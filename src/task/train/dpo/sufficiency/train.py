"""Weighted or unweighted DPO for one evidence-use policy role.

By default, training continues from the SFT adapter. The same entry point
supports Router, Gatekeeper, and Verifier preference pairs labelled by the
``score_rubric`` chain. The RL-only flags instead start from the base model.

Why hand-written and not trl
----------------------------
TRL's ``DPOTrainer`` does not expose a per-example weight knob. Our preference
strength is ``|gt_score - rag_score|`` and we must respect it (memory:
``project_training_strategy``). Rather than subclass trl's internal methods —
which would bind us to a specific trl version — we implement DPO directly on
top of ``transformers.Trainer``. The math is tiny; the bookkeeping (PEFT,
checkpointing, optimizer, logging) we keep from transformers.

Math
----
For each pair (prompt, chosen, rejected) with weight ``w``:

    logp_pol_c  = log P_policy   (chosen   | prompt)
    logp_pol_r  = log P_policy   (rejected | prompt)
    logp_ref_c  = log P_ref      (chosen   | prompt)
    logp_ref_r  = log P_ref      (rejected | prompt)

    logit       = β * ((logp_pol_c - logp_ref_c) - (logp_pol_r - logp_ref_r))
    loss_i      = -logsigmoid(logit)

    batch_loss  = sum(w_i * loss_i) / sum(w_i)

The default policy continues from SFT. For the RL-only ablation,
``--initialization base --uniform-pair-weights`` creates a fresh LoRA adapter
directly on the base model and assigns every retained preference pair weight 1.

The reference model is the SAME base model with the policy adapter **disabled**
(memory-efficient: ``with model.disable_adapter():`` swaps in the frozen base
LM without loading a second copy).

Inputs
------
Role-specific JSONL produced by ``src.task.train.dpo.*.data_preparation``:

    {prompt: str, chosen: str, rejected: str, weight: float, meta: {...}}

Outputs
-------
One LoRA adapter in ``--output-dir``. Full-method adapters continue from SFT;
RL-only adapters are fresh base-model LoRAs. They are not merged into the base.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.task.train.hub_utils import push_folder_to_hub
from src.task.train.paths import (
    BASE_MODEL_DIR,
    DPO_SUFFICIENCY_ADAPTER_DIR,
    DPO_SUFFICIENCY_DATA_DIR,
    SFT_ADAPTER_DIR,
)
from src.task.train.sft.data import build_completion_string, load_records


# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DPOKnobs:
    role: str = "sufficiency"
    base_model_dir: Path = BASE_MODEL_DIR
    sft_adapter_dir: Path = SFT_ADAPTER_DIR
    output_dir: Path = DPO_SUFFICIENCY_ADAPTER_DIR
    data_dir: Path = DPO_SUFFICIENCY_DATA_DIR
    seed: int = 42
    beta: float = 0.1
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    max_length: int = 4096
    max_steps: int = -1
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 10
    warmup_ratio: float = 0.05
    initialization: str = "sft"
    uniform_pair_weights: bool = False


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _tokenize_pair(
    row: dict, tokenizer, *, max_length: int
) -> Optional[dict[str, Any]]:
    """Tokenize one DPO row into chosen / rejected input_ids and a label mask."""
    prompt_str: str = row["prompt"]
    chosen_str = build_completion_string(row["chosen"])
    rejected_str = build_completion_string(row["rejected"])

    p_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(chosen_str, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(rejected_str, add_special_tokens=False)["input_ids"]

    longest = max(len(p_ids) + len(c_ids), len(p_ids) + len(r_ids))
    if longest > max_length:
        # Drop prompt tokens from the front while keeping the `<|im_start|>assistant\n`
        # tail (last few tokens of prompt). Simple right-truncate then left-truncate
        # rarely needed for sufficiency prompts; if overflow happens just hard-trim.
        overflow = longest - max_length
        if len(p_ids) <= overflow + 8:
            return None
        p_ids = p_ids[overflow:]

    chosen_input = p_ids + c_ids
    rejected_input = p_ids + r_ids
    chosen_labels = [-100] * len(p_ids) + list(c_ids)
    rejected_labels = [-100] * len(p_ids) + list(r_ids)

    return {
        "chosen_input_ids": chosen_input,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input,
        "rejected_labels": rejected_labels,
        "weight": float(row["weight"]),
        "id": row.get("id", ""),
    }


def build_dpo_dataset(split: str, tokenizer, knobs: DPOKnobs):
    """Load and tokenize the DPO split. Returns a ``datasets.Dataset``."""
    from datasets import Dataset

    path = knobs.data_dir / f"{split}.jsonl"
    rows = load_records(path)
    tokenized = []
    skipped = 0
    for r in rows:
        if knobs.uniform_pair_weights:
            r = {**r, "weight": 1.0}
        out = _tokenize_pair(r, tokenizer, max_length=knobs.max_length)
        if out is None:
            skipped += 1
            continue
        tokenized.append(out)
    if skipped:
        print(f"WARNING: skipped {skipped} pairs in {split} (too long).")
    return Dataset.from_list(tokenized)


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #


class DPOCollator:
    """Pad chosen + rejected sequences into a single batched dict.

    Output keys:
        chosen_input_ids, chosen_attention_mask, chosen_labels
        rejected_input_ids, rejected_attention_mask, rejected_labels
        weight                            (shape: [bsz])
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def _pad(self, seqs: list[list[int]], pad_value: int):
        max_len = max(len(s) for s in seqs)
        padded = [s + [pad_value] * (max_len - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]
        return padded, mask

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        import torch

        c_ids, c_mask = self._pad([r["chosen_input_ids"] for r in batch], self.pad_id)
        r_ids, r_mask = self._pad([r["rejected_input_ids"] for r in batch], self.pad_id)
        c_lbl, _ = self._pad([r["chosen_labels"] for r in batch], -100)
        r_lbl, _ = self._pad([r["rejected_labels"] for r in batch], -100)

        return {
            "chosen_input_ids": torch.tensor(c_ids, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(c_mask, dtype=torch.long),
            "chosen_labels": torch.tensor(c_lbl, dtype=torch.long),
            "rejected_input_ids": torch.tensor(r_ids, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(r_mask, dtype=torch.long),
            "rejected_labels": torch.tensor(r_lbl, dtype=torch.long),
            "weight": torch.tensor([r["weight"] for r in batch], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Per-sequence log-probability of the label tokens
# --------------------------------------------------------------------------- #


def _seq_logprob(model, input_ids, attention_mask, labels):
    """Sum of log P(token | prefix) over positions where label != -100.

    Returns a tensor of shape ``[bsz]`` (one log-prob per row).
    """
    import torch

    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    )
    logits = outputs.logits  # [bsz, seq, vocab]
    # Shift: predict token t+1 from logits at position t.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = (shift_labels != -100).to(shift_logits.dtype)

    # Clamp -100 to a valid index (will be masked out anyway).
    safe_labels = shift_labels.clamp_min(0)
    logp_all = torch.log_softmax(shift_logits, dim=-1)
    logp_token = logp_all.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    logp_token = logp_token * shift_mask  # zero out masked positions
    return logp_token.sum(dim=-1)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


def _build_weighted_dpo_trainer_cls():
    """Return a ``Trainer`` subclass implementing weighted DPO loss.

    Defined inside a factory so that the heavy ``transformers`` import only
    happens when the trainer is actually constructed.
    """
    from transformers import Trainer
    import torch
    import torch.nn.functional as F

    class WeightedDPOTrainer(Trainer):
        def __init__(self, *args, beta: float, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta = beta

        def _policy_and_ref_logp(self, model, input_ids, attention_mask, labels):
            """Return ``(policy_logp, ref_logp)`` for one side (chosen or rejected).

            Reference = base model with the adapter disabled. We use
            ``model.disable_adapter()`` so no second model copy is needed.
            """
            policy_logp = _seq_logprob(model, input_ids, attention_mask, labels)
            with torch.no_grad(), model.disable_adapter():
                ref_logp = _seq_logprob(model, input_ids, attention_mask, labels)
            return policy_logp, ref_logp

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            weights = inputs["weight"].to(model.device)
            c_ids = inputs["chosen_input_ids"].to(model.device)
            c_mask = inputs["chosen_attention_mask"].to(model.device)
            c_lbl = inputs["chosen_labels"].to(model.device)
            r_ids = inputs["rejected_input_ids"].to(model.device)
            r_mask = inputs["rejected_attention_mask"].to(model.device)
            r_lbl = inputs["rejected_labels"].to(model.device)

            pol_c, ref_c = self._policy_and_ref_logp(model, c_ids, c_mask, c_lbl)
            pol_r, ref_r = self._policy_and_ref_logp(model, r_ids, r_mask, r_lbl)

            logits = self.beta * ((pol_c - ref_c) - (pol_r - ref_r))
            per_sample_loss = -F.logsigmoid(logits)

            denom = weights.sum().clamp_min(1e-8)
            loss = (per_sample_loss * weights).sum() / denom

            # Log handy metrics; Trainer collects these via .log()
            with torch.no_grad():
                metrics = {
                    "dpo_accuracy": (logits > 0).float().mean().item(),
                    "logits_mean": logits.mean().item(),
                    "weight_mean": weights.mean().item(),
                }
            self._latest_metrics = metrics

            if return_outputs:
                return loss, {"logits": logits.detach()}
            return loss

        def log(self, logs: dict, *args, **kwargs):  # type: ignore[override]
            if getattr(self, "_latest_metrics", None):
                for k, v in self._latest_metrics.items():
                    logs.setdefault(k, v)
                self._latest_metrics = {}
            return super().log(logs, *args, **kwargs)

    return WeightedDPOTrainer


# --------------------------------------------------------------------------- #
# Model loading: base + trainable LoRA, optionally initialized from SFT
# --------------------------------------------------------------------------- #


def load_policy_model(knobs: DPOKnobs):
    """Load the base model with either an SFT-initialized or fresh LoRA.

    Returns ``(model, tokenizer)``.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if knobs.initialization == "sft" and not knobs.sft_adapter_dir.exists():
        raise SystemExit(
            f"SFT adapter not found at {knobs.sft_adapter_dir}\n"
            "  -> run SFT first: bash scripts/train/sft/train.sh"
        )

    # Tokenizer: prefer the one saved next to the SFT adapter (matches the
    # training-time tokenizer). Fall back to base if missing.
    tok_dir = (
        knobs.sft_adapter_dir
        if knobs.initialization == "sft"
        and (knobs.sft_adapter_dir / "tokenizer_config.json").exists()
        else knobs.base_model_dir
    )
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        knobs.base_model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False

    if knobs.initialization == "sft":
        model = PeftModel.from_pretrained(
            base, str(knobs.sft_adapter_dir), is_trainable=True
        )
    else:
        from src.task.train.sft.trainer import wrap_model_with_lora

        model = wrap_model_with_lora(base)
    return model, tok


def load_policy_with_sft_adapter(knobs: DPOKnobs):
    """Backward-compatible alias for the default SFT-initialized loader."""
    if knobs.initialization != "sft":
        raise ValueError("load_policy_with_sft_adapter requires initialization='sft'")
    return load_policy_model(knobs)


# --------------------------------------------------------------------------- #
# Training args
# --------------------------------------------------------------------------- #


def build_dpo_training_args(knobs: DPOKnobs):
    from transformers import TrainingArguments

    from src.task.train.hub_utils import configure_wandb

    stage = (
        f"rl_only_{knobs.role}"
        if knobs.initialization == "base" and knobs.uniform_pair_weights
        else f"dpo_{knobs.role}"
    )
    report_to, run_name = configure_wandb(
        output_dir=knobs.output_dir, stage=stage
    )

    return TrainingArguments(
        output_dir=str(knobs.output_dir),
        per_device_train_batch_size=knobs.per_device_train_batch_size,
        per_device_eval_batch_size=knobs.per_device_eval_batch_size,
        gradient_accumulation_steps=knobs.gradient_accumulation_steps,
        num_train_epochs=knobs.num_train_epochs,
        max_steps=knobs.max_steps,
        learning_rate=knobs.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=knobs.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=knobs.logging_steps,
        save_steps=knobs.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=knobs.eval_steps,
        report_to=report_to,
        run_name=run_name,
        seed=knobs.seed,
        remove_unused_columns=False,
        label_names=["chosen_labels", "rejected_labels"],
        optim="adamw_torch",
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-model-dir", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument(
        "--role", choices=("routing", "sufficiency", "judgment"),
        default="sufficiency",
    )
    parser.add_argument("--sft-adapter-dir", type=Path, default=SFT_ADAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DPO_SUFFICIENCY_ADAPTER_DIR)
    parser.add_argument("--data-dir", type=Path, default=DPO_SUFFICIENCY_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--initialization",
        choices=("sft", "base"),
        default="sft",
        help="Initialize the trainable LoRA from SFT (default) or from the base model.",
    )
    parser.add_argument(
        "--uniform-pair-weights",
        action="store_true",
        help="Set every retained preference-pair weight to 1 (standard DPO ablation).",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=os.environ.get("HF_REPO_ID", ""),
        help=(
            "If set, push the final adapter to this Hugging Face Hub repo "
            "(e.g. 'your-org/etd-policy-dpo-sufficiency'). Defaults to env "
            "HF_REPO_ID. Empty disables push."
        ),
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        default=os.environ.get("HF_PRIVATE", "").lower() in {"1", "true", "yes"},
        help="Create the Hub repo as private (default: public, or env HF_PRIVATE=1).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    knobs = DPOKnobs(
        role=args.role,
        base_model_dir=args.base_model_dir,
        sft_adapter_dir=args.sft_adapter_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        initialization=args.initialization,
        uniform_pair_weights=args.uniform_pair_weights,
    )

    if not knobs.base_model_dir.exists():
        raise SystemExit(f"base model not found: {knobs.base_model_dir}")
    if not (knobs.data_dir / "train.jsonl").exists():
        raise SystemExit(
            f"DPO data not found: {knobs.data_dir}/train.jsonl\n"
            "  -> run: bash scripts/train/dpo/prepare_data.sh"
        )

    weighting = "uniform" if knobs.uniform_pair_weights else "outcome-weighted"
    print(f"== EtD DPO ({knobs.role}, {knobs.initialization}-init, {weighting}) ==")
    print(f"base_model_dir  = {knobs.base_model_dir}")
    print(f"sft_adapter_dir = {knobs.sft_adapter_dir}")
    print(f"initialization  = {knobs.initialization}")
    print(f"pair_weighting  = {weighting}")
    print(f"data_dir        = {knobs.data_dir}")
    print(f"output_dir      = {knobs.output_dir}")
    print(f"beta            = {knobs.beta}")

    model, tok = load_policy_model(knobs)

    train_ds = build_dpo_dataset("train", tok, knobs)
    print(f"train pairs = {len(train_ds)}")
    eval_ds = None
    if not args.no_eval:
        val_path = knobs.data_dir / "val.jsonl"
        if val_path.exists():
            eval_ds = build_dpo_dataset("val", tok, knobs)
            print(f"val pairs   = {len(eval_ds)}")
        else:
            print(f"WARNING: no val split at {val_path}; skipping eval.")

    training_args = build_dpo_training_args(knobs)
    if eval_ds is None:
        training_args.eval_strategy = "no"

    WeightedDPOTrainer = _build_weighted_dpo_trainer_cls()
    trainer = WeightedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DPOCollator(tok),
        processing_class=tok,
        beta=knobs.beta,
    )

    trainer.train()

    knobs.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(knobs.output_dir))
    tok.save_pretrained(str(knobs.output_dir))

    meta = {
        "base_model": str(knobs.base_model_dir),
        "role": knobs.role,
        "sft_adapter": (
            str(knobs.sft_adapter_dir) if knobs.initialization == "sft" else None
        ),
        "initialization": knobs.initialization,
        "pair_weighting": weighting,
        "data_dir": str(knobs.data_dir),
        "beta": knobs.beta,
        "seed": knobs.seed,
        "num_train_epochs": knobs.num_train_epochs,
        "learning_rate": knobs.learning_rate,
        "train_pairs": len(train_ds),
    }
    with (knobs.output_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved adapter + tokenizer + train_meta.json to {knobs.output_dir}")

    if args.hub_repo_id:
        print(f"pushing {knobs.output_dir} -> {args.hub_repo_id} ...")
        push_folder_to_hub(
            knobs.output_dir,
            repo_id=args.hub_repo_id,
            private=args.hub_private,
            commit_message=(
                f"EtD DPO {knobs.role} adapter ({len(train_ds)} pairs, "
                f"beta={knobs.beta}, epochs={knobs.num_train_epochs})"
            ),
        )


if __name__ == "__main__":
    main()

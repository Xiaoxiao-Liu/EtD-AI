"""Shared SFT trainer/config builders (multi-task policy on Qwen3-8B).

Pure transformers + peft (no trl dependency). Centralises:
- LoRA configuration (one shared adapter across [ROUTE] / [SUFFICE] / [JUDGE])
- ``TrainingArguments`` tuned for A800-80GB single-card bf16
- Base model + tokenizer loader
- A data collator that pads ``input_ids`` and ``labels`` together

Loss is masked at *dataset build time* (see ``src/task/train/sft/data.py``):
labels for system + user tokens are set to ``-100`` so the standard causal-LM
loss only counts assistant tokens. The collator only needs to pad.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.task.train.paths import BASE_MODEL_DIR, SFT_ADAPTER_DIR


# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainKnobs:
    """Runtime knobs read from the CLI. Everything else is fixed in code."""

    base_model_dir: Path = BASE_MODEL_DIR
    output_dir: Path = SFT_ADAPTER_DIR
    seed: int = 42
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.5e-4
    max_length: int = 4096
    max_steps: int = -1  # -1 == disabled (let epochs decide)
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10
    warmup_ratio: float = 0.03


# --------------------------------------------------------------------------- #
# Model + tokenizer
# --------------------------------------------------------------------------- #


def load_policy_model_and_tokenizer(knobs: TrainKnobs):
    """Load Qwen3-8B in bf16 with sdpa attention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(knobs.base_model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Causal LM training right-pads; collator below relies on this convention.
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        knobs.base_model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Required by gradient checkpointing.
    model.config.use_cache = False
    return model, tok


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #


def build_lora_config():
    """LoRA hitting all Qwen3 linear layers (attn + MLP)."""
    from peft import LoraConfig

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def wrap_model_with_lora(model):
    """Apply our LoRA config and return the PEFT-wrapped model.

    Moves the base model to GPU before wrapping. Wrapping a CPU model with
    peft 0.19 + torch 2.12 triggers a per-LoRA-Linear ``.to(cpu, bf16)`` path
    inside ``_move_adapter_to_device_of_base_layer`` that hangs CPU for a
    very long time on 8B-scale models; doing the move first avoids it.
    """
    import torch
    from peft import get_peft_model

    if torch.cuda.is_available():
        model = model.to("cuda")
    lora_config = build_lora_config()
    return get_peft_model(model, lora_config)


# --------------------------------------------------------------------------- #
# Training args
# --------------------------------------------------------------------------- #


def build_training_args(knobs: TrainKnobs):
    """Construct ``transformers.TrainingArguments`` from the knobs."""
    from transformers import TrainingArguments

    from src.task.train.hub_utils import configure_wandb

    report_to, run_name = configure_wandb(
        output_dir=knobs.output_dir, stage="sft"
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
        remove_unused_columns=False,  # keep ``task`` / ``id`` columns
        label_names=["labels"],
        optim="adamw_torch",
    )


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #


class SFTPadCollator:
    """Right-pad ``input_ids`` and ``labels`` to the longest sequence in batch.

    Labels are padded with ``-100`` so padded positions are excluded from loss.
    Attention mask is built from ``input_ids != pad_token_id``.

    Loss masking on system / user tokens is already done at dataset build time
    (labels there are ``-100``).
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer.pad_token_id is None — set pad_token first")

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        import torch

        max_len = max(len(r["input_ids"]) for r in batch)
        input_ids = []
        labels = []
        attention_mask = []
        for r in batch:
            ids = r["input_ids"]
            lbl = r["labels"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_n)
            labels.append(lbl + [-100] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

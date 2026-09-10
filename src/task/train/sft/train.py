"""Multi-task SFT for the EtD policy model (single shared LoRA adapter).

One model handles routing / sufficiency / judgment, distinguished by an
instruction tag in the user message. Three task JSONLs are merged into one
tokenized + label-masked dataset; loss is computed on assistant tokens only.

Implementation uses transformers ``Trainer`` + peft directly (no trl). The
ChatML prompt is hand-assembled in ``src/task/train/sft/data.py`` to avoid
Qwen3's chat-template ``<think>...</think>`` injection on the assistant turn.

Outputs::

    EtD/model/policy/sft/
        adapter_config.json
        adapter_model.safetensors
        tokenizer.* (saved alongside for self-contained inference)
        checkpoint-*/

Example::

    # Smoke test (2 steps, capped data):
    python -m src.task.train.sft.train \\
        --sample-cap-routing 16 --sample-cap-sufficiency 16 \\
        --sample-cap-judgment 16 --max-steps 2

    # Full training:
    python -m src.task.train.sft.train

The three wrapper modules ``src.task.train.sft.{routing,sufficiency,judgment}
.train`` re-export ``main`` from here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from src.task.train.hub_utils import push_folder_to_hub
from src.task.train.paths import BASE_MODEL_DIR, SFT_ADAPTER_DIR, SFT_TASKS
from src.task.train.sft.data import dataset_size_by_task, make_dataset
from src.task.train.sft.trainer import (
    SFTPadCollator,
    TrainKnobs,
    build_training_args,
    load_policy_model_and_tokenizer,
    wrap_model_with_lora,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=BASE_MODEL_DIR,
        help=f"Base model directory (default: {BASE_MODEL_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SFT_ADAPTER_DIR,
        help=f"Where to save the SFT adapter (default: {SFT_ADAPTER_DIR}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, override num_train_epochs and stop after N steps (smoke).",
    )
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(SFT_TASKS),
        choices=list(SFT_TASKS),
        help="Subset of tasks to train on (default: all three).",
    )
    parser.add_argument("--sample-cap-routing", type=int, default=None)
    parser.add_argument("--sample-cap-sufficiency", type=int, default=None)
    parser.add_argument("--sample-cap-judgment", type=int, default=None)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip building the val dataset and disable eval.",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=os.environ.get("HF_REPO_ID", ""),
        help=(
            "If set, push the final adapter to this Hugging Face Hub repo "
            "(e.g. 'your-org/etd-policy-sft'). Defaults to env HF_REPO_ID. "
            "Empty disables push."
        ),
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        default=os.environ.get("HF_PRIVATE", "").lower() in {"1", "true", "yes"},
        help="Create the Hub repo as private (default: public, or env HF_PRIVATE=1).",
    )
    return parser.parse_args()


def _sample_caps_from_args(args: argparse.Namespace) -> dict[str, int]:
    caps: dict[str, int] = {}
    for task in SFT_TASKS:
        v = getattr(args, f"sample_cap_{task}")
        if v is not None:
            caps[task] = v
    return caps


def main() -> None:
    args = _parse_args()

    knobs = TrainKnobs(
        base_model_dir=args.base_model_dir,
        output_dir=args.output_dir,
        seed=args.seed,
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
    )

    if not knobs.base_model_dir.exists():
        raise SystemExit(
            f"base model not found at {knobs.base_model_dir}\n"
            "  -> run: python -m src.common.download_model --model qwen3-8b"
        )

    print("== EtD SFT (multi-task, shared LoRA) ==")
    print(f"base_model_dir = {knobs.base_model_dir}")
    print(f"output_dir     = {knobs.output_dir}")
    print(f"tasks          = {args.tasks}")

    sample_caps = _sample_caps_from_args(args)
    if sample_caps:
        print(f"sample_caps    = {sample_caps}")

    # ---- model + tokenizer ---------------------------------------------- #
    model, tok = load_policy_model_and_tokenizer(knobs)

    # ---- dataset --------------------------------------------------------- #
    tasks = tuple(args.tasks)
    train_ds = make_dataset(
        "train",
        tok,
        max_length=knobs.max_length,
        tasks=tasks,
        sample_caps=sample_caps,
        seed=knobs.seed,
    )
    print(f"train rows     = {len(train_ds)}  by_task={dataset_size_by_task(train_ds)}")

    eval_ds: Optional[object] = None
    if not args.no_eval:
        try:
            eval_ds = make_dataset(
                "val",
                tok,
                max_length=knobs.max_length,
                tasks=tasks,
                sample_caps=None,
                seed=knobs.seed,
            )
            print(
                f"val rows       = {len(eval_ds)}  "
                f"by_task={dataset_size_by_task(eval_ds)}"
            )
        except FileNotFoundError as e:
            print(f"WARNING: skipping eval ({e})")
            eval_ds = None

    # ---- wrap with LoRA + assemble Trainer ------------------------------ #
    model = wrap_model_with_lora(model)
    model.print_trainable_parameters()

    training_args = build_training_args(knobs)
    if args.no_eval or eval_ds is None:
        training_args.eval_strategy = "no"

    collator = SFTPadCollator(tok)

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tok,
    )

    trainer.train()

    # ---- save final adapter + tokenizer + a small metadata file --------- #
    knobs.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(knobs.output_dir))
    tok.save_pretrained(str(knobs.output_dir))

    meta = {
        "base_model": str(knobs.base_model_dir),
        "tasks": list(tasks),
        "seed": knobs.seed,
        "num_train_epochs": knobs.num_train_epochs,
        "per_device_train_batch_size": knobs.per_device_train_batch_size,
        "gradient_accumulation_steps": knobs.gradient_accumulation_steps,
        "learning_rate": knobs.learning_rate,
        "max_length": knobs.max_length,
        "train_rows": len(train_ds),
        "train_by_task": dataset_size_by_task(train_ds),
        "sample_caps": sample_caps,
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
            commit_message=f"EtD SFT adapter ({len(train_ds)} rows, "
            f"epochs={knobs.num_train_epochs})",
        )


if __name__ == "__main__":
    main()

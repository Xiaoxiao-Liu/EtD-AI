"""Dataset assembly for multi-task SFT (transformers Trainer flavour).

Reads the three per-task JSONL files produced by
``src.task.train.sft.prepare_data`` and merges them into one HuggingFace
``datasets.Dataset`` that is already tokenized and label-masked:

    {
      "input_ids": list[int],    # full sequence: SYS + USER + ASSISTANT
      "labels":    list[int],    # SYS/USER -100, ASSISTANT tokens kept
      "task":      "routing" | "sufficiency" | "judgment",
      "id":        "<record id>",
    }

We **do not** use ``tokenizer.apply_chat_template`` because the Qwen3 chat
template forces a ``<think>...</think>`` block into the last assistant turn
even when no reasoning is present. For a policy that should output a bare
label string, that template behaviour fights us. Instead we hand-build the
ChatML string (the same special tokens Qwen3 was trained on)::

    <|im_start|>system\\n{SYSTEM}<|im_end|>\\n
    <|im_start|>user\\n{USER}<|im_end|>\\n
    <|im_start|>assistant\\n{LABEL}<|im_end|>

Loss is computed on the assistant tokens (label + the closing ``<|im_end|>``);
everything before ``<|im_start|>assistant\\n`` is masked to ``-100``.

For inference, the prompt is the same up to ``<|im_start|>assistant\\n``; the
model then generates until it emits ``<|im_end|>``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from src.task.train.paths import SFT_TASKS, sft_task_split_path
from src.task.train.sft.prompts import (
    SYSTEM_PROMPT,
    extract_target_label,
    render_user,
)

# Qwen3 special tokens used to frame each turn.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# --------------------------------------------------------------------------- #
# IO (kept aligned with src/task/train/sft/prepare_data.py)
# --------------------------------------------------------------------------- #


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# ChatML string builders
# --------------------------------------------------------------------------- #


def build_prompt_string(user_text: str, *, system_text: str = SYSTEM_PROMPT) -> str:
    """Assemble system + user + ``<|im_start|>assistant\\n`` (no label).

    Used both for inference (the model continues from here) and for SFT
    tokenization (we then concatenate the label tokens with loss enabled).
    """
    return (
        f"{IM_START}system\n{system_text}{IM_END}\n"
        f"{IM_START}user\n{user_text}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def build_completion_string(label: str) -> str:
    """The assistant-side string that follows the prompt. Ends with ``<|im_end|>``."""
    return f"{label}{IM_END}"


# --------------------------------------------------------------------------- #
# Per-record tokenization
# --------------------------------------------------------------------------- #


def tokenize_record(
    record: dict,
    task: str,
    tokenizer,
    *,
    max_length: int,
) -> Optional[dict]:
    """Tokenize one record and produce ``{input_ids, labels, task, id}``.

    Truncation policy: if the full sequence exceeds ``max_length``, the
    **user** portion is truncated from the right (keep system + label intact).
    Records whose label alone wouldn't fit are skipped (returned ``None``).
    """
    user_text = render_user(task, record.get("input", {}))
    label = extract_target_label(task, record.get("target", {}))

    prompt_str = build_prompt_string(user_text)
    completion_str = build_completion_string(label)

    # Tokenize without special tokens — every framing token is already in
    # the string. ``<|im_start|>`` etc. are part of the vocabulary as
    # "added tokens" and tokenize as single ids.
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_str, add_special_tokens=False)["input_ids"]

    if len(completion_ids) >= max_length:
        # Label alone too long — should not happen for our short labels.
        return None

    available = max_length - len(completion_ids)
    if len(prompt_ids) > available:
        # Drop tokens from the **middle of the user turn** to preserve the
        # system header and the ``<|im_start|>assistant\n`` tail. Simplest
        # right-truncation of the user content section works because the
        # assistant marker is appended last.
        # We truncate to leave the last few tokens (the assistant marker)
        # intact: chop from the start of the user content.
        # Implementation: re-render with a truncated user_text.
        # Estimate by tokens: shave roughly proportional to overage.
        overflow = len(prompt_ids) - available
        # Approximate: each english/chinese token ~ a few chars; use a
        # conservative ratio to avoid an infinite loop.
        approx_chars = max(8, overflow * 4)
        truncated_user = user_text[: max(0, len(user_text) - approx_chars)]
        prompt_str = build_prompt_string(truncated_user)
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > available:
            # Last-resort hard cap.
            prompt_ids = prompt_ids[-available:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "task": task,
        "id": record.get("id", ""),
    }


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def _load_records(task: str, split: str, sample_cap: Optional[int], rng: random.Random):
    path = sft_task_split_path(task, split)
    recs = load_records(path)
    if sample_cap is not None and sample_cap >= 0 and len(recs) > sample_cap:
        rng.shuffle(recs)
        recs = recs[:sample_cap]
    return recs


def make_dataset(
    split: str,
    tokenizer,
    *,
    max_length: int,
    tasks: tuple[str, ...] = SFT_TASKS,
    sample_caps: Optional[dict[str, int]] = None,
    seed: int = 42,
):
    """Build a tokenized + label-masked multi-task dataset for one split."""
    from datasets import Dataset

    rng = random.Random(seed)
    rows: list[dict] = []
    skipped = 0
    for task in tasks:
        cap = (sample_caps or {}).get(task)
        for rec in _load_records(task, split, cap, rng):
            tok_row = tokenize_record(rec, task, tokenizer, max_length=max_length)
            if tok_row is None:
                skipped += 1
                continue
            rows.append(tok_row)

    if skipped:
        print(f"WARNING: skipped {skipped} records (label longer than max_length).")

    rng.shuffle(rows)
    return Dataset.from_list(rows)


def dataset_size_by_task(ds) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in ds["task"]:
        counts[task] = counts.get(task, 0) + 1
    return counts

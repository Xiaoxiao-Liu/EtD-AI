"""Shared helpers for role-specific DPO preference datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from src.task.train.sft.data import build_prompt_string, load_records, write_jsonl
from src.task.train.sft.prompts import render_user


def load_split_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(
            f"PICO splits file not found: {path}\n"
            "  -> run: bash scripts/train/sft/prepare_data.sh"
        )
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    split_map = data.get("split_map")
    if not isinstance(split_map, dict):
        raise SystemExit(f"unexpected splits file format at {path}")
    return split_map


def make_prompt(task: str, input_obj: dict) -> str:
    return build_prompt_string(render_user(task, input_obj))


def build_and_write(
    *,
    input_path: Path,
    splits_path: Path,
    output_dir: Path,
    pair_builder: Callable[[dict], tuple[Optional[dict], str]],
    config: dict,
) -> None:
    split_map = load_split_map(splits_path)
    records = load_records(input_path)
    buckets: dict[str, list[dict]] = {"train": [], "val": []}
    dropped: Counter = Counter()

    for record in records:
        pico = record.get("pico_source_file")
        if not isinstance(pico, str):
            dropped["no_pico"] += 1
            continue
        split = split_map.get(pico)
        if split not in buckets:
            dropped["held_out_or_missing_split"] += 1
            continue
        pair, reason = pair_builder(record)
        if pair is None:
            dropped[reason or "filtered"] += 1
            continue
        buckets[split].append(pair)

    for split, rows in buckets.items():
        write_jsonl(rows, output_dir / f"{split}.jsonl")
    (output_dir / "test.jsonl").unlink(missing_ok=True)

    meta = {
        **config,
        "input": str(input_path),
        "splits_file": str(splits_path),
        "rows": {key: len(value) for key, value in buckets.items()},
        "chosen_distribution": {
            key: dict(Counter(row["chosen"] for row in value))
            for key, value in buckets.items()
        },
        "weight_summary": {
            key: {
                "n": len(value),
                "min": min((row["weight"] for row in value), default=None),
                "max": max((row["weight"] for row in value), default=None),
                "mean": (
                    sum(row["weight"] for row in value) / len(value)
                    if value else None
                ),
            }
            for key, value in buckets.items()
        },
        "dropped": dict(dropped),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)

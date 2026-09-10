"""Build the DPO sufficiency dataset from ``all_models.labeled.jsonl``.

For each labelled record that is eligible as an SFT sufficiency sample, emit a
DPO preference pair:

    - ``prompt``   : ChatML string up to ``<|im_start|>assistant\\n`` — identical
                     to what the SFT model sees at inference time
    - ``chosen``   : the actual ``evidence_sufficiency_label`` string
                     (``sufficient`` or ``retrieve_more``)
    - ``rejected`` : the opposite label
    - ``weight``   : ``abs(gt_score - rag_score)`` — preference strength.
                     Records below ``--min-weight`` are dropped as ambiguous.

We hand-build the ChatML string (the same way ``src.task.train.sft.data``
does) instead of using ``tokenizer.apply_chat_template`` — the Qwen3 template
forces ``<think>...</think>`` blocks into assistant turns and we want a clean
prompt → bare-label assistant continuation.

The PICO-level split is reused from ``dataset/train/sft/pico_splits.json``.
Only train/validation preference pairs are materialized; held-out test PICOs
are never converted into DPO training format.

Outputs::

    EtD/dataset/train/dpo/sufficiency/output/{train,val}.jsonl
    EtD/dataset/train/dpo/sufficiency/output/meta.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from src.task.train.paths import (
    DPO_SUFFICIENCY_DATA_DIR,
    LABELED_JSONL,
    PICO_SPLITS_FILE,
)
from src.task.train.sft.data import build_prompt_string, load_records, write_jsonl
from src.task.train.sft.prepare_data import build_sufficiency_record
from src.task.train.sft.prompts import render_user

OPPOSITE_LABEL = {
    "sufficient": "retrieve_more",
    "retrieve_more": "sufficient",
}


def _load_split_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(
            f"PICO splits file not found: {path}\n"
            "  -> run: bash scripts/train/sft/prepare_data.sh"
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    split_map = data.get("split_map")
    if not isinstance(split_map, dict):
        raise SystemExit(f"unexpected splits file format at {path}")
    return split_map


def _build_pair(record: dict, *, min_weight: float) -> Optional[dict]:
    """Return one DPO row, or ``None`` if this record is ineligible."""
    built = build_sufficiency_record(record)
    if built is None:
        return None

    gt = record.get("gt_score")
    rag = record.get("rag_score")
    if not isinstance(gt, (int, float)) or not isinstance(rag, (int, float)):
        return None
    weight = abs(float(gt) - float(rag))
    if weight < min_weight:
        return None

    chosen_label = built["target"]["sufficiency"]
    rejected_label = OPPOSITE_LABEL.get(chosen_label)
    if rejected_label is None:
        return None

    user_text = render_user("sufficiency", built["input"])
    prompt_str = build_prompt_string(user_text)

    return {
        "id": built["id"],
        "prompt": prompt_str,
        "chosen": chosen_label,
        "rejected": rejected_label,
        "weight": weight,
        "meta": {
            **built["meta"],
            "gt_score": float(gt),
            "rag_score": float(rag),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=LABELED_JSONL,
        help=f"Labelled JSONL (default: {LABELED_JSONL}).",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=PICO_SPLITS_FILE,
        help=f"PICO split map shared with SFT (default: {PICO_SPLITS_FILE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DPO_SUFFICIENCY_DATA_DIR,
        help=f"Output directory (default: {DPO_SUFFICIENCY_DATA_DIR}).",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        help=(
            "Drop pairs whose |gt_score - rag_score| < this threshold "
            "(default: 0.05; both scores live in [0,1])."
        ),
    )
    args = parser.parse_args()

    split_map = _load_split_map(args.splits_file)
    records = load_records(args.input)
    print(f"loaded {len(records)} labelled records from {args.input}")
    print(f"PICOs in split map: {len(split_map)}")

    buckets: dict[str, list[dict]] = {"train": [], "val": []}
    dropped: Counter = Counter()

    for rec in records:
        pico = rec.get("pico_source_file")
        if not isinstance(pico, str):
            dropped["no_pico"] += 1
            continue
        split = split_map.get(pico)
        if split not in buckets:
            dropped["held_out_or_missing_split"] += 1
            continue
        pair = _build_pair(rec, min_weight=args.min_weight)
        if pair is None:
            dropped["filtered"] += 1
            continue
        buckets[split].append(pair)

    for split, rows in buckets.items():
        out_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(rows, out_path)
        print(f"  {split}: {len(rows)} pairs -> {out_path}")
    (args.output_dir / "test.jsonl").unlink(missing_ok=True)

    label_dist = {
        split: dict(Counter(r["chosen"] for r in rows))
        for split, rows in buckets.items()
    }
    weight_summary = {
        split: {
            "n": len(rows),
            "min": min((r["weight"] for r in rows), default=None),
            "max": max((r["weight"] for r in rows), default=None),
            "mean": (sum(r["weight"] for r in rows) / len(rows) if rows else None),
        }
        for split, rows in buckets.items()
    }
    meta = {
        "input": str(args.input),
        "splits_file": str(args.splits_file),
        "min_weight": args.min_weight,
        "rows": {k: len(v) for k, v in buckets.items()},
        "label_distribution": label_dist,
        "weight_summary": weight_summary,
        "dropped": dict(dropped),
    }
    meta_path = args.output_dir / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote meta to {meta_path}")
    print(f"dropped: {dict(dropped)}")


if __name__ == "__main__":
    main()

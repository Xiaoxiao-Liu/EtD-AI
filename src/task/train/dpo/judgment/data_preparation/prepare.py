"""Build Verifier DPO pairs from single-candidate SFT quality records."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.task.train.dpo.common_prepare import build_and_write, make_prompt
from src.task.train.paths import (
    DPO_JUDGMENT_DATA_DIR,
    LABELED_JSONL,
    PICO_SPLITS_FILE,
)
from src.task.train.sft.prepare_data import build_judgment_records

OPPOSITE = {"accept": "reject", "reject": "accept"}


def _pair_from_built(source: dict, built: dict, *, min_weight: float, weight_mode: str):
    origin = built["meta"]["original_source"]
    utility_key = "no_score" if origin == "no_evidence" else "rag_score"
    utility = source.get(utility_key)
    if not isinstance(utility, (int, float)):
        return None, "missing_utility"
    chosen = built["target"]["quality"]
    # Confidence in accept grows with utility; confidence in reject grows with
    # its complement. This is an explicit experimental heuristic, not a label.
    weight = 1.0 if weight_mode == "unweighted" else (
        float(utility) if chosen == "accept" else 1.0 - float(utility)
    )
    if weight < min_weight:
        return None, "below_min_weight"
    return {
        "id": built["id"],
        "prompt": make_prompt("judgment", built["input"]),
        "chosen": chosen,
        "rejected": OPPOSITE[chosen],
        "weight": weight,
        "meta": {
            **built["meta"],
            "weight_mode": weight_mode,
            "candidate_utility": float(utility),
            "judgement_correctness": source.get(
                "no_evidence_judgement_correctness"
                if origin == "no_evidence" else "rag_judgement_correctness"
            ),
            "evidence_correctness_rag": source.get("evidence_correctness_rag")
            if origin == "rag_evidence" else None,
        },
    }, ""


def build_pairs(record: dict, *, min_weight: float, weight_mode: str):
    return [
        _pair_from_built(record, built, min_weight=min_weight, weight_mode=weight_mode)
        for built in build_judgment_records(record)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LABELED_JSONL)
    parser.add_argument("--splits-file", type=Path, default=PICO_SPLITS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DPO_JUDGMENT_DATA_DIR)
    parser.add_argument("--weight-mode", choices=("utility_confidence", "unweighted"), default="utility_confidence")
    parser.add_argument("--min-weight", type=float, default=0.05)
    args = parser.parse_args()

    # The Verifier can yield two pairs per source record, so flatten before
    # delegating to the common one-record builder.
    from src.task.train.dpo.common_prepare import load_split_map
    from src.task.train.sft.data import load_records, write_jsonl
    import json
    from collections import Counter

    split_map = load_split_map(args.splits_file)
    buckets = {"train": [], "val": []}
    dropped = Counter()
    for record in load_records(args.input):
        split = split_map.get(record.get("pico_source_file"))
        if split not in buckets:
            dropped["held_out_or_missing_split"] += 1
            continue
        pairs = build_pairs(record, min_weight=args.min_weight, weight_mode=args.weight_mode)
        if not pairs:
            dropped["invalid_sft_record"] += 1
        for pair, reason in pairs:
            if pair is None:
                dropped[reason] += 1
            else:
                buckets[split].append(pair)
    for split, rows in buckets.items():
        write_jsonl(rows, args.output_dir / f"{split}.jsonl")
    (args.output_dir / "test.jsonl").unlink(missing_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "task": "judgment", "weight_mode": args.weight_mode,
        "min_weight": args.min_weight,
        "input": str(args.input), "splits_file": str(args.splits_file),
        "rows": {k: len(v) for k, v in buckets.items()},
        "chosen_distribution": {k: dict(Counter(r["chosen"] for r in v)) for k, v in buckets.items()},
        "dropped": dict(dropped),
    }
    with (args.output_dir / "meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

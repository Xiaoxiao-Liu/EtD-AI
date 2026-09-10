"""Build Router DPO pairs from the existing Router SFT labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.task.train.dpo.common_prepare import build_and_write, make_prompt
from src.task.train.paths import (
    DPO_ROUTING_DATA_DIR,
    LABELED_JSONL,
    PICO_SPLITS_FILE,
)
from src.task.train.sft.prepare_data import build_routing_record

OPPOSITE = {
    "direct_answer": "retrieve_evidence",
    "retrieve_evidence": "direct_answer",
}


def build_pair(
    record: dict, *, min_weight: float, weight_mode: str,
    both_correct_weight: float = 0.3,
):
    built = build_routing_record(record)
    if built is None:
        return None, "invalid_sft_record"
    no_score, rag_score = record.get("no_score"), record.get("rag_score")
    if not isinstance(no_score, (int, float)) or not isinstance(rag_score, (int, float)):
        return None, "missing_utility"
    utility_gap = float(rag_score) - float(no_score)
    direct_correct = record.get("no_evidence_direct_gate") is True
    rag_correct = record.get("rag_hard_gate") is True
    expected = "direct_answer" if direct_correct else "retrieve_evidence"
    chosen = built["target"]["action"]
    if chosen != expected:
        return None, "stale_routing_label"

    if weight_mode == "unweighted":
        weight = 1.0
    elif weight_mode == "utility_gap":
        weight = abs(utility_gap)
    else:
        # Outcome-supervised action preference: exclusive success is strong;
        # when both paths are correct, direct wins because retrieval has cost.
        # If both observed paths fail, retrieval remains the only viable action
        # but receives only the score-gap confidence.
        if direct_correct and rag_correct:
            weight = both_correct_weight
        elif direct_correct != rag_correct:
            weight = 1.0
        else:
            weight = min(1.0, abs(utility_gap))
    if weight < min_weight:
        return None, "below_min_weight"
    return {
        "id": built["id"],
        "prompt": make_prompt("routing", built["input"]),
        "chosen": chosen,
        "rejected": OPPOSITE[chosen],
        "weight": weight,
        "meta": {
            **built["meta"],
            "weight_mode": weight_mode,
            "utility_gap": utility_gap,
            "both_correct_weight": both_correct_weight,
            "direct_correct": direct_correct,
            "rag_correct": rag_correct,
            "preference_reason": record.get("initial_action_preference_reason"),
        },
    }, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LABELED_JSONL)
    parser.add_argument("--splits-file", type=Path, default=PICO_SPLITS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DPO_ROUTING_DATA_DIR)
    parser.add_argument(
        "--weight-mode",
        choices=("outcome", "utility_gap", "unweighted"),
        default="outcome",
    )
    parser.add_argument("--both-correct-weight", type=float, default=0.3)
    parser.add_argument("--min-weight", type=float, default=0.05)
    args = parser.parse_args()
    build_and_write(
        input_path=args.input,
        splits_path=args.splits_file,
        output_dir=args.output_dir,
        pair_builder=lambda row: build_pair(
            row, min_weight=args.min_weight, weight_mode=args.weight_mode,
            both_correct_weight=args.both_correct_weight,
        ),
        config={
            "task": "routing", "weight_mode": args.weight_mode,
            "min_weight": args.min_weight,
            "both_correct_weight": args.both_correct_weight,
        },
    )


if __name__ == "__main__":
    main()

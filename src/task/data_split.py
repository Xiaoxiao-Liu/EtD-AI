"""Create the immutable PICO-level train/validation/test split.

Run this step before rubric scoring so exploratory analysis and supervision
construction can exclude the held-out test partition from the start.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


ETD_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ETD_ROOT / "dataset" / "pico_sections_icd11.json"
DEFAULT_OUTPUT = ETD_ROOT / "dataset" / "train" / "sft" / "pico_splits.json"


def split_picos(
    picos: list[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Return a deterministic ``source_file -> split`` mapping."""
    rng = random.Random(seed)
    shuffled = sorted(picos)
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * ratios[0])
    n_val = int(len(shuffled) * ratios[1])
    return {
        pico: (
            "train"
            if index < n_train
            else "val"
            if index < n_train + n_val
            else "test"
        )
        for index, pico in enumerate(shuffled)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing frozen split map.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"split already frozen at {args.output}; leaving it unchanged")
        return

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if min(args.train_ratio, args.val_ratio, test_ratio) < 0:
        raise SystemExit("train/val ratios must be non-negative and sum to at most 1")

    with args.input.open(encoding="utf-8") as stream:
        records = json.load(stream)
    picos = [record.get("source_file") for record in records]
    if any(not isinstance(pico, str) or not pico for pico in picos):
        raise SystemExit(f"every record in {args.input} must have a source_file")
    if len(set(picos)) != len(picos):
        raise SystemExit(f"duplicate source_file values found in {args.input}")

    split_map = split_picos(
        picos,
        (args.train_ratio, args.val_ratio, test_ratio),
        args.seed,
    )
    payload = {
        "created_before_rubric_scoring": True,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(test_ratio, 6),
        },
        "n_picos": len(picos),
        "split_count": dict(Counter(split_map.values())),
        "split_map": split_map,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(f"wrote immutable PICO split to {args.output}")
    print(payload["split_count"])


if __name__ == "__main__":
    main()

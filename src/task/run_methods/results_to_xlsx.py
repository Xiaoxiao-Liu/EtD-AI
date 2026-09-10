#!/usr/bin/env python3
"""Aggregate per-PICO pipeline result JSONs into a single XLSX for review.

Reads ``dataset/run_methods/<pipeline>/output/<model>/*.json`` (one JSON per
PICO) and writes one XLSX row per PICO with one column block per EtD criterion:
``<criterion>_question``, ``<criterion>_gt_judgement``,
``<criterion>_ai_judgement``, ``<criterion>_ai_reason``,
``<criterion>_judgement_result`` (1=match, 0=mismatch).

Usage (from EtD/)::

    python -m src.task.run_methods.results_to_xlsx --pipeline end2end
    python -m src.task.run_methods.results_to_xlsx --pipeline rag --model gpt-4o
    python -m src.task.run_methods.results_to_xlsx --pipeline with_evidence \
        --output /tmp/with_evidence.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ETD_ROOT = Path(__file__).resolve().parents[3]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

PIPELINE_CHOICES = ("end2end", "rag", "with_evidence")


def _format_gt_judgement(gt_judgement: dict) -> str:
    parts = [f' "{key}": "{value}"' for key, value in gt_judgement.items()]
    return ",\n        ".join(parts)


def _criterion_key(criterion: str) -> str:
    return (
        criterion.strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _judgement_match(ai_judgement: str, gt_judgement: dict) -> int:
    ai_value = str(ai_judgement).strip().strip('"').strip("'")
    gt_value = next(
        (key for key, value in gt_judgement.items() if value == "●"),
        None,
    )
    return 1 if ai_value == gt_value else 0


def _json_to_row(data: dict) -> dict:
    row: dict = {"PICO_QUESTION": data.get("PICO_QUESTION", "")}
    for item in data.get("subquestion", []):
        criterion = item.get("criterion", "")
        key = _criterion_key(criterion)
        gt = item.get("gt_judgement", {})
        ai = item.get("ai_judgement", "")
        row[f"{key}_criterion"] = criterion
        row[f"{key}_question"] = item.get("question", "")
        row[f"{key}_gt_judgement"] = _format_gt_judgement(gt)
        row[f"{key}_ai_judgement"] = ai
        row[f"{key}_ai_reason"] = item.get("ai_reason", "")
        row[f"{key}_judgement_result"] = _judgement_match(ai, gt)
    return row


def aggregate(results_dir: Path, output_path: Path) -> int:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required: pip install pandas openpyxl"
        ) from exc

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    rows = [_json_to_row(_load_json(p)) for p in sorted(results_dir.glob("*.json"))]
    if not rows:
        raise RuntimeError(f"No *.json found under {results_dir}")

    pd.DataFrame(rows).to_excel(output_path, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=PIPELINE_CHOICES,
        help="Which pipeline's result JSONs to aggregate.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model subdir under run_methods/<pipeline>/output/ (default: gpt-4o).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_ETD_ROOT / "dataset" / "run_methods",
        help="Override the run_methods/ root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output XLSX (default: <results-root>/<pipeline>/output/<model>/result_<pipeline>.xlsx).",
    )
    args = parser.parse_args()

    results_dir = args.results_root / args.pipeline / "output" / args.model
    output_path = args.output or results_dir / f"result_{args.pipeline}.xlsx"

    n = aggregate(results_dir, output_path)
    print(f"Wrote {n} rows -> {output_path}")


if __name__ == "__main__":
    main()

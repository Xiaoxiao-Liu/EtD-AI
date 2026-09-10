"""Merge per-record rubric JSON files back into a scored JSONL.

After ``judge_rubric`` writes one file per (judge, model, PICO, section) under::

    dataset/score_rubric/judge/output/<prompt_version>/<judge_model>/<record.model_name>/
        <source_stem>/section_<idx>.json

this script copies the LLM-judge score fields into each line of
``all_models.scored.jsonl`` (or another input JSONL) and writes a merged
copy under ``dataset/score_rubric/merge/output/``.

Example::

    python -m src.task.score_rubric.merge_rubric \\
        --input dataset/score_rubric/score/output/all_models.scored.jsonl \\
        --judge-model gpt-5.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from src.task.score_rubric.judge_rubric import JUDGE_OUTPUT_FIELDS, rubric_path, record_has_all_judge_fields

ETD_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ETD_ROOT / "dataset"
DEFAULT_INPUT = DATASET_DIR / "score_rubric" / "score" / "output" / "all_models.scored.jsonl"
DEFAULT_RUBRIC_DIR = DATASET_DIR / "score_rubric" / "judge" / "output"
DEFAULT_OUTPUT = DATASET_DIR / "score_rubric" / "merge" / "output" / "all_models.scored.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def merge_record(
    rec: dict[str, Any],
    *,
    rubric_dir: Path,
    judge_model: str,
) -> tuple[dict[str, Any], bool]:
    """Return (updated_record, found_rubric_file)."""
    model_name = rec.get("model_name")
    src = rec.get("pico_source_file")
    sidx = rec.get("section_index")
    if not isinstance(model_name, str) or not isinstance(src, str):
        return rec, False
    if not isinstance(sidx, int):
        return rec, False

    source_stem = src.rsplit(".", 1)[0]
    path = rubric_path(rubric_dir, judge_model, model_name, source_stem, sidx)
    if not path.is_file():
        return rec, False

    rubric = load_json(path)
    if not record_has_all_judge_fields(rubric, judge_model):
        return rec, False

    updated = dict(rec)
    for field in JUDGE_OUTPUT_FIELDS:
        if field in rubric:
            updated[field] = rubric[field]
    return updated, True


def merge_file(
    input_path: Path,
    output_path: Path,
    *,
    rubric_dir: Path,
    judge_model: str,
) -> dict[str, int]:
    stats = {"read": 0, "wrote": 0, "merged": 0, "missing_rubric": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            stats["read"] += 1
            rec = json.loads(line)
            updated, found = merge_record(
                rec, rubric_dir=rubric_dir, judge_model=judge_model
            )
            if found:
                stats["merged"] += 1
            else:
                stats["missing_rubric"] += 1
            fout.write(json.dumps(updated, ensure_ascii=False) + "\n")
            stats["wrote"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Scored JSONL to enrich "
            "(default: dataset/score_rubric/score/output/all_models.scored.jsonl)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL "
            "(default: dataset/score_rubric/merge/output/all_models.scored.jsonl)."
        ),
    )
    parser.add_argument(
        "--rubric-dir",
        default=str(DEFAULT_RUBRIC_DIR),
        help="Rubric JSON root (default: dataset/score_rubric/judge/output).",
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Judge model subdir name (e.g. gpt-5.5).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (ETD_ROOT / input_path).resolve()
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    if not output_path.is_absolute():
        output_path = (ETD_ROOT / output_path).resolve()
    rubric_dir = Path(args.rubric_dir)
    if not rubric_dir.is_absolute():
        rubric_dir = (ETD_ROOT / rubric_dir).resolve()

    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    stats = merge_file(
        input_path,
        output_path,
        rubric_dir=rubric_dir,
        judge_model=args.judge_model,
    )
    print(f"input  -> {input_path}")
    print(f"output -> {output_path}")
    print(f"judge_model -> {args.judge_model}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

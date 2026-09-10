#!/usr/bin/env python3
"""Score policy-workflow result JSONs and summarize action usage by criterion.

Reads ``*.json`` under a results directory (e.g.
``dataset/run_methods/policy_workflow/output/<model>/test/``), compares each
subquestion's deployed/AI judgement with the ``●`` option in ``gt_judgement``
via :func:`src.task.run_methods.judgement.judgement_compare`, scores whether
RAG top-1 chunks match ground-truth evidence (reusing
:func:`src.task.score_rubric.score_rubric.score_evidence_correctness`),
writes scores back in place, builds a per-criterion summary DataFrame, and
saves it to CSV.

Usage (from EtD/)::

    python -m src.task.run_methods.data_preparation.results_calculation
    python -m src.task.run_methods.data_preparation.results_calculation \\
        --results-dir dataset/run_methods/policy_workflow/output/gpt-4o/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_ETD_ROOT = Path(__file__).resolve().parents[4]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

from src.common.io import read_etd_file
from src.task.run_methods.judgement import judgement_compare
from src.task.score_rubric.score_rubric import score_evidence_correctness

DEFAULT_RESULTS_DIR = (
    _ETD_ROOT
    / "dataset"
    / "run_methods"
    / "policy_workflow"
    / "output"
    / "gpt-4o"
    / "test"
)
DEFAULT_DATASET = _ETD_ROOT / "dataset" / "pico_sections_icd11.json"

# Policy workflow decision fields and their possible labels.
ACTION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("route", ("direct_answer", "retrieve_evidence")),
    ("suffice", ("sufficient", "retrieve_more")),
    ("judge_first", ("accept", "reject", "pass")),  # pass: legacy outputs
    ("judge_fallback", ("accept", "reject", "pass")),
    ("deployed_source", ("no_evidence", "rag_evidence", "needs_human_review")),
)

# RAG top-1 evidence correctness labels (same as score_rubric).
EVIDENCE_LABELS = ("exact", "partial", "miss", "na")
RAG_TOP1_PREFIXES = ("rag_top1", "rag_more_top1")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_section_index_by_criterion(dataset_path: Path) -> dict[str, dict[str, int]]:
    """Map ``source_stem -> {criterion: section_index}`` from the EtD dataset."""
    etd_data = read_etd_file(dataset_path)
    out: dict[str, dict[str, int]] = {}
    for item in etd_data:
        stem = item["source_file"].rsplit(".", 1)[0]
        for i, section in enumerate(item.get("sections", [])):
            criterion = section.get("criterion")
            if criterion:
                out.setdefault(stem, {})[criterion] = i
    return out


def _ai_judgement(item: dict) -> str | None:
    """Deployed judgement (policy workflow) or ai_judgement (other pipelines)."""
    for key in ("deployed_judgement", "ai_judgement"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _action_label(item: dict, field: str) -> str | None:
    if field == "deployed_source":
        value = item.get(field)
    else:
        value = (item.get(field) or {}).get("label")
    return str(value).strip() if value else None


def score_top1_evidence(
    chunks: list | None,
    *,
    source_stem: str,
    criterion: str,
    section_map: dict[str, dict[str, int]],
) -> tuple[Optional[int], str]:
    """Score only the top-1 retrieved chunk against gt section metadata."""
    if not chunks:
        return None, "na"

    top1 = chunks[0] if isinstance(chunks[0], dict) else None
    if not top1:
        return None, "na"

    section_index = section_map.get(source_stem, {}).get(criterion)
    if section_index is None:
        return 0, "miss"

    record = {
        "pico_source_file": f"{source_stem}.json",
        "section_index": section_index,
        "criterion": criterion,
    }
    rag_sub = {"retrieved_chunks": [top1]}
    return score_evidence_correctness(record, rag_sub)


def score_subquestion_evidence(
    item: dict,
    *,
    source_stem: str,
    section_map: dict[str, dict[str, int]],
) -> None:
    """Write RAG top-1 evidence correctness fields on *item*."""
    route = _action_label(item, "route")
    criterion = item.get("criterion", "")

    if route != "retrieve_evidence":
        for prefix in RAG_TOP1_PREFIXES:
            item[f"{prefix}_evidence_correctness"] = None
            item[f"{prefix}_evidence_correctness_label"] = "na"
        return

    score, label = score_top1_evidence(
        item.get("rag_chunks"),
        source_stem=source_stem,
        criterion=criterion,
        section_map=section_map,
    )
    item["rag_top1_evidence_correctness"] = score
    item["rag_top1_evidence_correctness_label"] = label

    if _action_label(item, "suffice") == "retrieve_more":
        score_more, label_more = score_top1_evidence(
            item.get("rag_chunks_more"),
            source_stem=source_stem,
            criterion=criterion,
            section_map=section_map,
        )
        item["rag_more_top1_evidence_correctness"] = score_more
        item["rag_more_top1_evidence_correctness_label"] = label_more
    else:
        item["rag_more_top1_evidence_correctness"] = None
        item["rag_more_top1_evidence_correctness_label"] = "na"


def score_file(
    data: dict,
    *,
    section_map: dict[str, dict[str, int]],
) -> tuple[int, int]:
    """Update judgement_result and evidence scores. Returns (yes_count, no_count)."""
    source_stem = data.get("source_file", "")
    yes_count = no_count = human_review_count = 0
    for item in data.get("subquestion", []):
        score_subquestion_evidence(item, source_stem=source_stem, section_map=section_map)

        gt = item.get("gt_judgement")
        if not isinstance(gt, dict):
            item["judgement_result"] = "no"
            no_count += 1
            continue

        if item.get("deployed_source") == "needs_human_review":
            item["judgement_result"] = "needs_human_review"
            human_review_count += 1
            no_count += 1
            continue

        ai = _ai_judgement(item)
        result = judgement_compare(ai or "", gt)
        item["judgement_result"] = result
        if result == "yes":
            yes_count += 1
        else:
            no_count += 1
    return yes_count, no_count


def _iter_subquestion_rows(data: dict) -> list[dict]:
    rows: list[dict] = []
    for item in data.get("subquestion", []):
        row: dict = {
            "criterion": item.get("criterion", ""),
            "judgement_result": item.get("judgement_result"),
        }
        for field, _ in ACTION_SPECS:
            row[field] = _action_label(item, field)
        for prefix in RAG_TOP1_PREFIXES:
            row[f"{prefix}_label"] = item.get(f"{prefix}_evidence_correctness_label")
        rows.append(row)
    return rows


def build_action_stats_df(all_rows: list[dict]):
    """Aggregate action and RAG top-1 evidence counts per criterion."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required: pip install pandas") from exc

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    count_cols: list[str] = []
    for field, labels in ACTION_SPECS:
        for label in labels:
            col = f"{field}__{label}"
            count_cols.append(col)
            df[col] = (df[field] == label).astype(int)

    df["judgement_yes"] = (df["judgement_result"] == "yes").astype(int)
    df["judgement_no"] = (df["judgement_result"] == "no").astype(int)

    evidence_count_cols: list[str] = []
    for prefix in RAG_TOP1_PREFIXES:
        for label in EVIDENCE_LABELS:
            col = f"{prefix}__{label}"
            evidence_count_cols.append(col)
            df[col] = (df[f"{prefix}_label"] == label).astype(int)

    agg = df.groupby("criterion", sort=True).agg(
        n=("criterion", "size"),
        **{col: (col, "sum") for col in count_cols},
        judgement_yes=("judgement_yes", "sum"),
        judgement_no=("judgement_no", "sum"),
        **{col: (col, "sum") for col in evidence_count_cols},
    )
    agg["accuracy"] = agg["judgement_yes"] / agg["n"].replace(0, float("nan"))

    for prefix in RAG_TOP1_PREFIXES:
        scored = agg["n"] - agg[f"{prefix}__na"]
        agg[f"{prefix}_exact_rate"] = (
            agg[f"{prefix}__exact"] / scored.replace(0, float("nan"))
        )

    ordered = [
        "n",
        *count_cols,
        "judgement_yes",
        "judgement_no",
        "accuracy",
        *evidence_count_cols,
        *[f"{p}_exact_rate" for p in RAG_TOP1_PREFIXES],
    ]
    return agg[ordered].reset_index()


def score_directory(
    results_dir: Path,
    *,
    dataset_path: Path,
) -> tuple[dict[str, dict], Any]:
    """Score all JSON files, write back, return (file summary, action stats df)."""
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise RuntimeError(f"No *.json found under {results_dir}")

    section_map = build_section_index_by_criterion(dataset_path)
    summary: dict[str, dict] = {}
    all_rows: list[dict] = []
    for path in paths:
        data = _load_json(path)
        yes_count, no_count = score_file(data, section_map=section_map)
        all_rows.extend(_iter_subquestion_rows(data))
        _save_json(path, data)
        n = yes_count + no_count
        summary[path.name] = {
            "yes": yes_count,
            "no": no_count,
            "accuracy": yes_count / n if n else 0.0,
        }

    action_df = build_action_stats_df(all_rows)
    return summary, action_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory of per-PICO JSON files (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"EtD PICO dataset for gt section_index (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Path for the action-stats CSV "
            "(default: <results-dir>/action_stats_by_criterion.csv)."
        ),
    )
    args = parser.parse_args()
    output_csv = args.output_csv or (
        args.results_dir / "action_stats_by_criterion.csv"
    )

    summary, action_df = score_directory(
        args.results_dir,
        dataset_path=args.dataset,
    )
    total_yes = sum(s["yes"] for s in summary.values())
    total_no = sum(s["no"] for s in summary.values())
    total = total_yes + total_no
    acc = total_yes / total if total else 0.0

    print(f"Scored {len(summary)} files under {args.results_dir}")
    for name, stats in summary.items():
        print(
            f"  {name}: yes={stats['yes']} no={stats['no']} "
            f"acc={stats['accuracy']:.1%}"
        )
    print(f"Overall: yes={total_yes} no={total_no} acc={acc:.1%}")

    print("\n--- Action counts by criterion ---")
    if action_df.empty:
        print("(no subquestions found)")
    else:
        import pandas as pd

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        action_df.to_csv(output_csv, index=False)
        print(f"Saved -> {output_csv}")

        with pd.option_context(
            "display.max_columns", None,
            "display.width", 240,
            "display.float_format", lambda x: f"{x:.1%}" if 0 <= x <= 1 else f"{x:.0f}",
        ):
            print(action_df.to_string(index=False))


if __name__ == "__main__":
    main()

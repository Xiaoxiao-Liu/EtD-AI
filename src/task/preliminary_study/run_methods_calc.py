"""Aggregate gpt-4o preliminary-study metrics by EtD criterion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ETD_ROOT = Path(__file__).resolve().parents[3]
SCORED_JSONL = (
    ETD_ROOT / "dataset/score_rubric/merge/output/all_models.scored.jsonl"
)
OUTPUT_DIR = ETD_ROOT / "dataset/preliminary_study/output"
SPLITS_FILE = ETD_ROOT / "dataset/train/sft/pico_splits.json"
OUTPUT_CSV = OUTPUT_DIR / "gpt-4o_judgement_correctness_by_criterion.csv"
OUTPUT_RAG_LABEL_CSV = (
    OUTPUT_DIR / "gpt-4o_evidence_correctness_rag_label_by_criterion.csv"
)
MODEL_NAME = "gpt-4o"

RAG_EVIDENCE_LABELS = ("exact", "partial", "miss", "na")

CORRECTNESS_COLUMNS = {
    "no_evidence": "no_evidence_judgement_correctness",
    "rag_evidence": "rag_judgement_correctness",
    "gt_evidence": "with_evidence_judgement_correctness",
}


def _load_train_picos() -> set[str]:
    payload = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    split_map = payload.get("split_map", payload)
    if not isinstance(split_map, dict):
        raise SystemExit(f"invalid PICO split map: {SPLITS_FILE}")
    return {pico for pico, split in split_map.items() if split == "train"}


def load_gpt4o_correctness(path: Path) -> pd.DataFrame:
    train_picos = _load_train_picos()
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("model_name") != MODEL_NAME:
                continue
            if rec.get("pico_source_file") not in train_picos:
                continue
            rows.append(
                {
                    "criterion": rec["criterion"],
                    **{
                        col: rec[field]
                        for col, field in CORRECTNESS_COLUMNS.items()
                    },
                }
            )
    return pd.DataFrame(rows)


def load_gpt4o_rag_evidence_labels(path: Path) -> pd.DataFrame:
    train_picos = _load_train_picos()
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("model_name") != MODEL_NAME:
                continue
            if rec.get("pico_source_file") not in train_picos:
                continue
            rows.append(
                {
                    "criterion": rec["criterion"],
                    "evidence_correctness_rag_label": rec.get(
                        "evidence_correctness_rag_label"
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_rag_labels_by_criterion(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["criterion", "evidence_correctness_rag_label"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    for label in RAG_EVIDENCE_LABELS:
        if label not in summary.columns:
            summary[label] = 0
    summary = summary[list(RAG_EVIDENCE_LABELS)].astype(int)
    return summary.sort_index()


def summarize_by_criterion(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("criterion", sort=False)[list(CORRECTNESS_COLUMNS)]
        .sum()
        .astype(int)
    )
    return summary.sort_index()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rag_df = load_gpt4o_rag_evidence_labels(SCORED_JSONL)
    if rag_df.empty:
        raise SystemExit(f"No rows found for model_name={MODEL_NAME!r}")

    rag_summary = summarize_rag_labels_by_criterion(rag_df)
    print(f"model_name={MODEL_NAME}, evidence_correctness_rag_label")
    print(f"n_rows={len(rag_df)}")
    print()
    print(rag_summary.to_string())
    print()

    df = load_gpt4o_correctness(SCORED_JSONL)
    summary = summarize_by_criterion(df)
    print(f"model_name={MODEL_NAME}, judgement correctness")
    print(f"n_rows={len(df)}")
    print()
    print(summary.to_string())
    print()

    rag_summary.to_csv(OUTPUT_RAG_LABEL_CSV, index_label="criterion")
    summary.to_csv(OUTPUT_CSV, index_label="criterion")
    print(f"saved: {OUTPUT_RAG_LABEL_CSV}")
    print(f"saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

"""Compute rubric sub-scores for extracted samples.

This script reads records from ``dataset/score_rubric/extract/output/all_models.jsonl``
and appends:

- judgement correctness (0/1, strict match) for:
  - no_evidence_judgement
  - rag_judgement
  - with_evidence_judgement
- evidence correctness (RAG-only, 0/1/2), based on retrieved chunk metadata:
  - 2: exact hit    (source_file + section_index + criterion all match)
  - 1: partial hit  (source_file matches and one of section_index / criterion matches)
  - 0: miss

Output is written as JSONL to ``dataset/score_rubric/score/output/``.

Example:
    python -m src.task.score_rubric.score_rubric --input dataset/score_rubric/extract/output/all_models.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

ETD_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ETD_ROOT / "dataset"
RESULTS_DIR = DATASET_DIR / "run_methods"
DEFAULT_INPUT = DATASET_DIR / "score_rubric" / "extract" / "output" / "all_models.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "score_rubric" / "score" / "output" / "all_models.scored.jsonl"

JUDGEMENT_FIELDS = (
    "no_evidence_judgement",
    "rag_judgement",
    "with_evidence_judgement",
)
JUDGEMENT_BULLETS = ("●", "○")
SPACE_RE = re.compile(r"\s+")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: Optional[str]) -> Optional[str]:
    """Normalize judgement-like text for matching."""
    if text is None or not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None

    # Reuse the same cleanup logic style used in extract_sample.py.
    if any(b in s for b in JUDGEMENT_BULLETS):
        if ":" in s:
            s = s.split(":", 1)[0]
        for b in JUDGEMENT_BULLETS:
            s = s.replace(b, "")
    s = s.strip()

    while s and s[0] in "\"'":
        s = s[1:]
    while s and s[-1] in "\"'":
        s = s[:-1]
    s = s.strip()
    if not s:
        return None

    s = s.replace("’", "'").replace("`", "'")
    s = SPACE_RE.sub(" ", s).strip().lower()
    return s or None


def score_judgement_correctness(
    pred: Optional[str],
    gold: Optional[str],
) -> Optional[int]:
    """Score judgement correctness by strict exact match only.

    Notes:
    - "probably yes" and "yes" are treated as different labels.
    - Exact textual match -> 1, all other non-null cases -> 0.
    """
    p = normalize_text(pred)
    g = normalize_text(gold)
    if p is None or g is None:
        return None
    if p == g:
        return 1

    return 0


def extract_rag_subquestion(
    record: dict[str, Any],
    rag_cache: dict[Path, Optional[dict[str, Any]]],
    default_model: Optional[str],
) -> Optional[dict[str, Any]]:
    """Load matching RAG subquestion payload for one DPO record."""
    source_file = record.get("pico_source_file")
    section_index = record.get("section_index")
    if not isinstance(source_file, str) or not isinstance(section_index, int):
        return None

    source_stem = source_file.rsplit(".", 1)[0]
    model = record.get("model_name") or default_model
    if not isinstance(model, str) or not model:
        return None

    rag_path = RESULTS_DIR / "rag" / "output" / model / f"{source_stem}_rag.json"
    if rag_path not in rag_cache:
        rag_cache[rag_path] = load_json(rag_path) if rag_path.exists() else None
    payload = rag_cache[rag_path]
    if not payload:
        return None

    subs = payload.get("subquestion", [])
    if not isinstance(subs, list) or section_index >= len(subs) or section_index < 0:
        return None
    sub = subs[section_index]
    return sub if isinstance(sub, dict) else None


def score_evidence_correctness(
    record: dict[str, Any],
    rag_sub: Optional[dict[str, Any]],
) -> tuple[Optional[int], str]:
    """Score RAG evidence correctness from retrieved chunk metadata."""
    if rag_sub is None:
        return None, "na"

    chunks = rag_sub.get("retrieved_chunks")
    if not isinstance(chunks, list):
        return 0, "miss"

    target_source = record.get("pico_source_file")
    target_idx = record.get("section_index")
    target_criterion = record.get("criterion")
    if not isinstance(target_source, str) or not isinstance(target_idx, int):
        return 0, "miss"

    exact = False
    partial = False
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        ch_source = ch.get("source_file")
        ch_idx = ch.get("section_index")
        ch_criterion = ch.get("criterion")

        if ch_source != target_source:
            continue
        source_match = True
        idx_match = ch_idx == target_idx
        criterion_match = ch_criterion == target_criterion

        if source_match and idx_match and criterion_match:
            exact = True
            break
        if source_match and (idx_match or criterion_match):
            partial = True

    if exact:
        return 2, "exact"
    if partial:
        return 1, "partial"
    return 0, "miss"


def score_records(
    input_path: Path,
    output_path: Path,
    default_model: Optional[str],
) -> tuple[int, int]:
    rag_cache: dict[Path, Optional[dict[str, Any]]] = {}

    read_count = 0
    write_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            read_count += 1
            rec = json.loads(line)

            gold = rec.get("gt_judgement_label")
            for field in JUDGEMENT_FIELDS:
                score = score_judgement_correctness(rec.get(field), gold)
                rec[f"{field}_correctness"] = score

            rag_sub = extract_rag_subquestion(rec, rag_cache, default_model)
            ev_score, ev_label = score_evidence_correctness(rec, rag_sub)
            rec["evidence_correctness_rag"] = ev_score
            rec["evidence_correctness_rag_label"] = ev_label

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            write_count += 1

    return read_count, write_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input JSONL path (default: dataset/score_rubric/extract/output/all_models.jsonl).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL path "
            "(default: dataset/score_rubric/score/output/all_models.scored.jsonl)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Fallback model name for locating run_methods/rag/output/<model>/ files "
            "when record.model_name is missing."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    read_count, write_count = score_records(
        input_path=input_path,
        output_path=output_path,
        default_model=args.model,
    )
    print(f"read={read_count}, wrote={write_count}")
    print(f"output -> {output_path}")


if __name__ == "__main__":
    main()


"""Compute policy labels from rubric scores for SFT/DPO data construction.

This script reads records from ``dataset/score_rubric/merge/output/all_models.scored.jsonl``
(produced by ``src.task.score_rubric.score_rubric`` +
``src.task.score_rubric.judge_rubric`` + ``src.task.score_rubric.merge_rubric``)
and appends, for each record:

Normalized weighted scores (each in [0, 1]):
- ``no_score``   : score of the Closed-book (direct) output
- ``rag_score``  : score of the RAG output
- ``gt_score``   : score of the with-evidence (oracle) output

Hard-gate booleans (True = pass):
- ``no_evidence_hard_gate`` : ``no_evidence_judgement_correctness != 0``
- ``rag_hard_gate``         : ``rag_judgement_correctness != 0``
- ``gt_hard_gate``          : ``with_evidence_judgement_correctness != 0``
- ``rag_sufficiency_gate``  : RAG passes general gate AND
                              ``evidence_correctness_rag != 0`` AND
                              ``rag_reasoning_grounding != 0``
- ``no_evidence_direct_gate``: Closed-book passes general gate
                              (i.e. ``no_evidence_judgement_correctness != 0``).
                              Note: previously also required
                              ``no_evidence_confidence_calibration != 0``, but
                              that field is missing on ~55% of records and
                              suppressed direct_answer to ~16%; with the
                              calibration requirement dropped, direct_answer
                              rises to ~41%, consistent with the share of
                              records where no_score >= rag_score.

Four policy labels:
- ``initial_action_label``:
      ``direct_answer`` whenever the Closed-book output is correct.  In
      particular, if both Closed-book and Open-book are correct, prefer the cheaper
      direct action.  Use ``retrieve_evidence`` when the direct output fails.
- ``evidence_sufficiency_label``:
      ``sufficient`` if RAG passes the sufficiency gate AND
      ``gt_score - rag_score < evidence_sufficiency_threshold`` (default 0.3),
      else ``retrieve_more``.
- ``deployable_selected_source`` (+ ``low_quality_case``):
      Pick the source (``no_evidence`` / ``rag_evidence``) that passes its
      general gate and has the higher weighted score. If neither passes,
      still pick the higher-scoring one but flag ``low_quality_case = true``.
- ``oracle_gap_label``:
      ``large_oracle_gap`` if ``gt_score - max(no_score, rag_score) >=
      oracle_gap_threshold`` (default 0.5), else ``small_oracle_gap``.

See ``paper/rubrics_label_instruction.txt`` for the original spec.

Example:
    python -m src.task.score_rubric.label_policy \\
        --input dataset/score_rubric/merge/output/all_models.scored.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

ETD_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ETD_ROOT / "dataset"
DEFAULT_INPUT = DATASET_DIR / "score_rubric" / "merge" / "output" / "all_models.scored.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "score_rubric" / "label" / "output" / "all_models.labeled.jsonl"


# ---------------------------------------------------------------------------
# Field metadata: (record_key, max_raw_value)
# Each rubric field is normalized to [0, 1] by dividing by its max.
# ---------------------------------------------------------------------------
# judgement_correctness comes from strict label match, so max is 1 (binary).
# evidence_correctness_rag is 0/1/2 (miss/partial/exact), so max is 2.
# All other rubric dimensions are LLM-judged 0/1/2, so max is 2.

# no_score: 4 dims (Closed-book has no "grounding")
NO_SCORE_COMPONENTS: tuple[tuple[str, float, float], ...] = (
    ("no_evidence_judgement_correctness", 0.40, 1.0),  # judgment correctness
    ("no_evidence_reasoning_relevance",   0.25, 2.0),  # reasoning validity
    ("no_evidence_confidence_calibration", 0.20, 2.0),  # confidence calibration
    ("no_evidence_etd_consistency",       0.15, 2.0),  # EtD consistency
)

# rag_score: 6 dims
RAG_SCORE_COMPONENTS: tuple[tuple[str, float, float], ...] = (
    ("rag_judgement_correctness", 0.35, 1.0),  # judgment correctness
    ("evidence_correctness_rag",  0.20, 2.0),  # evidence relevance
    ("rag_reasoning_grounding",   0.20, 2.0),  # evidence grounding
    ("rag_reasoning_relevance",   0.15, 2.0),  # reasoning validity
    ("rag_confidence_calibration", 0.05, 2.0),  # confidence calibration
    ("rag_etd_consistency",       0.05, 2.0),  # EtD consistency
)

# gt_score: 5 dims (with_evidence == oracle)
GT_SCORE_COMPONENTS: tuple[tuple[str, float, float], ...] = (
    ("with_evidence_judgement_correctness", 0.40, 1.0),
    ("with_evidence_reasoning_grounding",   0.25, 2.0),
    ("with_evidence_reasoning_relevance",   0.20, 2.0),
    ("with_evidence_confidence_calibration", 0.10, 2.0),
    ("with_evidence_etd_consistency",       0.05, 2.0),
)


def _to_float(v: Any) -> Optional[float]:
    """Coerce a rubric value to float, returning None for missing/invalid."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    return None


def compute_weighted_score(
    record: dict[str, Any],
    components: tuple[tuple[str, float, float], ...],
) -> tuple[Optional[float], list[str]]:
    """Compute a normalized weighted score in [0, 1].

    Missing components are treated as 0 (their weight is still consumed),
    but the list of missing field names is returned for diagnostics. If
    every component is missing, returns ``None``.
    """
    total = 0.0
    missing: list[str] = []
    present_any = False
    for key, weight, max_val in components:
        raw = _to_float(record.get(key))
        if raw is None:
            missing.append(key)
            continue
        present_any = True
        total += weight * (raw / max_val)
    if not present_any:
        return None, missing
    if total < 0.0:
        total = 0.0
    if total > 1.0:
        total = 1.0
    return total, missing


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------

def _nonzero(v: Any) -> bool:
    """True iff value is a numeric, non-None, non-zero rubric score."""
    f = _to_float(v)
    return f is not None and f != 0.0


def compute_gates(record: dict[str, Any]) -> dict[str, bool]:
    """Compute all hard gates per the instruction file."""
    no_ev_gate = _nonzero(record.get("no_evidence_judgement_correctness"))
    rag_gate = _nonzero(record.get("rag_judgement_correctness"))
    gt_gate = _nonzero(record.get("with_evidence_judgement_correctness"))

    rag_sufficiency_gate = (
        rag_gate
        and _nonzero(record.get("evidence_correctness_rag"))
        and _nonzero(record.get("rag_reasoning_grounding"))
    )

    no_ev_direct_gate = no_ev_gate

    return {
        "no_evidence_hard_gate": no_ev_gate,
        "rag_hard_gate": rag_gate,
        "gt_hard_gate": gt_gate,
        "rag_sufficiency_gate": rag_sufficiency_gate,
        "no_evidence_direct_gate": no_ev_direct_gate,
    }


# ---------------------------------------------------------------------------
# Policy labels
# ---------------------------------------------------------------------------

def _verifier_label(
    correctness: Any,
    reasoning_relevance: Any,
    *,
    evidence_correctness: Any = None,
    reasoning_grounding: Any = None,
    require_evidence_support: bool = False,
) -> str:
    """Return ``accept`` only for a correct, supported candidate.

    A Closed-book candidate must have a correct judgment and a relevant
    rationale.  A RAG candidate must additionally use relevant evidence and
    ground its reasoning in that evidence.  Missing rubric values fail closed.
    """
    if not _nonzero(correctness) or not _nonzero(reasoning_relevance):
        return "reject"
    if require_evidence_support and (
        not _nonzero(evidence_correctness) or not _nonzero(reasoning_grounding)
    ):
        return "reject"
    return "accept"


def label_record(
    record: dict[str, Any],
    evidence_sufficiency_threshold: float,
    oracle_gap_threshold: float,
) -> dict[str, Any]:
    """Compute weighted scores, gates, and 4 policy labels for one record."""
    no_score, no_missing = compute_weighted_score(record, NO_SCORE_COMPONENTS)
    rag_score, rag_missing = compute_weighted_score(record, RAG_SCORE_COMPONENTS)
    gt_score, gt_missing = compute_weighted_score(record, GT_SCORE_COMPONENTS)

    gates = compute_gates(record)

    out: dict[str, Any] = {
        "no_score": no_score,
        "rag_score": rag_score,
        "gt_score": gt_score,
        **gates,
    }

    # Unknown reference correctness is not an observed failure. Keep diagnostics,
    # but emit no policy targets; existing SFT/DPO builders then exclude the row.
    correctness_keys = ("no_evidence_judgement_correctness", "rag_judgement_correctness",
                        "with_evidence_judgement_correctness")
    if all(record.get(key) is None for key in correctness_keys):
        out.update(
            label_status="missing_correctness", initial_action_label=None,
            initial_action_preference_reason="missing_correctness",
            evidence_sufficiency_label=None, no_evidence_arbit_label=None,
            rag_arbit_label=None, deployable_selected_source=None,
            low_quality_case=True, oracle_gap=None, oracle_gap_label=None,
            _missing_fields={"no_score": no_missing, "rag_score": rag_missing, "gt_score": gt_missing},
        )
        return out
    out["label_status"] = "ok"

    # --- per-candidate quality gate (JUDGE accept/reject) ---
    out["no_evidence_arbit_label"] = _verifier_label(
        record.get("no_evidence_judgement_correctness"),
        record.get("no_evidence_reasoning_relevance"),
    )
    out["rag_arbit_label"] = _verifier_label(
        record.get("rag_judgement_correctness"),
        record.get("rag_reasoning_relevance"),
        evidence_correctness=record.get("evidence_correctness_rag"),
        reasoning_grounding=record.get("rag_reasoning_grounding"),
        require_evidence_support=True,
    )

    # --- initial_action_label ---
    direct_correct = gates["no_evidence_direct_gate"]
    rag_correct = gates["rag_hard_gate"]
    if direct_correct:
        out["initial_action_label"] = "direct_answer"
        out["initial_action_preference_reason"] = (
            "both_correct_prefer_direct" if rag_correct else "direct_only_correct"
        )
    else:
        out["initial_action_label"] = "retrieve_evidence"
        out["initial_action_preference_reason"] = (
            "retrieval_only_correct" if rag_correct else "direct_failed"
        )

    # --- evidence_sufficiency_label ---
    if (
        gates["rag_sufficiency_gate"]
        and rag_score is not None
        and gt_score is not None
        and (gt_score - rag_score) < evidence_sufficiency_threshold
    ):
        out["evidence_sufficiency_label"] = "sufficient"
    else:
        out["evidence_sufficiency_label"] = "retrieve_more"

    # --- deployable_selected_source (+ low_quality_case) ---
    no_pass = gates["no_evidence_hard_gate"]
    rag_pass = gates["rag_hard_gate"]
    no_s = no_score if no_score is not None else float("-inf")
    rag_s = rag_score if rag_score is not None else float("-inf")

    if no_pass and rag_pass:
        choice = "no_evidence" if no_s >= rag_s else "rag_evidence"
        low_quality = False
    elif no_pass and not rag_pass:
        choice = "no_evidence"
        low_quality = False
    elif rag_pass and not no_pass:
        choice = "rag_evidence"
        low_quality = False
    else:
        choice = "no_evidence" if no_s >= rag_s else "rag_evidence"
        low_quality = True
    out["deployable_selected_source"] = choice
    out["low_quality_case"] = low_quality

    # --- oracle_gap_label ---
    if (
        gt_score is not None
        and (no_score is not None or rag_score is not None)
    ):
        best_deployable = max(
            v for v in (no_score, rag_score) if v is not None
        )
        gap = gt_score - best_deployable
        out["oracle_gap"] = gap
        out["oracle_gap_label"] = (
            "large_oracle_gap"
            if gap >= oracle_gap_threshold
            else "small_oracle_gap"
        )
    else:
        out["oracle_gap"] = None
        out["oracle_gap_label"] = "small_oracle_gap"

    # Diagnostics: which fields were missing per score group.
    out["_missing_fields"] = {
        "no_score": no_missing,
        "rag_score": rag_missing,
        "gt_score": gt_missing,
    }

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def label_file(
    input_path: Path,
    output_path: Path,
    evidence_sufficiency_threshold: float,
    oracle_gap_threshold: float,
) -> dict[str, int]:
    """Stream-process the input JSONL, writing labelled records to output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "read": 0,
        "wrote": 0,
        "missing_correctness": 0,
        "direct_answer": 0,
        "retrieve_evidence": 0,
        "sufficient": 0,
        "retrieve_more": 0,
        "selected_no_evidence": 0,
        "selected_rag_evidence": 0,
        "low_quality_case": 0,
        "large_oracle_gap": 0,
        "small_oracle_gap": 0,
        "no_evidence_arbit_accept": 0,
        "no_evidence_arbit_reject": 0,
        "rag_arbit_accept": 0,
        "rag_arbit_reject": 0,
    }

    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            stats["read"] += 1
            rec = json.loads(line)
            labels = label_record(
                rec,
                evidence_sufficiency_threshold=evidence_sufficiency_threshold,
                oracle_gap_threshold=oracle_gap_threshold,
            )
            rec.update(labels)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["wrote"] += 1

            if labels.get("label_status") == "missing_correctness":
                stats["missing_correctness"] += 1
                continue
            stats[labels["initial_action_label"]] += 1
            stats[labels["evidence_sufficiency_label"]] += 1
            stats[f"selected_{labels['deployable_selected_source']}"] += 1
            if labels["low_quality_case"]:
                stats["low_quality_case"] += 1
            stats[labels["oracle_gap_label"]] += 1
            stats[f"no_evidence_arbit_{labels['no_evidence_arbit_label']}"] += 1
            stats[f"rag_arbit_{labels['rag_arbit_label']}"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Input scored JSONL path "
            "(default: dataset/score_rubric/merge/output/all_models.scored.jsonl)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL path "
            "(default: dataset/score_rubric/label/output/all_models.labeled.jsonl)."
        ),
    )
    parser.add_argument(
        "--evidence-sufficiency-threshold",
        type=float,
        default=0.3,
        help="gt_score - rag_score < T => sufficient (default 0.3).",
    )
    parser.add_argument(
        "--oracle-gap-threshold",
        type=float,
        default=0.5,
        help="gt_score - max(no, rag) >= T => large_oracle_gap (default 0.5).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (ETD_ROOT / input_path).resolve()
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    if not output_path.is_absolute():
        output_path = (ETD_ROOT / output_path).resolve()

    stats = label_file(
        input_path=input_path,
        output_path=output_path,
        evidence_sufficiency_threshold=args.evidence_sufficiency_threshold,
        oracle_gap_threshold=args.oracle_gap_threshold,
    )

    print(f"input  -> {input_path}")
    print(f"output -> {output_path}")
    print("stats:")
    for k, v in stats.items():
        print(f"  {k:>22}: {v}")


if __name__ == "__main__":
    main()

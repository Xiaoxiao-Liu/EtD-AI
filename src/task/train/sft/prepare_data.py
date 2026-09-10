"""Prepare SFT training data from ``all_models.labeled.jsonl``.

Splits one labelled JSONL into three task-specific datasets — ``routing``,
``sufficiency``, ``judgment`` — matching the workflow described in
``memory/project_training_strategy.md``:

- Step 1 (routing)     : input = PICO + criterion + question
                         target = initial_action_label
- Step 2 (sufficiency) : input = PICO + criterion + question + rag_evidence
                         target = evidence_sufficiency_label  (label only —
                                  policy does NOT generate judgement text)
- Step 3 (judgment)    : input = PICO + criterion + question + rag_evidence +
                                  sufficiency_assessment (the SUFFICE label) +
                                  both executor candidates (no_evidence and rag,
                                  each with judgement + rationale)
                         target = deployable_selected_source ∈
                                  {no_evidence, rag_evidence}

The PICO-level split must already have been frozen before rubric scoring by
``scripts/prepare_pico_splits.sh``.  This module reuses that map and writes
only train/validation supervision; held-out test records are ignored.

Outputs::

    dataset/train/sft/pico_splits.json
    dataset/train/sft/<task>/output/{train,val}.jsonl
    dataset/train/sft/<task>/output/meta.json

Field whitelist enforced per task. Gold / oracle fields never reach input:
    gt_evidence, gt_reasoning, gt_judgement_label,
    with_evidence_judgement, ai_reason_with_evidence,
    all *_score / *_correctness / *_gate / oracle_gap*.

JUDGMENT may use sibling-executor outputs (no_evidence_judgement, rag_judgement,
ai_reason_e2e, ai_reason_rag) because those are produced at deployment time
in parallel mode, not gold.

Example::

    python -m src.task.train.sft.prepare_data
    bash scripts/prepare_pico_splits.sh
    python -m src.task.train.sft.prepare_data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

ETD_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = ETD_ROOT / "dataset"
DEFAULT_INPUT = DATASET_DIR / "score_rubric" / "label" / "output" / "all_models.labeled.jsonl"
DEFAULT_OUTPUT_ROOT = DATASET_DIR / "train" / "sft"
DEFAULT_SPLITS_FILE = DATASET_DIR / "train" / "sft" / "pico_splits.json"

INSTRUCTION_TAGS = {
    "routing": "[ROUTE]",
    "sufficiency": "[SUFFICE]",
    "judgment": "[JUDGE]",
}


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Per-task record builders
#
# Each builder returns a fully prepared training record, or None to drop.
# The builders intentionally use a *whitelist* of fields when populating
# ``input`` — any gold / oracle field never reaches the prompt side.
# --------------------------------------------------------------------------- #


def _common_meta(rec: dict) -> dict:
    return {
        "model_name": rec.get("model_name"),
        "pico_source_file": rec.get("pico_source_file"),
        "section_index": rec.get("section_index"),
    }


def build_routing_record(rec: dict) -> Optional[dict]:
    label = rec.get("initial_action_label")
    if label not in ("direct_answer", "retrieve_evidence"):
        return None
    return {
        "id": rec.get("id"),
        "instruction": INSTRUCTION_TAGS["routing"],
        "input": {
            "PICO": rec.get("PICO", ""),
            "criterion": rec.get("criterion", ""),
            "criterion_question": rec.get("criterion_question", ""),
        },
        "target": {
            "action": label,
        },
        "meta": {
            **_common_meta(rec),
            "no_score": rec.get("no_score"),
            "rag_score": rec.get("rag_score"),
        },
    }


def build_sufficiency_record(rec: dict) -> Optional[dict]:
    if rec.get("low_quality_case") is True:
        return None
    label = rec.get("evidence_sufficiency_label")
    if label not in ("sufficient", "retrieve_more"):
        return None

    # Avoid teaching "confidently wrong": if rag is judged sufficient
    # but its judgement is actually wrong, drop this record.
    if label == "sufficient" and rec.get("rag_judgement_correctness") == 0:
        return None

    rag_evidence = rec.get("rag_evidence")
    if not isinstance(rag_evidence, str) or not rag_evidence.strip():
        # Sufficiency must see evidence — no evidence, no decision.
        return None

    return {
        "id": rec.get("id"),
        "instruction": INSTRUCTION_TAGS["sufficiency"],
        "input": {
            "PICO": rec.get("PICO", ""),
            "criterion": rec.get("criterion", ""),
            "criterion_question": rec.get("criterion_question", ""),
            "rag_evidence": rag_evidence,
        },
        "target": {"sufficiency": label},
        "meta": {
            **_common_meta(rec),
            "rag_score": rec.get("rag_score"),
            "gt_score": rec.get("gt_score"),
        },
    }


def build_judgment_records(rec: dict) -> list[dict]:
    """Build 0, 1, or 2 single-candidate quality-gate records.

    Each record evaluates one candidate as accept/reject. The no_evidence
    candidate is evaluated without evidence context; the rag candidate
    includes rag_evidence and sufficiency_assessment.
    """
    if rec.get("low_quality_case") is True:
        return []

    results: list[dict] = []
    pico = rec.get("PICO", "")
    criterion = rec.get("criterion", "")
    criterion_question = rec.get("criterion_question", "")
    base_meta = _common_meta(rec)

    # --- no_evidence candidate ---
    no_arbit = rec.get("no_evidence_arbit_label")
    if no_arbit in ("accept", "reject"):
        no_j = rec.get("no_evidence_judgement")
        no_r = rec.get("ai_reason_e2e")
        if isinstance(no_j, str) and no_j.strip() and isinstance(no_r, str) and no_r.strip():
            results.append({
                "id": f"{rec.get('id')}::no_evidence",
                "instruction": INSTRUCTION_TAGS["judgment"],
                "input": {
                    "PICO": pico,
                    "criterion": criterion,
                    "criterion_question": criterion_question,
                    "source": "no_evidence",
                    "candidate": {
                        "judgement": no_j,
                        "rationale": no_r,
                    },
                },
                "target": {"quality": no_arbit},
                "meta": {
                    **base_meta,
                    "original_source": "no_evidence",
                    "no_score": rec.get("no_score"),
                },
            })

    # --- rag candidate ---
    rag_arbit = rec.get("rag_arbit_label")
    if rag_arbit in ("accept", "reject"):
        rag_j = rec.get("rag_judgement")
        rag_r = rec.get("ai_reason_rag")
        rag_ev = rec.get("rag_evidence")
        sufficiency = rec.get("evidence_sufficiency_label")
        if (isinstance(rag_j, str) and rag_j.strip()
                and isinstance(rag_r, str) and rag_r.strip()
                and isinstance(rag_ev, str) and rag_ev.strip()):
            results.append({
                "id": f"{rec.get('id')}::rag_evidence",
                "instruction": INSTRUCTION_TAGS["judgment"],
                "input": {
                    "PICO": pico,
                    "criterion": criterion,
                    "criterion_question": criterion_question,
                    "source": "rag_evidence",
                    "candidate": {
                        "judgement": rag_j,
                        "rationale": rag_r,
                    },
                    "rag_evidence": rag_ev,
                    "sufficiency_assessment": sufficiency or "",
                },
                "target": {"quality": rag_arbit},
                "meta": {
                    **base_meta,
                    "original_source": "rag_evidence",
                    "rag_score": rec.get("rag_score"),
                },
            })

    return results


TASK_BUILDERS: dict[str, Callable[[dict], list[dict]]] = {
    "routing": lambda rec: [r] if (r := build_routing_record(rec)) else [],
    "sufficiency": lambda rec: [r] if (r := build_sufficiency_record(rec)) else [],
    "judgment": build_judgment_records,
}

# Which field in the built record to bucket by, for label-distribution stats.
TASK_LABEL_PATHS: dict[str, tuple[str, ...]] = {
    "routing": ("target", "action"),
    "sufficiency": ("target", "sufficiency"),
    "judgment": ("target", "quality"),
}


# --------------------------------------------------------------------------- #
# Per-task driver
# --------------------------------------------------------------------------- #


def _get_nested(d: dict, path: tuple[str, ...]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def build_task_datasets(
    task: str,
    records: list[dict],
    split_map: dict[str, str],
    output_root: Path,
) -> dict[str, Any]:
    """Build and write train/validation JSONL for one task."""
    builder = TASK_BUILDERS[task]
    split_records: dict[str, list[dict]] = {"train": [], "val": []}

    for rec in records:
        pico = rec.get("pico_source_file")
        if not isinstance(pico, str):
            continue
        split = split_map.get(pico)
        if split not in split_records:
            continue
        for built in builder(rec):
            split_records[split].append(built)

    out_dir = output_root / task / "output"
    for split_name, recs in split_records.items():
        write_jsonl(recs, out_dir / f"{split_name}.jsonl")
    # A stale test file would be training-shaped held-out data; remove it.
    (out_dir / "test.jsonl").unlink(missing_ok=True)

    label_path = TASK_LABEL_PATHS[task]
    label_distribution = {
        split_name: dict(Counter(_get_nested(r, label_path) for r in recs))
        for split_name, recs in split_records.items()
    }

    meta = {
        "task": task,
        "instruction": INSTRUCTION_TAGS[task],
        "splits": {k: len(v) for k, v in split_records.items()},
        "total": sum(len(v) for v in split_records.values()),
        "label_distribution": label_distribution,
    }
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Labelled JSONL "
            "(default: dataset/score_rubric/label/output/all_models.labeled.jsonl)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root dir for per-task data (default: dataset/train/sft).",
    )
    parser.add_argument(
        "--splits-file",
        default=str(DEFAULT_SPLITS_FILE),
        help=(
            "Precomputed PICO split map created before Step 3 "
            "(default: dataset/train/sft/pico_splits.json)."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_BUILDERS.keys()),
        choices=list(TASK_BUILDERS.keys()),
        help="Subset of tasks to build (default: all).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_root)
    splits_file = Path(args.splits_file)

    records = load_records(input_path)
    print(f"loaded {len(records)} records from {input_path}")
    if not records:
        print("WARNING: input is empty. Proceeding to write zero-row splits/meta.")

    if not splits_file.exists():
        raise SystemExit(
            f"PICO split map not found: {splits_file}\n"
            "  -> run: bash scripts/prepare_pico_splits.sh before Step 3"
        )
    with splits_file.open(encoding="utf-8") as f:
        split_payload = json.load(f)
    split_map = split_payload.get("split_map")
    if not isinstance(split_map, dict):
        raise SystemExit(f"unexpected split-map format: {splits_file}")
    split_counts = Counter(split_map.values())
    print(
        f"reusing frozen PICO split -> {splits_file}\n"
        f"  train PICOs={split_counts.get('train', 0)}, "
        f"val PICOs={split_counts.get('val', 0)}, "
        f"test PICOs={split_counts.get('test', 0)}"
    )

    for task in args.tasks:
        meta = build_task_datasets(task, records, split_map, output_root)
        print(
            f"[{task}] total={meta['total']}, splits={meta['splits']}\n"
            f"        label_distribution={meta['label_distribution']}"
        )

    print("done.")


if __name__ == "__main__":
    main()

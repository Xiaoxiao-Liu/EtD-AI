"""Batch-extract DPO-ready samples by joining ``pico_sections_icd11.json``
with the end2end / rag / with_evidence result JSONs of a given model.

All records are written to a single JSONL file (one line = one record),
which is the convenient format for downstream training pipelines
(``datasets.load_dataset("json", data_files=..., split="train")``, TRL DPO,
etc.).

Each record has the following fields::

    id                        # globally unique, "<source_stem>::<idx>::<model>"
    pico_source_file          # original source_file basename in pico_sections
    section_index             # 0-based section index within the PICO
    PICO,                     # pico.Question
    criterion,                # sections[i].criterion
    criterion_question,       # sections[i].question
    rag_evidence,             # rag result's subquestion[i].retrieved_evidence
    gt_evidence,              # sections[i].research_evidence (may be dict)
    no_evidence_judgement,    # end2end ai_judgement (cleaned)
    ai_reason_e2e,            # end2end ai_reason
    rag_judgement,            # rag ai_judgement (cleaned)
    ai_reason_rag,            # rag ai_reason
    with_evidence_judgement,  # with_evidence ai_judgement (cleaned)
    ai_reason_with_evidence,  # with_evidence ai_reason
    gt_judgement_label,       # ● selection from judgement_extract
    gt_reasoning,             # sections[i].additional_considerations ("" if missing)
    model_name                # the result-model these judgements come from

By default only the development partitions (``train,val``) are emitted.  The
held-out test partition must be processed explicitly with
``--include-splits test`` and written to a separate evaluation file.

Output::

    EtD/dataset/score_rubric/extract/output/all_models.jsonl   (default; all models in one file)

By default we only emit a PICO when ALL three method results exist for the
chosen model (matching the original instruction "三种 method 都有的"). Pass
``--allow-partial`` to also emit PICOs with missing methods (missing fields
become ``null``).

Examples::

    python -m src.task.score_rubric.data_preparation.extract_sample --models gpt-4o
    python -m src.task.score_rubric.data_preparation.extract_sample --models gpt-4o,gpt-5.5
    python -m src.task.score_rubric.data_preparation.extract_sample --pico a_daily_lower_dose_...for_whom
    python -m src.task.score_rubric.data_preparation.extract_sample --allow-partial --limit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

ETD_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = ETD_ROOT / "dataset"
RESULTS_DIR = DATASET_DIR / "run_methods"
PICO_SECTIONS_FILE = DATASET_DIR / "pico_sections_icd11.json"
DEFAULT_OUTPUT = DATASET_DIR / "score_rubric" / "extract" / "output" / "all_models.jsonl"
DEFAULT_SPLITS_FILE = DATASET_DIR / "train" / "sft" / "pico_splits.json"

JUDGEMENT_BULLETS = ("●", "○")


def clean_judgement(val: Optional[str]) -> Optional[str]:
    """Strip dataset bullet artefacts (●, ○) and surrounding quotes from a
    raw ai_judgement string, keeping only the textual label.

    Examples (left = raw model output, right = cleaned)::

        "Probably yes"           -> "Probably yes"
        "●"                      -> None
        "Varies: '●'"            -> "Varies"
        "Don't know\\": '●'"     -> "Don't know"
    """
    if val is None or not isinstance(val, str):
        return val
    s = val.strip()

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
    return s if s else None


def parse_gt_label(
    judgement_extract: list[str],
) -> Union[str, list[str], None]:
    """Extract the selected option(s) (those prefixed with ●) from a
    ``judgement_extract`` list. Returns ``str`` for single-select,
    ``list[str]`` for multi-select, or ``None`` if nothing selected.
    """
    selected_raw = [
        item.lstrip("●").strip()
        for item in judgement_extract or []
        if isinstance(item, str) and item.lstrip().startswith("●")
    ]
    selected = [s for s in (clean_judgement(x) for x in selected_raw) if s]
    if not selected:
        return None
    if len(selected) == 1:
        return selected[0]
    return selected


def load_json(p: Path):
    with p.open() as f:
        return json.load(f)


def load_split_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(
            f"PICO split map not found: {path}\n"
            "  -> run: bash scripts/prepare_pico_splits.sh before Step 3"
        )
    payload = load_json(path)
    split_map = payload.get("split_map") if isinstance(payload, dict) else None
    if not isinstance(split_map, dict):
        raise SystemExit(f"unexpected split-map format: {path}")
    return split_map


def _check_alignment(
    source_stem: str,
    sections: list[dict],
    name: str,
    payload: Optional[dict],
) -> bool:
    """Return True if the result payload aligns with the source sections by
    (criterion, question). Logs warnings to stderr on mismatch."""
    if payload is None:
        return True
    subs = payload.get("subquestion", [])
    if len(subs) != len(sections):
        print(
            f"  [warn] {name} subquestion length mismatch for {source_stem}: "
            f"sections={len(sections)}, {name}={len(subs)}; treating as missing",
            file=sys.stderr,
        )
        return False
    for i, section in enumerate(sections):
        sub = subs[i]
        if sub.get("criterion") != section.get("criterion") or sub.get(
            "question"
        ) != section.get("question"):
            print(
                f"  [warn] {name} subq[{i}] mismatch in {source_stem}: "
                f"{sub.get('criterion')!r}/{sub.get('question')!r} vs "
                f"{section.get('criterion')!r}/{section.get('question')!r}",
                file=sys.stderr,
            )
            return False
    return True


def extract_one(
    item: dict,
    *,
    model: str,
    allow_partial: bool,
) -> Optional[list[dict]]:
    """Build records for one PICO. Returns ``None`` to indicate skip."""
    source_file = item["source_file"]
    source_stem = source_file.split(".")[0]
    paths = {
        "end2end": RESULTS_DIR / "end2end" / "output" / model / f"{source_stem}_e2e.json",
        "rag": RESULTS_DIR / "rag" / "output" / model / f"{source_stem}_rag.json",
        "with_evidence": (
            RESULTS_DIR / "with_evidence" / "output" / model / f"{source_stem}_with_evidence.json"
        ),
    }
    payloads: dict[str, Optional[dict]] = {}
    for name, path in paths.items():
        payloads[name] = load_json(path) if path.exists() else None

    if not allow_partial and any(v is None for v in payloads.values()):
        return None

    sections = item["sections"]
    for name, payload in payloads.items():
        if not _check_alignment(source_stem, sections, name, payload):
            payloads[name] = None
    if not allow_partial and any(v is None for v in payloads.values()):
        return None

    records: list[dict] = []
    for i, section in enumerate(sections):
        e2e_sub = payloads["end2end"]["subquestion"][i] if payloads["end2end"] else None
        rag_sub = payloads["rag"]["subquestion"][i] if payloads["rag"] else None
        we_sub = (
            payloads["with_evidence"]["subquestion"][i]
            if payloads["with_evidence"]
            else None
        )

        records.append(
            {
                "id": f"{source_stem}::{i}::{model}",
                "pico_source_file": source_file,
                "section_index": i,
                "PICO": item["pico"]["Question"],
                "criterion": section.get("criterion"),
                "criterion_question": section.get("question"),
                "rag_evidence": rag_sub.get("retrieved_evidence") if rag_sub else None,
                "gt_evidence": section.get("research_evidence"),
                "no_evidence_judgement": clean_judgement(
                    e2e_sub.get("ai_judgement") if e2e_sub else None
                ),
                "ai_reason_e2e": e2e_sub.get("ai_reason") if e2e_sub else None,
                "rag_judgement": clean_judgement(
                    rag_sub.get("ai_judgement") if rag_sub else None
                ),
                "ai_reason_rag": rag_sub.get("ai_reason") if rag_sub else None,
                "with_evidence_judgement": clean_judgement(
                    we_sub.get("ai_judgement") if we_sub else None
                ),
                "ai_reason_with_evidence": (
                    we_sub.get("ai_reason") if we_sub else None
                ),
                "gt_judgement_label": parse_gt_label(
                    section.get("judgement_extract", [])
                ),
                "gt_reasoning": section.get("additional_considerations") or "",
                "model_name": model,
            }
        )
    return records


def parse_models_arg(models: str | None, model: str | None) -> list[str]:
    """Resolve --models (comma-separated) or legacy --model."""
    if models:
        return [m.strip() for m in models.split(",") if m.strip()]
    if model:
        return [model.strip()]
    return ["gpt-4o"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names under run_methods/<method>/output/ (e.g. gpt-4o,gpt-5.5).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Single model alias for --models (deprecated; prefer --models).",
    )
    parser.add_argument(
        "--pico",
        default=None,
        help="If set, only process the PICO whose source_file stem matches.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL path (default: dataset/score_rubric/extract/output/all_models.jsonl).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Also emit PICOs missing some method results (null filled).",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=DEFAULT_SPLITS_FILE,
        help=f"Precomputed PICO split map (default: {DEFAULT_SPLITS_FILE}).",
    )
    parser.add_argument(
        "--include-splits",
        default="train,val",
        help=(
            "Comma-separated splits to emit (default: train,val). "
            "Use 'test' only for a separate final-evaluation pipeline."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after emitting this many PICOs per model (smoke test).",
    )
    args = parser.parse_args()

    include_splits = {
        value.strip() for value in args.include_splits.split(",") if value.strip()
    }
    valid_splits = {"train", "val", "test"}
    if not include_splits or not include_splits <= valid_splits:
        parser.error(
            f"--include-splits must be a non-empty subset of {sorted(valid_splits)}"
        )
    split_map = load_split_map(args.splits_file)

    model_list = parse_models_arg(args.models, args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pico_data = load_json(PICO_SECTIONS_FILE)

    scanned = pico_emitted = pico_skipped = record_count = 0
    seen_ids: set[str] = set()
    with out_path.open("w", encoding="utf-8") as fout:
        for model in model_list:
            model_pico_emitted = 0
            for item in pico_data:
                source_file = item.get("source_file")
                split = split_map.get(source_file)
                if split is None:
                    raise SystemExit(
                        f"PICO {source_file!r} is missing from {args.splits_file}"
                    )
                if split not in include_splits:
                    continue
                source_stem = item["source_file"].split(".")[0]
                if args.pico and source_stem != args.pico:
                    continue
                scanned += 1
                records = extract_one(
                    item, model=model, allow_partial=args.allow_partial
                )
                if records is None:
                    pico_skipped += 1
                    continue
                for rec in records:
                    rec["data_split"] = split
                    if rec["id"] in seen_ids:
                        print(
                            f"  [warn] duplicate id {rec['id']}",
                            file=sys.stderr,
                        )
                    seen_ids.add(rec["id"])
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    record_count += 1
                pico_emitted += 1
                model_pico_emitted += 1
                if args.limit is not None and model_pico_emitted >= args.limit:
                    break

    print(
        f"models={model_list}, include_splits={sorted(include_splits)}, "
        f"scanned PICOs={scanned}, "
        f"emitted PICOs={pico_emitted}, skipped(missing results)={pico_skipped}, "
        f"records={record_count}"
    )
    print(f"output -> {out_path}")


if __name__ == "__main__":
    main()

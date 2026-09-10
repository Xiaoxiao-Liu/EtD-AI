"""LLM-judged rubric scores for the 4 non-rule dimensions.

Reads ``dataset/score_rubric/score/output/all_models.scored.jsonl`` produced by
``score_rubric.py`` and, for every (PICO, criterion) record, asks a judge
LLM to score four dimensions for each of three methods
(``no_evidence`` / ``rag_evidence`` / ``gt_evidence``):

    reasoning_relevance      (0/1/2)
    reasoning_grounding      (0/1/2; None for no_evidence)
    confidence_calibration   (0/1/2)
    etd_consistency          (0/1/2)

Results are saved as one JSON file per (PICO, criterion) under

    dataset/score_rubric/judge/output/<prompt_version>/<judge_model>/<judged_model>/<source_stem>/section_<idx>.json

so that:
- every completed record is persisted immediately (crash-safe)
- resume = skip only a complete result with matching prompt version and input fingerprint
- multiple judge models can coexist without overwriting each other

Each JSON file contains the full updated record: the original 22 fields
from ``*.scored.jsonl`` plus the 13 new judge fields (12 scores +
``judge_model_name``).

Examples::

    # judge demo file with gpt-5, single-threaded
    python -m src.task.score_rubric.judge_rubric \\
        --input dataset/score_rubric/score/output/demo.scored.jsonl \\
        --judge-model gpt-5 --workers 1

    # full run, 8 parallel workers
    python -m src.task.score_rubric.judge_rubric --judge-model gpt-5 --workers 8

    # smoke test, first 5 pending records only
    python -m src.task.score_rubric.judge_rubric --judge-model gpt-5 --limit 5
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.common.judge_client import JudgeClient as ChatBot
from src.task.score_rubric.judge_prompts import (
    PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, context_index, flatten_evidence,
)

ETD_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ETD_ROOT / "dataset"
DEFAULT_INPUT = DATASET_DIR / "score_rubric" / "score" / "output" / "all_models.scored.jsonl"
DEFAULT_RUBRIC_DIR = DATASET_DIR / "score_rubric" / "judge" / "output"

# --------------------------------------------------------------------------- #
# Field naming (kept consistent with score_rubric.py output)
# --------------------------------------------------------------------------- #

# (method, field_prefix, judgement_field, reason_field,
#  judgement_correctness_field, evidence_correctness_field_or_None)
METHODS: tuple[tuple[str, str, str, str, str, Optional[str]], ...] = (
    (
        "no_evidence",
        "no_evidence",
        "no_evidence_judgement",
        "ai_reason_e2e",
        "no_evidence_judgement_correctness",
        None,
    ),
    (
        "rag_evidence",
        "rag",
        "rag_judgement",
        "ai_reason_rag",
        "rag_judgement_correctness",
        "evidence_correctness_rag",
    ),
    (
        "gt_evidence",
        "with_evidence",
        "with_evidence_judgement",
        "ai_reason_with_evidence",
        "with_evidence_judgement_correctness",
        None,
    ),
)

JUDGE_DIMS = (
    "reasoning_relevance",
    "reasoning_grounding",
    "confidence_calibration",
    "etd_consistency",
)


def output_field(prefix: str, dim: str) -> str:
    return f"{prefix}_{dim}"


# --------------------------------------------------------------------------- #
# Prompts live in judge_prompts.py; persist their version with every result.
# --------------------------------------------------------------------------- #


def format_score(val: Any) -> str:
    if val is None:
        return "NA"
    return str(val)


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(raw: str) -> Optional[dict]:
    """Extract the first JSON object from the model response."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def coerce_score(val: Any, allow_na: bool = False) -> Any:
    """Coerce a judge score to int 0/1/2 (or 'NA' if allowed)."""
    if allow_na and isinstance(val, str) and val.strip().upper() == "NA":
        return "NA"
    if isinstance(val, bool):
        return None
    if isinstance(val, int) and val in (0, 1, 2):
        return val
    if isinstance(val, float) and val in (0.0, 1.0, 2.0):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if s in {"0", "1", "2"}:
            return int(s)
    return None


def record_has_all_judge_fields(rec: dict, judge_model: str) -> bool:
    """Resume check: True iff this record already has every judge field
    written under the same judge_model (None values count, since some
    fields are intentionally None - e.g. no_evidence_reasoning_grounding).
    """
    if rec.get("judge_model_name") != judge_model or rec.get("judge_prompt_version") != PROMPT_VERSION:
        return False
    for _method, prefix, *_ in METHODS:
        for dim in JUDGE_DIMS:
            if output_field(prefix, dim) not in rec:
                return False
    return True


# --------------------------------------------------------------------------- #
# Per-record file IO
# --------------------------------------------------------------------------- #


def rubric_path(
    rubric_dir: Path,
    judge_model: str,
    judged_model: str,
    source_stem: str,
    section_index: int,
) -> Path:
    """Resolve the on-disk path for one (PICO, criterion) judge result."""
    return (
        rubric_dir
        / PROMPT_VERSION
        / judge_model
        / judged_model
        / source_stem
        / f"section_{section_index}.json"
    )


def write_single_record(path: Path, record: dict) -> None:
    """Write one record as pretty JSON, atomically (<file>.tmp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def existing_judge_model(path: Path) -> Optional[str]:
    """Read judge_model_name from an existing rubric file, or None on error."""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    val = existing.get("judge_model_name")
    return val if isinstance(val, str) else None


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #


def judge_one_method(
    agent: ChatBot, record: dict, method: str, prefix: str,
    judgement_field: str, reason_field: str, jc_field: str,
    ec_field: Optional[str], context: Optional[list] = None,
    audit: Optional[dict] = None,
) -> dict[str, Any]:
    """Score one candidate; malformed responses fail instead of becoming silent nulls."""
    judgement, reason = record.get(judgement_field), record.get(reason_field)
    if not isinstance(judgement, str) or not judgement.strip() or not isinstance(reason, str) or not reason.strip():
        if audit is not None:
            audit[method] = {"status": "missing_candidate", "calls": []}
        return {output_field(prefix, d): None for d in JUDGE_DIMS}
    system, user = build_prompt(record, method, context)
    before = len(getattr(agent, "calls", []))
    response = agent.call(system, user)
    parsed = parse_judge_response(response)
    expected = set(JUDGE_DIMS) - ({"reasoning_grounding"} if method == "no_evidence" else set())
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ValueError(f"Invalid judge JSON schema for {method}")
    for dim, value in parsed.items():
        if value is None and dim in {"reasoning_grounding", "etd_consistency"}:
            if dim == "etd_consistency" and context:
                raise ValueError("Consistency null despite supplied context")
            continue
        if type(value) is not int or value not in (0, 1, 2):
            raise ValueError(f"Invalid judge score for {method}/{dim}")
    if not context and parsed.get("etd_consistency") is not None:
        raise ValueError("Consistency must be null without other-criterion context")
    if audit is not None:
        audit[method] = {
            "status": "ok", "raw_response": response,
            "prompt_sha256": sha256((system + "\n" + user).encode()).hexdigest(),
            "calls": getattr(agent, "calls", [])[before:],
        }
    return {output_field(prefix, dim): parsed.get(dim) for dim in JUDGE_DIMS}


def input_fingerprint(record: dict, context: Optional[dict]) -> str:
    return sha256(json.dumps([record, context], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def judge_and_save(record: dict, judge_model: str, out_path: Path,
                   context: Optional[dict] = None) -> str:
    agent = ChatBot(judge_model)
    updated = dict(record)
    audit = {}
    for method, prefix, jud_f, rea_f, jc_f, ec_f in METHODS:
        updated.update(judge_one_method(agent, record, method, prefix, jud_f, rea_f,
            jc_f, ec_f, (context or {}).get(method), audit))
    updated.update(judge_model_name=judge_model, judge_prompt_version=PROMPT_VERSION,
                   judge_input_sha256=input_fingerprint(record, context), judge_audit=audit)
    write_single_record(out_path, updated)
    return "ok"


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def judged_model_from_record(record: dict) -> Optional[str]:
    """Return the model that produced judgements for this record."""
    name = record.get("model_name")
    return name if isinstance(name, str) and name.strip() else None


# Fields written by judge_rubric (merged back into *.scored.jsonl).
JUDGE_OUTPUT_FIELDS: tuple[str, ...] = tuple(
    output_field(prefix, dim)
    for _method, prefix, *_ in METHODS
    for dim in JUDGE_DIMS
) + ("judge_model_name", "judge_prompt_version", "judge_input_sha256", "judge_audit")


def run(
    input_path: Path,
    rubric_dir: Path,
    judge_model: str,
    workers: int,
    overwrite: bool,
    limit: Optional[int],
) -> tuple[int, int, int]:
    with input_path.open(encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin if line.strip()]

    contexts = context_index(records)
    pending: list[tuple[dict, Path]] = []
    skipped = 0
    for rec in records:
        src = rec.get("pico_source_file")
        sidx = rec.get("section_index")
        judged_model = judged_model_from_record(rec)
        if not isinstance(src, str) or not isinstance(sidx, int):
            skipped += 1
            continue
        if judged_model is None:
            skipped += 1
            continue
        source_stem = src.rsplit(".", 1)[0]
        out_path = rubric_path(
            rubric_dir, judge_model, judged_model, source_stem, sidx
        )
        if not overwrite and out_path.exists():
            try:
                cached = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError):
                cached = {}
            if record_has_all_judge_fields(cached, judge_model) and cached.get("judge_input_sha256") == input_fingerprint(rec, contexts.get(rec["id"])):
                skipped += 1
                continue
        pending.append((rec, out_path))
        if limit is not None and len(pending) >= limit:
            break

    print(
        f"input={input_path.name}, total={len(records)}, "
        f"skipped(resume/invalid)={skipped}, pending={len(pending)}, "
        f"judge_model={judge_model}, workers={workers}\n"
        f"rubric_dir={rubric_dir} (per-record model_name subdirs)"
    )

    if not pending:
        print("Nothing to do.")
        return len(records), 0, skipped

    disable_progress = not sys.stderr.isatty()
    desc = f"judge ({judge_model})"

    if workers <= 1:
        for completed, (rec, out_path) in enumerate(tqdm(
            pending, desc=desc, unit="rec", disable=disable_progress
        ), 1):
            judge_and_save(rec, judge_model, out_path, contexts.get(rec["id"]))
            if disable_progress and (completed % 25 == 0 or completed == len(pending)):
                print(f"judged {completed}/{len(pending)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(judge_and_save, rec, judge_model, out_path, contexts.get(rec["id"]))
                for rec, out_path in pending
            ]
            for completed, fut in enumerate(tqdm(
                as_completed(futures), total=len(futures), desc=desc,
                unit="rec", disable=disable_progress,
            ), 1):
                # Invalid output raises; it must not be cached as completed null scores.
                fut.result()
                if disable_progress and (completed % 25 == 0 or completed == len(pending)):
                    print(f"judged {completed}/{len(pending)}", flush=True)

    return len(records), len(pending), skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "Input scored JSONL "
            "(default: dataset/score_rubric/score/output/all_models.scored.jsonl)."
        ),
    )
    parser.add_argument(
        "--rubric-dir",
        default=str(DEFAULT_RUBRIC_DIR),
        help=(
            "Root directory for per-record rubric JSON files "
            "(default: dataset/score_rubric/judge/output). Layout: "
            "<rubric_dir>/<prompt_version>/<judge_model>/<record.model_name>/"
            "<source_stem>/section_<idx>.json"
        ),
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="LLM judge model name (e.g. gpt-5, claude-3-5-sonnet, deepseek-v3).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent worker threads for LLM calls (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-judge records that already have a rubric file from this judge.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge the first N pending records (for smoke tests).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    rubric_dir = Path(args.rubric_dir)
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    total, judged, skipped = run(
        input_path=input_path,
        rubric_dir=rubric_dir,
        judge_model=args.judge_model,
        workers=args.workers,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(
        f"done. total={total}, judged_now={judged}, skipped={skipped}\n"
        f"rubric_dir -> {rubric_dir}"
    )


if __name__ == "__main__":
    main()

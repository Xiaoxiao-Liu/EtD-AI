"""Controlled pilot: original vs criterion-anchored prompts using gpt-5.5.

Baseline outputs and RAG passages are reused from the existing gpt-5.5 runs.
Only refined prompts call the API. Outputs are isolated under
dataset/pilot/judgment_definitions/.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.parsing import extract_model_output, parse_judgement_options
from src.task.run_methods.judgement import judgement_compare
from src.task.run_methods.judgment_definitions import format_judgment_definitions

DATA = ROOT / "dataset" / "pico_sections_icd11.json"
SPLITS = ROOT / "dataset" / "train" / "sft" / "pico_splits.json"
OUT = ROOT / "dataset" / "pilot" / "judgment_definitions"
MODEL = "gpt-5.5"
METHODS = ("end2end", "rag", "with_evidence")
SUFFIX = {"end2end": "_e2e.json", "rag": "_rag.json", "with_evidence": "_with_evidence.json"}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def api_call(system: str, user: str) -> str:
    load_env(ROOT.parent / ".env")
    base = os.environ["GPT_4_URL"].rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {os.environ['GPT_4_KEY']}", "Content-Type": "application/json"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read())
            return body["choices"][0]["message"]["content"]
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ConnectionError,
            http.client.RemoteDisconnected,
        ):
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def baseline_path(method: str, stem: str) -> Path:
    return ROOT / "dataset" / "run_methods" / method / "output" / MODEL / f"{stem}{SUFFIX[method]}"


def select_items(n: int, seed: int):
    data = json.loads(DATA.read_text())
    split_map = json.loads(SPLITS.read_text())["split_map"]
    test = [item for item in data if split_map.get(item["source_file"]) == "test"]
    rng = random.Random(seed)
    rng.shuffle(test)
    eligible = [item for item in test if all(baseline_path(m, Path(item["source_file"]).stem).exists() for m in METHODS)]
    return eligible[:n]


def format_options(options) -> str:
    return "\n".join(f"- {option}" for option in options)


def exact_correct(judgment, gold_options) -> bool:
    value = str(judgment).strip().strip('"').strip("'")
    return value in set(gold_options or [])


def prompts(item: dict, section: dict, method: str, evidence: str):
    pico = item["pico"]
    options = parse_judgement_options(section["judgement_extract"])
    system = f"""You are an expert guideline panel member using the GRADE Evidence-to-Decision (EtD) framework.

PICO question:
Population: {pico['Population']}
Intervention: {pico['Intervention']}
Comparison: {pico['Comparison']}
Outcomes: {pico['Main outcomes']}

Select exactly one offered judgment. Use the criterion-specific definitions as decision boundaries. Do not infer the reference answer. Keep the reason to 1-2 sentences.

Output format:
judgement: "<one offered option>"
reason: "<brief justification>"""
    evidence_heading = ""
    if method == "rag":
        evidence_heading = f"\nRetrieved research evidence:\n{evidence or 'No relevant evidence retrieved.'}\n"
    elif method == "with_evidence":
        evidence_heading = f"\nResearch evidence:\n{evidence or 'No research evidence provided.'}\n"
    definitions = format_judgment_definitions(section["criterion"], options)
    user = f"""EtD criterion: {section['criterion']}
EtD subquestion: {section['question']}
{evidence_heading}
Judgment options:
{format_options(options)}

Criterion-specific definitions:
{definitions}

Select the single best option. Use "Varies" only for genuine heterogeneity and "Don't know" only when no substantive option is supportable."""
    return system, user, options


def make_jobs(items):
    jobs = []
    for item in items:
        stem = Path(item["source_file"]).stem
        baselines = {m: json.loads(baseline_path(m, stem).read_text()) for m in METHODS}
        for idx, section in enumerate(item["sections"]):
            for method in METHODS:
                old = baselines[method]["subquestion"][idx]
                evidence = old.get("retrieved_evidence", "") if method == "rag" else section.get("research_evidence", "")
                jobs.append((item, idx, section, method, evidence, old))
    return jobs


def run(n: int, seed: int, workers: int):
    items = select_items(n, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"model": MODEL, "seed": seed, "n_picos": n, "source_files": [x["source_file"] for x in items]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    output_path = OUT / "records.jsonl"
    existing = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            row = json.loads(line); existing[row["id"]] = row
    jobs = make_jobs(items)

    def one(job):
        item, idx, section, method, evidence, old = job
        row_id = f"{Path(item['source_file']).stem}::{idx}::{method}"
        if row_id in existing:
            return existing[row_id]
        system, user, options = prompts(item, section, method, evidence)
        response = api_call(system, user)
        judgment, reason = extract_model_output(response)
        return {
            "id": row_id, "source_file": item["source_file"], "section_index": idx,
            "criterion": section["criterion"], "method": method, "model": MODEL,
            "gt_judgement": [k for k, v in options.items() if v == "●"],
            "baseline_judgement": old.get("ai_judgement"),
            "baseline_reason": old.get("ai_reason"),
            "baseline_correct": exact_correct(
                old.get("ai_judgement"), [k for k, v in options.items() if v == "●"]
            ),
            "refined_judgement": judgment, "refined_reason": reason,
            "refined_correct": exact_correct(
                judgment, [k for k, v in options.items() if v == "●"]
            ),
            "format_error": judgment is None or judgment not in options,
            "response": response,
        }

    rows = list(existing.values())
    pending = [job for job in jobs if f"{Path(job[0]['source_file']).stem}::{job[1]}::{job[3]}" not in existing]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, job) for job in pending]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            output_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
            print(f"completed {i}/{len(pending)}")
    analyze(rows)


def analyze(rows=None):
    if rows is None:
        rows = [json.loads(x) for x in (OUT / "records.jsonl").read_text().splitlines()]
    groups = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    summary = {"overall": summarize(rows), "by_method": {k: summarize(v) for k, v in groups.items()}}
    criteria = {}
    for row in rows:
        criteria.setdefault(row["criterion"], []).append(row)
    summary["by_criterion"] = {k: summarize(v) for k, v in criteria.items()}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize(rows):
    n = len(rows)
    baseline = [exact_correct(x.get("baseline_judgement"), x.get("gt_judgement")) for x in rows]
    refined = [exact_correct(x.get("refined_judgement"), x.get("gt_judgement")) for x in rows]
    return {
        "n": n,
        "baseline_accuracy": sum(baseline) / n if n else None,
        "refined_accuracy": sum(refined) / n if n else None,
        "delta": (sum(r - b for r, b in zip(refined, baseline)) / n) if n else None,
        "format_error_rate": sum(bool(x["format_error"]) for x in rows) / n if n else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-picos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="dataset/pilot/judgment_definitions",
        help="Output directory, relative to the EtD project root unless absolute.",
    )
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    requested_out = Path(args.output_dir)
    OUT = requested_out if requested_out.is_absolute() else ROOT / requested_out
    analyze() if args.analyze_only else run(args.n_picos, args.seed, args.workers)

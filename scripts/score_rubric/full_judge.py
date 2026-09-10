"""Resumable full GPT-5.5 judging with bounded retries and a live status file."""
from __future__ import annotations
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.task.score_rubric import judge_rubric as judge
from src.task.score_rubric.merge_rubric import merge_file, DEFAULT_OUTPUT

RUN = ROOT / 'analysis_reports/full_judge_20260909'
MODEL = 'gpt-5.5'
WORKERS = 16


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = [json.loads(line) for line in judge.DEFAULT_INPUT.read_text().splitlines() if line.strip()]
    contexts = judge.context_index(records)
    jobs = []
    seen = set()
    for rec in records:
        path = judge.rubric_path(judge.DEFAULT_RUBRIC_DIR, MODEL, rec['model_name'],
                                 Path(rec['pico_source_file']).stem, rec['section_index'])
        assert path not in seen, 'Duplicate output path'
        seen.add(path)
        try:
            cached = json.loads(path.read_text())
        except (OSError, ValueError):
            cached = {}
        if (judge.record_has_all_judge_fields(cached, MODEL) and
            cached.get('judge_input_sha256') == judge.input_fingerprint(rec, contexts.get(rec['id']))):
            continue
        jobs.append((rec, path))
    state = dict(pid=os.getpid(), model=MODEL, workers=WORKERS,
                 prompt_version=judge.PROMPT_VERSION, total=len(records),
                 completed=len(records)-len(jobs), resumed=len(records)-len(jobs),
                 input_sha256=hashlib.sha256(judge.DEFAULT_INPUT.read_bytes()).hexdigest(),
                 started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='running')

    def save():
        state['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        judge.write_single_record(RUN / 'status.json', state)

    save()
    print(json.dumps(state), flush=True)
    # All futures are observed; individual errors never conceal other successes.
    for round_number in range(1, 4):
        state['round'] = round_number
        failed = []
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(judge.judge_and_save, rec, MODEL, path,
                                   contexts.get(rec['id'])): (rec, path) for rec, path in jobs}
            for future in cf.as_completed(futures):
                rec, path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((rec, path))
                    with (RUN / 'errors.jsonl').open('a') as log:
                        log.write(json.dumps(dict(record_id=rec['id'], round=round_number,
                                                  error_type=type(exc).__name__)) + '\n')
                else:
                    state['completed'] += 1
                state['failed_this_round'] = len(failed)
                save()
                if state['completed'] % 25 == 0:
                    print(json.dumps(state), flush=True)
        jobs = failed
        if not jobs:
            break
        print(f'Retrying {len(jobs)} failed records after round {round_number}', flush=True)
    if jobs:
        state.update(status='needs_attention', remaining=len(jobs))
        save()
        return 1
    state['status'] = 'merging'
    save()
    temporary = DEFAULT_OUTPUT.with_suffix('.jsonl.tmp')
    stats = merge_file(judge.DEFAULT_INPUT, temporary,
                       rubric_dir=judge.DEFAULT_RUBRIC_DIR, judge_model=MODEL)
    assert stats['missing_rubric'] == 0 and stats['merged'] == len(records), stats
    os.replace(temporary, DEFAULT_OUTPUT)
    state.update(status='completed', merge=stats, merged_output=str(DEFAULT_OUTPUT), remaining=0)
    save()
    print(json.dumps(state), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

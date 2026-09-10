"""Train-only, PICO-disjoint judge prompt pilot. No executor generation.

Example: python scripts/score_rubric/pilot_judge_prompt.py prepare
         python scripts/score_rubric/pilot_judge_prompt.py run --phase development --version v2
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import itertools
import importlib.util
from functools import lru_cache
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.judge_client import JudgeClient
from src.task.score_rubric import judge_prompts as prompts

OUT = ROOT / 'analysis_reports/judge_prompt_20260909'
MODELS = ('gpt-4o', 'gpt-5.5', 'deepseek-v4-pro')
EXECUTORS = ('gpt-4o', 'gpt-5.5', 'claude-opus-4-6-thinking')
INPUT = ROOT / 'dataset/score_rubric/score/output/all_models.scored.jsonl'
METHODS = tuple(prompts.METHOD_KEYS)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def prepare():
    if (OUT / 'manifest.json').exists():
        raise SystemExit('Pilot already frozen; reuse manifest, do not resample.')
    rows = [json.loads(s) for s in INPUT.read_text().splitlines()]
    splits = json.loads((ROOT / 'dataset/train/sft/pico_splits.json').read_text())['split_map']
    train = [r for r in rows if splits[r['pico_source_file']] == 'train']
    rng = random.Random(20260909)
    rng.shuffle(train)
    criteria = sorted({r['criterion'] for r in train})
    used_picos = set()
    batches = {}
    for phase in ('development', 'validation'):
        chosen = []
        for ci, criterion in enumerate(criteria):
            for j in range(5):
                model = EXECUTORS[(ci * 5 + j) % 3]
                pool = [r for r in train if r['criterion'] == criterion
                        and r['model_name'] == model and r['pico_source_file'] not in used_picos]
                assert pool, (phase, criterion, model)
                # Include missing-reference edge cases, then varied fixed-baseline outcomes.
                preferred = [r for r in pool if not r.get('gt_judgement_label')] if j == 0 else []
                if not preferred:
                    target = (j % 2, (j // 2) % 2)
                    preferred = [r for r in pool if (r.get('no_evidence_judgement_correctness'),
                                r.get('rag_judgement_correctness')) == target]
                row = (preferred or pool)[0]
                chosen.append(row)
                used_picos.add(row['pico_source_file'])
        batches[phase] = chosen
        dump(OUT / f'{phase}.json', chosen)
    ctx = prompts.context_index(train)
    dump(OUT / 'contexts.json', {r['id']: ctx[r['id']] for b in batches.values() for r in b})
    dump(OUT / 'manifest.json', {
        'seed': 20260909, 'source_sha256': sha256(INPUT.read_bytes()).hexdigest(),
        'judges': MODELS, 'phases': {k: [r['id'] for r in v] for k,v in batches.items()},
        'counts': {k: {'records': len(v), 'picos': len({r['pico_source_file'] for r in v}),
            'executor': dict(Counter(r['model_name'] for r in v)),
            'criteria': dict(Counter(r['criterion'] for r in v)),
            'missing_reference_judgment': sum(not r.get('gt_judgement_label') for r in v)}
            for k,v in batches.items()},
        'split_policy': 'Both phases use train PICOs only; phases share no PICO. Official val/test untouched.',
        'primary_checks': ['valid JSON/schema', 'missing context handling', 'dimension independence',
                           'pairwise ordinal agreement', 'downstream action agreement'],
        'limitations': 'No independent clinician gold ratings. Agreement is not accuracy. No test conclusions.',
    })
    print('Frozen 60 development + 60 validation records; disjoint train PICOs.', flush=True)


def legacy_prompt(record, method):
    module = ast.parse((OUT / 'legacy_judge_rubric.py').read_text())
    constants = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('SYSTEM_PROMPT', 'USER_PROMPT_WITH_GROUNDING', 'USER_PROMPT_NO_EVIDENCE'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    jud, rea, correct = prompts.METHOD_KEYS[method]
    kwargs = dict(pico=record.get('PICO',''), criterion=record['criterion'],
        criterion_question=record['criterion_question'], ai_judgement=record[jud], ai_reason=record[rea],
        gt_judgement=record.get('gt_judgement_label',''), gt_reasoning=record.get('gt_reasoning') or '(none)',
        judgement_correctness='NA' if record.get(correct) is None else str(record[correct]))
    template = constants['USER_PROMPT_NO_EVIDENCE']
    if method != 'no_evidence':
        value = record.get('rag_evidence' if method == 'rag_evidence' else 'gt_evidence')
        if isinstance(value, dict):
            evidence = value.get('text','')
            if value.get('tables'): evidence += f"\n\n[tables omitted: {len(value['tables'])} table(s)]"
            evidence = evidence or '无'
        else: evidence = prompts.flatten_evidence(value) or '无'
        kwargs.update(method=method, available_evidence=evidence,
            evidence_correctness_rag=str(record.get('evidence_correctness_rag','NA')) if method == 'rag_evidence' else 'NA',
            evidence_correctness_rag_label=record.get('evidence_correctness_rag_label') or 'NA')
        template = constants['USER_PROMPT_WITH_GROUNDING']
    return constants['SYSTEM_PROMPT'], template.format(**kwargs)


def parse(raw, method, version):
    cleaned = raw.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[1].rsplit('```',1)[0].strip()
    scores = json.loads(cleaned)
    expected = {'reasoning_relevance', 'confidence_calibration', 'etd_consistency'}
    if method != 'no_evidence': expected.add('reasoning_grounding')
    if not isinstance(scores, dict) or set(scores) != expected:
        raise ValueError('Schema keys mismatch')
    for key, value in scores.items():
        if value is None and version != 'legacy' and key in {'reasoning_grounding','etd_consistency'}:continue
        if type(value) is not int or value not in (0,1,2):
            raise ValueError(f'Invalid score: {key}')
    return scores


@lru_cache(maxsize=None)
def prompt_module(version):
    snapshot = OUT / f'prompts_{version}.py'
    if not snapshot.exists():
        return prompts
    spec = importlib.util.spec_from_file_location(f'pilot_{version}', snapshot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(row, method, model, version, phase, context):
    token = sha256((row['id']+'|'+method).encode()).hexdigest()[:20]
    path = OUT / 'responses' / version / phase / model / f'{token}.json'
    system, user = legacy_prompt(row,method) if version == 'legacy' else prompt_module(version).build_prompt(row,method,context)
    digest = sha256((system+'\n'+user).encode()).hexdigest()
    prior_calls = []
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('status') == 'ok':
            assert old['prompt_sha256'] == digest, 'Prompt changed under frozen version; choose a new version'
            return 'cached'
        prior_calls = old.get('calls', [])
    client = JudgeClient(model)
    result = dict(record_id=row['id'], pico=row['pico_source_file'], executor=row['model_name'],
        criterion=row['criterion'], method=method, judge=model, version=version, phase=phase,
        prompt_sha256=digest, system=system, user=user)
    try:
        raw = client.call(system,user)
        result['raw'] = raw
        result['scores'] = parse(raw,method,version)
        if version not in ('legacy', 'v2', 'v2_1') and context and result['scores'].get('etd_consistency') is None:
            raise ValueError('Consistency null despite supplied context')
        result['status'] = 'ok'
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__)
    result['calls'] = prior_calls + client.calls
    dump(path,result)
    return result['status']


def run(args):
    rows = json.loads((OUT / f'{args.phase}.json').read_text())
    if args.limit: rows = rows[:args.limit]
    contexts = json.loads((OUT / 'contexts.json').read_text())
    jobs = [(r,m,j) for r in rows for m in METHODS for j in MODELS]
    # Interleave judges; bounded concurrency and bounded retries in client.
    stats=Counter(); start=time.monotonic()
    print(f'{args.version}/{args.phase}: {len(rows)} records, {len(jobs)} planned scores',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,r,m,j,args.version,args.phase,contexts[r['id']][m]) for r,m,j in jobs]
        for i,future in enumerate(as_completed(futures),1):
            stats[future.result()]+=1
            if i % 9 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} {dict(stats)} elapsed={time.monotonic()-start:.0f}s',flush=True)
    dump(OUT / f'run_{args.version}_{args.phase}.json',dict(stats))


def summarize():
    grouped=defaultdict(list)
    for p in (OUT/'responses').rglob('*.json'):
        r=json.loads(p.read_text()); grouped[(r['version'],r['phase'])].append(r)
    summary={}
    for (version,phase),rows in grouped.items():
        key=version+'/'+phase
        out={'scores':len(rows),'status':dict(Counter(r['status'] for r in rows)),
             'calls':sum(len(r['calls']) for r in rows), 'judges':{},'agreement':{}}
        for model in MODELS:
            subset=[r for r in rows if r['judge']==model]
            good=[r for r in subset if r['status']=='ok']
            usage=Counter()
            for r in subset:
                for c in r['calls']:
                    for k in ('prompt_tokens','completion_tokens','total_tokens'):
                        usage[k]+=(c.get('usage') or {}).get(k,0)
            out['judges'][model]={'n':len(subset),'ok':len(good),'usage':dict(usage),
                'returned_models':dict(Counter(c.get('returned_model') for r in subset for c in r['calls'] if c.get('status')=='ok')),
                'dimensions':{d:dict(Counter(str(r['scores'].get(d)) for r in good if d in r['scores']))
                  for d in ('reasoning_relevance','reasoning_grounding','confidence_calibration','etd_consistency')}}
        idx={(r['record_id'],r['method'],r['judge']):r for r in rows if r['status']=='ok'}
        for a,b in itertools.combinations(MODELS,2):
            result={}
            for dim in ('reasoning_relevance','reasoning_grounding','confidence_calibration','etd_consistency'):
                pairs=[]
                for (rid,m,j),r in idx.items():
                    if j!=a or (rid,m,b) not in idx:continue
                    x,y=r['scores'].get(dim),idx[(rid,m,b)]['scores'].get(dim)
                    if type(x) is int and type(y) is int:pairs.append((x,y))
                if pairs:
                    n=len(pairs); mae=sum(abs(x-y) for x,y in pairs)/n
                    cx,cy=Counter(x for x,y in pairs),Counter(y for x,y in pairs)
                    expected=sum(cx[x]*cy[y]*abs(x-y) for x in cx for y in cy)/n**2
                    result[dim]={'n':n,'exact':sum(x==y for x,y in pairs)/n,'mae':mae,
                                 'linear_weighted_kappa':1-mae/expected if expected else None}
            out['agreement'][a+' vs '+b]=result
        summary[key]=out
    dump(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('prepare','run','summarize'))
    ap.add_argument('--phase',choices=('development','refinement','validation'),default='development')
    ap.add_argument('--version',default='v2')
    ap.add_argument('--limit',type=int)
    ap.add_argument('--workers',type=int,default=12)
    args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args)
    else:summarize()

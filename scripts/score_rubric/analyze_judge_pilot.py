"""Read-only analysis plus report artifacts for the train-only judge pilot."""
from __future__ import annotations
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.task.score_rubric.label_policy import label_record
from src.task.score_rubric.judge_prompts import context_index
OUT=ROOT/'analysis_reports/judge_prompt_20260909'
MODELS=('gpt-4o','gpt-5.5','deepseek-v4-pro')
PREFIX={'no_evidence':'no_evidence','rag_evidence':'rag','gt_evidence':'with_evidence'}
TARGETS=('initial_action_label','evidence_sufficiency_label','no_evidence_arbit_label','rag_arbit_label')


def main():
    source={r['id']:r for phase in ('development','validation')
            for r in json.loads((OUT/f'{phase}.json').read_text())}
    groups=defaultdict(list)
    for path in (OUT/'responses').rglob('*.json'):
        row=json.loads(path.read_text())
        groups[(row['version'],row['phase'])].append(row)
    summary={}
    policy_rows=[]
    for (version,phase),rows in groups.items():
        valid=[r for r in rows if r['status']=='ok']
        by_record=defaultdict(list)
        for r in valid:by_record[(r['record_id'],r['judge'])].append(r)
        outcomes={}
        for (rid,judge),entries in by_record.items():
            if len(entries)!=3:continue
            rec=dict(source[rid])
            for entry in entries:
                prefix=PREFIX[entry['method']]
                rec.update({prefix+'_'+k:v for k,v in entry['scores'].items()})
            labels=label_record(rec,0.3,0.5)
            outcomes[(rid,judge)]=labels
            policy_rows.append(dict(version=version,phase=phase,record_id=rid,judge=judge,
                                    executor=rec['model_name'],**labels))
        checks={}
        for judge in MODELS:
            subset=[r for r in valid if r['judge']==judge]
            wrong_grounded=[r for r in subset if r['method']!='no_evidence'
                and source[r['record_id']].get(PREFIX[r['method']]+'_judgement_correctness')==0]
            no_context_null=[r for r in subset if r['scores'].get('etd_consistency') is None]
            checks[judge]={
                'complete_record_triplets':sum(j==judge for _,j in outcomes),
                'missing_correctness_triplets':sum(v.get('label_status')=='missing_correctness' for (rid,j),v in outcomes.items() if j==judge),
                'wrong_label_grounding_n':len(wrong_grounded),
                'wrong_label_grounding_positive':sum((r['scores'].get('reasoning_grounding') or 0)>0 for r in wrong_grounded),
                'unexpected_consistency_null':len(no_context_null) if version!='legacy' else None,
                'action_distributions':{k:dict(Counter(str(v.get(k)) for (rid,j),v in outcomes.items() if j==judge)) for k in TARGETS},
            }
        agreement={}
        for a,b in itertools.combinations(MODELS,2):
            shared={rid for rid,j in outcomes if j==a} & {rid for rid,j in outcomes if j==b}
            stats={}
            for k in TARGETS:
                pairs=[(outcomes[(rid,a)].get(k),outcomes[(rid,b)].get(k)) for rid in shared]
                pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
                stats[k]={'n':len(pairs),'agreement':sum(x==y for x,y in pairs)/len(pairs) if pairs else None}
            agreement[a+' vs '+b]=stats
        summary[version+'/'+phase]={'checks':checks,'action_agreement':agreement}
    (OUT/'policy_agreement.json').write_text(json.dumps(summary,indent=2))
    (OUT/'pilot_policy_labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in policy_rows))
    print('Wrote policy_agreement.json and pilot_policy_labels.jsonl (pilot artifacts only).')

if __name__=='__main__':main()

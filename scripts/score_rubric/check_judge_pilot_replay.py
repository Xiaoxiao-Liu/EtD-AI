"""Validate production parsing/save/merge using cached pilot responses; zero API calls."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.task.score_rubric.judge_rubric import judge_and_save, rubric_path
from src.task.score_rubric.merge_rubric import merge_record
from src.task.score_rubric.judge_prompts import PROMPT_VERSION
OUT=ROOT/'analysis_reports/judge_prompt_20260909'


def main():
    records=json.loads((OUT/'validation.json').read_text())
    contexts=json.loads((OUT/'contexts.json').read_text())
    responses={}
    for path in (OUT/'responses/v2_2/validation').rglob('*.json'):
        r=json.loads(path.read_text())
        assert r['status']=='ok',path
        responses[(r['record_id'],r['judge'],r['method'])]=r
    assert len(responses)==540,len(responses)
    count=0
    with tempfile.TemporaryDirectory() as tmp:
        for rec in records:
            for judge in ('gpt-4o','gpt-5.5','deepseek-v4-pro'):
                class ReplayClient:
                    def __init__(self,*args):self.calls=[]
                    def call(self,system,user):
                        data=json.loads(user.split('INPUT DATA (JSON):\n')[1].split('\nOUTPUT VALUE CONSTRAINTS')[0])
                        r=responses[(rec['id'],judge,data['condition'])]
                        assert hashlib.sha256((system+'\n'+user).encode()).hexdigest()==r['prompt_sha256']
                        self.calls.extend(r['calls'])
                        return r['raw']
                path=rubric_path(Path(tmp),judge,rec['model_name'],Path(rec['pico_source_file']).stem,rec['section_index'])
                with patch('src.task.score_rubric.judge_rubric.ChatBot',ReplayClient):
                    judge_and_save(rec,judge,path,contexts[rec['id']])
                merged,found=merge_record(rec,rubric_dir=Path(tmp),judge_model=judge)
                assert found and merged['judge_prompt_version']==PROMPT_VERSION
                assert set(merged['judge_audit'])=={'no_evidence','rag_evidence','gt_evidence'}
                count+=1
    result={'record_triplets':count,'cached_method_responses':len(responses),
            'new_api_calls':0,'status':'passed','prompt_version':PROMPT_VERSION}
    (OUT/'replay_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()

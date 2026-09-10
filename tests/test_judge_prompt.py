from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.task.score_rubric.judge_prompts import build_prompt, context_index, flatten_evidence, evidence_status, PROMPT_VERSION
from src.task.score_rubric.judge_rubric import (
    METHODS, JUDGE_DIMS, judge_one_method, judge_and_save, rubric_path,
    record_has_all_judge_fields, input_fingerprint,
)
from src.task.score_rubric.merge_rubric import merge_record


def row(id='a', source='p.json', executor='gpt-4o', idx=0):
    return dict(id=id, pico_source_file=source, model_name=executor, section_index=idx,
        PICO='Context', criterion='Problem', criterion_question='Is this a priority?',
        no_evidence_judgement='Yes', ai_reason_e2e='Important burden.',
        rag_judgement='Yes', ai_reason_rag='Important burden.',
        with_evidence_judgement='Yes', ai_reason_with_evidence='Important burden.',
        rag_evidence='A major burden.', gt_evidence={'text':'A major burden.', 'tables':[['N',100]]},
        no_evidence_judgement_correctness=0, rag_judgement_correctness=0,
        with_evidence_judgement_correctness=0, gt_judgement_label=None, gt_reasoning=None)

CONTEXT = [{'criterion':'Feasibility','judgment':'Yes'}]


class FakeClient:
    def __init__(self, response): self.response=response; self.calls=[]
    def call(self, system, user):
        self.calls.append({'status':'ok'})
        return self.response


class JudgePromptTest(unittest.TestCase):
    def test_context_never_crosses_executor_or_pico_and_excludes_self(self):
        rows=[row(),row('b',idx=1),row('c',executor='gpt-5.5',idx=2),row('d',source='other',idx=3)]
        context=context_index(rows)['a']['rag_evidence']
        self.assertEqual(len(context),1)
        self.assertNotIn('rationale',context[0])

    def test_closed_book_does_not_receive_rag_or_gold_evidence(self):
        rec=row();rec['rag_evidence']='SECRET_RAG';rec['gt_evidence']='SECRET_GOLD'
        system,user=build_prompt(rec,'no_evidence',CONTEXT)
        self.assertNotIn('SECRET_RAG',user);self.assertNotIn('SECRET_GOLD',user)
        self.assertNotIn('4. reasoning_grounding:',user)

    def test_missing_reference_is_explicit_and_tables_preserved(self):
        _,user=build_prompt(row(),'gt_evidence',CONTEXT)
        data=json.loads(user.split('INPUT DATA (JSON):\n')[1].split('\nOUTPUT VALUE CONSTRAINTS')[0])
        self.assertIsNone(data['reference']['judgment'])
        self.assertIn('100',data['available_evidence'])
        self.assertNotIn('tables omitted',flatten_evidence(row()['gt_evidence']))

    def test_wrong_label_can_have_grounded_reasoning(self):
        agent=FakeClient(json.dumps(dict(reasoning_relevance=2,reasoning_grounding=2,
                         confidence_calibration=2,etd_consistency=2)))
        scored=judge_one_method(agent,row(),*METHODS[1],context=CONTEXT)
        self.assertEqual(scored['rag_reasoning_grounding'],2)

    def test_bad_json_and_illegal_scores_fail(self):
        for raw in ('[]','{}','not JSON',json.dumps(dict(reasoning_relevance=True,
                confidence_calibration=2,etd_consistency=2))):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                judge_one_method(FakeClient(raw),row(),*METHODS[0],context=CONTEXT)

    def test_missing_candidate_does_not_call_api(self):
        rec=row();rec['ai_reason_e2e']=' '
        client=FakeClient('{}');audit={}
        scores=judge_one_method(client,rec,*METHODS[0],context=CONTEXT,audit=audit)
        self.assertFalse(client.calls)
        self.assertTrue(all(v is None for v in scores.values()))
        self.assertEqual(audit['no_evidence']['status'],'missing_candidate')

    def test_no_context_requires_explicit_null_consistency(self):
        client=FakeClient(json.dumps(dict(reasoning_relevance=2,confidence_calibration=2,etd_consistency=None)))
        scores=judge_one_method(client,row(),*METHODS[0],context=[])
        self.assertIsNone(scores['no_evidence_etd_consistency'])
        client.response=client.response.replace('null','2')
        with self.assertRaises(ValueError):judge_one_method(client,row(),*METHODS[0],context=[])

    def test_input_fingerprint_changes_with_context(self):
        self.assertNotEqual(input_fingerprint(row(),{}),input_fingerprint(row(),{'rag_evidence':CONTEXT}))

    def test_empty_evidence_detection(self):
        self.assertEqual(evidence_status('react-empty: 612'), 'empty')
        self.assertEqual(evidence_status('No research evidence identified'), 'no_studies_reported')
        self.assertEqual(evidence_status('There is no research evidence react-empty: 123'), 'no_studies_reported')
        self.assertEqual(evidence_status('See corresponding narrative for companion PICO.'), 'cross_reference_only')
        self.assertEqual(evidence_status('A trial of 100 patients found no benefit.'), 'available')

    def test_missing_correctness_is_not_negative_policy_supervision(self):
        from src.task.score_rubric.label_policy import label_record
        from src.task.train.sft.prepare_data import build_routing_record, build_sufficiency_record, build_judgment_records
        rec=row()
        for key in ('no_evidence_judgement_correctness','rag_judgement_correctness','with_evidence_judgement_correctness'):
            rec[key]=None
        rec.update(label_record(rec,0.3,0.5))
        self.assertEqual(rec['label_status'],'missing_correctness')
        self.assertIsNone(build_routing_record(rec))
        self.assertIsNone(build_sufficiency_record(rec))
        self.assertEqual(build_judgment_records(rec),[])

    def test_missing_correctness_file_reports_exclusion(self):
        from src.task.score_rubric.label_policy import label_file
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'input.jsonl';output=Path(tmp)/'output.jsonl'
            source.write_text(json.dumps({'id':'missing-reference'})+'\n')
            stats=label_file(source,output,0.3,0.5)
            self.assertEqual(stats['missing_correctness'],1)
            self.assertEqual(stats['retrieve_evidence'],0)
            self.assertIsNone(json.loads(output.read_text())['initial_action_label'])

    def test_versioned_save_and_merge(self):
        class AutoClient:
            def __init__(self,*args):self.calls=[]
            def call(self,system,user):
                dims=['reasoning_relevance','confidence_calibration','etd_consistency']
                if '4. reasoning_grounding:' in user:dims.append('reasoning_grounding')
                return json.dumps(dict.fromkeys(dims,2))
        rec=row();contexts={m[0]:CONTEXT for m in METHODS}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=rubric_path(root,'fake',rec['model_name'],'p',0)
            self.assertIn(PROMPT_VERSION,str(path))
            with patch('src.task.score_rubric.judge_rubric.ChatBot',AutoClient):
                judge_and_save(rec,'fake',path,contexts)
            saved=json.loads(path.read_text())
            self.assertTrue(record_has_all_judge_fields(saved,'fake'))
            merged,found=merge_record(rec,rubric_dir=root,judge_model='fake')
            self.assertTrue(found)
            self.assertEqual(merged['judge_prompt_version'],PROMPT_VERSION)
            saved['judge_prompt_version']='old'
            path.write_text(json.dumps(saved))
            _,found=merge_record(rec,rubric_dir=root,judge_model='fake')
            self.assertFalse(found)

if __name__=='__main__':unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.task.train.dpo.judgment.data_preparation.prepare import build_pairs
from src.task.train.dpo.routing.data_preparation.prepare import build_pair
from src.task.train.paths import LABELED_JSONL
from src.task.score_rubric.label_policy import label_record


FORBIDDEN_PROMPT_KEYS = (
    "gt_evidence:",
    "gt_score:",
    "reference_judgment:",
    "judgement_correctness:",
    "evidence_correctness_rag:",
)


def records():
    with LABELED_JSONL.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class PreferenceDataTest(unittest.TestCase):
    def test_router_prefers_direct_when_both_paths_are_correct(self):
        record = {
            "id": "router-both-correct",
            "PICO": "pico",
            "criterion": "criterion",
            "criterion_question": "question",
            "initial_action_label": "direct_answer",
            "initial_action_preference_reason": "both_correct_prefer_direct",
            "no_score": 0.70,
            "rag_score": 0.95,
            "no_evidence_direct_gate": True,
            "rag_hard_gate": True,
        }
        pair, reason = build_pair(
            record, min_weight=0.05, weight_mode="outcome",
            both_correct_weight=0.3,
        )
        self.assertEqual(reason, "")
        self.assertIsNotNone(pair)
        self.assertEqual(pair["chosen"], "direct_answer")
        self.assertEqual(pair["weight"], 0.3)
        self.assertEqual({pair["chosen"], pair["rejected"]}, {"direct_answer", "retrieve_evidence"})
        for key in FORBIDDEN_PROMPT_KEYS:
            self.assertNotIn(key, pair["prompt"].lower())

    def test_verifier_pairs_use_accept_reject(self):
        record = {
            "id": "verifier",
            "PICO": "pico",
            "criterion": "criterion",
            "criterion_question": "question",
            "low_quality_case": False,
            "no_evidence_arbit_label": "accept",
            "rag_arbit_label": "reject",
            "no_evidence_judgement": "Yes",
            "ai_reason_e2e": "Relevant rationale",
            "rag_judgement": "No",
            "ai_reason_rag": "Unsupported rationale",
            "rag_evidence": "Evidence",
            "evidence_sufficiency_label": "retrieve_more",
            "no_score": 0.8,
            "rag_score": 0.2,
        }
        pairs = [
            pair for pair, _ in build_pairs(
                record, min_weight=0.05, weight_mode="utility_confidence"
            ) if pair
        ]
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            self.assertEqual({pair["chosen"], pair["rejected"]}, {"accept", "reject"})
            self.assertGreaterEqual(pair["weight"], 0.05)
            self.assertLessEqual(pair["weight"], 1.0)
            for key in FORBIDDEN_PROMPT_KEYS:
                self.assertNotIn(key, pair["prompt"].lower())

    def test_policy_labels_match_locked_action_preferences(self):
        base = {
            "no_evidence_judgement_correctness": 1,
            "no_evidence_reasoning_relevance": 2,
            "no_evidence_confidence_calibration": 2,
            "no_evidence_etd_consistency": 2,
            "rag_judgement_correctness": 1,
            "evidence_correctness_rag": 2,
            "rag_reasoning_grounding": 2,
            "rag_reasoning_relevance": 2,
            "rag_confidence_calibration": 2,
            "rag_etd_consistency": 2,
            "with_evidence_judgement_correctness": 1,
            "with_evidence_reasoning_grounding": 2,
            "with_evidence_reasoning_relevance": 2,
            "with_evidence_confidence_calibration": 2,
            "with_evidence_etd_consistency": 2,
        }
        labels = label_record(
            base, evidence_sufficiency_threshold=0.3, oracle_gap_threshold=0.5,
        )
        self.assertEqual(labels["initial_action_label"], "direct_answer")
        self.assertEqual(labels["initial_action_preference_reason"], "both_correct_prefer_direct")
        self.assertEqual(labels["evidence_sufficiency_label"], "sufficient")
        self.assertEqual(labels["no_evidence_arbit_label"], "accept")
        self.assertEqual(labels["rag_arbit_label"], "accept")

        unsupported_rag = {**base, "rag_reasoning_grounding": 0}
        labels = label_record(
            unsupported_rag,
            evidence_sufficiency_threshold=0.3,
            oracle_gap_threshold=0.5,
        )
        self.assertEqual(labels["evidence_sufficiency_label"], "retrieve_more")
        self.assertEqual(labels["rag_arbit_label"], "reject")

    @unittest.skipUnless(LABELED_JSONL.exists(), "labelled dataset is not present")
    def test_only_development_preference_splits_are_written(self):
        for role in ("routing", "sufficiency", "judgment"):
            root = Path("dataset/train/dpo") / role / "output"
            split_picos = {}
            for split in ("train", "val"):
                with (root / f"{split}.jsonl").open(encoding="utf-8") as stream:
                    split_picos[split] = {
                        json.loads(line)["meta"]["pico_source_file"]
                        for line in stream if line.strip()
                    }
            self.assertFalse(split_picos["train"] & split_picos["val"])
            self.assertFalse((root / "test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

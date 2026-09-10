"""Policy client: wraps a ChatBot so it can play the three SFT sub-routines.

The SFT prompts (src/task/train/sft/prompts.py) instruct the model to emit a
single bare label per call:

  [ROUTE]   -> direct_answer  | retrieve_evidence
  [SUFFICE] -> sufficient     | retrieve_more
  [JUDGE]   -> accept         | reject

A trained Qwen3-8B policy would obey this; a stock chat model (gpt-4o) often
adds whitespace, quotes, or a brief explanation. Parsing here is permissive:
case-insensitive substring match against the allowed labels, first match wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from src.common.agent import ChatBot
from src.task.train.sft.prompts import SYSTEM_PROMPT, render_user


ROUTE_LABELS = ("direct_answer", "retrieve_evidence")
SUFFICE_LABELS = ("sufficient", "retrieve_more")
JUDGE_LABELS = ("accept", "reject")


def _parse_label(response: Optional[str], allowed: tuple[str, ...]) -> Optional[str]:
    if not response:
        return None
    text = response.strip().lower()
    text = re.sub(r"[`*\"'\s]+", "_", text)
    # Direct equality check first (cheap, matches a well-behaved policy).
    for label in allowed:
        if text == label:
            return label
    # Permissive: substring search; whichever allowed label appears earliest wins.
    earliest_pos = len(text) + 1
    earliest_label: Optional[str] = None
    for label in allowed:
        pos = text.find(label)
        if pos >= 0 and pos < earliest_pos:
            earliest_pos = pos
            earliest_label = label
    return earliest_label


@dataclass
class PolicyDecision:
    label: Optional[str]
    raw: Optional[str]


class PolicyClient:
    """LLM-backed policy: routes / decides sufficiency / judges candidates."""

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model_name = model_name
        self.agent = ChatBot(model_name)

    def _call(self, task: str, input_obj: dict, allowed: tuple[str, ...]) -> PolicyDecision:
        user = render_user(task, input_obj)
        raw = self.agent.call(SYSTEM_PROMPT, user)
        label = _parse_label(raw, allowed)
        return PolicyDecision(label=label, raw=raw)

    def route(self, pico: Any, criterion: str, criterion_question: str) -> PolicyDecision:
        return self._call(
            "routing",
            {
                "PICO": pico,
                "criterion": criterion,
                "criterion_question": criterion_question,
            },
            ROUTE_LABELS,
        )

    def suffice(
        self,
        pico: Any,
        criterion: str,
        criterion_question: str,
        rag_evidence: str,
    ) -> PolicyDecision:
        return self._call(
            "sufficiency",
            {
                "PICO": pico,
                "criterion": criterion,
                "criterion_question": criterion_question,
                "rag_evidence": rag_evidence,
            },
            SUFFICE_LABELS,
        )

    def judge(
        self,
        pico: Any,
        criterion: str,
        criterion_question: str,
        candidate: dict,
        source: str,
        rag_evidence: Optional[str] = None,
        sufficiency_assessment: Optional[str] = None,
    ) -> PolicyDecision:
        input_obj: dict = {
            "PICO": pico,
            "criterion": criterion,
            "criterion_question": criterion_question,
            "source": source,
            "candidate": candidate,
        }
        if source == "rag_evidence" and rag_evidence:
            input_obj["rag_evidence"] = rag_evidence
            input_obj["sufficiency_assessment"] = sufficiency_assessment or ""
        return self._call("judgment", input_obj, JUDGE_LABELS)

"""Prompt templates for the EtD policy model (multi-task SFT + DPO).

The policy is one model that handles three sub-routines, distinguished by an
instruction tag in the user message:

    [ROUTE]    routing       -> direct_answer  | retrieve_evidence
    [SUFFICE]  sufficiency   -> sufficient     | retrieve_more
    [JUDGE]    judgment      -> accept         | reject

These templates are imported by:
- ``src/task/train/sft/data.py``           (SFT dataset rendering)
- ``src/task/train/dpo/sufficiency/data_preparation/prepare.py``
                                           (DPO prompts must match SFT-2 exactly)

Style follows ``src/task/run_methods/prompts.py`` (string.Template).
"""

from __future__ import annotations

from string import Template
from typing import Any

# --------------------------------------------------------------------------- #
# System prompt (shared across all three sub-routines)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are the EtD policy model. You operate as one of three sub-routines, "
    "selected by the leading instruction tag in the user message:\n"
    "\n"
    "  [ROUTE]   Decide if the model can answer directly or must retrieve evidence.\n"
    "            Output exactly one of:  direct_answer  |  retrieve_evidence\n"
    "  [SUFFICE] Given retrieved RAG evidence, decide if it is sufficient.\n"
    "            Output exactly one of:  sufficient  |  retrieve_more\n"
    "  [JUDGE]   Given a single candidate judgement and its source, assess quality.\n"
    "            Output exactly one of:  accept  |  reject\n"
    "\n"
    "You MUST output only the label string. No explanation, no punctuation, "
    "no quotation marks, no extra whitespace."
)

# --------------------------------------------------------------------------- #
# User prompt templates
# --------------------------------------------------------------------------- #

ROUTING_USER_TEMPLATE = Template(
    """[ROUTE]
PICO: $PICO
Criterion: $CRITERION
Question: $CRITERION_QUESTION

Decide:"""
)

SUFFICIENCY_USER_TEMPLATE = Template(
    """[SUFFICE]
PICO: $PICO
Criterion: $CRITERION
Question: $CRITERION_QUESTION

Retrieved evidence:
$RAG_EVIDENCE

Decide:"""
)

JUDGMENT_USER_TEMPLATE = Template(
    """[JUDGE]
PICO: $PICO
Criterion: $CRITERION
Question: $CRITERION_QUESTION

Source: $SOURCE
Candidate judgement: $CANDIDATE_JUDGEMENT
Candidate rationale: $CANDIDATE_RATIONALE
$EVIDENCE_BLOCK
Assess quality:"""
)


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #


def _pico_to_text(pico: Any) -> str:
    """Stringify the PICO field (it may be a dict or a plain string)."""
    if isinstance(pico, str):
        return pico
    if isinstance(pico, dict):
        # Preserve key order; format as ``key: value`` lines for readability.
        return "\n".join(f"{k}: {v}" for k, v in pico.items())
    return str(pico)


def render_user(task: str, input_obj: dict) -> str:
    """Render the user-turn content for one record.

    ``input_obj`` is the ``record["input"]`` dict produced by
    ``src.task.train.sft.prepare_data``. The field whitelist is already
    enforced upstream; this function only formats.
    """
    pico_text = _pico_to_text(input_obj.get("PICO", ""))
    criterion = input_obj.get("criterion", "")
    question = input_obj.get("criterion_question", "")

    if task == "routing":
        return ROUTING_USER_TEMPLATE.substitute(
            PICO=pico_text,
            CRITERION=criterion,
            CRITERION_QUESTION=question,
        )
    if task == "sufficiency":
        return SUFFICIENCY_USER_TEMPLATE.substitute(
            PICO=pico_text,
            CRITERION=criterion,
            CRITERION_QUESTION=question,
            RAG_EVIDENCE=input_obj.get("rag_evidence", ""),
        )
    if task == "judgment":
        source = input_obj.get("source", "")
        candidate = input_obj.get("candidate", {}) or {}
        evidence_block = ""
        if source == "rag_evidence":
            rag_ev = input_obj.get("rag_evidence", "")
            suff = input_obj.get("sufficiency_assessment", "")
            evidence_block = (
                f"\nRetrieved evidence:\n{rag_ev}\n\n"
                f"Sufficiency assessment: {suff}"
            )
        return JUDGMENT_USER_TEMPLATE.substitute(
            PICO=pico_text,
            CRITERION=criterion,
            CRITERION_QUESTION=question,
            SOURCE=source,
            CANDIDATE_JUDGEMENT=candidate.get("judgement", ""),
            CANDIDATE_RATIONALE=candidate.get("rationale", ""),
            EVIDENCE_BLOCK=evidence_block,
        )
    raise ValueError(f"unknown task: {task!r}")


# Mapping from task -> the target-dict key carrying the label string.
TARGET_KEYS: dict[str, str] = {
    "routing": "action",
    "sufficiency": "sufficiency",
    "judgment": "quality",
}


def extract_target_label(task: str, target_obj: dict) -> str:
    """Pull the label string from a record's ``target`` dict."""
    key = TARGET_KEYS[task]
    label = target_obj.get(key)
    if not isinstance(label, str) or not label:
        raise ValueError(f"missing target.{key} for task {task!r}")
    return label

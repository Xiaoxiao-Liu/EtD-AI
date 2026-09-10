"""Inference workflow that mirrors the paper's Fig.1 deployment pipeline.

For each (PICO × subquestion) pair:

  ROUTE
    ├── direct_answer        → generate candidate_no_evidence → deploy it.
    └── retrieve_evidence    → retrieve top-k from RAG corpus.
                                SUFFICE
                                  ├── sufficient
                                  └── retrieve_more  → re-retrieve with more.
                                Generate a candidate and run the single-candidate
                                JUDGE accept/reject gate with fallback if needed.

All decisions and intermediates are saved so we can score downstream.

This module does NOT touch existing code. It imports:
  - ChatBot                 (LLM client)
  - BGERetriever            (RAG retrieval)
  - END2END / RAG prompts   (candidate generation; existing run_methods.prompts)
  - SFT policy prompts      (ROUTE/SUFFICE/JUDGE; existing train.sft.prompts via PolicyClient)
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.common.agent import ChatBot
from src.common.io import read_etd_file, save_output
from src.common.parsing import extract_model_output, parse_judgement_options
from src.task.build_rag.retriever import BGERetriever, DEFAULT_TOP_K
from src.task.run_methods.judgement import judgement_compare
from src.task.run_methods.policy_workflow.policy_client import PolicyClient
from src.task.run_methods.prompts import (
    END2END_SYSTEM_PROMPT,
    END2END_USER_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
)
from src.task.run_methods.rag import build_retrieval_query


def _format_options(options: dict) -> str:
    return "\n".join(f"- {key}" for key in options)


def _gen_candidate_no_evidence(agent: ChatBot, pico: dict, section: dict, options: dict) -> dict:
    sys_prompt = END2END_SYSTEM_PROMPT.substitute(
        POPULATION=pico["Population"],
        INTERVENTION=pico["Intervention"],
        COMPARISON=pico["Comparison"],
        MAIN_OUTCOMES=pico["Main outcomes"],
    )
    user_prompt = END2END_USER_PROMPT.substitute(
        CRITERION=section["criterion"],
        SUBQUESTION=section["question"],
        JUDGEMENT_OPTIONS=_format_options(options),
    )
    response = agent.call(sys_prompt, user_prompt)
    judgement, reason = extract_model_output(response)
    return {
        "judgement": judgement,
        "rationale": reason,
        "raw": response,
    }


def _gen_candidate_rag(
    agent: ChatBot,
    pico: dict,
    section: dict,
    options: dict,
    retrieved_text: str,
) -> dict:
    sys_prompt = RAG_SYSTEM_PROMPT.substitute(
        POPULATION=pico["Population"],
        INTERVENTION=pico["Intervention"],
        COMPARISON=pico["Comparison"],
        MAIN_OUTCOMES=pico["Main outcomes"],
    )
    user_prompt = RAG_USER_PROMPT.substitute(
        CRITERION=section["criterion"],
        SUBQUESTION=section["question"],
        RESEARCH_EVIDENCE=retrieved_text or "No relevant evidence retrieved.",
        JUDGEMENT_OPTIONS=_format_options(options),
    )
    response = agent.call(sys_prompt, user_prompt)
    judgement, reason = extract_model_output(response)
    return {
        "judgement": judgement,
        "rationale": reason,
        "raw": response,
    }


def _pico_for_policy(pico: dict) -> dict:
    return {
        "Population": pico.get("Population", ""),
        "Intervention": pico.get("Intervention", ""),
        "Comparison": pico.get("Comparison", ""),
        "Main outcomes": pico.get("Main outcomes", ""),
    }


def _cand_dict(cand: dict) -> dict:
    """Extract {judgement, rationale} for policy input."""
    return {
        "judgement": cand.get("judgement") or "",
        "rationale": cand.get("rationale") or "",
    }


def _run_rag_subflow(
    pico: dict,
    section: dict,
    options: dict,
    *,
    source_file: str,
    pico_text: dict,
    criterion: str,
    question: str,
    policy: PolicyClient,
    generator: ChatBot,
    retriever: BGERetriever,
    top_k: int,
    top_k_more: int,
    exclude_self_pico: bool,
    record: dict,
) -> tuple[Optional[dict], str, Optional[str]]:
    """Run retrieve → SUFFICE → generate cand_rag. Returns (cand_rag, evidence_used, suffice_label)."""
    query = build_retrieval_query(pico, section)
    retrieved_text, retrieved_chunks = retriever.retrieve_formatted(
        query,
        top_k=top_k,
        exclude_source_file=source_file if exclude_self_pico else None,
    )
    record["rag_evidence"] = retrieved_text
    record["rag_chunks"] = retrieved_chunks

    suffice = policy.suffice(pico_text, criterion, question, retrieved_text)
    record["suffice"] = {"label": suffice.label, "raw": suffice.raw}

    if suffice.label == "retrieve_more":
        retrieved_text_more, retrieved_chunks_more = retriever.retrieve_formatted(
            query,
            top_k=top_k_more,
            exclude_source_file=source_file if exclude_self_pico else None,
        )
        record["rag_evidence_more"] = retrieved_text_more
        record["rag_chunks_more"] = retrieved_chunks_more
        evidence_for_cand = retrieved_text_more
    else:
        evidence_for_cand = retrieved_text

    cand_rag = _gen_candidate_rag(generator, pico, section, options, evidence_for_cand)
    record["candidate_rag"] = cand_rag
    return cand_rag, evidence_for_cand, suffice.label


def process_subquestion(
    pico: dict,
    section: dict,
    *,
    source_file: str,
    policy: PolicyClient,
    generator: ChatBot,
    retriever: BGERetriever,
    top_k: int,
    top_k_more: int,
    exclude_self_pico: bool,
) -> dict:
    options = parse_judgement_options(section["judgement_extract"])
    pico_text = _pico_for_policy(pico)
    criterion = section["criterion"]
    question = section["question"]

    record: dict = {
        "Population": pico["Population"],
        "Intervention": pico["Intervention"],
        "Comparison": pico["Comparison"],
        "Main outcomes": pico["Main outcomes"],
        "criterion": criterion,
        "question": question,
        "gt_judgement": options,
        "additional_consideration": section.get("additional_considerations"),
    }

    rag_subflow_kwargs = dict(
        source_file=source_file,
        pico_text=pico_text,
        criterion=criterion,
        question=question,
        policy=policy,
        generator=generator,
        retriever=retriever,
        top_k=top_k,
        top_k_more=top_k_more,
        exclude_self_pico=exclude_self_pico,
        record=record,
    )

    # ---- Step 1: ROUTE ---------------------------------------------------- #
    route = policy.route(pico_text, criterion, question)
    record["route"] = {"label": route.label, "raw": route.raw}

    deployed_source: Optional[str] = None
    deployed: Optional[dict] = None

    if route.label == "direct_answer":
        # --- direct_answer path ---
        cand_no = _gen_candidate_no_evidence(generator, pico, section, options)
        record["candidate_no_evidence"] = cand_no

        judge_no = policy.judge(
            pico_text, criterion, question,
            candidate=_cand_dict(cand_no),
            source="no_evidence",
        )
        record["judge_first"] = {
            "label": judge_no.label, "source": "no_evidence", "raw": judge_no.raw,
        }

        if judge_no.label == "accept":
            deployed_source, deployed = "no_evidence", cand_no
        else:
            # Fallback: run RAG sub-flow
            cand_rag, evidence_used, suffice_label = _run_rag_subflow(
                pico, section, options, **rag_subflow_kwargs,
            )
            judge_rag = policy.judge(
                pico_text, criterion, question,
                candidate=_cand_dict(cand_rag),
                source="rag_evidence",
                rag_evidence=evidence_used,
                sufficiency_assessment=suffice_label or "retrieve_more",
            )
            record["judge_fallback"] = {
                "label": judge_rag.label, "source": "rag_evidence", "raw": judge_rag.raw,
            }
            if judge_rag.label == "accept":
                deployed_source, deployed = "rag_evidence", cand_rag
            else:
                deployed_source = "needs_human_review"

    else:
        # --- retrieve_evidence path ---
        cand_rag, evidence_used, suffice_label = _run_rag_subflow(
            pico, section, options, **rag_subflow_kwargs,
        )
        judge_rag = policy.judge(
            pico_text, criterion, question,
            candidate=_cand_dict(cand_rag),
            source="rag_evidence",
            rag_evidence=evidence_used,
            sufficiency_assessment=suffice_label or "retrieve_more",
        )
        record["judge_first"] = {
            "label": judge_rag.label, "source": "rag_evidence", "raw": judge_rag.raw,
        }

        if judge_rag.label == "accept":
            deployed_source, deployed = "rag_evidence", cand_rag
        else:
            # Fallback: try no_evidence
            cand_no = _gen_candidate_no_evidence(generator, pico, section, options)
            record["candidate_no_evidence"] = cand_no

            judge_no = policy.judge(
                pico_text, criterion, question,
                candidate=_cand_dict(cand_no),
                source="no_evidence",
            )
            record["judge_fallback"] = {
                "label": judge_no.label, "source": "no_evidence", "raw": judge_no.raw,
            }
            if judge_no.label == "accept":
                deployed_source, deployed = "no_evidence", cand_no
            else:
                deployed_source = "needs_human_review"

    record["deployed_source"] = deployed_source
    record["deployed_judgement"] = (deployed or {}).get("judgement")
    record["deployed_rationale"] = (deployed or {}).get("rationale")
    if deployed_source == "needs_human_review":
        record["judgement_result"] = "needs_human_review"
    else:
        record["judgement_result"] = judgement_compare(
            record["deployed_judgement"], options
        )
    return record


def process_item(
    item: dict,
    results_dir: Path,
    *,
    policy,
    generator: ChatBot,
    retriever: BGERetriever,
    top_k: int,
    top_k_more: int,
    exclude_self_pico: bool,
) -> tuple[str, Optional[str]]:
    source_stem = item["source_file"].split(".")[0]
    output_path = results_dir / f"{source_stem}_policy.json"
    if output_path.exists():
        return item["source_file"], "skip"

    results: dict = {
        "PICO_QUESTION": item["pico"]["Question"],
        "source_file": source_stem,
        "policy_model": policy.model_name,
        "generator_model": generator.model_name,
        "subquestion": [],
    }

    for idx in range(len(item["sections"])):
        section = item["sections"][idx]
        record = process_subquestion(
            item["pico"],
            section,
            source_file=item["source_file"],
            policy=policy,
            generator=generator,
            retriever=retriever,
            top_k=top_k,
            top_k_more=top_k_more,
            exclude_self_pico=exclude_self_pico,
        )
        results["subquestion"].append(record)

    save_output(results, output_path)
    return item["source_file"], None


SPLIT_CHOICES = ("test", "val", "train", "all")
DEFAULT_PICO_SPLITS_PATH = Path("dataset/train/sft/pico_splits.json")


def _load_pico_splits(etd_root: Path) -> dict[str, str]:
    """Load PICO -> split_name mapping written by src/task/train/sft/prepare_data.py.

    Raises FileNotFoundError with a clear hint if the split file is missing.
    """
    path = etd_root / DEFAULT_PICO_SPLITS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"pico_splits.json not found at {path}. "
            f"Run: bash scripts/train/sft/prepare_data.sh first "
            f"(or pass --split all to bypass split filtering)."
        )
    with path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    split_map = splits.get("split_map") if isinstance(splits, dict) else None
    if not isinstance(split_map, dict):
        raise ValueError(
            f"Unexpected pico_splits.json schema at {path}: "
            f"expected top-level 'split_map' dict."
        )
    return split_map


def _filter_by_split(etd_data: list, split: str, etd_root: Path) -> list:
    """Return only PICOs whose source_file is in the requested split.

    `split == 'all'` skips filtering entirely (legacy behaviour). Any other
    value reads pico_splits.json and keeps only matching items.
    """
    if split == "all":
        return etd_data
    if split not in SPLIT_CHOICES:
        raise ValueError(
            f"split must be one of {SPLIT_CHOICES}, got {split!r}"
        )
    split_map = _load_pico_splits(etd_root)
    filtered = [
        item for item in etd_data
        if split_map.get(item["source_file"]) == split
    ]
    return filtered


def run(
    *,
    etd_root: Path,
    dataset_filename: str,
    policy,
    generator: ChatBot,
    retriever: BGERetriever,
    top_k: int = DEFAULT_TOP_K,
    top_k_more: int = 10,
    exclude_self_pico: bool = False,
    workers: int = 1,
    limit_picos: Optional[int] = None,
    results_subdir: str = "policy_workflow",
    split: str = "test",
    output_tag: Optional[str] = None,
) -> None:
    data_path = etd_root / "dataset" / dataset_filename
    etd_data_all = read_etd_file(data_path)
    etd_data = _filter_by_split(etd_data_all, split, etd_root)

    # Output subdir: ``<generator>__<policy>`` so a local-policy run never
    # overwrites an API-policy run with the same generator (and vice versa).
    # Caller may override via ``output_tag``.
    tag = output_tag or f"{generator.model_name}__{policy.model_name}"

    # Tag the output dir by split so train/val/test runs never overwrite
    # each other. `all` keeps the legacy flat layout for backward compat.
    if split == "all":
        results_dir = (
            etd_root
            / "dataset"
            / "run_methods"
            / results_subdir
            / "output"
            / tag
        )
    else:
        results_dir = (
            etd_root
            / "dataset"
            / "run_methods"
            / results_subdir
            / "output"
            / tag
            / split
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    scan_items = etd_data[:limit_picos] if limit_picos else etd_data
    pending = []
    for item in scan_items:
        stem = item["source_file"].split(".")[0]
        if not (results_dir / f"{stem}_policy.json").exists():
            pending.append(item)

    print(
        f"split={split}, "
        f"total_in_file={len(etd_data_all)}, in_split={len(etd_data)}, "
        f"scan={len(scan_items)}, "
        f"skipped(existing)={len(scan_items) - len(pending)}, "
        f"pending={len(pending)}, workers={workers}, "
        f"policy={policy.model_name}, generator={generator.model_name}, "
        f"out={results_dir}"
    )
    if not pending:
        print("Nothing to do.")
        return

    disable_progress = not sys.stderr.isatty()
    desc = f"policy_workflow ({tag})"

    if workers <= 1:
        for item in tqdm(pending, desc=desc, unit="item", disable=disable_progress):
            process_item(
                item,
                results_dir,
                policy=policy,
                generator=generator,
                retriever=retriever,
                top_k=top_k,
                top_k_more=top_k_more,
                exclude_self_pico=exclude_self_pico,
            )
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_item,
                item,
                results_dir,
                policy=policy,
                generator=generator,
                retriever=retriever,
                top_k=top_k,
                top_k_more=top_k_more,
                exclude_self_pico=exclude_self_pico,
            )
            for item in pending
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(pending),
            desc=desc,
            unit="item",
            disable=disable_progress,
        ):
            future.result()

"""CLI entry point for the policy workflow runner.

Mirrors src/task/run_methods/run_rag.py in style. Reuses the existing BGE-M3
retriever, prompt templates, and ChatBot — no edits to those files.

Typical invocation (local policy — default):
  CUDA_VISIBLE_DEVICES=0 python -m src.task.run_methods.policy_workflow.run \
      --generator-model gpt-4o --split test

API policy (no GPU needed):
  python -m src.task.run_methods.policy_workflow.run \
      --policy-backend api --policy-model gpt-4o \
      --generator-model gpt-4o --workers 2

Ablation — SFT-only (skip all DPO adapters):
  CUDA_VISIBLE_DEVICES=0 python -m src.task.run_methods.policy_workflow.run \
      --dpo-roles none --generator-model gpt-4o --split test

Ablation — RL-only (base-init, unweighted DPO; no SFT adapter):
  CUDA_VISIBLE_DEVICES=0 python -m src.task.run_methods.policy_workflow.run \
      --rl-only --generator-model gpt-4o --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ETD_ROOT = Path(__file__).resolve().parents[4]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

from src.task.build_rag.paths import (
    default_bge_model_dir,
    default_corpus_path,
    default_index_dir,
)
DATASET_FILENAME = "pico_sections_icd11.json"

BACKEND_CHOICES = ("local", "api")
DEFAULT_TOP_K = 5
SPLIT_CHOICES = ("test", "val", "train", "all")


def _build_policy(args):
    """Construct the policy client based on --policy-backend."""
    if args.policy_backend == "local":
        from src.task.run_methods.policy_workflow.local_policy_client import (
            LocalPolicyClient,
        )

        roles = set() if args.dpo_roles == "none" else set(args.dpo_roles.split(","))
        if args.no_dpo_sufficiency:
            roles.discard("sufficiency")
        return LocalPolicyClient(
            use_dpo_routing="routing" in roles,
            use_dpo_sufficiency="sufficiency" in roles,
            use_dpo_judgment="judgment" in roles,
            rl_only=args.rl_only,
        )

    from src.task.run_methods.policy_workflow.policy_client import PolicyClient

    return PolicyClient(model_name=args.policy_model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the policy workflow: ROUTE -> retrieve -> SUFFICE -> "
            "generate candidates -> JUDGE."
        )
    )
    parser.add_argument(
        "--policy-backend",
        choices=BACKEND_CHOICES,
        default="local",
        help=(
            "Policy backend. 'local' loads Qwen3-8B + LoRA adapters on GPU; "
            "'api' uses a chat LLM via ChatBot. Default: local."
        ),
    )
    parser.add_argument(
        "--policy-model",
        default="gpt-4o",
        help=(
            "LLM playing the policy role (only used when --policy-backend=api). "
            "Default: gpt-4o."
        ),
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help=(
            "Load the three base-initialized, uniformly weighted DPO adapters "
            "without loading the SFT adapter. Only valid for local policy."
        ),
    )
    parser.add_argument(
        "--dpo-roles",
        default="routing,sufficiency,judgment",
        help=(
            "Comma-separated local DPO roles: routing,sufficiency,judgment; "
            "use 'none' for SFT-only. Default enables all three roles."
        ),
    )
    parser.add_argument(
        "--no-dpo-sufficiency",
        action="store_true",
        help=(
            "Ablation: use SFT adapter for ALL tasks (skip DPO adapter for "
            "sufficiency). Only meaningful with --policy-backend=local."
        ),
    )
    parser.add_argument(
        "--generator-model",
        default="gpt-4o",
        help=(
            "LLM that generates candidate judgements. Also used in the output "
            "directory name. Default: gpt-4o."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="RAG top_k for the first retrieval (default: 5).",
    )
    parser.add_argument(
        "--top-k-more",
        type=int,
        default=10,
        help="RAG top_k when SUFFICE says retrieve_more (default: 10).",
    )
    parser.add_argument(
        "--exclude-self-pico",
        action="store_true",
        help=(
            "Leave-one-PICO-out retrieval (deployment-like). Default: include "
            "all chunks (evaluation-like)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Thread count for parallel PICO processing (default: 1).",
    )
    parser.add_argument(
        "--limit-picos",
        type=int,
        default=None,
        help="Only process first N PICOs (after split filter). Use 2 for smoke.",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="test",
        help=(
            "Which PICO split to evaluate on (default: test). "
            "Reads dataset/train/sft/pico_splits.json. Use 'all' to bypass "
            "filtering — but DO NOT use 'all' for evaluation once a local "
            "Qwen3-8B policy is plugged in (data leakage)."
        ),
    )
    args = parser.parse_args()

    if args.rl_only and args.policy_backend != "local":
        parser.error("--rl-only requires --policy-backend local")

    if args.dpo_roles != "none":
        requested_roles = set(args.dpo_roles.split(","))
        unknown_roles = requested_roles - {"routing", "sufficiency", "judgment"}
        if unknown_roles:
            parser.error(f"unknown --dpo-roles: {sorted(unknown_roles)}")

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.top_k < 1 or args.top_k_more < 1:
        parser.error("--top-k / --top-k-more must be >= 1")
    if args.limit_picos is not None and args.limit_picos < 1:
        parser.error("--limit-picos must be >= 1")

    # Local policy is not thread-safe (single GPU model); force serial.
    if args.policy_backend == "local" and args.workers > 1:
        print(
            f"WARNING: --policy-backend=local forces workers=1 "
            f"(was {args.workers}). GPU inference is not thread-safe."
        )
        args.workers = 1

    # Build policy client (local LoRA or API).
    policy = _build_policy(args)

    # Build generator (always an API ChatBot).
    from src.common.agent import ChatBot
    from src.task.build_rag.corpus import require_corpus
    from src.task.build_rag.retriever import BGERetriever
    from src.task.run_methods.policy_workflow.workflow import run
    generator = ChatBot(args.generator_model)

    corpus_path = default_corpus_path(_ETD_ROOT)
    index_dir = default_index_dir(_ETD_ROOT)
    bge_dir = default_bge_model_dir(_ETD_ROOT)

    require_corpus(corpus_path)
    retriever = BGERetriever.load(corpus_path, index_dir, model_dir=bge_dir)

    run(
        etd_root=_ETD_ROOT,
        dataset_filename=DATASET_FILENAME,
        policy=policy,
        generator=generator,
        retriever=retriever,
        top_k=args.top_k,
        top_k_more=args.top_k_more,
        exclude_self_pico=args.exclude_self_pico,
        workers=args.workers,
        limit_picos=args.limit_picos,
        split=args.split,
    )


if __name__ == "__main__":
    main()

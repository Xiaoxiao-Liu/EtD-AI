#!/usr/bin/env bash
# Run the policy workflow (ROUTE -> retrieve -> SUFFICE -> generate -> JUDGE).
#
# Two backends:
#   POLICY_BACKEND=local  (default) — loads Qwen3-8B + LoRA on GPU, needs
#                         CUDA_VISIBLE_DEVICES set. Workers forced to 1.
#   POLICY_BACKEND=api    — uses a chat LLM via ChatBot, no GPU needed.
#
# Requires pre-built BGE-M3 corpus + index (scripts/build_rag/*).
#
# Examples:
#   # 默认：本地 policy（SFT+DPO）+ gpt-4o generator，test split
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_methods/policy_workflow/run_policy_workflow.sh
#
#   # 冒烟测试：只跑 2 个 PICO
#   CUDA_VISIBLE_DEVICES=0 LIMIT=2 bash scripts/run_methods/policy_workflow/run_policy_workflow.sh
#
#   # Ablation：SFT-only（跳过 DPO adapter）
#   CUDA_VISIBLE_DEVICES=0 NO_DPO_SUFFICIENCY=1 bash scripts/run_methods/policy_workflow/run_policy_workflow.sh
#
#   # Ablation：RL-only（base-init + unweighted DPO，不加载 SFT）
#   CUDA_VISIBLE_DEVICES=0 RL_ONLY=1 bash scripts/run_methods/policy_workflow/run_policy_workflow.sh
#
#   # API policy（不需要 GPU）
#   POLICY_BACKEND=api POLICY_MODEL=gpt-4o WORKERS=2 \
#       bash scripts/run_methods/policy_workflow/run_policy_workflow.sh

set -euo pipefail

cd "$(dirname "$0")/../../.."

POLICY_BACKEND="${POLICY_BACKEND:-local}"
POLICY_MODEL="${POLICY_MODEL:-gpt-4o}"
GENERATOR_MODEL="${GENERATOR_MODEL:-gpt-4o}"
TOP_K="${TOP_K:-5}"
TOP_K_MORE="${TOP_K_MORE:-10}"
WORKERS="${WORKERS:-1}"
LIMIT="${LIMIT:-}"
EXCLUDE_SELF_PICO="${EXCLUDE_SELF_PICO:-0}"
SPLIT="${SPLIT:-test}"
NO_DPO_SUFFICIENCY="${NO_DPO_SUFFICIENCY:-0}"
DPO_ROLES="${DPO_ROLES:-routing,sufficiency,judgment}"
RL_ONLY="${RL_ONLY:-0}"

cmd=(python -m src.task.run_methods.policy_workflow.run
     --policy-backend "$POLICY_BACKEND"
     --policy-model "$POLICY_MODEL"
     --generator-model "$GENERATOR_MODEL"
     --dpo-roles "$DPO_ROLES"
     --top-k "$TOP_K"
     --top-k-more "$TOP_K_MORE"
     --workers "$WORKERS"
     --split "$SPLIT")

if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit-picos "$LIMIT")
fi
if [[ "$EXCLUDE_SELF_PICO" == "1" ]]; then
    cmd+=(--exclude-self-pico)
fi
if [[ "$NO_DPO_SUFFICIENCY" == "1" ]]; then
    cmd+=(--no-dpo-sufficiency)
fi
if [[ "$RL_ONLY" == "1" ]]; then
    cmd+=(--rl-only)
fi

echo "+ ${cmd[*]}"
exec "${cmd[@]}"

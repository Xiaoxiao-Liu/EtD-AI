#!/usr/bin/env bash
# Run the RAG pipeline. Requires pre-built corpus + vector index
# (see scripts/build_rag/build_corpus.sh, scripts/build_rag/build_index.sh).
#
# Evaluation mode (default): retrieve from the full corpus, including the
# gold PICO's own evidence.
#
# Examples:
#   bash scripts/run_methods/run_rag.sh
#   MODEL_NAME=gpt-4o LIMIT=1 bash scripts/run_methods/run_rag.sh
#   MODEL_NAME=gpt-4o WORKERS=4 bash scripts/run_methods/run_rag.sh --top-k 5
#   EXCLUDE_SELF_PICO=1 bash scripts/run_methods/run_rag.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

# ---------------------- 可编辑参数 ---------------------- #

MODEL_NAME="${MODEL_NAME:-gpt-5.5}"
# 被评 LLM 模型名；结果写到 dataset/run_methods/rag/output/$MODEL_NAME/    【可修改】

WORKERS="${WORKERS:-20}"
# 并发线程数                                                         【可修改】

TOP_K="${TOP_K:-5}"
# 每个 subquestion 检索的 evidence chunk 数                           【可修改】

LIMIT="${LIMIT:-}"
# 只处理前 N 个 PICO；留空 = 全部。例: LIMIT=1 只跑 1 个 PICO 冒烟   【可修改】

EXCLUDE_SELF_PICO="${EXCLUDE_SELF_PICO:-0}"
# 1 = leave-one-PICO-out；0 = 评测模式（含 gold PICO evidence）       【可修改】

# ---------------------- 组装命令 ---------------------- #

cmd=(python -m src.task.run_methods.run_rag
     --model "$MODEL_NAME"
     --workers "$WORKERS"
     --top-k "$TOP_K")

[[ -n "$LIMIT" ]] && cmd+=(--limit-picos "$LIMIT")
[[ "$EXCLUDE_SELF_PICO" == "1" ]] && cmd+=(--exclude-self-pico)

echo "+ ${cmd[*]} $*"
"${cmd[@]}" "$@"

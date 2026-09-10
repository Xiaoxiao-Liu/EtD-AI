#!/usr/bin/env bash
# ============================================================================
# LLM judge rubric 评分脚本
# ----------------------------------------------------------------------------
# 对 all_models.scored.jsonl 里每条 (PICO, criterion, model) 调用 judge LLM，
# 打 4 个维度 (no_evidence 时 grounding 不评):
#   - reasoning_relevance
#   - reasoning_grounding   (no_evidence 该字段为 null)
#   - confidence_calibration
#   - etd_consistency
#
# 每条评完立即落盘:
#   dataset/score_rubric/judge/output/<prompt_version>/<judge_model>/<record.model_name>/
#       <source_stem>/section_<idx>.json
#
# 评完后运行 scripts/score_rubric/merge.sh 合并回 scored jsonl
#
# 用法:
#   bash scripts/score_rubric/judge.sh
#   INPUT=dataset/score_rubric/score/output/all_models.scored.jsonl LIMIT=5 bash scripts/score_rubric/judge.sh
#   OVERWRITE=1 bash scripts/score_rubric/judge.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

# ---------------------- 可编辑参数 ---------------------- #

INPUT="${INPUT:-dataset/score_rubric/score/output/all_models.scored.jsonl}"
RUBRIC_DIR="${RUBRIC_DIR:-dataset/score_rubric/judge/output}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"
WORKERS="${WORKERS:-4}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"

# ---------------------- 组装命令 ---------------------- #

cmd=(python -m src.task.score_rubric.judge_rubric
     --input "$INPUT"
     --rubric-dir "$RUBRIC_DIR"
     --judge-model "$JUDGE_MODEL"
     --workers "$WORKERS")

[[ -n "$LIMIT"  ]] && cmd+=(--limit  "$LIMIT")
[[ "$OVERWRITE" == "1" ]] && cmd+=(--overwrite)

echo "+ ${cmd[*]}"
"${cmd[@]}" "$@"

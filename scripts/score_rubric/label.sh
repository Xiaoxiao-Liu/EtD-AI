#!/usr/bin/env bash
# Policy-label 计算脚本（需先 score_rubric/merge.sh 产出的 all_models.scored.jsonl）
#
# 用法:
#   bash scripts/score_rubric/label.sh
#   INPUT=dataset/score_rubric/merge/output/all_models.scored.jsonl \
#       bash scripts/score_rubric/label.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

INPUT="${INPUT:-dataset/score_rubric/merge/output/all_models.scored.jsonl}"
OUTPUT="${OUTPUT:-}"
EVIDENCE_SUFFICIENCY_THRESHOLD="${EVIDENCE_SUFFICIENCY_THRESHOLD:-0.3}"
ORACLE_GAP_THRESHOLD="${ORACLE_GAP_THRESHOLD:-0.5}"

cmd=(python -m src.task.score_rubric.label_policy
     --input "$INPUT"
     --evidence-sufficiency-threshold "$EVIDENCE_SUFFICIENCY_THRESHOLD"
     --oracle-gap-threshold "$ORACLE_GAP_THRESHOLD")

[[ -n "$OUTPUT" ]] && cmd+=(--output "$OUTPUT")

echo "+ ${cmd[*]}"
"${cmd[@]}" "$@"

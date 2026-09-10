#!/usr/bin/env bash
# Merge judge/<judge>/<model>/.../section_*.json back into scored JSONL.
# Output lands at dataset/score_rubric/merge/output/all_models.scored.jsonl
#
# Example:
#   bash scripts/score_rubric/merge.sh
#   JUDGE_MODEL=gpt-5.5 INPUT=dataset/score_rubric/score/output/all_models.scored.jsonl \
#       bash scripts/score_rubric/merge.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

INPUT="${INPUT:-dataset/score_rubric/score/output/all_models.scored.jsonl}"
OUTPUT="${OUTPUT:-}"
RUBRIC_DIR="${RUBRIC_DIR:-dataset/score_rubric/judge/output}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"

cmd=(python -m src.task.score_rubric.merge_rubric
     --input "$INPUT"
     --rubric-dir "$RUBRIC_DIR"
     --judge-model "$JUDGE_MODEL")

[[ -n "$OUTPUT" ]] && cmd+=(--output "$OUTPUT")

echo "+ ${cmd[*]}"
"${cmd[@]}"

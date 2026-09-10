#!/usr/bin/env bash
# Rule-based rubric scores (judgement + evidence correctness).
#
# Example:
#   bash scripts/score_rubric/score.sh
#   INPUT=dataset/score_rubric/extract/output/all_models.jsonl bash scripts/score_rubric/score.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

INPUT="${INPUT:-dataset/score_rubric/extract/output/all_models.jsonl}"
OUTPUT="${OUTPUT:-}"

cmd=(python -m src.task.score_rubric.score_rubric --input "$INPUT")
[[ -n "$OUTPUT" ]] && cmd+=(--output "$OUTPUT")

echo "+ ${cmd[*]}"
"${cmd[@]}" "$@"

#!/usr/bin/env bash
# Join end2end / rag / with_evidence results into one JSONL (all models).
#
# Examples:
#   bash scripts/score_rubric/extract.sh
#   MODELS=gpt-4o,gpt-5.5 bash scripts/score_rubric/extract.sh
#   MODELS=gpt-4o LIMIT=1 bash scripts/score_rubric/extract.sh
#   INCLUDE_SPLITS=test OUTPUT=dataset/score_rubric/extract/output/test_models.jsonl \
#       bash scripts/score_rubric/extract.sh  # final evaluation only
set -euo pipefail

cd "$(dirname "$0")/../.."

MODELS="${MODELS:-gpt-4o}"
OUTPUT="${OUTPUT:-dataset/score_rubric/extract/output/all_models.jsonl}"
LIMIT="${LIMIT:-}"
SPLITS_FILE="${SPLITS_FILE:-dataset/train/sft/pico_splits.json}"
INCLUDE_SPLITS="${INCLUDE_SPLITS:-train,val}"

cmd=(python -m src.task.score_rubric.data_preparation.extract_sample
     --models "$MODELS"
     --output "$OUTPUT"
     --splits-file "$SPLITS_FILE"
     --include-splits "$INCLUDE_SPLITS")

[[ -n "$LIMIT" ]] && cmd+=(--limit "$LIMIT")

echo "+ ${cmd[*]} $*"
"${cmd[@]}" "$@"

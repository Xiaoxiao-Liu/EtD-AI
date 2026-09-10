#!/usr/bin/env bash
# Freeze the PICO-level split before Step 3 rubric scoring.
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT="${INPUT:-dataset/pico_sections_icd11.json}"
OUTPUT="${OUTPUT:-dataset/train/sft/pico_splits.json}"
TRAIN_RATIO="${TRAIN_RATIO:-0.70}"
VAL_RATIO="${VAL_RATIO:-0.15}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

cmd=(python -m src.task.data_split \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED")

[[ "$FORCE" == "1" ]] && cmd+=(--force)
"${cmd[@]}"

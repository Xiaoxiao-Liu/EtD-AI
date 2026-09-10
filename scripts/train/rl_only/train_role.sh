#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

ROLE="${1:?usage: train_role.sh routing|sufficiency|judgment}"
shift
case "$ROLE" in
  routing|sufficiency|judgment) ;;
  *) echo "unknown role: $ROLE" >&2; exit 2 ;;
esac

cmd=(python -m src.task.train.dpo.sufficiency.train
  --role "$ROLE"
  --initialization base
  --uniform-pair-weights
  --base-model-dir "${BASE_MODEL_DIR:-model/Qwen3-8B}"
  --data-dir "${DATA_DIR:-dataset/train/rl_only/$ROLE/output}"
  --output-dir "${OUTPUT_DIR:-model/policy/rl_only_$ROLE}"
  --seed "${SEED:-42}" --beta "${BETA:-0.1}"
  --num-train-epochs "${EPOCHS:-1}"
  --per-device-train-batch-size "${BATCH:-2}"
  --per-device-eval-batch-size "${EVAL_BATCH:-2}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}"
  --learning-rate "${LR:-5e-6}" --max-length "${MAX_LENGTH:-4096}"
  --max-steps "${MAX_STEPS:--1}" --eval-steps "${EVAL_STEPS:-200}"
  --save-steps "${SAVE_STEPS:-200}" --logging-steps "${LOGGING_STEPS:-10}"
  --warmup-ratio "${WARMUP_RATIO:-0.05}")

echo "+ ${cmd[*]} $*"
"${cmd[@]}" "$@"

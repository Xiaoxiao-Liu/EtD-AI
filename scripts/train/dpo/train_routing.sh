#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

cmd=(python -m src.task.train.dpo.routing.train
  --role routing
  --base-model-dir "${BASE_MODEL_DIR:-model/Qwen3-8B}"
  --sft-adapter-dir "${SFT_ADAPTER_DIR:-model/policy/sft}"
  --output-dir "${OUTPUT_DIR:-model/policy/dpo_routing}"
  --data-dir "${DATA_DIR:-dataset/train/dpo/routing/output}"
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

#!/usr/bin/env bash
# ============================================================================
# DPO sufficiency 训练 (sample-weighted, weight = |gt_score - rag_score|)
# ----------------------------------------------------------------------------
# 在 SFT adapter (model/policy/sft/) 基础上继续训, 只对 sufficiency 子流程
# 做偏好对齐, 产出:
#   model/policy/dpo_sufficiency/  (新一份 LoRA adapter; 不与 SFT 合并)
#
# 参考: src/task/train/dpo/sufficiency/train.py (手写 weighted DPO, 不依赖 trl,
# reference model = 同一 base + SFT adapter 禁用, 省一倍显存)。
#
# 用法:
#   bash scripts/train/dpo/train_sufficiency.sh
#
#   # smoke
#   MAX_STEPS=2 bash scripts/train/dpo/train_sufficiency.sh
#
#   # 自定义超参
#   BETA=0.2 LR=3e-6 EPOCHS=2 bash scripts/train/dpo/train_sufficiency.sh
#
#   # 接入 wandb (默认开启):
#   WANDB_PROJECT=etd-ai bash scripts/train/dpo/train_sufficiency.sh
#
#   # 训完自动推到 Hugging Face Hub:
#   HF_REPO_ID=your-org/etd-policy-dpo-sufficiency \
#     bash scripts/train/dpo/train_sufficiency.sh
#
# 环境变量速查:
#   WANDB_DISABLED=1                关闭 wandb
#   WANDB_PROJECT=etd-ai            wandb project 名
#   WANDB_RUN_NAME=...              自定义 run name
#   HF_REPO_ID=org/repo             非空才会推送
#   HF_PRIVATE=1                    创建私有 repo
#
# 前置条件:
#   1) bash scripts/train/sft/train.sh         # 产出 model/policy/sft/
#   2) bash scripts/train/dpo/prepare_data.sh  # 仅产出 train/val.jsonl
#   3) wandb login                             # 首次用 wandb
#   4) huggingface-cli login                   # 如需推送
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/../../.."

# ---------------------- 路径 ---------------------- #
BASE_MODEL_DIR="${BASE_MODEL_DIR:-model/Qwen3-8B}"
SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:-model/policy/sft}"
OUTPUT_DIR="${OUTPUT_DIR:-model/policy/dpo_sufficiency}"
DATA_DIR="${DATA_DIR:-dataset/train/dpo/sufficiency/output}"

# ---------------------- 训练超参 ---------------------- #
SEED="${SEED:-42}"
BETA="${BETA:-0.1}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-2}"
EVAL_BATCH="${EVAL_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:--1}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

# ---------------------- 组装命令 ---------------------- #
cmd=(python -m src.task.train.dpo.sufficiency.train
     --base-model-dir "$BASE_MODEL_DIR"
     --sft-adapter-dir "$SFT_ADAPTER_DIR"
     --output-dir "$OUTPUT_DIR"
     --data-dir "$DATA_DIR"
     --seed "$SEED"
     --beta "$BETA"
     --num-train-epochs "$EPOCHS"
     --per-device-train-batch-size "$BATCH"
     --per-device-eval-batch-size "$EVAL_BATCH"
     --gradient-accumulation-steps "$GRAD_ACCUM"
     --learning-rate "$LR"
     --max-length "$MAX_LENGTH"
     --max-steps "$MAX_STEPS"
     --eval-steps "$EVAL_STEPS"
     --save-steps "$SAVE_STEPS"
     --logging-steps "$LOGGING_STEPS"
     --warmup-ratio "$WARMUP_RATIO")

echo "+ ${cmd[*]} $*"
"${cmd[@]}" "$@"

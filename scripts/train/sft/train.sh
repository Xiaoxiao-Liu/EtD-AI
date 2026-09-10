#!/usr/bin/env bash
# ============================================================================
# 多任务 SFT 入口 (routing / sufficiency / judgment 共训, 共享一份 LoRA adapter)
# ----------------------------------------------------------------------------
# 一次跑完三 task 的 SFT warm-up, 产出
#   model/policy/sft/  (adapter + tokenizer + train_meta.json)
#
# 三个 per-task wrapper (train_routing.sh / train_sufficiency.sh /
# train_judgment.sh) 都等价于这个脚本——共训意味着 "选哪个 task" 没有意义,
# 一份 adapter 同时学三件事。保留三个文件是为了和 plan 的目录结构对齐。
#
# 用法:
#   bash scripts/train/sft/train.sh
#
#   # smoke test (2 step, 每 task 各取 16 条)
#   MAX_STEPS=2 SAMPLE_CAP_ROUTING=16 SAMPLE_CAP_SUFFICIENCY=16 \
#     SAMPLE_CAP_JUDGMENT=16 bash scripts/train/sft/train.sh
#
#   # 自定义超参
#   LR=1e-4 EPOCHS=2 BATCH=4 GRAD_ACCUM=8 bash scripts/train/sft/train.sh
#
#   # 透传额外 flag (例如只训其中两个 task):
#   bash scripts/train/sft/train.sh --tasks routing sufficiency
#
#   # 接入 wandb (默认开启, 想关就置 WANDB_DISABLED=1):
#   WANDB_PROJECT=etd-ai bash scripts/train/sft/train.sh
#
#   # 训完自动推送到 Hugging Face Hub (需先 huggingface-cli login):
#   HF_REPO_ID=your-org/etd-policy-sft bash scripts/train/sft/train.sh
#   HF_REPO_ID=your-org/etd-policy-sft HF_PRIVATE=1 bash scripts/train/sft/train.sh
#
# 环境变量速查:
#   WANDB_DISABLED=1                关闭 wandb (默认开启)
#   WANDB_PROJECT=etd-ai            wandb project 名 (默认 etd-ai)
#   WANDB_RUN_NAME=...              自定义 run name (默认 sft-<output>-<时间戳>)
#   HF_REPO_ID=org/repo             非空才会训完后推送
#   HF_PRIVATE=1                    创建私有 repo
#
# 前置条件:
#   1) bash scripts/download_model.sh  (或 python -m src.common.download_model --model qwen3-8b)
#   2) bash scripts/train/sft/prepare_data.sh
#   3) wandb login                  (首次用 wandb)
#   4) huggingface-cli login        (如需推送)
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/../../.."

# ---------------------- 路径 ---------------------- #
BASE_MODEL_DIR="${BASE_MODEL_DIR:-model/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-model/policy/sft}"

# ---------------------- 训练超参 ---------------------- #
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-2}"                # per-device train batch
EVAL_BATCH="${EVAL_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"     # 有效 batch = BATCH * GRAD_ACCUM = 32
LR="${LR:-1.5e-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:--1}"       # >0 时盖过 EPOCHS, 用于 smoke
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

# ---------------------- 每 task 抽样上限 (smoke / 调试用) ---------------------- #
SAMPLE_CAP_ROUTING="${SAMPLE_CAP_ROUTING:-}"
SAMPLE_CAP_SUFFICIENCY="${SAMPLE_CAP_SUFFICIENCY:-}"
SAMPLE_CAP_JUDGMENT="${SAMPLE_CAP_JUDGMENT:-}"

# ---------------------- 组装命令 ---------------------- #
cmd=(python -m src.task.train.sft.train
     --base-model-dir "$BASE_MODEL_DIR"
     --output-dir "$OUTPUT_DIR"
     --seed "$SEED"
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

[[ -n "$SAMPLE_CAP_ROUTING"     ]] && cmd+=(--sample-cap-routing     "$SAMPLE_CAP_ROUTING")
[[ -n "$SAMPLE_CAP_SUFFICIENCY" ]] && cmd+=(--sample-cap-sufficiency "$SAMPLE_CAP_SUFFICIENCY")
[[ -n "$SAMPLE_CAP_JUDGMENT"    ]] && cmd+=(--sample-cap-judgment    "$SAMPLE_CAP_JUDGMENT")

echo "+ ${cmd[*]} $*"
"${cmd[@]}" "$@"

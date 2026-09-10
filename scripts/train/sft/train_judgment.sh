#!/usr/bin/env bash
# ============================================================================
# SFT judgment 训练入口 (薄 wrapper)
# ----------------------------------------------------------------------------
# 注意: 三个 SFT 子任务 (routing / sufficiency / judgment) 共享一份 LoRA
# adapter, 由 src/task/train/sft/train.py 一次性多任务共训。所以这个脚本
# 与 train_routing.sh / train_sufficiency.sh / train.sh 完全等价——保留它
# 只是为了和 plan 的目录结构对齐。
#
# 推荐入口: bash scripts/train/sft/train.sh
# ============================================================================
set -euo pipefail
exec "$(dirname "$0")/train.sh" "$@"

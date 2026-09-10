#!/usr/bin/env bash
# 使用 Step 3 前已冻结的 PICO split，把 development labels 拆成 3 个 SFT task
# 的 train/val；held-out test 不生成训练格式文件。
#
# 用法:
#   bash scripts/train/sft/prepare_data.sh
#   INPUT=dataset/score_rubric/label/output/all_models.labeled.jsonl \
#       bash scripts/train/sft/prepare_data.sh
#
# 输出:
#   dataset/train/sft/pico_splits.json                       (Step 3 前创建，只读复用)
#   dataset/train/sft/<task>/output/{train,val}.jsonl
#   dataset/train/sft/<task>/output/meta.json                (label 分布、各 split 行数)
set -euo pipefail

cd "$(dirname "$0")/../../.."

INPUT="${INPUT:-dataset/score_rubric/label/output/all_models.labeled.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-dataset/train/sft}"
SPLITS_FILE="${SPLITS_FILE:-dataset/train/sft/pico_splits.json}"

cmd=(python -m src.task.train.sft.prepare_data
     --input "$INPUT"
     --output-root "$OUTPUT_ROOT"
     --splits-file "$SPLITS_FILE")

echo "+ ${cmd[*]}"
"${cmd[@]}" "$@"

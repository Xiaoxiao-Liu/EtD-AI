#!/usr/bin/env bash
# ============================================================================
# 构造 sufficiency 子流程的 DPO preference pair 数据
# ----------------------------------------------------------------------------
# 从 all_models.labeled.jsonl + dataset/train/sft/pico_splits.json (与 SFT
# 共享同一份 PICO 切分, 防泄漏) 生成 train/val preference pairs:
#   dataset/train/dpo/sufficiency/output/{train,val}.jsonl
#   dataset/train/dpo/sufficiency/output/meta.json
#
# 每条 pair:
#   {id, prompt (ChatML 字符串到 <|im_start|>assistant\n 为止),
#    chosen, rejected, weight = |gt_score - rag_score|, meta}
#
# 默认丢掉 weight < MIN_WEIGHT 的模糊样本 (gt 与 rag 分数差异过小)。
#
# 用法:
#   bash scripts/train/dpo/prepare_data.sh
#   MIN_WEIGHT=0.1 bash scripts/train/dpo/prepare_data.sh
#
# 前置条件:
#   bash scripts/prepare_pico_splits.sh      # Step 3 前冻结 pico_splits.json
#   bash scripts/train/sft/prepare_data.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/../../.."

INPUT="${INPUT:-dataset/score_rubric/label/output/all_models.labeled.jsonl}"
SPLITS_FILE="${SPLITS_FILE:-dataset/train/sft/pico_splits.json}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset/train/dpo/sufficiency/output}"
MIN_WEIGHT="${MIN_WEIGHT:-0.05}"

cmd=(python -m src.task.train.dpo.sufficiency.data_preparation.prepare
     --input "$INPUT"
     --splits-file "$SPLITS_FILE"
     --output-dir "$OUTPUT_DIR"
     --min-weight "$MIN_WEIGHT")

echo "+ ${cmd[*]} $*"
"${cmd[@]}" "$@"

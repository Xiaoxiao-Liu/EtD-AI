#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

python -m src.task.train.dpo.routing.data_preparation.prepare \
  --weight-mode "${ROUTER_WEIGHT_MODE:-outcome}" \
  --both-correct-weight "${ROUTER_BOTH_CORRECT_WEIGHT:-0.3}" \
  --min-weight "${ROUTER_MIN_WEIGHT:-0.05}"
python -m src.task.train.dpo.sufficiency.data_preparation.prepare \
  --min-weight "${GATEKEEPER_MIN_WEIGHT:-0.05}"
python -m src.task.train.dpo.judgment.data_preparation.prepare \
  --weight-mode "${VERIFIER_WEIGHT_MODE:-utility_confidence}" \
  --min-weight "${VERIFIER_MIN_WEIGHT:-0.05}"

echo "Prepared Router, Gatekeeper, and Verifier DPO datasets."

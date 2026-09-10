#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# Build the same binary preferences as the full method, but retain ambiguous
# pairs. Training overrides every stored weight to 1, yielding standard DPO.
python -m src.task.train.dpo.routing.data_preparation.prepare \
  --weight-mode unweighted \
  --min-weight 0 \
  --output-dir dataset/train/rl_only/routing/output
python -m src.task.train.dpo.sufficiency.data_preparation.prepare \
  --min-weight 0 \
  --output-dir dataset/train/rl_only/sufficiency/output
python -m src.task.train.dpo.judgment.data_preparation.prepare \
  --weight-mode unweighted \
  --min-weight 0 \
  --output-dir dataset/train/rl_only/judgment/output

echo "Prepared unweighted Router, Gatekeeper, and Verifier preference data."

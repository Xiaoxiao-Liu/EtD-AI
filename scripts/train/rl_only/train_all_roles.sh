#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

for role in routing sufficiency judgment; do
  bash scripts/train/rl_only/train_role.sh "$role" "$@"
done

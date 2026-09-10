#!/usr/bin/env bash
# Run the with-evidence pipeline (gold research_evidence in prompt).
# e.g.  bash scripts/run_methods/run_with_evidence.sh --workers 4
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m src.task.run_methods.run_with_evidence "$@"

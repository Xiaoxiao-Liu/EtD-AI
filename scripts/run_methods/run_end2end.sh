#!/usr/bin/env bash
# Run the end2end (Closed-book) pipeline. Forwards args to the Python entry,
# e.g.  bash scripts/run_methods/run_end2end.sh --workers 4
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m src.task.run_methods.run_end2end "$@"

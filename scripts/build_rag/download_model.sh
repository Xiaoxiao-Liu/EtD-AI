#!/usr/bin/env bash
# Download BGE-M3 embedding model into models/bge-m3 (idempotent).
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m src.common.download_model "$@"

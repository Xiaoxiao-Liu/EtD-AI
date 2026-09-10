#!/usr/bin/env bash
# Build the RAG corpus (chunked evidence JSONL) from pico_sections_icd11.json.
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m src.task.build_rag.build_corpus "$@"

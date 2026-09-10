#!/usr/bin/env bash
# Build BGE-M3 vector index over the RAG corpus. Pass --force to re-encode.
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m src.task.build_rag.build_index "$@"

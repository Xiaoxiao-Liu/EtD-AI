#!/usr/bin/env python3
"""
Build BGE-M3 vector index from dataset/build_rag/corpus/output/corpus.json.

Encodes all chunks once and saves embeddings under
dataset/build_rag/index/output/bge-m3/.
Run after build_corpus.py (or use build_corpus.py --build-index).

Usage (from EtD/):
    python -m src.task.build_rag.build_index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ETD_ROOT = Path(__file__).resolve().parents[3]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

from src.task.build_rag.corpus import require_corpus
from src.task.build_rag.paths import default_bge_model_dir, default_corpus_path, default_index_dir
from src.task.build_rag.retriever import build_rag_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BGE-M3 embeddings for the RAG corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=default_corpus_path(_ETD_ROOT),
        help="Path to corpus.json",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=default_index_dir(_ETD_ROOT),
        help="Directory for embeddings.npy and meta.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_bge_model_dir(_ETD_ROOT),
        help="Local BGE-M3 model directory (EtD/model/bge-m3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Encoding batch size (default: 12).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if a matching index already exists.",
    )
    args = parser.parse_args()

    require_corpus(args.corpus)
    build_rag_index(
        args.corpus,
        args.index_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()

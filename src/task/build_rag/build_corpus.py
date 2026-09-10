#!/usr/bin/env python3
"""
Build the RAG evidence corpus from pico_sections_icd11.json.

Extracts each section's research_evidence (by EtD criterion), normalizes
tables/dicts to text, chunks for retrieval, and saves with PICO context.

Optional: encode vectors in the same run (--build-index).

Usage (from EtD/):
    python -m src.task.build_rag.build_corpus
    python -m src.task.build_rag.build_corpus --build-index   # corpus + BGE vectors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ETD_ROOT = Path(__file__).resolve().parents[3]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

from src.task.build_rag.corpus import build_and_save_corpus, require_corpus
from src.task.build_rag.paths import (
    DEFAULT_CORPUS_META_PATH,
    DEFAULT_CORPUS_PATH,
    default_bge_model_dir,
    default_index_dir,
)
from src.task.build_rag.retriever import build_rag_index

DEFAULT_DATASET = _ETD_ROOT / "dataset" / "pico_sections_icd11.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RAG corpus from pico_sections_icd11.json."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Source JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Output corpus JSON (default: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_CORPUS_META_PATH,
        help=f"Output stats JSON (default: {DEFAULT_CORPUS_META_PATH})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Max characters per retrieval chunk (default: 1200).",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=150,
        help="Overlap between consecutive chunks (default: 150).",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="After corpus build, run BGE-M3 encoding (same as build_index.py).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Vector index output dir (default: dataset/rag_index/bge-m3).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_bge_model_dir(_ETD_ROOT),
        help="Local BGE-M3 model directory.",
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="With --build-index, re-encode even if index exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="BGE encoding batch size when --build-index is set.",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset not found: {args.dataset}")

    chunks = build_and_save_corpus(
        args.dataset,
        args.output,
        args.meta,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )

    print(f"Saved {len(chunks)} chunks -> {args.output}")
    print(f"Stats -> {args.meta}")

    if args.build_index:
        index_dir = args.index_dir or default_index_dir(_ETD_ROOT)
        require_corpus(args.output)
        print("Building vector index ...")
        build_rag_index(
            args.output,
            index_dir,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            force=args.force_index,
        )


if __name__ == "__main__":
    main()

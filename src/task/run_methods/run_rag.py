import argparse
import sys
from pathlib import Path

_ETD_ROOT = Path(__file__).resolve().parents[3]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

from src.task.build_rag.corpus import require_corpus
from src.task.build_rag.paths import default_bge_model_dir, default_corpus_path, default_index_dir
from src.task.build_rag.retriever import BGERetriever
from src.task.run_methods.rag import RagVariableExtractor
from src.task.run_methods.runner import run_queries

DATASET_FILENAME = "pico_sections_icd11.json"
CORPUS_PATH = default_corpus_path(_ETD_ROOT)
INDEX_DIR = default_index_dir(_ETD_ROOT)
BGE_MODEL_DIR = default_bge_model_dir(_ETD_ROOT)


def main(
    model_name: str,
    workers: int = 1,
    *,
    top_k: int = 5,
    exclude_self_pico: bool = False,
    limit_picos: int | None = None,
) -> None:
    require_corpus(CORPUS_PATH)
    retriever = BGERetriever.load(
        CORPUS_PATH,
        INDEX_DIR,
        model_dir=BGE_MODEL_DIR,
    )
    extractor = RagVariableExtractor(
        retriever,
        top_k=top_k,
        exclude_self_pico=exclude_self_pico,
    )

    run_queries(
        etd_root=_ETD_ROOT,
        dataset_filename=DATASET_FILENAME,
        results_subdir="rag",
        output_suffix="_rag.json",
        model_name=model_name,
        extractor=extractor,
        workers=workers,
        desc_label="rag",
        limit_picos=limit_picos,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run RAG EtD queries. Requires pre-built corpus and index "
            "(src/task/build_rag/build_corpus.py, src/task/build_rag/build_index.py)."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for LLM calls and run_methods/rag/output/<model>/ output dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Thread count for parallel item processing (default: 1). Try 4-8.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence chunks to retrieve per subquestion (default: 5).",
    )
    parser.add_argument(
        "--limit-picos",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only consider the first N PICO items in the dataset (default: all). "
            "Use 1 for a single-PICO smoke test."
        ),
    )
    parser.add_argument(
        "--exclude-self-pico",
        action="store_true",
        help=(
            "Leave-one-PICO-out mode: exclude chunks whose source_file matches "
            "the current PICO file. Default OFF (evaluation mode): retrieve "
            "from the full corpus, including the gold PICO's own evidence."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.limit_picos is not None and args.limit_picos < 1:
        parser.error("--limit-picos must be >= 1")

    main(
        model_name=args.model,
        workers=args.workers,
        top_k=args.top_k,
        exclude_self_pico=args.exclude_self_pico,
        limit_picos=args.limit_picos,
    )

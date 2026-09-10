import argparse
import sys
from pathlib import Path

_ETD_ROOT = Path(__file__).resolve().parents[3]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

from src.task.run_methods.end2end import End2EndVariableExtractor
from src.task.run_methods.runner import run_queries

MODEL_NAME = "deepseek-v4-pro"


def main(workers: int = 1, *, limit_picos: int | None = None) -> None:
    run_queries(
        etd_root=_ETD_ROOT,
        dataset_filename="pico_sections_icd11.json",
        results_subdir="end2end",
        output_suffix="_e2e.json",
        model_name=MODEL_NAME,
        extractor=End2EndVariableExtractor(),
        workers=workers,
        desc_label="end2end",
        limit_picos=limit_picos,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end EtD judgement queries.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Thread count for parallel item processing (default: 1). Try 4-8.",
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
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit_picos is not None and args.limit_picos < 1:
        parser.error("--limit-picos must be >= 1")
    main(workers=args.workers, limit_picos=args.limit_picos)

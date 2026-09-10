import argparse
import sys
from pathlib import Path

_ETD_ROOT = Path(__file__).resolve().parents[3]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

from src.task.run_methods.runner import run_queries
from src.task.run_methods.with_evidence import WithEvidenceVariableExtractor

MODEL_NAME = "deepseek-v4-pro"


def main(workers: int = 1) -> None:
    run_queries(
        etd_root=_ETD_ROOT,
        dataset_filename="pico_sections_icd11.json",
        results_subdir="with_evidence",
        output_suffix="_with_evidence.json",
        model_name=MODEL_NAME,
        extractor=WithEvidenceVariableExtractor(),
        workers=workers,
        desc_label="with_evidence",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run with-evidence EtD judgement queries."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Thread count for parallel item processing (default: 1). Try 4-8.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    main(workers=args.workers)

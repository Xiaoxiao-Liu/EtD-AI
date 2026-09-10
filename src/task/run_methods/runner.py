import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

from src.common.agent import ChatBot
from src.common.io import read_etd_file, save_output
from src.common.parsing import extract_model_output
from src.task.run_methods.base import BaseVariableExtractor
from src.task.run_methods.judgement import judgement_compare


def process_item(
    item: dict,
    results_dir: Path,
    model_name: str,
    extractor: BaseVariableExtractor,
    output_suffix: str,
    etd_data: list,
) -> Tuple[str, Optional[str]]:
    """Process one PICO item. Returns (source_file, None) on success or skip."""
    source_stem = item["source_file"].split(".")[0]
    output_path = results_dir / f"{source_stem}{output_suffix}"

    if output_path.exists():
        return item["source_file"], "skip"

    agent = ChatBot(model_name)
    results_dict = {
        "PICO_QUESTION": item["pico"]["Question"],
        "subquestion": [],
        "source_file": source_stem,
    }

    for idx in range(len(item["sections"])):
        system_prompt, user_prompt, subquestion_dict = extractor.extract(
            etd_data, item, idx
        )
        response = agent.call(system_prompt, user_prompt)
        judgement, reason = extract_model_output(response)

        subquestion_dict["ai_judgement"] = judgement
        subquestion_dict["ai_reason"] = reason
        subquestion_dict["additional_consideration"] = item["sections"][idx][
            "additional_considerations"
        ]
        subquestion_dict["judgement_result"] = judgement_compare(
            judgement, subquestion_dict["gt_judgement"]
        )
        subquestion_dict["category"] = item["pico"]["category"]
        subquestion_dict["output"] = response
        results_dict["subquestion"].append(subquestion_dict)

    save_output(results_dict, output_path)
    return item["source_file"], None


def run_queries(
    *,
    etd_root: Path,
    dataset_filename: str,
    results_subdir: str,
    output_suffix: str,
    model_name: str,
    extractor: BaseVariableExtractor,
    workers: int = 1,
    desc_label: str,
    limit_picos: Optional[int] = None,
) -> None:
    data_path = etd_root / "dataset" / dataset_filename
    etd_data = read_etd_file(data_path)

    results_dir = etd_root / "dataset" / "run_methods" / results_subdir / "output" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    scan_items = (
        etd_data[:limit_picos] if limit_picos is not None else etd_data
    )
    pending = []
    for item in scan_items:
        source_stem = item["source_file"].split(".")[0]
        if not (results_dir / f"{source_stem}{output_suffix}").exists():
            pending.append(item)

    skipped_existing = len(scan_items) - len(pending)
    limit_note = (
        f", limit_picos={limit_picos}" if limit_picos is not None else ""
    )
    print(
        f"total={len(etd_data)}, scan={len(scan_items)}, "
        f"skipped(existing)={skipped_existing}, "
        f"pending={len(pending)}, workers={workers}{limit_note}"
    )

    if not pending:
        print("Nothing to do.")
        return

    disable_progress = not sys.stderr.isatty()
    desc = f"{desc_label} ({model_name})"

    if workers <= 1:
        for item in tqdm(
            pending,
            desc=desc,
            unit="item",
            disable=disable_progress,
        ):
            process_item(
                item, results_dir, model_name, extractor, output_suffix, etd_data
            )
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_item,
                item,
                results_dir,
                model_name,
                extractor,
                output_suffix,
                etd_data,
            )
            for item in pending
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(pending),
            desc=desc,
            unit="item",
            disable=disable_progress,
        ):
            future.result()

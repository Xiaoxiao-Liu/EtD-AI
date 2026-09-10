#!/usr/bin/env python3
"""Download model snapshots into EtD/model/.

Currently supported (use --model <name>):
    bge-m3    -> EtD/model/bge-m3         RAG embedding (BAAI/bge-m3)
    qwen3-8b  -> EtD/model/Qwen3-8B       policy backbone for SFT/DPO (Qwen/Qwen3-8B)

Note on Qwen3 naming: there is no ``Qwen/Qwen3-8B-Instruct`` repo on the Hub.
Qwen3 ships a single post-trained model at ``Qwen/Qwen3-8B`` that combines
instruction-following and thinking modes in one weight set. (The ``-Instruct``
suffix only exists for the multimodal ``Qwen/Qwen3-VL-8B-Instruct`` variants.)

Default model is ``bge-m3`` so existing scripts (``scripts/build_rag.sh``)
keep working unchanged.

Usage (from EtD/):

    # RAG embedding (default):
    python -m src.common.download_model

    # Policy backbone:
    python -m src.common.download_model --model qwen3-8b

    # Override repo / output:
    python -m src.common.download_model \\
        --repo-id Qwen/Qwen3-8B-Base \\
        --output-dir model/Qwen3-8B-Base

    # Re-download:
    python -m src.common.download_model --model qwen3-8b --force
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ETD_ROOT = Path(__file__).resolve().parents[2]
if str(_ETD_ROOT) not in sys.path:
    sys.path.insert(0, str(_ETD_ROOT))

from src.task.build_rag.paths import BGE_M3_DIR, BGE_M3_REPO_ID, require_bge_model_dir


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the registry.

    sentinel_files: any file in this list, present under target_dir, is
    treated as proof that the snapshot is already on disk (--force overrides).
    """
    name: str
    repo_id: str
    default_dir: Path
    sentinel_files: tuple[str, ...]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "bge-m3": ModelSpec(
        name="bge-m3",
        repo_id=BGE_M3_REPO_ID,
        default_dir=BGE_M3_DIR,
        sentinel_files=("config.json",),
    ),
    "qwen3-8b": ModelSpec(
        name="qwen3-8b",
        repo_id="Qwen/Qwen3-8B",
        default_dir=_ETD_ROOT / "model" / "Qwen3-8B",
        sentinel_files=("config.json", "tokenizer.json"),
    ),
}


# --------------------------------------------------------------------------- #
# Download core
# --------------------------------------------------------------------------- #


def _snapshot_exists(target_dir: Path, sentinel_files: tuple[str, ...]) -> bool:
    """True iff any sentinel file is present in target_dir."""
    return any((target_dir / name).is_file() for name in sentinel_files)


def download_snapshot(
    repo_id: str,
    target_dir: Path,
    *,
    sentinel_files: tuple[str, ...] = (),
    force: bool = False,
    allow_patterns: Optional[list[str]] = None,
) -> Path:
    """Download a HF snapshot into ``target_dir``.

    - Skips if a sentinel file is already present (unless ``force``).
    - ``allow_patterns`` lets callers narrow what to fetch (e.g. skip onnx/.bin
      shards). ``None`` = fetch everything.
    """
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not force and sentinel_files and _snapshot_exists(target_dir, sentinel_files):
        print(f"snapshot already present at {target_dir}, skip (use --force to re-download)")
        return target_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    print(f"downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        allow_patterns=allow_patterns,
    )
    print(f"done. saved to {target_dir}")
    return target_dir


# --------------------------------------------------------------------------- #
# Per-model conveniences (used by main and importable from other code)
# --------------------------------------------------------------------------- #


def download_bge_m3(
    target_dir: Path = BGE_M3_DIR,
    *,
    repo_id: str = BGE_M3_REPO_ID,
    force: bool = False,
) -> Path:
    spec = MODEL_REGISTRY["bge-m3"]
    out = download_snapshot(
        repo_id=repo_id,
        target_dir=target_dir,
        sentinel_files=spec.sentinel_files,
        force=force,
    )
    require_bge_model_dir(out)
    return out


def download_qwen3_8b(
    target_dir: Optional[Path] = None,
    *,
    repo_id: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download Qwen3-8B (post-trained, thinking+instruct) into EtD/model/Qwen3-8B."""
    spec = MODEL_REGISTRY["qwen3-8b"]
    return download_snapshot(
        repo_id=repo_id or spec.repo_id,
        target_dir=target_dir or spec.default_dir,
        sentinel_files=spec.sentinel_files,
        force=force,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="bge-m3",
        help="Model preset to download (default: bge-m3).",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override the registry repo_id (e.g. Qwen/Qwen3-8B-Base).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the target directory (default: per-model registry entry).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a sentinel file already exists.",
    )
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.model]
    repo_id = args.repo_id or spec.repo_id
    target_dir = args.output_dir or spec.default_dir

    out = download_snapshot(
        repo_id=repo_id,
        target_dir=target_dir,
        sentinel_files=spec.sentinel_files,
        force=args.force,
    )

    # BGE-M3 has a stricter "is it usable?" check that other modules rely on;
    # apply only when we know that's the model we just fetched.
    if args.model == "bge-m3" and repo_id == BGE_M3_REPO_ID:
        require_bge_model_dir(out)


if __name__ == "__main__":
    main()

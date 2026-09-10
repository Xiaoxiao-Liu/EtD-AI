"""Shared logging / Hub helpers used by both SFT and DPO entry points.

Two utilities live here so that the SFT and DPO trainers stay focused on
training logic:

1. ``configure_wandb()`` — read environment variables to decide whether to
   enable Weights & Biases logging and, if so, derive a stable ``run_name``
   and ``project``. Returns ``(report_to, run_name)`` to feed straight into
   ``transformers.TrainingArguments``.

2. ``push_folder_to_hub()`` — upload an adapter directory (after
   ``trainer.save_model``) to the Hugging Face Hub. Used by both SFT and DPO
   when ``--push-to-hub`` (or ``HF_REPO_ID``) is set.

Both functions are best-effort: they never raise into the training loop. If
``wandb`` / ``huggingface_hub`` aren't installed or login is missing, we log
a warning and continue — failed Hub push must not lose a finished checkpoint.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Weights & Biases
# --------------------------------------------------------------------------- #


def _wandb_enabled() -> bool:
    """Disabled when ``WANDB_DISABLED`` is set to a truthy value."""
    raw = os.environ.get("WANDB_DISABLED", "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def configure_wandb(
    *,
    output_dir: Path,
    stage: str,
    default_project: str = "etd-ai",
) -> tuple[str, Optional[str]]:
    """Decide whether to log to wandb and derive a ``(report_to, run_name)``.

    Side effect: sets ``WANDB_PROJECT`` for the trainer if not already set.

    Args:
        output_dir: model output dir; the basename is folded into ``run_name``.
        stage: ``"sft"`` or ``"dpo_sufficiency"`` — folded into ``run_name``.
        default_project: used when ``WANDB_PROJECT`` is not in the env.

    Returns:
        ``(report_to, run_name)`` — ``report_to`` is ``"wandb"`` or ``"none"``;
        ``run_name`` is ``None`` when disabled.
    """
    if not _wandb_enabled():
        return "none", None

    project = os.environ.get("WANDB_PROJECT", default_project)
    os.environ.setdefault("WANDB_PROJECT", project)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = os.environ.get("WANDB_RUN_NAME") or f"{stage}-{output_dir.name}-{ts}"
    return "wandb", run_name


# --------------------------------------------------------------------------- #
# Hugging Face Hub
# --------------------------------------------------------------------------- #


def push_folder_to_hub(
    folder: Path,
    *,
    repo_id: str,
    private: bool = False,
    commit_message: str = "upload adapter",
    repo_type: str = "model",
) -> None:
    """Upload ``folder`` (e.g. an adapter dir) to ``repo_id`` on the HF Hub.

    Requires the user to have run ``huggingface-cli login`` (or set
    ``HF_TOKEN`` in the env). This helper is best-effort: any failure is
    printed but never re-raised — the local checkpoint is still safe on disk.
    """
    if not repo_id:
        return
    if not folder.exists():
        print(f"WARNING: push_folder_to_hub: folder not found: {folder}")
        return
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print(
            "WARNING: huggingface_hub not installed; skipping push.\n"
            "        -> pip install 'huggingface_hub>=0.25'"
        )
        return

    try:
        create_repo(
            repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True
        )
        api = HfApi()
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        print(f"pushed {folder} -> https://huggingface.co/{repo_id}")
    except Exception as e:
        print(
            f"WARNING: failed to push {folder} to {repo_id}: {e}\n"
            "        local checkpoint is intact; you can retry with\n"
            f"        huggingface-cli upload {repo_id} {folder} ."
        )

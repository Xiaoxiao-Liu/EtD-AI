"""Central path constants for the training subsystem.

Mirrors the pattern in ``src/task/build_rag/paths.py`` so other modules
import a single source of truth instead of duplicating ``Path(__file__)...``
expressions.

Layout::

    EtD/
        model/
            Qwen3-8B/                                <- base (downloaded)
            policy/
                sft/                                 <- LoRA adapter from SFT
                dpo_routing/                         <- Router LoRA adapter from DPO
                dpo_sufficiency/                     <- LoRA adapter from DPO
                dpo_judgment/                        <- Verifier LoRA adapter from DPO
                rl_only_routing/                     <- Router LoRA, base-init DPO
                rl_only_sufficiency/                 <- Gatekeeper LoRA, base-init DPO
                rl_only_judgment/                    <- Verifier LoRA, base-init DPO
        dataset/
            score_rubric/label/output/all_models.labeled.jsonl
            train/
                sft/
                    pico_splits.json                 <- PICO -> split (shared)
                    {routing,sufficiency,judgment}/output/{train,val}.jsonl
                dpo/
                    routing/output/{train,val}.jsonl
                    sufficiency/output/{train,val}.jsonl
                    judgment/output/{train,val}.jsonl
                rl_only/
                    {routing,sufficiency,judgment}/output/{train,val}.jsonl
"""

from __future__ import annotations

from pathlib import Path

ETD_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #

MODEL_DIR = ETD_ROOT / "model"
BASE_MODEL_DIR = MODEL_DIR / "Qwen3-8B"

# --------------------------------------------------------------------------- #
# Adapter outputs
# --------------------------------------------------------------------------- #

POLICY_DIR = MODEL_DIR / "policy"
SFT_ADAPTER_DIR = POLICY_DIR / "sft"
DPO_ROUTING_ADAPTER_DIR = POLICY_DIR / "dpo_routing"
DPO_SUFFICIENCY_ADAPTER_DIR = POLICY_DIR / "dpo_sufficiency"
DPO_JUDGMENT_ADAPTER_DIR = POLICY_DIR / "dpo_judgment"
RL_ONLY_ROUTING_ADAPTER_DIR = POLICY_DIR / "rl_only_routing"
RL_ONLY_SUFFICIENCY_ADAPTER_DIR = POLICY_DIR / "rl_only_sufficiency"
RL_ONLY_JUDGMENT_ADAPTER_DIR = POLICY_DIR / "rl_only_judgment"

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

DATASET_DIR = ETD_ROOT / "dataset"

LABELED_JSONL = (
    DATASET_DIR / "score_rubric" / "label" / "output" / "all_models.labeled.jsonl"
)

SFT_DATA_ROOT = DATASET_DIR / "train" / "sft"
PICO_SPLITS_FILE = SFT_DATA_ROOT / "pico_splits.json"

DPO_DATA_ROOT = DATASET_DIR / "train" / "dpo"
DPO_ROUTING_DATA_DIR = DPO_DATA_ROOT / "routing" / "output"
DPO_SUFFICIENCY_DATA_DIR = DPO_DATA_ROOT / "sufficiency" / "output"
DPO_JUDGMENT_DATA_DIR = DPO_DATA_ROOT / "judgment" / "output"

RL_ONLY_DATA_ROOT = DATASET_DIR / "train" / "rl_only"
RL_ONLY_ROUTING_DATA_DIR = RL_ONLY_DATA_ROOT / "routing" / "output"
RL_ONLY_SUFFICIENCY_DATA_DIR = RL_ONLY_DATA_ROOT / "sufficiency" / "output"
RL_ONLY_JUDGMENT_DATA_DIR = RL_ONLY_DATA_ROOT / "judgment" / "output"

SFT_TASKS: tuple[str, ...] = ("routing", "sufficiency", "judgment")


def sft_task_split_path(task: str, split: str) -> Path:
    """Resolve ``dataset/train/sft/<task>/output/<split>.jsonl``."""
    if task not in SFT_TASKS:
        raise ValueError(f"unknown task: {task!r} (expected one of {SFT_TASKS})")
    if split not in ("train", "val"):
        raise ValueError(f"unknown split: {split!r}")
    return SFT_DATA_ROOT / task / "output" / f"{split}.jsonl"

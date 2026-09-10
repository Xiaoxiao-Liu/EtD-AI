"""Thin wrapper. The EtD policy is co-trained across [ROUTE]/[SUFFICE]/[JUDGE]
on a single shared LoRA adapter — running this entry is identical to running
``python -m src.task.train.sft.train``.

The historic per-task directory layout is preserved only because the existing
shell wrappers (``scripts/train/sft/train_sufficiency.sh`` etc.) point here.
"""

from src.task.train.sft.train import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import subprocess
import argparse
from experiments.section2.data_augmentation import (
    BEST_MODEL,
    EPOCHS,
    pretty_print_command,
)
from experiments.section3.lr_exploration import BEST_AUG
from experiments.section3.batch_size_exploration import BEST_LR

BEST_BATCH_SIZE = 64
OPTIMIZER = "sgd"

BASE_PARAMS = [
    "-s", "veri",
    "-t", "veri",
    "-a", BEST_MODEL,
    "--root", "src/datasets",
    "--height", "224",
    "--width", "224",
    "--optim", OPTIMIZER,
    "--lr", str(BEST_LR),
    "--max-epoch", str(EPOCHS),
    "--stepsize", "20","40",
    "--train-batch-size", str(BEST_BATCH_SIZE),
    "--test-batch-size", "100",
]


def run_sgd_experiment(dry_run=True):
    save_dir = f"logs/{BEST_MODEL}_{BEST_AUG}_{BEST_LR:.0e}_{BEST_BATCH_SIZE}_{OPTIMIZER}_{EPOCHS}"
    params = BASE_PARAMS + ["--save-dir", save_dir]

    if "erase" in BEST_AUG:
        params.append("--random-erase")
    if "jitter" in BEST_AUG:
        params.append("--color-jitter")
    if "color" in BEST_AUG:
        params.append("--color-aug")

    cmd = ["python", "main.py"] + params
    pretty_print_command(cmd, OPTIMIZER, dry_run=dry_run)

    if not dry_run:
        print(f"Running SGD experiment...")
        subprocess.run(cmd, check=True)
        print("Experiment completed!")

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SGD optimizer experiment")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing"
    )
    args = parser.parse_args()
    run_sgd_experiment(dry_run=args.dry_run)

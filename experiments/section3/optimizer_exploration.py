#!/usr/bin/env python3
import subprocess
import argparse
from experiments.section2.data_augmentation import (
    BEST_MODEL,
    EPOCHS,
    pretty_print_command,
    SPACING,
)
from experiments.section3.lr_exploration import BEST_AUG
from experiments.section3.batch_size_exploration import BEST_LR

BEST_BATCH_SIZE = 64
OPTIMIZERS = ["adam", "amsgrad", "sgd", "rmsprop"]

BASE_PARAMS = [
    "-s", "veri",
    "-t", "veri",
    "-a", BEST_MODEL,
    "--root", "src/datasets",
    "--height", "224",
    "--width", "224",
    "--lr", str(BEST_LR),
    "--max-epoch", str(EPOCHS),
    "--stepsize", "20", "40",
    "--train-batch-size", str(BEST_BATCH_SIZE),
    "--test-batch-size", "100",
]


def run_experiment(optim, dry_run=True):
    save_dir = f"logs/{BEST_MODEL}_{BEST_AUG}_{BEST_LR:.0e}_{optim}_{EPOCHS}"
    print(save_dir)
    cmd = (
        ["python", "main.py"] + BASE_PARAMS + ["--optim", optim, "--save-dir", save_dir]
    )
    params = BASE_PARAMS + ["--save-dir", save_dir]

    if "erase" in BEST_AUG:
        params.append("--random-erase")
    if "jitter" in BEST_AUG:
        params.append("--color-jitter")
    if "color" in BEST_AUG:
        params.append("--color-aug")

    pretty_print_command(cmd, optim, dry_run=dry_run)
    if not dry_run:
        subprocess.run(cmd, check=True)
    return save_dir


def run_optimizer_experimentss(dry_run=True):
    print("=" * SPACING)
    print(f"RUNNING {len(OPTIMIZERS)} LEARNING RATE EXPERIMENTS")
    print("=" * SPACING)
    for i, lr in enumerate(OPTIMIZERS):
        print(f"{i+1}. {(lr)}")
    print("=" * SPACING + "\n")
    if (
        not dry_run
        and input("Execute all experiments? This will take a while ;) (y/n): ").lower()
        != "y"
    ):
        print("Aborted.")
        return

    for optim in OPTIMIZERS:
        run_experiment(optim, dry_run=dry_run)

    print("This was a DRY RUN." if dry_run else "All experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimizer experiment")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing"
    )
    args = parser.parse_args()
    run_optimizer_experimentss(dry_run=args.dry_run)

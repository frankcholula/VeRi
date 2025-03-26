#!/usr/bin/env python3
import subprocess
from experiments.section2.data_augmentation import BEST_MODEL, EPOCHS, SPACING, pretty_print_command
from experiments.section3.lr_exploration import BEST_AUG

BEST_AUG = "base"
BEST_LR = 0.0001
BATCH_SIZES = [16, 32, 64, 128, 256]


BASE_PARAMS = [
    "-s","veri",
    "-t", "veri",
    "-a", BEST_MODEL,
    "--root", "src/datasets",
    "--height", "224",
    "--width", "224",
    "--optim", "amsgrad",
    "--lr", str(BEST_LR),
    "--max-epoch",str(EPOCHS),
    "--stepsize", "20", "40",
    "--test-batch-size", "100",
]


def run_experiment(batch_size, dry_run=True):
    save_dir = f"logs/{BEST_MODEL}_{BEST_AUG}_{BEST_LR:.0e}_{batch_size}_{EPOCHS}"
    print(save_dir)
    cmd = ["python", "main.py"] + BASE_PARAMS +  ["--test-batch-size", str(batch_size) ,"--save-dir", save_dir]
    pretty_print_command(cmd, batch_size, dry_run=dry_run)
    if not dry_run:
        subprocess.run(cmd, check=True)
    return save_dir


def run_batch_size_experiments(dry_run=True):
    print("=" * SPACING)
    print(f"RUNNING {len(BATCH_SIZES)} BATCH SIZE EXPERIMENTS")
    print("=" * SPACING)
    for i, batch_size in enumerate(BATCH_SIZES):
        print(f"{i+1}. {(batch_size)}")
    print("=" * SPACING + "\n")

    # execuet experiments
    if (
        not dry_run
        and input("Execute all experiments? This will take a while ;) (y/n): ").lower()
        != "y"
    ):
        print("Aborted.")
        return

    for batch_size in BATCH_SIZES:
        run_experiment(batch_size, dry_run=dry_run)

    print("This was a DRY RUN." if dry_run else "All experiments completed!")


if __name__ == "__main__":
    run_batch_size_experiments(dry_run=True)

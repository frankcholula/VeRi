#!/usr/bin/env python3
import subprocess
from experiments.section2.data_augmentation import BEST_MODEL, SPACING, EPOCHS, pretty_print_command

BEST_AUG = "base"
LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01]

BASE_PARAMS = [
    "-s","veri",
    "-t", "veri",
    "-a", BEST_MODEL,
    "--root", "src/datasets",
    "--height", "224",
    "--width", "224",
    "--optim", "amsgrad",
    "--max-epoch",str(EPOCHS),
    "--stepsize", "20", "40",
    "--train-batch-size", "64",
    "--test-batch-size", "100",
]

def run_experiment(lr, dry_run=True):
    save_dir = f"logs/{BEST_MODEL}_{BEST_AUG}_{lr:.0e}_{EPOCHS}"
    print(save_dir)
    cmd = ["python", "main.py"] + BASE_PARAMS + ["--lr", str(lr), "--save-dir", save_dir]
    pretty_print_command(cmd, lr, dry_run=dry_run)
    if not dry_run:
        subprocess.run(cmd, check=True)
    return save_dir


def run_lr_experiments(dry_run=True):
    print("=" * SPACING)
    print(f"RUNNING {len(LRS)} LEARNING RATE EXPERIMENTS")
    print("=" * SPACING)
    for i, lr in enumerate(LRS):
        print(f"{i+1}. {(lr)}")
    print("=" * SPACING + "\n")

    # execuet experiments
    if (
        not dry_run
        and input("Execute all experiments? This will take a while ;) (y/n): ").lower()
        != "y"
    ):
        print("Aborted.")
        return

    for lr in LRS:
        run_experiment(lr, dry_run=dry_run)

    print("This was a DRY RUN." if dry_run else "All experiments completed!")


if __name__ == "__main__":
    run_lr_experiments(dry_run=True)

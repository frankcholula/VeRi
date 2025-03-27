#!/usr/bin/env python3
import subprocess
import itertools
import argparse

BEST_MODEL = "resnet50_fc512"
EPOCHS = 10
BASE_PARAMS = [
    "-s","veri",
    "-t", "veri",
    "-a", BEST_MODEL,
    "--root", "src/datasets",
    "--height", "224",
    "--width", "224",
    "--optim", "amsgrad",
    "--lr", "0.0003",
    "--max-epoch", str(EPOCHS),
    "--stepsize", "20", "40",
    "--train-batch-size", "64",
    "--test-batch-size", "100",
]

AUG_TO_FLAG = {
    "erase": "--random-erase",
    "jitter": "--color-jitter",
    "color": "--color-aug",
}

SPACING = 60


def get_aug_name(augmentations):
    if augmentations:
        return "-".join(sorted(augmentations))
    else:
        return "base"


def pretty_print_command(cmd, experiment_name, dry_run=True):
    """Pretty print for better readability."""
    parts = []
    i = 2  # Skip 'python' and 'main.py'
    while i < len(cmd):
        if cmd[i].startswith("-"):
            current_part = cmd[i]
            i += 1
            # Collect all values for this parameter
            while i < len(cmd) and not cmd[i].startswith("-"):
                current_part += f" {cmd[i]}"
                i += 1
            parts.append(current_part)
        else:
            parts.append(cmd[i])
            i += 1

    formatted = (
        "python main.py \\\n" + "\n".join(f"    {part} \\" for part in parts)[:-2]
    )

    prefix = f"\n[DRY RUN] " if dry_run else "\n"
    print(f"{prefix}Command for `{experiment_name}`:")
    print("-" * SPACING)
    print(f"{formatted}")
    print("-" * SPACING)


def run_experiment(experiment_name, dry_run=True):
    aug_name = get_aug_name(experiment_name)
    aug_flags = [AUG_TO_FLAG[aug] for aug in experiment_name]
    save_dir = f"logs/{BEST_MODEL}_{aug_name}_10"
    cmd = ["python", "main.py"] + BASE_PARAMS + aug_flags + ["--save-dir", save_dir]
    pretty_print_command(cmd, aug_name, dry_run)

    if not dry_run:
        subprocess.run(cmd, check=True)


def run_aug_experiments(dry_run=True):
    # Generate all combinations
    all_experiments = [
        list(combo)
        for r in range(1, len(AUG_TO_FLAG) + 1)
        for combo in itertools.combinations(AUG_TO_FLAG.keys(), r)
    ]

    # Print experiment summary
    print("=" * SPACING)
    print(f"RUNNING {len(all_experiments)} AUGMENTATION EXPERIMENTS")
    print("=" * SPACING)
    for i, augs in enumerate(all_experiments):
        print(f"{i+1}. {get_aug_name(augs)}")
    print("=" * SPACING + "\n")

    # execuet experiments
    if (
        not dry_run
        and input("Execute all experiments? This will take a while ;) (y/n): ").lower()
        != "y"
    ):
        print("Aborted.")
        return

    for experiment in all_experiments:
        run_experiment(experiment, dry_run=dry_run)

    print("This was a DRY RUN." if dry_run else "All experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data augmentation experiments")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing"
    )
    args = parser.parse_args()
    run_aug_experiments(dry_run=args.dry_run)

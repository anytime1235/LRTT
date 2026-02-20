#!/usr/bin/env python3
"""Migrate existing trials from old sweep to new study with SQLite storage."""

import os
import yaml
import csv
import optuna


def parse_trial_results(trial_dir):
    """Parse trial hyperparameters and results."""
    args_file = os.path.join(trial_dir, "args.yaml")
    summary_file = os.path.join(trial_dir, "summary.csv")

    if not os.path.exists(args_file):
        return None

    # Parse hyperparameters
    with open(args_file, 'r') as f:
        args = yaml.safe_load(f)

    params = {
        "lr": float(args.get("lr", 0)),
        "transfer_lr": float(args.get("transfer_lr", 0)),
        "transfer_every": int(args.get("transfer_every", 0)),
        "fast_lr": float(args.get("fast_lr", 0)),
    }

    # Parse best accuracy
    best_acc = 0.0
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "eval_top1" in row:
                        acc = float(row["eval_top1"])
                        best_acc = max(best_acc, acc)
        except Exception as e:
            print(f"Warning: Could not parse {summary_file}: {e}")

    return params, best_acc


def main():
    old_output_dir = "/root/results/sweep_tikitaka"
    new_output_dir = "/root/results/sweep_tikitaka_earlystop"
    storage = f"sqlite:///{new_output_dir}/optuna_study.db"
    study_name = "mdmlp_tikitaka_cifar10_earlystop"

    os.makedirs(new_output_dir, exist_ok=True)

    print(f"Migrating trials from {old_output_dir}")
    print(f"to study: {study_name}")
    print(f"storage: {storage}\n")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # Find and migrate trials
    migrated_count = 0
    for trial_num in range(100):  # Check up to 100 trials
        trial_dir = os.path.join(old_output_dir, f"trial_{trial_num:04d}")

        if not os.path.exists(trial_dir):
            continue

        result = parse_trial_results(trial_dir)
        if result is None:
            continue

        params, best_acc = result

        # Add trial to study
        trial = study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions={
                    "lr": optuna.distributions.FloatDistribution(1e-3, 0.1, log=True),
                    "transfer_lr": optuna.distributions.FloatDistribution(0.01, 10.0, log=True),
                    "transfer_every": optuna.distributions.CategoricalDistribution([1, 10, 20, 100]),
                    "fast_lr": optuna.distributions.FloatDistribution(0.01, 10.0, log=True),
                },
                values=[best_acc],
            )
        )

        print(f"✅ Migrated Trial {trial_num}: acc={best_acc:.2f}%, params={params}")
        migrated_count += 1

    print(f"\n{'='*60}")
    print(f"Migration complete: {migrated_count} trials migrated")
    if study.best_trial:
        print(f"Best trial so far: {study.best_trial.number}")
        print(f"Best accuracy: {study.best_value:.2f}%")
        print(f"Best params: {study.best_params}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

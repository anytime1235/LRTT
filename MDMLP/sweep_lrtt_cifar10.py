#!/usr/bin/env python3
"""Optuna-based hyperparameter sweep for MDMLP + LRTT on CIFAR-10.

Searches over: lr, transfer_lr, transfer_every, rank(1-3).
Each trial runs train_analog.py for a configured number of epochs and
reports the best validation accuracy to Optuna.

Usage:
    python sweep_lrtt_cifar10.py --n-trials 50 --epochs 50
    python sweep_lrtt_cifar10.py --n-trials 100 --epochs 200 --study-name full_sweep
"""

import argparse
import os
import subprocess
import sys
import json
import csv
from datetime import datetime

import optuna
from optuna.samplers import TPESampler


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna sweep for MDMLP + LRTT")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--data-dir", type=str, default="/data/cifar10", help="Path to CIFAR-10 data")
    parser.add_argument("--model", type=str, default="mdmlp_patch4_lap2_dim64_depth8_32",
                        help="Model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--study-name", type=str, default="mdmlp_lrtt_cifar10",
                        help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///sweep.db). None=in-memory")
    parser.add_argument("--output-dir", type=str, default="./output/sweep",
                        help="Base output directory for trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def objective(trial, args):
    """Optuna objective: run one MDMLP+LRTT training and return best val accuracy."""

    # === Search space ===
    lr = trial.suggest_float("lr", 1e-3, 0.1, log=True)
    transfer_lr = trial.suggest_float("transfer_lr", 1e-3, 10.0, log=True)
    transfer_every = trial.suggest_categorical("transfer_every", [1, 10, 100, 1000])

    # Trial output directory (train_analog.py creates it via get_outdir)
    exp_name = f"trial_{trial.number:04d}"
    exp_dir = os.path.join(args.output_dir, exp_name)

    # Build command
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train_analog.py"),
        args.data_dir,
        "-c", os.path.join(os.path.dirname(__file__), "ymls", "cifar10_analog.yml"),
        "--model", args.model,
        "--analog",
        "--epochs", str(args.epochs),
        "--lr", str(lr),
        "--lrtt-rank", "4",
        "--transfer-every", str(transfer_every),
        "--transfer-lr", str(transfer_lr),
        "--warmup-epochs", "0",
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--validate-c-only",
        "--output", args.output_dir,
        "--experiment", exp_name,
    ]

    # Run training subprocess
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: lr={lr:.5f}, t_lr={transfer_lr:.4f}, "
          f"te={transfer_every}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per trial
        )
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out!")
        return 0.0

    # Parse best accuracy from output
    best_acc = 0.0
    for line in result.stdout.split("\n"):
        if "*** Best metric:" in line:
            try:
                best_acc = float(line.split("*** Best metric:")[1].split("(")[0].strip())
            except (ValueError, IndexError):
                pass

    # Also try parsing from summary.csv
    summary_path = os.path.join(exp_dir, "summary.csv")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "eval_top1" in row:
                        acc = float(row["eval_top1"])
                        best_acc = max(best_acc, acc)
        except Exception:
            pass

    # Save trial params + result
    trial_info = {
        "trial": trial.number,
        "lr": lr,
        "transfer_lr": transfer_lr,
        "transfer_every": transfer_every,
        "best_accuracy": best_acc,
    }
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "trial_params.json"), "w") as f:
        json.dump(trial_info, f, indent=2)

    print(f"\nTrial {trial.number} complete: best_acc={best_acc:.2f}%")

    # Report to Optuna for pruning
    trial.report(best_acc, step=args.epochs)

    return best_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Optuna study
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
    )

    # Print results
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.2f}%")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results summary
    results_path = os.path.join(args.output_dir, "sweep_results.json")
    results = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save aggregated CSV
    csv_path = os.path.join(args.output_dir, "all_trials.csv")
    param_keys = list(study.best_params.keys()) if study.best_trial else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial"] + param_keys + ["best_accuracy", "state"])
        for t in study.trials:
            row = [t.number] + [t.params.get(k, "") for k in param_keys] + [t.value, str(t.state)]
            writer.writerow(row)
    print(f"Aggregated CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

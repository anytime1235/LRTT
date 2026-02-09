#!/usr/bin/env python3
"""Optuna sweep: Digital MDMLP with only stem.Linear + head.Linear trainable.

All layers frozen except:
- stem.Linear (16 -> 64, 1,088 params)
- head.Linear (64 -> 10, 650 params)
Total trainable: 1,738 params out of 301,234 (0.6%)

Tests whether input projection + classifier can learn together.

Usage:
    python sweep_digital_stem_head.py --n-trials 30 --epochs 30
"""

import argparse
import os
import subprocess
import sys
import json
import csv

import optuna
from optuna.samplers import TPESampler


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep: digital, stem+head trainable")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--data-dir", type=str, default="/data/cifar10", help="Path to CIFAR-10")
    parser.add_argument("--model", type=str, default="mdmlp_patch4_lap2_dim64_depth8_32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--study-name", type=str, default="mdmlp_digital_stem_head")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/sweep_stem_head")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def objective(trial, args):
    """Train with only stem.Linear + head.Linear unfrozen."""

    lr = trial.suggest_float("lr", 1e-4, 1.0, log=True)

    # Trial output directory (train_analog.py creates it via get_outdir)
    exp_name = f"trial_{trial.number:04d}"
    exp_dir = os.path.join(args.output_dir, exp_name)

    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train_analog.py"),
        args.data_dir,
        "-c", os.path.join(os.path.dirname(__file__), "ymls", "cifar10_sgd.yml"),
        "--model", args.model,
        "--freeze-mode", "stem-head",
        "--epochs", str(args.epochs),
        "--lr", str(lr),
        "--warmup-epochs", "0",
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--output", args.output_dir,
        "--experiment", exp_name,
    ]

    print(f"\n{'='*60}")
    print(f"[stem+head] Trial {trial.number}: lr={lr:.5f}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out!")
        return 0.0

    best_acc = _parse_best_acc(result.stdout, exp_dir)

    trial_info = {
        "trial": trial.number, "mode": "stem-head",
        "lr": lr,
        "best_accuracy": best_acc,
    }
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "trial_params.json"), "w") as f:
        json.dump(trial_info, f, indent=2)

    print(f"\nTrial {trial.number} complete: best_acc={best_acc:.2f}%")
    trial.report(best_acc, step=args.epochs)
    return best_acc


def _parse_best_acc(stdout, exp_dir):
    best_acc = 0.0
    for line in stdout.split("\n"):
        if "*** Best metric:" in line:
            try:
                best_acc = float(line.split("*** Best metric:")[1].split("(")[0].strip())
            except (ValueError, IndexError):
                pass
    summary_path = os.path.join(exp_dir, "summary.csv")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                for row in csv.DictReader(f):
                    if "eval_top1" in row:
                        best_acc = max(best_acc, float(row["eval_top1"]))
        except Exception:
            pass
    return best_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        direction="maximize", sampler=sampler, load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n" + "=" * 80)
    print("STEM+HEAD SWEEP COMPLETE")
    print("=" * 80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.2f}%")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    results_path = os.path.join(args.output_dir, "sweep_results.json")
    results = {
        "mode": "stem-head",
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
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

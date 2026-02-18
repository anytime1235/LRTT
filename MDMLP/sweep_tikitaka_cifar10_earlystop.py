#!/usr/bin/env python3
"""Optuna-based hyperparameter sweep for MDMLP + TikiTaka v1 with early stopping.

Features:
- MedianPruner for pruning underperforming trials
- Real-time monitoring via Popen
- Early stopping within trials via --patience-epochs
- SQLite storage for persistence and smart continuation
- Avoids re-exploring previous trial regions

Usage:
    python sweep_tikitaka_cifar10_earlystop.py --n-trials 45 --epochs 30 --patience 4
    python sweep_tikitaka_cifar10_earlystop.py --n-trials 100 --epochs 50 --patience 5 --resume
"""

import argparse
import os
import subprocess
import sys
import json
import csv
import time
import re

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna sweep for MDMLP + TikiTaka v1 with early stopping")
    parser.add_argument("--n-trials", type=int, default=45, help="Number of NEW Optuna trials to run")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience in epochs")
    parser.add_argument("--data-dir", type=str, default="/data/cifar10", help="Path to CIFAR-10 data")
    parser.add_argument("--model", type=str, default="mdmlp_patch4_lap2_dim64_depth8_32",
                        help="Model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--study-name", type=str, default="mdmlp_tikitaka_cifar10_earlystop",
                        help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL. Default: sqlite:///output_dir/optuna_study.db")
    parser.add_argument("--output-dir", type=str, default="/root/results/sweep_tikitaka_earlystop",
                        help="Base output directory for trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing study (loads previous trials)")
    parser.add_argument("--pruner-warmup", type=int, default=5,
                        help="Number of epochs before pruner starts evaluating")
    parser.add_argument("--pruner-interval", type=int, default=1,
                        help="Interval (in epochs) for pruning checks")
    return parser.parse_args()


def objective(trial, args):
    """Optuna objective with real-time monitoring and pruning."""

    # === Search space ===
    lr = trial.suggest_float("lr", 1e-3, 0.1, log=True)
    transfer_lr = trial.suggest_float("transfer_lr", 0.01, 10.0, log=True)
    transfer_every = trial.suggest_categorical("transfer_every", [1, 10, 20, 100])
    fast_lr = trial.suggest_float("fast_lr", 0.01, 10.0, log=True)

    # Trial output directory
    exp_name = f"trial_{trial.number:04d}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train_analog.py"),
        args.data_dir,
        "-c", os.path.join(os.path.dirname(__file__), "ymls", "cifar10_tikitaka.yml"),
        "--model", args.model,
        "--analog",
        "--algo", "tikitaka",
        "--epochs", str(args.epochs),
        "--lr", str(lr),
        "--transfer-every", str(transfer_every),
        "--transfer-lr", str(transfer_lr),
        "--fast-lr", str(fast_lr),
        "--warmup-epochs", "0",
        "--patience-epochs", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--validate-c-only",
        "--output", args.output_dir,
        "--experiment", exp_name,
    ]

    print(f"\n{'='*80}")
    print(f"Trial {trial.number}: lr={lr:.5f}, t_lr={transfer_lr:.4f}, "
          f"te={transfer_every}, fast_lr={fast_lr:.4f}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Run with Popen for real-time monitoring
    summary_path = os.path.join(exp_dir, "summary.csv")
    best_acc = 0.0
    current_epoch = 0

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Monitor output in real-time
        log_file = os.path.join(exp_dir, "train.log")
        with open(log_file, "w") as f:
            for line in process.stdout:
                f.write(line)
                f.flush()

                # Parse epoch and accuracy from output
                # Example: "Train: 0  [  0/391]  Loss: 2.345  Acc@1: 12.50"
                # Example: "Test: [391/391]  Loss: 2.123  Acc@1: 25.30"

                if "Test:" in line and "Acc@1:" in line:
                    # Try to extract accuracy
                    match = re.search(r'Acc@1:\s+(\d+\.?\d*)', line)
                    if match:
                        acc = float(match.group(1))
                        best_acc = max(best_acc, acc)

                # Check summary.csv for intermediate results
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r") as csvf:
                            reader = csv.DictReader(csvf)
                            rows = list(reader)
                            if rows:
                                latest_row = rows[-1]
                                if "epoch" in latest_row:
                                    current_epoch = int(latest_row["epoch"]) + 1  # 0-indexed
                                if "eval_top1" in latest_row:
                                    acc = float(latest_row["eval_top1"])
                                    best_acc = max(best_acc, acc)

                                    # Report intermediate value for pruning
                                    trial.report(acc, step=current_epoch)

                                    # Check if should prune
                                    if trial.should_prune():
                                        print(f"\n⚠️  Trial {trial.number} pruned at epoch {current_epoch} (acc={acc:.2f}%)")
                                        process.terminate()
                                        process.wait(timeout=10)
                                        raise optuna.TrialPruned()
                    except (ValueError, KeyError, FileNotFoundError):
                        pass

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            print(f"⚠️  Trial {trial.number} failed with return code {return_code}")
            return 0.0

    except subprocess.TimeoutExpired:
        print(f"⚠️  Trial {trial.number} timed out!")
        process.kill()
        return 0.0

    except optuna.TrialPruned:
        raise  # Re-raise to let Optuna handle it

    except Exception as e:
        print(f"⚠️  Trial {trial.number} encountered error: {e}")
        return 0.0

    # Parse final best accuracy from summary.csv
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "eval_top1" in row:
                        acc = float(row["eval_top1"])
                        best_acc = max(best_acc, acc)
        except Exception as e:
            print(f"Warning: Could not parse summary.csv: {e}")

    # Save trial params + result
    trial_info = {
        "trial": trial.number,
        "algo": "tikitaka",
        "lr": lr,
        "transfer_lr": transfer_lr,
        "transfer_every": transfer_every,
        "fast_lr": fast_lr,
        "best_accuracy": best_acc,
        "final_epoch": current_epoch,
    }
    with open(os.path.join(exp_dir, "trial_params.json"), "w") as f:
        json.dump(trial_info, f, indent=2)

    print(f"\n✅ Trial {trial.number} complete: best_acc={best_acc:.2f}% (epoch {current_epoch})")

    return best_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup storage
    if args.storage is None:
        args.storage = f"sqlite:///{os.path.join(args.output_dir, 'optuna_study.db')}"

    print(f"\n{'='*80}")
    print(f"Starting Optuna sweep with early stopping")
    print(f"Storage: {args.storage}")
    print(f"Study name: {args.study_name}")
    print(f"Output dir: {args.output_dir}")
    print(f"Patience: {args.patience} epochs")
    print(f"{'='*80}\n")

    # Setup pruner and sampler
    pruner = MedianPruner(
        n_startup_trials=3,  # Don't prune first 3 trials
        n_warmup_steps=args.pruner_warmup,  # Wait for warmup epochs
        interval_steps=args.pruner_interval,
    )

    sampler = TPESampler(seed=args.seed)

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Print existing trials if resuming
    if len(study.trials) > 0:
        print(f"\n📊 Resuming study with {len(study.trials)} existing trials")
        print(f"Best so far: Trial {study.best_trial.number} with {study.best_value:.2f}%")
        print(f"Best params: {study.best_params}")
        print(f"\nWill run {args.n_trials} NEW trials\n")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 80)
    print("TIKITAKA v1 SWEEP WITH EARLY STOPPING COMPLETE")
    print("=" * 80)

    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    if study.best_trial:
        print(f"\n🏆 Best trial: {study.best_trial.number}")
        print(f"Best accuracy: {study.best_value:.2f}%")
        print(f"Best params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

    # Save results summary
    results_path = os.path.join(args.output_dir, "sweep_results.json")
    results = {
        "algo": "tikitaka",
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_accuracy": study.best_value if study.best_trial else 0.0,
        "best_params": study.best_params if study.best_trial else {},
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "datetime_start": str(t.datetime_start),
                "datetime_complete": str(t.datetime_complete),
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
            row = [t.number] + [t.params.get(k, "") for k in param_keys] + [t.value if t.value else 0.0, str(t.state)]
            writer.writerow(row)
    print(f"Aggregated CSV saved to: {csv_path}")

    print(f"\n💾 Study saved to: {args.storage}")
    print(f"To resume later, run with --resume flag\n")


if __name__ == "__main__":
    main()

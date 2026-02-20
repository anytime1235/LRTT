#!/usr/bin/env python3
"""Optuna sweep for transfer_lr only, with lr and te fixed.

Usage:
    python sweep_tlr_only.py --te 10 --n-trials 10 --epochs 10
    python sweep_tlr_only.py --te 1000 --n-trials 10 --epochs 10
    python sweep_tlr_only.py --te 10 --n-trials 10 --epochs 10 --resume
"""

import argparse
import os
import re
import subprocess
import sys
import json
import csv
import glob
import yaml

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.distributions import FloatDistribution


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep transfer_lr only")
    parser.add_argument("--te", type=int, required=True, help="transfer_every (fixed)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (fixed)")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default="/data/cifar10")
    parser.add_argument("--model", type=str, default="mdmlp_patch4_lap2_dim64_depth8_32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing trials in output dir")
    return parser.parse_args()


def _get_tlr_distribution(te):
    if te <= 10:
        return FloatDistribution(0.003, 0.04, log=True)
    elif te <= 100:
        return FloatDistribution(0.01, 0.5, log=True)
    else:
        return FloatDistribution(0.05, 2.0, log=True)


def _load_existing_trials(output_dir, te, min_epochs=2):
    """Scan existing trial directories and return list of frozen trials."""
    trials = []
    dist = _get_tlr_distribution(te)
    trial_dirs = sorted(glob.glob(os.path.join(output_dir, "trial_*")))

    for trial_dir in trial_dirs:
        args_path = os.path.join(trial_dir, "args.yaml")
        summary_path = os.path.join(trial_dir, "summary.csv")
        if not os.path.exists(args_path) or not os.path.exists(summary_path):
            continue

        with open(args_path, "r") as f:
            trial_args = yaml.safe_load(f)
        transfer_lr = trial_args.get("transfer_lr")
        if transfer_lr is None:
            continue

        # Parse summary.csv for best accuracy and epoch count
        best_acc = 0.0
        n_epochs = 0
        with open(summary_path, "r") as f:
            for row in csv.DictReader(f):
                if "eval_top1" in row:
                    try:
                        best_acc = max(best_acc, float(row["eval_top1"]))
                        n_epochs += 1
                    except (ValueError, TypeError):
                        continue  # skip duplicate header rows

        # Skip incomplete trials (e.g. interrupted at epoch 0)
        if n_epochs < min_epochs:
            trial_name = os.path.basename(trial_dir)
            print(f"  Skipping {trial_name}: only {n_epochs} epoch(s), will re-run")
            continue

        trial_name = os.path.basename(trial_dir)
        trial_num = int(trial_name.split("_")[1])

        frozen = optuna.trial.create_trial(
            params={"transfer_lr": transfer_lr},
            distributions={"transfer_lr": dist},
            values=[best_acc],
        )
        trials.append((trial_num, frozen, transfer_lr, best_acc))
        print(f"  Loaded {trial_name}: t_lr={transfer_lr:.6f}, acc={best_acc:.2f}%")

    return trials


def _parse_epoch_acc(line):
    m = re.search(r'Test:\s*\[\s*(\d+)/\d+\].*Acc@1:\s*[\d.]+\s*\(([\d.]+)\)', line)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def objective(trial, args):
    te = args.te

    # Transfer_lr search range depends on te
    if te <= 10:
        transfer_lr = trial.suggest_float("transfer_lr", 0.003, 0.04, log=True)
    elif te <= 100:
        transfer_lr = trial.suggest_float("transfer_lr", 0.01, 0.5, log=True)
    else:
        transfer_lr = trial.suggest_float("transfer_lr", 0.05, 2.0, log=True)

    exp_name = f"trial_{trial.number:04d}"
    output_dir = f"./output/sweep_tlr_te{te}"
    exp_dir = os.path.join(output_dir, exp_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable, os.path.join(script_dir, "train_analog.py"),
        args.data_dir,
        "-c", os.path.join(script_dir, "ymls", "cifar10_analog.yml"),
        "--model", args.model,
        "--analog",
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--lrtt-rank", "4",
        "--transfer-every", str(te),
        "--transfer-lr", str(transfer_lr),
        "--c-desired-bl", "31",
        "--sched", "none",
        "--warmup-epochs", "0",
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--validate-c-only",
        "--patience", str(args.patience),
        "--lrtt-all",
        "--output", output_dir,
        "--experiment", exp_name,
    ]

    steps_per_epoch = 391
    transfers_per_ep = steps_per_epoch // te

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: te={te}, lr={args.lr}, t_lr={transfer_lr:.6f}, "
          f"transfers/ep~{transfers_per_ep}")
    print(f"{'='*60}\n")

    os.makedirs(exp_dir, exist_ok=True)
    best_acc = 0.0
    last_epoch = -1

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line_s = line.rstrip()
            if any(kw in line_s for kw in ["Test:", "C-only", "*** Best", "Early stopping"]):
                print(f"  [T{trial.number}] {line_s}")

            parsed = _parse_epoch_acc(line_s)
            if parsed:
                epoch, acc = parsed
                if epoch > last_epoch:
                    last_epoch = epoch
                    best_acc = max(best_acc, acc)
                    trial.report(best_acc, step=epoch)
                    if trial.should_prune():
                        print(f"  [T{trial.number}] PRUNED at epoch {epoch} (acc={best_acc:.2f}%)")
                        proc.kill()
                        proc.wait()
                        raise optuna.TrialPruned()

        proc.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"Trial {trial.number} timed out!")
        return 0.0

    # Parse from summary.csv as fallback
    summary_path = os.path.join(exp_dir, "summary.csv")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                for row in csv.DictReader(f):
                    if "eval_top1" in row:
                        best_acc = max(best_acc, float(row["eval_top1"]))
        except Exception:
            pass

    trial_info = {
        "trial": trial.number,
        "lr": args.lr,
        "transfer_lr": transfer_lr,
        "transfer_every": te,
        "best_accuracy": best_acc,
    }
    with open(os.path.join(exp_dir, "trial_params.json"), "w") as f:
        json.dump(trial_info, f, indent=2)

    print(f"\nTrial {trial.number} complete: best_acc={best_acc:.2f}%")
    return best_acc


def main():
    args = parse_args()
    te = args.te
    output_dir = f"./output/sweep_tlr_te{te}"
    os.makedirs(output_dir, exist_ok=True)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=f"sweep_tlr_te{te}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    n_new_trials = args.n_trials
    if args.resume:
        print(f"\n{'='*60}")
        print(f"RESUMING: scanning existing trials in {output_dir}")
        print(f"{'='*60}")
        existing = _load_existing_trials(output_dir, te)
        for _, frozen_trial, _, _ in existing:
            study.add_trial(frozen_trial)
        n_loaded = len(existing)
        # Re-seed sampler to avoid replaying same suggestions
        sampler = TPESampler(seed=args.seed + n_loaded)
        study.sampler = sampler
        print(f"\nLoaded {n_loaded} existing trials into study (sampler re-seeded).")
        if n_loaded >= args.n_trials:
            n_new_trials = 0
            print("All requested trials already done. Set higher --n-trials to run more.")
        else:
            n_new_trials = args.n_trials - n_loaded
            print(f"Will run {n_new_trials} new trials (total target: {args.n_trials})")
        print()

    if n_new_trials > 0:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=n_new_trials,
        )

    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE (te={te})")
    print(f"{'='*80}")
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.2f}%")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    results = {
        "te": te,
        "lr": args.lr,
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
    with open(os.path.join(output_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {os.path.join(output_dir, 'sweep_results.json')}")

    csv_path = os.path.join(output_dir, "all_trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "transfer_lr", "best_accuracy", "state"])
        for t in study.trials:
            writer.writerow([t.number, t.params.get("transfer_lr", ""), t.value, str(t.state)])
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

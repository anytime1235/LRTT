#!/usr/bin/env python3
"""Optuna sweep for MDMLP + LRTT-ALL with hybrid reinit, 200ep cosine annealing.

Grid over transfer_every: [1, 10, 100, 1000]
Search lr and transfer_lr with Optuna per each te value.
reinit_mode=hybrid, cosine scheduler, early stopping.

Usage:
    python sweep_hybrid_200ep.py --n-trials 40 --epochs 200 --patience 20
"""

import argparse
import os
import re
import subprocess
import sys
import json
import csv
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna sweep: LRTT-ALL hybrid reinit, 200ep cosine")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Number of Optuna trials (total across all te values)")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per trial")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--data-dir", type=str, default="/data/cifar10")
    parser.add_argument("--model", type=str, default="mdmlp_patch4_lap2_dim64_depth8_32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="./output/sweep_hybrid_200ep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay-factor", type=float, default=1.0,
                        help="Decay factor for hybrid reinit (default: 1.0)")
    parser.add_argument("--timeout-per-trial", type=int, default=14400,
                        help="Timeout per trial in seconds (default: 4h)")
    return parser.parse_args()


def _parse_epoch_acc(line):
    """Parse accuracy from a Test: line. Returns (epoch, acc) or None."""
    m = re.search(r'Test:\s*\[\s*(\d+)/\d+\].*Acc@1:\s*[\d.]+\s*\(([\d.]+)\)', line)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def run_trial(trial, args, transfer_every):
    """Run one training trial with given transfer_every."""

    lr = trial.suggest_float("lr", 0.01, 0.2, log=True)
    transfer_lr = trial.suggest_float("transfer_lr", 0.001, 1.0, log=True)

    exp_name = f"te{transfer_every}_trial_{trial.number:04d}"
    exp_dir = os.path.join(args.output_dir, exp_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable, os.path.join(script_dir, "train_analog.py"),
        args.data_dir,
        "-c", os.path.join(script_dir, "ymls", "cifar10_analog.yml"),
        "--model", args.model,
        "--analog",
        "--epochs", str(args.epochs),
        "--lr", str(lr),
        "--lrtt-rank", "4",
        "--transfer-every", str(transfer_every),
        "--transfer-lr", str(transfer_lr),
        "--c-desired-bl", "31",
        "--sched", "cosine",
        "--warmup-epochs", "5",
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--validate-c-only",
        "--patience", str(args.patience),
        "--lrtt-all",
        "--reinit-mode", "hybrid",
        "--decay-factor", str(args.decay_factor),
        "--output", args.output_dir,
        "--experiment", exp_name,
    ]

    steps_per_epoch = 391
    transfers_per_ep = steps_per_epoch // transfer_every if transfer_every <= steps_per_epoch else f"1/{transfer_every // steps_per_epoch + 1}ep"

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} | te={transfer_every}, lr={lr:.5f}, t_lr={transfer_lr:.5f}, "
          f"transfers/ep~{transfers_per_ep}")
    print(f"reinit=hybrid, decay_factor={args.decay_factor}, sched=cosine, patience={args.patience}")
    print(f"{'='*70}\n")

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

        proc.wait(timeout=args.timeout_per_trial)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"Trial {trial.number} timed out!")

    # Fallback: parse from summary.csv
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
        "transfer_every": transfer_every,
        "lr": lr,
        "transfer_lr": transfer_lr,
        "reinit_mode": "hybrid",
        "decay_factor": args.decay_factor,
        "sched": "cosine",
        "epochs": args.epochs,
        "patience": args.patience,
        "best_accuracy": best_acc,
        "stopped_epoch": last_epoch,
    }
    with open(os.path.join(exp_dir, "trial_params.json"), "w") as f:
        json.dump(trial_info, f, indent=2)

    print(f"\nTrial {trial.number} (te={transfer_every}) complete: "
          f"best_acc={best_acc:.2f}% (stopped at epoch {last_epoch})")
    return best_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    te_values = [1, 10, 100, 1000]
    trials_per_te = args.n_trials // len(te_values)

    all_results = {}

    for te in te_values:
        print(f"\n{'#'*80}")
        print(f"# SWEEP: transfer_every = {te}  ({trials_per_te} trials)")
        print(f"{'#'*80}")

        study_name = f"hybrid_200ep_te{te}"
        sampler = TPESampler(seed=args.seed)
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=10)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            lambda trial: run_trial(trial, args, te),
            n_trials=trials_per_te,
        )

        print(f"\n--- te={te} Best: trial {study.best_trial.number}, "
              f"acc={study.best_value:.2f}%, params={study.best_params}")

        all_results[te] = {
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

    # Print final summary
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE - SUMMARY")
    print(f"{'='*80}")

    best_overall_acc = 0
    best_overall_te = None
    for te in te_values:
        r = all_results[te]
        print(f"  te={te:>4d}: best_acc={r['best_accuracy']:.2f}%  "
              f"lr={r['best_params']['lr']:.5f}  t_lr={r['best_params']['transfer_lr']:.5f}")
        if r['best_accuracy'] > best_overall_acc:
            best_overall_acc = r['best_accuracy']
            best_overall_te = te

    print(f"\nBest overall: te={best_overall_te}, acc={best_overall_acc:.2f}%")

    # Save results
    results_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "reinit_mode": "hybrid",
                "decay_factor": args.decay_factor,
                "sched": "cosine",
                "warmup_epochs": 5,
                "epochs": args.epochs,
                "patience": args.patience,
                "te_values": te_values,
                "trials_per_te": trials_per_te,
            },
            "best_overall": {
                "transfer_every": best_overall_te,
                "accuracy": best_overall_acc,
                "params": all_results[best_overall_te]["best_params"],
            },
            "per_te_results": {str(k): v for k, v in all_results.items()},
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()

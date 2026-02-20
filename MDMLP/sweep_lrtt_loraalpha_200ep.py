#!/usr/bin/env python3
"""Optuna sweep for MDMLP + LRTT with decay reinit, 200ep, lora_alpha grid.

Grid over transfer_every: [50, 100, 500, 1000] and lora_alpha: [0.1, 0.5, 1.0]
Search lr and transfer_lr with Optuna per each combination.
reinit_mode=decay, stem/head=FloatingPoint, cosine scheduler, early stopping.

Usage:
    python sweep_lrtt_loraalpha_200ep.py --n-trials 48 --epochs 200 --patience 10
"""

import argparse
import os
import re
import subprocess
import sys
import json
import csv

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def parse_args():
    parser = argparse.ArgumentParser(description="LRTT sweep: decay+lora_alpha grid")
    parser.add_argument("--n-trials", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="/data/cifar10")
    parser.add_argument("--model", type=str, default="mdmlp_patch4_lap2_dim64_depth8_32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="/root/results/sweep_lrtt_loraalpha_200ep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay-factor", type=float, default=1.0)
    parser.add_argument("--timeout-per-trial", type=int, default=14400)
    return parser.parse_args()


def _parse_epoch_acc(line):
    m = re.search(r'Test:\s*\[\s*(\d+)/\d+\].*Acc@1:\s*[\d.]+\s*\(([\d.]+)\)', line)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def run_trial(trial, args, transfer_every, lora_alpha):
    lr = trial.suggest_float("lr", 0.05, 0.2, log=True)
    transfer_lr = trial.suggest_float("transfer_lr", 0.001, 1.0, log=True)

    exp_name = f"te{transfer_every}_la{lora_alpha}_t{trial.number:04d}"
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
        "--lora-alpha", str(lora_alpha),
        "--c-desired-bl", "31",
        "--sched", "cosine",
        "--warmup-epochs", "2",
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--validate-c-only",
        "--patience", str(args.patience),
        "--reinit-mode", "decay",
        "--decay-factor", str(args.decay_factor),
        "--output", args.output_dir,
        "--experiment", exp_name,
    ]

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} | te={transfer_every}, lora_alpha={lora_alpha}, lr={lr:.5f}, t_lr={transfer_lr:.5f}")
    print(f"{'='*70}\n")

    os.makedirs(exp_dir, exist_ok=True)
    best_acc = 0.0
    last_epoch = -1

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line_s = line.rstrip()
            if any(kw in line_s for kw in ["Test:", "C-only", "*** Best", "Early"]):
                print(f"  [T{trial.number}] {line_s}")

            parsed = _parse_epoch_acc(line_s)
            if parsed:
                epoch, acc = parsed
                if epoch > last_epoch:
                    last_epoch = epoch
                    best_acc = max(best_acc, acc)
                    trial.report(best_acc, step=epoch)
                    if trial.should_prune():
                        print(f"  PRUNED at epoch {epoch}")
                        proc.kill()
                        proc.wait()
                        raise optuna.TrialPruned()

        proc.wait(timeout=args.timeout_per_trial)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

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
        "lora_alpha": lora_alpha,
        "lr": lr,
        "transfer_lr": transfer_lr,
        "best_accuracy": best_acc,
        "stopped_epoch": last_epoch,
    }
    with open(os.path.join(exp_dir, "trial_params.json"), "w") as f:
        json.dump(trial_info, f, indent=2)

    print(f"\nTrial {trial.number} complete: best_acc={best_acc:.2f}%")
    return best_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    te_values = [50, 100, 500, 1000]
    lora_alpha_values = [0.01, 0.1, 0.5, 1.0]

    # Skip already completed combinations
    completed = [("te50_la0.01", 64.95), ("te50_la0.1", 59.48)]
    completed_keys = [k for k, _ in completed]

    total_comb = len(te_values) * len(lora_alpha_values)
    trials_per_comb = max(4, args.n_trials // total_comb)

    print(f"\n{'='*80}")
    print(f"LRTT Sweep: te={te_values}, lora_alpha={lora_alpha_values}")
    print(f"Total combinations: {total_comb}, Trials per comb: {trials_per_comb}")
    print(f"Already completed: {len(completed_keys)}")
    print(f"Remaining: {total_comb - len(completed_keys)}")
    print(f"{'='*80}\n")

    all_results = {}

    # Load already completed results
    for key, acc in completed:
        all_results[key] = {"best_accuracy": acc, "status": "previously_completed"}
        print(f"✓ Skipping {key} (already completed: {acc:.2f}%)")

    for te in te_values:
        for lora_alpha in lora_alpha_values:
            comb_key = f"te{te}_la{lora_alpha}"

            # Skip if already completed
            if comb_key in completed_keys:
                continue

            print(f"\n{'#'*80}")
            print(f"# SWEEP: transfer_every={te}, lora_alpha={lora_alpha}")
            print(f"{'#'*80}")

            study = optuna.create_study(
                study_name=f"lrtt_{comb_key}",
                direction="maximize",
                sampler=TPESampler(seed=args.seed),
                pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=5),
            )

            study.optimize(
                lambda trial: run_trial(trial, args, te, lora_alpha),
                n_trials=trials_per_comb,
            )

            print(f"\n--- {comb_key} Best: {study.best_value:.2f}%, params={study.best_params}")

            all_results[comb_key] = {
                "transfer_every": te,
                "lora_alpha": lora_alpha,
                "best_accuracy": study.best_value,
                "best_params": study.best_params,
            }

    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")

    best_key = max(all_results.keys(), key=lambda k: all_results[k]['best_accuracy'])
    best_acc = all_results[best_key]['best_accuracy']

    for key in all_results.keys():
        r = all_results[key]
        print(f"  {key}: {r['best_accuracy']:.2f}%")

    print(f"\nBest: {best_key}, acc={best_acc:.2f}%")

    with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
        json.dump({"best": best_key, "best_accuracy": best_acc, "all": all_results}, f, indent=2)


if __name__ == "__main__":
    main()

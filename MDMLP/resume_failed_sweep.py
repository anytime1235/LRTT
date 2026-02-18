#!/usr/bin/env python3
"""Resume failed combinations from sweep_lrtt_loraalpha_200ep.py
Run sequentially to avoid memory competition issues.
"""

import argparse
import os
import re
import subprocess
import sys
import json
import csv
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Resume LRTT sweep for failed combinations")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="/data/cifar10")
    parser.add_argument("--model", type=str, default="mdmlp_patch4_lap2_dim64_depth8_32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="/root/results/sweep_lrtt_loraalpha_200ep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay-factor", type=float, default=1.0)
    parser.add_argument("--timeout-per-trial", type=int, default=14400)
    parser.add_argument("--skip-large-alpha", action="store_true", default=False,
                        help="Skip lora_alpha >= 0.5 to avoid OOM")
    return parser.parse_args()


def _parse_epoch_acc(line):
    m = re.search(r'Test:\s*\[\s*(\d+)/\d+\].*Acc@1:\s*[\d.]+\s*\(([\d.]+)\)', line)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def run_single_trial(args, transfer_every, lora_alpha, lr, transfer_lr, trial_num):
    """Run a single trial for a given combination"""
    exp_name = f"te{transfer_every}_la{lora_alpha}_t{trial_num:04d}"
    exp_dir = os.path.join(args.output_dir, exp_name)

    # Check if already completed
    summary_path = os.path.join(exp_dir, "summary.csv")
    if os.path.exists(summary_path):
        print(f"  ✓ {exp_name} already completed, skipping...")
        try:
            with open(summary_path, "r") as f:
                lines = list(csv.DictReader(f))
                if lines and "eval_top1" in lines[-1]:
                    best_acc = max(float(row["eval_top1"]) for row in lines)
                    return best_acc
        except Exception:
            pass
        return 0.0

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
    print(f"Trial {trial_num} | te={transfer_every}, lora_alpha={lora_alpha}")
    print(f"  lr={lr:.5f}, transfer_lr={transfer_lr:.5f}")
    print(f"{'='*70}\n")

    os.makedirs(exp_dir, exist_ok=True)
    best_acc = 0.0
    last_epoch = -1

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line_s = line.rstrip()
            if any(kw in line_s for kw in ["Test:", "C-only", "*** Best", "Early", "Error", "error", "OOM", "out of memory"]):
                print(f"  {line_s}")

            parsed = _parse_epoch_acc(line_s)
            if parsed:
                epoch, acc = parsed
                if epoch > last_epoch:
                    last_epoch = epoch
                    best_acc = max(best_acc, acc)

        proc.wait(timeout=args.timeout_per_trial)
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Trial timed out")
        proc.kill()
        proc.wait()
    except KeyboardInterrupt:
        print(f"  ⚠ Interrupted by user")
        proc.kill()
        proc.wait()
        raise

    # Read from summary.csv if available
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                for row in csv.DictReader(f):
                    if "eval_top1" in row:
                        best_acc = max(best_acc, float(row["eval_top1"]))
        except Exception:
            pass

    trial_info = {
        "trial": trial_num,
        "transfer_every": transfer_every,
        "lora_alpha": lora_alpha,
        "lr": lr,
        "transfer_lr": transfer_lr,
        "best_accuracy": best_acc,
        "stopped_epoch": last_epoch,
    }
    with open(os.path.join(exp_dir, "trial_params.json"), "w") as f:
        json.dump(trial_info, f, indent=2)

    print(f"\n✓ Trial {trial_num} complete: best_acc={best_acc:.2f}%")
    return best_acc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Failed combinations from original sweep (excluding te50_la0.01 and te50_la0.1 which succeeded)
    failed_configs = [
        # te=50, lora_alpha >= 0.5 (may still OOM with high alpha)
        (50, 0.5, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (50, 1.0, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        # te=100, all lora_alpha
        (100, 0.01, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (100, 0.1, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (100, 0.5, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (100, 1.0, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        # te=500, all lora_alpha
        (500, 0.01, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (500, 0.1, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (500, 0.5, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (500, 1.0, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        # te=1000, all lora_alpha
        (1000, 0.01, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (1000, 0.1, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (1000, 0.5, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
        (1000, 1.0, [(0.08404, 0.71145), (0.13793, 0.06251), (0.06207, 0.00294), (0.05419, 0.39676)]),
    ]

    if args.skip_large_alpha:
        print("⚠ Skipping lora_alpha >= 0.5 configurations to avoid OOM")
        failed_configs = [(te, la, trials) for te, la, trials in failed_configs if la < 0.5]

    print(f"\n{'='*80}")
    print(f"Resuming failed LRTT sweep combinations")
    print(f"Total combinations to retry: {len(failed_configs)}")
    print(f"Trials per combination: 4")
    print(f"Total trials: {len(failed_configs) * 4}")
    print(f"Sequential execution (one at a time)")
    print(f"{'='*80}\n")

    results = {}

    for te, lora_alpha, trial_params in failed_configs:
        comb_key = f"te{te}_la{lora_alpha}"

        print(f"\n{'#'*80}")
        print(f"# COMBINATION: transfer_every={te}, lora_alpha={lora_alpha}")
        print(f"{'#'*80}")

        best_acc = 0.0
        best_params = None

        for trial_num, (lr, transfer_lr) in enumerate(trial_params):
            acc = run_single_trial(args, te, lora_alpha, lr, transfer_lr, trial_num)
            if acc > best_acc:
                best_acc = acc
                best_params = {"lr": lr, "transfer_lr": transfer_lr}

            # Small delay between trials to ensure memory is freed
            time.sleep(2)

        print(f"\n--- {comb_key} Best: {best_acc:.2f}%, params={best_params}")

        results[comb_key] = {
            "transfer_every": te,
            "lora_alpha": lora_alpha,
            "best_accuracy": best_acc,
            "best_params": best_params,
        }

        # Save intermediate results
        with open(os.path.join(args.output_dir, "resume_sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("RESUME SWEEP COMPLETE")
    print(f"{'='*80}")

    for key in sorted(results.keys()):
        r = results[key]
        print(f"  {key}: {r['best_accuracy']:.2f}%")

    best_key = max(results.keys(), key=lambda k: results[k]['best_accuracy'])
    best_acc = results[best_key]['best_accuracy']
    print(f"\nBest: {best_key}, acc={best_acc:.2f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Bayesian hyperparameter sweep for classifier-only SQuAD finetuning.
Freezes all layers except qa_outputs, sweeps learning rate.

Uses TPE (Tree-structured Parzen Estimator) via optuna for Bayesian search.

Usage:
    cd /data/LRTT_transformer/lora_training
    python ../experiments/sweep_classifier_only_squad.py

    # Dry run
    python ../experiments/sweep_classifier_only_squad.py --dry-run
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Installing optuna...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna
    from optuna.samplers import TPESampler

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
QA_SCRIPT = BASE_DIR / "lora_training" / "experiments" / "run_classifier_only_squad.py"
# Use the script we created
QA_SCRIPT = SCRIPT_DIR / "run_classifier_only_squad.py"
WORKING_DIR = BASE_DIR / "lora_training"
RESULTS_DIR = Path("/data/AIMC_LoRA_results")
BEST_RESULTS_DIR = RESULTS_DIR / "digital" / "qa"

NUM_TRIALS = 50
NUM_EPOCHS = 3
BATCH_SIZE = 32
MODEL_NAME = "google/mobilebert-uncased"

# Sweep ranges
LR_MIN = 1e-6
LR_MAX = 1e-1


def objective(trial: optuna.Trial) -> float:
    """Single trial: train classifier-only on SQuAD, return eval_f1."""

    lr = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)

    trial_dir = RESULTS_DIR / "sweep_classifier_only_squad" / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(QA_SCRIPT),
        "--model_name_or_path", MODEL_NAME,
        "--dataset_name", "squad",
        "--do_train", "--do_eval",
        "--per_device_train_batch_size", str(BATCH_SIZE),
        "--num_train_epochs", str(NUM_EPOCHS),
        "--learning_rate", str(lr),
        "--logging_steps", "100",
        "--save_strategy", "no",
        "--output_dir", str(trial_dir),
        "--report_to", "none",
        "--overwrite_output_dir",
        "--warmup_ratio", "0.1",
    ]

    print(f"\n{'='*60}")
    print(f"Trial {trial.number:03d} | lr={lr:.2e}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKING_DIR),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per trial
        )

        if result.returncode != 0:
            print(f"[FAILED] Trial {trial.number}: {result.stderr[-500:]}")
            # Save error info
            with open(trial_dir / "error.txt", "w") as f:
                f.write(result.stderr)
            return 0.0

        # Parse eval results
        eval_file = trial_dir / "eval_results.json"
        if eval_file.exists():
            with open(eval_file) as f:
                eval_results = json.load(f)
            f1 = eval_results.get("eval_f1", 0.0)
            em = eval_results.get("eval_exact_match", 0.0)
            print(f"[OK] Trial {trial.number}: F1={f1:.2f}, EM={em:.2f}, lr={lr:.2e}")

            # Also save trial config
            trial_info = {
                "trial": trial.number,
                "learning_rate": lr,
                "eval_f1": f1,
                "eval_exact_match": em,
                "timestamp": datetime.now().isoformat(),
            }
            with open(trial_dir / "trial_config.json", "w") as f:
                json.dump(trial_info, f, indent=2)

            return f1
        else:
            # Try parsing from train_results
            train_file = trial_dir / "train_results.json"
            if train_file.exists():
                print(f"[WARN] Trial {trial.number}: no eval_results.json, checking stdout")

            # Parse from stdout as fallback
            for line in result.stdout.split('\n'):
                if 'eval_f1' in line:
                    try:
                        # Try to extract from printed results
                        parts = line.strip().split(':')
                        if len(parts) == 2:
                            return float(parts[1].strip())
                    except:
                        pass

            print(f"[WARN] Trial {trial.number}: could not parse eval results")
            return 0.0

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Trial {trial.number}")
        return 0.0
    except Exception as e:
        print(f"[ERROR] Trial {trial.number}: {e}")
        return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Set wandb offline
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DISABLED"] = "true"

    print(f"\n{'='*60}")
    print(f"CLASSIFIER-ONLY SQUAD BAYESIAN SWEEP")
    print(f"{'='*60}")
    print(f"Trials: {args.num_trials}")
    print(f"LR range: [{LR_MIN:.0e}, {LR_MAX:.0e}] (log scale)")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Model: {MODEL_NAME}")
    print(f"Best results → {BEST_RESULTS_DIR}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[DRY RUN] Would run Bayesian sweep with optuna TPE sampler")
        return

    # Create optuna study (maximize F1)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="classifier_only_squad",
    )

    study.optimize(objective, n_trials=args.num_trials)

    # ============================================================
    # Save results
    # ============================================================
    sweep_dir = RESULTS_DIR / "sweep_classifier_only_squad"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save all trials
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "trial": t.number,
            "learning_rate": t.params.get("learning_rate"),
            "eval_f1": t.value,
            "state": str(t.state),
        })

    with open(sweep_dir / "all_trials.json", "w") as f:
        json.dump(all_trials, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best LR: {study.best_params['learning_rate']:.2e}")

    # Top 5 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    print(f"\nTop 5 trials:")
    for t in sorted_trials[:5]:
        print(f"  Trial {t.number:03d}: F1={t.value:.4f}, lr={t.params.get('learning_rate', 0):.2e}")

    # ============================================================
    # Copy best trial results to digital/qa directory
    # ============================================================
    best_trial_dir = sweep_dir / f"trial_{study.best_trial.number:03d}"
    BEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy result files
    import shutil
    for fname in ["eval_results.json", "train_results.json", "trial_config.json"]:
        src = best_trial_dir / fname
        if src.exists():
            shutil.copy2(src, BEST_RESULTS_DIR / fname)

    # Save sweep summary in best results dir
    summary = {
        "experiment": "classifier_only_squad",
        "best_trial": study.best_trial.number,
        "best_f1": study.best_value,
        "best_learning_rate": study.best_params["learning_rate"],
        "num_trials": args.num_trials,
        "num_epochs": NUM_EPOCHS,
        "model": MODEL_NAME,
        "lr_range": [LR_MIN, LR_MAX],
        "timestamp": datetime.now().isoformat(),
    }
    with open(BEST_RESULTS_DIR / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest results saved to: {BEST_RESULTS_DIR}")
    print(f"All trials saved to: {sweep_dir}/all_trials.json")


if __name__ == "__main__":
    main()

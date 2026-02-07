#!/usr/bin/env python
"""
Sixt1c (6T1C) LoRA Hyperparameter Sweep Script for SQuAD.

Sweeps lora_alpha and analog_lr for different LoRA ranks on SQuAD.

Sweep parameters:
- lora_alpha: log_uniform(0.01, 100)
- analog_lr: log_uniform(1e-5, 1e-1)

Fixed parameters:
- sixt1c_dt_batch_sec: 1.0
- num_train_epochs: 3
- lora_dropout: 0.1
- warmup_ratio: 0.1

Usage:
    # Run sweep for rank 1
    python sweep_sixt1c_lora_squad.py --lora_rank 1

    # Run sweep for rank 4
    python sweep_sixt1c_lora_squad.py --lora_rank 4

    # Run sweep for rank 8
    python sweep_sixt1c_lora_squad.py --lora_rank 8

    # Run with nohup (background)
    nohup python sweep_sixt1c_lora_squad.py --lora_rank 1 > /data/AIMC_LoRA_results/sweep_sixt1c_lora_squad_r1.log 2>&1 &
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BASE_DIR,
    QA_SCRIPT,
    MODEL_NAME,
    RESULTS_DIR,
    SIXT1C_TRAINING_PARAMS,
    SIXT1C_PARAMS,
    ensure_directories,
)


# ============================================================================
# Sweep Configuration
# ============================================================================

# SQuAD task
SWEEP_TASKS: List[str] = [
    "squad",
]

# Number of trials per task
NUM_TRIALS = 50

# Sweep parameter ranges (log scale)
LORA_ALPHA_MIN = 0.01
LORA_ALPHA_MAX = 100.0
ANALOG_LR_MIN = 1e-5
ANALOG_LR_MAX = 1e-1

# Fixed parameters for sweep (LRTT-style: linear scheduler with warmup)
FIXED_PARAMS = {
    "sixt1c_dt_batch_sec": 1.0,
    "num_train_epochs": 3,
    "lora_dropout": 0.1,
    "warmup_steps": 500,
    "lr_scheduler_type": "linear",
    "save_strategy": "no",  # Avoid checkpoint save errors with analog tiles
}


def log_uniform(low: float, high: float, rng: np.random.Generator) -> float:
    """Sample from log-uniform distribution."""
    log_low = np.log(low)
    log_high = np.log(high)
    return float(np.exp(rng.uniform(log_low, log_high)))


def build_training_args(
    params: Dict[str, Any],
    output_dir: Path,
    run_name: str,
) -> List[str]:
    """Build command line arguments for training script."""
    args = []

    for key, value in params.items():
        if isinstance(value, bool):
            args.append(f"--{key}")
            args.append("true" if value else "false")
        else:
            args.append(f"--{key}")
            args.append(str(value))

    args.extend(["--output_dir", str(output_dir)])
    args.extend(["--run_name", run_name])

    return args


def run_single_experiment(
    task_name: str,
    lora_rank: int,
    lora_alpha: float,
    analog_lr: float,
    trial_idx: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run a single sweep experiment.

    Args:
        task_name: GLUE task name
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        analog_lr: Analog optimizer learning rate
        trial_idx: Trial index
        dry_run: If True, only print commands without executing

    Returns:
        Dictionary with experiment results
    """
    # Create output directory
    sweep_dir = RESULTS_DIR / "sweep_sixt1c_lora_squad" / f"rank{lora_rank}" / task_name
    output_dir = sweep_dir / f"trial_{trial_idx:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build run name
    run_name = f"sweep_sixt1c_squad_r{lora_rank}_t{trial_idx:03d}_alpha{lora_alpha:.4f}_lr{analog_lr:.2e}"

    # Build training parameters
    params = SIXT1C_TRAINING_PARAMS.copy()

    # Override with fixed sweep parameters
    params["num_train_epochs"] = FIXED_PARAMS["num_train_epochs"]
    params["warmup_steps"] = FIXED_PARAMS["warmup_steps"]
    params["lr_scheduler_type"] = FIXED_PARAMS["lr_scheduler_type"]
    params["save_strategy"] = FIXED_PARAMS["save_strategy"]

    # Build command
    cmd = [
        sys.executable,
        str(QA_SCRIPT),
        "--model_name_or_path", MODEL_NAME,
        "--dataset_name", "squad",
        "--baseline_mode", "analog",
        "--analog_device", "sixt1c",
    ]

    # Add training arguments
    cmd.extend(build_training_args(params, output_dir, run_name))
    cmd.extend(["--overwrite_output_dir"])

    # Add sixt1c-specific arguments
    sixt1c_params = SIXT1C_PARAMS.copy()
    cmd.extend(["--pcm_model", sixt1c_params["pcm_model"]])
    cmd.extend(["--sixt1c_dt_batch_sec", str(FIXED_PARAMS["sixt1c_dt_batch_sec"])])
    cmd.extend(["--sixt1c_include_retention", str(sixt1c_params["include_retention"])])
    cmd.extend(["--output_noise_level", str(sixt1c_params["output_noise_level"])])
    cmd.extend(["--analog_optimizer", sixt1c_params["analog_optimizer"]])
    cmd.extend(["--num_evaluation_drift_values", "0"])
    cmd.extend(["--num_evaluation_repetition", "1"])

    # Add swept parameters
    cmd.extend(["--analog_lr", str(analog_lr)])
    cmd.extend(["--lora_rank", str(lora_rank)])
    cmd.extend(["--lora_alpha", str(lora_alpha)])
    cmd.extend(["--lora_dropout", str(FIXED_PARAMS["lora_dropout"])])

    # Log experiment info
    experiment_info = {
        "task_name": task_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "analog_lr": analog_lr,
        "trial_idx": trial_idx,
        "output_dir": str(output_dir),
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"  Task: {task_name}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha:.4f}")
    print(f"  Analog LR: {analog_lr:.2e}")
    print(f"  Trial: {trial_idx}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[DRY RUN] Command: {' '.join(cmd)}")
        experiment_info["status"] = "dry_run"
        return experiment_info

    # Save experiment config
    config_path = output_dir / "sweep_config.json"
    with open(config_path, "w") as f:
        json.dump(experiment_info, f, indent=2)

    # Execute command
    try:
        working_dir = BASE_DIR / "lora_training"
        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            check=True,
            capture_output=False,
        )
        experiment_info["status"] = "success"
        print(f"\n[SUCCESS] {run_name}")
    except subprocess.CalledProcessError as e:
        experiment_info["status"] = "failed"
        experiment_info["error"] = str(e)
        print(f"\n[FAILED] {run_name}: {e}")

    return experiment_info


def run_sweep(
    lora_rank: int,
    tasks: List[str] = None,
    num_trials: int = NUM_TRIALS,
    seed: int = 42,
    dry_run: bool = False,
    start_trial: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run hyperparameter sweep for a given LoRA rank.

    Args:
        lora_rank: LoRA rank to use
        tasks: List of tasks to sweep (default: all SWEEP_TASKS)
        num_trials: Number of trials per task
        seed: Random seed for reproducibility
        dry_run: If True, only print commands without executing

    Returns:
        List of experiment results
    """
    if tasks is None:
        tasks = SWEEP_TASKS

    # Initialize random generator
    rng = np.random.default_rng(seed)

    # Track results
    all_results = []

    print(f"\n{'='*60}")
    print(f"SIXT1C LoRA HYPERPARAMETER SWEEP (SQuAD)")
    print(f"{'='*60}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"Tasks: {tasks}")
    print(f"Trials per task: {num_trials}")
    print(f"Total experiments: {len(tasks) * num_trials}")
    print(f"Seed: {seed}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\nSweep ranges:")
    print(f"  lora_alpha: log_uniform({LORA_ALPHA_MIN}, {LORA_ALPHA_MAX})")
    print(f"  analog_lr: log_uniform({ANALOG_LR_MIN}, {ANALOG_LR_MAX})")
    print(f"\nFixed parameters:")
    for key, value in FIXED_PARAMS.items():
        print(f"  {key}: {value}")

    # Ensure directories exist
    ensure_directories()

    # Pre-sample all hyperparameters for reproducibility
    all_samples = {}
    for task in tasks:
        all_samples[task] = []
        for trial_idx in range(num_trials):
            lora_alpha = log_uniform(LORA_ALPHA_MIN, LORA_ALPHA_MAX, rng)
            analog_lr = log_uniform(ANALOG_LR_MIN, ANALOG_LR_MAX, rng)
            all_samples[task].append({
                "lora_alpha": lora_alpha,
                "analog_lr": analog_lr,
            })

    # Save sweep plan
    sweep_plan_dir = RESULTS_DIR / "sweep_sixt1c_lora_squad" / f"rank{lora_rank}"
    sweep_plan_dir.mkdir(parents=True, exist_ok=True)
    sweep_plan_path = sweep_plan_dir / "sweep_plan.json"
    sweep_plan = {
        "lora_rank": lora_rank,
        "tasks": tasks,
        "num_trials": num_trials,
        "seed": seed,
        "samples": all_samples,
        "timestamp": datetime.now().isoformat(),
    }
    with open(sweep_plan_path, "w") as f:
        json.dump(sweep_plan, f, indent=2)
    print(f"\nSweep plan saved to: {sweep_plan_path}")

    # Run experiments
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*60}")

        for trial_idx in range(start_trial, num_trials):
            samples = all_samples[task][trial_idx]

            result = run_single_experiment(
                task_name=task,
                lora_rank=lora_rank,
                lora_alpha=samples["lora_alpha"],
                analog_lr=samples["analog_lr"],
                trial_idx=trial_idx,
                dry_run=dry_run,
            )
            all_results.append(result)

    # Save all results
    results_path = sweep_plan_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSweep results saved to: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: {len(all_results)}")

    if not dry_run:
        successful = sum(1 for r in all_results if r.get("status") == "success")
        failed = sum(1 for r in all_results if r.get("status") == "failed")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed experiments:")
            for r in all_results:
                if r.get("status") == "failed":
                    print(f"  - {r['run_name']}: {r.get('error', 'Unknown error')}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Sixt1c LoRA hyperparameter sweep on SQuAD"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        required=True,
        help="LoRA rank to use (e.g., 1, 4, 8)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to run (default: squad)"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=NUM_TRIALS,
        help=f"Number of trials per task (default: {NUM_TRIALS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--start_trial",
        type=int,
        default=0,
        help="Trial index to start from (for resuming)"
    )

    args = parser.parse_args()

    # Validate tasks
    if args.tasks:
        invalid_tasks = [t for t in args.tasks if t not in SWEEP_TASKS]
        if invalid_tasks:
            print(f"Error: Invalid tasks: {invalid_tasks}")
            print(f"Valid tasks: {SWEEP_TASKS}")
            sys.exit(1)

    # Run sweep
    run_sweep(
        lora_rank=args.lora_rank,
        tasks=args.tasks,
        num_trials=args.num_trials,
        seed=args.seed,
        dry_run=args.dry_run,
        start_trial=args.start_trial,
    )


if __name__ == "__main__":
    main()

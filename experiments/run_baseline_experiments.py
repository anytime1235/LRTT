#!/usr/bin/env python
"""
Master runner script for baseline experiments.
Executes digital and analog baseline experiments for SQuAD QA and GLUE benchmark.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BASE_DIR,
    QA_SCRIPT,
    GLUE_SCRIPT,
    MODEL_NAME,
    GLUE_TASKS,
    DIGITAL_TRAINING_PARAMS,
    ANALOG_TRAINING_PARAMS,
    SIXT1C_TRAINING_PARAMS,
    ANALOG_PARAMS,
    SIXT1C_PARAMS,
    TIKITAKA_PARAMS,
    QUICK_TEST_PARAMS,
    DIGITAL_QA_DIR,
    DIGITAL_GLUE_DIR,
    ANALOG_QA_DIR,
    ANALOG_GLUE_DIR,
    SIXT1C_QA_DIR,
    SIXT1C_GLUE_DIR,
    TIKITAKA_QA_DIR,
    TIKITAKA_GLUE_DIR,
    WANDB_PROJECT,
    ensure_directories,
)


def build_training_args(
    params: Dict[str, Any],
    output_dir: Path,
    run_name: str,
) -> List[str]:
    """
    Build command line arguments for training script.

    Args:
        params: Dictionary of training parameters
        output_dir: Output directory for results
        run_name: Name for wandb run

    Returns:
        List of command line arguments
    """
    args = []

    for key, value in params.items():
        if isinstance(value, bool):
            # HuggingFace TrainingArguments expects "true"/"false" for boolean args
            args.append(f"--{key}")
            args.append("true" if value else "false")
        else:
            args.append(f"--{key}")
            args.append(str(value))

    args.extend(["--output_dir", str(output_dir)])
    args.extend(["--run_name", run_name])

    return args


def run_qa_experiment(
    experiment_type: str,
    quick_test: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Run SQuAD QA experiment.

    Args:
        experiment_type: 'digital', 'analog', 'sixt1c', or 'tikitaka'
        quick_test: If True, use reduced parameters for quick testing
        dry_run: If True, only print commands without executing

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print(f"Running QA Experiment: {experiment_type.upper()}")
    print("=" * 60)

    # Determine output directory
    if experiment_type == "digital":
        output_dir = DIGITAL_QA_DIR
    elif experiment_type == "sixt1c":
        output_dir = SIXT1C_QA_DIR
    elif experiment_type == "tikitaka":
        output_dir = TIKITAKA_QA_DIR
    else:
        output_dir = ANALOG_QA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build base parameters (sixt1c/tikitaka use same training params as analog)
    if experiment_type == "digital":
        params = DIGITAL_TRAINING_PARAMS.copy()
    elif experiment_type == "sixt1c":
        params = SIXT1C_TRAINING_PARAMS.copy()
    else:
        params = ANALOG_TRAINING_PARAMS.copy()

    # Apply quick test parameters if needed
    if quick_test:
        params.update({
            "max_train_samples": QUICK_TEST_PARAMS["max_train_samples"],
            "max_eval_samples": QUICK_TEST_PARAMS["max_eval_samples"],
            "num_train_epochs": QUICK_TEST_PARAMS["num_train_epochs"],
        })

    # Build command
    run_name = f"qa_{experiment_type}_baseline"
    if quick_test:
        run_name += "_quicktest"

    # For sixt1c/tikitaka, baseline_mode is "analog" but analog_device varies
    baseline_mode = "analog" if experiment_type in ["sixt1c", "tikitaka"] else experiment_type

    cmd = [
        sys.executable,
        str(QA_SCRIPT),
        "--model_name_or_path", MODEL_NAME,
        "--dataset_name", "squad",
        "--baseline_mode", baseline_mode,
    ]

    # Add training arguments
    cmd.extend(build_training_args(params, output_dir, run_name))

    # Add analog-specific arguments if needed
    if experiment_type == "analog":
        analog_params = ANALOG_PARAMS.copy()
        if quick_test:
            analog_params["num_evaluation_drift_values"] = QUICK_TEST_PARAMS["num_evaluation_drift_values"]
            analog_params["num_evaluation_repetition"] = QUICK_TEST_PARAMS["num_evaluation_repetition"]

        cmd.extend(["--analog_device", "pcm"])
        cmd.extend(["--pcm_model", analog_params["pcm_model"]])
        cmd.extend(["--output_noise_level", str(analog_params["output_noise_level"])])
        cmd.extend(["--analog_optimizer", analog_params["analog_optimizer"]])
        cmd.extend(["--analog_lr", str(analog_params["analog_lr"])])
        cmd.extend(["--num_evaluation_drift_values", str(analog_params["num_evaluation_drift_values"])])
        cmd.extend(["--num_evaluation_repetition", str(analog_params["num_evaluation_repetition"])])

    elif experiment_type == "sixt1c":
        sixt1c_params = SIXT1C_PARAMS.copy()
        if quick_test:
            sixt1c_params["num_evaluation_drift_values"] = QUICK_TEST_PARAMS["num_evaluation_drift_values"]
            sixt1c_params["num_evaluation_repetition"] = QUICK_TEST_PARAMS["num_evaluation_repetition"]

        cmd.extend(["--analog_device", "sixt1c"])
        cmd.extend(["--pcm_model", sixt1c_params["pcm_model"]])
        cmd.extend(["--sixt1c_dt_batch_sec", str(sixt1c_params["dt_batch_sec"])])
        cmd.extend(["--sixt1c_include_retention", str(sixt1c_params["include_retention"])])
        # write_noise_std=0 is hardcoded in sixt1c_config.py
        cmd.extend(["--output_noise_level", str(sixt1c_params["output_noise_level"])])
        cmd.extend(["--analog_optimizer", sixt1c_params["analog_optimizer"]])
        cmd.extend(["--analog_lr", str(sixt1c_params["analog_lr"])])
        cmd.extend(["--num_evaluation_drift_values", str(sixt1c_params["num_evaluation_drift_values"])])
        cmd.extend(["--num_evaluation_repetition", str(sixt1c_params["num_evaluation_repetition"])])

    elif experiment_type == "tikitaka":
        tikitaka_params = TIKITAKA_PARAMS.copy()
        if quick_test:
            tikitaka_params["num_evaluation_drift_values"] = QUICK_TEST_PARAMS["num_evaluation_drift_values"]
            tikitaka_params["num_evaluation_repetition"] = QUICK_TEST_PARAMS["num_evaluation_repetition"]

        cmd.extend(["--analog_device", "tikitaka"])
        cmd.extend(["--tikitaka_transfer_every", str(tikitaka_params["transfer_every"])])
        cmd.extend(["--tikitaka_transfer_lr", str(tikitaka_params["transfer_lr"])])
        cmd.extend(["--tikitaka_fast_lr", str(tikitaka_params["fast_lr"])])
        cmd.extend(["--tikitaka_auto_granularity", str(tikitaka_params["auto_granularity"])])
        cmd.extend(["--tikitaka_in_chop_prob", str(tikitaka_params["in_chop_prob"])])
        cmd.extend(["--tikitaka_target_modules"] + tikitaka_params["target_modules"])
        cmd.extend(["--analog_optimizer", tikitaka_params["analog_optimizer"]])
        cmd.extend(["--analog_lr", str(tikitaka_params["analog_lr"])])

        # Drift evaluation parameters
        if tikitaka_params.get("enable_drift", False):
            cmd.extend(["--tikitaka_enable_drift"])
            cmd.extend(["--tikitaka_drift_nu", str(tikitaka_params["drift_nu"])])
            cmd.extend(["--tikitaka_drift_nu_dtod", str(tikitaka_params["drift_nu_dtod"])])

        cmd.extend(["--num_evaluation_drift_values", str(tikitaka_params["num_evaluation_drift_values"])])
        cmd.extend(["--num_evaluation_repetition", str(tikitaka_params["num_evaluation_repetition"])])

    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] Skipping execution")
        return True

    # Execute command
    try:
        working_dir = BASE_DIR / "lora_training"
        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            check=True,
            capture_output=False,
        )
        print(f"\nQA experiment ({experiment_type}) completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running QA experiment ({experiment_type}): {e}")
        return False


def run_glue_experiment(
    task_name: str,
    experiment_type: str,
    quick_test: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Run GLUE experiment for a specific task.

    Args:
        task_name: GLUE task name
        experiment_type: 'digital', 'analog', 'sixt1c', or 'tikitaka'
        quick_test: If True, use reduced parameters for quick testing
        dry_run: If True, only print commands without executing

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "-" * 50)
    print(f"Running GLUE Task: {task_name.upper()} ({experiment_type})")
    print("-" * 50)

    # Determine output directory
    if experiment_type == "digital":
        output_dir = DIGITAL_GLUE_DIR / task_name
    elif experiment_type == "sixt1c":
        output_dir = SIXT1C_GLUE_DIR / task_name
    elif experiment_type == "tikitaka":
        output_dir = TIKITAKA_GLUE_DIR / task_name
    else:
        output_dir = ANALOG_GLUE_DIR / task_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build base parameters (sixt1c/tikitaka use same training params as analog)
    if experiment_type == "digital":
        params = DIGITAL_TRAINING_PARAMS.copy()
    elif experiment_type == "sixt1c":
        params = SIXT1C_TRAINING_PARAMS.copy()
    else:
        params = ANALOG_TRAINING_PARAMS.copy()

    # Apply quick test parameters if needed
    if quick_test:
        params.update({
            "max_train_samples": QUICK_TEST_PARAMS["max_train_samples"],
            "max_eval_samples": QUICK_TEST_PARAMS["max_eval_samples"],
            "num_train_epochs": QUICK_TEST_PARAMS["num_train_epochs"],
        })

    # Build command
    run_name = f"glue_{task_name}_{experiment_type}_baseline"
    if quick_test:
        run_name += "_quicktest"

    # For sixt1c/tikitaka, baseline_mode is "analog" but analog_device varies
    baseline_mode = "analog" if experiment_type in ["sixt1c", "tikitaka"] else experiment_type

    cmd = [
        sys.executable,
        str(GLUE_SCRIPT),
        "--model_name_or_path", MODEL_NAME,
        "--task_name", task_name,
        "--baseline_mode", baseline_mode,
    ]

    # Add training arguments
    cmd.extend(build_training_args(params, output_dir, run_name))

    # Add analog-specific arguments if needed
    if experiment_type == "analog":
        analog_params = ANALOG_PARAMS.copy()
        if quick_test:
            analog_params["num_evaluation_drift_values"] = QUICK_TEST_PARAMS["num_evaluation_drift_values"]
            analog_params["num_evaluation_repetition"] = QUICK_TEST_PARAMS["num_evaluation_repetition"]

        cmd.extend(["--analog_device", "pcm"])
        cmd.extend(["--pcm_model", analog_params["pcm_model"]])
        cmd.extend(["--output_noise_level", str(analog_params["output_noise_level"])])
        cmd.extend(["--analog_optimizer", analog_params["analog_optimizer"]])
        cmd.extend(["--analog_lr", str(analog_params["analog_lr"])])
        cmd.extend(["--num_evaluation_drift_values", str(analog_params["num_evaluation_drift_values"])])
        cmd.extend(["--num_evaluation_repetition", str(analog_params["num_evaluation_repetition"])])

    elif experiment_type == "sixt1c":
        sixt1c_params = SIXT1C_PARAMS.copy()
        if quick_test:
            sixt1c_params["num_evaluation_drift_values"] = QUICK_TEST_PARAMS["num_evaluation_drift_values"]
            sixt1c_params["num_evaluation_repetition"] = QUICK_TEST_PARAMS["num_evaluation_repetition"]

        cmd.extend(["--analog_device", "sixt1c"])
        cmd.extend(["--pcm_model", sixt1c_params["pcm_model"]])
        cmd.extend(["--sixt1c_dt_batch_sec", str(sixt1c_params["dt_batch_sec"])])
        cmd.extend(["--sixt1c_include_retention", str(sixt1c_params["include_retention"])])
        # write_noise_std=0 is hardcoded in sixt1c_config.py
        cmd.extend(["--output_noise_level", str(sixt1c_params["output_noise_level"])])
        cmd.extend(["--analog_optimizer", sixt1c_params["analog_optimizer"]])
        cmd.extend(["--analog_lr", str(sixt1c_params["analog_lr"])])
        cmd.extend(["--num_evaluation_drift_values", str(sixt1c_params["num_evaluation_drift_values"])])
        cmd.extend(["--num_evaluation_repetition", str(sixt1c_params["num_evaluation_repetition"])])

    elif experiment_type == "tikitaka":
        tikitaka_params = TIKITAKA_PARAMS.copy()
        if quick_test:
            tikitaka_params["num_evaluation_drift_values"] = QUICK_TEST_PARAMS["num_evaluation_drift_values"]
            tikitaka_params["num_evaluation_repetition"] = QUICK_TEST_PARAMS["num_evaluation_repetition"]

        cmd.extend(["--analog_device", "tikitaka"])
        cmd.extend(["--tikitaka_transfer_every", str(tikitaka_params["transfer_every"])])
        cmd.extend(["--tikitaka_transfer_lr", str(tikitaka_params["transfer_lr"])])
        cmd.extend(["--tikitaka_fast_lr", str(tikitaka_params["fast_lr"])])
        cmd.extend(["--tikitaka_auto_granularity", str(tikitaka_params["auto_granularity"])])
        cmd.extend(["--tikitaka_in_chop_prob", str(tikitaka_params["in_chop_prob"])])
        cmd.extend(["--tikitaka_target_modules"] + tikitaka_params["target_modules"])
        cmd.extend(["--analog_optimizer", tikitaka_params["analog_optimizer"]])
        cmd.extend(["--analog_lr", str(tikitaka_params["analog_lr"])])

        # Drift evaluation parameters
        if tikitaka_params.get("enable_drift", False):
            cmd.extend(["--tikitaka_enable_drift"])
            cmd.extend(["--tikitaka_drift_nu", str(tikitaka_params["drift_nu"])])
            cmd.extend(["--tikitaka_drift_nu_dtod", str(tikitaka_params["drift_nu_dtod"])])

        cmd.extend(["--num_evaluation_drift_values", str(tikitaka_params["num_evaluation_drift_values"])])
        cmd.extend(["--num_evaluation_repetition", str(tikitaka_params["num_evaluation_repetition"])])

    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] Skipping execution")
        return True

    # Execute command
    try:
        working_dir = BASE_DIR / "lora_training_glue"
        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            check=True,
            capture_output=False,
        )
        print(f"\nGLUE task {task_name} ({experiment_type}) completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running GLUE task {task_name} ({experiment_type}): {e}")
        return False


def run_digital_baseline(
    quick_test: bool = False,
    dry_run: bool = False,
    tasks: List[str] = None,
    skip_qa: bool = False,
    skip_glue: bool = False,
) -> Dict[str, bool]:
    """
    Run all digital baseline experiments.

    Args:
        quick_test: If True, use reduced parameters
        dry_run: If True, only print commands
        tasks: List of GLUE tasks to run (None for all)
        skip_qa: If True, skip QA experiment
        skip_glue: If True, skip GLUE experiments

    Returns:
        Dictionary mapping experiment names to success status
    """
    print("\n" + "=" * 60)
    print("DIGITAL BASELINE EXPERIMENTS")
    print("=" * 60)

    results = {}

    # Run QA experiment
    if not skip_qa:
        results["qa_digital"] = run_qa_experiment("digital", quick_test, dry_run)

    # Run GLUE experiments
    if not skip_glue:
        glue_tasks = tasks if tasks else GLUE_TASKS
        for task in glue_tasks:
            results[f"glue_{task}_digital"] = run_glue_experiment(task, "digital", quick_test, dry_run)

    return results


def run_analog_baseline(
    quick_test: bool = False,
    dry_run: bool = False,
    tasks: List[str] = None,
    skip_qa: bool = False,
    skip_glue: bool = False,
) -> Dict[str, bool]:
    """
    Run all analog baseline experiments.

    Args:
        quick_test: If True, use reduced parameters
        dry_run: If True, only print commands
        tasks: List of GLUE tasks to run (None for all)
        skip_qa: If True, skip QA experiment
        skip_glue: If True, skip GLUE experiments

    Returns:
        Dictionary mapping experiment names to success status
    """
    print("\n" + "=" * 60)
    print("ANALOG BASELINE EXPERIMENTS (PCM)")
    print("=" * 60)

    results = {}

    # Run QA experiment
    if not skip_qa:
        results["qa_analog"] = run_qa_experiment("analog", quick_test, dry_run)

    # Run GLUE experiments
    if not skip_glue:
        glue_tasks = tasks if tasks else GLUE_TASKS
        for task in glue_tasks:
            results[f"glue_{task}_analog"] = run_glue_experiment(task, "analog", quick_test, dry_run)

    return results


def run_sixt1c_baseline(
    quick_test: bool = False,
    dry_run: bool = False,
    tasks: List[str] = None,
    skip_qa: bool = False,
    skip_glue: bool = False,
) -> Dict[str, bool]:
    """
    Run all sixt1c (6T1C) baseline experiments.

    Args:
        quick_test: If True, use reduced parameters
        dry_run: If True, only print commands
        tasks: List of GLUE tasks to run (None for all)
        skip_qa: If True, skip QA experiment
        skip_glue: If True, skip GLUE experiments

    Returns:
        Dictionary mapping experiment names to success status
    """
    print("\n" + "=" * 60)
    print("SIXT1C (6T1C) BASELINE EXPERIMENTS")
    print("=" * 60)

    results = {}

    # Run QA experiment
    if not skip_qa:
        results["qa_sixt1c"] = run_qa_experiment("sixt1c", quick_test, dry_run)

    # Run GLUE experiments
    if not skip_glue:
        glue_tasks = tasks if tasks else GLUE_TASKS
        for task in glue_tasks:
            results[f"glue_{task}_sixt1c"] = run_glue_experiment(task, "sixt1c", quick_test, dry_run)

    return results


def run_tikitaka_baseline(
    quick_test: bool = False,
    dry_run: bool = False,
    tasks: List[str] = None,
    skip_qa: bool = False,
    skip_glue: bool = False,
) -> Dict[str, bool]:
    """
    Run all TikiTaka (2-tile: Fast 6T1C + Slow SoftBounds) baseline experiments.

    TikiTaka uses a 2-tile structure:
    - Fast tile (6T1C): receives SGD updates
    - Slow tile (SoftBounds): stable weight storage, visible in forward pass
    - Periodic transfer from Fast -> Slow updates the C tile

    Args:
        quick_test: If True, use reduced parameters
        dry_run: If True, only print commands
        tasks: List of GLUE tasks to run (None for all)
        skip_qa: If True, skip QA experiment
        skip_glue: If True, skip GLUE experiments

    Returns:
        Dictionary mapping experiment names to success status
    """
    print("\n" + "=" * 60)
    print("TIKITAKA (2-TILE) BASELINE EXPERIMENTS")
    print("=" * 60)

    results = {}

    # Run QA experiment
    if not skip_qa:
        results["qa_tikitaka"] = run_qa_experiment("tikitaka", quick_test, dry_run)

    # Run GLUE experiments
    if not skip_glue:
        glue_tasks = tasks if tasks else GLUE_TASKS
        for task in glue_tasks:
            results[f"glue_{task}_tikitaka"] = run_glue_experiment(task, "tikitaka", quick_test, dry_run)

    return results


def collect_and_plot_results():
    """Collect results and generate plots."""
    print("\n" + "=" * 60)
    print("COLLECTING RESULTS AND GENERATING PLOTS")
    print("=" * 60)

    from collect_results import create_summary_csv
    from generate_plots import generate_all_plots

    summary_df = create_summary_csv()
    if summary_df is not None and not summary_df.empty:
        generate_all_plots(summary_df)
    else:
        print("No results to plot")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiments for MobileBERT on SQuAD QA and GLUE benchmark"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["digital", "analog", "sixt1c", "tikitaka", "both", "all"],
        default="both",
        help="Experiment mode: digital, analog, sixt1c, tikitaka, both (digital+analog), or all (default: both)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run with reduced parameters for quick testing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Specific GLUE tasks to run (default: all)"
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip SQuAD QA experiments"
    )
    parser.add_argument(
        "--skip-glue",
        action="store_true",
        help="Skip GLUE experiments"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip results collection and plot generation"
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect results and generate plots (skip training)"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("BASELINE EXPERIMENTS CONFIGURATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Quick test: {args.quick_test}")
    print(f"Dry run: {args.dry_run}")
    print(f"Tasks: {args.tasks if args.tasks else 'all'}")
    print(f"Skip QA: {args.skip_qa}")
    print(f"Skip GLUE: {args.skip_glue}")
    print(f"Skip plots: {args.skip_plots}")
    print(f"Collect only: {args.collect_only}")
    print(f"Model: {MODEL_NAME}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Ensure directories exist
    ensure_directories()

    # Track all results
    all_results = {}

    if not args.collect_only:
        # Run digital baseline
        if args.mode in ["digital", "both", "all"]:
            digital_results = run_digital_baseline(
                quick_test=args.quick_test,
                dry_run=args.dry_run,
                tasks=args.tasks,
                skip_qa=args.skip_qa,
                skip_glue=args.skip_glue,
            )
            all_results.update(digital_results)

        # Run analog baseline (PCM)
        if args.mode in ["analog", "both", "all"]:
            analog_results = run_analog_baseline(
                quick_test=args.quick_test,
                dry_run=args.dry_run,
                tasks=args.tasks,
                skip_qa=args.skip_qa,
                skip_glue=args.skip_glue,
            )
            all_results.update(analog_results)

        # Run sixt1c baseline (6T1C)
        if args.mode in ["sixt1c", "all"]:
            sixt1c_results = run_sixt1c_baseline(
                quick_test=args.quick_test,
                dry_run=args.dry_run,
                tasks=args.tasks,
                skip_qa=args.skip_qa,
                skip_glue=args.skip_glue,
            )
            all_results.update(sixt1c_results)

        # Run TikiTaka baseline (2-tile: Fast 6T1C + Slow SoftBounds)
        if args.mode in ["tikitaka", "all"]:
            tikitaka_results = run_tikitaka_baseline(
                quick_test=args.quick_test,
                dry_run=args.dry_run,
                tasks=args.tasks,
                skip_qa=args.skip_qa,
                skip_glue=args.skip_glue,
            )
            all_results.update(tikitaka_results)

    # Collect results and generate plots
    if not args.skip_plots and not args.dry_run:
        collect_and_plot_results()

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if all_results:
        successful = sum(1 for v in all_results.values() if v)
        failed = sum(1 for v in all_results.values() if not v)
        print(f"Total experiments: {len(all_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed experiments:")
            for name, success in all_results.items():
                if not success:
                    print(f"  - {name}")

    print("=" * 60)


if __name__ == "__main__":
    main()

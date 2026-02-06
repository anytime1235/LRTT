"""
Results collection module for baseline experiments.
Parses JSON results from training runs and converts them to CSV format.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    RESULTS_DIR,
    DIGITAL_QA_DIR,
    DIGITAL_GLUE_DIR,
    ANALOG_QA_DIR,
    ANALOG_GLUE_DIR,
    SIXT1C_QA_DIR,
    SIXT1C_GLUE_DIR,
    GLUE_TASKS,
    GLUE_METRICS,
    DRIFT_VALUES_SECONDS,
    DRIFT_LABELS,
    MODEL_NAME,
)


def parse_qa_results(output_dir: Path, experiment_type: str) -> pd.DataFrame:
    """
    Parse QA experiment results from output directory.

    Args:
        output_dir: Directory containing experiment results
        experiment_type: 'digital', 'analog', or 'sixt1c'

    Returns:
        DataFrame with parsed results
    """
    results = []

    if experiment_type == "digital":
        # For digital, look for eval_results.json
        eval_file = output_dir / "eval_results.json"
        if eval_file.exists():
            with open(eval_file, "r") as f:
                metrics = json.load(f)
            results.append({
                "experiment_type": "digital",
                "model": MODEL_NAME,
                "dataset": "squad",
                "drift_seconds": 0,
                "drift_label": "t0",
                "exact_match_mean": metrics.get("eval_exact_match", metrics.get("exact_match", 0)),
                "f1_mean": metrics.get("eval_f1", metrics.get("f1", 0)),
                "exact_match_std": 0.0,
                "f1_std": 0.0,
            })
    else:
        # For analog and sixt1c, look for drift-specific results
        for drift_sec in DRIFT_VALUES_SECONDS:
            mean_file = output_dir / f"eval_mean_driftsecond={drift_sec}s_results.json"
            std_file = output_dir / f"eval_std_driftsecond={drift_sec}s_results.json"

            if mean_file.exists():
                with open(mean_file, "r") as f:
                    mean_metrics = json.load(f)
                std_metrics = {}
                if std_file.exists():
                    with open(std_file, "r") as f:
                        std_metrics = json.load(f)

                results.append({
                    "experiment_type": experiment_type,  # 'analog' or 'sixt1c'
                    "model": MODEL_NAME,
                    "dataset": "squad",
                    "drift_seconds": drift_sec,
                    "drift_label": DRIFT_LABELS.get(drift_sec, str(drift_sec)),
                    "exact_match_mean": mean_metrics.get("eval_exact_match", mean_metrics.get("exact_match", 0)),
                    "f1_mean": mean_metrics.get("eval_f1", mean_metrics.get("f1", 0)),
                    "exact_match_std": std_metrics.get("eval_exact_match", std_metrics.get("exact_match", 0)),
                    "f1_std": std_metrics.get("eval_f1", std_metrics.get("f1", 0)),
                })

    return pd.DataFrame(results)


def parse_glue_results(output_dir: Path, task_name: str, experiment_type: str) -> pd.DataFrame:
    """
    Parse GLUE experiment results from output directory.

    Args:
        output_dir: Directory containing experiment results
        task_name: GLUE task name
        experiment_type: 'digital', 'analog', or 'sixt1c'

    Returns:
        DataFrame with parsed results
    """
    results = []
    primary_metric = GLUE_METRICS.get(task_name, "accuracy")
    task_dir = output_dir / task_name if (output_dir / task_name).exists() else output_dir

    if experiment_type == "digital":
        eval_file = task_dir / "eval_results.json"
        if eval_file.exists():
            with open(eval_file, "r") as f:
                metrics = json.load(f)

            metric_value = metrics.get(f"eval_{primary_metric}", metrics.get(primary_metric, 0))
            results.append({
                "experiment_type": "digital",
                "task_name": task_name,
                "drift_seconds": 0,
                "drift_label": "t0",
                "metric_name": primary_metric,
                "metric_mean": metric_value,
                "metric_std": 0.0,
            })
    else:
        # For analog and sixt1c, look for drift-specific results
        for drift_sec in DRIFT_VALUES_SECONDS:
            mean_file = task_dir / f"eval_mean_driftsecond={drift_sec}s_results.json"
            std_file = task_dir / f"eval_std_driftsecond={drift_sec}s_results.json"

            if mean_file.exists():
                with open(mean_file, "r") as f:
                    mean_metrics = json.load(f)
                std_metrics = {}
                if std_file.exists():
                    with open(std_file, "r") as f:
                        std_metrics = json.load(f)

                metric_mean = mean_metrics.get(f"eval_{primary_metric}", mean_metrics.get(primary_metric, 0))
                metric_std = std_metrics.get(f"eval_{primary_metric}", std_metrics.get(primary_metric, 0))

                results.append({
                    "experiment_type": experiment_type,  # 'analog' or 'sixt1c'
                    "task_name": task_name,
                    "drift_seconds": drift_sec,
                    "drift_label": DRIFT_LABELS.get(drift_sec, str(drift_sec)),
                    "metric_name": primary_metric,
                    "metric_mean": metric_mean,
                    "metric_std": metric_std,
                })

    return pd.DataFrame(results)


def collect_all_qa_results() -> Dict[str, pd.DataFrame]:
    """
    Collect all QA results (digital, analog, and sixt1c).

    Returns:
        Dictionary with 'digital', 'analog', and 'sixt1c' DataFrames
    """
    results = {}

    # Digital QA results
    if DIGITAL_QA_DIR.exists():
        results["digital"] = parse_qa_results(DIGITAL_QA_DIR, "digital")
        if not results["digital"].empty:
            results["digital"].to_csv(DIGITAL_QA_DIR / "results.csv", index=False)
            print(f"Saved digital QA results to {DIGITAL_QA_DIR / 'results.csv'}")

    # Analog QA results (PCM)
    if ANALOG_QA_DIR.exists():
        results["analog"] = parse_qa_results(ANALOG_QA_DIR, "analog")
        if not results["analog"].empty:
            results["analog"].to_csv(ANALOG_QA_DIR / "results.csv", index=False)
            print(f"Saved analog QA results to {ANALOG_QA_DIR / 'results.csv'}")

    # Sixt1c QA results (6T1C)
    if SIXT1C_QA_DIR.exists():
        results["sixt1c"] = parse_qa_results(SIXT1C_QA_DIR, "sixt1c")
        if not results["sixt1c"].empty:
            results["sixt1c"].to_csv(SIXT1C_QA_DIR / "results.csv", index=False)
            print(f"Saved sixt1c QA results to {SIXT1C_QA_DIR / 'results.csv'}")

    return results


def collect_all_glue_results() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Collect all GLUE results (digital, analog, and sixt1c).

    Returns:
        Nested dictionary with experiment type -> task -> DataFrame
    """
    results = {"digital": {}, "analog": {}, "sixt1c": {}}

    for task_name in GLUE_TASKS:
        # Digital GLUE results
        if DIGITAL_GLUE_DIR.exists():
            df = parse_glue_results(DIGITAL_GLUE_DIR, task_name, "digital")
            if not df.empty:
                results["digital"][task_name] = df
                task_csv_path = DIGITAL_GLUE_DIR / f"{task_name}_results.csv"
                df.to_csv(task_csv_path, index=False)
                print(f"Saved digital GLUE {task_name} results to {task_csv_path}")

        # Analog GLUE results (PCM)
        if ANALOG_GLUE_DIR.exists():
            df = parse_glue_results(ANALOG_GLUE_DIR, task_name, "analog")
            if not df.empty:
                results["analog"][task_name] = df
                task_csv_path = ANALOG_GLUE_DIR / f"{task_name}_results.csv"
                df.to_csv(task_csv_path, index=False)
                print(f"Saved analog GLUE {task_name} results to {task_csv_path}")

        # Sixt1c GLUE results (6T1C)
        if SIXT1C_GLUE_DIR.exists():
            df = parse_glue_results(SIXT1C_GLUE_DIR, task_name, "sixt1c")
            if not df.empty:
                results["sixt1c"][task_name] = df
                task_csv_path = SIXT1C_GLUE_DIR / f"{task_name}_results.csv"
                df.to_csv(task_csv_path, index=False)
                print(f"Saved sixt1c GLUE {task_name} results to {task_csv_path}")

    return results


def create_summary_csv() -> pd.DataFrame:
    """
    Create a summary CSV combining all experiment results.

    Returns:
        Summary DataFrame
    """
    summary_rows = []

    # Collect QA results
    qa_results = collect_all_qa_results()
    for exp_type, df in qa_results.items():
        if not df.empty:
            for _, row in df.iterrows():
                summary_rows.append({
                    "experiment_type": row["experiment_type"],
                    "task_type": "qa",
                    "task_name": "squad",
                    "drift_seconds": row["drift_seconds"],
                    "drift_label": row["drift_label"],
                    "primary_metric": "f1",
                    "metric_mean": row["f1_mean"],
                    "metric_std": row["f1_std"],
                    "secondary_metric": "exact_match",
                    "secondary_mean": row["exact_match_mean"],
                    "secondary_std": row["exact_match_std"],
                })

    # Collect GLUE results
    glue_results = collect_all_glue_results()
    for exp_type, tasks_dict in glue_results.items():
        for task_name, df in tasks_dict.items():
            if not df.empty:
                for _, row in df.iterrows():
                    summary_rows.append({
                        "experiment_type": row["experiment_type"],
                        "task_type": "glue",
                        "task_name": task_name,
                        "drift_seconds": row["drift_seconds"],
                        "drift_label": row["drift_label"],
                        "primary_metric": row["metric_name"],
                        "metric_mean": row["metric_mean"],
                        "metric_std": row["metric_std"],
                        "secondary_metric": None,
                        "secondary_mean": None,
                        "secondary_std": None,
                    })

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        summary_path = RESULTS_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")

    return summary_df


def scan_output_directory(output_dir: Path) -> Dict[str, Any]:
    """
    Scan an output directory and return all available JSON result files.

    Args:
        output_dir: Directory to scan

    Returns:
        Dictionary mapping file names to their contents
    """
    results = {}
    if not output_dir.exists():
        return results

    for file_path in output_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                results[file_path.name] = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {file_path}")

    return results


def main():
    """Main function to collect all results."""
    print("=" * 60)
    print("Collecting Baseline Experiment Results")
    print("=" * 60)
    print()

    # Create summary
    summary_df = create_summary_csv()

    print()
    print("=" * 60)
    print("Results Collection Complete")
    print("=" * 60)

    if not summary_df.empty:
        print(f"\nTotal entries in summary: {len(summary_df)}")
        print(f"\nSummary by experiment type:")
        print(summary_df.groupby(["experiment_type", "task_type"]).size())
    else:
        print("\nNo results found to collect.")

    return summary_df


if __name__ == "__main__":
    main()

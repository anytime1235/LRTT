"""
Plot generation module for baseline experiments.
Creates visualizations comparing digital vs analog performance and drift effects.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from config import (
    RESULTS_DIR,
    PLOTS_DIR,
    GLUE_TASKS,
    GLUE_METRICS,
    DRIFT_VALUES_SECONDS,
    DRIFT_LABELS,
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_summary_data() -> Optional[pd.DataFrame]:
    """Load the summary CSV file."""
    summary_path = RESULTS_DIR / "summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    print(f"Warning: Summary file not found at {summary_path}")
    return None


def plot_digital_vs_analog_comparison(summary_df: pd.DataFrame, output_path: Path = None) -> None:
    """
    Create a bar chart comparing digital vs analog baseline performance at t0.

    Args:
        summary_df: Summary DataFrame with all results
        output_path: Path to save the plot
    """
    if output_path is None:
        output_path = PLOTS_DIR / "digital_vs_analog_comparison.png"

    # Filter for t0 (drift_seconds = 0)
    t0_data = summary_df[summary_df["drift_seconds"] == 0].copy()

    if t0_data.empty:
        print("No t0 data available for digital vs analog comparison")
        return

    # Pivot for easier plotting
    pivot_data = t0_data.pivot_table(
        index="task_name",
        columns="experiment_type",
        values="metric_mean",
        aggfunc="first"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set up bar positions
    x = np.arange(len(pivot_data.index))
    width = 0.35

    # Plot bars
    colors = {"digital": "#2ecc71", "analog": "#e74c3c"}

    if "digital" in pivot_data.columns:
        bars1 = ax.bar(x - width/2, pivot_data["digital"], width,
                       label="Digital", color=colors["digital"], alpha=0.8)
    if "analog" in pivot_data.columns:
        bars2 = ax.bar(x + width/2, pivot_data["analog"], width,
                       label="Analog (t0)", color=colors["analog"], alpha=0.8)

    # Customize plot
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Digital vs Analog Baseline Performance (at t0)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_data.index, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    if "digital" in pivot_data.columns:
        autolabel(bars1)
    if "analog" in pivot_data.columns:
        autolabel(bars2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved digital vs analog comparison plot to {output_path}")


def plot_drift_vs_metrics(summary_df: pd.DataFrame, output_path: Path = None) -> None:
    """
    Create line plots showing metric degradation over drift time.

    Args:
        summary_df: Summary DataFrame with all results
        output_path: Path to save the plot
    """
    if output_path is None:
        output_path = PLOTS_DIR / "drift_vs_metrics.png"

    # Filter for analog results only
    analog_data = summary_df[summary_df["experiment_type"] == "analog"].copy()

    if analog_data.empty:
        print("No analog data available for drift vs metrics plot")
        return

    # Get unique tasks
    tasks = analog_data["task_name"].unique()
    n_tasks = len(tasks)

    # Calculate grid size
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_tasks == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_tasks))

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = analog_data[analog_data["task_name"] == task].sort_values("drift_seconds")

        if task_data.empty:
            continue

        # Plot with error bars
        drift_labels = [DRIFT_LABELS.get(d, str(d)) for d in task_data["drift_seconds"]]

        ax.errorbar(
            range(len(task_data)),
            task_data["metric_mean"],
            yerr=task_data["metric_std"],
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=5,
            color=colors[idx],
            label=task
        )

        ax.set_xticks(range(len(task_data)))
        ax.set_xticklabels(drift_labels, rotation=45, ha="right")
        ax.set_xlabel("Drift Time")
        ax.set_ylabel(f"{task_data['primary_metric'].iloc[0]}")
        ax.set_title(f"{task.upper()}", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Metric Degradation Over Drift Time (Analog)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved drift vs metrics plot to {output_path}")


def plot_task_comparison(summary_df: pd.DataFrame, output_path: Path = None) -> None:
    """
    Create a grouped bar chart comparing all tasks across different drift times.

    Args:
        summary_df: Summary DataFrame with all results
        output_path: Path to save the plot
    """
    if output_path is None:
        output_path = PLOTS_DIR / "task_comparison.png"

    # Filter for analog results
    analog_data = summary_df[summary_df["experiment_type"] == "analog"].copy()

    if analog_data.empty:
        print("No analog data available for task comparison plot")
        return

    # Get unique tasks and drift times
    tasks = sorted(analog_data["task_name"].unique())
    drift_times = sorted(analog_data["drift_seconds"].unique())

    # Create pivot table
    pivot_data = analog_data.pivot_table(
        index="task_name",
        columns="drift_seconds",
        values="metric_mean",
        aggfunc="first"
    ).reindex(tasks)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Set up bar positions
    x = np.arange(len(tasks))
    n_drifts = len(drift_times)
    width = 0.8 / n_drifts

    # Color palette for drift times
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_drifts))

    # Plot bars for each drift time
    for i, drift in enumerate(drift_times):
        if drift in pivot_data.columns:
            offset = (i - n_drifts / 2 + 0.5) * width
            label = DRIFT_LABELS.get(drift, str(drift))
            ax.bar(x + offset, pivot_data[drift], width,
                   label=label, color=colors[i], alpha=0.85)

    # Customize plot
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Task Performance Comparison Across Drift Times (Analog)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.legend(title="Drift Time", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved task comparison plot to {output_path}")


def plot_drift_heatmap(summary_df: pd.DataFrame, output_path: Path = None) -> None:
    """
    Create a heatmap showing performance across tasks and drift times.

    Args:
        summary_df: Summary DataFrame with all results
        output_path: Path to save the plot
    """
    if output_path is None:
        output_path = PLOTS_DIR / "drift_heatmap.png"

    # Filter for analog results
    analog_data = summary_df[summary_df["experiment_type"] == "analog"].copy()

    if analog_data.empty:
        print("No analog data available for drift heatmap")
        return

    # Create pivot table for heatmap
    pivot_data = analog_data.pivot_table(
        index="task_name",
        columns="drift_seconds",
        values="metric_mean",
        aggfunc="first"
    )

    # Rename columns to drift labels
    pivot_data.columns = [DRIFT_LABELS.get(d, str(d)) for d in pivot_data.columns]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Metric Value"}
    )

    ax.set_xlabel("Drift Time", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title("Performance Heatmap: Tasks vs Drift Time (Analog)", fontsize=14, fontweight="bold")

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved drift heatmap to {output_path}")


def plot_performance_degradation_rate(summary_df: pd.DataFrame, output_path: Path = None) -> None:
    """
    Create a plot showing the rate of performance degradation for each task.

    Args:
        summary_df: Summary DataFrame with all results
        output_path: Path to save the plot
    """
    if output_path is None:
        output_path = PLOTS_DIR / "degradation_rate.png"

    # Filter for analog results
    analog_data = summary_df[summary_df["experiment_type"] == "analog"].copy()

    if analog_data.empty:
        print("No analog data available for degradation rate plot")
        return

    # Calculate relative degradation from t0
    tasks = analog_data["task_name"].unique()
    degradation_data = []

    for task in tasks:
        task_data = analog_data[analog_data["task_name"] == task].sort_values("drift_seconds")
        if task_data.empty:
            continue

        t0_value = task_data[task_data["drift_seconds"] == 0]["metric_mean"].values
        if len(t0_value) == 0:
            continue
        t0_value = t0_value[0]

        for _, row in task_data.iterrows():
            if t0_value > 0:
                relative_perf = row["metric_mean"] / t0_value
            else:
                relative_perf = 0
            degradation_data.append({
                "task_name": task,
                "drift_seconds": row["drift_seconds"],
                "drift_label": row["drift_label"],
                "relative_performance": relative_perf,
            })

    if not degradation_data:
        print("No degradation data available")
        return

    deg_df = pd.DataFrame(degradation_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each task
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    for idx, task in enumerate(tasks):
        task_data = deg_df[deg_df["task_name"] == task].sort_values("drift_seconds")
        if task_data.empty:
            continue

        ax.plot(
            range(len(task_data)),
            task_data["relative_performance"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=task,
            color=colors[idx]
        )

    # Get drift labels for x-axis
    drift_order = sorted(deg_df["drift_seconds"].unique())
    drift_labels = [DRIFT_LABELS.get(d, str(d)) for d in drift_order]

    ax.set_xticks(range(len(drift_labels)))
    ax.set_xticklabels(drift_labels, rotation=45, ha="right")
    ax.set_xlabel("Drift Time", fontsize=12)
    ax.set_ylabel("Relative Performance (vs t0)", fontsize=12)
    ax.set_title("Performance Degradation Rate (Analog)", fontsize=14, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (t0)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved degradation rate plot to {output_path}")


def generate_all_plots(summary_df: pd.DataFrame = None) -> None:
    """
    Generate all plots from the summary data.

    Args:
        summary_df: Optional summary DataFrame. If not provided, loads from file.
    """
    if summary_df is None:
        summary_df = load_summary_data()

    if summary_df is None or summary_df.empty:
        print("No data available for plotting")
        return

    # Ensure plots directory exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Plots")
    print("=" * 60)
    print()

    # Generate all plots
    plot_digital_vs_analog_comparison(summary_df)
    plot_drift_vs_metrics(summary_df)
    plot_task_comparison(summary_df)
    plot_drift_heatmap(summary_df)
    plot_performance_degradation_rate(summary_df)

    print()
    print("=" * 60)
    print("Plot Generation Complete")
    print("=" * 60)
    print(f"All plots saved to: {PLOTS_DIR}")


def main():
    """Main function to generate all plots."""
    generate_all_plots()


if __name__ == "__main__":
    main()

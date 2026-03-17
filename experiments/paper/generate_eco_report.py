#!/usr/bin/env python
# coding=utf-8
"""Post-experiment analysis for ECO comparison experiments.

Reads summary.json and diagnostic CSVs from E0-E3 experiments.
Generates:
  - E1 F1 bar chart (mean±std over seeds)
  - E2 pulse-type heatmap
  - VRC_K vs K line plots
  - TTv1 transfer histograms
  - Markdown report

Usage:
    python generate_eco_report.py --results-dir results/paper
    python generate_eco_report.py --e1-dir results/paper/eco_E1_main
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not available, skipping plots")


# ============================================================================
# Data loading
# ============================================================================

def load_summaries(base_dir):
    """Load all summary.json files from subdirectories."""
    summaries = {}
    if not os.path.isdir(base_dir):
        return summaries
    for tag in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, tag, "summary.json")
        if os.path.isfile(path):
            with open(path) as f:
                summaries[tag] = json.load(f)
    return summaries


def load_csv_records(csv_path):
    """Load CSV file as list of dicts."""
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_carry_path_summary(base_dir, tag):
    """Load carry_path_summary.json for a given experiment tag."""
    path = os.path.join(base_dir, tag, "carry_path_summary.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================================
# E1: Main comparison
# ============================================================================

E1_METHODS = [
    ("single_rpu_deterministic", "SingleRPU (det)"),
    ("single_rpu_stochastic", "SingleRPU (stoch)"),
    ("eco_ref_rtn", "ECO ref (RTN)"),
    ("eco_ref_stochastic", "ECO ref (stoch)"),
    ("mixed_precision", "MixedPrecision"),
    ("ttv1_hidden_buffer", "TTv1 HiddenBuf"),
    ("ttv1_residual_lane", "TTv1 ResLane"),
    ("ttv1_residual_lane_noreset", "TTv1 ResLane NR"),
]
E1_SEEDS = [42, 43, 44]


def analyze_e1(e1_dir, output_dir):
    """Analyze E1 main comparison results."""
    summaries = load_summaries(e1_dir)

    results = {}
    for prefix, label in E1_METHODS:
        f1s = []
        for s in E1_SEEDS:
            tag = f"{prefix}_s{s}"
            if tag in summaries:
                f1s.append(summaries[tag]["results"]["best_f1"])
        if f1s:
            results[label] = {"mean": np.mean(f1s), "std": np.std(f1s),
                              "values": f1s}
        else:
            results[label] = {"mean": 0, "std": 0, "values": []}

    # Bar chart
    if HAS_MPL and results:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = list(results.keys())
        means = [results[l]["mean"] for l in labels]
        stds = [results[l]["std"] for l in labels]

        bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
                      color=["#2196F3", "#42A5F5",  # SingleRPU
                             "#FF9800", "#FFB74D",   # ECO
                             "#4CAF50",              # MixedPrec
                             "#9C27B0", "#BA68C8", "#CE93D8"],  # TTv1
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Best F1")
        ax.set_title("E1: Main Comparison (mean ± std over 3 seeds)")
        ax.set_ylim(bottom=max(0, min(means) - 5))
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.3, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "e1_f1_bar.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: e1_f1_bar.png")

    return results


# ============================================================================
# E2: Pulse factorial
# ============================================================================

def analyze_e2(e2_dir, output_dir):
    """Analyze E2 TTv1 pulse factorial results."""
    summaries = load_summaries(e2_dir)

    modes = ["hidden_buffer", "residual_lane"]
    fast_pulses = ["stochastic", "deterministic"]
    transfer_pulses = ["stochastic", "deterministic"]

    grid = {}
    for mode in modes:
        grid[mode] = {}
        for fp in fast_pulses:
            for tp in transfer_pulses:
                tag = f"{mode}_fp{fp}_tp{tp}"
                if tag in summaries:
                    grid[mode][(fp, tp)] = summaries[tag]["results"]["best_f1"]
                else:
                    grid[mode][(fp, tp)] = 0.0

    # Heatmap
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            data = np.array([
                [grid[mode].get(("stochastic", "stochastic"), 0),
                 grid[mode].get(("stochastic", "deterministic"), 0)],
                [grid[mode].get(("deterministic", "stochastic"), 0),
                 grid[mode].get(("deterministic", "deterministic"), 0)],
            ])
            im = ax.imshow(data, cmap="YlOrRd", aspect="auto",
                          vmin=max(0, data[data > 0].min() - 2) if data.any() else 0,
                          vmax=data.max() + 1 if data.max() > 0 else 1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["stoch", "det"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["stoch", "det"])
            ax.set_xlabel("Transfer pulse")
            ax.set_ylabel("Fast pulse")
            ax.set_title(mode.replace("_", " ").title())
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                           fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle("E2: Pulse Type Factorial (Best F1)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "e2_pulse_heatmap.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: e2_pulse_heatmap.png")

    return grid


# ============================================================================
# VRC_K vs K plots
# ============================================================================

def analyze_vrc(base_dir, output_dir, experiment="eco_E1_main"):
    """Plot VRC_K vs K from carry-path window CSVs."""
    exp_dir = os.path.join(base_dir, experiment) if experiment else base_dir

    # Collect window data per method
    method_vrc = defaultdict(lambda: defaultdict(list))

    for tag in sorted(os.listdir(exp_dir)):
        csv_path = os.path.join(exp_dir, tag, "carry_path_window.csv")
        records = load_csv_records(csv_path)
        if not records:
            continue

        # Extract method name (strip seed suffix)
        method = tag.rsplit("_s", 1)[0] if "_s" in tag else tag

        for r in records:
            K = int(r["window_K"])
            vrc = float(r["VRC_K"])
            method_vrc[method][K].append(vrc)

    if not method_vrc:
        print("  No VRC window data found, skipping VRC plot")
        return

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(method_vrc)))

        for (method, k_data), color in zip(sorted(method_vrc.items()), colors):
            ks = sorted(k_data.keys())
            means = [np.mean(k_data[k]) for k in ks]
            stds = [np.std(k_data[k]) for k in ks]
            ax.errorbar(ks, means, yerr=stds, marker="o", label=method,
                       color=color, capsize=3)

        ax.set_xlabel("Window Size K")
        ax.set_ylabel("VRC_K (Vector Recovery Cosine)")
        ax.set_title("Windowed Recovery: VRC_K vs K")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "vrc_vs_k.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: vrc_vs_k.png")

    return dict(method_vrc)


# ============================================================================
# TTv1 transfer histograms
# ============================================================================

def analyze_transfers(base_dir, output_dir, experiment="eco_E1_main"):
    """Plot TTv1 transfer metric histograms."""
    exp_dir = os.path.join(base_dir, experiment) if experiment else base_dir

    all_e2e_cos = defaultdict(list)
    all_handoff_cos = defaultdict(list)

    for tag in sorted(os.listdir(exp_dir)):
        if "ttv1" not in tag:
            continue
        csv_path = os.path.join(exp_dir, tag, "carry_path_transfer.csv")
        records = load_csv_records(csv_path)
        if not records:
            continue

        method = tag.rsplit("_s", 1)[0] if "_s" in tag else tag
        for r in records:
            all_e2e_cos[method].append(float(r["EndToEndCos"]))
            all_handoff_cos[method].append(float(r["HandoffCos"]))

    if not all_e2e_cos:
        print("  No TTv1 transfer data found, skipping transfer plots")
        return

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for method, values in sorted(all_e2e_cos.items()):
            axes[0].hist(values, bins=50, alpha=0.5, label=method, density=True)
        axes[0].set_xlabel("EndToEnd Cosine")
        axes[0].set_ylabel("Density")
        axes[0].set_title("TTv1 End-to-End Transfer Cosine")
        axes[0].legend(fontsize=7)

        for method, values in sorted(all_handoff_cos.items()):
            axes[1].hist(values, bins=50, alpha=0.5, label=method, density=True)
        axes[1].set_xlabel("Handoff Cosine")
        axes[1].set_ylabel("Density")
        axes[1].set_title("TTv1 Handoff Cosine")
        axes[1].legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "ttv1_transfer_hist.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: ttv1_transfer_hist.png")


# ============================================================================
# Markdown report
# ============================================================================

def generate_report(e1_results, e2_grid, e3_dir, output_dir):
    """Generate markdown report."""
    lines = ["# ECO Comparison Experiment Report\n"]

    # E1 results
    lines.append("## E1: Main Comparison (4 epochs, 3 seeds)\n")
    lines.append("| Method | Mean F1 | Std | Seeds |")
    lines.append("|--------|---------|-----|-------|")
    if e1_results:
        for label, data in e1_results.items():
            vals = ", ".join(f"{v:.1f}" for v in data["values"])
            lines.append(f"| {label} | {data['mean']:.2f} | {data['std']:.2f} | {vals} |")
    lines.append("")

    # E2 results
    lines.append("## E2: TTv1 Pulse Factorial\n")
    if e2_grid:
        for mode, grid in e2_grid.items():
            lines.append(f"### {mode}\n")
            lines.append("| Fast \\ Transfer | Stochastic | Deterministic |")
            lines.append("|-----------------|------------|---------------|")
            for fp in ["stochastic", "deterministic"]:
                s = grid.get((fp, "stochastic"), 0)
                d = grid.get((fp, "deterministic"), 0)
                lines.append(f"| {fp} | {s:.2f} | {d:.2f} |")
            lines.append("")

    # E3 results
    lines.append("## E3: 14-bit Upper Bound\n")
    e3_summaries = load_summaries(e3_dir) if e3_dir else {}
    for tag, data in sorted(e3_summaries.items()):
        r = data["results"]
        lines.append(f"- **{tag}**: best_f1={r['best_f1']:.2f}, "
                     f"final_f1={r['final_f1']:.2f}, em={r['final_em']:.2f}")
    lines.append("")

    # Interpretation
    lines.append("## Key Findings\n")
    if e1_results:
        # Sort by mean F1
        ranked = sorted(e1_results.items(), key=lambda x: -x[1]["mean"])
        lines.append(f"1. **Best method**: {ranked[0][0]} (F1={ranked[0][1]['mean']:.2f})")

        eco_stoch = e1_results.get("ECO ref (stoch)", {})
        mp = e1_results.get("MixedPrecision", {})
        if eco_stoch.get("mean", 0) > 0 and mp.get("mean", 0) > 0:
            gap = eco_stoch["mean"] - mp["mean"]
            lines.append(f"2. **ECO-MixedPrecision gap**: {gap:+.2f} F1 points")
            if abs(gap) < 1.0:
                lines.append("   - MixedPrecision is a close surrogate for ECO")
            else:
                lines.append("   - Significant gap; MixedPrecision may not fully "
                            "capture ECO behavior")

        srpu_det = e1_results.get("SingleRPU (det)", {})
        srpu_stoch = e1_results.get("SingleRPU (stoch)", {})
        ttv1_hb = e1_results.get("TTv1 HiddenBuf", {})
        if (srpu_stoch.get("mean", 0) > 0 and ttv1_hb.get("mean", 0) > 0):
            improvement = ttv1_hb["mean"] - srpu_stoch["mean"]
            lines.append(f"3. **TTv1 vs SingleRPU**: {improvement:+.2f} F1 points "
                        f"(hidden_buffer vs stoch)")
    lines.append("")

    report_path = os.path.join(output_dir, "eco_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate ECO comparison report")
    parser.add_argument("--results-dir", type=str, default="results/paper",
                        help="Base results directory")
    parser.add_argument("--e1-dir", type=str, default=None)
    parser.add_argument("--e2-dir", type=str, default=None)
    parser.add_argument("--e3-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots and report")
    args = parser.parse_args()

    e1_dir = args.e1_dir or os.path.join(args.results_dir, "eco_E1_main")
    e2_dir = args.e2_dir or os.path.join(args.results_dir, "eco_E2_pulse")
    e3_dir = args.e3_dir or os.path.join(args.results_dir, "eco_E3_14bit")
    output_dir = args.output_dir or os.path.join(args.results_dir, "eco_report")

    os.makedirs(output_dir, exist_ok=True)

    print("=== Generating ECO Comparison Report ===")

    print("\nAnalyzing E1 (main comparison)...")
    e1_results = analyze_e1(e1_dir, output_dir)

    print("\nAnalyzing E2 (pulse factorial)...")
    e2_grid = analyze_e2(e2_dir, output_dir)

    print("\nAnalyzing VRC windows...")
    analyze_vrc(args.results_dir, output_dir)

    print("\nAnalyzing TTv1 transfers...")
    analyze_transfers(args.results_dir, output_dir)

    print("\nGenerating report...")
    generate_report(e1_results, e2_grid, e3_dir, output_dir)

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()

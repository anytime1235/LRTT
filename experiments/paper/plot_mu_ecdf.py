#!/usr/bin/env python
# coding=utf-8
"""Plot measured μ ECDF from D1 sub-pulse diagnostic results.

Reads mu_distribution.json from each run directory and generates
ECDF plots comparing actual measured μ distributions across
bit-widths, methods, and layers.

Usage:
    python plot_mu_ecdf.py [--results-dir DIR] [--output PATH] [--step STEP]

Requires: D1 diagnostic re-run with updated update_diagnostics.py
that saves mu_distribution.json (histogram of per-element μ values).
"""

import argparse
import os
import json
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================================
# Data loading
# ============================================================================

def load_mu_distribution(run_dir):
    """Load mu_distribution.json from a run directory.

    Returns None if file doesn't exist.
    """
    path = os.path.join(run_dir, "mu_distribution.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_histograms(data, step=None, tile_filter=None):
    """Sum histogram counts across tiles (and optionally filter by step).

    Args:
        data: Loaded mu_distribution.json dict.
        step: If set, only include this step. If None, include all steps.
        tile_filter: Optional function(tile_name) -> bool.

    Returns:
        (bin_edges, total_counts) numpy arrays.
    """
    bin_edges = np.array(data["bin_edges"])
    total_counts = np.zeros(len(bin_edges) - 1)

    for h in data["histograms"]:
        if step is not None and h["step"] != step:
            continue
        if tile_filter is not None and not tile_filter(h["tile_name"]):
            continue
        total_counts += np.array(h["counts"])

    return bin_edges, total_counts


def histograms_by_group(data, group_fn, step=None):
    """Group histograms by a key function.

    Args:
        data: Loaded mu_distribution.json dict.
        group_fn: function(histogram_entry) -> group_key string.
        step: If set, only include this step.

    Returns:
        dict {group_key: total_counts array}
    """
    bin_edges = np.array(data["bin_edges"])
    groups = {}
    for h in data["histograms"]:
        if step is not None and h["step"] != step:
            continue
        key = group_fn(h)
        if key not in groups:
            groups[key] = np.zeros(len(bin_edges) - 1)
        groups[key] += np.array(h["counts"])
    return groups


def counts_to_ecdf(bin_edges, counts):
    """Convert histogram counts to ECDF (x, y) arrays.

    Returns:
        x: right bin edges (μ values)
        y: cumulative fraction P(μ ≤ x)
    """
    cumulative = np.cumsum(counts)
    total = cumulative[-1]
    if total == 0:
        return bin_edges[1:], np.zeros_like(counts, dtype=float)
    return bin_edges[1:], cumulative / total


def frac_below_threshold(bin_edges, counts, threshold=1.0):
    """Compute fraction of elements with μ < threshold from histogram."""
    total = counts.sum()
    if total == 0:
        return 0.0
    mask = bin_edges[1:] <= threshold
    return counts[mask].sum() / total


def get_layer_index(tile_name):
    """Extract layer index from tile name."""
    m = re.search(r'layer\.(\d+)', tile_name)
    return int(m.group(1)) if m else -1


def get_subtype(tile_name):
    """Extract attention subtype from tile name."""
    if "query" in tile_name:
        return "Q"
    elif "key" in tile_name:
        return "K"
    elif "value" in tile_name:
        return "V"
    elif "dense" in tile_name:
        return "O"
    return "other"


# ============================================================================
# Plotting
# ============================================================================

BIT_COLORS = {8: '#d62728', 10: '#ff7f0e', 12: '#2ca02c', 14: '#1f77b4'}
BIT_LABELS = {8: '8b', 10: '10b', 12: '12b', 14: '14b'}
LAYER_COLORS = {0: '#1f77b4', 5: '#ff7f0e', 11: '#d62728'}


def plot_ecdf_by_bitwidth(ax, base_dir, method_prefix, bits_list,
                          step=None, title=None):
    """Plot ECDF curves for different bit-widths of one method.

    Returns dict of {bits: frac_mu_lt_1} for annotation.
    """
    fracs = {}
    for bits in bits_list:
        tag = f"{method_prefix}_{bits}b"
        run_dir = os.path.join(base_dir, tag)
        data = load_mu_distribution(run_dir)
        if data is None:
            continue

        bin_edges, counts = aggregate_histograms(data, step=step)
        x, y = counts_to_ecdf(bin_edges, counts)
        frac = frac_below_threshold(bin_edges, counts, 1.0)
        fracs[bits] = frac

        ax.plot(x, y, color=BIT_COLORS.get(bits, 'gray'),
                label=f"{bits}b (frac<1={frac:.3f})", linewidth=1.5)

    # μ=1 boundary
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(1.05, 0.5, 'μ=1', transform=ax.get_xaxis_transform(),
            fontsize=9, alpha=0.7)

    # Sub-pulse region shading
    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1e-6,
               1.0, alpha=0.05, color='red')

    ax.set_xscale('log')
    ax.set_xlabel('μ (expected pulse count per element)')
    ax.set_ylabel('P(μ ≤ x)')
    ax.set_xlim(1e-6, 1e4)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)

    return fracs


def plot_stoch_vs_det(ax, base_dir, step=None):
    """Plot stochastic vs deterministic comparison at 8b and 10b."""
    configs = [
        ("single_rpu_stoch_8b", "stoch 8b", '#d62728', '-'),
        ("single_rpu_det_8b", "det 8b", '#d62728', '--'),
        ("single_rpu_stoch_10b", "stoch 10b", '#ff7f0e', '-'),
        ("single_rpu_det_10b", "det 10b", '#ff7f0e', '--'),
    ]

    for tag, label, color, ls in configs:
        run_dir = os.path.join(base_dir, tag)
        data = load_mu_distribution(run_dir)
        if data is None:
            continue

        bin_edges, counts = aggregate_histograms(data, step=step)
        x, y = counts_to_ecdf(bin_edges, counts)
        frac = frac_below_threshold(bin_edges, counts, 1.0)
        ax.plot(x, y, color=color, linestyle=ls,
                label=f"{label} (frac<1={frac:.3f})", linewidth=1.5)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('μ')
    ax.set_ylabel('P(μ ≤ x)')
    ax.set_xlim(1e-6, 1e4)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Stochastic vs Deterministic (same μ distribution)', fontsize=10)
    ax.text(0.02, 0.15,
            'μ distribution is identical\n(same delta_target / dw_min)\n'
            'Difference is in pulse execution:\n'
            '  det: round(μ)=0 → dead\n'
            '  stoch: P(1)=μ → noisy but unbiased',
            transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


def plot_per_layer(ax, base_dir, bits, step=None):
    """Plot μ ECDF breakdown by layer index."""
    tag = f"single_rpu_stoch_{bits}b"
    run_dir = os.path.join(base_dir, tag)
    data = load_mu_distribution(run_dir)
    if data is None:
        ax.text(0.5, 0.5, f'No data: {tag}', transform=ax.transAxes,
                ha='center')
        return

    bin_edges = np.array(data["bin_edges"])
    groups = histograms_by_group(
        data,
        group_fn=lambda h: get_layer_index(h["tile_name"]),
        step=step,
    )

    for layer_idx in sorted(groups.keys()):
        counts = groups[layer_idx]
        x, y = counts_to_ecdf(bin_edges, counts)
        frac = frac_below_threshold(bin_edges, counts, 1.0)
        color = LAYER_COLORS.get(layer_idx, 'gray')
        ax.plot(x, y, color=color,
                label=f"Layer {layer_idx} (frac<1={frac:.3f})", linewidth=1.5)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('μ')
    ax.set_ylabel('P(μ ≤ x)')
    ax.set_xlim(1e-6, 1e4)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Per-layer μ ECDF ({bits}b stochastic)', fontsize=10)


def plot_per_subtype(ax, base_dir, bits, step=None):
    """Plot μ ECDF breakdown by attention subtype (Q/K/V/O)."""
    tag = f"single_rpu_stoch_{bits}b"
    run_dir = os.path.join(base_dir, tag)
    data = load_mu_distribution(run_dir)
    if data is None:
        ax.text(0.5, 0.5, f'No data: {tag}', transform=ax.transAxes,
                ha='center')
        return

    bin_edges = np.array(data["bin_edges"])
    subtype_colors = {'Q': '#1f77b4', 'K': '#ff7f0e', 'V': '#2ca02c', 'O': '#d62728'}

    groups = histograms_by_group(
        data,
        group_fn=lambda h: get_subtype(h["tile_name"]),
        step=step,
    )

    for subtype in ['Q', 'K', 'V', 'O']:
        if subtype not in groups:
            continue
        counts = groups[subtype]
        x, y = counts_to_ecdf(bin_edges, counts)
        frac = frac_below_threshold(bin_edges, counts, 1.0)
        ax.plot(x, y, color=subtype_colors.get(subtype, 'gray'),
                label=f"{subtype} (frac<1={frac:.3f})", linewidth=1.5)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('μ')
    ax.set_ylabel('P(μ ≤ x)')
    ax.set_xlim(1e-6, 1e4)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Per-subtype μ ECDF ({bits}b stochastic)', fontsize=10)


def plot_step_evolution(ax, base_dir, bits, method_prefix="single_rpu_stoch"):
    """Plot μ ECDF at different training steps to show evolution."""
    tag = f"{method_prefix}_{bits}b"
    run_dir = os.path.join(base_dir, tag)
    data = load_mu_distribution(run_dir)
    if data is None:
        ax.text(0.5, 0.5, f'No data: {tag}', transform=ax.transAxes,
                ha='center')
        return

    # Find available steps
    steps = sorted(set(h["step"] for h in data["histograms"]))
    step_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(steps)))

    bin_edges = np.array(data["bin_edges"])
    for i, s in enumerate(steps):
        _, counts = aggregate_histograms(data, step=s)
        x, y = counts_to_ecdf(bin_edges, counts)
        frac = frac_below_threshold(bin_edges, counts, 1.0)
        ax.plot(x, y, color=step_colors[i],
                label=f"step {s} (frac<1={frac:.3f})", linewidth=1.5)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('μ')
    ax.set_ylabel('P(μ ≤ x)')
    ax.set_xlim(1e-6, 1e4)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'μ ECDF evolution over training ({bits}b stoch)', fontsize=10)


def plot_frac_bar(ax, base_dir, step=None):
    """Bar chart of frac(μ<1) per bit-width, measured from actual ECDF."""
    methods = [
        ("single_rpu_stoch", [8, 10, 12, 14], "stoch"),
    ]

    bits_all = [8, 10, 12, 14]
    fracs = []
    labels = []

    for bits in bits_all:
        tag = f"single_rpu_stoch_{bits}b"
        run_dir = os.path.join(base_dir, tag)
        data = load_mu_distribution(run_dir)
        if data is None:
            fracs.append(0)
            labels.append(f"{bits}b")
            continue
        bin_edges, counts = aggregate_histograms(data, step=step)
        frac = frac_below_threshold(bin_edges, counts, 1.0)
        fracs.append(frac * 100)
        labels.append(f"{bits}b")

    colors = [BIT_COLORS.get(b, 'gray') for b in bits_all]
    bars = ax.bar(labels, fracs, color=colors, alpha=0.8, edgecolor='black')

    # Value labels on bars
    for bar, frac in zip(bars, fracs):
        if frac > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{frac:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('frac(μ < 1) [%]')
    ax.set_xlabel('Bit-width')
    ax.set_ylim(0, 105)
    ax.set_title('Measured sub-pulse fraction by bit-width', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot measured μ ECDF")
    parser.add_argument("--results-dir", type=str,
                        default="results/paper/diag_D1_subpulse",
                        help="Base directory containing run subdirectories")
    parser.add_argument("--output", type=str,
                        default="/root/mu_distribution_measured_ecdf.png",
                        help="Output plot path")
    parser.add_argument("--step", type=int, default=None,
                        help="Only plot data from this step (default: all steps)")
    parser.add_argument("--layer-bits", type=int, default=14,
                        help="Bit-width for per-layer breakdown (default: 14)")
    args = parser.parse_args()

    base_dir = args.results_dir

    # Check if any mu_distribution.json exists
    found = False
    for tag in os.listdir(base_dir) if os.path.isdir(base_dir) else []:
        if os.path.exists(os.path.join(base_dir, tag, "mu_distribution.json")):
            found = True
            break

    if not found:
        print("=" * 60)
        print("ERROR: No mu_distribution.json found in any run directory.")
        print(f"  Searched: {base_dir}/*/mu_distribution.json")
        print()
        print("The D1 experiment needs to be re-run with the updated")
        print("update_diagnostics.py that saves μ histograms.")
        print()
        print("Run:")
        print("  cd /root/LRTT/experiments/paper")
        print("  bash launchers/diag_D1_subpulse.sh")
        print("=" * 60)
        return

    # Create figure: 3×2 layout
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Measured μ Distribution — D1 Sub-pulse Diagnostic',
                 fontsize=14, fontweight='bold', y=0.98)

    step_label = f" (step {args.step})" if args.step else " (all steps)"

    # (0,0): Main ECDF per bit-width
    plot_ecdf_by_bitwidth(
        axes[0, 0], base_dir, "single_rpu_stoch", [8, 10, 12, 14],
        step=args.step,
        title=f'Measured μ ECDF — single_rpu stochastic{step_label}')

    # (0,1): Stoch vs Det
    plot_stoch_vs_det(axes[0, 1], base_dir, step=args.step)

    # (1,0): Per-layer breakdown
    plot_per_layer(axes[1, 0], base_dir, args.layer_bits, step=args.step)

    # (1,1): Per-subtype breakdown
    plot_per_subtype(axes[1, 1], base_dir, args.layer_bits, step=args.step)

    # (2,0): Step evolution at 8b
    plot_step_evolution(axes[2, 0], base_dir, 8)

    # (2,1): frac(μ<1) bar chart
    plot_frac_bar(axes[2, 1], base_dir, step=args.step)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

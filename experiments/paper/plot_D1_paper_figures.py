#!/usr/bin/env python3
"""D1 Sub-pulse Regime — Paper Figures

Main Figure (2-panel):
  (a) μ ECDF by bit-width (stochastic 8/10/12/14b)
  (b) Update quality vs bit-width: cosine similarity + recovery ratio

Supplementary Figures (4 panels):
  S1. Stochastic vs Deterministic comparison
  S2. Per-layer & per-subtype μ breakdown (14b)
  S3. Actual vs Target norm per tile
  S4. Training loss curves
"""

import json
import os
import csv
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# ── Paths ──
BASE = "/root/LRTT/experiments/paper/results/paper/diag_D1_subpulse"
OUT_DIR = "/root"

# ── Style ──
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

BIT_COLORS = {6: "#9467bd", 8: "#d62728", 10: "#ff7f0e", 12: "#2ca02c", 14: "#1f77b4"}
BIT_MARKERS = {6: "^", 8: "o", 10: "s", 12: "D", 14: "v"}


# ═══════════════════════════════════════════════════════════════════
#  Data loaders
# ═══════════════════════════════════════════════════════════════════

def load_diagnostics_summary(tag):
    path = os.path.join(BASE, tag, "update_diagnostics_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_mu_distribution(tag):
    path = os.path.join(BASE, tag, "mu_distribution.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_training_log(tag):
    path = os.path.join(BASE, tag, "training_log.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read().strip()
    if content.count("\n") < 1:
        return None
    rows = []
    for r in csv.DictReader(content.split("\n")):
        rows.append(r)
    return rows


def aggregate_metrics(tag):
    """Aggregate key metrics across all layers for a given experiment."""
    diag = load_diagnostics_summary(tag)
    if diag is None:
        return None
    layers = list(diag.keys())
    return {
        "mu_p50": np.mean([diag[l]["mean_eff_mu_p50"] for l in layers]),
        "mu_mean": np.mean([diag[l]["mean_eff_mu_mean"] for l in layers]),
        "mu_max": np.max([diag[l]["max_eff_mu_max"] for l in layers]),
        "frac_lt1": np.mean([diag[l]["mean_eff_frac_mu_lt_1"] for l in layers]),
        "cosine": np.mean([diag[l]["mean_cosine_sim"] for l in layers]),
        "zero_frac": np.mean([diag[l]["mean_zero_frac"] for l in layers]),
        "recovery": np.mean([diag[l]["final_recovery_ratio"] for l in layers]),
        "nsr": np.mean([diag[l]["final_cum_nsr"] for l in layers]),
        "actual_norm": np.mean([diag[l]["mean_actual_norm"] for l in layers]),
        "target_norm": np.mean([diag[l]["mean_target_norm"] for l in layers]),
        "residual_norm": np.mean([diag[l]["mean_residual_norm"] for l in layers]),
    }


def mu_ecdf(data, step=None, tile_filter=None):
    """Compute ECDF from mu_distribution.json."""
    bin_edges = np.array(data["bin_edges"])
    total_counts = np.zeros(len(bin_edges) - 1)
    for h in data["histograms"]:
        if step is not None and h["step"] != step:
            continue
        if tile_filter is not None and not tile_filter(h["tile_name"]):
            continue
        total_counts += np.array(h["counts"])
    cumulative = np.cumsum(total_counts)
    total = cumulative[-1]
    if total == 0:
        return bin_edges[1:], np.zeros_like(total_counts)
    return bin_edges[1:], cumulative / total


def frac_below(data, threshold=1.0, step=None):
    bin_edges = np.array(data["bin_edges"])
    total_counts = np.zeros(len(bin_edges) - 1)
    for h in data["histograms"]:
        if step is not None and h["step"] != step:
            continue
        total_counts += np.array(h["counts"])
    total = total_counts.sum()
    if total == 0:
        return 0.0
    mask = bin_edges[1:] <= threshold
    return total_counts[mask].sum() / total


def get_subtype(name):
    if "query" in name:
        return "Q"
    elif "key" in name:
        return "K"
    elif "value" in name:
        return "V"
    elif "dense" in name:
        return "O"
    return "?"


def get_layer_idx(name):
    m = re.search(r"layer\.(\d+)", name)
    return int(m.group(1)) if m else -1


# ═══════════════════════════════════════════════════════════════════
#  MAIN FIGURE: 2-panel
# ═══════════════════════════════════════════════════════════════════

def plot_main_figure():
    fig, (ax_ecdf, ax_quality) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── (a) μ ECDF by bit-width ──
    bits_list = [8, 10, 12, 14]
    for bits in bits_list:
        data = load_mu_distribution(f"single_rpu_stoch_{bits}b")
        if data is None:
            continue
        x, y = mu_ecdf(data)
        frac = frac_below(data, 1.0)
        ax_ecdf.plot(x, y, color=BIT_COLORS[bits], linewidth=1.8,
                     label=f"{bits}b  (P(μ<1)={frac:.3f})")

    # μ=1 boundary
    ax_ecdf.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_ecdf.axvspan(1e-6, 1.0, alpha=0.06, color="red")
    ax_ecdf.text(0.08, 0.55, "sub-pulse\nregion",
                 transform=ax_ecdf.transAxes, fontsize=9, color="#c0392b",
                 fontstyle="italic", alpha=0.8)
    ax_ecdf.text(1.3, 0.5, "μ = 1", fontsize=9, alpha=0.6,
                 rotation=90, va="center")

    ax_ecdf.set_xscale("log")
    ax_ecdf.set_xlabel("μ  (expected pulse count per element)")
    ax_ecdf.set_ylabel("P(μ ≤ x)")
    ax_ecdf.set_xlim(1e-5, 1e3)
    ax_ecdf.set_ylim(0, 1.05)
    ax_ecdf.legend(loc="lower right", framealpha=0.9)
    ax_ecdf.grid(True, alpha=0.25)
    ax_ecdf.set_title("(a)  μ ECDF — Single RPU Stochastic", fontweight="bold")

    # ── (b) Update quality vs bit-width ──
    cosines, recoveries, bits_plot = [], [], []
    for bits in bits_list:
        m = aggregate_metrics(f"single_rpu_stoch_{bits}b")
        if m is None:
            continue
        bits_plot.append(bits)
        cosines.append(m["cosine"])
        recoveries.append(m["recovery"])

    # Left y-axis: cosine similarity
    color_cos = "#2980b9"
    ax_quality.set_xlabel("Bit-width")
    ax_quality.set_ylabel("Cosine Similarity (direction fidelity)", color=color_cos)
    ln1 = ax_quality.plot(bits_plot, cosines, "o-", color=color_cos, linewidth=2,
                          markersize=8, label="Cosine Similarity", zorder=3)
    ax_quality.tick_params(axis="y", labelcolor=color_cos)
    ax_quality.set_ylim(0, 0.35)
    ax_quality.axhline(1.0, color=color_cos, linestyle=":", alpha=0.3)

    # Right y-axis: recovery ratio (log)
    ax_r = ax_quality.twinx()
    color_rec = "#e74c3c"
    ax_r.set_ylabel("Recovery Ratio  (‖actual‖ / ‖target‖)", color=color_rec)
    ln2 = ax_r.plot(bits_plot, recoveries, "s--", color=color_rec, linewidth=2,
                    markersize=8, label="Recovery Ratio", zorder=3)
    ax_r.tick_params(axis="y", labelcolor=color_rec)
    ax_r.set_yscale("log")
    ax_r.set_ylim(1, 100)
    ax_r.axhline(1.0, color=color_rec, linestyle=":", alpha=0.3)

    # Ideal annotations
    ax_quality.annotate("ideal: cos → 1", xy=(13.5, 0.32), fontsize=7.5,
                        color=color_cos, alpha=0.6, fontstyle="italic")
    ax_r.annotate("ideal: recovery → 1", xy=(12.5, 1.3), fontsize=7.5,
                  color=color_rec, alpha=0.6, fontstyle="italic")

    ax_quality.set_xticks(bits_plot)
    ax_quality.grid(True, alpha=0.25)
    ax_quality.set_title("(b)  Update Quality vs Bit-width (Stochastic)",
                         fontweight="bold")

    # Combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax_quality.legend(lines, labels, loc="center right", framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "D1_main_figure.png")
    fig.savefig(out_path)
    plt.close()
    print(f"Main figure saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY FIGURE: 4-panel
# ═══════════════════════════════════════════════════════════════════

def plot_supplementary():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── S1: Stochastic vs Deterministic ──
    ax = axes[0, 0]
    configs = [
        ("single_rpu_stoch_8b",  "Stoch 8b",  BIT_COLORS[8],  "-"),
        ("single_rpu_det_8b",    "Det 8b",    BIT_COLORS[8],  "--"),
        ("single_rpu_stoch_14b", "Stoch 14b", BIT_COLORS[14], "-"),
        ("single_rpu_det_14b",   "Det 14b",   BIT_COLORS[14], "--"),
    ]
    for tag, label, color, ls in configs:
        data = load_mu_distribution(tag)
        if data is None:
            continue
        x, y = mu_ecdf(data)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=1.5, label=label)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvspan(1e-6, 1.0, alpha=0.05, color="red")
    ax.set_xscale("log")
    ax.set_xlabel("μ")
    ax.set_ylabel("P(μ ≤ x)")
    ax.set_xlim(1e-5, 1e3)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.set_title("(S1)  Stoch vs Det: same μ, different execution", fontweight="bold")

    # Annotation box
    ax.text(0.02, 0.22,
            "μ distributions are identical\n"
            "(same Δtarget / dw_min)\n\n"
            "Det: floor(μ)=0 → update=0\n"
            "Stoch: P(pulse=1)=μ → noisy",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

    # ── S2: Per-subtype μ breakdown (14b stoch) ──
    ax = axes[0, 1]
    data_14 = load_mu_distribution("single_rpu_stoch_14b")
    if data_14 is not None:
        subtype_colors = {"Q": "#1f77b4", "K": "#ff7f0e", "V": "#2ca02c", "O": "#d62728"}
        bin_edges = np.array(data_14["bin_edges"])

        for subtype in ["Q", "K", "V", "O"]:
            counts = np.zeros(len(bin_edges) - 1)
            for h in data_14["histograms"]:
                if get_subtype(h["tile_name"]) == subtype:
                    counts += np.array(h["counts"])
            cumulative = np.cumsum(counts)
            total = cumulative[-1]
            if total > 0:
                frac = counts[bin_edges[1:] <= 1.0].sum() / total
                ax.plot(bin_edges[1:], cumulative / total,
                        color=subtype_colors[subtype], linewidth=1.5,
                        label=f"{subtype}  (P(μ<1)={frac:.3f})")

        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("μ")
        ax.set_ylabel("P(μ ≤ x)")
        ax.set_xlim(1e-5, 1e3)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)
    ax.set_title("(S2)  Per-subtype μ ECDF (14b Stochastic)", fontweight="bold")

    # ── S3: Actual vs Target norm per bit-width ──
    ax = axes[1, 0]
    bits_list = [8, 10, 12, 14]

    # Per-tile data for stochastic
    x_pos = np.arange(len(bits_list))
    width = 0.3
    target_means, actual_means = [], []
    for bits in bits_list:
        m = aggregate_metrics(f"single_rpu_stoch_{bits}b")
        if m:
            target_means.append(m["target_norm"])
            actual_means.append(m["actual_norm"])
        else:
            target_means.append(0)
            actual_means.append(0)

    bars1 = ax.bar(x_pos - width / 2, target_means, width, label="Target norm (ideal FP32)",
                   color="#3498db", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x_pos + width / 2, actual_means, width, label="Actual norm (pulsed)",
                   color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b}b" for b in bits_list])
    ax.set_xlabel("Bit-width")
    ax.set_ylabel("Mean update norm (Frobenius)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_title("(S3)  Target vs Actual Update Norm (Stochastic)", fontweight="bold")

    # Add ratio annotations
    for i, (t, a) in enumerate(zip(target_means, actual_means)):
        if t > 0:
            ratio = a / t
            ax.text(i, max(a, t) * 1.5, f"{ratio:.0f}×",
                    ha="center", fontsize=8, fontweight="bold", color="#c0392b")

    # ── S4: Training loss curves ──
    ax = axes[1, 1]
    for mode, ls in [("stoch", "-"), ("det", "--")]:
        for bits in bits_list:
            tag = f"single_rpu_{mode}_{bits}b"
            rows = load_training_log(tag)
            if rows is None:
                continue
            steps = [int(r["step"]) for r in rows]
            losses = [float(r["loss"]) for r in rows]
            label = f"{mode} {bits}b"
            ax.plot(steps, losses, linestyle=ls, color=BIT_COLORS[bits],
                    linewidth=1.3, alpha=0.85, label=label)

    # Custom legend: separate stoch/det
    stoch_line = Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5)
    det_line = Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5)
    handles = [stoch_line, det_line]
    labels = ["Stochastic", "Deterministic"]
    for bits in bits_list:
        handles.append(Line2D([0], [0], color=BIT_COLORS[bits], marker="o",
                              linestyle="None", markersize=5))
        labels.append(f"{bits}b")

    ax.legend(handles, labels, fontsize=7, ncol=2, loc="upper right")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_ylim(1.5, 6.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("(S4)  Training Loss: Stoch (—) vs Det (--)", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "D1_supplementary_figures.png")
    fig.savefig(out_path)
    plt.close()
    print(f"Supplementary figure saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    plot_main_figure()
    plot_supplementary()
    print("Done.")

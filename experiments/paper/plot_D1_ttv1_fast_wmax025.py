#!/usr/bin/env python3
"""D1 TTv1 Fast Tile Sub-pulse — μ ECDF Figure

Main Figure (2-panel):
  (a) μ ECDF by fast tile bit-width (TTv1 w_max_fast=0.25, 8/10/12/14b)
  (b) Update quality vs bit-width: cosine similarity + recovery ratio

Comparable to D1_main_figure.png but for TTv1 fast tile.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──
BASE = "/root/LRTT/experiments/paper/results/paper/diag_D1_ttv1_fast_wmax025"
OUT_DIR = "/root"

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

BIT_COLORS = {8: "#d62728", 10: "#ff7f0e", 12: "#2ca02c", 14: "#1f77b4"}


def load_mu_distribution(tag):
    path = os.path.join(BASE, tag, "mu_distribution.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_diagnostics_summary(tag):
    path = os.path.join(BASE, tag, "update_diagnostics_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def mu_ecdf(data, step=None, tile_filter=None):
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


def aggregate_metrics(tag):
    diag = load_diagnostics_summary(tag)
    if diag is None:
        return None
    layers = list(diag.keys())
    return {
        "mu_p50": np.mean([diag[l]["mean_eff_mu_p50"] for l in layers]),
        "frac_lt1": np.mean([diag[l]["mean_eff_frac_mu_lt_1"] for l in layers]),
        "cosine": np.mean([diag[l]["mean_cosine_sim"] for l in layers]),
        "recovery": np.mean([diag[l]["final_recovery_ratio"] for l in layers]),
    }


def plot_main_figure():
    fig, (ax_ecdf, ax_quality) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── (a) μ ECDF by bit-width ──
    bits_list = [8, 10, 12, 14]
    for bits in bits_list:
        data = load_mu_distribution(f"ttv1_fast{bits}b_wmax025")
        if data is None:
            print(f"  [SKIP] ttv1_fast{bits}b_wmax025: no mu_distribution.json")
            continue
        x, y = mu_ecdf(data)
        frac = frac_below(data, 1.0)
        dw_min = 2 * 0.25 / (2 ** bits)
        ax_ecdf.plot(x, y, color=BIT_COLORS[bits], linewidth=1.8,
                     label=f"{bits}b  dw={dw_min:.1e}  P(μ<1)={frac:.3f}")

    # μ=1 boundary
    ax_ecdf.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_ecdf.axvspan(1e-6, 1.0, alpha=0.06, color="red")
    ax_ecdf.text(0.08, 0.55, "sub-pulse\nregion",
                 transform=ax_ecdf.transAxes, fontsize=9, color="#c0392b",
                 fontstyle="italic", alpha=0.8)

    ax_ecdf.set_xscale("log")
    ax_ecdf.set_xlabel("μ  (expected pulse count per element)")
    ax_ecdf.set_ylabel("P(μ ≤ x)")
    ax_ecdf.set_xlim(1e-5, 1e3)
    ax_ecdf.set_ylim(0, 1.05)
    ax_ecdf.legend(loc="lower right", framealpha=0.9)
    ax_ecdf.grid(True, alpha=0.25)
    ax_ecdf.set_title("(a)  μ ECDF — TTv1 Fast Tile (w_max=0.25)", fontweight="bold")

    # ── (b) Update quality vs bit-width ──
    cosines, recoveries, bits_plot = [], [], []
    for bits in bits_list:
        m = aggregate_metrics(f"ttv1_fast{bits}b_wmax025")
        if m is None:
            continue
        bits_plot.append(bits)
        cosines.append(m["cosine"])
        recoveries.append(m["recovery"])

    if bits_plot:
        color_cos = "#2980b9"
        ax_quality.set_xlabel("Fast Tile Bit-width")
        ax_quality.set_ylabel("Cosine Similarity", color=color_cos)
        ln1 = ax_quality.plot(bits_plot, cosines, "o-", color=color_cos, linewidth=2,
                              markersize=8, label="Cosine Similarity", zorder=3)
        ax_quality.tick_params(axis="y", labelcolor=color_cos)
        ax_quality.set_ylim(0, max(cosines) * 1.3 if cosines else 0.5)

        ax_r = ax_quality.twinx()
        color_rec = "#e74c3c"
        ax_r.set_ylabel("Recovery Ratio", color=color_rec)
        ln2 = ax_r.plot(bits_plot, recoveries, "s--", color=color_rec, linewidth=2,
                        markersize=8, label="Recovery Ratio", zorder=3)
        ax_r.tick_params(axis="y", labelcolor=color_rec)
        ax_r.set_yscale("log")
        ax_r.axhline(1.0, color=color_rec, linestyle=":", alpha=0.3)

        ax_quality.set_xticks(bits_plot)
        ax_quality.grid(True, alpha=0.25)
        ax_quality.set_title("(b)  Update Quality — TTv1 Fast Tile (w_max=0.25)",
                             fontweight="bold")

        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax_quality.legend(lines, labels, loc="center right", framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "D1_ttv1_fast_wmax025_figure.png")
    fig.savefig(out_path)
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    plot_main_figure()
    print("Done.")

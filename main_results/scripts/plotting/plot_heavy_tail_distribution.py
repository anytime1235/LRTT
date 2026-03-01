"""plot_heavy_tail_distribution.py — Visualize gradient heavy-tail distribution per sublayer.

Uses SQuAD seed_42 data:
  - NPZ: per-vector absmax arrays → ECDF of absmax (inter-vector spread)
  - CSV: ratio quantiles (q50, q90, q99) → intra-vector element distribution

Outputs:
  1. 2x3 ECDF of per-vector absmax (log-x) for each sublayer, all 12 layers
  2. 2x3 intra-vector ratio box/quantile plot (element/absmax distribution)
  3. Combined summary: ODR vs QZR scatter with sublayer color

Usage:
  python plot_heavy_tail_distribution.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SQUAD_NPZ = "/data/main_results/results/squad/seed_42/absmax_raw_A_baseline_7b.npz"
SQUAD_CSV = "/data/main_results/results/squad/seed_42/metrics_A_rootcause_summary.csv"
COLA_NPZ = "/data/main_results/results/glue/cola/seed_42/absmax_raw_A_baseline.npz"
COLA_CSV = "/data/main_results/results/glue/cola/seed_42/metrics_A_rootcause_summary.csv"
OUT_DIR = "/data/main_results/results/figures/heavy_tail"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
}
plt.rcParams.update(RCPARAMS)

SUBLAYERS = ["FFN1", "K", "Q", "V", "O", "FFN2"]
SUBLAYER_COLORS = {
    "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
    "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B",
}
N_LAYERS = 12
LAYER_CMAP = plt.cm.viridis
LAYER_COLORS = [LAYER_CMAP(i / (N_LAYERS - 1)) for i in range(N_LAYERS)]


def _save(fig, basename):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"{basename}.{ext}"),
                    dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR}/{basename}.{{pdf,png}}")


def ecdf(arr, max_points=3000):
    """Compute ECDF, subsampled for plotting."""
    s = np.sort(arr)
    n = len(s)
    y = np.arange(1, n + 1) / n
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        return s[idx], y[idx]
    return s, y


# ---------------------------------------------------------------------------
# Figure 1: Per-vector absmax ECDF (log-x) — inter-vector spread
# ---------------------------------------------------------------------------
def fig1_absmax_ecdf(npz_path, task_name):
    """2x3 grid: ECDF of per-vector absmax for each sublayer, 12 layers overlaid."""
    npz = np.load(npz_path)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{task_name} — Per-Vector Absmax ECDF (log scale)\n"
                 f"Heavy tail = wide spread between layers/vectors",
                 fontsize=13, fontweight="bold")

    for si, sub in enumerate(SUBLAYERS):
        ax = axes[si // 3, si % 3]
        for layer in range(N_LAYERS):
            key = f"L{layer}_{sub}"
            if key not in npz:
                continue
            arr = npz[key]
            arr = arr[arr > 0]  # exclude zero vectors
            if len(arr) == 0:
                continue
            x, y = ecdf(arr)
            ax.plot(x, y, color=LAYER_COLORS[layer], alpha=0.7,
                    linewidth=1.2, label=f"L{layer}")

        ax.set_xscale("log")
        ax.set_xlabel("absmax (log)")
        ax.set_ylabel("ECDF")
        ax.set_title(f"{sub}", fontweight="bold",
                     color=SUBLAYER_COLORS[sub])
        ax.grid(True, alpha=0.3)
        if si == 0:
            ax.legend(ncol=3, fontsize=6, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"{task_name.lower()}_absmax_ecdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Intra-vector ratio distribution (element/absmax)
# ---------------------------------------------------------------------------
def fig2_ratio_quantiles(csv_path, task_name):
    """2x3 grid: ratio quantiles (q50, q90, q99) per layer, showing intra-vector spread.

    Lower ratio_q50 = heavier tail (median element much smaller than max).
    """
    df = pd.read_csv(csv_path)
    df = df[df["layer_idx"] <= 10]  # exclude L11 for CoLA

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{task_name} — Intra-Vector Element Distribution (element/absmax)\n"
                 f"Lower ratio = more extreme outliers, heavier tail",
                 fontsize=13, fontweight="bold")

    # Quantization threshold line
    zero_thresh = 1.0 / 126  # 7-bit: 1/(2^7 - 2)

    for si, sub in enumerate(SUBLAYERS):
        ax = axes[si // 3, si % 3]
        sub_df = df[df["sublayer"] == sub].sort_values("layer_idx")
        layers = sub_df["layer_idx"].values

        q50 = sub_df["ratio_q50"].values
        q90 = sub_df["ratio_q90"].values
        q99 = sub_df["ratio_q99"].values

        # Plot as filled area between quantiles
        ax.fill_between(layers, q50, q99, alpha=0.15,
                         color=SUBLAYER_COLORS[sub])
        ax.fill_between(layers, q50, q90, alpha=0.25,
                         color=SUBLAYER_COLORS[sub])
        ax.plot(layers, q99, "^-", color=SUBLAYER_COLORS[sub],
                markersize=5, label="p99", alpha=0.8)
        ax.plot(layers, q90, "s-", color=SUBLAYER_COLORS[sub],
                markersize=5, label="p90", alpha=0.8)
        ax.plot(layers, q50, "o-", color=SUBLAYER_COLORS[sub],
                markersize=6, label="median", linewidth=2)

        # Zero threshold
        ax.axhline(zero_thresh, color="red", linestyle="--",
                    linewidth=1.5, alpha=0.7, label=f"7b threshold ({zero_thresh:.4f})")

        ax.set_xlabel("Layer")
        ax.set_ylabel("ratio (element / absmax)")
        ax.set_title(f"{sub}  (ODR={sub_df['ODR'].mean():.0f})",
                     fontweight="bold", color=SUBLAYER_COLORS[sub])
        ax.set_xticks(range(11))
        ax.set_ylim(-0.02, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"{task_name.lower()}_ratio_quantiles")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Absmax histogram comparison (log-scale) — all sublayers overlaid
# ---------------------------------------------------------------------------
def fig3_absmax_histogram_overlay(npz_path, task_name):
    """Single plot: log-scale histogram of absmax for all 6 sublayers (L0-10 pooled)."""
    npz = np.load(npz_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"{task_name} — Absmax Distribution by Sublayer (L0-10 pooled)",
                 fontsize=13, fontweight="bold")

    for sub in SUBLAYERS:
        all_vals = []
        for layer in range(11):
            key = f"L{layer}_{sub}"
            if key in npz:
                arr = npz[key]
                all_vals.append(arr[arr > 0])
        if not all_vals:
            continue
        pooled = np.concatenate(all_vals)
        log_vals = np.log10(pooled)
        ax.hist(log_vals, bins=100, alpha=0.4, density=True,
                color=SUBLAYER_COLORS[sub], label=f"{sub} (n={len(pooled):,})")

    ax.set_xlabel("log10(absmax)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f"{task_name.lower()}_absmax_histogram")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: ODR vs QZR scatter with ratio_q50 annotation
# ---------------------------------------------------------------------------
def fig4_odr_qzr_scatter(csv_path, task_name):
    """Scatter: log(ODR) vs QZR_nonzero, colored by sublayer, sized by layer."""
    df = pd.read_csv(csv_path)
    df = df[df["layer_idx"] <= 10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{task_name} — ODR vs QZR Correlation (L0-10)",
                 fontsize=13, fontweight="bold")

    # Left: ODR vs QZR_nonzero
    ax = axes[0]
    for sub in SUBLAYERS:
        s = df[df["sublayer"] == sub]
        ax.scatter(s["ODR"], s["QZR_nonzero"],
                   c=SUBLAYER_COLORS[sub], label=sub,
                   s=50 + s["layer_idx"] * 8, alpha=0.7,
                   edgecolors="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("ODR (log scale)")
    ax.set_ylabel("QZR_nonzero")
    ax.set_title("(a) log(ODR) vs QZR_nonzero")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: ratio_q50 vs QZR_nonzero
    ax = axes[1]
    for sub in SUBLAYERS:
        s = df[df["sublayer"] == sub]
        ax.scatter(s["ratio_q50"], s["QZR_nonzero"],
                   c=SUBLAYER_COLORS[sub], label=sub,
                   s=50 + s["layer_idx"] * 8, alpha=0.7,
                   edgecolors="white", linewidth=0.5)
    ax.set_xlabel("ratio_q50 (median element / absmax)")
    ax.set_ylabel("QZR_nonzero")
    ax.set_title("(b) Median ratio vs QZR_nonzero")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add threshold line
    zero_thresh = 1.0 / 126
    axes[1].axvline(zero_thresh, color="red", linestyle="--",
                     alpha=0.5, label="7b threshold")

    plt.tight_layout()
    _save(fig, f"{task_name.lower()}_odr_qzr_scatter")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== SQuAD ===")
    print("  Fig 1: Absmax ECDF")
    fig1_absmax_ecdf(SQUAD_NPZ, "SQuAD")
    print("  Fig 2: Ratio quantiles")
    fig2_ratio_quantiles(SQUAD_CSV, "SQuAD")
    print("  Fig 3: Absmax histogram")
    fig3_absmax_histogram_overlay(SQUAD_NPZ, "SQuAD")
    print("  Fig 4: ODR vs QZR scatter")
    fig4_odr_qzr_scatter(SQUAD_CSV, "SQuAD")

    print("\n=== CoLA ===")
    print("  Fig 1: Absmax ECDF")
    fig1_absmax_ecdf(COLA_NPZ, "CoLA")
    print("  Fig 2: Ratio quantiles")
    fig2_ratio_quantiles(COLA_CSV, "CoLA")
    print("  Fig 3: Absmax histogram")
    fig3_absmax_histogram_overlay(COLA_NPZ, "CoLA")
    print("  Fig 4: ODR vs QZR scatter")
    fig4_odr_qzr_scatter(COLA_CSV, "CoLA")

    print("\nDone.")


if __name__ == "__main__":
    main()

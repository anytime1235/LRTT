"""comprehensive_qzr_odr_analysis.py — Complete QZR/ODR/ECDF diagnostic analysis.

Generates 7 publication-quality diagnostic figures:
  1. QZR Complete Heatmap Grid (2×3, all 6 sublayers, bits×layer)
  2. ODR Complete Heatmap Grid (2×3, all 6 sublayers, bits×layer, log10)
  3. QZR vs Bits Line Plots (2×3, per sublayer, 12 layers as lines)
  4. QZR-ODR Correlation Scatter (2×3, per bit config)
  5. QZR-ODR Correlation Coefficient vs Bits (1×2)
  6. ECDF Summary — Worst Sublayers across Bits (2×3)
  7. Solution Effectiveness — QZR & ODR Delta Heatmaps (2×2)

Usage:
  python comprehensive_qzr_odr_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_DIR = "/data/main_results/results/csv"
NPZ_DIR = "/data/main_results/results/npz"
OUT_DIR = "/data/main_results/results/figures/diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LAYERS = 12
SUBLAYERS = ["Q", "K", "V", "O", "FFN1", "FFN2"]
BITS_ORDER = [4, 6, 7, 8, 10, 12]

RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
}
plt.rcParams.update(RCPARAMS)

LAYER_CMAP = plt.cm.coolwarm
LAYER_COLORS = [LAYER_CMAP(i / (N_LAYERS - 1)) for i in range(N_LAYERS)]

BITS_COLORS = {4: "#e41a1c", 6: "#ff7f00", 7: "#333333",
               8: "#4daf4a", 10: "#377eb8", 12: "#984ea3"}

SUBLAYER_COLORS = {
    "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
    "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B",
}

SUBLAYER_MARKERS = {
    "Q": "o", "K": "s", "V": "D", "O": "^", "FFN1": "v", "FFN2": "P",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, basename):
    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"{basename}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR}/{basename}.{{pdf,png}}")


def annotate_heatmap(ax, mat, fmt=".3f", fs=7, threshold=0.3):
    """Add text annotations to heatmap cells."""
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            val = mat[ri, ci]
            if np.isnan(val):
                continue
            vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
            norm = (val - vmin) / (vmax - vmin + 1e-12)
            color = "white" if norm > 0.55 else "black"
            ax.text(ci, ri, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=fs, color=color, fontweight="bold")


def ecdf(arr, max_points=4000):
    s = np.sort(arr)
    y = np.arange(1, len(s) + 1) / len(s)
    if len(s) > max_points:
        idx = np.linspace(0, len(s) - 1, max_points, dtype=int)
        return s[idx], y[idx]
    return s, y


def load_data():
    """Load all three CSV datasets and merge B + A for complete bit coverage."""
    df_a = pd.read_csv(os.path.join(CSV_DIR, "metrics_paper_A_rootcause_summary.csv"))
    df_b = pd.read_csv(os.path.join(CSV_DIR, "metrics_paper_B_bitsweep_summary.csv"))
    df_c = pd.read_csv(os.path.join(CSV_DIR, "metrics_paper_C_solutions_summary.csv"))

    # Build unified bit-sweep DataFrame: B already contains all 6 bit configs
    # Check if 7b baseline is in B; if not, add from A
    bits_in_b = sorted(df_b["dac_bits"].unique())
    if 7 not in bits_in_b:
        df_ab = pd.concat([df_b, df_a], ignore_index=True)
    else:
        df_ab = df_b.copy()

    return df_a, df_b, df_c, df_ab


def build_matrix(df, sublayer, metric, bits_list=None):
    """Build bits×layer matrix for a single sublayer.

    Returns (matrix, bits_list).
    """
    sub = df[df["sublayer"] == sublayer]
    if bits_list is None:
        bits_list = sorted(sub["dac_bits"].unique())
    mat = np.full((len(bits_list), N_LAYERS), np.nan)
    for bi, b in enumerate(bits_list):
        for li in range(N_LAYERS):
            rows = sub[(sub["dac_bits"] == b) & (sub["layer_idx"] == li)]
            if len(rows) > 0:
                mat[bi, li] = rows[metric].mean()
    return mat, bits_list


# ===========================================================================
# Figure 1: QZR Complete Heatmap Grid (2×3)
# ===========================================================================
def fig1_qzr_heatmap_grid(df_ab):
    print("\n[Fig1] QZR Complete Heatmap Grid (all 6 sublayers) ...")

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(
        "QZR (Quantization Zero Rate) — All Sublayers × Bit Resolutions\n"
        "Fraction of non-zero gradient elements quantized to 0 after AbsMax NM + DAC",
        fontsize=13, y=1.03,
    )

    for si, sl in enumerate(SUBLAYERS):
        ax = axes[si // 3, si % 3]
        mat, bits_list = build_matrix(df_ab, sl, "QZR_nonzero", BITS_ORDER)

        im = ax.imshow(mat, aspect="auto", cmap="plasma", origin="upper",
                       vmin=0.0, vmax=0.8)
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(bits_list)))
        ax.set_yticklabels([f"{b}b" for b in bits_list], fontsize=9)
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("DAC Bits")
        ax.set_title(f"({chr(97+si)}) {sl}", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85, label="QZR_nonzero")
        annotate_heatmap(ax, mat, fmt=".2f", fs=6, threshold=0.3)

    plt.tight_layout()
    _save(fig, "qzr_complete_heatmap_6sublayers")
    plt.close(fig)


# ===========================================================================
# Figure 2: ODR Complete Heatmap Grid (2×3, log10 scale)
# ===========================================================================
def fig2_odr_heatmap_grid(df_ab):
    print("\n[Fig2] ODR Complete Heatmap Grid (all 6 sublayers, log10) ...")

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(
        "ODR (Overflow/Deviation Rate) — All Sublayers × Bit Resolutions\n"
        "log$_{10}$(ODR): gradient overflow rate; largely independent of bit resolution",
        fontsize=13, y=1.03,
    )

    for si, sl in enumerate(SUBLAYERS):
        ax = axes[si // 3, si % 3]
        mat, bits_list = build_matrix(df_ab, sl, "ODR", BITS_ORDER)

        # Apply log10 (handle zeros)
        mat_log = np.where(mat > 0, np.log10(mat), np.nan)

        im = ax.imshow(mat_log, aspect="auto", cmap="hot_r", origin="upper")
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(bits_list)))
        ax.set_yticklabels([f"{b}b" for b in bits_list], fontsize=9)
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("DAC Bits")
        ax.set_title(f"({chr(97+si)}) {sl}", fontsize=11, fontweight="bold")
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("log$_{10}$(ODR)")
        annotate_heatmap(ax, mat_log, fmt=".1f", fs=6)

    plt.tight_layout()
    _save(fig, "odr_complete_heatmap_6sublayers")
    plt.close(fig)


# ===========================================================================
# Figure 3: QZR vs Bits Line Plots (2×3, 12 layers per sublayer)
# ===========================================================================
def fig3_qzr_vs_bits_lines(df_ab):
    print("\n[Fig3] QZR vs Bits Line Plots (12 layers per sublayer) ...")

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        "QZR vs Bit Resolution — Per Layer Trajectories\n"
        "Color: blue (shallow L0) → red (deep L11); higher bits → lower QZR",
        fontsize=13, y=1.03,
    )

    for si, sl in enumerate(SUBLAYERS):
        ax = axes[si // 3, si % 3]
        mat, bits_list = build_matrix(df_ab, sl, "QZR_nonzero", BITS_ORDER)

        for li in range(N_LAYERS):
            values = mat[:, li]
            valid = ~np.isnan(values)
            if valid.any():
                ax.plot(np.array(bits_list)[valid], values[valid],
                        color=LAYER_COLORS[li], marker="o", ms=4, lw=1.5,
                        alpha=0.85, label=f"L{li}")

        ax.set_xlabel("DAC Bits")
        ax.set_ylabel("QZR_nonzero")
        ax.set_title(f"({chr(97+si)}) {sl}", fontsize=11, fontweight="bold")
        ax.set_xticks(bits_list)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=-0.02)
        ax.legend(fontsize=6, ncol=3, loc="upper right",
                  framealpha=0.7, handlelength=1.0)

    plt.tight_layout()
    _save(fig, "qzr_vs_bits_layerlines_6sublayers")
    plt.close(fig)


# ===========================================================================
# Figure 4: QZR-ODR Correlation Scatter (2×3, per bit config)
# ===========================================================================
def fig4_qzr_odr_scatter(df_ab):
    print("\n[Fig4] QZR-ODR Correlation Scatter (per bit config) ...")

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        "QZR vs ODR Correlation — Per Bit Resolution\n"
        "Each point = one (layer, sublayer) pair; color by sublayer type",
        fontsize=13, y=1.03,
    )

    corr_results = {}

    for bi, bits in enumerate(BITS_ORDER):
        ax = axes[bi // 3, bi % 3]
        sub = df_ab[df_ab["dac_bits"] == bits].copy()

        if len(sub) == 0:
            ax.set_title(f"({chr(97+bi)}) {bits}b — no data")
            continue

        # Filter valid data
        sub = sub[(sub["ODR"] > 0) & sub["QZR_nonzero"].notna()].copy()
        sub["log10_ODR"] = np.log10(sub["ODR"])

        # Scatter by sublayer
        for sl in SUBLAYERS:
            sl_data = sub[sub["sublayer"] == sl]
            if len(sl_data) == 0:
                continue
            ax.scatter(sl_data["log10_ODR"], sl_data["QZR_nonzero"],
                       color=SUBLAYER_COLORS[sl], marker=SUBLAYER_MARKERS[sl],
                       s=40, alpha=0.75, label=sl, edgecolors="white",
                       linewidth=0.3)

        # Compute correlation
        if len(sub) >= 3:
            r_p, p_p = stats.pearsonr(sub["log10_ODR"], sub["QZR_nonzero"])
            r_s, p_s = stats.spearmanr(sub["log10_ODR"], sub["QZR_nonzero"])
            corr_results[bits] = {"pearson_r": r_p, "pearson_p": p_p,
                                  "spearman_rho": r_s, "spearman_p": p_s,
                                  "n": len(sub)}

            # Regression line
            slope, intercept = np.polyfit(sub["log10_ODR"], sub["QZR_nonzero"], 1)
            x_range = np.linspace(sub["log10_ODR"].min(), sub["log10_ODR"].max(), 50)
            ax.plot(x_range, slope * x_range + intercept, "k--", lw=1.5, alpha=0.6)

            ax.text(0.03, 0.97,
                    f"Pearson r={r_p:.3f} (p={p_p:.1e})\n"
                    f"Spearman ρ={r_s:.3f} (p={p_s:.1e})\n"
                    f"n={len(sub)}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="wheat", alpha=0.8))

        ax.set_xlabel("log$_{10}$(ODR)")
        ax.set_ylabel("QZR_nonzero")
        ax.set_title(f"({chr(97+bi)}) {bits}-bit", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.7, ncol=2)

    plt.tight_layout()
    _save(fig, "qzr_odr_correlation_scatter")
    plt.close(fig)

    return corr_results


# ===========================================================================
# Figure 5: QZR-ODR Correlation Coefficient vs Bits (1×2)
# ===========================================================================
def fig5_correlation_vs_bits(df_ab):
    print("\n[Fig5] QZR-ODR Correlation Coefficient vs Bits ...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "QZR-ODR Correlation Strength vs Bit Resolution\n"
        "Does the relationship between ODR and QZR depend on quantization precision?",
        fontsize=12, y=1.04,
    )

    # Compute correlations: overall + per sublayer
    corr_data = {"overall": {"pearson": [], "spearman": [], "bits": []}}
    for sl in SUBLAYERS:
        corr_data[sl] = {"pearson": [], "spearman": [], "bits": []}

    for bits in BITS_ORDER:
        sub = df_ab[df_ab["dac_bits"] == bits].copy()
        sub = sub[(sub["ODR"] > 0) & sub["QZR_nonzero"].notna()].copy()
        if len(sub) < 3:
            continue
        sub["log10_ODR"] = np.log10(sub["ODR"])

        # Overall
        r_p, _ = stats.pearsonr(sub["log10_ODR"], sub["QZR_nonzero"])
        r_s, _ = stats.spearmanr(sub["log10_ODR"], sub["QZR_nonzero"])
        corr_data["overall"]["pearson"].append(r_p)
        corr_data["overall"]["spearman"].append(r_s)
        corr_data["overall"]["bits"].append(bits)

        # Per sublayer
        for sl in SUBLAYERS:
            sl_sub = sub[sub["sublayer"] == sl]
            if len(sl_sub) >= 3:
                r_p_sl, _ = stats.pearsonr(sl_sub["log10_ODR"], sl_sub["QZR_nonzero"])
                r_s_sl, _ = stats.spearmanr(sl_sub["log10_ODR"], sl_sub["QZR_nonzero"])
            else:
                r_p_sl, r_s_sl = np.nan, np.nan
            corr_data[sl]["pearson"].append(r_p_sl)
            corr_data[sl]["spearman"].append(r_s_sl)
            corr_data[sl]["bits"].append(bits)

    # Panel (a): Pearson r
    ax = axes[0]
    bits_arr = corr_data["overall"]["bits"]
    ax.plot(bits_arr, corr_data["overall"]["pearson"],
            "k-o", lw=2.5, ms=8, label="Overall", zorder=10)
    for sl in SUBLAYERS:
        ax.plot(corr_data[sl]["bits"], corr_data[sl]["pearson"],
                color=SUBLAYER_COLORS[sl], marker=SUBLAYER_MARKERS[sl],
                ms=5, lw=1.2, alpha=0.7, label=sl)
    ax.set_xlabel("DAC Bits")
    ax.set_ylabel("Pearson r")
    ax.set_title("(a) Pearson Correlation (log₁₀ODR vs QZR)", fontsize=10)
    ax.set_xticks(BITS_ORDER)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.7)

    # Panel (b): Spearman ρ
    ax = axes[1]
    ax.plot(bits_arr, corr_data["overall"]["spearman"],
            "k-o", lw=2.5, ms=8, label="Overall", zorder=10)
    for sl in SUBLAYERS:
        ax.plot(corr_data[sl]["bits"], corr_data[sl]["spearman"],
                color=SUBLAYER_COLORS[sl], marker=SUBLAYER_MARKERS[sl],
                ms=5, lw=1.2, alpha=0.7, label=sl)
    ax.set_xlabel("DAC Bits")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("(b) Spearman Correlation (log₁₀ODR vs QZR)", fontsize=10)
    ax.set_xticks(BITS_ORDER)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.7)

    plt.tight_layout()
    _save(fig, "qzr_odr_correlation_vs_bits")
    plt.close(fig)

    return corr_data


# ===========================================================================
# Figure 6: ECDF Summary — Worst Sublayers across Bits (2×3)
# ===========================================================================
def fig6_ecdf_worst_sublayers(df_ab):
    print("\n[Fig6] ECDF Summary — Worst sublayers across bits ...")

    # Identify top-3 worst sublayers by mean QZR at baseline 7b
    baseline = df_ab[df_ab["dac_bits"] == 7]
    sl_mean = baseline.groupby("sublayer")["QZR_nonzero"].mean().sort_values(ascending=False)
    worst_sublayers = sl_mean.head(3).index.tolist()
    print(f"  Worst sublayers (by mean QZR@7b): {worst_sublayers}")

    # For each worst sublayer, find the worst layer (highest QZR at 7b)
    worst_configs = []
    for sl in worst_sublayers:
        sl_bl = baseline[baseline["sublayer"] == sl]
        worst_layer = sl_bl.sort_values("QZR_nonzero", ascending=False).iloc[0]["layer_idx"]
        worst_configs.append((sl, int(worst_layer)))
    print(f"  Worst (sublayer, layer): {worst_configs}")

    # Load NPZ data for 4b, 7b, 12b
    npz_files = {
        4: os.path.join(NPZ_DIR, "absmax_raw_B_sweep_4b.npz"),
        7: os.path.join(NPZ_DIR, "absmax_raw_A_baseline_7b.npz"),
        12: os.path.join(NPZ_DIR, "absmax_raw_B_sweep_12b.npz"),
    }
    npz_data = {}
    for b, path in npz_files.items():
        if os.path.exists(path):
            npz_data[b] = dict(np.load(path))

    if len(npz_data) == 0:
        print("  [SKIP] No NPZ data available")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Absmax ECDF — Worst Sublayer×Layer Combinations\n"
        "Comparing 4-bit (most quantization) vs 7-bit (baseline) vs 12-bit (least quantization)",
        fontsize=12, y=1.03,
    )

    # Top row: absmax ECDF for 3 worst configs
    for ci, (sl, li) in enumerate(worst_configs):
        ax = axes[0, ci]
        key = f"L{li}_{sl}"
        qzr_7b = baseline[(baseline["sublayer"] == sl) &
                           (baseline["layer_idx"] == li)]["QZR_nonzero"].values
        qzr_label = f" (QZR@7b={qzr_7b[0]:.3f})" if len(qzr_7b) > 0 else ""

        for bits in sorted(npz_data.keys()):
            if key not in npz_data[bits]:
                continue
            arr = npz_data[bits][key]
            xs, ys = ecdf(arr)
            lw = 2.2 if bits == 7 else 1.5
            ax.plot(xs, ys, color=BITS_COLORS[bits], lw=lw, alpha=0.85,
                    label=f"{bits}b")

        ax.set_xscale("log")
        ax.set_xlabel(r"$\|\delta_{vec}\|_\infty$")
        ax.set_ylabel("ECDF")
        ax.set_title(f"L{li} {sl}{qzr_label}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.7)

    # Bottom row: QZR bar comparison for same configs across bits
    bits_compare = [4, 6, 7, 8, 10, 12]
    for ci, (sl, li) in enumerate(worst_configs):
        ax = axes[1, ci]
        qzr_vals = []
        bit_labels = []
        colors = []
        for b in bits_compare:
            row = df_ab[(df_ab["dac_bits"] == b) &
                        (df_ab["sublayer"] == sl) &
                        (df_ab["layer_idx"] == li)]
            if len(row) > 0:
                qzr_vals.append(row["QZR_nonzero"].values[0])
                bit_labels.append(f"{b}b")
                colors.append(BITS_COLORS.get(b, "gray"))

        bars = ax.bar(range(len(qzr_vals)), qzr_vals, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(bit_labels)))
        ax.set_xticklabels(bit_labels)
        ax.set_ylabel("QZR_nonzero")
        ax.set_title(f"L{li} {sl} — QZR by bits", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

        # Annotate bars
        for bar, val in zip(bars, qzr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")

    plt.tight_layout()
    _save(fig, "ecdf_worst_sublayers_bits_comparison")
    plt.close(fig)


# ===========================================================================
# Figure 7: Solution Effectiveness — QZR & ODR Delta Heatmaps (2×2)
# ===========================================================================
def fig7_solution_deltas(df_c):
    print("\n[Fig7] Solution Effectiveness — QZR & ODR Delta Heatmaps ...")

    variants = ["sto_round", "nm_thres_cal", "p99_clip"]
    variant_labels = ["sto_round", "nm_thres_cal", "p99_clip"]

    # Extract baseline from C
    bl = df_c[df_c["variant"] == "baseline"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Solution Effectiveness — QZR Change (ΔQZR) & ODR Ratio\n"
        "ΔQZR = variant − baseline (negative = improvement); "
        "ODR ratio = variant / baseline (< 1 = reduced overflow)",
        fontsize=12, y=1.03,
    )

    for col_idx, sl in enumerate(["K", "V"]):
        # QZR delta: variant - baseline
        ax = axes[0, col_idx]
        mat_qzr = np.full((len(variants), N_LAYERS), np.nan)
        for vi, var in enumerate(variants):
            var_df = df_c[(df_c["variant"] == var) & (df_c["sublayer"] == sl)]
            bl_df = bl[bl["sublayer"] == sl]
            for li in range(N_LAYERS):
                v_row = var_df[var_df["layer_idx"] == li]
                b_row = bl_df[bl_df["layer_idx"] == li]
                if len(v_row) > 0 and len(b_row) > 0:
                    mat_qzr[vi, li] = (v_row["QZR_nonzero"].values[0] -
                                       b_row["QZR_nonzero"].values[0])

        vabs = max(abs(np.nanmin(mat_qzr)), abs(np.nanmax(mat_qzr)), 0.01)
        im = ax.imshow(mat_qzr, aspect="auto", cmap="RdBu_r", origin="upper",
                       vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variant_labels, fontsize=9)
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("Solution Variant")
        ax.set_title(f"({'a' if col_idx == 0 else 'b'}) {sl} — ΔQZR (variant − baseline)",
                     fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85, label="ΔQZR_nonzero")
        annotate_heatmap(ax, mat_qzr, fmt=".3f", fs=6)

        # ODR ratio: variant / baseline
        ax = axes[1, col_idx]
        mat_odr = np.full((len(variants), N_LAYERS), np.nan)
        for vi, var in enumerate(variants):
            var_df = df_c[(df_c["variant"] == var) & (df_c["sublayer"] == sl)]
            bl_df = bl[bl["sublayer"] == sl]
            for li in range(N_LAYERS):
                v_row = var_df[var_df["layer_idx"] == li]
                b_row = bl_df[bl_df["layer_idx"] == li]
                if len(v_row) > 0 and len(b_row) > 0:
                    bl_odr = b_row["ODR"].values[0]
                    if bl_odr > 0:
                        mat_odr[vi, li] = v_row["ODR"].values[0] / bl_odr

        im = ax.imshow(mat_odr, aspect="auto", cmap="RdYlGn_r", origin="upper")
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variant_labels, fontsize=9)
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("Solution Variant")
        ax.set_title(f"({'c' if col_idx == 0 else 'd'}) {sl} — ODR ratio (variant / baseline)",
                     fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85, label="ODR ratio")
        annotate_heatmap(ax, mat_odr, fmt=".2f", fs=6)

    plt.tight_layout()
    _save(fig, "solution_effectiveness_qzr_odr_delta")
    plt.close(fig)


# ===========================================================================
# Console Output
# ===========================================================================
def print_summary_tables(df_ab, corr_data):
    """Print summary tables to console."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE QZR/ODR SUMMARY")
    print("=" * 80)

    # QZR summary per sublayer × bits
    print("\n--- QZR_nonzero Mean per Sublayer × Bits ---")
    header = f"{'Sublayer':<8}" + "".join(f"  {b}b     " for b in BITS_ORDER)
    print(header)
    print("-" * len(header))
    for sl in SUBLAYERS:
        row = f"{sl:<8}"
        for b in BITS_ORDER:
            sub = df_ab[(df_ab["dac_bits"] == b) & (df_ab["sublayer"] == sl)]
            if len(sub) > 0:
                row += f"  {sub['QZR_nonzero'].mean():.4f}  "
            else:
                row += "    N/A    "
        print(row)

    # ODR summary
    print("\n--- ODR Mean per Sublayer × Bits ---")
    header = f"{'Sublayer':<8}" + "".join(f"  {b}b       " for b in BITS_ORDER)
    print(header)
    print("-" * len(header))
    for sl in SUBLAYERS:
        row = f"{sl:<8}"
        for b in BITS_ORDER:
            sub = df_ab[(df_ab["dac_bits"] == b) & (df_ab["sublayer"] == sl)]
            if len(sub) > 0:
                row += f"  {sub['ODR'].mean():>9.2f}  "
            else:
                row += "      N/A    "
        print(row)

    # Correlation table
    if corr_data:
        print("\n--- Pearson & Spearman Correlation (log10(ODR) vs QZR) per Bits ---")
        print(f"{'Bits':<6} {'Pearson r':>10} {'Spearman ρ':>12} {'n':>5}")
        print("-" * 35)
        overall = corr_data.get("overall", {})
        for i, b in enumerate(overall.get("bits", [])):
            p = overall["pearson"][i]
            s = overall["spearman"][i]
            print(f"{b}b    {p:>10.4f} {s:>12.4f}")

    # Worst layers per sublayer
    print("\n--- Worst Layers per Sublayer (highest QZR@7b) ---")
    baseline = df_ab[df_ab["dac_bits"] == 7]
    for sl in SUBLAYERS:
        sl_bl = baseline[baseline["sublayer"] == sl].sort_values(
            "QZR_nonzero", ascending=False)
        if len(sl_bl) > 0:
            top = sl_bl.head(3)
            layers = ", ".join(
                f"L{int(r['layer_idx'])}({r['QZR_nonzero']:.4f})"
                for _, r in top.iterrows()
            )
            print(f"  {sl:<6}: {layers}")

    print("\n" + "=" * 80)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(f"CSV dir: {CSV_DIR}")
    print(f"NPZ dir: {NPZ_DIR}")
    print(f"Out dir: {OUT_DIR}")

    df_a, df_b, df_c, df_ab = load_data()
    print(f"Loaded: A={len(df_a)} rows, B={len(df_b)} rows, "
          f"C={len(df_c)} rows, AB_merged={len(df_ab)} rows")
    print(f"Bits in AB: {sorted(df_ab['dac_bits'].unique())}")

    # Figure 1: QZR heatmap grid
    fig1_qzr_heatmap_grid(df_ab)

    # Figure 2: ODR heatmap grid
    fig2_odr_heatmap_grid(df_ab)

    # Figure 3: QZR vs bits line plots
    fig3_qzr_vs_bits_lines(df_ab)

    # Figure 4: QZR-ODR correlation scatter
    corr_results = fig4_qzr_odr_scatter(df_ab)

    # Figure 5: Correlation vs bits
    corr_data = fig5_correlation_vs_bits(df_ab)

    # Figure 6: ECDF worst sublayers
    fig6_ecdf_worst_sublayers(df_ab)

    # Figure 7: Solution deltas
    fig7_solution_deltas(df_c)

    # Console summary
    print_summary_tables(df_ab, corr_data)

    print(f"\nAll 7 figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

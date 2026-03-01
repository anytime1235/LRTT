"""plot_ecdf_comprehensive.py — Layer-wise & bit-resolution absmax ECDF plots.

Generates publication-quality ECDF figures from saved NPZ raw data.

IMPORTANT NOTE on what absmax ECDF means:
  - NPZ stores per-vector absmax = max(|δ_i|) for each gradient vector
  - AbsMax Noise Management normalizes each vector to [-1,1] BEFORE DAC quantization
  - Therefore raw absmax magnitude ≠ quantization underflow
  - Quantization underflow is determined by within-vector RATIO distribution
  - absmax ECDF shows gradient magnitude distribution & inter-layer variation
  - Use ratio CDF (from CSV) or QZR metrics for actual quantization analysis

Figures generated:
  1-6. Per-sublayer (Q/K/V/O/FFN1/FFN2) all-layer × all-bits 3x4 grid
  7.   All-sublayer layerwise comparison (baseline 7b)
  8.   QZR-based quantization impact heatmap (from CSV, correct metric)
  9.   Solution comparison absmax ECDF

Usage:
  python plot_ecdf_comprehensive.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--npz-dir", type=str,
                    default="/data/main_results/results/npz")
parser.add_argument("--csv-dir", type=str,
                    default="/data/main_results/results/csv")
parser.add_argument("--out-dir", type=str,
                    default="/data/main_results/results/figures/diagnostic")
args = parser.parse_args()

NPZ_DIR = args.npz_dir
CSV_DIR = args.csv_dir
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LAYERS = 12
INP_BOUND = 1.0
SUBLAYERS_ALL = ["Q", "K", "V", "O", "FFN1", "FFN2"]

BITS_LIST = [4, 6, 8, 10, 12]
BASELINE_BITS = 7

SOLUTION_VARIANTS = {
    "baseline":      "absmax_raw_A_baseline_7b.npz",
    "nm_thres_cal":  "absmax_raw_C_nm_thres_cal.npz",
    "p99_clip":      "absmax_raw_C_p99_clip.npz",
    "sto_round":     "absmax_raw_C_sto_round.npz",
}

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

# Colors
LAYER_CMAP = plt.cm.coolwarm
LAYER_COLORS = [LAYER_CMAP(i / (N_LAYERS - 1)) for i in range(N_LAYERS)]

BITS_COLORS = {4: "#e41a1c", 6: "#ff7f00", 7: "#333333",
               8: "#4daf4a", 10: "#377eb8", 12: "#984ea3"}

SOLUTION_COLORS = {
    "baseline": "#4C72B0", "nm_thres_cal": "#DD8452",
    "p99_clip": "#55A868", "sto_round": "#C44E52",
}

SUBLAYER_COLORS = {
    "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
    "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(filename):
    path = os.path.join(NPZ_DIR, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    return dict(np.load(path))


def ecdf(arr, max_points=4000):
    s = np.sort(arr)
    y = np.arange(1, len(s) + 1) / len(s)
    if len(s) > max_points:
        idx = np.linspace(0, len(s) - 1, max_points, dtype=int)
        return s[idx], y[idx]
    return s, y


def _save(fig, basename):
    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"{basename}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR}/{basename}.{{pdf,png}}")


def load_all_npz():
    """Load baseline + all bit-sweep NPZs. Returns {bits: {key: array}}."""
    npz_data = {}
    bl = load_npz("absmax_raw_A_baseline_7b.npz")
    if bl is not None:
        npz_data[BASELINE_BITS] = bl
    for bits in BITS_LIST:
        d = load_npz(f"absmax_raw_B_sweep_{bits}b.npz")
        if d is not None:
            npz_data[bits] = d
    return npz_data


# ===========================================================================
# Fig 1-6: Per-sublayer, all 12 layers × all bit-resolutions (3×4 grid)
# ===========================================================================

def fig_sublayer_grid(sl, npz_data):
    """One 3×4 figure: 12 panels (one per layer), each with 6 bit-res curves."""
    all_bits = sorted(npz_data.keys())

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(
        f"{sl} Sublayer — Absmax ECDF per Layer × Bit Resolution\n"
        r"$X = \|\delta_{vec}\|_\infty$ (raw gradient magnitude, before AbsMax NM scaling). "
        "Shape = gradient magnitude spread; different bits → different forward quantization noise",
        fontsize=12, y=1.02,
    )

    for li in range(N_LAYERS):
        ax = axes[li // 4, li % 4]
        key = f"L{li}_{sl}"

        has_data = False
        for bits in all_bits:
            if key not in npz_data[bits]:
                continue
            arr = npz_data[bits][key]
            xs, ys = ecdf(arr)
            lw = 2.2 if bits == BASELINE_BITS else 1.3
            ax.plot(xs, ys, color=BITS_COLORS.get(bits, "gray"),
                    lw=lw, alpha=0.85, label=f"{bits}b")
            has_data = True

        if has_data:
            ax.set_xscale("log")
        ax.set_title(f"L{li} {sl}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.02, 1.05)
        if li >= 8:
            ax.set_xlabel(r"$\|\delta\|_\infty$", fontsize=9)
        if li % 4 == 0:
            ax.set_ylabel("ECDF", fontsize=9)
        ax.tick_params(labelsize=7)

    # Shared legend at bottom
    handles = [Line2D([0], [0], color=BITS_COLORS.get(b, "gray"),
                      lw=2.0 if b == BASELINE_BITS else 1.3,
                      label=f"{b}-bit{'  (baseline)' if b == BASELINE_BITS else ''}")
               for b in all_bits]
    fig.legend(handles=handles, loc="lower center", ncol=len(all_bits),
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, f"ecdf_{sl}_all_layers_all_bits")
    plt.close(fig)


# ===========================================================================
# Fig 7: All-sublayer layerwise comparison (baseline 7b, 2×3 grid)
# ===========================================================================

def fig_layerwise_all_sublayers():
    """6-panel (2×3): Q/K/V/O/FFN1/FFN2 — each panel shows 12 layers."""
    print("\n[Fig7] Layerwise ECDF, all 6 sublayers (baseline 7b) ...")
    data = load_npz("absmax_raw_A_baseline_7b.npz")
    if data is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Absmax ECDF by Encoder Layer — Baseline (DAC 7-bit), All 6 Sublayers\n"
        r"$X = \|\delta_{vec}\|_\infty$; color gradient: blue (shallow) → red (deep layer)",
        fontsize=12, y=1.02,
    )

    for si, sl in enumerate(SUBLAYERS_ALL):
        ax = axes[si // 3, si % 3]
        for li in range(N_LAYERS):
            key = f"L{li}_{sl}"
            if key not in data:
                continue
            arr = data[key]
            xs, ys = ecdf(arr)
            ax.plot(xs, ys, color=LAYER_COLORS[li], lw=1.2, alpha=0.85,
                    label=f"L{li}")

        ax.set_xscale("log")
        ax.set_xlabel(r"$\|\delta_{vec}\|_\infty$")
        ax.set_ylabel("ECDF")
        ax.set_title(f"({chr(97+si)}) {sl}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=6, ncol=3, loc="lower right",
                  framealpha=0.7, handlelength=1.0)

    plt.tight_layout()
    _save(fig, "ecdf_layerwise_all_sublayers_baseline7b")
    plt.close(fig)


# ===========================================================================
# Fig 8: QZR heatmap from CSV (correct quantization metric)
# ===========================================================================

def fig_qzr_heatmap():
    """Correct underflow analysis using QZR_nonzero from CSV (post-NM-scaling metric).

    Panel layout:
      (a) Baseline 7b QZR heatmap (layer × sublayer)
      (b) Bit-sweep QZR heatmap for K (layer × bits)
      (c) Bit-sweep QZR heatmap for V (layer × bits)
      (d) Solution QZR heatmap for K (layer × variant)
    """
    print("\n[Fig8] QZR heatmap (correct quantization metric from CSV) ...")

    csv_a = os.path.join(CSV_DIR, "metrics_paper_A_rootcause_summary.csv")
    csv_b = os.path.join(CSV_DIR, "metrics_paper_B_bitsweep_summary.csv")
    csv_c = os.path.join(CSV_DIR, "metrics_paper_C_solutions_summary.csv")

    has_a = os.path.exists(csv_a)
    has_b = os.path.exists(csv_b)
    has_c = os.path.exists(csv_c)

    if not has_a:
        print("  [SKIP] No rootcause summary CSV")
        return

    df_a = pd.read_csv(csv_a) if has_a else None
    df_b = pd.read_csv(csv_b) if has_b else None
    df_c = pd.read_csv(csv_c) if has_c else None

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        "Quantization Zero Rate (QZR) — Correct Post-NM-Scaling Metric\n"
        "QZR = fraction of gradient elements quantized to 0 after AbsMax normalization + DAC",
        fontsize=12, y=1.02,
    )

    # Helper
    def annotate(ax, mat, fmt=".3f", fs=7):
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                val = mat[ri, ci]
                if np.isnan(val):
                    continue
                color = "white" if val > 0.3 else "black"
                ax.text(ci, ri, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=fs, color=color, fontweight="bold")

    # (a) Baseline: layer × sublayer
    ax = axes[0, 0]
    qzr_col = "QZR_nonzero" if "QZR_nonzero" in df_a.columns else "QZR_mean"
    mat_a = np.full((N_LAYERS, len(SUBLAYERS_ALL)), np.nan)
    for _, row in df_a.iterrows():
        li = int(row["layer_idx"])
        sl = row["sublayer"]
        if sl in SUBLAYERS_ALL and li < N_LAYERS:
            mat_a[li, SUBLAYERS_ALL.index(sl)] = row[qzr_col]
    im = ax.imshow(mat_a, aspect="auto", cmap="plasma", origin="upper",
                   vmin=0.0, vmax=0.5)
    ax.set_xticks(range(len(SUBLAYERS_ALL)))
    ax.set_xticklabels(SUBLAYERS_ALL, fontsize=9)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=8)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
    ax.set_title("(a) Baseline 7b — QZR per layer × sublayer", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, label="QZR")
    annotate(ax, mat_a)

    # (b) Bit-sweep: K sublayer, bits × layer
    ax = axes[0, 1]
    if df_b is not None:
        bits_in_data = sorted(df_b["dac_bits"].unique())
        mat_bk = np.full((len(bits_in_data), N_LAYERS), np.nan)
        k_df = df_b[df_b["sublayer"] == "K"]
        for bi, b in enumerate(bits_in_data):
            for li in range(N_LAYERS):
                sub = k_df[(k_df["dac_bits"] == b) & (k_df["layer_idx"] == li)]
                if len(sub) > 0:
                    mat_bk[bi, li] = sub[qzr_col].mean()
        im = ax.imshow(mat_bk, aspect="auto", cmap="plasma", origin="upper",
                       vmin=0.0, vmax=0.5)
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(bits_in_data)))
        ax.set_yticklabels([f"{b}b" for b in bits_in_data], fontsize=9)
        ax.set_xlabel("Encoder Layer"); ax.set_ylabel("DAC bits")
        ax.set_title("(b) K — QZR by bit-resolution × layer", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85, label="QZR")
        annotate(ax, mat_bk, fmt=".2f", fs=6)
    else:
        ax.text(0.5, 0.5, "No bit-sweep CSV", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title("(b) K — QZR by bit-resolution × layer")

    # (c) Bit-sweep: V sublayer, bits × layer
    ax = axes[1, 0]
    if df_b is not None:
        mat_bv = np.full((len(bits_in_data), N_LAYERS), np.nan)
        v_df = df_b[df_b["sublayer"] == "V"]
        for bi, b in enumerate(bits_in_data):
            for li in range(N_LAYERS):
                sub = v_df[(v_df["dac_bits"] == b) & (v_df["layer_idx"] == li)]
                if len(sub) > 0:
                    mat_bv[bi, li] = sub[qzr_col].mean()
        im = ax.imshow(mat_bv, aspect="auto", cmap="plasma", origin="upper",
                       vmin=0.0, vmax=0.5)
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(bits_in_data)))
        ax.set_yticklabels([f"{b}b" for b in bits_in_data], fontsize=9)
        ax.set_xlabel("Encoder Layer"); ax.set_ylabel("DAC bits")
        ax.set_title("(c) V — QZR by bit-resolution × layer", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85, label="QZR")
        annotate(ax, mat_bv, fmt=".2f", fs=6)
    else:
        ax.text(0.5, 0.5, "No bit-sweep CSV", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title("(c) V — QZR by bit-resolution × layer")

    # (d) Solutions: K sublayer, variant × layer
    ax = axes[1, 1]
    variants = ["baseline", "sto_round", "nm_thres_cal", "p99_clip"]
    if df_c is not None:
        variants_present = [v for v in variants if v in df_c["variant"].unique()]
        mat_ck = np.full((len(variants_present), N_LAYERS), np.nan)
        k_df = df_c[df_c["sublayer"] == "K"]
        for vi, variant in enumerate(variants_present):
            vdf = k_df[k_df["variant"] == variant]
            for _, row in vdf.iterrows():
                li = int(row["layer_idx"])
                if li < N_LAYERS:
                    mat_ck[vi, li] = row[qzr_col]
        im = ax.imshow(mat_ck, aspect="auto", cmap="plasma", origin="upper",
                       vmin=0.0, vmax=0.5)
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(variants_present)))
        ax.set_yticklabels(variants_present, fontsize=9)
        ax.set_xlabel("Encoder Layer"); ax.set_ylabel("Solution Variant")
        ax.set_title("(d) K — QZR by solution × layer", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85, label="QZR")
        annotate(ax, mat_ck, fmt=".2f", fs=6)
    else:
        ax.text(0.5, 0.5, "No solutions CSV", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title("(d) K — QZR by solution × layer")

    plt.tight_layout()
    _save(fig, "qzr_heatmap_correct_metric")
    plt.close(fig)


# ===========================================================================
# Fig 9: Solution comparison absmax ECDF — all sublayers, worst layers
# ===========================================================================

def fig_solution_ecdf():
    """6-panel (2×3): worst-3 layers × {K row, V row}, 4 solution curves each."""
    print("\n[Fig9] Solution comparison ECDF ...")

    csv_path = os.path.join(CSV_DIR, "metrics_paper_A_rootcause_summary.csv")
    if os.path.exists(csv_path):
        df_a = pd.read_csv(csv_path)
        qzr_col = "QZR_nonzero" if "QZR_nonzero" in df_a.columns else "QZR_mean"
        k_df = df_a[df_a["sublayer"] == "K"].sort_values(qzr_col, ascending=False)
        worst_layers = k_df.head(3)["layer_idx"].astype(int).tolist()
    else:
        worst_layers = [7, 9, 10]

    sol_data = {}
    for variant, filename in SOLUTION_VARIANTS.items():
        d = load_npz(filename)
        if d is not None:
            sol_data[variant] = d

    if len(sol_data) == 0:
        print("  [SKIP] No solution NPZ data")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Absmax ECDF — Solution Comparison (DAC 7-bit)\n"
        r"Do solutions change the gradient magnitude distribution? "
        "(Note: absmax distribution ≠ quantization underflow; see QZR heatmap for that)",
        fontsize=11, y=1.02,
    )

    for row_idx, sl in enumerate(["K", "V"]):
        for col_idx, li in enumerate(worst_layers):
            ax = axes[row_idx, col_idx]
            key = f"L{li}_{sl}"

            for variant, color in SOLUTION_COLORS.items():
                if variant not in sol_data or key not in sol_data[variant]:
                    continue
                arr = sol_data[variant][key]
                xs, ys = ecdf(arr)
                lw = 2.2 if variant == "baseline" else 1.5
                ax.plot(xs, ys, color=color, lw=lw, alpha=0.85,
                        label=variant)

            ax.set_xscale("log")
            ax.set_xlabel(r"$\|\delta_{vec}\|_\infty$")
            ax.set_ylabel("ECDF")
            ax.set_title(f"L{li} {sl}", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(-0.02, 1.05)
            ax.legend(fontsize=8, loc="lower right", framealpha=0.7)

    plt.tight_layout()
    _save(fig, "ecdf_solution_comparison_worst_KV")
    plt.close(fig)


# ===========================================================================
# Fig 10: Spread metric — IQR of log(absmax), layer × sublayer × bits
# ===========================================================================

def fig_spread_metric():
    """The gradient magnitude spread (IQR of log-absmax) indicates outlier severity.

    Wider spread → absmax varies more across vectors → more outlier-dominant vectors
    → higher QZR after AbsMax NM normalization.
    """
    print("\n[Fig10] Gradient magnitude spread (IQR log-absmax) ...")
    npz_data = load_all_npz()
    if len(npz_data) == 0:
        return

    all_bits = sorted(npz_data.keys())

    # Compute IQR(log10(absmax)) for K and V at each (bits, layer)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Gradient Magnitude Spread — IQR of log$_{10}$(absmax)\n"
        "Wider spread → more heterogeneous vector magnitudes → "
        "AbsMax NM creates larger dynamic range → more quantization zeros",
        fontsize=11, y=1.04,
    )

    for si, (sl, ax) in enumerate(zip(["K", "V"], axes)):
        mat = np.full((len(all_bits), N_LAYERS), np.nan)
        for bi, bits in enumerate(all_bits):
            for li in range(N_LAYERS):
                key = f"L{li}_{sl}"
                if key in npz_data[bits]:
                    arr = npz_data[bits][key]
                    log_arr = np.log10(arr[arr > 0] + 1e-15)
                    mat[bi, li] = np.percentile(log_arr, 75) - np.percentile(log_arr, 25)

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", origin="upper")
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=8)
        ax.set_yticks(range(len(all_bits)))
        ax.set_yticklabels([f"{b}b" for b in all_bits], fontsize=9)
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("DAC Bit Resolution")
        ax.set_title(f"({chr(97+si)}) {sl} sublayer", fontsize=11)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("IQR of log₁₀(absmax)")

        for bi in range(mat.shape[0]):
            for li in range(mat.shape[1]):
                val = mat[bi, li]
                if np.isnan(val):
                    continue
                vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
                norm = (val - vmin) / (vmax - vmin + 1e-12)
                color = "white" if norm > 0.55 else "black"
                ax.text(li, bi, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold")

    plt.tight_layout()
    _save(fig, "spread_iqr_log_absmax_KV_bits")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"NPZ dir: {NPZ_DIR}")
    print(f"CSV dir: {CSV_DIR}")
    print(f"Out dir: {OUT_DIR}")

    # Load all NPZ data once
    npz_data = load_all_npz()
    all_bits = sorted(npz_data.keys())
    print(f"Loaded NPZ for bits: {all_bits}")

    # Fig 1-6: Per-sublayer 3×4 grid (all layers × all bits)
    for sl in SUBLAYERS_ALL:
        print(f"\n[Fig] {sl} sublayer — all layers × all bits ...")
        fig_sublayer_grid(sl, npz_data)

    # Fig 7: All-sublayer layerwise (baseline only)
    fig_layerwise_all_sublayers()

    # Fig 8: QZR heatmap (correct metric)
    fig_qzr_heatmap()

    # Fig 9: Solution comparison
    fig_solution_ecdf()

    # Fig 10: Spread metric
    fig_spread_metric()

    # Clean up old incorrect figures (from previous version with wrong threshold)
    old_files = [
        "ecdf_underflow_heatmap_KV_bits",
        "ecdf_KV_detail_underflow_baseline7b",
        "ecdf_layerwise_QKVO_baseline7b",
        "ecdf_bitresolution_worst_KV",
    ]
    removed = 0
    for base in old_files:
        for ext in ["pdf", "png"]:
            path = os.path.join(OUT_DIR, f"{base}.{ext}")
            if os.path.exists(path):
                os.remove(path)
                removed += 1
    if removed:
        print(f"\nCleaned up {removed} old files with incorrect threshold comparison")

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

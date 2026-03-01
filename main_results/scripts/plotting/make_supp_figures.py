"""make_supp_figures.py — Supplementary Figures S1/S2/S3 for paper.

S1: Root Cause detail (2x3) — EZR, cosine, l2_retention, ratio bar, CDF, diagnosis text
S2: Bit-Sweep (2x2) — QZR vs bits, cosine vs bits, K/V QZR heatmaps
S3: Solutions comparison — Negative results (2x3)

Usage:
  python make_supp_figures.py --out-dir /data/results/tikitakav1 --run-tag v3
  python make_supp_figures.py --figures S2S3  # generate only S2 and S3
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Supplementary Figures S1/S2/S3")
parser.add_argument("--out-dir", type=str, default="/data/results/tikitakav1")
parser.add_argument("--run-tag", type=str, default="v3")
parser.add_argument("--figures", type=str, default="S1S2S3",
                    help="Which supplementary figures to generate (default: S1S2S3)")
args = parser.parse_args()

OUT_DIR = args.out_dir
RUN_TAG = args.run_tag
RUN_S1 = "S1" in args.figures.upper()
RUN_S2 = "S2" in args.figures.upper()
RUN_S3 = "S3" in args.figures.upper()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LAYERS = 12
DAC_BITS = 7
INP_BOUND = 1.0
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]

SUBLAYER_COLORS = {
    "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
    "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B",
}
SUBLAYER_MARKERS = {
    "Q": "o", "K": "s", "V": "^", "O": "D", "FFN1": "p", "FFN2": "h",
}

VARIANT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
VARIANTS = ["baseline", "sto_round", "nm_thres_cal", "p99_clip"]

# CSV paths
CSV_A_SUMMARY = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_summary.csv")
CSV_A_CDF = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_cdf.csv")
CSV_B_SUMMARY = os.path.join(OUT_DIR, "metrics_paper_B_bitsweep_summary.csv")
CSV_C_SUMMARY = os.path.join(OUT_DIR, "metrics_paper_C_solutions_summary.csv")

# ---------------------------------------------------------------------------
# Publication RC params
# ---------------------------------------------------------------------------
RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def to_mat(df, col):
    """Pivot DataFrame into (12, 6) ndarray [layer x sublayer]."""
    mat = np.full((N_LAYERS, len(SUBLAYER_ORDER)), np.nan)
    for _, row in df.iterrows():
        li = int(row["layer_idx"])
        sl = row["sublayer"]
        if sl in SUBLAYER_ORDER and li < N_LAYERS:
            mat[li, SUBLAYER_ORDER.index(sl)] = row[col]
    return mat


def annotate_heatmap(ax, mat, vmin, vmax, fmt=".3f", fontsize=6):
    """Write cell values on heatmap with auto-contrast text color."""
    for li in range(mat.shape[0]):
        for si in range(mat.shape[1]):
            val = mat[li, si]
            if np.isnan(val):
                continue
            txt = f"{val:{fmt}}"
            norm = (val - vmin) / (vmax - vmin + 1e-12)
            color = "white" if norm > 0.6 or norm < 0.15 else "black"
            ax.text(si, li, txt, ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")


def _save(fig, out_dir, basename, run_tag):
    """Save figure as PDF + PNG."""
    for ext in ["pdf", "png"]:
        path = os.path.join(out_dir, f"{basename}_{run_tag}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")


def _heatmap_panel(ax, mat, title, cmap, vmin, vmax, label, fmt=".3f"):
    """Generic heatmap panel for (12x6) layer-sublayer matrices."""
    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER, fontsize=8)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("Encoder Layer")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(label)
    annotate_heatmap(ax, mat, vmin, vmax, fmt=fmt)
    return im


# ===========================================================================
# Supplementary Figure S1: Root Cause Detail (2x3)
# ===========================================================================

def figure_S1(df_a, df_cdf):
    """S1: EZR, cosine_sim, l2_retention heatmaps + ratio bar + CDF + diagnosis."""
    zero_thresh = INP_BOUND / (2**DAC_BITS - 2)

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        "Supplementary Figure S1 — Root Cause Diagnosis Detail\n"
        f"(BERT-base, DAC={DAC_BITS}-bit, 6 sublayers)",
        fontsize=12, y=1.01,
    )

    # (a) EZR heatmap — YlOrRd, 0-1
    ezr_mat = to_mat(df_a, "EZR")
    _heatmap_panel(axes[0, 0], ezr_mat,
                   "(a) Exact Zero Rate (EZR)", "YlOrRd",
                   0.0, 1.0, "EZR")

    # (b) cosine_sim heatmap — RdYlGn, 0.99-1.0
    cos_mat = to_mat(df_a, "cosine_sim")
    _heatmap_panel(axes[0, 1], cos_mat,
                   "(b) Cosine Similarity", "RdYlGn",
                   0.99, 1.0, "cosine_sim", fmt=".4f")

    # (c) l2_retention heatmap — RdYlGn, 0.98-1.01
    l2r_mat = to_mat(df_a, "l2_retention")
    _heatmap_panel(axes[0, 2], l2r_mat,
                   r"(c) L2 Retention $\|{\delta_q}\| / \|{\delta}\|$", "RdYlGn",
                   0.98, 1.01, "l2_retention", fmt=".4f")

    # (d) ratio_q50 bar chart — per-sublayer mean + zero_thresh line
    ax = axes[1, 0]
    sublayer_means = {
        sl: float(df_a[df_a["sublayer"] == sl]["ratio_q50"].mean())
        if len(df_a[df_a["sublayer"] == sl]) > 0 else 0.0
        for sl in SUBLAYER_ORDER
    }
    colors6 = [SUBLAYER_COLORS[sl] for sl in SUBLAYER_ORDER]
    bars = ax.bar(SUBLAYER_ORDER,
                  [sublayer_means[sl] for sl in SUBLAYER_ORDER],
                  color=colors6, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(zero_thresh, color="red", ls="--", lw=1.5,
               label=f"zero_thresh = {zero_thresh:.5f}")
    for bar, sl in zip(bars, SUBLAYER_ORDER):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + zero_thresh * 0.05,
                f"{sublayer_means[sl]:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("ratio_q50 = median(|dy|/absmax)")
    ax.set_title("(d) Median ratio per sublayer\nratio < zero_thresh = quantized to zero",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (e) CDF worst-3 K layers
    ax = axes[1, 1]
    k_df = df_a[df_a["sublayer"] == "K"].sort_values("QZR_nonzero", ascending=False)
    worst3 = k_df.head(3)
    colors3 = ["#e41a1c", "#377eb8", "#4daf4a"]
    for (_, row), col in zip(worst3.iterrows(), colors3):
        li = int(row["layer_idx"])
        cdf_sub = df_cdf[(df_cdf["layer_idx"] == li) & (df_cdf["sublayer"] == "K")]
        if len(cdf_sub) == 0:
            continue
        cdf_sub = cdf_sub.sort_values("ratio")
        ax.plot(cdf_sub["ratio"].values, cdf_sub["cdf"].values,
                color=col, lw=1.5,
                label=f"K L{li}  QZR={row['QZR_nonzero']:.3f}")
    ax.axvline(zero_thresh, color="red", ls="--", lw=1.5,
               label=f"zero_thresh={zero_thresh:.5f}")
    ax.set_xlabel(r"$|\delta|$ / absmax ratio")
    ax.set_ylabel("CDF")
    ax.set_title("(e) CDF of gradient ratio — worst 3 K layers", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)

    # (f) Diagnosis text panel
    ax = axes[1, 2]
    ax.axis("off")
    kv = df_a[df_a["sublayer"].isin(["K", "V"])]
    ffn = df_a[df_a["sublayer"].isin(["FFN1", "FFN2"])]
    lines = [
        "Root Cause Diagnosis Summary",
        "=" * 36,
        f"DAC bits:     {DAC_BITS}",
        f"inp_res:      1/{2**DAC_BITS-2} = {zero_thresh:.6f}",
        f"step_size:    {2.0 * INP_BOUND / (2**DAC_BITS - 2):.6f}",
        f"zero_thresh:  {zero_thresh:.6f}",
        "",
        "K/V Statistics (mean across layers):",
        f"  EZR:          {kv['EZR'].mean():.4f}",
        f"  QZR_all:      {kv['QZR_all'].mean():.4f}",
        f"  QZR_nonzero:  {kv['QZR_nonzero'].mean():.4f}",
        f"  cosine_sim:   {kv['cosine_sim'].mean():.4f}",
        f"  l2_retention: {kv['l2_retention'].mean():.4f}",
        f"  ratio_q50:    {kv['ratio_q50'].mean():.6f}",
        "",
    ]
    if len(ffn) > 0:
        lines += [
            "FFN Statistics (mean across layers):",
            f"  EZR:          {ffn['EZR'].mean():.4f}",
            f"  QZR_nonzero:  {ffn['QZR_nonzero'].mean():.4f}",
            f"  cosine_sim:   {ffn['cosine_sim'].mean():.4f}",
            f"  l2_retention: {ffn['l2_retention'].mean():.4f}",
            "",
        ]
    # Verdict
    kv_qzr_mean = kv['QZR_nonzero'].mean()
    if kv_qzr_mean > 0.3:
        verdict = "SEVERE: K/V gradients heavily zeroed by DAC quantization"
    elif kv_qzr_mean > 0.1:
        verdict = "MODERATE: K/V gradient underflow present"
    else:
        verdict = "MILD: K/V gradient fidelity acceptable"
    lines += [f"Verdict: {verdict}"]

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    ax.set_title("(f) Diagnosis Summary", fontsize=10)

    plt.tight_layout()
    _save(fig, OUT_DIR, "supp_figure_S1", RUN_TAG)
    plt.close(fig)


# ===========================================================================
# Supplementary Figure S2: Bit-Sweep (2x2)
# ===========================================================================

def figure_S2(df_b):
    """S2: QZR vs bits, cosine vs bits, K QZR heatmap, V QZR heatmap."""
    bits_list = sorted(df_b["dac_bits"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Supplementary Figure S2 — IO Resolution Sweep\n"
        f"(BERT-base, bits={bits_list})",
        fontsize=12, y=1.01,
    )

    # (a) QZR_nonzero vs bits — line per sublayer
    ax = axes[0, 0]
    for sl in SUBLAYER_ORDER:
        sub = df_b[df_b["sublayer"] == sl].groupby("dac_bits")["QZR_nonzero"].mean()
        if len(sub) == 0:
            continue
        ax.plot(sub.index, sub.values,
                color=SUBLAYER_COLORS[sl], marker=SUBLAYER_MARKERS[sl],
                label=sl, lw=1.5, markersize=6)
    if 7 in bits_list:
        ax.axvline(7, color="gray", ls=":", lw=1.2, label="baseline 7b")
    ax.set_xlabel("bits (dac_bits)")
    ax.set_ylabel("QZR_nonzero")
    ax.set_title("(a) QZR_nonzero vs bits per sublayer", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bits_list)

    # (b) cosine_sim vs bits — line per sublayer
    ax = axes[0, 1]
    for sl in SUBLAYER_ORDER:
        sub = df_b[df_b["sublayer"] == sl].groupby("dac_bits")["cosine_sim"].mean()
        if len(sub) == 0:
            continue
        ax.plot(sub.index, sub.values,
                color=SUBLAYER_COLORS[sl], marker=SUBLAYER_MARKERS[sl],
                label=sl, lw=1.5, markersize=6)
    if 7 in bits_list:
        ax.axvline(7, color="gray", ls=":", lw=1.2, label="baseline 7b")
    ax.set_xlabel("bits (dac_bits)")
    ax.set_ylabel("cosine_sim")
    ax.set_title("(b) cosine_sim vs bits per sublayer", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bits_list)

    # (c) K QZR heatmap — bits(y) x layer(x), plasma 0-0.5
    ax = axes[1, 0]
    k_df = df_b[df_b["sublayer"] == "K"]
    mat_k = np.full((len(bits_list), N_LAYERS), np.nan)
    for bi, b in enumerate(bits_list):
        for li in range(N_LAYERS):
            sub = k_df[(k_df["dac_bits"] == b) & (k_df["layer_idx"] == li)]
            if len(sub) > 0:
                mat_k[bi, li] = sub["QZR_nonzero"].mean()
    im = ax.imshow(mat_k, aspect="auto", cmap="plasma", origin="upper",
                   vmin=0.0, vmax=0.5)
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_yticks(range(len(bits_list)))
    ax.set_yticklabels([f"{b}b" for b in bits_list])
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("bits")
    ax.set_title("(c) K QZR_nonzero: bits x layer", fontsize=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("QZR_nonzero")
    # Annotate
    for bi in range(mat_k.shape[0]):
        for li in range(mat_k.shape[1]):
            val = mat_k[bi, li]
            if np.isnan(val):
                continue
            norm = (val - 0.0) / (0.5 + 1e-12)
            color = "white" if norm > 0.6 or norm < 0.15 else "black"
            ax.text(li, bi, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    # (d) V QZR heatmap — bits(y) x layer(x), plasma 0-0.5
    ax = axes[1, 1]
    v_df = df_b[df_b["sublayer"] == "V"]
    mat_v = np.full((len(bits_list), N_LAYERS), np.nan)
    for bi, b in enumerate(bits_list):
        for li in range(N_LAYERS):
            sub = v_df[(v_df["dac_bits"] == b) & (v_df["layer_idx"] == li)]
            if len(sub) > 0:
                mat_v[bi, li] = sub["QZR_nonzero"].mean()
    im = ax.imshow(mat_v, aspect="auto", cmap="plasma", origin="upper",
                   vmin=0.0, vmax=0.5)
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_yticks(range(len(bits_list)))
    ax.set_yticklabels([f"{b}b" for b in bits_list])
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("bits")
    ax.set_title("(d) V QZR_nonzero: bits x layer", fontsize=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("QZR_nonzero")
    for bi in range(mat_v.shape[0]):
        for li in range(mat_v.shape[1]):
            val = mat_v[bi, li]
            if np.isnan(val):
                continue
            norm = (val - 0.0) / (0.5 + 1e-12)
            color = "white" if norm > 0.6 or norm < 0.15 else "black"
            ax.text(li, bi, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    plt.tight_layout()
    _save(fig, OUT_DIR, "supp_figure_S2", RUN_TAG)
    plt.close(fig)


# ===========================================================================
# Supplementary Figure S3: Solutions Comparison — Negative Results (2x3)
# ===========================================================================

def figure_S3(df_c):
    """S3: Grouped bars (QZR, rel_l2_error, l2_retention) + K/V heatmaps + delta text."""
    # Only keep variants present in data
    variants_present = [v for v in VARIANTS if v in df_c["variant"].unique()]
    colors_present = VARIANT_COLORS[:len(variants_present)]
    kv_df = df_c[df_c["sublayer"].isin(["K", "V"])]

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        "Supplementary Figure S3 — Solutions Comparison (Negative Results)\n"
        f"(BERT-base, DAC={DAC_BITS}-bit, variants: {variants_present})",
        fontsize=12, y=1.01,
    )

    # --- Grouped bar helper ---
    def _grouped_bar(ax, metric, title, ylabel, higher_better=False):
        n_sl = 2  # K, V
        x = np.arange(n_sl)
        n_var = len(variants_present)
        width = 0.8 / n_var
        offsets = np.linspace(-(n_var - 1) * width / 2, (n_var - 1) * width / 2, n_var)

        for vi, (variant, color) in enumerate(zip(variants_present, colors_present)):
            var_df = kv_df[kv_df["variant"] == variant]
            vals = []
            for sl in ["K", "V"]:
                sub = var_df[var_df["sublayer"] == sl][metric]
                vals.append(float(sub.mean()) if len(sub) > 0 else float("nan"))
            bars = ax.bar(x + offsets[vi], vals, width=width, label=variant,
                          color=color, alpha=0.8, edgecolor="k", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{val:.4f}", ha="center", va="bottom",
                            fontsize=7, rotation=45)
        ax.set_xticks(x)
        ax.set_xticklabels(["K", "V"])
        ax.set_xlabel("Sublayer")
        ax.set_ylabel(ylabel)
        better = "higher = better" if higher_better else "lower = better"
        ax.set_title(f"{title}\n({better})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # (a) K/V QZR grouped bar
    _grouped_bar(axes[0, 0], "QZR_nonzero",
                 "(a) K/V Mean QZR_nonzero per variant",
                 "QZR_nonzero", higher_better=False)

    # (b) K/V rel_l2_error grouped bar
    _grouped_bar(axes[0, 1], "rel_l2_error",
                 "(b) K/V Mean rel_l2_error per variant",
                 "rel_l2_error", higher_better=False)

    # (c) K/V l2_retention grouped bar
    _grouped_bar(axes[0, 2], "l2_retention",
                 "(c) K/V Mean l2_retention per variant",
                 "l2_retention", higher_better=True)

    # --- Heatmap: variant(y) x layer(x) ---
    def _variant_layer_heatmap(ax, sublayer, title):
        mat = np.full((len(variants_present), N_LAYERS), np.nan)
        sub_df = df_c[df_c["sublayer"] == sublayer]
        for vi, variant in enumerate(variants_present):
            v_df = sub_df[sub_df["variant"] == variant]
            for _, row in v_df.iterrows():
                li = int(row["layer_idx"])
                if li < N_LAYERS:
                    mat[vi, li] = row["QZR_nonzero"]
        im = ax.imshow(mat, aspect="auto", cmap="plasma", origin="upper",
                       vmin=0.0, vmax=0.5)
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(variants_present)))
        ax.set_yticklabels(variants_present, fontsize=9)
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("Variant")
        ax.set_title(title, fontsize=10)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("QZR_nonzero")
        # Annotate
        for vi in range(mat.shape[0]):
            for li in range(mat.shape[1]):
                val = mat[vi, li]
                if np.isnan(val):
                    continue
                norm = (val - 0.0) / (0.5 + 1e-12)
                color = "white" if norm > 0.6 or norm < 0.15 else "black"
                ax.text(li, vi, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold")

    # (d) K QZR heatmap: variant x layer
    _variant_layer_heatmap(axes[1, 0], "K",
                           "(d) K QZR_nonzero: variant x layer")

    # (e) V QZR heatmap: variant x layer
    _variant_layer_heatmap(axes[1, 1], "V",
                           "(e) V QZR_nonzero: variant x layer")

    # (f) Delta summary text
    ax = axes[1, 2]
    ax.axis("off")
    base_df = kv_df[kv_df["variant"] == "baseline"]
    base_k_qzr = base_df[base_df["sublayer"] == "K"]["QZR_nonzero"].mean()
    base_v_qzr = base_df[base_df["sublayer"] == "V"]["QZR_nonzero"].mean()
    base_k_l2r = base_df[base_df["sublayer"] == "K"]["l2_retention"].mean()
    base_v_l2r = base_df[base_df["sublayer"] == "V"]["l2_retention"].mean()
    base_k_cos = base_df[base_df["sublayer"] == "K"]["cosine_sim"].mean()
    base_v_cos = base_df[base_df["sublayer"] == "V"]["cosine_sim"].mean()

    lines = [
        "Delta vs Baseline Summary",
        "=" * 50,
        "",
        f"{'Variant':<15} {'K_QZR':>8} {'dK_QZR':>8} {'V_QZR':>8} {'dV_QZR':>8}",
        "-" * 50,
    ]
    for variant in variants_present:
        v_kv = kv_df[kv_df["variant"] == variant]
        k_q = v_kv[v_kv["sublayer"] == "K"]["QZR_nonzero"].mean()
        v_q = v_kv[v_kv["sublayer"] == "V"]["QZR_nonzero"].mean()
        dk = k_q - base_k_qzr
        dv = v_q - base_v_qzr
        lines.append(f"{variant:<15} {k_q:>8.4f} {dk:>+8.4f} {v_q:>8.4f} {dv:>+8.4f}")
    lines += [
        "",
        f"{'Variant':<15} {'K_cos':>8} {'dK_cos':>8} {'V_cos':>8} {'dV_cos':>8}",
        "-" * 50,
    ]
    for variant in variants_present:
        v_kv = kv_df[kv_df["variant"] == variant]
        k_c = v_kv[v_kv["sublayer"] == "K"]["cosine_sim"].mean()
        v_c = v_kv[v_kv["sublayer"] == "V"]["cosine_sim"].mean()
        lines.append(
            f"{variant:<15} {k_c:>8.4f} {k_c - base_k_cos:>+8.4f} "
            f"{v_c:>8.4f} {v_c - base_v_cos:>+8.4f}"
        )
    lines += [
        "",
        f"{'Variant':<15} {'K_l2r':>8} {'dK_l2r':>8} {'V_l2r':>8} {'dV_l2r':>8}",
        "-" * 50,
    ]
    for variant in variants_present:
        v_kv = kv_df[kv_df["variant"] == variant]
        k_l = v_kv[v_kv["sublayer"] == "K"]["l2_retention"].mean()
        v_l = v_kv[v_kv["sublayer"] == "V"]["l2_retention"].mean()
        lines.append(
            f"{variant:<15} {k_l:>8.4f} {k_l - base_k_l2r:>+8.4f} "
            f"{v_l:>8.4f} {v_l - base_v_l2r:>+8.4f}"
        )

    ax.text(0.03, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    ax.set_title("(f) Delta vs Baseline", fontsize=10)

    plt.tight_layout()
    _save(fig, OUT_DIR, "supp_figure_S3", RUN_TAG)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    plt.rcParams.update(RCPARAMS)
    os.makedirs(OUT_DIR, exist_ok=True)

    if RUN_S1:
        print("\n[S1] Root Cause Detail ...")
        df_a = pd.read_csv(CSV_A_SUMMARY)
        df_cdf = pd.read_csv(CSV_A_CDF)
        print(f"  A_summary: {len(df_a)} rows, A_cdf: {len(df_cdf)} rows")
        figure_S1(df_a, df_cdf)

    if RUN_S2:
        print("\n[S2] Bit-Sweep ...")
        df_b = pd.read_csv(CSV_B_SUMMARY)
        print(f"  B_summary: {len(df_b)} rows")
        figure_S2(df_b)

    if RUN_S3:
        print("\n[S3] Solutions Comparison ...")
        df_c = pd.read_csv(CSV_C_SUMMARY)
        print(f"  C_summary: {len(df_c)} rows")
        figure_S3(df_c)

    print("\nDone.")


if __name__ == "__main__":
    main()

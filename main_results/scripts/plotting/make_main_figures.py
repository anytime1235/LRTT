"""make_main_figures.py — Main Figure 1 for paper (1x3 layout).

Panels:
  (a) QZR_nonzero heatmap (12 layer x 6 sublayer)
  (b) ODR heatmap (log10 scale, 12x6)
  (c) Worst-K CDF (Layer 7,9,10) + zero_thresh line

Usage:
  python make_main_figures.py --out-dir /data/results/tikitakav1 --run-tag v3
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
parser = argparse.ArgumentParser(description="Main Figure 1 — AIMC BERT-base")
parser.add_argument("--out-dir", type=str, default="/data/results/tikitakav1")
parser.add_argument("--run-tag", type=str, default="v3")
args = parser.parse_args()

OUT_DIR = args.out_dir
RUN_TAG = args.run_tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LAYERS = 12
DAC_BITS = 7
INP_BOUND = 1.0
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]

CSV_A_SUMMARY = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_summary.csv")
CSV_A_CDF = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_cdf.csv")

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


def annotate_heatmap(ax, mat, vmin, vmax, fmt=".3f", fontsize=7):
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


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def panel_a_qzr(ax, df_a):
    """(a) QZR_nonzero heatmap — plasma, 0-0.5."""
    mat = to_mat(df_a, "QZR_nonzero")
    vmin, vmax = 0.0, 0.5
    im = ax.imshow(mat, aspect="auto", cmap="plasma", origin="upper",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("Encoder Layer")
    ax.set_title("(a) Quantization Zero Rate (QZR)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("QZR (fraction rounded to 0)")
    annotate_heatmap(ax, mat, vmin, vmax)
    return im


def panel_b_odr(ax, df_a):
    """(b) ODR heatmap — log10, hot_r, auto range."""
    odr_mat = to_mat(df_a, "ODR")
    log_odr = np.log10(np.clip(odr_mat, 1e-3, None))
    vmin, vmax = float(np.nanmin(log_odr)), float(np.nanmax(log_odr))
    im = ax.imshow(log_odr, aspect="auto", cmap="hot_r", origin="upper",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("Encoder Layer")
    ax.set_title(r"(b) Outlier Dominance Ratio (log$_{10}$ ODR)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(r"log$_{10}$(ODR)")
    annotate_heatmap(ax, log_odr, vmin, vmax, fmt=".1f")
    return im


def panel_c_cdf(ax, df_cdf, df_a):
    """(c) Worst-K CDF curves + zero_thresh vertical line."""
    zero_thresh = INP_BOUND / (2**DAC_BITS - 2)

    # Find worst-3 K layers from summary
    k_df = df_a[df_a["sublayer"] == "K"].sort_values("QZR_nonzero", ascending=False)
    worst3 = k_df.head(3)

    colors3 = ["#e41a1c", "#377eb8", "#4daf4a"]

    for (_, row), col in zip(worst3.iterrows(), colors3):
        li = int(row["layer_idx"])
        sl = row["sublayer"]
        # Filter CDF data for this layer+sublayer
        cdf_sub = df_cdf[(df_cdf["layer_idx"] == li) & (df_cdf["sublayer"] == sl)]
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
    ax.set_title("(c) CDF of gradient ratio — worst K layers")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plt.rcParams.update(RCPARAMS)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading CSVs ...")
    df_a = pd.read_csv(CSV_A_SUMMARY)
    df_cdf = pd.read_csv(CSV_A_CDF)
    print(f"  A_summary: {len(df_a)} rows")
    print(f"  A_cdf:     {len(df_cdf)} rows")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    panel_a_qzr(axes[0], df_a)
    panel_b_odr(axes[1], df_a)
    panel_c_cdf(axes[2], df_cdf, df_a)

    plt.tight_layout()
    _save(fig, OUT_DIR, "main_figure1", RUN_TAG)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()

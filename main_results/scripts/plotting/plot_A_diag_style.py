"""Reproduce diag_backward_outlier.py figure style from paper_figures CSV A.

Layout identical to diag_backward_outlier.py create_figure():
  [0,0] Heatmap: ODR (log10) per layer × sublayer
  [0,1] Heatmap: QZR          per layer × sublayer
  [0,2] Scatter: ODR vs QZR   (color = encoder depth)
  [1,0] absmax quantile profile — Top-1 worst layer (by QZR_nz)
  [1,1] absmax quantile profile — Top-2 worst layer
  [1,2] absmax quantile profile — Top-3 worst layer

Note: ECDF requires raw per-vector absmax which is not in CSV.
      Substituted with absmax quantile bar profiles (q50/q90/q99/q999).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/data/results/tikitakav1"
CSV_A   = f"{OUT_DIR}/metrics_paper_A_rootcause.csv"
FIG_OUT = f"{OUT_DIR}/fig_A_diag_backward_outlier_style.png"

df = pd.read_csv(CSV_A)

SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]
N_LAYERS = 12
DAC_BITS = 7
INP_BOUND = 1.0


def to_mat(col):
    mat = np.full((N_LAYERS, len(SUBLAYER_ORDER)), np.nan)
    for _, row in df.iterrows():
        li = int(row["layer_idx"])
        sl = row["sublayer"]
        if sl in SUBLAYER_ORDER and li < N_LAYERS:
            mat[li, SUBLAYER_ORDER.index(sl)] = row[col]
    return mat


def annotate(ax, mat, vmin, vmax, fmt=".3f"):
    for li in range(mat.shape[0]):
        for si in range(mat.shape[1]):
            val = mat[li, si]
            if np.isnan(val):
                continue
            txt = f"{val:{fmt}}"
            norm = (val - vmin) / (vmax - vmin + 1e-12)
            color = "white" if norm > 0.6 or norm < 0.15 else "black"
            ax.text(si, li, txt, ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")


# --- Top-3 worst layers by QZR_nonzero_mean ---
top3 = df.nlargest(3, "QZR_nonzero_mean")[
    ["layer_name", "layer_idx", "sublayer", "QZR_nonzero_mean", "ODR_mean",
     "absmax_q50", "absmax_q90", "absmax_q99", "absmax_q999", "p_clip",
     "cosine_sim"]
].reset_index(drop=True)

# DAC zero threshold
dac_thresh = INP_BOUND / (2**DAC_BITS - 1)   # Δ/2

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    "Backward Gradient Outlier Diagnosis — BERT-base QKVO+FFN Analog Tiles\n"
    f"(AbsMax noise management, nm_thres=0, DAC={DAC_BITS}-bit, ADC=9-bit, "
    f"N=200 steps, batch=8, 6 sublayers)",
    fontsize=11, y=1.01,
)

# ─── [0,0] Heatmap ODR (log10) ───
ax = axes[0, 0]
odr_mat = to_mat("ODR_mean")
log_odr = np.log10(np.clip(odr_mat, 1e-3, None))
vmin_o, vmax_o = np.nanmin(log_odr), np.nanmax(log_odr)
im = ax.imshow(log_odr, aspect="auto", cmap="hot_r", origin="upper")
ax.set_xticks(range(6)); ax.set_xticklabels(SUBLAYER_ORDER, fontsize=9)
ax.set_yticks(range(N_LAYERS))
ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=8)
ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
ax.set_title(r"(a) ODR$_\ell$ — Outlier Dominance Ratio (log$_{10}$)", fontsize=10)
plt.colorbar(im, ax=ax, label=r"log$_{10}$(ODR)", shrink=0.85)
annotate(ax, log_odr, vmin_o, vmax_o, fmt=".1f")

# ─── [0,1] Heatmap QZR ───
ax = axes[0, 1]
qzr_mat = to_mat("QZR_nonzero_mean")
im = ax.imshow(qzr_mat, aspect="auto", cmap="plasma",
               origin="upper", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(SUBLAYER_ORDER, fontsize=9)
ax.set_yticks(range(N_LAYERS))
ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=8)
ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
ax.set_title(f"(b) QZR — Quant. Zero Rate (DAC {DAC_BITS}-bit, AbsMax)", fontsize=10)
plt.colorbar(im, ax=ax, label="QZR (fraction rounded to 0)", shrink=0.85)
annotate(ax, qzr_mat, 0, 1)

# ─── [0,2] Scatter: ODR vs QZR (color = layer depth) ───
ax = axes[0, 2]
sc = ax.scatter(
    df["ODR_mean"], df["QZR_nonzero_mean"],
    c=df["layer_idx"], cmap="viridis",
    alpha=0.8, s=60, edgecolors="k", linewidths=0.4,
)
ax.set_xlabel(r"ODR$_\ell$ (Outlier Dominance Ratio)")
ax.set_ylabel(r"QZR$_\ell$ (Quantization Zero Rate)")
ax.set_title("(c) ODR vs QZR — Outlier→Collapse Correlation", fontsize=10)
ax.set_xscale("log")
for _, row in top3.iterrows():
    ax.annotate(
        f"L{int(row['layer_idx'])}{row['sublayer']}",
        (row["ODR_mean"], row["QZR_nonzero_mean"]),
        fontsize=8, fontweight="bold",
        xytext=(5, 5), textcoords="offset points",
    )
plt.colorbar(sc, ax=ax, label="Encoder depth")
ax.grid(True, alpha=0.3)

# ─── [1,0..2] absmax quantile profile — Top-3 worst layers ───
q_names   = ["q50", "q90", "q99", "q999"]
q_cols    = ["absmax_q50", "absmax_q90", "absmax_q99", "absmax_q999"]
bar_color = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]

for k, (panel_ax, (_, row)) in enumerate(zip(axes[1, :3], top3.iterrows())):
    li   = int(row["layer_idx"])
    sl   = row["sublayer"]
    qzr  = row["QZR_nonzero_mean"]
    label = f"L{li}{sl} (QZR={qzr:.3f})"

    vals = [row[c] for c in q_cols]

    bars = panel_ax.bar(q_names, vals, color=bar_color, alpha=0.85,
                        edgecolor="k", linewidth=0.5)

    # DAC zero threshold line
    panel_ax.axhline(dac_thresh, color="red", ls="--", lw=1.5,
                     label=f"Δ/2 = {dac_thresh:.4f}\n(DAC zero threshold)")

    # inp_bound line
    panel_ax.axhline(INP_BOUND, color="darkgreen", ls=":", lw=1.2,
                     label=f"inp_bound = {INP_BOUND}")

    # value labels on bars
    for bar, val in zip(bars, vals):
        if val > 0:
            panel_ax.text(bar.get_x() + bar.get_width() / 2,
                          val * 1.05,
                          f"{val:.2e}" if val < 0.01 else f"{val:.4f}",
                          ha="center", va="bottom", fontsize=8, fontweight="bold")

    panel_ax.set_xlabel("absmax quantile")
    panel_ax.set_ylabel(r"$\|\delta_{vec}\|_\infty$")
    panel_ax.set_title(f"(d{k+1}) {label}", fontsize=10)
    panel_ax.legend(fontsize=7, loc="upper left")
    panel_ax.set_yscale("log")
    panel_ax.grid(True, alpha=0.3, axis="y")

    # extra info
    panel_ax.text(0.98, 0.02,
                  f"p_clip={row['p_clip']:.4f}\n"
                  f"cosine={row['cosine_sim']:.4f}\n"
                  f"ODR={row['ODR_mean']:.1f}",
                  transform=panel_ax.transAxes,
                  ha="right", va="bottom", fontsize=8,
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIG_OUT}")

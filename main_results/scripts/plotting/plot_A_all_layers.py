"""Plot all 72 layers from CSV A — layer_idx × sublayer with key metrics visible."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_A = "/data/results/tikitakav1/metrics_paper_A_rootcause.csv"
OUT   = "/data/results/tikitakav1/fig_A_all_layers_detail.png"

df = pd.read_csv(CSV_A)
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]
N_LAYERS = 12

# Build 12×6 matrices
def to_mat(col):
    mat = np.full((N_LAYERS, len(SUBLAYER_ORDER)), np.nan)
    for _, row in df.iterrows():
        li = int(row["layer_idx"])
        sl = row["sublayer"]
        if sl in SUBLAYER_ORDER and li < N_LAYERS:
            mat[li, SUBLAYER_ORDER.index(sl)] = row[col]
    return mat

metrics = [
    ("QZR_nonzero_mean", "QZR_nonzero (lower=better)", "plasma", 0, 1),
    ("cosine_sim",       "Cosine Similarity (higher=better)", "RdYlGn", 0, 1),
    ("ODR_mean",         "ODR (Outlier Dominance Ratio)", "hot_r", None, None),
    ("EZR_mean",         "EZR (Exact Zero Ratio)", "YlOrRd", 0, 1),
    ("p_clip",           "P(clip) = P(|δ|>1)", "Reds", 0, None),
    ("absmax_q99",       "absmax q99", "magma", None, None),
]

fig, axes = plt.subplots(2, 3, figsize=(26, 16))
fig.suptitle(
    f"Figure A Detail — All 72 Layers (12 encoder × 6 sublayers)\n"
    f"DAC=7-bit, ADC=9-bit, 200 steps × batch 8, baseline (nm_thres=0)",
    fontsize=13, y=1.01
)

for ax, (col, title, cmap, vmin, vmax) in zip(axes.flat, metrics):
    mat = to_mat(col)

    # auto range if None
    if vmin is None:
        vmin = np.nanmin(mat)
    if vmax is None:
        vmax = np.nanmax(mat)

    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER, fontsize=10, fontweight="bold")
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=9)
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("Encoder Layer")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Annotate each cell with numeric value
    for li in range(N_LAYERS):
        for si in range(len(SUBLAYER_ORDER)):
            val = mat[li, si]
            if np.isnan(val):
                continue
            # Format: use scientific for very small, otherwise 3-4 digits
            if abs(val) < 0.001 and val != 0:
                txt = f"{val:.1e}"
            elif abs(val) >= 100:
                txt = f"{val:.0f}"
            elif abs(val) >= 10:
                txt = f"{val:.1f}"
            else:
                txt = f"{val:.3f}"
            # contrast color
            norm_val = (val - vmin) / (vmax - vmin + 1e-12)
            color = "white" if norm_val > 0.6 or norm_val < 0.15 else "black"
            ax.text(si, li, txt, ha="center", va="center",
                    fontsize=6.5, color=color, fontweight="bold")

plt.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}")

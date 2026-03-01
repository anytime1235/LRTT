"""
Plot all CSVs in diag_fwd_io — one figure per CSV file, all columns visualized.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

OUT_DIR = Path("/data/main_results/results/figures/diag_fwd_io")
OUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("/data/main_results/results/csv/diag_fwd_io")

SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]
ADC_LABELS     = ["adc4", "adc6", "adc8", "adc10", "adc12"]
ADC_BITS       = [4, 6, 8, 10, 12]
ADC_COLORS     = dict(zip(ADC_LABELS, cm.viridis(np.linspace(0.1, 0.9, 5))))

METRIC_COLS_MAC = [
    "mac_snr_db", "mac_nmse", "cosine",
    "out_clip_ratio", "ref_deadzone_ratio",
    "mean_abs_err", "median_abs_err", "p95_abs_err",
]

# ─────────────────────────────────────────────────────────────
# Figure 1 : summary_adc_sweep.csv
# ─────────────────────────────────────────────────────────────
def plot_summary_adc_sweep():
    df = pd.read_csv(DATA_DIR / "summary_adc_sweep.csv")
    sublayers = ["Q", "K", "V", "O", "FFN1", "FFN2"]

    snr_cols   = [f"mac_snr_{s}_mean"      for s in sublayers]
    clip_cols  = [f"out_clip_ratio_{s}_mean" for s in sublayers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ADC Sweep Summary  (summary_adc_sweep.csv)", fontsize=13, fontweight="bold")

    # — subplot 1: MAC SNR per sublayer vs ADC bits
    ax = axes[0]
    for col, sl in zip(snr_cols, sublayers):
        ax.plot(df["adc_bits"], df[col], marker="o", label=sl)
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("MAC SNR (dB)")
    ax.set_title("MAC SNR per Sublayer vs ADC Bits")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # — subplot 2: Out Clip Ratio per sublayer vs ADC bits
    ax = axes[1]
    for col, sl in zip(clip_cols, sublayers):
        ax.semilogy(df["adc_bits"], df[col].replace(0, np.nan), marker="o", label=sl)
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("Out Clip Ratio (log scale)")
    ax.set_title("Out Clip Ratio per Sublayer vs ADC Bits")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # — subplot 3: mean MAC SNR & mean clip ratio on twin axes
    ax = axes[2]
    color1, color2 = "steelblue", "tomato"
    ax.plot(df["adc_bits"], df["mac_snr_mean"], marker="o", color=color1, label="mac_snr_mean")
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("Mean MAC SNR (dB)", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax2 = ax.twinx()
    ax2.semilogy(df["adc_bits"], df["out_clip_ratio_mean"].replace(0, np.nan),
                 marker="s", linestyle="--", color=color2, label="out_clip_ratio_mean")
    ax2.set_ylabel("Mean Out Clip Ratio (log)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax.set_title("Mean MAC SNR & Clip Ratio vs ADC Bits")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig1_summary_adc_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# Figure 2 : adc{N}_module_mac_summary.csv  (all 5 combined)
# ─────────────────────────────────────────────────────────────
def plot_module_mac_summary():
    frames = []
    for lbl in ADC_LABELS:
        d = pd.read_csv(DATA_DIR / f"{lbl}_module_mac_summary.csv")
        d["label"] = lbl
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["sublayer"] = pd.Categorical(df["sublayer"], categories=SUBLAYER_ORDER, ordered=True)
    df = df.sort_values(["label", "layer_idx", "sublayer"])

    metrics = METRIC_COLS_MAC
    n = len(metrics)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Module MAC Summary  (adc4/6/8/10/12_module_mac_summary.csv)", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        for lbl in ADC_LABELS:
            sub = df[df["label"] == lbl]
            grp = sub.groupby("layer_idx")[metric].mean()
            ax.plot(grp.index, grp.values, marker=".", label=lbl, color=ADC_COLORS[lbl], linewidth=1.2)
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if metric in ("mac_nmse", "out_clip_ratio", "ref_deadzone_ratio"):
            ax.set_yscale("symlog", linthresh=1e-6)

    # last subplot: sublayer-wise mac_snr_db heatmap per ADC
    ax = axes_flat[-1]
    pivot = df.groupby(["label", "sublayer"])["mac_snr_db"].mean().unstack("sublayer")
    pivot = pivot.reindex(columns=SUBLAYER_ORDER)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER, fontsize=8)
    ax.set_yticks(range(len(ADC_LABELS)))
    ax.set_yticklabels(ADC_LABELS, fontsize=8)
    ax.set_title("MAC SNR (dB) — sublayer × ADC heatmap", fontsize=10)
    for ii in range(len(ADC_LABELS)):
        for jj in range(len(SUBLAYER_ORDER)):
            ax.text(jj, ii, f"{pivot.values[ii, jj]:.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="MAC SNR (dB)")

    plt.tight_layout()
    out = OUT_DIR / "fig2_module_mac_summary_all_adc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# Figure 3 : adc{N}_layer_mac_metrics.csv  (step-aggregated)
# ─────────────────────────────────────────────────────────────
def plot_layer_mac_metrics():
    print("Loading adc layer_mac_metrics (large files)...")
    frames = []
    for lbl in ADC_LABELS:
        d = pd.read_csv(DATA_DIR / f"{lbl}_layer_mac_metrics.csv")
        d["label"] = lbl
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # aggregate: mean over all steps per (label, layer_idx, sublayer)
    grp = df.groupby(["label", "layer_idx", "sublayer"])[METRIC_COLS_MAC].mean().reset_index()
    grp["sublayer"] = pd.Categorical(grp["sublayer"], categories=SUBLAYER_ORDER, ordered=True)

    metrics = METRIC_COLS_MAC
    n_metrics = len(metrics)
    n_sublayers = len(SUBLAYER_ORDER)

    fig, axes = plt.subplots(n_metrics, n_sublayers, figsize=(28, 22), sharex="col")
    fig.suptitle("Layer MAC Metrics (mean over steps)  (adc*_layer_mac_metrics.csv)",
                 fontsize=13, fontweight="bold")

    for row, metric in enumerate(metrics):
        for col, sl in enumerate(SUBLAYER_ORDER):
            ax = axes[row][col]
            sub = grp[grp["sublayer"] == sl]
            for lbl in ADC_LABELS:
                d = sub[sub["label"] == lbl].sort_values("layer_idx")
                ax.plot(d["layer_idx"], d[metric], marker=".", linewidth=1,
                        label=lbl, color=ADC_COLORS[lbl])
            if row == 0:
                ax.set_title(sl, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(metric, fontsize=8)
            if row == n_metrics - 1:
                ax.set_xlabel("Layer idx", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if metric in ("mac_nmse", "out_clip_ratio", "ref_deadzone_ratio"):
                ax.set_yscale("symlog", linthresh=1e-6)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    plt.tight_layout()
    out = OUT_DIR / "fig3_layer_mac_metrics_all_adc.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# Figure 4 : single_run_layer_mac_metrics.csv
# ─────────────────────────────────────────────────────────────
def plot_single_run_layer_mac():
    df = pd.read_csv(DATA_DIR / "single_run_layer_mac_metrics.csv")
    df["sublayer"] = pd.Categorical(df["sublayer"], categories=SUBLAYER_ORDER, ordered=True)

    metrics = METRIC_COLS_MAC
    n_metrics = len(metrics)
    n_sublayers = len(SUBLAYER_ORDER)

    fig, axes = plt.subplots(n_metrics, n_sublayers, figsize=(28, 22), sharex="col")
    fig.suptitle("Single Run — Layer MAC Metrics  (single_run_layer_mac_metrics.csv)",
                 fontsize=13, fontweight="bold")

    colors = cm.tab10(np.linspace(0, 1, df["step"].nunique())) if "step" in df.columns else ["steelblue"]
    steps  = sorted(df["step"].unique()) if "step" in df.columns else [None]

    for row, metric in enumerate(metrics):
        for col, sl in enumerate(SUBLAYER_ORDER):
            ax = axes[row][col]
            sub = df[df["sublayer"] == sl].sort_values("layer_idx")
            if "step" in df.columns:
                for s, c in zip(steps, colors):
                    d = sub[sub["step"] == s]
                    ax.plot(d["layer_idx"], d[metric], marker=".", linewidth=1,
                            label=f"step {s}", color=c)
            else:
                ax.plot(sub["layer_idx"], sub[metric], marker=".", linewidth=1)
            if row == 0:
                ax.set_title(sl, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(metric, fontsize=8)
            if row == n_metrics - 1:
                ax.set_xlabel("Layer idx", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if metric in ("mac_nmse", "out_clip_ratio", "ref_deadzone_ratio"):
                ax.set_yscale("symlog", linthresh=1e-6)
            if row == 0 and col == 0 and "step" in df.columns:
                ax.legend(fontsize=6)

    plt.tight_layout()
    out = OUT_DIR / "fig4_single_run_layer_mac_metrics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# Figure 5 : single_run_module_mac_summary.csv
# ─────────────────────────────────────────────────────────────
def plot_single_run_module_mac_summary():
    df = pd.read_csv(DATA_DIR / "single_run_module_mac_summary.csv")
    df["sublayer"] = pd.Categorical(df["sublayer"], categories=SUBLAYER_ORDER, ordered=True)

    metrics = METRIC_COLS_MAC
    n = len(metrics)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Single Run — Module MAC Summary  (single_run_module_mac_summary.csv)",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()
    cmap = cm.get_cmap("tab10", len(SUBLAYER_ORDER))

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        for j, sl in enumerate(SUBLAYER_ORDER):
            sub = df[df["sublayer"] == sl].sort_values("layer_idx")
            ax.plot(sub["layer_idx"], sub[metric], marker=".", label=sl,
                    color=cmap(j), linewidth=1.3)
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if metric in ("mac_nmse", "out_clip_ratio", "ref_deadzone_ratio"):
            ax.set_yscale("symlog", linthresh=1e-6)

    # last subplot: heatmap sublayer × layer_idx for mac_snr_db
    ax = axes_flat[-1]
    pivot = df.pivot_table(index="sublayer", columns="layer_idx", values="mac_snr_db", aggfunc="mean")
    pivot = pivot.reindex(index=SUBLAYER_ORDER)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), fontsize=6)
    ax.set_yticks(range(len(SUBLAYER_ORDER)))
    ax.set_yticklabels(SUBLAYER_ORDER, fontsize=8)
    ax.set_title("MAC SNR (dB) heatmap", fontsize=10)
    plt.colorbar(im, ax=ax, label="MAC SNR (dB)")

    plt.tight_layout()
    out = OUT_DIR / "fig5_single_run_module_mac_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# Figure 6 : single_run_logit_metrics.csv
# ─────────────────────────────────────────────────────────────
def plot_single_run_logit_metrics():
    df = pd.read_csv(DATA_DIR / "single_run_logit_metrics.csv")
    # Only 1 row — show all columns as bar chart + print values as table
    metrics = [c for c in df.columns if c != "step"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Single Run — Logit Metrics  (single_run_logit_metrics.csv)",
                 fontsize=13, fontweight="bold")

    # bar chart for each metric value
    ax = axes[0]
    vals = df[metrics].iloc[0].values
    colors_bar = ["tomato" if v < 0 else "steelblue" for v in vals]
    bars = ax.bar(range(len(metrics)), vals, color=colors_bar)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_title("All Logit Metrics (single sample)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # pair comparison: start vs end metrics
    ax = axes[1]
    pair_metrics = [("mse_start", "mse_end"),
                    ("kl_start", "kl_end"),
                    ("flip_start", "flip_end")]
    labels_pair, before_vals, after_vals = [], [], []
    for s, e in pair_metrics:
        if s in df.columns and e in df.columns:
            labels_pair.append(s.replace("_start", ""))
            before_vals.append(df[s].iloc[0])
            after_vals.append(df[e].iloc[0])

    x = np.arange(len(labels_pair))
    w = 0.35
    ax.bar(x - w/2, before_vals, w, label="before (start)", color="steelblue")
    ax.bar(x + w/2, after_vals,  w, label="after  (end)",   color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_pair, fontsize=10)
    ax.set_title("Before vs After Metrics")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig6_single_run_logit_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# Figure 7 : single_run_weight_delta_metrics.csv
# ─────────────────────────────────────────────────────────────
def plot_single_run_weight_delta():
    df = pd.read_csv(DATA_DIR / "single_run_weight_delta_metrics.csv")
    df["sublayer"] = pd.Categorical(df["sublayer"], categories=SUBLAYER_ORDER, ordered=True)

    metrics = ["dw_zero_ratio", "dw_1lsb_ratio", "dw_absmean", "min_nonzero_delta"]
    n_metrics = len(metrics)
    n_sublayers = len(SUBLAYER_ORDER)

    steps   = sorted(df["step"].unique())
    cmap    = cm.get_cmap("tab10", len(steps))
    colors_step = {s: cmap(i) for i, s in enumerate(steps)}

    fig, axes = plt.subplots(n_metrics, n_sublayers, figsize=(24, 14), sharex="col")
    fig.suptitle("Single Run — Weight Delta Metrics  (single_run_weight_delta_metrics.csv)",
                 fontsize=13, fontweight="bold")

    for row, metric in enumerate(metrics):
        for col, sl in enumerate(SUBLAYER_ORDER):
            ax = axes[row][col]
            sub = df[df["sublayer"] == sl].sort_values(["step", "layer_idx"])
            for s in steps:
                d = sub[sub["step"] == s].sort_values("layer_idx")
                ax.plot(d["layer_idx"], d[metric], marker=".", linewidth=1,
                        label=f"step {s}", color=colors_step[s])
            if row == 0:
                ax.set_title(sl, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(metric, fontsize=8)
            if row == n_metrics - 1:
                ax.set_xlabel("Layer idx", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if metric in ("dw_absmean", "min_nonzero_delta"):
                ax.set_yscale("log")
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    plt.tight_layout()
    out = OUT_DIR / "fig7_single_run_weight_delta.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Generating plots for all CSVs in diag_fwd_io ===\n")
    plot_summary_adc_sweep()
    plot_module_mac_summary()
    plot_layer_mac_metrics()
    plot_single_run_layer_mac()
    plot_single_run_module_mac_summary()
    plot_single_run_logit_metrics()
    plot_single_run_weight_delta()
    print(f"\nAll plots saved to: {OUT_DIR}")

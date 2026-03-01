"""plot_mitigation_adc_io.py — Compare baseline vs mitigation sweep results.

Usage:
  python plot_mitigation_adc_io.py \
      --results-dir ./results/diag_fwd_io_mitigations \
      --tags baseline,obcal_per_module,mp_base6 \
      --out-dir ./plots/diag_fwd_io_mitigations
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="./results/diag_fwd_io_mitigations")
parser.add_argument("--tags",        default="baseline,obcal_per_module,mp_base6",
    help="Comma-separated tags to compare")
parser.add_argument("--out-dir",     default="./plots/diag_fwd_io_mitigations")
parser.add_argument("--dpi",         type=int, default=300)
args = parser.parse_args()

RESULTS = Path(args.results_dir)
PLOTS   = Path(args.out_dir)
PLOTS.mkdir(parents=True, exist_ok=True)

TAGS = [t.strip() for t in args.tags.split(",")]
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]
ADC_BITS = [4, 6, 8, 10, 12]

# =============================================================================
# Helper: load sweep summary CSV for a tag
# =============================================================================

def load_sweep_summary(tag: str) -> pd.DataFrame:
    """Try {tag}/{tag}_sweep_summary.csv, then {tag}/summary_adc_sweep.csv."""
    candidates = [
        RESULTS / tag / f"{tag}_sweep_summary.csv",
        RESULTS / tag / "summary_adc_sweep.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


def load_module_mac_summary(tag: str, adc_bits: int) -> pd.DataFrame:
    """Load {tag}/{tag}_adc{adc_bits}_module_mac_summary.csv."""
    p = RESULTS / tag / f"{tag}_adc{adc_bits}_module_mac_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_logit_eval(tag: str, adc_bits: int) -> pd.DataFrame:
    p = RESULTS / tag / f"{tag}_adc{adc_bits}_logit_eval.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


# =============================================================================
# Figure M1 — MAC SNR mean vs ADC bits, one line per tag
# =============================================================================

def fig_M1_snr_vs_adc():
    fig, ax = plt.subplots(figsize=(8, 5))
    saved_arrays = {}

    for tag in TAGS:
        df = load_sweep_summary(tag)
        if df.empty or "mac_snr_mean" not in df.columns or "adc_bits" not in df.columns:
            print(f"  [M1] Skipping tag={tag}: missing sweep summary or columns")
            continue
        df_sorted = df.sort_values("adc_bits")
        ax.plot(df_sorted["adc_bits"], df_sorted["mac_snr_mean"],
                marker="o", label=tag)
        saved_arrays[f"{tag}_adc_bits"] = df_sorted["adc_bits"].values
        saved_arrays[f"{tag}_mac_snr_mean"] = df_sorted["mac_snr_mean"].values

    ax.set_xlabel("ADC bits")
    ax.set_ylabel("MAC SNR mean (dB)")
    ax.set_title("MAC SNR vs ADC bits — Mitigation Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ext in ("png", "pdf"):
        path = PLOTS / f"fig_M1_snr_vs_adc.{ext}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved → {path}")
    np.savez_compressed(PLOTS / "fig_M1_snr_vs_adc.npz", **saved_arrays)
    plt.close(fig)


# =============================================================================
# Figure M2 — SNR heatmap at ADC=6, two panels (baseline vs best mitigation)
# =============================================================================

def fig_M2_snr_heatmap_adc6():
    ref_tag  = TAGS[0] if TAGS else "baseline"
    best_tag = TAGS[-1] if len(TAGS) > 1 else TAGS[0]

    def pivot_snr(tag):
        df = load_module_mac_summary(tag, 6)
        if df.empty:
            return None
        pivot = df.pivot_table(index="layer_idx", columns="sublayer", values="mac_snr_db")
        return pivot.reindex(columns=SUBLAYER_ORDER)

    piv_ref  = pivot_snr(ref_tag)
    piv_best = pivot_snr(best_tag)

    if piv_ref is None and piv_best is None:
        print("  [M2] No module_mac_summary data found, skipping")
        return

    n_panels = 2 if (piv_ref is not None and piv_best is not None) else 1
    pivots = [p for p in [piv_ref, piv_best] if p is not None]
    labels_plot = [ref_tag, best_tag][:n_panels]

    vmin = min(p.values[~np.isnan(p.values)].min() for p in pivots)
    vmax = max(p.values[~np.isnan(p.values)].max() for p in pivots)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)
    for i, (piv, lbl) in enumerate(zip(pivots, labels_plot)):
        ax = axes[0][i]
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(SUBLAYER_ORDER)))
        ax.set_xticklabels(SUBLAYER_ORDER)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index.tolist())
        ax.set_xlabel("Sublayer")
        ax.set_ylabel("Encoder Layer")
        ax.set_title(f"{lbl} — MAC SNR (dB) @ ADC=6")
        plt.colorbar(im, ax=ax)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = PLOTS / f"fig_M2_snr_heatmap_adc6.{ext}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# =============================================================================
# Figure M3 — O deadzone ratio vs ADC bits, baseline vs mitigations
# =============================================================================

def fig_M3_O_deadzone_vs_adc():
    fig, ax = plt.subplots(figsize=(8, 5))

    for tag in TAGS:
        df = load_sweep_summary(tag)
        col = "out_clip_ratio_O_mean"
        # Fall back to loading per-adc module_mac_summary if not in sweep
        if df.empty or col not in df.columns:
            # Try to gather per-ADC bits from module summaries
            rows = []
            for ab in ADC_BITS:
                mdf = load_module_mac_summary(tag, ab)
                if mdf.empty:
                    continue
                o_df = mdf[mdf["sublayer"] == "O"]
                if o_df.empty:
                    continue
                rows.append({
                    "adc_bits": ab,
                    "ref_deadzone_ratio_O": o_df["ref_deadzone_ratio"].mean(),
                })
            if not rows:
                print(f"  [M3] Skipping tag={tag}: no O deadzone data")
                continue
            tmp = pd.DataFrame(rows).sort_values("adc_bits")
            ax.plot(tmp["adc_bits"], tmp["ref_deadzone_ratio_O"], marker="o", label=tag)
        else:
            df_sorted = df.sort_values("adc_bits")
            ax.plot(df_sorted["adc_bits"], df_sorted[col], marker="o", label=tag)

    ax.set_xlabel("ADC bits")
    ax.set_ylabel("O sublayer deadzone ratio")
    ax.set_title("Output (O) Deadzone Ratio vs ADC bits")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ext in ("png", "pdf"):
        path = PLOTS / f"fig_M3_O_deadzone_vs_adc.{ext}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# =============================================================================
# Supplementary S1 — Full heatmaps (NMSE, cosine, clip, deadzone)
# =============================================================================

def fig_S1_full_heatmaps(adc_bits: int = 6):
    metrics = [
        ("mac_snr_db",         "MAC SNR (dB)",        "viridis"),
        ("mac_nmse",           "NMSE",                "Reds"),
        ("cosine",             "Cosine similarity",   "Blues"),
        ("out_clip_ratio",     "Clip ratio",          "Oranges"),
        ("ref_deadzone_ratio", "Deadzone ratio",      "Purples"),
    ]
    tags_to_plot = TAGS[:2]  # baseline + first mitigation

    for tag in tags_to_plot:
        df = load_module_mac_summary(tag, adc_bits)
        if df.empty:
            print(f"  [S1] No module_mac_summary for tag={tag}, adc={adc_bits}")
            continue
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        for ax, (col, title, cmap) in zip(axes, metrics):
            if col not in df.columns:
                ax.set_title(f"{title}\n(no data)")
                continue
            pivot = df.pivot_table(index="layer_idx", columns="sublayer", values=col)
            pivot = pivot.reindex(columns=SUBLAYER_ORDER)
            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(SUBLAYER_ORDER)))
            ax.set_xticklabels(SUBLAYER_ORDER, fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
            ax.set_title(f"{title}", fontsize=9)
            plt.colorbar(im, ax=ax)
        fig.suptitle(f"[S1] {tag} — Full heatmaps @ ADC={adc_bits}", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = PLOTS / f"fig_S1_full_heatmaps_{tag}_adc{adc_bits}.{ext}"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            print(f"  Saved → {path}")
        plt.close(fig)


# =============================================================================
# Supplementary S2 — Boxplots of FFN1 and V SNR across layers at ADC=6
# =============================================================================

def fig_S2_boxplots_ffn1_v(adc_bits: int = 6):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sl in zip(axes, ["FFN1", "V"]):
        for tag in TAGS:
            df = load_module_mac_summary(tag, adc_bits)
            if df.empty:
                continue
            sl_df = df[df["sublayer"] == sl]
            if sl_df.empty:
                continue
            # Sort by layer_idx, collect per-layer mac_snr_db
            vals = sl_df.sort_values("layer_idx")["mac_snr_db"].values
            ax.plot(sl_df.sort_values("layer_idx")["layer_idx"].values, vals,
                    marker="o", label=tag)
        ax.set_title(f"{sl} MAC SNR (dB) @ ADC={adc_bits}")
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("MAC SNR (dB)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = PLOTS / f"fig_S2_ffn1_v_snr_by_layer_adc{adc_bits}.{ext}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# =============================================================================
# Supplementary S3 — Flip rate vs margin scatter (logit eval)
# =============================================================================

def fig_S3_flip_vs_margin(adc_bits: int = 6):
    fig, ax = plt.subplots(figsize=(7, 5))
    found = False
    for tag in TAGS:
        df = load_logit_eval(tag, adc_bits)
        if df.empty or "flip" not in df.columns or "margin" not in df.columns:
            continue
        ax.scatter(df["margin"], df["flip"], label=tag, alpha=0.7, s=30)
        found = True
    if not found:
        print(f"  [S3] No logit_eval data for adc={adc_bits}, skipping")
        plt.close(fig)
        return
    ax.set_xlabel("Logit margin (top-1 − top-2)")
    ax.set_ylabel("Flip rate")
    ax.set_title(f"Flip Rate vs Logit Margin @ ADC={adc_bits}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for ext in ("png", "pdf"):
        path = PLOTS / f"fig_S3_flip_vs_margin_adc{adc_bits}.{ext}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# =============================================================================
# Mitigation report (markdown)
# =============================================================================

def write_mitigation_report():
    lines = ["# Mitigation Report\n",
             f"Tags: {', '.join(TAGS)}\n",
             "\n## Summary Table @ ADC=6\n",
             "| Tag | FFN1 SNR (dB) | V NMSE | O deadzone | L11-FFN1 SNR |",
             "|-----|--------------|--------|------------|--------------|"]
    for tag in TAGS:
        df = load_module_mac_summary(tag, 6)
        if df.empty:
            lines.append(f"| {tag} | — | — | — | — |")
            continue
        ffn1_snr = df[df["sublayer"] == "FFN1"]["mac_snr_db"].mean()
        v_nmse   = df[df["sublayer"] == "V"]["mac_nmse"].mean()
        o_dead   = df[df["sublayer"] == "O"]["ref_deadzone_ratio"].mean()
        l11_ffn1 = df[(df["sublayer"] == "FFN1") & (df["layer_idx"] == 11)]["mac_snr_db"]
        l11_val  = l11_ffn1.values[0] if len(l11_ffn1) > 0 else float("nan")
        lines.append(
            f"| {tag} | {ffn1_snr:.2f} | {v_nmse:.4f} | {o_dead:.4f} | {l11_val:.2f} |"
        )

    lines.append("\n## ADC=4 Summary\n")
    lines.append("| Tag | FFN1 SNR (dB) | O deadzone |")
    lines.append("|-----|--------------|------------|")
    for tag in TAGS:
        df = load_module_mac_summary(tag, 4)
        if df.empty:
            lines.append(f"| {tag} | — | — |")
            continue
        ffn1_snr = df[df["sublayer"] == "FFN1"]["mac_snr_db"].mean()
        o_dead   = df[df["sublayer"] == "O"]["ref_deadzone_ratio"].mean()
        lines.append(f"| {tag} | {ffn1_snr:.2f} | {o_dead:.4f} |")

    report_path = PLOTS / "mitigation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved report → {report_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"[PlotMitigation] Results dir: {RESULTS}")
    print(f"[PlotMitigation] Tags: {TAGS}")
    print(f"[PlotMitigation] Output dir: {PLOTS}")

    print("\n--- Figure M1: SNR vs ADC bits ---")
    fig_M1_snr_vs_adc()

    print("\n--- Figure M2: SNR heatmap at ADC=6 ---")
    fig_M2_snr_heatmap_adc6()

    print("\n--- Figure M3: O deadzone vs ADC bits ---")
    fig_M3_O_deadzone_vs_adc()

    print("\n--- Supplementary S1: Full heatmaps ---")
    fig_S1_full_heatmaps(adc_bits=6)

    print("\n--- Supplementary S2: FFN1/V SNR by layer ---")
    fig_S2_boxplots_ffn1_v(adc_bits=6)

    print("\n--- Supplementary S3: Flip rate vs margin ---")
    fig_S3_flip_vs_margin(adc_bits=6)

    print("\n--- Mitigation Report ---")
    write_mitigation_report()

    print(f"\n[PlotMitigation] Done. All outputs in {PLOTS}")

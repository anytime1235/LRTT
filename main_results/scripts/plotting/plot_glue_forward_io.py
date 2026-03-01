"""plot_glue_forward_io.py — Paper-ready plots for GLUE forward I/O diagnostics.

Reads CSV/NPZ artifacts produced by diag_forward_io_glue.py and generates
publication-quality figures.

Main Plots:
  1. MAC SNR vs ADC bits — per sublayer line plot (seed mean±std if available)
  2. Heatmap at adc=6 and adc=8 — layer × sublayer SNR (bottleneck highlighted)
  3. O deadzone vs ADC bits — with/without out_bound calibration

Supplementary Plots:
  4. Full metric heatmaps (NMSE, cosine, clip, deadzone) across all ADC levels
  5. Layer-wise box plots for FFN1 and V sublayers at adc=6
  6. Distribution plots (per-layer SNR spread)

Usage:
  python plot_glue_forward_io.py --results-dir ./results/diag_fwd_io_glue/baseline_sst2 \\
      --task sst2 --tag baseline_sst2 --out-dir ./results/paper_plots_glue

  # Compare baseline vs calibrated
  python plot_glue_forward_io.py \\
      --results-dir ./results/diag_fwd_io_glue/baseline_sst2 \\
      --calib-dir   ./results/diag_fwd_io_glue/obcal_sst2 \\
      --task sst2 --out-dir ./results/paper_plots_glue

  # Seed sweep (error bars from seed_sweep_summary.csv)
  python plot_glue_forward_io.py \\
      --results-dir ./results/diag_fwd_io_glue/seed_sweep_sst2 \\
      --seed-sweep --task sst2 --out-dir ./results/paper_plots_glue
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Paper-ready plots for GLUE forward I/O diagnostics"
)
parser.add_argument("--results-dir", type=str, required=True,
                    help="Directory with diagnostic CSV/NPZ from diag_forward_io_glue.py")
parser.add_argument("--calib-dir",   type=str, default=None,
                    help="Calibrated out-bound results dir (for comparison plot)")
parser.add_argument("--task",        type=str, default="sst2",
                    help="GLUE task name (for plot titles)")
parser.add_argument("--tag",         type=str, default=None,
                    help="Run tag prefix (auto-detected from files if omitted)")
parser.add_argument("--out-dir",     type=str, default="./results/paper_plots_glue",
                    help="Output directory for plots")
parser.add_argument("--seed-sweep",  action="store_true",
                    help="Load seed_sweep_summary.csv for error bars")
parser.add_argument("--adc-highlight", type=int, nargs="+", default=[6, 8],
                    help="ADC bit values to show heatmaps for (default: 6 8)")
parser.add_argument("--dpi",         type=int, default=150,
                    help="Plot DPI (default: 150)")
parser.add_argument("--format",      type=str, default="png",
                    choices=["png", "pdf", "both"],
                    help="Output format (default: png)")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
SL_COLOR = {sl: COLORS[i % len(COLORS)] for i, sl in enumerate(SUBLAYER_ORDER)}

TASK   = args.task
TAG    = args.tag or ""
DPI    = args.dpi


def _savefig(fig, name: str):
    """Save figure in requested format(s) and close."""
    if args.format in ("png", "both"):
        p = os.path.join(args.out_dir, f"{name}.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"  Saved → {p}")
    if args.format in ("pdf", "both"):
        p = os.path.join(args.out_dir, f"{name}.pdf")
        fig.savefig(p, bbox_inches="tight")
        print(f"  Saved → {p}")
    plt.close(fig)


def _save_npz(name: str, **arrays):
    """Save numpy arrays used in plots for reproducibility."""
    path = os.path.join(args.out_dir, f"{name}.npz")
    np.savez_compressed(path, **arrays)
    print(f"  Saved arrays → {path}")


# =============================================================================
# Data Loading
# =============================================================================

def _find_sweep_csv(results_dir: str) -> pd.DataFrame | None:
    """Find and load sweep summary CSV."""
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_sweep_summary.csv"):
            path = os.path.join(results_dir, fname)
            df = pd.read_csv(path)
            print(f"  Loaded sweep summary: {path} ({len(df)} rows)")
            return df
    return None


def _find_mac_metrics_csv(results_dir: str, adc_bits: int) -> pd.DataFrame | None:
    """Find layer_mac_metrics CSV for a specific adc_bits."""
    for fname in sorted(os.listdir(results_dir)):
        if f"adc{adc_bits}" in fname and fname.endswith("_layer_mac_metrics.csv"):
            path = os.path.join(results_dir, fname)
            df = pd.read_csv(path)
            print(f"  Loaded mac metrics: {path} ({len(df)} rows)")
            return df
    return None


def _find_seed_sweep_csv(results_dir: str) -> pd.DataFrame | None:
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_seed_sweep_summary.csv"):
            path = os.path.join(results_dir, fname)
            df = pd.read_csv(path)
            print(f"  Loaded seed sweep summary: {path}")
            return df
    return None


def _find_module_mac_csv(results_dir: str, adc_bits: int) -> pd.DataFrame | None:
    for fname in sorted(os.listdir(results_dir)):
        if f"adc{adc_bits}" in fname and fname.endswith("_module_mac_summary.csv"):
            path = os.path.join(results_dir, fname)
            df = pd.read_csv(path)
            print(f"  Loaded module MAC summary: {path}")
            return df
    return None


# =============================================================================
# Plot 1: MAC SNR vs ADC Bits (per sublayer, with optional seed error bars)
# =============================================================================

def plot_snr_vs_adc(df: pd.DataFrame, seed_df: pd.DataFrame = None,
                    calib_df: pd.DataFrame = None):
    """Main plot 1: MAC SNR vs ADC bits, per sublayer."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for sl in SUBLAYER_ORDER:
        col_mean = f"mac_snr_{sl}_mean"
        if col_mean not in df.columns:
            continue
        y = df[col_mean].values
        x = df["adc_bits"].values

        # Seed error bars if available
        if seed_df is not None:
            mean_col = f"mac_snr_{sl}_mean_mean"
            std_col  = f"mac_snr_{sl}_mean_std"
            if mean_col in seed_df.columns:
                y = seed_df[mean_col].values
                x = seed_df["adc_bits"].values
                yerr = seed_df[std_col].values if std_col in seed_df.columns else None
                ax.errorbar(x, y, yerr=yerr, marker="o", label=sl,
                            color=SL_COLOR[sl], capsize=4)
                continue

        ax.plot(x, y, marker="o", label=sl, color=SL_COLOR[sl])

    # Calibrated overlay (dashed)
    if calib_df is not None:
        for sl in SUBLAYER_ORDER:
            col = f"mac_snr_{sl}_mean"
            if col in calib_df.columns:
                ax.plot(calib_df["adc_bits"], calib_df[col],
                        linestyle="--", marker="s", color=SL_COLOR[sl],
                        alpha=0.6, label=f"{sl} (calib)")

    ax.set_xlabel("ADC Bits", fontsize=12)
    ax.set_ylabel("MAC SNR (dB)", fontsize=12)
    ax.set_title(f"BERT-base {TASK.upper()} — Forward MAC SNR vs ADC Bits", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df["adc_bits"].unique())

    _save_npz("plot1_snr_vs_adc",
              adc_bits=df["adc_bits"].values,
              **{f"snr_{sl}": df.get(f"mac_snr_{sl}_mean",
                                     pd.Series(dtype=float)).values
                 for sl in SUBLAYER_ORDER})
    _savefig(fig, "plot1_snr_vs_adc")


# =============================================================================
# Plot 2: Layer × Sublayer Heatmap at Selected ADC Bits
# =============================================================================

def plot_heatmaps_at_adc(results_dir: str, adc_highlight: list):
    """Plot 2: SNR heatmap for each highlighted ADC bit value."""
    for adc_bits in adc_highlight:
        mac_df = _find_mac_metrics_csv(results_dir, adc_bits)
        if mac_df is None:
            print(f"  [skip] No mac_metrics CSV for adc={adc_bits}")
            continue

        # Aggregate over steps
        agg = mac_df.groupby(["layer_idx", "sublayer"])[
            ["mac_snr_db", "ref_deadzone_ratio", "out_clip_ratio"]
        ].mean().reset_index()

        pivot_snr = agg.pivot(index="layer_idx", columns="sublayer", values="mac_snr_db")
        pivot_snr = pivot_snr.reindex(columns=[c for c in SUBLAYER_ORDER if c in pivot_snr.columns])

        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(pivot_snr.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot_snr.columns)))
        ax.set_xticklabels(list(pivot_snr.columns), fontsize=11)
        ax.set_yticks(range(len(pivot_snr.index)))
        ax.set_yticklabels([f"L{i}" for i in pivot_snr.index], fontsize=9)
        ax.set_xlabel("Sublayer", fontsize=12)
        ax.set_ylabel("Encoder Layer", fontsize=12)
        ax.set_title(f"{TASK.upper()} — MAC SNR (dB) at ADC={adc_bits} bits", fontsize=13)
        plt.colorbar(im, ax=ax, label="MAC SNR (dB)")

        # Highlight bottleneck (lowest SNR cell)
        if pivot_snr.values.size > 0:
            flat_min = np.nanargmin(pivot_snr.values)
            ri, ci = np.unravel_index(flat_min, pivot_snr.values.shape)
            ax.add_patch(plt.Rectangle(
                (ci - 0.5, ri - 0.5), 1, 1,
                fill=False, edgecolor="red", lw=2.5, label="Bottleneck"
            ))
            ax.legend(loc="upper right", fontsize=9)

        _save_npz(f"plot2_heatmap_adc{adc_bits}",
                  pivot_snr=pivot_snr.values,
                  columns=np.array(list(pivot_snr.columns)),
                  index=np.array(list(pivot_snr.index)))
        _savefig(fig, f"plot2_heatmap_snr_adc{adc_bits}")


# =============================================================================
# Plot 3: O Sublayer Deadzone vs ADC Bits (baseline vs calibrated)
# =============================================================================

def plot_o_deadzone_vs_adc(df: pd.DataFrame, calib_df: pd.DataFrame = None):
    """Plot 3: O sublayer ref_deadzone_ratio vs ADC bits."""
    col = "ref_deadzone_ratio_O_mean"
    if col not in df.columns:
        print(f"  [skip] Column {col} not found in sweep summary")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["adc_bits"], df[col], marker="o", label="Baseline", color="tab:blue")

    if calib_df is not None and col in calib_df.columns:
        ax.plot(calib_df["adc_bits"], calib_df[col], marker="s", linestyle="--",
                label="Out-bound Calibrated", color="tab:orange")

    ax.set_xlabel("ADC Bits", fontsize=12)
    ax.set_ylabel("O Deadzone Ratio", fontsize=12)
    ax.set_title(f"{TASK.upper()} — Output (O) Sublayer Deadzone vs ADC Bits", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df["adc_bits"].unique())

    _save_npz("plot3_o_deadzone_vs_adc",
              adc_bits=df["adc_bits"].values,
              deadzone_O=df[col].values)
    _savefig(fig, "plot3_o_deadzone_vs_adc")


# =============================================================================
# Supplementary Plot 4: Full Metric Heatmaps Across All ADC Levels
# =============================================================================

def plot_supp_full_heatmaps(results_dir: str):
    """Supp 4: Heatmaps for all 4 metrics × all available ADC levels."""
    adc_csvs = {}
    for fname in sorted(os.listdir(results_dir)):
        if "layer_mac_metrics" in fname and fname.endswith(".csv"):
            # Try to extract adc bits from filename
            import re
            m = re.search(r"adc(\d+)", fname)
            if m:
                bits = int(m.group(1))
                adc_csvs[bits] = os.path.join(results_dir, fname)

    if not adc_csvs:
        print("  [skip] No layer_mac_metrics CSVs found for supp heatmaps")
        return

    metrics = [
        ("mac_nmse",           "MAC NMSE",       "hot_r"),
        ("cosine",             "Cosine Sim",      "Blues"),
        ("out_clip_ratio",     "Clip Ratio",      "Reds"),
        ("ref_deadzone_ratio", "Deadzone Ratio",  "Purples"),
    ]

    for bits, csv_path in sorted(adc_csvs.items()):
        mac_df = pd.read_csv(csv_path)
        agg = mac_df.groupby(["layer_idx", "sublayer"])[
            [m[0] for m in metrics]
        ].mean().reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes.flatten()
        fig.suptitle(f"{TASK.upper()} — ADC={bits} bits: Metric Heatmaps", fontsize=14)

        for ax, (metric, title, cmap) in zip(axes, metrics):
            if metric not in agg.columns:
                ax.set_visible(False)
                continue
            pivot = agg.pivot(index="layer_idx", columns="sublayer", values=metric)
            pivot = pivot.reindex(columns=[c for c in SUBLAYER_ORDER if c in pivot.columns])
            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(list(pivot.columns), fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"L{i}" for i in pivot.index], fontsize=7)
            ax.set_title(title, fontsize=11)
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        _savefig(fig, f"supp4_full_metrics_adc{bits}")


# =============================================================================
# Supplementary Plot 5: Layer-wise Box Plots for FFN1 and V at ADC=6
# =============================================================================

def plot_supp_boxplots(results_dir: str, adc_bits: int = 6):
    """Supp 5: Per-layer SNR box plot for FFN1 and V sublayers."""
    mac_df = _find_mac_metrics_csv(results_dir, adc_bits)
    if mac_df is None:
        print(f"  [skip] No mac_metrics CSV for adc={adc_bits} (box plots)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{TASK.upper()} — Per-layer SNR at ADC={adc_bits} bits", fontsize=13)

    for ax, sublayer in zip(axes, ["FFN1", "V"]):
        sl_df = mac_df[mac_df["sublayer"] == sublayer]
        if sl_df.empty:
            ax.set_visible(False)
            continue
        data = [sl_df[sl_df["layer_idx"] == li]["mac_snr_db"].dropna().values
                for li in range(12)]
        ax.boxplot(data, positions=range(12), widths=0.6)
        ax.set_xlabel("Encoder Layer Index", fontsize=11)
        ax.set_ylabel("MAC SNR (dB)", fontsize=11)
        ax.set_title(f"{sublayer} Sublayer", fontsize=12)
        ax.set_xticks(range(12))
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig(fig, f"supp5_boxplots_adc{adc_bits}")


# =============================================================================
# Supplementary Plot 6: Per-Layer SNR Distribution
# =============================================================================

def plot_supp_snr_distribution(results_dir: str, adc_bits: int = 6):
    """Supp 6: Distribution of SNR across all layers (violin plot)."""
    mac_df = _find_mac_metrics_csv(results_dir, adc_bits)
    if mac_df is None:
        print(f"  [skip] No mac_metrics CSV for adc={adc_bits} (distribution)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    data  = [mac_df[mac_df["sublayer"] == sl]["mac_snr_db"].dropna().values
             for sl in SUBLAYER_ORDER]
    parts = ax.violinplot(data, positions=range(len(SUBLAYER_ORDER)),
                          showmedians=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(SL_COLOR[SUBLAYER_ORDER[i]])
        body.set_alpha(0.7)

    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER, fontsize=11)
    ax.set_ylabel("MAC SNR (dB)", fontsize=12)
    ax.set_title(f"{TASK.upper()} — SNR Distribution at ADC={adc_bits} bits", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    _savefig(fig, f"supp6_snr_distribution_adc{adc_bits}")


# =============================================================================
# Main
# =============================================================================

def main():
    results_dir = args.results_dir
    calib_dir   = args.calib_dir

    print(f"\n[PlotGlue] results_dir={results_dir}")
    print(f"[PlotGlue] out_dir={args.out_dir}")

    # Load sweep summary
    sweep_df = _find_sweep_csv(results_dir)
    seed_df  = _find_seed_sweep_csv(results_dir) if args.seed_sweep else None
    calib_df = _find_sweep_csv(calib_dir) if calib_dir else None

    if sweep_df is None:
        print("[WARNING] No sweep_summary.csv found — skipping plots 1, 3")
    else:
        # Plot 1: SNR vs ADC bits
        plot_snr_vs_adc(sweep_df, seed_df=seed_df, calib_df=calib_df)
        # Plot 3: O deadzone vs ADC bits
        plot_o_deadzone_vs_adc(sweep_df, calib_df=calib_df)

    # Plot 2: Heatmaps at selected ADC bits
    plot_heatmaps_at_adc(results_dir, args.adc_highlight)

    # Supplementary plots
    plot_supp_full_heatmaps(results_dir)
    for bits in args.adc_highlight:
        plot_supp_boxplots(results_dir, adc_bits=bits)
        plot_supp_snr_distribution(results_dir, adc_bits=bits)

    print(f"\n[PlotGlue] All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()

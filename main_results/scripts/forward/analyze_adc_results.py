#!/usr/bin/env python3
"""analyze_adc_results.py — ADC 4/6/8 bit 결과 병합 분석 스크립트.

Usage:
    /data/venvs/lrtt/bin/python /data/main_results/analyze_adc_results.py \
        --results-dir /data/main_results/results/diag_fwd_io_mitigations/baseline \
        --tag baseline \
        --adc-bits 4 6 8 \
        --out-dir /data/main_results/results/diag_fwd_io_mitigations/analysis_adc468
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]
METRICS = ["mac_snr_db", "mac_nmse", "cosine", "out_clip_ratio"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csvs(results_dir: str, tag: str, adc_bits_list: list[int]):
    """Load and merge CSVs for all ADC bit widths."""
    summary_rows = []
    module_dfs = []
    layer_dfs = []

    for bits in adc_bits_list:
        prefix = os.path.join(results_dir, f"{tag}_adc{bits}")

        sr = pd.read_csv(f"{prefix}_summary_row.csv")
        summary_rows.append(sr)

        mm = pd.read_csv(f"{prefix}_module_mac_summary.csv")
        mm["adc_bits"] = bits
        module_dfs.append(mm)

        lm = pd.read_csv(f"{prefix}_layer_mac_metrics.csv")
        lm["adc_bits"] = bits
        layer_dfs.append(lm)

    sweep_df = pd.concat(summary_rows, ignore_index=True).sort_values("adc_bits").reset_index(drop=True)
    module_df = pd.concat(module_dfs, ignore_index=True)
    layer_df = pd.concat(layer_dfs, ignore_index=True)

    return sweep_df, module_df, layer_df


# ---------------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------------

def build_summary_table(module_df: pd.DataFrame, adc_bits_list: list[int]) -> pd.DataFrame:
    """Sublayer × (adc_bits × metrics) summary table."""
    rows = []
    for sl in SUBLAYER_ORDER + ["mean"]:
        row = {"sublayer": sl}
        for bits in adc_bits_list:
            sub = module_df[module_df["adc_bits"] == bits]
            if sl == "mean":
                grp = sub
            else:
                grp = sub[sub["sublayer"] == sl]
            for m in METRICS:
                row[f"adc{bits}_{m}"] = grp[m].mean() if not grp.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_snr_gain_table(summary_table: pd.DataFrame, adc_bits_list: list[int]) -> pd.DataFrame:
    """SNR gain across bit widths per sublayer."""
    rows = []
    bits_sorted = sorted(adc_bits_list)
    pairs = [(bits_sorted[i], bits_sorted[j])
             for i in range(len(bits_sorted))
             for j in range(i + 1, len(bits_sorted))]

    for _, row in summary_table.iterrows():
        sl = row["sublayer"]
        entry = {"sublayer": sl}
        for lo, hi in pairs:
            snr_lo = row.get(f"adc{lo}_mac_snr_db", float("nan"))
            snr_hi = row.get(f"adc{hi}_mac_snr_db", float("nan"))
            entry[f"snr_gain_{lo}to{hi}_db"] = snr_hi - snr_lo
        rows.append(entry)

    df = pd.DataFrame(rows)
    # Identify bottleneck sublayer (lowest SNR at lowest bit width)
    lowest_bits = bits_sorted[0]
    col = f"adc{lowest_bits}_mac_snr_db"
    sub_only = summary_table[summary_table["sublayer"] != "mean"]
    bottleneck_idx = sub_only[col].idxmin()
    bottleneck = sub_only.loc[bottleneck_idx, "sublayer"]
    print(f"\n  Bottleneck sublayer (adc{lowest_bits}): {bottleneck}"
          f" (SNR = {sub_only.loc[bottleneck_idx, col]:.3f} dB)")
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_adc_sweep(sweep_df: pd.DataFrame, out_dir: str):
    """Plot A: MAC SNR per sublayer vs ADC bits; Plot B: clip ratio."""
    bits = sweep_df["adc_bits"].values

    # Plot A: SNR per sublayer
    fig, ax = plt.subplots(figsize=(8, 5))
    for sl in SUBLAYER_ORDER:
        col = f"mac_snr_{sl}_mean"
        if col in sweep_df.columns:
            ax.plot(bits, sweep_df[col].values, marker="o", label=sl)
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("MAC SNR (dB)")
    ax.set_title("Forward MAC SNR vs ADC bits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "plot_A_snr_vs_adc.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # Plot B: clip ratio per sublayer
    fig, ax = plt.subplots(figsize=(8, 5))
    for sl in SUBLAYER_ORDER:
        col = f"out_clip_ratio_{sl}_mean"
        if col in sweep_df.columns:
            ax.plot(bits, sweep_df[col].values, marker="o", label=sl)
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("Output Clip Ratio")
    ax.set_title("Output Clip Ratio vs ADC bits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "plot_B_clip_ratio_vs_adc.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_layer_snr_comparison(layer_df: pd.DataFrame, adc_bits_list: list[int], out_dir: str):
    """Per-layer MAC SNR comparison across ADC bits (subplot per sublayer)."""
    n_sl = len(SUBLAYER_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes_flat = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(adc_bits_list)))

    for ax_idx, sl in enumerate(SUBLAYER_ORDER):
        ax = axes_flat[ax_idx]
        for bits, color in zip(sorted(adc_bits_list), colors):
            sub = layer_df[(layer_df["sublayer"] == sl) & (layer_df["adc_bits"] == bits)]
            if sub.empty:
                continue
            agg = sub.groupby("layer_idx")["mac_snr_db"].mean().reset_index()
            agg = agg.sort_values("layer_idx")
            ax.plot(agg["layer_idx"], agg["mac_snr_db"], marker="o",
                    label=f"adc{bits}", color=color)
        ax.set_title(sl)
        ax.set_ylabel("MAC SNR (dB)")
        ax.set_xlabel("Layer")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Layer MAC SNR: ADC 4/6/8 Comparison", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "analysis_layer_snr_comparison.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_error_dist_comparison(module_df: pd.DataFrame, adc_bits_list: list[int], out_dir: str):
    """Mean and p95 abs error bar chart per sublayer across ADC bits."""
    bits_sorted = sorted(adc_bits_list)
    n_sl = len(SUBLAYER_ORDER)
    x = np.arange(n_sl)
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, err_col, title in zip(
        axes,
        ["mean_abs_err", "p95_abs_err"],
        ["Mean Absolute Error", "P95 Absolute Error"],
    ):
        for i, bits in enumerate(bits_sorted):
            sub = module_df[module_df["adc_bits"] == bits]
            vals = []
            for sl in SUBLAYER_ORDER:
                grp = sub[sub["sublayer"] == sl]
                vals.append(grp[err_col].mean() if not grp.empty else 0.0)
            offset = (i - len(bits_sorted) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=f"adc{bits}")

        ax.set_xticks(x)
        ax.set_xticklabels(SUBLAYER_ORDER)
        ax.set_ylabel(err_col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Error Distribution by Sublayer: ADC 4/6/8", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "analysis_error_dist_comparison.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_console_summary(sweep_df: pd.DataFrame, summary_table: pd.DataFrame,
                           snr_gain_df: pd.DataFrame, adc_bits_list: list[int]):
    bits_sorted = sorted(adc_bits_list)
    print("\n" + "=" * 60)
    print("## ADC 4/6/8 분석 요약")
    print("=" * 60)

    # Overall SNR
    print("\n### 전체 평균 MAC SNR (dB)")
    for bits in bits_sorted:
        row = sweep_df[sweep_df["adc_bits"] == bits]
        if row.empty:
            continue
        snr = row["mac_snr_mean"].values[0]
        print(f"  adc{bits}: {snr:.3f} dB")

    # Sublayer SNR table
    print(f"\n### Sublayer 별 MAC SNR (dB)")
    header = f"  {'Sublayer':<8}" + "".join(f"  adc{b:>2}" for b in bits_sorted)
    print(header)
    print("  " + "-" * (8 + 8 * len(bits_sorted)))
    for _, row in summary_table.iterrows():
        sl = row["sublayer"]
        vals = "".join(f"  {row[f'adc{b}_mac_snr_db']:>6.2f}" for b in bits_sorted)
        print(f"  {sl:<8}{vals}")

    # SNR gain
    print("\n### SNR Gain (dB)")
    gain_cols = [c for c in snr_gain_df.columns if c.startswith("snr_gain_")]
    hdr = f"  {'Sublayer':<8}" + "".join(f"  {c.replace('snr_gain_','').replace('_db',''):>10}" for c in gain_cols)
    print(hdr)
    print("  " + "-" * (8 + 12 * len(gain_cols)))
    for _, row in snr_gain_df.iterrows():
        sl = row["sublayer"]
        vals = "".join(f"  {row[c]:>10.3f}" for c in gain_cols)
        print(f"  {sl:<8}{vals}")

    # Bottleneck
    lowest_bits = bits_sorted[0]
    col = f"adc{lowest_bits}_mac_snr_db"
    sub_only = summary_table[summary_table["sublayer"] != "mean"]
    bottleneck_sl = sub_only.loc[sub_only[col].idxmin(), "sublayer"]
    bottleneck_snr = sub_only[col].min()
    print(f"\n### 병목 Sublayer")
    print(f"  adc{lowest_bits}에서 SNR 최저: **{bottleneck_sl}** ({bottleneck_snr:.3f} dB)")

    # Monotonicity check
    mean_snrs = [sweep_df[sweep_df["adc_bits"] == b]["mac_snr_mean"].values[0] for b in bits_sorted]
    is_monotone = all(mean_snrs[i] <= mean_snrs[i + 1] for i in range(len(mean_snrs) - 1))
    print(f"\n### 단조 증가 검증")
    print(f"  mac_snr_mean 단조 증가: {'✓' if is_monotone else '✗'}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ADC 4/6/8 결과 분석")
    p.add_argument("--results-dir", required=True, help="베이스라인 결과 디렉토리")
    p.add_argument("--tag", default="baseline", help="파일 prefix 태그")
    p.add_argument("--adc-bits", nargs="+", type=int, default=[4, 6, 8])
    p.add_argument("--out-dir", required=True, help="분석 결과 출력 디렉토리")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/5] Loading CSVs from {args.results_dir} ...")
    sweep_df, module_df, layer_df = load_csvs(args.results_dir, args.tag, args.adc_bits)

    # Save merged sweep summary
    sweep_path = os.path.join(args.out_dir, "sweep_summary.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  Saved → {sweep_path}")

    print("[2/5] Building analysis tables ...")
    summary_table = build_summary_table(module_df, args.adc_bits)
    summary_path = os.path.join(args.out_dir, "analysis_summary_table.csv")
    summary_table.to_csv(summary_path, index=False)
    print(f"  Saved → {summary_path}")

    snr_gain_df = build_snr_gain_table(summary_table, args.adc_bits)
    gain_path = os.path.join(args.out_dir, "analysis_snr_gain.csv")
    snr_gain_df.to_csv(gain_path, index=False)
    print(f"  Saved → {gain_path}")

    print("[3/5] Plotting ADC sweep ...")
    plot_adc_sweep(sweep_df, args.out_dir)

    print("[4/5] Plotting per-layer SNR comparison ...")
    plot_layer_snr_comparison(layer_df, args.adc_bits, args.out_dir)

    print("[4/5] Plotting error distribution ...")
    plot_error_dist_comparison(module_df, args.adc_bits, args.out_dir)

    print("[5/5] Console summary ...")
    print_console_summary(sweep_df, summary_table, snr_gain_df, args.adc_bits)

    print(f"Done. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()

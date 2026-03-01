"""fwd_bwd_sensitivity_comparison.py — Forward vs Backward Sensitivity 통합 비교

Forward ADC (activation quantization) vs Backward DAC (gradient quantization)
sensitivity를 비교하여, V와 K의 순서가 반전되는 현상을 시각화.

Figures:
  1. Forward vs Backward Sensitivity Summary (1×2 grouped bar)
  2. Sensitivity Rank Table Heatmap (2×1)
  3. Mixed-Precision Recommendation Matrix (single heatmap)

Data sources:
  - Backward DAC: metrics_paper_B_bitsweep_summary.csv
  - Forward ADC sweep: diag_fwd_io/summary_adc_sweep.csv
  - Forward ADC per-layer: diag_fwd_io/single_run_module_mac_summary.csv
  - Forward ADC layer metrics: diag_fwd_io/adc{4,6,8,10,12}_layer_mac_metrics.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(BASE, "results", "csv")
FWD_DIR = os.path.join(CSV_DIR, "diag_fwd_io")
OUT_DIR = os.path.join(BASE, "results", "figures", "diagnostic")
os.makedirs(OUT_DIR, exist_ok=True)

SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]
SUBLAYER_COLORS = {
    "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
    "O": "#C44E52", "FFN1": "#8172B3", "FFN2": "#937860",
}

# ── load data ────────────────────────────────────────────────────────────────
def load_backward_dac():
    """Load backward DAC bitsweep summary (432 rows).
    Returns sweep (bits 4/6/8/10/12) and baseline (bits 7) combined."""
    path = os.path.join(CSV_DIR, "metrics_paper_B_bitsweep_summary.csv")
    df = pd.read_csv(path)
    # Keep sweep + baseline variants (baseline has 7-bit data)
    df = df[df["variant"].isin(["sweep", "baseline"])].copy()
    return df

def load_forward_adc_sweep():
    """Load forward ADC sweep summary (5 rows: ADC 4/6/8/10/12)."""
    path = os.path.join(FWD_DIR, "summary_adc_sweep.csv")
    return pd.read_csv(path)

def load_forward_adc_perlayer():
    """Load forward ADC per-layer summary at 8-bit (72 rows)."""
    path = os.path.join(FWD_DIR, "single_run_module_mac_summary.csv")
    return pd.read_csv(path)

def load_forward_adc_layer_bits():
    """Load per-layer forward ADC metrics at each bit width."""
    frames = []
    for bits in [4, 6, 8, 10, 12]:
        path = os.path.join(FWD_DIR, f"adc{bits}_layer_mac_metrics.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Average across steps per (layer_idx, sublayer)
            agg = df.groupby(["layer_idx", "sublayer"]).agg(
                mac_nmse=("mac_nmse", "mean"),
                mac_snr_db=("mac_snr_db", "mean"),
                cosine=("cosine", "mean"),
            ).reset_index()
            agg["adc_bits"] = bits
            frames.append(agg)
    return pd.concat(frames, ignore_index=True)


# ── Figure 1: Forward vs Backward Sensitivity Summary (1×2 grouped bar) ─────
def fig1_sensitivity_summary(bwd_df, fwd_perlayer):
    """Grouped bar: (a) Forward NMSE@8-bit, (b) Backward QZR@7-bit."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Forward ADC: mean NMSE per sublayer @ ADC 8-bit
    fwd_agg = fwd_perlayer.groupby("sublayer")["mac_nmse"].mean()
    fwd_agg = fwd_agg.reindex(SUBLAYER_ORDER)

    ax = axes[0]
    bars = ax.bar(SUBLAYER_ORDER, fwd_agg.values,
                  color=[SUBLAYER_COLORS[s] for s in SUBLAYER_ORDER],
                  edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean NMSE", fontsize=12)
    ax.set_title("(a) Forward ADC Sensitivity @ 8-bit", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sublayer", fontsize=11)
    # Annotate values
    for bar, val in zip(bars, fwd_agg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    # Highlight V and K
    fwd_rank = fwd_agg.rank(ascending=False)
    for i, s in enumerate(SUBLAYER_ORDER):
        if s in ("V", "K"):
            bars[i].set_edgecolor("red" if s == "V" else "blue")
            bars[i].set_linewidth(2.5)

    # (b) Backward DAC: mean QZR_nonzero per sublayer @ DAC 7-bit
    bwd_7 = bwd_df[bwd_df["dac_bits"] == 7].copy()
    bwd_agg = bwd_7.groupby("sublayer")["QZR_nonzero"].mean()
    bwd_agg = bwd_agg.reindex(SUBLAYER_ORDER)

    ax = axes[1]
    bars = ax.bar(SUBLAYER_ORDER, bwd_agg.values,
                  color=[SUBLAYER_COLORS[s] for s in SUBLAYER_ORDER],
                  edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean QZR (nonzero)", fontsize=12)
    ax.set_title("(b) Backward DAC Sensitivity @ 7-bit", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sublayer", fontsize=11)
    for bar, val in zip(bars, bwd_agg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    # Highlight V and K
    for i, s in enumerate(SUBLAYER_ORDER):
        if s in ("V", "K"):
            bars[i].set_edgecolor("blue" if s == "K" else "red")
            bars[i].set_linewidth(2.5)

    # Add rank annotations at bottom
    fwd_ranks = fwd_agg.dropna().rank(ascending=False).astype(int)
    bwd_ranks = bwd_agg.dropna().rank(ascending=False).astype(int)
    rank_text_fwd = "Rank (worst→best): " + " > ".join(
        fwd_agg.sort_values(ascending=False).index)
    rank_text_bwd = "Rank (worst→best): " + " > ".join(
        bwd_agg.sort_values(ascending=False).index)
    axes[0].text(0.5, -0.18, rank_text_fwd, transform=axes[0].transAxes,
                 ha="center", fontsize=10, style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffc107"))
    axes[1].text(0.5, -0.18, rank_text_bwd, transform=axes[1].transAxes,
                 ha="center", fontsize=10, style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#d1ecf1", ec="#17a2b8"))

    fig.suptitle("Forward (ADC) vs Backward (DAC): Sublayer Sensitivity Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_fwd_bwd_sensitivity_summary.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig1] Saved fig_fwd_bwd_sensitivity_summary.png/pdf")
    return fwd_agg, bwd_agg


# ── Figure 2: Sensitivity Rank Heatmap (2×1) ────────────────────────────────
def fig2_rank_heatmap(bwd_df, fwd_layer_bits):
    """Heatmap of sensitivity ranking per bits for Forward and Backward."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"hspace": 0.35})

    # (a) Forward ADC: sublayer ranking per adc_bits (1=worst=highest NMSE)
    fwd_pivot = fwd_layer_bits.groupby(["adc_bits", "sublayer"])["mac_nmse"].mean().unstack()
    fwd_pivot = fwd_pivot[SUBLAYER_ORDER]
    fwd_ranks = fwd_pivot.rank(axis=1, ascending=False).astype(int)

    ax = axes[0]
    _draw_rank_heatmap(ax, fwd_ranks, "ADC bits",
                       "(a) Forward ADC — Sublayer Sensitivity Ranking (1=worst)")

    # (b) Backward DAC: sublayer ranking per dac_bits (1=worst=highest QZR)
    # Use all available bits including baseline 7-bit
    bwd_pivot = bwd_df.groupby(["dac_bits", "sublayer"])["QZR_nonzero"].mean().unstack()
    bwd_pivot = bwd_pivot.reindex(columns=SUBLAYER_ORDER).sort_index()
    bwd_ranks = bwd_pivot.rank(axis=1, ascending=False).astype(int)

    ax = axes[1]
    _draw_rank_heatmap(ax, bwd_ranks, "DAC bits",
                       "(b) Backward DAC — Sublayer Sensitivity Ranking (1=worst)")

    fig.suptitle("Sensitivity Ranking: Forward vs Backward",
                 fontsize=14, fontweight="bold", y=1.01)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_fwd_bwd_rank_heatmap.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig2] Saved fig_fwd_bwd_rank_heatmap.png/pdf")
    return fwd_ranks, bwd_ranks


def _draw_rank_heatmap(ax, rank_df, ylabel, title):
    """Draw a rank heatmap on the given axis."""
    n_rows, n_cols = rank_df.shape
    # Color: 1 (worst) = red, 6 (best) = green
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=1, vmax=6)

    data = rank_df.values
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(rank_df.columns, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rank_df.index, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = int(data[i, j])
            color = "white" if val <= 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # Highlight V and K columns
    v_idx = list(rank_df.columns).index("V")
    k_idx = list(rank_df.columns).index("K")
    for idx, c in [(v_idx, "red"), (k_idx, "blue")]:
        rect = plt.Rectangle((idx - 0.5, -0.5), 1, n_rows,
                              linewidth=2.5, edgecolor=c, facecolor="none")
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Rank (1=worst, 6=best)")


# ── Figure 3: Mixed-Precision Recommendation Matrix ─────────────────────────
def fig3_recommendation_matrix(bwd_df, fwd_layer_bits):
    """Heatmap: minimum bits to meet quality threshold per sublayer × direction."""
    # Forward threshold: NMSE < 0.005
    fwd_thresh = 0.005
    # Backward threshold: QZR_nonzero < 0.10
    bwd_thresh = 0.10

    fwd_bits_list = sorted(fwd_layer_bits["adc_bits"].unique())
    bwd_bits_list = sorted(bwd_df["dac_bits"].unique())

    min_bits = {}
    for direction, label in [("forward", "Forward ADC bits"), ("backward", "Backward DAC bits")]:
        min_bits[label] = {}
        for sub in SUBLAYER_ORDER:
            if direction == "forward":
                found = None
                for b in fwd_bits_list:
                    val = fwd_layer_bits[
                        (fwd_layer_bits["adc_bits"] == b) &
                        (fwd_layer_bits["sublayer"] == sub)
                    ]["mac_nmse"].mean()
                    if val < fwd_thresh:
                        found = b
                        break
                min_bits[label][sub] = found if found else fwd_bits_list[-1]
            else:
                found = None
                for b in bwd_bits_list:
                    val = bwd_df[
                        (bwd_df["dac_bits"] == b) &
                        (bwd_df["sublayer"] == sub)
                    ]["QZR_nonzero"].mean()
                    if val < bwd_thresh:
                        found = b
                        break
                min_bits[label][sub] = found if found else bwd_bits_list[-1]

    rec_df = pd.DataFrame(min_bits).T
    rec_df = rec_df[SUBLAYER_ORDER]

    # Add "Recommended (max)" row
    rec_row = rec_df.max(axis=0)
    rec_df.loc["Recommended\n(max of both)"] = rec_row

    fig, ax = plt.subplots(figsize=(10, 4))
    data = rec_df.values.astype(float)
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=data.min() - 0.5, vmax=data.max() + 0.5)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(rec_df)))
    ax.set_yticklabels(rec_df.index, fontsize=11)
    ax.set_title("Mixed-Precision Recommendation: Minimum Bits to Meet Quality Threshold",
                 fontsize=13, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i, j])
            color = "white" if val >= 10 else "black"
            weight = "bold" if i == data.shape[0] - 1 else "normal"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=14, fontweight=weight, color=color)

    # Highlight recommendation row
    rect = plt.Rectangle((-0.5, data.shape[0] - 1.5), data.shape[1], 1,
                          linewidth=3, edgecolor="black", facecolor="none",
                          linestyle="--")
    ax.add_patch(rect)

    plt.colorbar(im, ax=ax, shrink=0.7, label="Bits")

    # Add threshold info
    ax.text(0.5, -0.25,
            f"Thresholds — Forward: NMSE < {fwd_thresh}  |  Backward: QZR < {bwd_thresh}",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#4caf50"))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_fwd_bwd_recommendation_matrix.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig3] Saved fig_fwd_bwd_recommendation_matrix.png/pdf")
    return rec_df


# ── Console output ───────────────────────────────────────────────────────────
def print_comparison_table(fwd_agg, bwd_agg):
    """Print ranking comparison table."""
    print("\n" + "=" * 70)
    print("  Forward vs Backward — Sublayer Sensitivity Ranking")
    print("=" * 70)

    fwd_ranked = fwd_agg.sort_values(ascending=False)
    bwd_ranked = bwd_agg.sort_values(ascending=False)

    print(f"\n{'Sublayer':<10} {'Fwd NMSE':>12} {'Fwd Rank':>10} {'Bwd QZR':>12} {'Bwd Rank':>10}")
    print("-" * 56)
    fwd_ranks = fwd_agg.rank(ascending=False).astype(int)
    bwd_ranks = bwd_agg.rank(ascending=False).astype(int)
    for s in SUBLAYER_ORDER:
        marker = ""
        if s == "V":
            marker = " ◄ V: fwd rank worse"
        elif s == "K":
            marker = " ◄ K: bwd rank worse"
        print(f"{s:<10} {fwd_agg[s]:>12.5f} {fwd_ranks[s]:>10d} "
              f"{bwd_agg[s]:>12.4f} {bwd_ranks[s]:>10d}{marker}")

    print(f"\nForward rank  (worst→best): {' > '.join(fwd_ranked.index)}")
    print(f"Backward rank (worst→best): {' > '.join(bwd_ranked.index)}")


def print_physical_explanation():
    """Print why V and K swap between forward and backward."""
    print("\n" + "=" * 70)
    print("  Physical Explanation: Why V and K Swap")
    print("=" * 70)
    explanation = """
  Forward (V가 민감):
    - V projection의 activation 분포가 Q/K보다 넓음 (dynamic range ↑)
    - ADC 양자화 시 동일 bit수에서 step_size가 커짐 → NMSE ↑
    - V의 평균 SNR = 25.4 dB vs K의 29.0 dB (@ ADC 8-bit sweep)

  Backward (K가 민감):
    - K gradient의 ODR(outlier dominance ratio) = 41.1 vs V의 27.9
    - Attention score backward에서 K gradient는 Q·softmax(QK^T)의
      outer product로 형성 → softmax의 peaky 분포가 K grad에 전파
    - AbsMax normalization 후 within-vector quantization에서
      outlier가 resolution을 독점 → 나머지 원소가 zero로 양자화됨
    - 결과: K의 QZR(0.20) > V의 QZR(0.14) @ DAC 7-bit

  핵심 요약:
    - Forward: activation magnitude 분포가 sensitivity를 결정
    - Backward: gradient outlier 구조가 sensitivity를 결정
    - 두 메커니즘이 다르므로 sublayer 순서가 달라짐"""
    print(explanation)


def print_recommendation(rec_df):
    """Print final mixed-precision recommendation."""
    print("\n" + "=" * 70)
    print("  Mixed-Precision Recommendation (Forward + Backward)")
    print("=" * 70)
    print(f"\n{rec_df.to_string()}\n")

    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  Forward (ADC):  FFN1 > V > FFN2 > Q ≈ K > O  (V 나쁨)    │")
    print("│  Backward (DAC): FFN1 > K > Q > V > O ≈ FFN2  (K 나쁨)    │")
    print("│                                                             │")
    print("│  → V는 activation 분포가 넓어서 forward에서 민감            │")
    print("│  → K는 gradient outlier가 심해서 backward에서 민감           │")
    print("│  → Mixed-precision은 BOTH 방향을 고려해야 함                │")
    print("└─────────────────────────────────────────────────────────────┘")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    bwd_df = load_backward_dac()
    fwd_sweep = load_forward_adc_sweep()
    fwd_perlayer = load_forward_adc_perlayer()
    fwd_layer_bits = load_forward_adc_layer_bits()

    print(f"  Backward DAC: {len(bwd_df)} rows, bits={sorted(bwd_df['dac_bits'].unique())}")
    print(f"  Forward ADC sweep: {len(fwd_sweep)} rows")
    print(f"  Forward ADC per-layer (8-bit): {len(fwd_perlayer)} rows")
    print(f"  Forward ADC layer×bits: {len(fwd_layer_bits)} rows")

    # Generate figures
    fwd_agg, bwd_agg = fig1_sensitivity_summary(bwd_df, fwd_perlayer)
    fig2_rank_heatmap(bwd_df, fwd_layer_bits)
    rec_df = fig3_recommendation_matrix(bwd_df, fwd_layer_bits)

    # Console output
    print_comparison_table(fwd_agg, bwd_agg)
    print_physical_explanation()
    print_recommendation(rec_df)
    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

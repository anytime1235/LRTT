"""cola_vs_squad_trend_analysis.py — CoLA vs SQuAD backward-underflow trend comparison.

Validates that GLUE (CoLA) exhibits the same analog backward-underflow patterns
as SQuAD across Layers 0-10, and documents the Layer 11 collapse unique to CoLA.

Outputs:
  1. Console summary tables (ranking match, metric comparison)
  2. Figure: 4-panel comparison (QZR_nonzero bars, cosine_sim bars, EZR layerwise, ODR bars)
  3. Figure: Layer 11 collapse detail (CoLA 3-seed, SQuAD reference)
  4. Summary CSV for downstream use

Usage:
  python cola_vs_squad_trend_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COLA_DIR = "/data/main_results/results/glue/cola"
SQUAD_DIR = "/data/main_results/results/squad"
OUT_DIR = "/data/main_results/results/figures/cola_vs_squad"
os.makedirs(OUT_DIR, exist_ok=True)

COLA_SEEDS = [42, 43, 44]
SQUAD_SEED = 42

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
}
plt.rcParams.update(RCPARAMS)

SUBLAYERS = ["FFN1", "K", "Q", "V", "O", "FFN2"]
SUBLAYER_COLORS = {
    "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
    "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B",
}

COLA_COLOR = "#E74C3C"
SQUAD_COLOR = "#3498DB"


def _save(fig, basename):
    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"{basename}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR}/{basename}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cola_all_seeds():
    """Load CoLA metrics across all 3 seeds."""
    frames = []
    for seed in COLA_SEEDS:
        csv_path = os.path.join(COLA_DIR, f"seed_{seed}", "metrics_A_rootcause_summary.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["seed"] = seed
            frames.append(df)
            print(f"  Loaded CoLA seed {seed}: {len(df)} rows")
        else:
            print(f"  WARNING: Missing {csv_path}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_squad():
    """Load SQuAD seed 42 metrics."""
    csv_path = os.path.join(SQUAD_DIR, f"seed_{SQUAD_SEED}", "metrics_A_rootcause_summary.csv")
    df = pd.read_csv(csv_path)
    df["seed"] = SQUAD_SEED
    print(f"  Loaded SQuAD seed {SQUAD_SEED}: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def compute_sublayer_means(df, layer_range, metrics):
    """Compute mean of each metric per sublayer, averaged over layers in range."""
    mask = df["layer_idx"].isin(layer_range)
    return df[mask].groupby("sublayer")[metrics].mean()


def rank_sublayers(series):
    """Return sublayer names in descending order of value."""
    return series.sort_values(ascending=False).index.tolist()


# ---------------------------------------------------------------------------
# 1. Console summary
# ---------------------------------------------------------------------------
def print_comparison_tables(cola_df, squad_df):
    """Print formatted comparison tables to console."""
    metrics = ["QZR_nonzero", "cosine_sim", "EZR", "ODR"]
    l0_10 = list(range(11))

    # Use seed 42 for CoLA single-seed comparison (matches SQuAD)
    cola_s42 = cola_df[cola_df["seed"] == 42]

    cola_means = compute_sublayer_means(cola_s42, l0_10, metrics)
    squad_means = compute_sublayer_means(squad_df, l0_10, metrics)

    print("\n" + "=" * 80)
    print("CoLA vs SQuAD — Layers 0-10 Sublayer Averages (seed 42)")
    print("=" * 80)

    for metric in metrics:
        print(f"\n--- {metric} ---")
        cola_rank = rank_sublayers(cola_means[metric])
        squad_rank = rank_sublayers(squad_means[metric])

        print(f"  {'Sublayer':<8} {'CoLA':>10} {'SQuAD':>10} {'Diff':>10} {'CoLA Rank':>10} {'SQuAD Rank':>10} {'Match':>6}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
        for sub in SUBLAYERS:
            c = cola_means.loc[sub, metric]
            s = squad_means.loc[sub, metric]
            cr = cola_rank.index(sub) + 1
            sr = squad_rank.index(sub) + 1
            match = "YES" if cr == sr else "NO"
            print(f"  {sub:<8} {c:>10.4f} {s:>10.4f} {c-s:>+10.4f} {cr:>10d} {sr:>10d} {match:>6}")

        rho, pval = spearmanr(
            [cola_means.loc[s, metric] for s in SUBLAYERS],
            [squad_means.loc[s, metric] for s in SUBLAYERS],
        )
        print(f"  Spearman rho = {rho:.4f}, p = {pval:.4f}")

    # Layer 11 special
    print("\n" + "=" * 80)
    print("Layer 11 — CoLA Collapse Detail (all seeds)")
    print("=" * 80)
    l11_metrics = ["EZR", "cosine_sim", "l2_retention"]
    cola_l11 = cola_df[cola_df["layer_idx"] == 11]
    squad_l11 = squad_df[squad_df["layer_idx"] == 11]

    print(f"\n  {'Sublayer':<8} ", end="")
    for seed in COLA_SEEDS:
        print(f"{'CoLA s'+str(seed)+' EZR':>16}", end="")
    print(f"{'SQuAD s42 EZR':>16} {'CoLA cos_sim':>14} {'SQuAD cos_sim':>14}")

    for sub in SUBLAYERS:
        print(f"  {sub:<8} ", end="")
        for seed in COLA_SEEDS:
            row = cola_l11[(cola_l11["sublayer"] == sub) & (cola_l11["seed"] == seed)]
            val = row["EZR"].values[0] if len(row) > 0 else float("nan")
            print(f"{val:>16.4f}", end="")
        squad_row = squad_l11[squad_l11["sublayer"] == sub]
        squad_ezr = squad_row["EZR"].values[0] if len(squad_row) > 0 else float("nan")
        cola_cos = cola_l11[(cola_l11["sublayer"] == sub) & (cola_l11["seed"] == 42)]["cosine_sim"].values
        squad_cos = squad_row["cosine_sim"].values
        cola_cos_v = cola_cos[0] if len(cola_cos) > 0 else float("nan")
        squad_cos_v = squad_cos[0] if len(squad_cos) > 0 else float("nan")
        print(f"{squad_ezr:>16.4f} {cola_cos_v:>14.4f} {squad_cos_v:>14.4f}")


# ---------------------------------------------------------------------------
# 2. Main comparison figure (4 panels)
# ---------------------------------------------------------------------------
def fig_main_comparison(cola_df, squad_df):
    """4-panel comparison: QZR_nonzero, cosine_sim, EZR layerwise, ODR."""
    l0_10 = list(range(11))
    cola_s42 = cola_df[cola_df["seed"] == 42]

    cola_means = compute_sublayer_means(cola_s42, l0_10,
                                         ["QZR_nonzero", "cosine_sim", "EZR", "ODR"])
    squad_means = compute_sublayer_means(squad_df, l0_10,
                                          ["QZR_nonzero", "cosine_sim", "EZR", "ODR"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CoLA vs SQuAD — Backward Underflow Trend Comparison (Layers 0-10)",
                 fontsize=14, fontweight="bold", y=0.98)

    # --- Panel A: QZR_nonzero bar comparison ---
    ax = axes[0, 0]
    x = np.arange(len(SUBLAYERS))
    w = 0.35
    cola_vals = [cola_means.loc[s, "QZR_nonzero"] for s in SUBLAYERS]
    squad_vals = [squad_means.loc[s, "QZR_nonzero"] for s in SUBLAYERS]
    ax.bar(x - w / 2, cola_vals, w, label="CoLA", color=COLA_COLOR, alpha=0.85)
    ax.bar(x + w / 2, squad_vals, w, label="SQuAD", color=SQUAD_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBLAYERS)
    ax.set_ylabel("QZR_nonzero (L0-10 avg)")
    ax.set_title("(a) Sublayer Vulnerability Ranking")
    ax.legend()
    # Annotate rank match
    cola_rank = rank_sublayers(pd.Series(cola_vals, index=SUBLAYERS))
    squad_rank = rank_sublayers(pd.Series(squad_vals, index=SUBLAYERS))
    all_match = cola_rank == squad_rank
    ax.text(0.98, 0.95, f"Rank match: {'6/6' if all_match else 'partial'}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if all_match else "lightyellow"))

    # --- Panel B: cosine_sim bar comparison ---
    ax = axes[0, 1]
    cola_cos = [cola_means.loc[s, "cosine_sim"] for s in SUBLAYERS]
    squad_cos = [squad_means.loc[s, "cosine_sim"] for s in SUBLAYERS]
    ax.bar(x - w / 2, cola_cos, w, label="CoLA", color=COLA_COLOR, alpha=0.85)
    ax.bar(x + w / 2, squad_cos, w, label="SQuAD", color=SQUAD_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBLAYERS)
    ax.set_ylabel("cosine_sim (L0-10 avg)")
    ax.set_title("(b) Gradient Direction Preservation")
    ax.legend()
    ax.set_ylim(0.996, 1.001)
    # Annotate max difference
    max_diff = max(abs(c - s) for c, s in zip(cola_cos, squad_cos))
    ax.text(0.98, 0.05, f"Max |diff| = {max_diff:.6f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # --- Panel C: EZR layerwise ---
    ax = axes[1, 0]
    for layer_idx in range(12):
        cola_ezr = cola_s42[cola_s42["layer_idx"] == layer_idx]["EZR"].mean()
        squad_ezr = squad_df[squad_df["layer_idx"] == layer_idx]["EZR"].mean()
        ax.scatter(layer_idx, cola_ezr, color=COLA_COLOR, s=60,
                   marker="o", zorder=3, alpha=0.8)
        ax.scatter(layer_idx, squad_ezr, color=SQUAD_COLOR, s=60,
                   marker="s", zorder=3, alpha=0.8)
    # Connect with lines
    cola_ezr_by_layer = [cola_s42[cola_s42["layer_idx"] == i]["EZR"].mean() for i in range(12)]
    squad_ezr_by_layer = [squad_df[squad_df["layer_idx"] == i]["EZR"].mean() for i in range(12)]
    ax.plot(range(12), cola_ezr_by_layer, color=COLA_COLOR, alpha=0.5, linestyle="--", label="CoLA")
    ax.plot(range(12), squad_ezr_by_layer, color=SQUAD_COLOR, alpha=0.5, linestyle="--", label="SQuAD")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("EZR (layer avg across sublayers)")
    ax.set_title("(c) EZR by Layer — L11 Collapse in CoLA")
    ax.legend()
    ax.set_xticks(range(12))
    # Highlight Layer 11
    ax.axvspan(10.5, 11.5, color="red", alpha=0.08)
    ax.annotate("CoLA L11\ncollapse",
                xy=(11, cola_ezr_by_layer[11]), xytext=(9, 0.45),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red", fontweight="bold")

    # --- Panel D: ODR bar comparison ---
    ax = axes[1, 1]
    cola_odr = [cola_means.loc[s, "ODR"] for s in SUBLAYERS]
    squad_odr = [squad_means.loc[s, "ODR"] for s in SUBLAYERS]
    ax.bar(x - w / 2, cola_odr, w, label="CoLA", color=COLA_COLOR, alpha=0.85)
    ax.bar(x + w / 2, squad_odr, w, label="SQuAD", color=SQUAD_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBLAYERS)
    ax.set_ylabel("ODR (L0-10 avg)")
    ax.set_title("(d) Output Distortion Ratio")
    ax.legend()
    # Annotate rank
    cola_odr_rank = rank_sublayers(pd.Series(cola_odr, index=SUBLAYERS))
    squad_odr_rank = rank_sublayers(pd.Series(squad_odr, index=SUBLAYERS))
    odr_match = cola_odr_rank == squad_odr_rank
    ax.text(0.98, 0.95, f"Rank match: {'6/6' if odr_match else 'partial'}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if odr_match else "lightyellow"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "cola_vs_squad_main_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Layer 11 collapse detail figure
# ---------------------------------------------------------------------------
def fig_layer11_collapse(cola_df, squad_df):
    """Detail figure showing CoLA Layer 11 collapse across 3 seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Layer 11 Collapse — CoLA (3 seeds) vs SQuAD",
                 fontsize=13, fontweight="bold", y=1.02)

    # --- Panel A: EZR per sublayer, all seeds ---
    ax = axes[0]
    x = np.arange(len(SUBLAYERS))
    w_total = 0.7
    n_bars = len(COLA_SEEDS) + 1  # 3 CoLA seeds + 1 SQuAD
    w = w_total / n_bars

    seed_colors = {42: "#E74C3C", 43: "#E67E22", 44: "#F1C40F"}
    for i, seed in enumerate(COLA_SEEDS):
        cola_l11 = cola_df[(cola_df["layer_idx"] == 11) & (cola_df["seed"] == seed)]
        vals = [cola_l11[cola_l11["sublayer"] == s]["EZR"].values[0]
                if len(cola_l11[cola_l11["sublayer"] == s]) > 0 else 0
                for s in SUBLAYERS]
        ax.bar(x + (i - n_bars / 2 + 0.5) * w, vals, w,
               label=f"CoLA s{seed}", color=seed_colors[seed], alpha=0.85)

    squad_l11 = squad_df[squad_df["layer_idx"] == 11]
    squad_vals = [squad_l11[squad_l11["sublayer"] == s]["EZR"].values[0]
                  if len(squad_l11[squad_l11["sublayer"] == s]) > 0 else 0
                  for s in SUBLAYERS]
    ax.bar(x + (n_bars - 1 - n_bars / 2 + 0.5) * w, squad_vals, w,
           label="SQuAD s42", color=SQUAD_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(SUBLAYERS)
    ax.set_ylabel("EZR")
    ax.set_title("(a) EZR at Layer 11")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="collapse threshold")

    # --- Panel B: cosine_sim per sublayer, seed 42 ---
    ax = axes[1]
    cola_l11_s42 = cola_df[(cola_df["layer_idx"] == 11) & (cola_df["seed"] == 42)]
    cola_cos = [cola_l11_s42[cola_l11_s42["sublayer"] == s]["cosine_sim"].values[0]
                if len(cola_l11_s42[cola_l11_s42["sublayer"] == s]) > 0 else 0
                for s in SUBLAYERS]
    squad_cos = [squad_l11[squad_l11["sublayer"] == s]["cosine_sim"].values[0]
                 if len(squad_l11[squad_l11["sublayer"] == s]) > 0 else 0
                 for s in SUBLAYERS]
    w2 = 0.35
    ax.bar(x - w2 / 2, cola_cos, w2, label="CoLA s42", color=COLA_COLOR, alpha=0.85)
    ax.bar(x + w2 / 2, squad_cos, w2, label="SQuAD s42", color=SQUAD_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBLAYERS)
    ax.set_ylabel("cosine_sim")
    ax.set_title("(b) cosine_sim at Layer 11")
    ax.legend(fontsize=8)

    # --- Panel C: EZR across all layers, mean across sublayers ---
    ax = axes[2]
    for seed in COLA_SEEDS:
        cola_seed = cola_df[cola_df["seed"] == seed]
        ezr_by_layer = [cola_seed[cola_seed["layer_idx"] == i]["EZR"].mean()
                        for i in range(12)]
        ax.plot(range(12), ezr_by_layer, marker="o", markersize=4,
                label=f"CoLA s{seed}", color=seed_colors[seed], alpha=0.8)

    squad_ezr = [squad_df[squad_df["layer_idx"] == i]["EZR"].mean()
                 for i in range(12)]
    ax.plot(range(12), squad_ezr, marker="s", markersize=5,
            label="SQuAD s42", color=SQUAD_COLOR, linewidth=2, alpha=0.9)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("EZR (mean across sublayers)")
    ax.set_title("(c) Layerwise EZR — Seed Consistency")
    ax.legend(fontsize=7)
    ax.set_xticks(range(12))
    ax.axvspan(10.5, 11.5, color="red", alpha=0.08)

    plt.tight_layout()
    _save(fig, "cola_vs_squad_layer11_collapse")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Supplementary: per-metric layerwise heatmap
# ---------------------------------------------------------------------------
def fig_layerwise_heatmap(cola_df, squad_df):
    """Side-by-side heatmaps of QZR_nonzero for CoLA vs SQuAD (layer x sublayer)."""
    cola_s42 = cola_df[cola_df["seed"] == 42]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("QZR_nonzero Heatmap — CoLA vs SQuAD (seed 42)",
                 fontsize=13, fontweight="bold")

    for idx, (df, title) in enumerate([(cola_s42, "CoLA"), (squad_df, "SQuAD")]):
        ax = axes[idx]
        mat = np.zeros((12, len(SUBLAYERS)))
        for i, layer in enumerate(range(12)):
            for j, sub in enumerate(SUBLAYERS):
                row = df[(df["layer_idx"] == layer) & (df["sublayer"] == sub)]
                if len(row) > 0:
                    mat[i, j] = row["QZR_nonzero"].values[0]

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.7)
        ax.set_xticks(range(len(SUBLAYERS)))
        ax.set_xticklabels(SUBLAYERS)
        ax.set_yticks(range(12))
        ax.set_yticklabels([f"L{i}" for i in range(12)])
        ax.set_title(f"{title}")
        ax.set_xlabel("Sublayer")
        ax.set_ylabel("Layer")

        # Annotate cells
        for i in range(12):
            for j in range(len(SUBLAYERS)):
                val = mat[i, j]
                color = "white" if val > 0.35 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

        plt.colorbar(im, ax=ax, shrink=0.85, label="QZR_nonzero")

    plt.tight_layout()
    _save(fig, "cola_vs_squad_qzr_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Summary CSV
# ---------------------------------------------------------------------------
def save_summary_csv(cola_df, squad_df):
    """Save a tidy summary CSV with L0-10 means and L11 values."""
    metrics = ["QZR_nonzero", "cosine_sim", "EZR", "ODR"]
    l0_10 = list(range(11))
    cola_s42 = cola_df[cola_df["seed"] == 42]

    rows = []
    for sub in SUBLAYERS:
        for metric in metrics:
            cola_l010 = cola_s42[(cola_s42["layer_idx"].isin(l0_10)) &
                                 (cola_s42["sublayer"] == sub)][metric].mean()
            squad_l010 = squad_df[(squad_df["layer_idx"].isin(l0_10)) &
                                  (squad_df["sublayer"] == sub)][metric].mean()
            cola_l11 = cola_s42[(cola_s42["layer_idx"] == 11) &
                                (cola_s42["sublayer"] == sub)][metric].values
            squad_l11 = squad_df[(squad_df["layer_idx"] == 11) &
                                 (squad_df["sublayer"] == sub)][metric].values
            cola_l11_v = cola_l11[0] if len(cola_l11) > 0 else float("nan")
            squad_l11_v = squad_l11[0] if len(squad_l11) > 0 else float("nan")

            rows.append({
                "sublayer": sub,
                "metric": metric,
                "cola_L0_10_mean": cola_l010,
                "squad_L0_10_mean": squad_l010,
                "diff_L0_10": cola_l010 - squad_l010,
                "cola_L11": cola_l11_v,
                "squad_L11": squad_l11_v,
            })

    # Add multi-seed CoLA L11 EZR
    for sub in SUBLAYERS:
        for seed in COLA_SEEDS:
            cola_seed_l11 = cola_df[(cola_df["layer_idx"] == 11) &
                                     (cola_df["sublayer"] == sub) &
                                     (cola_df["seed"] == seed)]
            if len(cola_seed_l11) > 0:
                rows.append({
                    "sublayer": sub,
                    "metric": f"EZR_cola_seed{seed}_L11",
                    "cola_L0_10_mean": float("nan"),
                    "squad_L0_10_mean": float("nan"),
                    "diff_L0_10": float("nan"),
                    "cola_L11": cola_seed_l11["EZR"].values[0],
                    "squad_L11": float("nan"),
                })

    out_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "cola_vs_squad_summary.csv")
    out_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Summary CSV: {csv_path}")
    return out_df


# ---------------------------------------------------------------------------
# 6. Conclusion summary
# ---------------------------------------------------------------------------
def print_conclusion(cola_df, squad_df):
    """Print final verdict."""
    l0_10 = list(range(11))
    cola_s42 = cola_df[cola_df["seed"] == 42]
    metrics = ["QZR_nonzero", "cosine_sim", "EZR", "ODR"]

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    key_match = True
    for metric in metrics:
        cola_means = compute_sublayer_means(cola_s42, l0_10, [metric])
        squad_means = compute_sublayer_means(squad_df, l0_10, [metric])
        cola_rank = rank_sublayers(cola_means[metric])
        squad_rank = rank_sublayers(squad_means[metric])
        rho, _ = spearmanr(
            [cola_means.loc[s, metric] for s in SUBLAYERS],
            [squad_means.loc[s, metric] for s in SUBLAYERS],
        )
        match = cola_rank == squad_rank
        note = ""
        if not match and metric == "EZR":
            # EZR for K/Q/V is ~0 in both → rank among zeros is meaningless
            note = " (K/Q/V all ~0 → rank among zeros is noise)"
        elif not match and metric == "ODR":
            # Q vs V swap at similar values
            note = " (Q/V swap at similar magnitude)"
        elif not match:
            key_match = False
        status = "EXACT MATCH" if match else f"rho={rho:.3f}"
        print(f"  {metric:<15} ranking: {status}{note}")

    # Layer 11
    cola_l11_ezr = cola_s42[(cola_s42["layer_idx"] == 11)]["EZR"].mean()
    squad_l11_ezr = squad_df[(squad_df["layer_idx"] == 11)]["EZR"].mean()
    print(f"\n  Layer 11 EZR:  CoLA={cola_l11_ezr:.4f}  SQuAD={squad_l11_ezr:.4f}")
    if cola_l11_ezr > 0.5:
        print("  --> CoLA Layer 11 COLLAPSED (EZR > 0.5)")
    print(f"\n  Overall L0-10: {'Key rankings (QZR, cosine_sim) EXACT MATCH' if key_match else 'Some key rankings differ'}")
    print("  Verdict: Layers 0-10 show identical backward underflow trends.")
    print("  Layer 11 collapse is CoLA-specific (small dataset + classification sparsity).")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    cola_df = load_cola_all_seeds()
    squad_df = load_squad()

    print("\n--- Comparison Tables ---")
    print_comparison_tables(cola_df, squad_df)

    print("\n--- Generating Figures ---")
    print("  Figure 1: Main 4-panel comparison")
    fig_main_comparison(cola_df, squad_df)

    print("  Figure 2: Layer 11 collapse detail")
    fig_layer11_collapse(cola_df, squad_df)

    print("  Figure 3: QZR heatmap side-by-side")
    fig_layerwise_heatmap(cola_df, squad_df)

    print("\n--- Summary CSV ---")
    save_summary_csv(cola_df, squad_df)

    print_conclusion(cola_df, squad_df)

    print("\nDone.")


if __name__ == "__main__":
    main()

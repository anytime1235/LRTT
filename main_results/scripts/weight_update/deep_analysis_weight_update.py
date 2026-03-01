#!/usr/bin/env python3
"""Deep Analysis: TikiTaka Weight Update Diagnostics.

Comprehensive 7-stage analysis of weight update diagnostics across
dw_min settings, layers, sublayers, training phases, and modes.

Builds on analyze_seed_variance.py (seed variance → single seed sufficient)
to provide actionable insights for analog training optimization.

Usage:
    python deep_analysis_weight_update.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, spearmanr, linregress

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning)

FIG_DIR = "/data/main_results/scripts/diagnosis/figures"

# ── Paths ─────────────────────────────────────────────────────────────────────

PRIMARY_BASE = "/data/main_results/weight_update/squad/tiki"
V3_BASE = "/data/main_results/scripts/diagnosis/results/diag_weight_update"

DW_CONFIGS = {
    0.0005: {"run": "run_f2fad8e3f307", "prefix": "dw0p0005"},
    0.005:  {"run": "run_4249343b8006", "prefix": "dw0p0050"},
}
SEEDS = [0, 1, 2]

# ── Health Thresholds ─────────────────────────────────────────────────────────

HEALTH_THRESHOLDS = {
    "dw_zero_ratio":         {"good_hi": 0.95, "warn_hi": 0.99},      # <0.95 GOOD, 0.95-0.99 WARN, >=0.99 BAD
    "update_vs_grad_cosine": {"good_lo": 0.05, "warn_lo": 0.01},      # >0.05 GOOD, 0.01-0.05 WARN, <0.01 BAD
    "sign_mismatch_ratio":   {"good_hi": 0.90, "warn_hi": 0.95},      # <0.90 GOOD, 0.90-0.95 WARN, >=0.95 BAD
    "pulse_ok_frac":         {"good_lo": 0.5,  "warn_lo": 0.2},       # >0.5 GOOD, 0.2-0.5 WARN, <0.2 BAD
    "transfer_efficiency":   {"good_lo": 0.001, "warn_lo": 1e-4},     # >0.001 GOOD, 1e-4-0.001 WARN, <1e-4 BAD
}

# Key metrics for cross-analysis
KEY_METRICS = [
    "dw_zero_ratio", "dw_absmean", "grad_absmean", "grad_deadzone_ratio",
    "update_vs_grad_cosine", "eff_lr_slope", "BL_mean", "pulse_ok_frac",
    "sign_mismatch_ratio", "rel_update_error",
]

# TikiTaka-specific metrics
TIKI_METRICS = [
    "dw_fast_absmean", "dw_slow_absmean", "dw_fast_zero_ratio", "dw_slow_zero_ratio",
    "dw_fast_vs_grad_cosine", "hidden_absmean", "hidden_absmax",
    "transfer_duty", "transfer_spike", "transfer_efficiency",
    "buffer_above_thresh_ratio",
]

# All metrics for correlation
CORR_METRICS = [
    "dw_zero_ratio", "dw_absmean", "dw_1lsb_ratio",
    "grad_absmean", "grad_deadzone_ratio",
    "update_vs_grad_cosine", "eff_lr_slope",
    "BL_mean", "BL_hit_ratio",
    "pulse_ok_frac", "pulse_under_frac", "pulse_over_frac",
    "pulse_sat_ratio", "bound_sat_ratio",
    "sign_mismatch_ratio", "rel_update_error",
    "dw_fast_absmean", "dw_slow_absmean",
    "transfer_efficiency", "transfer_duty",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def ci95_t(arr):
    """Compute mean, std, 95% CI using t-distribution."""
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    m = arr.mean()
    if n == 1:
        return m, 0.0, m, m
    sd = arr.std(ddof=1)
    t_val = t_dist.ppf(0.975, n - 1)
    ci_half = t_val * sd / np.sqrt(n)
    return m, sd, m - ci_half, m + ci_half


def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    g1, g2 = np.array(group1, dtype=float), np.array(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_sd < 1e-30:
        return np.inf if abs(m1 - m2) > 1e-30 else 0.0
    return (m1 - m2) / pooled_sd


def health_classify(metric, value):
    """Classify a metric value as GOOD/WARN/BAD."""
    if metric not in HEALTH_THRESHOLDS:
        return "N/A"
    th = HEALTH_THRESHOLDS[metric]
    if np.isnan(value):
        return "N/A"
    # Metrics where lower is better (dw_zero_ratio, sign_mismatch_ratio)
    if "good_hi" in th:
        if value < th["good_hi"]:
            return "GOOD"
        elif value < th["warn_hi"]:
            return "WARN"
        else:
            return "BAD"
    # Metrics where higher is better
    else:
        if value > th["good_lo"]:
            return "GOOD"
        elif value > th["warn_lo"]:
            return "WARN"
        else:
            return "BAD"


def section_header(title, level=1):
    """Print a formatted section header."""
    if level == 1:
        print("\n" + "=" * 90)
        print(f"  {title}")
        print("=" * 90)
    else:
        print(f"\n  --- {title} ---")


def fmt_sci(val, width=12):
    """Format a number in scientific notation or fixed."""
    if isinstance(val, (int, np.integer)):
        return f"{val:>{width}d}"
    if np.isnan(val) or np.isinf(val):
        return f"{'nan' if np.isnan(val) else 'inf':>{width}}"
    if abs(val) < 1e-3 and val != 0:
        return f"{val:>{width}.4e}"
    elif abs(val) < 1:
        return f"{val:>{width}.6f}"
    elif abs(val) < 100:
        return f"{val:>{width}.4f}"
    else:
        return f"{val:>{width}.2f}"


def get_metric_cols(df):
    """Return numeric metric columns."""
    exclude = {"layer_idx", "sublayer", "trace_every", "is_transfer_step",
               "step", "mode", "module_name", "dw_min", "desired_bl",
               "sto_round_update", "update_bl_management", "update_management", "seed"}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in (np.float64, np.float32, np.int64)]


# ── Plot Helpers & Functions ──────────────────────────────────────────────────

_SAVED_FIGURES = []


def annotate_heatmap(ax, mat, fmt=".2f"):
    """Add contrast-aware text annotations to a heatmap."""
    norm = plt.Normalize(vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            n = norm(val)
            color = "white" if n > 0.6 or n < 0.15 else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")


def plot_dose_response(results):
    """Plot 1: Cohen's d effect size and fold change bar charts."""
    if not results:
        return
    metrics = [r["metric"] for r in results]
    cohens_vals = [abs(r["cohens_d"]) if not np.isinf(r["cohens_d"]) else 0 for r in results]
    folds = [r["fold"] for r in results]
    sensitivities = [r["sensitivity"] for r in results]

    color_map = {"STRONG": "#d62728", "MODERATE": "#ff7f0e", "WEAK": "#2ca02c"}
    colors = [color_map.get(s, "grey") for s in sensitivities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(metrics) * 0.35)))

    # Left: |Cohen's d|
    y_pos = np.arange(len(metrics))
    ax1.barh(y_pos, cohens_vals, color=colors, edgecolor="k", linewidth=0.4)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics, fontsize=8)
    ax1.set_xlabel("|Cohen's d|")
    ax1.set_title("(a) dw_min Dose-Response: Effect Size")
    ax1.axvline(0.8, color="grey", ls="--", lw=1, alpha=0.7, label="d=0.8 (large)")
    ax1.axvline(2.0, color="grey", ls=":", lw=1, alpha=0.7, label="d=2.0 (very large)")
    ax1.legend(fontsize=7)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # Right: fold change
    ax2.barh(y_pos, folds, color=colors, edgecolor="k", linewidth=0.4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics, fontsize=8)
    ax2.set_xlabel("Fold Change (dw=0.005 / dw=0.0005)")
    ax2.set_title("(b) dw_min Dose-Response: Fold Change")
    ax2.axvline(1.0, color="black", ls="-", lw=1, alpha=0.5, label="fold=1")
    ax2.axvline(10.0, color="grey", ls="--", lw=1, alpha=0.7, label="fold=10 (expected)")
    ax2.legend(fontsize=7)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_dose_response.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _SAVED_FIGURES.append(path)


def plot_bottleneck_heatmap(store):
    """Plot 2: Bottleneck score heatmaps for each dw_min (12 layers x 4 sublayers)."""
    health_metrics = list(HEALTH_THRESHOLDS.keys())
    bottleneck_weight = {"BAD": 3, "WARN": 1, "GOOD": 0, "N/A": 0}
    dw_vals = sorted(DW_CONFIGS.keys())
    sublayers = ["Q", "K", "V", "O"]

    fig, axes = plt.subplots(1, len(dw_vals), figsize=(6 * len(dw_vals) + 2, 8))
    if len(dw_vals) == 1:
        axes = [axes]

    vmin, vmax = 0, 15
    for idx, dw_min in enumerate(dw_vals):
        avg_df = store.get_seed_averaged_summary(dw_min)
        if avg_df is None:
            continue
        mat = np.full((12, len(sublayers)), 0.0)
        for _, row in avg_df.iterrows():
            li = int(row["layer_idx"])
            sub = row["sublayer"]
            if sub not in sublayers or li >= 12:
                continue
            si = sublayers.index(sub)
            score = 0
            for metric in health_metrics:
                if metric in row.index:
                    status = health_classify(metric, row[metric])
                    score += bottleneck_weight.get(status, 0)
            mat[li, si] = score

        ax = axes[idx]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", origin="upper",
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(sublayers)))
        ax.set_xticklabels(sublayers, fontsize=9)
        ax.set_yticks(range(12))
        ax.set_yticklabels([f"L{i}" for i in range(12)], fontsize=8)
        ax.set_xlabel("Sublayer")
        ax.set_ylabel("Encoder Layer")
        ax.set_title(f"dw_min={dw_min}", fontsize=10)
        annotate_heatmap(ax, mat, fmt=".0f")

    fig.suptitle("Bottleneck Score Heatmap (0=healthy, 15=worst)", fontsize=12, y=1.02)
    fig.colorbar(im, ax=axes, label="Bottleneck Score", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_bottleneck_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _SAVED_FIGURES.append(path)


def plot_transfer_pipeline(store):
    """Plot 3: Transfer pipeline bar charts (fast/slow/efficiency + loss %)."""
    dw_vals = sorted(DW_CONFIGS.keys())
    transfer_metrics = ["dw_fast_absmean", "dw_slow_absmean", "transfer_efficiency"]

    # Gather data
    data = {}
    loss_data = {}
    for dw_min in dw_vals:
        avg_df = store.get_seed_averaged_summary(dw_min)
        if avg_df is None:
            continue
        avail = [m for m in transfer_metrics if m in avg_df.columns]
        data[dw_min] = {m: avg_df[m].mean() for m in avail}
        if "dw_fast_absmean" in avg_df.columns and "dw_slow_absmean" in avg_df.columns:
            fast = avg_df["dw_fast_absmean"].mean()
            slow = avg_df["dw_slow_absmean"].mean()
            loss_data[dw_min] = (1 - slow / fast) * 100 if fast > 1e-30 else 100.0

    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: grouped bar chart
    avail_metrics = [m for m in transfer_metrics if all(m in data[dw] for dw in data)]
    x = np.arange(len(avail_metrics))
    width = 0.35
    for i, dw_min in enumerate(dw_vals):
        if dw_min not in data:
            continue
        vals = [data[dw_min].get(m, 0) for m in avail_metrics]
        offset = -width / 2 + i * width
        ax1.bar(x + offset, vals, width, label=f"dw={dw_min}", edgecolor="k", linewidth=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(avail_metrics, fontsize=8, rotation=15, ha="right")
    ax1.set_ylabel("Value (log scale)")
    ax1.set_yscale("log")
    ax1.set_title("(a) Transfer Pipeline Metrics")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: transfer loss %
    if loss_data:
        dws = list(loss_data.keys())
        losses = [loss_data[dw] for dw in dws]
        bars = ax2.bar([f"dw={dw}" for dw in dws], losses,
                       color=["#4c72b0", "#dd8452"], edgecolor="k", linewidth=0.4)
        for bar, loss in zip(bars, losses):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{loss:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Transfer Loss (%)")
        ax2.set_title("(b) Transfer Pipeline Loss (1 - slow/fast)")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_transfer_pipeline.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _SAVED_FIGURES.append(path)


def plot_temporal_dynamics(store):
    """Plot 4: Temporal dynamics line plots across training phases."""
    plot_metrics = ["dw_absmean", "grad_absmean", "update_vs_grad_cosine",
                    "pulse_ok_frac", "transfer_efficiency"]
    dw_vals = sorted(DW_CONFIGS.keys())
    n_dw = len(dw_vals)

    fig, axes = plt.subplots(n_dw, len(plot_metrics),
                             figsize=(4 * len(plot_metrics), 4 * n_dw),
                             squeeze=False)

    for row_idx, dw_min in enumerate(dw_vals):
        step_dfs = []
        for seed in SEEDS:
            key = (dw_min, seed)
            if key in store.primary_steps:
                step_dfs.append(store.primary_steps[key].copy())
        if not step_dfs:
            continue
        combined = pd.concat(step_dfs, ignore_index=True)
        avail = [m for m in plot_metrics if m in combined.columns]
        step_avg = combined.groupby("step")[avail].mean().reset_index().sort_values("step")
        steps = step_avg["step"].values

        for col_idx, metric in enumerate(plot_metrics):
            ax = axes[row_idx, col_idx]
            if metric in step_avg.columns:
                ax.plot(steps, step_avg[metric].values, lw=1.2, color="#1f77b4")
            # Phase lines
            for phase_step, label in [(125, "early|mid"), (255, "mid|late")]:
                ax.axvline(phase_step, color="grey", ls="--", lw=0.8, alpha=0.7)
                ax.text(phase_step, ax.get_ylim()[1], label, fontsize=6,
                        ha="center", va="bottom", rotation=90, alpha=0.6)
            ax.set_xlabel("Step", fontsize=8)
            ax.set_title(f"{metric}\n(dw={dw_min})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Temporal Training Dynamics", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_temporal_dynamics.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _SAVED_FIGURES.append(path)


def plot_correlation_matrix(corr_matrix, metric_names):
    """Plot 5: Spearman correlation matrix heatmap."""
    n = len(metric_names)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(8, n * 0.5)))
    sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                xticklabels=metric_names, yticklabels=metric_names,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Spearman Correlation Matrix (lower triangle)", fontsize=11)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_correlation_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _SAVED_FIGURES.append(path)


# ── DataStore ─────────────────────────────────────────────────────────────────

class DataStore:
    """Load and manage all diagnostic data sources."""

    def __init__(self):
        self.primary_summary = {}   # (dw_min, seed) → DataFrame
        self.primary_steps = {}     # (dw_min, seed) → DataFrame
        self.mode_summary = {}      # "single"|"tiki" → DataFrame
        self.agg_layer = {}         # "v3_mseed"|"v3_protA" → DataFrame

    def load_primary_tiki(self):
        """Load primary multi-seed TikiTaka data (dw=0.0005 and dw=0.005)."""
        print("\n  Loading primary TikiTaka data...")
        for dw_min, cfg in DW_CONFIGS.items():
            for seed in SEEDS:
                tag = f"{cfg['prefix']}_seed{seed}"
                base = os.path.join(PRIMARY_BASE, cfg["run"], tag)
                summ_path = os.path.join(base, f"{tag}_summary.csv")
                step_path = os.path.join(base, f"{tag}_step_metrics.csv")
                if os.path.exists(summ_path):
                    self.primary_summary[(dw_min, seed)] = pd.read_csv(summ_path)
                else:
                    print(f"    [WARN] Missing: {summ_path}")
                if os.path.exists(step_path):
                    self.primary_steps[(dw_min, seed)] = pd.read_csv(step_path)
                else:
                    print(f"    [WARN] Missing: {step_path}")
        n_summ = len(self.primary_summary)
        n_step = len(self.primary_steps)
        print(f"    Loaded {n_summ} summary + {n_step} step_metrics CSVs")

    def load_mode_compare(self):
        """Load v3_single and v3_tiki for mode comparison."""
        print("  Loading mode comparison data...")
        for mode in ["single", "tiki"]:
            path = os.path.join(V3_BASE, f"v3_{mode}", f"v3_{mode}_summary.csv")
            if os.path.exists(path):
                self.mode_summary[mode] = pd.read_csv(path)
                print(f"    v3_{mode}: {len(self.mode_summary[mode])} rows")
            else:
                print(f"    [WARN] Missing: {path}")

    def load_multiseed(self):
        """Load v3_mseed and v3_protA aggregated layer summaries."""
        print("  Loading multi-seed aggregated data...")
        for tag in ["v3_mseed", "v3_protA"]:
            path = os.path.join(V3_BASE, tag, "aggregated_layer_summary.csv")
            if os.path.exists(path):
                self.agg_layer[tag] = pd.read_csv(path)
                print(f"    {tag}: {len(self.agg_layer[tag])} rows")
            else:
                print(f"    [INFO] Not found: {path}")

    def report_inventory(self):
        """Print loading summary."""
        section_header("Data Inventory", 2)
        print(f"    Primary summary CSVs : {len(self.primary_summary)}")
        print(f"    Primary step CSVs    : {len(self.primary_steps)}")
        print(f"    Mode compare (v3)    : {list(self.mode_summary.keys())}")
        print(f"    Aggregated layers    : {list(self.agg_layer.keys())}")
        # Show primary data breakdown
        for dw_min in DW_CONFIGS:
            seeds_loaded = [s for s in SEEDS if (dw_min, s) in self.primary_summary]
            print(f"    dw_min={dw_min}: seeds {seeds_loaded}")

    def get_seed_averaged_summary(self, dw_min):
        """Average summary across seeds for a given dw_min. Returns DataFrame with same index."""
        dfs = []
        for seed in SEEDS:
            key = (dw_min, seed)
            if key in self.primary_summary:
                df = self.primary_summary[key].copy()
                df["seed"] = seed
                dfs.append(df)
        if not dfs:
            return None
        combined = pd.concat(dfs, ignore_index=True)
        mcols = get_metric_cols(combined)
        avg = combined.groupby(["layer_idx", "sublayer"])[mcols].mean().reset_index()
        return avg


# ── Analysis 1: dw_min Dose-Response Profile ─────────────────────────────────

def analysis_1_dose_response(store):
    """dw_min dose-response: seed-averaged metrics, fold change, Cohen's d."""
    section_header("ANALYSIS 1: dw_min Dose-Response Profile")

    dw_vals = sorted(DW_CONFIGS.keys())
    dw_lo, dw_hi = dw_vals[0], dw_vals[1]
    ratio_dw = dw_hi / dw_lo  # expected 10x

    # Collect per-seed global means for each (dw_min, metric)
    metrics_to_check = KEY_METRICS + [
        "dw_fast_absmean", "dw_slow_absmean", "transfer_efficiency",
        "dw_1lsb_ratio", "pulse_under_frac", "pulse_over_frac",
        "BL_hit_ratio", "dw_absmax", "dw_q99",
    ]
    metrics_to_check = list(dict.fromkeys(metrics_to_check))  # deduplicate

    results = []
    for metric in metrics_to_check:
        vals_lo, vals_hi = [], []
        for seed in SEEDS:
            for dw_min, vals_list in [(dw_lo, vals_lo), (dw_hi, vals_hi)]:
                key = (dw_min, seed)
                if key in store.primary_summary and metric in store.primary_summary[key].columns:
                    v = store.primary_summary[key][metric].mean()
                    vals_list.append(v)

        if len(vals_lo) < 2 or len(vals_hi) < 2:
            continue

        m_lo, sd_lo, ci_lo_lo, ci_lo_hi = ci95_t(vals_lo)
        m_hi, sd_hi, ci_hi_lo, ci_hi_hi = ci95_t(vals_hi)
        fold = m_hi / m_lo if abs(m_lo) > 1e-30 else np.inf
        d = cohens_d(vals_lo, vals_hi)
        abs_d = abs(d) if not np.isnan(d) and not np.isinf(d) else float("inf")

        if abs_d > 2:
            sensitivity = "STRONG"
        elif abs_d > 0.8:
            sensitivity = "MODERATE"
        else:
            sensitivity = "WEAK"

        results.append({
            "metric": metric,
            "mean_lo": m_lo, "ci_lo": f"[{ci_lo_lo:.4g}, {ci_lo_hi:.4g}]",
            "mean_hi": m_hi, "ci_hi": f"[{ci_hi_lo:.4g}, {ci_hi_hi:.4g}]",
            "fold": fold, "cohens_d": d, "sensitivity": sensitivity,
        })

    # Print table
    print(f"\n  dw_min comparison: {dw_lo} vs {dw_hi} (expected {ratio_dw:.0f}x scaling)")
    print(f"  Seeds: {SEEDS} (n={len(SEEDS)} per condition)\n")

    hdr = f"  {'Metric':<26} {'Mean(lo)':>12} {'Mean(hi)':>12} {'Fold':>8} {'Cohen d':>9} {'Sens':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        d_str = f"{r['cohens_d']:.2f}" if not np.isinf(r['cohens_d']) else "inf"
        print(f"  {r['metric']:<26} {fmt_sci(r['mean_lo'])} {fmt_sci(r['mean_hi'])} "
              f"{r['fold']:>8.3f} {d_str:>9} {r['sensitivity']:>10}")

    # dw_absmean scaling check
    dw_absmean_row = next((r for r in results if r["metric"] == "dw_absmean"), None)
    if dw_absmean_row:
        actual_fold = dw_absmean_row["fold"]
        print(f"\n  dw_absmean scaling check: expected ~{ratio_dw:.0f}x, "
              f"actual {actual_fold:.2f}x → "
              f"{'MATCHES' if 0.5 * ratio_dw < actual_fold < 2 * ratio_dw else 'MISMATCH'}")

    # Summary
    strong = sum(1 for r in results if r["sensitivity"] == "STRONG")
    moderate = sum(1 for r in results if r["sensitivity"] == "MODERATE")
    weak = sum(1 for r in results if r["sensitivity"] == "WEAK")
    print(f"\n  Sensitivity summary: {strong} STRONG, {moderate} MODERATE, {weak} WEAK")

    # Print 95% CIs for key metrics
    section_header("95% CI Details (t-distribution)", 2)
    for r in results:
        if r["metric"] in KEY_METRICS[:6]:
            print(f"    {r['metric']:<26} dw={dw_lo}: {r['ci_lo']:>30}  "
                  f"dw={dw_hi}: {r['ci_hi']:>30}")

    try:
        plot_dose_response(results)
    except Exception as e:
        print(f"  [WARN] plot_dose_response failed: {e}")

    return results


# ── Analysis 2: Layer/Sublayer Bottleneck Map ─────────────────────────────────

def analysis_2_bottleneck_map(store):
    """Identify bottleneck layers and sublayer archetypes."""
    section_header("ANALYSIS 2: Layer/Sublayer Bottleneck Map")

    health_metrics = list(HEALTH_THRESHOLDS.keys())
    bottleneck_weight = {"BAD": 3, "WARN": 1, "GOOD": 0, "N/A": 0}

    for dw_min in sorted(DW_CONFIGS.keys()):
        avg_df = store.get_seed_averaged_summary(dw_min)
        if avg_df is None:
            print(f"  [SKIP] No data for dw_min={dw_min}")
            continue

        section_header(f"dw_min = {dw_min}", 2)

        # Build health map
        rows = []
        for _, row in avg_df.iterrows():
            lidx, sub = int(row["layer_idx"]), row["sublayer"]
            entry = {"layer_idx": lidx, "sublayer": sub, "score": 0}
            flags = []
            for metric in health_metrics:
                if metric in row.index:
                    val = row[metric]
                    status = health_classify(metric, val)
                    entry[metric] = status
                    entry[f"{metric}_val"] = val
                    entry["score"] += bottleneck_weight.get(status, 0)
                    if status != "GOOD":
                        flags.append(f"{metric}={status}")
            entry["flags"] = ", ".join(flags) if flags else "-"
            rows.append(entry)

        bn_df = pd.DataFrame(rows).sort_values("score", ascending=False)

        # Print top bottlenecks
        print(f"\n  {'Layer':>5} {'Sub':>4} {'Score':>6}  Health Flags")
        print("  " + "-" * 70)
        for _, r in bn_df.head(12).iterrows():
            score_mark = "***" if r["score"] >= 6 else "** " if r["score"] >= 3 else "   "
            print(f"  L{int(r['layer_idx']):02d}   {r['sublayer']:<4} {r['score']:>5}  {score_mark} {r['flags']}")

        # Sublayer archetype analysis: Q/K vs V/O
        section_header("Sublayer Archetype (Q/K vs V/O)", 2)
        for metric in health_metrics:
            if metric not in avg_df.columns:
                continue
            qk = avg_df[avg_df["sublayer"].isin(["Q", "K"])][metric].mean()
            vo = avg_df[avg_df["sublayer"].isin(["V", "O"])][metric].mean()
            diff_pct = 100 * (vo - qk) / abs(qk) if abs(qk) > 1e-15 else 0
            marker = " <--" if abs(diff_pct) > 20 else ""
            print(f"    {metric:<26} Q/K avg={qk:.6g}  V/O avg={vo:.6g}  "
                  f"diff={diff_pct:+.1f}%{marker}")

        # Layer depth gradient: Spearman correlation with layer_idx
        section_header("Layer Depth Gradient (Spearman rho with layer_idx)", 2)
        depth_metrics = ["dw_zero_ratio", "dw_absmean", "grad_absmean",
                         "update_vs_grad_cosine", "pulse_ok_frac", "BL_mean",
                         "transfer_efficiency", "sign_mismatch_ratio"]
        # Average across sublayers per layer
        layer_avg = avg_df.groupby("layer_idx")[depth_metrics].mean().reset_index()
        for metric in depth_metrics:
            if metric not in layer_avg.columns:
                continue
            vals = layer_avg[metric].values
            idxs = layer_avg["layer_idx"].values
            if np.std(vals) < 1e-15:
                rho_str, trend = "const", "-"
            else:
                rho, pval = spearmanr(idxs, vals)
                rho_str = f"{rho:+.3f}"
                if abs(rho) > 0.7:
                    trend = "STRONG " + ("increase" if rho > 0 else "decrease")
                elif abs(rho) > 0.4:
                    trend = "moderate " + ("increase" if rho > 0 else "decrease")
                else:
                    trend = "flat"
            print(f"    {metric:<26} rho={rho_str:>8}  trend={trend}")

    try:
        plot_bottleneck_heatmap(store)
    except Exception as e:
        print(f"  [WARN] plot_bottleneck_heatmap failed: {e}")


# ── Analysis 3: TikiTaka Transfer Dynamics ────────────────────────────────────

def analysis_3_transfer_dynamics(store):
    """Analyze fast/slow tile capture and transfer pipeline efficiency."""
    section_header("ANALYSIS 3: TikiTaka Transfer Dynamics")

    for dw_min in sorted(DW_CONFIGS.keys()):
        avg_df = store.get_seed_averaged_summary(dw_min)
        if avg_df is None:
            continue

        section_header(f"dw_min = {dw_min}", 2)

        # 1) Fast tile capture efficiency: dw_fast_absmean / grad_absmean
        if "dw_fast_absmean" in avg_df.columns and "grad_absmean" in avg_df.columns:
            avg_df = avg_df.copy()
            avg_df["fast_capture_ratio"] = np.where(
                avg_df["grad_absmean"] > 1e-30,
                avg_df["dw_fast_absmean"] / avg_df["grad_absmean"],
                np.nan
            )
            fc_mean = avg_df["fast_capture_ratio"].mean()
            fc_std = avg_df["fast_capture_ratio"].std()
            print(f"\n  Fast tile capture efficiency (dw_fast / grad):")
            print(f"    Mean = {fc_mean:.6g} +/- {fc_std:.4g}")

        # 2) Transfer pipeline loss: 1 - (dw_slow_absmean / dw_fast_absmean)
        if "dw_slow_absmean" in avg_df.columns and "dw_fast_absmean" in avg_df.columns:
            avg_df["transfer_loss"] = np.where(
                avg_df["dw_fast_absmean"] > 1e-30,
                1.0 - (avg_df["dw_slow_absmean"] / avg_df["dw_fast_absmean"]),
                np.nan
            )
            tl_mean = avg_df["transfer_loss"].mean()
            tl_pct = tl_mean * 100
            print(f"\n  Transfer pipeline loss (1 - slow/fast):")
            print(f"    Mean loss = {tl_pct:.4f}%")
            print(f"    → {tl_pct:.2f}% of fast-tile signal is lost before reaching slow tile")

        # 3) Transfer metrics summary table
        transfer_cols = ["dw_fast_absmean", "dw_slow_absmean", "transfer_duty",
                         "transfer_spike", "transfer_efficiency", "buffer_above_thresh_ratio"]
        available = [c for c in transfer_cols if c in avg_df.columns]
        if available:
            print(f"\n  {'Metric':<28} {'Mean':>14} {'Std':>12} {'Min':>12} {'Max':>12}")
            print("  " + "-" * 70)
            for col in available:
                vals = avg_df[col].dropna()
                print(f"  {col:<28} {fmt_sci(vals.mean())} {fmt_sci(vals.std())} "
                      f"{fmt_sci(vals.min())} {fmt_sci(vals.max())}")

        # 4) Hidden weights check (forget_buffer mode indicator)
        if "hidden_absmean" in avg_df.columns:
            h_mean = avg_df["hidden_absmean"].mean()
            h_all_zero = (avg_df["hidden_absmean"] == 0).all()
            print(f"\n  Hidden weights: mean_absmean = {h_mean:.6g}")
            if h_all_zero:
                print(f"    → All hidden_absmean = 0 → forget_buffer mode detected")
            else:
                print(f"    → Non-zero hidden weights present → standard transfer mode")

        # 5) Per-layer transfer efficiency
        if "transfer_efficiency" in avg_df.columns:
            print(f"\n  Per-layer transfer efficiency:")
            print(f"  {'Layer':>5} {'Q':>12} {'K':>12} {'V':>12} {'O':>12}")
            print("  " + "-" * 55)
            for lidx in range(12):
                vals = []
                for sub in ["Q", "K", "V", "O"]:
                    mask = (avg_df["layer_idx"] == lidx) & (avg_df["sublayer"] == sub)
                    v = avg_df.loc[mask, "transfer_efficiency"]
                    vals.append(v.values[0] if len(v) > 0 else np.nan)
                print(f"  L{lidx:02d}   " + "  ".join(fmt_sci(v) for v in vals))

    # Cross-dw_min transfer comparison
    section_header("Cross dw_min Transfer Comparison", 2)
    dw_vals = sorted(DW_CONFIGS.keys())
    if len(dw_vals) == 2:
        compare_metrics = ["dw_fast_absmean", "dw_slow_absmean",
                           "transfer_efficiency", "transfer_duty"]
        avg0 = store.get_seed_averaged_summary(dw_vals[0])
        avg1 = store.get_seed_averaged_summary(dw_vals[1])
        if avg0 is not None and avg1 is not None:
            print(f"\n  {'Metric':<28} {'dw=' + str(dw_vals[0]):>14} {'dw=' + str(dw_vals[1]):>14} {'Ratio':>10}")
            print("  " + "-" * 70)
            for metric in compare_metrics:
                if metric in avg0.columns and metric in avg1.columns:
                    m0 = avg0[metric].mean()
                    m1 = avg1[metric].mean()
                    ratio = m1 / m0 if abs(m0) > 1e-30 else np.inf
                    print(f"  {metric:<28} {fmt_sci(m0)} {fmt_sci(m1)} {ratio:>10.3f}")

    try:
        plot_transfer_pipeline(store)
    except Exception as e:
        print(f"  [WARN] plot_transfer_pipeline failed: {e}")


# ── Analysis 4: Temporal Training Dynamics ────────────────────────────────────

def analysis_4_temporal_dynamics(store):
    """Analyze training dynamics across time phases."""
    section_header("ANALYSIS 4: Temporal Training Dynamics")

    # Phase definitions
    PHASES = {
        "early":  (0, 125),
        "mid":    (126, 255),
        "late":   (256, 383),
    }

    temporal_metrics = [
        "dw_zero_ratio", "dw_absmean", "grad_absmean",
        "update_vs_grad_cosine", "pulse_ok_frac", "BL_mean",
        "sign_mismatch_ratio", "transfer_efficiency",
        "dw_fast_absmean", "dw_slow_absmean",
    ]

    for dw_min in sorted(DW_CONFIGS.keys()):
        section_header(f"dw_min = {dw_min}", 2)

        # Collect step_metrics across seeds, average per (step, metric)
        step_dfs = []
        for seed in SEEDS:
            key = (dw_min, seed)
            if key in store.primary_steps:
                df = store.primary_steps[key].copy()
                step_dfs.append(df)

        if not step_dfs:
            print("  [SKIP] No step data")
            continue

        combined = pd.concat(step_dfs, ignore_index=True)
        available_metrics = [m for m in temporal_metrics if m in combined.columns]

        # Average across layers and seeds per step
        step_avg = combined.groupby("step")[available_metrics].mean().reset_index()
        step_avg = step_avg.sort_values("step")
        steps = step_avg["step"].values
        n_steps = len(steps)

        print(f"\n  Time points: {n_steps} (step range: {steps.min()}-{steps.max()})")

        # 3-phase analysis
        print(f"\n  {'Metric':<26} {'Early':>12} {'Mid':>12} {'Late':>12} {'Late/Early':>12}")
        print("  " + "-" * 70)

        for metric in available_metrics:
            phase_means = {}
            for phase_name, (lo, hi) in PHASES.items():
                mask = (step_avg["step"] >= lo) & (step_avg["step"] <= hi)
                phase_vals = step_avg.loc[mask, metric]
                phase_means[phase_name] = phase_vals.mean() if len(phase_vals) > 0 else np.nan

            ratio = (phase_means["late"] / phase_means["early"]
                     if abs(phase_means["early"]) > 1e-30 else np.nan)
            print(f"  {metric:<26} {fmt_sci(phase_means['early'])} "
                  f"{fmt_sci(phase_means['mid'])} {fmt_sci(phase_means['late'])} "
                  f"{ratio:>12.4f}" if not np.isnan(ratio) else
                  f"  {metric:<26} {fmt_sci(phase_means['early'])} "
                  f"{fmt_sci(phase_means['mid'])} {fmt_sci(phase_means['late'])} "
                  f"{'N/A':>12}")

        # Temporal stability: CV across time
        section_header("Temporal Stability (CV across time)", 2)
        print(f"  {'Metric':<26} {'Temporal CV':>12} {'Stability':>12}")
        print("  " + "-" * 55)
        for metric in available_metrics:
            vals = step_avg[metric].dropna().values
            if len(vals) > 1 and abs(vals.mean()) > 1e-15:
                tcv = vals.std(ddof=1) / abs(vals.mean())
            else:
                tcv = np.nan
            stability = ("STABLE" if tcv < 0.1 else
                         "MODERATE" if tcv < 0.3 else
                         "VOLATILE") if not np.isnan(tcv) else "N/A"
            tcv_str = f"{tcv:.4f}" if not np.isnan(tcv) else "N/A"
            print(f"  {metric:<26} {tcv_str:>12} {stability:>12}")

        # Trend detection: linear regression
        section_header("Trend Detection (Linear Regression)", 2)
        print(f"  {'Metric':<26} {'Slope':>14} {'R^2':>8} {'Trend':>15}")
        print("  " + "-" * 68)
        for metric in available_metrics:
            vals = step_avg[metric].dropna().values
            x = step_avg.loc[step_avg[metric].notna(), "step"].values
            if len(vals) < 3:
                continue
            if np.std(vals) < 1e-20:
                print(f"  {metric:<26} {'0':>14} {'1.000':>8} {'constant':>15}")
                continue
            slope, intercept, r_val, p_val, std_err = linregress(x, vals)
            r_sq = r_val ** 2
            if r_sq > 0.5 and abs(slope) > 1e-15:
                trend = "INCREASING" if slope > 0 else "DECREASING"
            else:
                trend = "no trend"
            print(f"  {metric:<26} {slope:>14.6g} {r_sq:>8.3f} {trend:>15}")

    try:
        plot_temporal_dynamics(store)
    except Exception as e:
        print(f"  [WARN] plot_temporal_dynamics failed: {e}")


# ── Analysis 5: Cross-Metric Correlation ──────────────────────────────────────

def analysis_5_correlation(store):
    """Compute Spearman correlation matrix across metrics."""
    section_header("ANALYSIS 5: Cross-Metric Correlation")

    # Pool all summary data (6 CSVs: 2 dw_min × 3 seeds)
    all_dfs = []
    for key, df in store.primary_summary.items():
        df_copy = df.copy()
        df_copy["dw_min"] = key[0]
        df_copy["seed"] = key[1]
        all_dfs.append(df_copy)

    if not all_dfs:
        print("  [SKIP] No primary summary data")
        return

    pooled = pd.concat(all_dfs, ignore_index=True)
    available = [m for m in CORR_METRICS if m in pooled.columns]
    n_rows = len(pooled)
    print(f"\n  Pooled {n_rows} rows ({len(all_dfs)} CSVs)")
    print(f"  Metrics: {len(available)}")

    # Compute Spearman correlation matrix
    corr_data = pooled[available].dropna(axis=1, how="all")
    available = list(corr_data.columns)
    n_metrics = len(available)

    corr_matrix = np.full((n_metrics, n_metrics), np.nan)
    for i in range(n_metrics):
        for j in range(i, n_metrics):
            x = corr_data[available[i]].values
            y = corr_data[available[j]].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3:
                continue
            if np.std(x[mask]) < 1e-15 or np.std(y[mask]) < 1e-15:
                corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                continue
            rho, _ = spearmanr(x[mask], y[mask])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # Find strong pairs (|rho| > 0.8)
    strong_pairs = []
    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            rho = corr_matrix[i, j]
            if not np.isnan(rho) and abs(rho) > 0.8:
                strong_pairs.append((available[i], available[j], rho))
    strong_pairs.sort(key=lambda x: -abs(x[2]))

    print(f"\n  Strong correlations (|rho| > 0.8): {len(strong_pairs)} pairs")
    print(f"\n  {'Metric A':<26} {'Metric B':<26} {'Spearman rho':>13}")
    print("  " + "-" * 68)
    for a, b, rho in strong_pairs[:20]:
        marker = " (+)" if rho > 0 else " (-)"
        print(f"  {a:<26} {b:<26} {rho:>12.4f}{marker}")

    # Key questions
    section_header("Key Diagnostic Questions", 2)

    # Q1: dw_zero_ratio ↔ grad_deadzone_ratio redundancy
    if "dw_zero_ratio" in available and "grad_deadzone_ratio" in available:
        i1 = available.index("dw_zero_ratio")
        i2 = available.index("grad_deadzone_ratio")
        rho = corr_matrix[i1, i2]
        redundant = abs(rho) > 0.9 if not np.isnan(rho) else False
        print(f"    dw_zero_ratio ↔ grad_deadzone_ratio: rho={rho:.4f} "
              f"→ {'REDUNDANT' if redundant else 'INDEPENDENT'}")

    # Q2: transfer_efficiency ↔ update_vs_grad_cosine
    if "transfer_efficiency" in available and "update_vs_grad_cosine" in available:
        i1 = available.index("transfer_efficiency")
        i2 = available.index("update_vs_grad_cosine")
        rho = corr_matrix[i1, i2]
        associated = abs(rho) > 0.5 if not np.isnan(rho) else False
        print(f"    transfer_efficiency ↔ update_vs_grad_cosine: rho={rho:.4f} "
              f"→ {'ASSOCIATED' if associated else 'INDEPENDENT'}")

    # Q3: BL_mean ↔ pulse_ok_frac
    if "BL_mean" in available and "pulse_ok_frac" in available:
        i1 = available.index("BL_mean")
        i2 = available.index("pulse_ok_frac")
        rho = corr_matrix[i1, i2]
        print(f"    BL_mean ↔ pulse_ok_frac: rho={rho:.4f}")

    # Identify metric clusters
    section_header("Metric Clusters (connected by |rho| > 0.8)", 2)
    # Simple union-find clustering
    parent = list(range(n_metrics))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b, rho in strong_pairs:
        i1 = available.index(a)
        i2 = available.index(b)
        union(i1, i2)

    clusters = {}
    for i in range(n_metrics):
        root = find(i)
        clusters.setdefault(root, []).append(available[i])

    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    for idx, (root, members) in enumerate(multi_clusters.items(), 1):
        print(f"    Cluster {idx}: {', '.join(members)}")

    try:
        plot_correlation_matrix(corr_matrix, available)
    except Exception as e:
        print(f"  [WARN] plot_correlation_matrix failed: {e}")


# ── Analysis 6: TikiTaka vs SingleRPU Mode Comparison ─────────────────────────

def analysis_6_mode_comparison(store):
    """Compare v3_single vs v3_tiki on shared metrics."""
    section_header("ANALYSIS 6: TikiTaka vs SingleRPU Mode Comparison")

    if "single" not in store.mode_summary or "tiki" not in store.mode_summary:
        missing = [m for m in ["single", "tiki"] if m not in store.mode_summary]
        print(f"  [SKIP] Missing mode data: {missing}")
        return

    df_single = store.mode_summary["single"]
    df_tiki = store.mode_summary["tiki"]
    print(f"\n  SingleRPU: {len(df_single)} rows, TikiTaka: {len(df_tiki)} rows")
    print(f"  NOTE: v3 data has only 5 training steps → early-training comparison only\n")

    # Find shared numeric columns
    single_mcols = set(get_metric_cols(df_single))
    tiki_mcols = set(get_metric_cols(df_tiki))
    shared = sorted(single_mcols & tiki_mcols)

    # Shared metrics that are meaningful for both modes
    compare_metrics = [m for m in [
        "dw_zero_ratio", "dw_absmean", "dw_1lsb_ratio",
        "grad_absmean", "grad_deadzone_ratio",
        "update_vs_grad_cosine", "eff_lr_slope",
        "BL_mean", "BL_hit_ratio",
        "pulse_ok_frac", "pulse_under_frac",
        "pulse_sat_ratio", "bound_sat_ratio",
        "sign_mismatch_ratio", "rel_update_error",
    ] if m in shared]

    print(f"  {'Metric':<26} {'Single':>12} {'TikiTaka':>12} {'Tiki/Single':>12} {'Winner':>8}")
    print("  " + "-" * 75)

    for metric in compare_metrics:
        m_single = df_single[metric].mean()
        m_tiki = df_tiki[metric].mean()
        ratio = m_tiki / m_single if abs(m_single) > 1e-30 else np.inf

        # Determine which is "better" based on metric semantics
        lower_better = metric in ["dw_zero_ratio", "grad_deadzone_ratio",
                                   "sign_mismatch_ratio", "rel_update_error",
                                   "pulse_sat_ratio", "bound_sat_ratio",
                                   "pulse_under_frac"]
        if abs(ratio - 1.0) < 0.05:
            winner = "TIE"
        elif lower_better:
            winner = "Single" if m_single < m_tiki else "Tiki"
        else:
            winner = "Tiki" if m_tiki > m_single else "Single"

        print(f"  {metric:<26} {fmt_sci(m_single)} {fmt_sci(m_tiki)} "
              f"{ratio:>12.4f} {winner:>8}")

    # TikiTaka-only metrics
    tiki_only = sorted(tiki_mcols - single_mcols)
    tiki_only_interesting = [m for m in TIKI_METRICS if m in tiki_only]
    if tiki_only_interesting:
        section_header("TikiTaka-Only Metrics", 2)
        print(f"  {'Metric':<28} {'Mean':>14} {'Std':>12}")
        print("  " + "-" * 58)
        for metric in tiki_only_interesting:
            vals = df_tiki[metric].dropna()
            if len(vals) == 0:
                continue
            print(f"  {metric:<28} {fmt_sci(vals.mean())} {fmt_sci(vals.std())}")

    # Sublayer-wise comparison (v3 includes FFN1/FFN2)
    sublayers_v3 = sorted(df_single["sublayer"].unique())
    section_header(f"Sublayer Breakdown (sublayers: {sublayers_v3})", 2)
    for metric in ["dw_zero_ratio", "dw_absmean", "update_vs_grad_cosine"]:
        if metric not in shared:
            continue
        print(f"\n  {metric}:")
        print(f"  {'Sublayer':<8} {'Single':>12} {'TikiTaka':>12} {'Ratio':>10}")
        print("  " + "-" * 45)
        for sub in sublayers_v3:
            ms = df_single[df_single["sublayer"] == sub][metric].mean()
            mt = df_tiki[df_tiki["sublayer"] == sub][metric].mean()
            r = mt / ms if abs(ms) > 1e-30 else np.nan
            r_str = f"{r:.4f}" if not np.isnan(r) else "N/A"
            print(f"  {sub:<8} {fmt_sci(ms)} {fmt_sci(mt)} {r_str:>10}")


# ── Analysis 7: Actionable Recommendations ────────────────────────────────────

def analysis_7_recommendations(store):
    """Synthesize all findings into prioritized recommendations."""
    section_header("ANALYSIS 7: Actionable Recommendations")

    dw_vals = sorted(DW_CONFIGS.keys())
    recommendations = []

    # Gather key data points for each dw_min
    for dw_min in dw_vals:
        avg_df = store.get_seed_averaged_summary(dw_min)
        if avg_df is None:
            continue

        info = {"dw_min": dw_min}
        for metric in ["dw_zero_ratio", "dw_absmean", "pulse_ok_frac", "BL_mean",
                        "update_vs_grad_cosine", "sign_mismatch_ratio",
                        "transfer_efficiency", "dw_fast_absmean", "dw_slow_absmean",
                        "grad_absmean", "grad_deadzone_ratio", "pulse_under_frac"]:
            if metric in avg_df.columns:
                info[metric] = avg_df[metric].mean()

        # Transfer loss
        if "dw_fast_absmean" in info and "dw_slow_absmean" in info:
            fast = info["dw_fast_absmean"]
            slow = info["dw_slow_absmean"]
            info["transfer_loss_pct"] = (1 - slow / fast) * 100 if fast > 1e-30 else 100.0

        # Health counts
        health_counts = {"GOOD": 0, "WARN": 0, "BAD": 0}
        for _, row in avg_df.iterrows():
            for metric in HEALTH_THRESHOLDS:
                if metric in row.index:
                    status = health_classify(metric, row[metric])
                    if status in health_counts:
                        health_counts[status] += 1
        info["health"] = health_counts
        recommendations.append(info)

    # Print diagnostic overview
    section_header("Diagnostic Overview", 2)
    for info in recommendations:
        dw = info["dw_min"]
        h = info["health"]
        total = h["GOOD"] + h["WARN"] + h["BAD"]
        print(f"\n  dw_min = {dw}:")
        print(f"    Health: {h['GOOD']} GOOD, {h['WARN']} WARN, {h['BAD']} BAD (of {total})")
        for key in ["dw_zero_ratio", "dw_absmean", "pulse_ok_frac", "BL_mean",
                     "update_vs_grad_cosine", "transfer_efficiency"]:
            if key in info:
                status = health_classify(key, info[key])
                print(f"    {key:<28} = {info[key]:.6g}  [{status}]")
        if "transfer_loss_pct" in info:
            print(f"    {'transfer_loss_pct':<28} = {info['transfer_loss_pct']:.4f}%")

    # Priority 1: Critical Issues
    section_header("Priority 1: CRITICAL", 2)
    p1_issues = []

    for info in recommendations:
        dw = info["dw_min"]

        # Pulse underutilization check
        if "pulse_ok_frac" in info and info["pulse_ok_frac"] < 0.2:
            bl_val = info.get("BL_mean", "?")
            p1_issues.append(
                f"[dw={dw}] Severe pulse underutilization: pulse_ok_frac={info['pulse_ok_frac']:.4f}, "
                f"BL_mean={bl_val:.2f}. "
                f"Most updates are sub-threshold. Consider increasing desired_bl or adjusting dw_min."
            )

        # Complete signal loss
        if "dw_zero_ratio" in info and info["dw_zero_ratio"] > 0.99:
            p1_issues.append(
                f"[dw={dw}] Near-total update failure: dw_zero_ratio={info['dw_zero_ratio']:.4f}. "
                f"Effective learning is blocked."
            )

        # Transfer pipeline loss
        if "transfer_loss_pct" in info and info["transfer_loss_pct"] > 99:
            p1_issues.append(
                f"[dw={dw}] Transfer pipeline loss={info['transfer_loss_pct']:.2f}%. "
                f"Slow tile receives almost no signal from fast tile."
            )

    if p1_issues:
        for i, issue in enumerate(p1_issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("    No critical issues detected.")

    # Priority 2: Important
    section_header("Priority 2: IMPORTANT", 2)
    p2_issues = []

    for info in recommendations:
        dw = info["dw_min"]

        # Gradient alignment
        if "update_vs_grad_cosine" in info and info["update_vs_grad_cosine"] < 0.05:
            p2_issues.append(
                f"[dw={dw}] Weak gradient alignment: cosine={info['update_vs_grad_cosine']:.6f}. "
                f"Updates are nearly orthogonal to gradient direction."
            )

        # High sign mismatch
        if "sign_mismatch_ratio" in info and info["sign_mismatch_ratio"] > 0.90:
            p2_issues.append(
                f"[dw={dw}] High sign mismatch: {info['sign_mismatch_ratio']:.4f}. "
                f"Most updates have wrong sign relative to gradient."
            )

        # Pulse range issues
        if "pulse_ok_frac" in info and 0.2 <= info["pulse_ok_frac"] < 0.5:
            p2_issues.append(
                f"[dw={dw}] Moderate pulse underutilization: pulse_ok_frac={info['pulse_ok_frac']:.4f}. "
                f"Less than half of updates are in the effective pulse range."
            )

    if p2_issues:
        for i, issue in enumerate(p2_issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("    No important issues detected.")

    # Priority 3: Monitor
    section_header("Priority 3: MONITOR", 2)
    p3_issues = []

    for info in recommendations:
        dw = info["dw_min"]

        # Moderate deadzone
        if "grad_deadzone_ratio" in info and info["grad_deadzone_ratio"] > 0.5:
            p3_issues.append(
                f"[dw={dw}] High gradient deadzone: {info['grad_deadzone_ratio']:.4f} of gradients "
                f"fall below quantization threshold."
            )

        # Moderate dw_zero_ratio
        if "dw_zero_ratio" in info and 0.95 <= info["dw_zero_ratio"] < 0.99:
            p3_issues.append(
                f"[dw={dw}] Elevated zero-update ratio: {info['dw_zero_ratio']:.4f}. "
                f"Many weight updates are quantized to zero."
            )

    if p3_issues:
        for i, issue in enumerate(p3_issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("    No monitoring items.")

    # dw_min Selection Guide
    section_header("dw_min Selection Guide", 2)
    if len(recommendations) >= 2:
        r0, r1 = recommendations[0], recommendations[1]
        dw0, dw1 = r0["dw_min"], r1["dw_min"]
        h0, h1 = r0["health"], r1["health"]

        print(f"\n  Comparison: dw_min={dw0} vs dw_min={dw1}")
        print(f"    Health (BAD count): {h0['BAD']} vs {h1['BAD']}")
        print(f"    Health (GOOD count): {h0['GOOD']} vs {h1['GOOD']}")

        # Determine recommendation
        if h0["BAD"] < h1["BAD"]:
            better = dw0
        elif h1["BAD"] < h0["BAD"]:
            better = dw1
        elif h0["GOOD"] > h1["GOOD"]:
            better = dw0
        else:
            better = dw1

        print(f"\n  >>> Recommended dw_min = {better} <<<")

        # Rationale
        for info in recommendations:
            dw = info["dw_min"]
            pros, cons = [], []
            if info.get("pulse_ok_frac", 0) > 0.3:
                pros.append(f"pulse_ok_frac={info['pulse_ok_frac']:.3f}")
            else:
                cons.append(f"pulse_ok_frac={info.get('pulse_ok_frac', 0):.3f}")
            if info.get("dw_zero_ratio", 1) < 0.95:
                pros.append(f"dw_zero_ratio={info['dw_zero_ratio']:.3f}")
            else:
                cons.append(f"dw_zero_ratio={info.get('dw_zero_ratio', 1):.3f}")
            if info.get("update_vs_grad_cosine", 0) > 0.05:
                pros.append(f"cosine={info['update_vs_grad_cosine']:.4f}")
            else:
                cons.append(f"cosine={info.get('update_vs_grad_cosine', 0):.4f}")

            print(f"\n    dw_min={dw}:")
            if pros:
                print(f"      Pros: {', '.join(pros)}")
            if cons:
                print(f"      Cons: {', '.join(cons)}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  Deep Analysis: TikiTaka Weight Update Diagnostics")
    print("=" * 90)

    os.makedirs(FIG_DIR, exist_ok=True)

    store = DataStore()
    store.load_primary_tiki()
    store.load_mode_compare()
    store.load_multiseed()
    store.report_inventory()

    # Verify minimum data
    if len(store.primary_summary) < 4:
        print("[ERROR] Need at least 2 seeds per dw_min for primary data. Aborting.")
        sys.exit(1)

    # Run all 7 analyses
    analysis_1_dose_response(store)
    analysis_2_bottleneck_map(store)
    analysis_3_transfer_dynamics(store)
    analysis_4_temporal_dynamics(store)
    analysis_5_correlation(store)
    analysis_6_mode_comparison(store)
    analysis_7_recommendations(store)

    print("=" * 90)
    print("  Deep Analysis Complete")
    print("=" * 90)

    if _SAVED_FIGURES:
        print(f"\n  Saved {len(_SAVED_FIGURES)} figures:")
        for fig_path in _SAVED_FIGURES:
            print(f"    - {fig_path}")
    else:
        print("\n  No figures were saved.")


if __name__ == "__main__":
    main()

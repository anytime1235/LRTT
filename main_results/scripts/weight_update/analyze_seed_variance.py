#!/usr/bin/env python3
"""Seed Variance Analysis for TikiTaka Weight Update Diagnostics.

Quantifies seed-to-seed variability across dw_min sweep experiments
to determine whether a single seed is sufficient for reliable conclusions.

Usage:
    python analyze_seed_variance.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, pearsonr, spearmanr
from itertools import combinations

# ── Configuration ──────────────────────────────────────────────────────────────

BASE = "/data/main_results/weight_update/squad/tiki"

DW_CONFIGS = {
    0.0005: {
        "run": "run_f2fad8e3f307",
        "prefix": "dw0p0005",
    },
    0.005: {
        "run": "run_4249343b8006",
        "prefix": "dw0p0050",
    },
}

SEEDS = [0, 1, 2]

# Key metrics for effect-size analysis (subset of all 44 numeric cols)
KEY_METRICS = [
    "dw_zero_ratio",
    "dw_absmean",
    "grad_absmean",
    "grad_deadzone_ratio",
    "update_vs_grad_cosine",
    "eff_lr_slope",
    "BL_mean",
    "pulse_ok_frac",
    "sign_mismatch_ratio",
    "rel_update_error",
]

CV_THRESHOLD = 0.10       # CV < 0.10 → seed variance negligible
VAR_RATIO_THRESHOLD = 10  # var_ratio > 10 → dw_min dominates


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_summary_csvs():
    """Load all 6 summary CSVs into a dict keyed by (dw_min, seed)."""
    data = {}
    for dw_min, cfg in DW_CONFIGS.items():
        for seed in SEEDS:
            tag = f"{cfg['prefix']}_seed{seed}"
            csv_path = os.path.join(
                BASE, cfg["run"], tag, f"{tag}_summary.csv"
            )
            if not os.path.exists(csv_path):
                print(f"  [WARNING] Missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            data[(dw_min, seed)] = df
    return data


def get_metric_cols(df):
    """Return numeric metric columns (exclude index/config cols)."""
    exclude = {"layer_idx", "sublayer", "trace_every", "is_transfer_step"}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in (np.float64, np.float32, np.int64)]


# ── Analysis 1: Per-Metric Global Seed Variance ──────────────────────────────

def analyze_per_metric_variance(data):
    """For each (dw_min, metric): compute mean, std, CV, 95% CI across seeds."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 1: Per-Metric Global Seed Variance")
    print("=" * 80)

    sample_df = next(iter(data.values()))
    metric_cols = get_metric_cols(sample_df)

    results = []
    for dw_min in DW_CONFIGS:
        for metric in metric_cols:
            seed_means = []
            for seed in SEEDS:
                key = (dw_min, seed)
                if key not in data:
                    continue
                val = data[key][metric].mean()
                seed_means.append(val)
            if len(seed_means) < 2:
                continue
            arr = np.array(seed_means)
            m = arr.mean()
            sd = arr.std(ddof=1)
            cv = sd / abs(m) if abs(m) > 1e-15 else np.nan
            n = len(arr)
            t_val = t_dist.ppf(0.975, n - 1)
            ci_half = t_val * sd / np.sqrt(n)
            results.append({
                "dw_min": dw_min,
                "metric": metric,
                "mean": m,
                "std": sd,
                "CV": cv,
                "ci95_lo": m - ci_half,
                "ci95_hi": m + ci_half,
            })

    df_res = pd.DataFrame(results)

    # Print summary for key metrics
    key_res = df_res[df_res["metric"].isin(KEY_METRICS)].copy()
    key_res = key_res.sort_values(["metric", "dw_min"])

    print(f"\n{'Metric':<28} {'dw_min':>8} {'Mean':>12} {'Std':>12} {'CV':>8} {'95% CI':>26}")
    print("-" * 96)
    for _, row in key_res.iterrows():
        ci_str = f"[{row['ci95_lo']:.6g}, {row['ci95_hi']:.6g}]"
        print(f"{row['metric']:<28} {row['dw_min']:>8.4f} {row['mean']:>12.6g} "
              f"{row['std']:>12.4g} {row['CV']:>8.4f} {ci_str:>26}")

    # Summary stats
    print(f"\n  All metrics: max CV = {df_res['CV'].max():.4f}, "
          f"median CV = {df_res['CV'].median():.4f}")
    high_cv = df_res[df_res["CV"] > CV_THRESHOLD]
    if len(high_cv) > 0:
        print(f"  Metrics with CV > {CV_THRESHOLD}:")
        for _, row in high_cv.sort_values("CV", ascending=False).head(10).iterrows():
            print(f"    {row['metric']:<28} dw_min={row['dw_min']:.4f}  CV={row['CV']:.4f}")
    else:
        print(f"  All metrics have CV < {CV_THRESHOLD}")

    return df_res


# ── Analysis 2: Per-Layer Variance ────────────────────────────────────────────

def analyze_per_layer_variance(data):
    """For each (dw_min, layer, sublayer, metric): compute CV across seeds."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 2: Per-Layer Seed Variance")
    print("=" * 80)

    sample_df = next(iter(data.values()))
    metric_cols = get_metric_cols(sample_df)

    all_cvs = []
    for dw_min in DW_CONFIGS:
        # Build per-seed DataFrames indexed by (layer_idx, sublayer)
        seed_dfs = {}
        for seed in SEEDS:
            key = (dw_min, seed)
            if key not in data:
                continue
            df = data[key].set_index(["layer_idx", "sublayer"])
            seed_dfs[seed] = df

        if len(seed_dfs) < 2:
            continue

        # For each (layer, sublayer, metric)
        common_idx = seed_dfs[SEEDS[0]].index
        for lidx, sub in common_idx:
            for metric in metric_cols:
                vals = []
                for seed in SEEDS:
                    if seed in seed_dfs:
                        vals.append(seed_dfs[seed].loc[(lidx, sub), metric])
                arr = np.array(vals, dtype=float)
                m = arr.mean()
                sd = arr.std(ddof=1)
                cv = sd / abs(m) if abs(m) > 1e-15 else np.nan
                all_cvs.append({
                    "dw_min": dw_min,
                    "layer_idx": lidx,
                    "sublayer": sub,
                    "metric": metric,
                    "mean": m,
                    "std": sd,
                    "CV": cv,
                })

    df_cv = pd.DataFrame(all_cvs)
    df_cv_valid = df_cv.dropna(subset=["CV"])

    # Report median and max CV per metric
    summary = df_cv_valid.groupby("metric")["CV"].agg(["median", "max"]).reset_index()
    summary = summary.sort_values("max", ascending=False)

    print(f"\n{'Metric':<28} {'Median CV':>10} {'Max CV':>10}")
    print("-" * 50)
    for _, row in summary.head(15).iterrows():
        flag = " *** " if row["max"] > CV_THRESHOLD else ""
        print(f"{row['metric']:<28} {row['median']:>10.4f} {row['max']:>10.4f}{flag}")

    # Identify most sensitive layers
    top_layers = df_cv_valid.nlargest(10, "CV")[
        ["dw_min", "layer_idx", "sublayer", "metric", "CV"]
    ]
    print(f"\n  Top 10 most seed-sensitive (layer, sublayer, metric) combinations:")
    for _, row in top_layers.iterrows():
        print(f"    dw_min={row['dw_min']:.4f}  L{int(row['layer_idx']):02d}/{row['sublayer']:<4s} "
              f"{row['metric']:<28s} CV={row['CV']:.4f}")

    overall_max_cv = df_cv_valid["CV"].max()
    overall_med_cv = df_cv_valid["CV"].median()
    print(f"\n  Overall: median CV = {overall_med_cv:.4f}, max CV = {overall_max_cv:.4f}")

    return df_cv, overall_max_cv


# ── Analysis 3: Effect Size — dw_min vs Seed ──────────────────────────────────

def analyze_effect_size(data):
    """Compute var_ratio = var(dw_min means) / avg(within-dw_min seed variance).

    High ratio → dw_min effect dominates seed noise → single seed sufficient.
    """
    print("\n" + "=" * 80)
    print("  ANALYSIS 3: Effect Size — dw_min vs Seed Variance")
    print("=" * 80)

    sample_df = next(iter(data.values()))
    metric_cols = get_metric_cols(sample_df)

    results = []
    for metric in metric_cols:
        # Per-dw_min: compute mean across seeds and within-dw_min variance
        dw_means = []
        within_vars = []
        for dw_min in DW_CONFIGS:
            seed_vals = []
            for seed in SEEDS:
                key = (dw_min, seed)
                if key not in data:
                    continue
                seed_vals.append(data[key][metric].mean())
            if len(seed_vals) < 2:
                continue
            arr = np.array(seed_vals)
            dw_means.append(arr.mean())
            within_vars.append(arr.var(ddof=1))

        if len(dw_means) < 2 or len(within_vars) < 2:
            continue

        var_between = np.var(dw_means, ddof=1)
        avg_var_within = np.mean(within_vars)

        if avg_var_within > 1e-30:
            var_ratio = var_between / avg_var_within
        else:
            var_ratio = np.inf if var_between > 1e-30 else np.nan

        results.append({
            "metric": metric,
            "dw_mean_0.0005": dw_means[0],
            "dw_mean_0.005": dw_means[1],
            "var_between_dw": var_between,
            "avg_var_within_seed": avg_var_within,
            "var_ratio": var_ratio,
        })

    df_eff = pd.DataFrame(results)

    # Print key metrics
    key_eff = df_eff[df_eff["metric"].isin(KEY_METRICS)].copy()
    key_eff = key_eff.sort_values("var_ratio", ascending=False)

    print(f"\n{'Metric':<28} {'dw=0.0005':>12} {'dw=0.005':>12} "
          f"{'Var(dw)':>12} {'Var(seed)':>12} {'Ratio':>10} {'Verdict':>10}")
    print("-" * 100)
    for _, row in key_eff.iterrows():
        ratio = row["var_ratio"]
        if np.isinf(ratio):
            verdict = ">>10"
            ratio_str = "inf"
        elif np.isnan(ratio):
            verdict = "N/A"
            ratio_str = "NaN"
        elif ratio > VAR_RATIO_THRESHOLD:
            verdict = "OK"
            ratio_str = f"{ratio:.1f}"
        else:
            verdict = "WARN"
            ratio_str = f"{ratio:.1f}"
        print(f"{row['metric']:<28} {row['dw_mean_0.0005']:>12.6g} {row['dw_mean_0.005']:>12.6g} "
              f"{row['var_between_dw']:>12.4g} {row['avg_var_within_seed']:>12.4g} "
              f"{ratio_str:>10} {verdict:>10}")

    # Summary
    key_ratios = key_eff["var_ratio"].replace([np.inf], np.nan).dropna()
    min_ratio = key_eff["var_ratio"].min()
    all_above = all(r > VAR_RATIO_THRESHOLD for r in key_eff["var_ratio"] if not np.isnan(r))
    print(f"\n  Key metrics: min var_ratio = {min_ratio:.1f}, "
          f"all > {VAR_RATIO_THRESHOLD}? {all_above}")

    return df_eff, all_above


# ── Analysis 4: Seed-Pair Layer-Profile Correlation ───────────────────────────

def analyze_seed_correlation(data):
    """Across 48 layer-sublayer profiles, compute seed-pair Pearson & Spearman."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 4: Seed-Pair Layer-Profile Correlation")
    print("=" * 80)

    results = []
    for dw_min in DW_CONFIGS:
        for metric in KEY_METRICS:
            for s1, s2 in combinations(SEEDS, 2):
                k1, k2 = (dw_min, s1), (dw_min, s2)
                if k1 not in data or k2 not in data:
                    continue
                v1 = data[k1].sort_values(["layer_idx", "sublayer"])[metric].values
                v2 = data[k2].sort_values(["layer_idx", "sublayer"])[metric].values

                # Skip constant arrays
                if np.std(v1) < 1e-15 or np.std(v2) < 1e-15:
                    results.append({
                        "dw_min": dw_min, "metric": metric,
                        "seed_pair": f"{s1}-{s2}",
                        "pearson_r": np.nan, "spearman_rho": np.nan,
                    })
                    continue

                pr, _ = pearsonr(v1, v2)
                sr, _ = spearmanr(v1, v2)
                results.append({
                    "dw_min": dw_min, "metric": metric,
                    "seed_pair": f"{s1}-{s2}",
                    "pearson_r": pr, "spearman_rho": sr,
                })

    df_corr = pd.DataFrame(results)
    df_valid = df_corr.dropna(subset=["spearman_rho"])

    # Summary per (dw_min, metric)
    summary = df_valid.groupby(["dw_min", "metric"]).agg(
        pearson_min=("pearson_r", "min"),
        spearman_min=("spearman_rho", "min"),
    ).reset_index()

    print(f"\n{'Metric':<28} {'dw_min':>8} {'Pearson min':>12} {'Spearman min':>13}")
    print("-" * 65)
    for _, row in summary.sort_values(["metric", "dw_min"]).iterrows():
        flag = " ***" if row["spearman_min"] < 0.95 else ""
        print(f"{row['metric']:<28} {row['dw_min']:>8.4f} "
              f"{row['pearson_min']:>12.4f} {row['spearman_min']:>13.4f}{flag}")

    overall_min_spearman = df_valid["spearman_rho"].min()
    overall_min_pearson = df_valid["pearson_r"].min()
    print(f"\n  Overall min Pearson r = {overall_min_pearson:.4f}, "
          f"min Spearman rho = {overall_min_spearman:.4f}")
    high_corr = overall_min_spearman > 0.95
    print(f"  All rho > 0.95? {high_corr}")

    return df_corr, high_corr


# ── Final Verdict ─────────────────────────────────────────────────────────────

def print_verdict(max_cv, all_ratio_above, high_corr, df_effect):
    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    key_eff = df_effect[df_effect["metric"].isin(KEY_METRICS)]
    n_ok = (key_eff["var_ratio"] > VAR_RATIO_THRESHOLD).sum()
    n_total = len(key_eff)
    frac_ok = n_ok / n_total if n_total > 0 else 0

    # Metrics where dw_min has negligible effect (both dw_min give ~same value)
    # — low var_ratio here is expected, not a sign of seed instability
    low_ratio_metrics = key_eff[key_eff["var_ratio"] <= VAR_RATIO_THRESHOLD]
    metrics_with_no_dw_effect = []
    for _, row in low_ratio_metrics.iterrows():
        m0, m1 = row["dw_mean_0.0005"], row["dw_mean_0.005"]
        rel_diff = abs(m0 - m1) / max(abs(m0), abs(m1), 1e-15)
        if rel_diff < 0.20:
            metrics_with_no_dw_effect.append(row["metric"])

    # Discriminating metrics: those where dw_min actually changes the value
    discriminating = key_eff[~key_eff["metric"].isin(metrics_with_no_dw_effect)]
    all_discrim_above = all(r > VAR_RATIO_THRESHOLD
                           for r in discriminating["var_ratio"]
                           if not np.isnan(r))

    print(f"\n  Criteria:")
    print(f"    1. Key metrics with var_ratio > {VAR_RATIO_THRESHOLD}: "
          f"{n_ok}/{n_total}")
    print(f"    2. Max per-layer CV < {CV_THRESHOLD}?  "
          f"{'YES' if max_cv < CV_THRESHOLD else 'NO'} (max={max_cv:.4f})")
    print(f"    3. All seed-pair Spearman rho > 0.95?  "
          f"{'YES' if high_corr else 'NO'}")

    if metrics_with_no_dw_effect:
        print(f"\n  Note: {len(metrics_with_no_dw_effect)} metric(s) show negligible dw_min effect "
              f"(both dw_min values are similar):")
        for m in metrics_with_no_dw_effect:
            print(f"    - {m}")
        print(f"  → Low var_ratio for these is expected (no signal to detect), not seed instability.")
        print(f"  Discriminating metrics (where dw_min matters): "
              f"all var_ratio > {VAR_RATIO_THRESHOLD}? {all_discrim_above}")

    if all_discrim_above and max_cv < CV_THRESHOLD:
        verdict = "SINGLE SEED SUFFICIENT"
        detail = ("dw_min effect vastly dominates seed noise for all discriminating metrics.\n"
                  "    Layer-level variance is minimal. Single seed results are reliable.")
    elif all_discrim_above or frac_ok >= 0.6:
        verdict = "SINGLE SEED SUFFICIENT FOR DIRECTIONAL CONCLUSIONS"
        detail = ("dw_min effect dominates seed noise for key discriminating metrics.\n"
                  "    Some per-layer CVs are high for metrics where both dw_min values are similar.\n"
                  "    → Single seed is reliable for sweep direction decisions;\n"
                  "      use 3 seeds only if precise layer-level ranking of noisy metrics is needed.")
    else:
        verdict = "THREE SEEDS RECOMMENDED"
        detail = ("Seed variance is non-negligible for multiple discriminating metrics.\n"
                  "    Multiple seeds needed for reliable conclusions.")

    print(f"\n  >>> {verdict} <<<")
    print(f"    {detail}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Seed Variance Analysis — TikiTaka Weight Update Diagnostics")
    print(f"  dw_min values: {list(DW_CONFIGS.keys())}")
    print(f"  Seeds: {SEEDS}")

    data = load_summary_csvs()
    print(f"  Loaded {len(data)} summary CSVs "
          f"({len(data[(list(DW_CONFIGS.keys())[0], SEEDS[0])]) if data else 0} rows each)")

    if len(data) < 4:
        print("[ERROR] Need at least 2 seeds per dw_min. Aborting.")
        sys.exit(1)

    # Analysis 1
    df_global_cv = analyze_per_metric_variance(data)

    # Analysis 2
    df_layer_cv, max_cv = analyze_per_layer_variance(data)

    # Analysis 3
    df_effect, all_ratio_above = analyze_effect_size(data)

    # Analysis 4
    df_corr, high_corr = analyze_seed_correlation(data)

    # Verdict
    print_verdict(max_cv, all_ratio_above, high_corr, df_effect)


if __name__ == "__main__":
    main()

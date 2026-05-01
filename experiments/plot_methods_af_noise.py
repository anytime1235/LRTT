#!/usr/bin/env python3
"""Plot AF / noise sensitivity comparison.

Reads results/methods_af_noise/{af,noise,anchor}_results.json (falls back to
*_partial.json) and produces:
  - figures/af_sensitivity.png    (x = gamma_up=gamma_down, y = best val acc)
  - figures/noise_sensitivity.png (x = noise_ratio,         y = best val acc)

Sweep methods (lrtt_v1, lrtt_v2) plotted as curves with mean +/- std shaded
bands across runs. Anchor methods (direct, tikitaka_v1) plotted as horizontal
reference lines (mean across anchor runs) with a thin shaded band for std.

Usage:
  python experiments/plot_methods_af_noise.py
  python experiments/plot_methods_af_noise.py --input_dir results/methods_af_noise_smoke
"""

import os, json, argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABELS = {
    "direct":      "Direct (Single RPU)",
    "tikitaka_v1": "TikiTaka v1",
    "lrtt_v1":     "LR-TT v1",
    "lrtt_v2":     "LR-TT v2",
}
METHOD_COLORS = {
    "direct":      "#888888",
    "tikitaka_v1": "#1f77b4",
    "lrtt_v1":     "#d62728",
    "lrtt_v2":     "#2ca02c",
}
SWEEP_METHODS = ["lrtt_v1", "lrtt_v2"]
ANCHOR_METHODS = ["direct", "tikitaka_v1"]
METHOD_ORDER = SWEEP_METHODS + ANCHOR_METHODS
AXIS_XLABEL = {
    "af":    r"AF: $\gamma_{up}=\gamma_{down}$",
    "noise": "Noise ratio (relative to 6T1C noise template)",
}
AXIS_TITLE = {
    "af":    "C-tile AF sensitivity (bits=8, lifetime=1000)",
    "noise": "C-tile noise sensitivity (bits=8, lifetime=1000)",
}


def aggregate(results, axis):
    """{method: (sorted_levels, mean_per_level, std_per_level, n_per_level)}"""
    by_method = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("axis", axis) != axis:
            continue
        by_method[r["method"]][r["level"]].append(r["best_acc"])
    out = {}
    for method, level_map in by_method.items():
        levels = sorted(level_map.keys())
        means = np.array([np.mean(level_map[L]) for L in levels])
        stds = np.array([np.std(level_map[L], ddof=1) if len(level_map[L]) > 1 else 0.0
                         for L in levels])
        ns = np.array([len(level_map[L]) for L in levels])
        out[method] = (np.array(levels), means, stds, ns)
    return out


def aggregate_anchor(anchor_results):
    """{method: (mean, std, n)} from anchor runs (single cell)."""
    by_method = defaultdict(list)
    for r in anchor_results:
        by_method[r["method"]].append(r["best_acc"])
    out = {}
    for method, accs in by_method.items():
        m = float(np.mean(accs))
        s = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        out[method] = (m, s, len(accs))
    return out


def plot_axis(sweep_results, anchor_agg, axis, fig_path):
    agg = aggregate(sweep_results, axis)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for method in SWEEP_METHODS:
        if method not in agg:
            continue
        levels, means, stds, ns = agg[method]
        color = METHOD_COLORS[method]
        ax.plot(levels, means, marker="o", lw=2.0, color=color,
                label=f"{METHOD_LABELS[method]} (n={int(ns.max())}/cell)")
        ax.fill_between(levels, means - stds, means + stds, alpha=0.15, color=color)

    sweep_levels = sorted({float(L) for m in agg.values() for L in m[0]})
    if sweep_levels:
        x_min, x_max = min(sweep_levels), max(sweep_levels)
    else:
        x_min, x_max = 0.0, 1.0

    for method in ANCHOR_METHODS:
        if method not in anchor_agg:
            continue
        mean, std, n = anchor_agg[method]
        color = METHOD_COLORS[method]
        ax.axhline(mean, color=color, lw=1.6, ls="--",
                   label=f"{METHOD_LABELS[method]} anchor "
                         f"(γ=0, n_ratio=0; n={n})")
        if std > 0:
            ax.fill_between([x_min, x_max], [mean - std] * 2, [mean + std] * 2,
                            color=color, alpha=0.10)

    if axis == "af":
        ax.set_xticks(sweep_levels)
    else:
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xticks(sweep_levels)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(AXIS_XLABEL[axis])
    ax.set_ylabel("Best val accuracy (%)")
    ax.set_title(AXIS_TITLE[axis])
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"  -> {fig_path}")


def print_summary(sweep_results, anchor_agg, axis):
    agg = aggregate(sweep_results, axis)
    print(f"\n[{axis}] sweep summary (mean +/- std over runs)")
    all_levels = sorted({float(L) for m in agg.values() for L in m[0]})
    print(f"  {'method':<12}  " + "  ".join(f"{L:>7}" for L in all_levels))
    for method in SWEEP_METHODS:
        if method not in agg:
            continue
        levels, means, stds, _ = agg[method]
        cells = []
        for L in all_levels:
            if L in levels:
                i = list(levels).index(L)
                cells.append(f"{means[i]:5.2f}+-{stds[i]:.2f}")
            else:
                cells.append("    -   ")
        print(f"  {method:<12}  " + "  ".join(f"{c:>11}" for c in cells))
    print("  --- anchors (single cell γ=0, n=0) ---")
    for method in ANCHOR_METHODS:
        if method in anchor_agg:
            m, s, n = anchor_agg[method]
            print(f"  {method:<12}  acc={m:5.2f}+-{s:.2f} (n={n})")


def _load_or_partial(path_base):
    """Load <base>.json or fall back to <base>_partial.json. Returns [] if neither."""
    if os.path.exists(path_base):
        with open(path_base) as f:
            return json.load(f)
    partial = path_base.replace(".json", "_partial.json")
    if os.path.exists(partial):
        print(f"  using partial: {partial}")
        with open(partial) as f:
            return json.load(f)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="results/methods_af_noise")
    ap.add_argument("--fig_dir", default=None,
                    help="default: <input_dir>/figures")
    args = ap.parse_args()

    fig_dir = args.fig_dir or os.path.join(args.input_dir, "figures")

    anchor_results = _load_or_partial(os.path.join(args.input_dir, "anchor_results.json"))
    anchor_agg = aggregate_anchor(anchor_results) if anchor_results else {}
    if anchor_agg:
        print(f"[anchor] loaded: " + ", ".join(
            f"{m}={v[0]:.2f}±{v[1]:.2f} (n={v[2]})" for m, v in anchor_agg.items()))
    else:
        print("[anchor] no anchor_results found — anchors will not appear on plots")

    for axis in ("af", "noise"):
        path = os.path.join(args.input_dir, f"{axis}_results.json")
        sweep_results = _load_or_partial(path)
        if not sweep_results and not anchor_agg:
            print(f"[{axis}] no data — skipping")
            continue
        print_summary(sweep_results, anchor_agg, axis)
        plot_axis(sweep_results, anchor_agg, axis,
                  os.path.join(fig_dir, f"{axis}_sensitivity.png"))


if __name__ == "__main__":
    main()

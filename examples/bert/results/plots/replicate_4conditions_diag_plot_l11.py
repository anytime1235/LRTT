#!/usr/bin/env python3
"""Same 5 plots as replicate_4conditions_diag_plot.py but for last_tile
(L11.attention.output.dense — the layer where collapse occurs).

Generates plots with `_L11` suffix in filename to keep alongside the original
L0.query plots.

Usage:
  python replicate_4conditions_diag_plot_l11.py [diag_dir]
"""
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_step_csv(runs, csv_path, tile, metric_keys):
    rows = {}
    for cond in CONDITIONS:
        if cond not in runs: continue
        for m in metric_keys:
            steps, vals = get_steps(runs[cond], m)
            for s, v in zip(steps, vals):
                rows.setdefault((cond, int(s)), {})[m] = float(v)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "step"] + list(metric_keys))
        for (cond, step) in sorted(rows.keys(), key=lambda k: (CONDITIONS.index(k[0]), k[1])):
            r = rows[(cond, step)]
            w.writerow([cond, step] + [f"{r[m]:.6g}" if m in r else "" for m in metric_keys])
    print(f"Saved: {csv_path}")

DIAG_DIR_DEFAULT = Path(__file__).parent / "diag_4conditions"

CONDITIONS = ["no_noise", "a_only", "b_only", "both"]
COLORS = {
    "no_noise": "#1f77b4",
    "a_only":   "#2ca02c",
    "b_only":   "#ff7f0e",
    "both":     "#d62728",
}
LABELS = {
    "no_noise": "No noise (gamma/gamma)",
    "a_only":   "A only (6t1c/gamma)",
    "b_only":   "B only (gamma/6t1c)",
    "both":     "Both (6t1c/6t1c)",
}
TILE_KEY = "last_tile"          # ★ L11.output.dense
FILE_SUFFIX = "_L11"             # ★ output filename suffix


def load_runs(diag_dir):
    runs = {}
    for cond in CONDITIONS:
        p = diag_dir / f"diag_{cond}.json"
        if not p.exists():
            print(f"WARNING: {p} not found, skipping {cond}")
            continue
        runs[cond] = json.loads(p.read_text())
    return runs


_KEY_ALIAS = {
    # Modern (C_eff) key name first, legacy (C_raw) fallback for older JSONs.
    "norm_C_raw":    ("norm_C_eff",   "norm_C_raw"),
    "delta_C_raw":   ("delta_C_eff",  "delta_C_raw"),
    "erank_C":       ("erank_C_eff",  "erank_C"),
}


def get_steps(d, key):
    keys_to_try = _KEY_ALIAS.get(key, (key,))
    steps, vals = [], []
    for r in d[TILE_KEY]["steps"]:
        v = None
        for k in keys_to_try:
            v = r.get(k)
            if v is not None:
                break
        if v is not None:
            steps.append(r["step"]); vals.append(v)
    return np.array(steps), np.array(vals)


def smooth(y, win=20):
    if len(y) < win:
        return y
    return np.convolve(y, np.ones(win)/win, mode="same")


def plot_overlay(ax, runs, key, smooth_win=0, title=None, ylabel=None, yscale="linear"):
    for cond in CONDITIONS:
        if cond not in runs: continue
        steps, vals = get_steps(runs[cond], key)
        if len(steps) == 0: continue
        if smooth_win > 0: vals = smooth(vals, smooth_win)
        ax.plot(steps, vals, color=COLORS[cond], label=LABELS[cond],
                linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Step")
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, fontsize=10)
    if yscale == "log": ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)


def main(diag_dir):
    runs = load_runs(diag_dir)
    if not runs:
        print(f"No diag JSONs found in {diag_dir}")
        return

    sample = next(iter(runs.values()))
    tile_name = sample.get(TILE_KEY, {}).get("name", TILE_KEY)
    tile_label_short = tile_name.replace("bert.encoder.", "").replace(".analog_module", "")
    tile_subtitle = f"(tile: {tile_label_short})"

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10,
                         "legend.fontsize": 8, "axes.linewidth": 0.8})

    # === Figure 1: Weight norm dynamics (H4) ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_overlay(axes[0], runs, "norm_A", title="||A||  (H4: magnitude growth)", ylabel="||A||")
    plot_overlay(axes[1], runs, "norm_B", title="||B||", ylabel="||B||")
    plot_overlay(axes[2], runs, "norm_AB", title="||A·B||  (effective fast contribution)", ylabel="||A·B||")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(f"Hypothesis 4: Weight magnitude evolution  {tile_subtitle}", fontsize=11, y=1.02)
    fig.tight_layout()
    out1 = diag_dir / f"diag_plot1_norms{FILE_SUFFIX}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    fig.savefig(out1.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig); print(f"Saved: {out1}")
    _save_step_csv(runs, out1.with_suffix(".csv"), TILE_KEY,
                   ["norm_A", "norm_B", "norm_AB"])

    # === Figure 2: Effective ranks (H2) ===
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    plot_overlay(axes[0], runs, "erank_A", title="erank(A)", ylabel="effective rank")
    plot_overlay(axes[1], runs, "erank_B", title="erank(B)", ylabel="effective rank")
    plot_overlay(axes[2], runs, "erank_AB", title="erank(A·B)  (H2: rank degradation)", ylabel="effective rank")
    plot_overlay(axes[3], runs, "erank_C", title="erank(C)", ylabel="effective rank")
    axes[0].legend(loc="best", framealpha=0.9, fontsize=7)
    fig.suptitle(f"Hypothesis 2: Effective rank evolution  {tile_subtitle}", fontsize=11, y=1.02)
    fig.tight_layout()
    out2 = diag_dir / f"diag_plot2_erank{FILE_SUFFIX}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    fig.savefig(out2.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig); print(f"Saved: {out2}")
    _save_step_csv(runs, out2.with_suffix(".csv"), TILE_KEY,
                   ["erank_A", "erank_B", "erank_AB", "erank_C"])

    # === Figure 3: Update magnitudes (H1) ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_overlay(axes[0], runs, "delta_A", title="||ΔA|| per step (smoothed)", ylabel="||ΔA||", smooth_win=20)
    plot_overlay(axes[1], runs, "delta_B", title="||ΔB|| per step (smoothed)", ylabel="||ΔB||", smooth_win=20)
    plot_overlay(axes[2], runs, "delta_C_raw", title="||ΔC|| per step (smoothed)", ylabel="||ΔC||", smooth_win=20)
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(f"Hypothesis 1: Update magnitude  {tile_subtitle}", fontsize=11, y=1.02)
    fig.tight_layout()
    out3 = diag_dir / f"diag_plot3_deltas{FILE_SUFFIX}.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    fig.savefig(out3.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig); print(f"Saved: {out3}")
    _save_step_csv(runs, out3.with_suffix(".csv"), TILE_KEY,
                   ["delta_A", "delta_B", "delta_C_raw"])

    # === Figure 4: C tile noise (H3') ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_overlay(axes[0], runs, "norm_C_raw", title="||C_raw||  (H3': noise accumulation)", ylabel="||C_raw||")
    plot_overlay(axes[1], runs, "erank_C_delta", title="erank(C - C_init)", ylabel="effective rank")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(f"Hypothesis 3': Cumulative noise in C tile  {tile_subtitle}", fontsize=11, y=1.02)
    fig.tight_layout()
    out4 = diag_dir / f"diag_plot4_C_noise{FILE_SUFFIX}.png"
    fig.savefig(out4, dpi=150, bbox_inches="tight")
    fig.savefig(out4.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig); print(f"Saved: {out4}")
    _save_step_csv(runs, out4.with_suffix(".csv"), TILE_KEY,
                   ["norm_C_raw", "erank_C_delta"])

    # === Figure 5: Learning trajectory (same per-epoch F1/loss; tile-independent) ===
    # NOTE: identical to original — skipping regeneration to avoid duplicate file

    # === Figure 6: Weight distribution evolution (g3c_weight_hist) ===
    quantities = ["A", "B", "C_eff"]
    any_hist = any(
        any("hist_A" in s for s in runs[c][TILE_KEY]["steps"])
        for c in CONDITIONS if c in runs
    )
    if any_hist:
        fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex="col")
        cmap = plt.cm.viridis
        for ci, cond in enumerate(CONDITIONS):
            if cond not in runs:
                for cj in range(3): axes[ci, cj].set_visible(False)
                continue
            steps = runs[cond][TILE_KEY]["steps"]
            hist_steps = [s for s in steps if "hist_A" in s]
            n_hist = len(hist_steps)
            for qi, q in enumerate(quantities):
                ax = axes[ci, qi]
                key = f"hist_{q}"
                for hi, s in enumerate(hist_steps):
                    h = s.get(key)
                    if h is None: continue
                    counts = np.asarray(h["counts"], dtype=float)
                    bin_edges = np.linspace(h["min"], h["max"], len(counts) + 1)
                    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
                    color = cmap(hi / max(1, n_hist - 1))
                    label = f"step {s['step']}" if (ci == 0 and qi == 2) else None
                    ax.plot(centers, counts, color=color, linewidth=1.1, label=label)
                if ci == 0: ax.set_title(f"{q}")
                if qi == 0: ax.set_ylabel(LABELS[cond], fontsize=8)
                ax.set_yscale("log")
                ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
            if ci == 0:
                axes[0, 2].legend(loc="upper right", fontsize=6, framealpha=0.9)
        fig.suptitle(f"Weight distribution evolution  {tile_subtitle}", fontsize=11, y=1.005)
        fig.tight_layout()
        out6 = diag_dir / f"diag_plot6_weight_hist{FILE_SUFFIX}.png"
        fig.savefig(out6, dpi=150, bbox_inches="tight")
        fig.savefig(out6.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig); print(f"Saved: {out6}")
        csv6 = out6.with_suffix(".csv")
        with csv6.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["condition", "step", "quantity", "bin_idx",
                        "bin_low", "bin_high", "bin_center", "count"])
            for cond in CONDITIONS:
                if cond not in runs: continue
                for s in runs[cond][TILE_KEY]["steps"]:
                    if "hist_A" not in s: continue
                    for q in quantities:
                        h = s.get(f"hist_{q}")
                        if h is None: continue
                        counts = h["counts"]
                        edges = np.linspace(h["min"], h["max"], len(counts) + 1)
                        for bi, c in enumerate(counts):
                            w.writerow([cond, s["step"], q, bi,
                                        f"{edges[bi]:.6g}", f"{edges[bi+1]:.6g}",
                                        f"{(edges[bi]+edges[bi+1])*0.5:.6g}",
                                        f"{c:.0f}"])
        print(f"Saved: {csv6}")

    # === Summary stats ===
    print("\n=== Summary statistics (at last_tile = L11.attention.output.dense) ===")
    print(f"{'condition':<12} {'best_F1':>8} {'final||A||':>11} {'final||B||':>11} {'final||AB||':>11} {'final erank(AB)':>16}")
    for cond in CONDITIONS:
        if cond not in runs: continue
        d = runs[cond]
        last = d[TILE_KEY]["steps"][-1]
        f1 = d.get("best_f1", 0)
        print(f"{cond:<12} {f1:>8.2f} {last.get('norm_A',0):>11.3f} {last.get('norm_B',0):>11.3f} "
              f"{last.get('norm_AB',0):>11.3f} {last.get('erank_AB',0) or 0:>16.2f}")


if __name__ == "__main__":
    diag_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DIAG_DIR_DEFAULT
    main(diag_dir)

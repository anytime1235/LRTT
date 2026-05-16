#!/usr/bin/env python3
"""Visualize 4-condition diagnostic comparison from diag JSONs.

Loads diag_{condition}.json files written by replicate_4conditions_diag.py,
then produces hypothesis-specific overlay plots.

Usage:
  python replicate_4conditions_diag_plot.py [diag_dir]   # default: ./diag_4conditions
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def get_steps(d, key, tile="first_tile"):
    """Extract (steps, values) for a metric from diagnostic JSON, skipping None.
    Honors KEY_ALIAS so plots work on both new (C_eff) and legacy (C_raw) JSONs."""
    keys_to_try = _KEY_ALIAS.get(key, (key,))
    steps = []
    vals = []
    for r in d[tile]["steps"]:
        v = None
        for k in keys_to_try:
            v = r.get(k)
            if v is not None:
                break
        if v is not None:
            steps.append(r["step"])
            vals.append(v)
    return np.array(steps), np.array(vals)


def smooth(y, win=20):
    if len(y) < win:
        return y
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="same")


def plot_overlay(ax, runs, key, smooth_win=0, title=None, ylabel=None,
                 yscale="linear", tile="first_tile"):
    for cond in CONDITIONS:
        if cond not in runs: continue
        steps, vals = get_steps(runs[cond], key, tile=tile)
        if len(steps) == 0: continue
        if smooth_win > 0:
            vals = smooth(vals, smooth_win)
        ax.plot(steps, vals, color=COLORS[cond], label=LABELS[cond],
                linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Step")
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, fontsize=10)
    if yscale == "log": ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)


def main(diag_dir, tile_key="first_tile", suffix=""):
    runs = load_runs(diag_dir)
    if not runs:
        print(f"No diag JSONs found in {diag_dir}")
        return

    # Capture tile names so plot titles can show which layer is being plotted.
    sample = next(iter(runs.values()))
    tile_name = sample.get(tile_key, {}).get("name", tile_key)
    tile_label_short = tile_name.replace("bert.encoder.", "").replace(".analog_module", "")
    tile_subtitle = f"(tile: {tile_label_short})"

    # Configure
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10,
                         "legend.fontsize": 8, "axes.linewidth": 0.8})

    # === Figure 1: Weight norm dynamics (Hypothesis 4) ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_overlay(axes[0], runs, "norm_A", title="‖A‖ over training (H4: magnitude growth)",
                 ylabel="‖A‖")
    plot_overlay(axes[1], runs, "norm_B", title="‖B‖ over training",
                 ylabel="‖B‖")
    plot_overlay(axes[2], runs, "norm_AB", title="‖A·B‖ (effective fast contribution)",
                 ylabel="‖A·B‖")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(f"Hypothesis 4: Weight magnitude evolution  {tile_subtitle}", fontsize=11, y=1.02)
    fig.tight_layout()
    out1 = diag_dir / "diag_plot1_norms.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    fig.savefig(out1.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # === Figure 2: Effective ranks (Hypothesis 2) ===
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    plot_overlay(axes[0], runs, "erank_A", title="erank(A)",
                 ylabel="effective rank")
    plot_overlay(axes[1], runs, "erank_B", title="erank(B)",
                 ylabel="effective rank")
    plot_overlay(axes[2], runs, "erank_AB", title="erank(A·B)  (H2: rank degradation)",
                 ylabel="effective rank")
    plot_overlay(axes[3], runs, "erank_C", title="erank(C)",
                 ylabel="effective rank")
    axes[0].legend(loc="best", framealpha=0.9, fontsize=7)
    fig.suptitle(f"Hypothesis 2: Effective rank evolution  {tile_subtitle}", fontsize=11, y=1.02)
    fig.tight_layout()
    out2 = diag_dir / "diag_plot2_erank.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    fig.savefig(out2.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # === Figure 3: Update magnitudes (Hypothesis 1: noise feedback) ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_overlay(axes[0], runs, "delta_A", title="‖ΔA‖ per step (smoothed)",
                 ylabel="‖ΔA‖", smooth_win=20)
    plot_overlay(axes[1], runs, "delta_B", title="‖ΔB‖ per step (smoothed)",
                 ylabel="‖ΔB‖", smooth_win=20)
    plot_overlay(axes[2], runs, "delta_C_raw", title="‖ΔC‖ per step (smoothed)",
                 ylabel="‖ΔC‖", smooth_win=20)
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(f"Hypothesis 1: Update magnitude (proxy for gradient + write noise)  {tile_subtitle}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out3 = diag_dir / "diag_plot3_deltas.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    fig.savefig(out3.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out3}")

    # === Figure 4: C tile noise accumulation (Hypothesis 3') ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_overlay(axes[0], runs, "norm_C_raw", title="‖C_raw‖ (H3': noise accumulation in C)",
                 ylabel="‖C_raw‖")
    plot_overlay(axes[1], runs, "erank_C_delta", title="erank(C - C_init)",
                 ylabel="effective rank")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(f"Hypothesis 3': Cumulative noise in C tile (transfer target)  {tile_subtitle}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out4 = diag_dir / "diag_plot4_C_noise.png"
    fig.savefig(out4, dpi=150, bbox_inches="tight")
    fig.savefig(out4.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out4}")

    # === Figure 5: F1 / loss epoch trajectory ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cond in CONDITIONS:
        if cond not in runs: continue
        d = runs[cond]
        eh = d.get("epoch_history", [])
        if not eh: continue
        epochs = [e["epoch"] for e in eh]
        # Diagnostic JSON stores eval F1 under key 'f1' (not 'eval_f1')
        f1s = [e.get("f1", e.get("eval_f1", 0)) for e in eh]
        losses = [e.get("train_loss", 0) for e in eh]
        axes[0].plot(epochs, f1s, "o-", color=COLORS[cond], label=LABELS[cond],
                     linewidth=1.4, markersize=5)
        axes[1].plot(epochs, losses, "o-", color=COLORS[cond], label=LABELS[cond],
                     linewidth=1.4, markersize=5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Eval F1")
    axes[0].set_title("Eval F1 per epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Train loss")
    axes[1].set_title("Train loss per epoch")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.suptitle("Learning trajectory", fontsize=11, y=1.02)
    fig.tight_layout()
    out5 = diag_dir / "diag_plot5_learning.png"
    fig.savefig(out5, dpi=150, bbox_inches="tight")
    fig.savefig(out5.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out5}")

    # === Summary stats ===
    print("\n=== Summary statistics ===")
    print(f"{'condition':<12} {'best_F1':>8} {'final||A||':>11} {'final||B||':>11} {'final||AB||':>11} {'final erank(AB)':>16}")
    for cond in CONDITIONS:
        if cond not in runs: continue
        d = runs[cond]
        last = d["first_tile"]["steps"][-1]
        f1 = d.get("best_f1", 0)
        print(f"{cond:<12} {f1:>8.2f} {last.get('norm_A',0):>11.3f} {last.get('norm_B',0):>11.3f} "
              f"{last.get('norm_AB',0):>11.3f} {last.get('erank_AB',0) or 0:>16.2f}")


if __name__ == "__main__":
    diag_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DIAG_DIR_DEFAULT
    main(diag_dir)

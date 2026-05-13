#!/usr/bin/env python3
"""Plot best val acc vs rank for the c-device sweep (3 devices × 2 reinit modes).

Reads the per-(device, reinit) study log, buckets trials by rank_exp, and
plots the best val accuracy per (device, reinit, rank) cell. One panel per
C device, with two lines per panel for gauss_a_zero and gauss_b_zero.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEVICES = ["idealizedpreset", "reramespreset", "ecrampreset"]
DEVICE_LABELS = {
    "idealizedpreset": "IdealizedPresetDevice",
    "reramespreset": "ReRamESPresetDevice",
    "ecrampreset": "EcRamPresetDevice",
}
REINIT_MODES = ["gauss_a_zero", "gauss_b_zero"]
REINIT_COLORS = {"gauss_a_zero": "#d62728", "gauss_b_zero": "#1f77b4"}
RANK_EXPS = [0, 2, 4, 6]
RANKS = [2 ** e for e in RANK_EXPS]


def study_log_path(results_dir: Path, reinit_mode: str, c_device: str) -> Path:
    return results_dir / (
        f"optuna_mlp_mnist_lrtt_bs64_sgd_{reinit_mode}_nowd_nomom_nonest_"
        f"onehot_constantstepideal_c{c_device}_perfect_no-stlr_split-reset_std_linear1_30ep.log"
    )


def best_per_rank(log_path: Path):
    """Return dict: rank_exp → best val acc (or None) parsed from the journal."""
    if not log_path.exists():
        return {e: None for e in RANK_EXPS}
    rank_best = {e: None for e in RANK_EXPS}
    params = {}
    with log_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = r.get("trial_id")
            if r.get("op_code") == 5 and r.get("param_name") == "rank_exp":
                params.setdefault(tid, {})["rank_exp"] = int(r["param_value_internal"])
            if r.get("op_code") == 6 and r.get("state") == 1:
                rk = params.get(tid, {}).get("rank_exp")
                if rk in rank_best:
                    v = r["values"][0]
                    if rank_best[rk] is None or v > rank_best[rk]:
                        rank_best[rk] = v
    return rank_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    args = ap.parse_args()

    here = Path(__file__).parent
    mlp_dir = here.parent
    results_dir = Path(args.results_dir) if args.results_dir else mlp_dir / "results" / "optuna_mlp_mnist_lrtt"
    out = Path(args.out) if args.out else here / "c_device_rank_sweep.png"

    # Collect all data
    data = {}   # (mode, device) → {rank_exp: best}
    for mode in REINIT_MODES:
        for dev in DEVICES:
            data[(mode, dev)] = best_per_rank(study_log_path(results_dir, mode, dev))

    # Plot: 1×3 panels (one per device)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, dev in zip(axes, DEVICES):
        for mode in REINIT_MODES:
            ys = [data[(mode, dev)][e] for e in RANK_EXPS]
            xs = RANKS
            color = REINIT_COLORS[mode]
            ax.plot(xs, ys, "-o", color=color, label=mode, linewidth=2, markersize=7)
            for x, y in zip(xs, ys):
                if y is not None:
                    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_xlabel("rank")
        ax.set_title(DEVICE_LABELS[dev])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Best val acc (%)")
    if args.ymin is not None or args.ymax is not None:
        for ax in axes:
            ax.set_ylim(args.ymin, args.ymax)
    axes[-1].legend(loc="lower right")
    fig.suptitle("LRTT MLP-MNIST — C device sweep (rank-exp ∈ {0,2,4,6}, AB=constantstepideal, noise=0)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Also print the data table
    print("\nbest val acc (%) by (device, mode, rank):")
    print(f"{'device':<22} {'mode':<14} " + " ".join(f"{f'rank={r}':>8}" for r in RANKS))
    for dev in DEVICES:
        for mode in REINIT_MODES:
            row = " ".join(f"{data[(mode,dev)][e]:>8.2f}" if data[(mode,dev)][e] is not None else f"{'—':>8}" for e in RANK_EXPS)
            print(f"{DEVICE_LABELS[dev]:<22} {mode:<14} {row}")


if __name__ == "__main__":
    main()

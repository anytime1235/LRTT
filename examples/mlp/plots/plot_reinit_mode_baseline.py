#!/usr/bin/env python3
"""Bar plot comparing baseline best val acc across reinit modes.

Compares gauss_a_zero, gauss_b_zero, and orthogonal_zero (relabeled as
fix_b_zero in the figure) at the prior baseline config: constantstepideal
A/B/C devices, no af/unr scaling, rank=8, transfer_every=10.

For orthogonal_zero the study log includes trials with varied rank/te (from
an earlier broad sweep), so we filter to rank_exp=3 (rank=8) and
transfer_every=10 for a fair comparison.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# (display_label, log filename, filter for rank=8/te=10)
ENTRIES = [
    (
        "gauss_a_zero",
        "optuna_mlp_mnist_lrtt_bs64_sgd_gauss_a_zero_nowd_nomom_nonest_onehot_"
        "aconstantstepideal_bconstantstepideal_cconstantstepideal_perfect_"
        "no-stlr_split-reset_std_linear1_30ep.log",
        False,
    ),
    (
        "gauss_b_zero",
        "optuna_mlp_mnist_lrtt_bs64_sgd_gauss_b_zero_nowd_nomom_nonest_onehot_"
        "aconstantstepideal_bconstantstepideal_cconstantstepideal_perfect_"
        "no-stlr_split-reset_std_linear1_30ep.log",
        False,
    ),
    (
        "fix_b_zero",
        "optuna_mlp_mnist_lrtt_bs64_sgd_orthogonal_zero_nowd_nomom_nonest_onehot_"
        "aconstantstepideal_bconstantstepideal_cconstantstepideal_perfect_"
        "no-stlr_linear1_30ep.log",
        True,  # filter rank_exp=3 & transfer_every=10
    ),
]


def collect(log: Path, do_filter: bool):
    """Return (best_val, n_complete) — optionally filtered to rank=8 te=10."""
    p_by_t = defaultdict(dict)
    res = {}
    if not log.exists():
        return None, 0
    with log.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("op_code") == 5:
                p_by_t[r["trial_id"]][r["param_name"]] = r["param_value_internal"]
            elif r.get("op_code") == 6 and r.get("state") == 1:
                res[r["trial_id"]] = r["values"][0]
    if do_filter:
        res = {
            tid: v for tid, v in res.items()
            if int(p_by_t[tid].get("rank_exp", -1)) == 3
            and int(p_by_t[tid].get("transfer_every", -1)) == 10
        }
    if not res:
        return None, 0
    return max(res.values()), len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = Path(__file__).parent
    mlp_dir = here.parent
    results_dir = Path(args.results_dir) if args.results_dir else mlp_dir / "results" / "optuna_mlp_mnist_lrtt"

    labels, bests, counts = [], [], []
    for label, fname, do_filter in ENTRIES:
        v, n = collect(results_dir / fname, do_filter)
        labels.append(label)
        bests.append(v)
        counts.append(n)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.arange(len(labels))
    colors = ["#3b82f6", "#f97316", "#8b5cf6"]
    bars = ax.bar(x, [v if v is not None else 0 for v in bests], width=0.55,
                  color=colors[:len(labels)])
    for b, v in zip(bars, bests):
        if v is not None:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best val acc (%)")
    ax.set_title("Reinit-mode baseline: best val acc (rank=8, te=10, no af/unr scaling)")
    ax.set_ylim(90, 100)
    ax.grid(axis="y", alpha=0.3)

    out = Path(args.out) if args.out else here / "bar_reinit_mode_baseline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    csv_out = out.with_suffix(".csv")
    with csv_out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reinit_mode", "best_val_acc", "n_complete"])
        for label, v, n in zip(labels, bests, counts):
            w.writerow([label,
                        f"{v:.3f}" if v is not None else "",
                        int(n)])
    print(f"Saved: {csv_out}")

    print("\nbaseline best val acc (rank=8, te=10):")
    for label, v, n in zip(labels, bests, counts):
        print(f"  {label:<14} {v:.3f}  (n={n})")


if __name__ == "__main__":
    main()

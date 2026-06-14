#!/usr/bin/env python3
"""Bar chart comparing F1 across 4 noise conditions (a_only/b_only/both/no_noise).

Reads summary_*.json files and emits a grouped bar chart showing each run's
final F1 per condition. Hyperparams from the latest run are annotated.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COND_ORDER = ["no_noise", "a_only", "b_only", "both"]
COND_LABELS = {
    "no_noise": "no noise\n(A: ideal, B: ideal)",
    "a_only":   "A only noise\n(A: 6t1c, B: ideal)",
    "b_only":   "B only noise\n(A: ideal, B: 6t1c)",
    "both":     "both noisy\n(A: 6t1c, B: 6t1c)",
}
COND_COLORS = {
    "no_noise": "#4c72b0",  # blue
    "a_only":   "#dd8452",  # orange
    "b_only":   "#55a467",  # green
    "both":     "#c44e52",  # red
}


def load_summaries(paths):
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        out.append((Path(p).stem, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+",
                    help="One or more summary_*.json files.")
    ap.add_argument("--out", default="f1_comparison.png",
                    help="Output PNG path (default in same dir as script).")
    ap.add_argument("--ymin", type=float, default=70.0)
    ap.add_argument("--ymax", type=float, default=90.0)
    ap.add_argument("--title", default="LRTT BERT SQuAD — 4-condition noise comparison")
    args = ap.parse_args()

    summaries = load_summaries(args.summaries)
    n_runs = len(summaries)
    n_cond = len(COND_ORDER)

    # Pull F1 values into a 2D array [run, condition]
    f1_grid = np.full((n_runs, n_cond), np.nan)
    for ri, (_, d) in enumerate(summaries):
        for r in d["results"]:
            if r["tag"] in COND_ORDER:
                f1_grid[ri, COND_ORDER.index(r["tag"])] = r["f1"]

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / n_runs
    x = np.arange(n_cond)

    for ri, (stem, _) in enumerate(summaries):
        offset = (ri - (n_runs - 1) / 2) * width
        ys = f1_grid[ri]
        for ci, cond in enumerate(COND_ORDER):
            ax.bar(x[ci] + offset, ys[ci], width,
                   color=COND_COLORS[cond],
                   edgecolor="black", linewidth=0.5,
                   label=stem.replace("summary_", "") if ci == 0 else None,
                   alpha=0.7 + 0.3 * (ri / max(n_runs - 1, 1)))
            ax.text(x[ci] + offset, ys[ci] + 0.15, f"{ys[ci]:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=10)
    ax.set_ylabel("F1 score")
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    if n_runs > 1:
        ax.legend(loc="upper right", title="Run", fontsize=9)

    # Annotate hyperparams from the most recent summary
    if summaries:
        h = summaries[-1][1].get("hyperparams", {})
        keys = ["lr", "tlr", "te", "rank", "fast_lr", "ab_dw_min", "c_dw_min",
                "batch_size", "warmup_steps", "seed"]
        hstr = ", ".join(f"{k}={h[k]}" for k in keys if k in h)
        fig.text(0.5, -0.04, hstr, ha="center", fontsize=8, color="gray",
                 wrap=True)

    fig.tight_layout()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also dump a CSV
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("run," + ",".join(COND_ORDER) + "\n")
        for ri, (stem, _) in enumerate(summaries):
            row = ",".join(f"{f1_grid[ri, ci]:.3f}" if not np.isnan(f1_grid[ri, ci]) else ""
                           for ci in range(n_cond))
            f.write(f"{stem},{row}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

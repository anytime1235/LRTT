#!/usr/bin/env python3
"""Plot replicate_4conditions JSON results as box plot + scatter (with means).

Usage:
  python replicate_4conditions_plot.py replicate_4conditions_20260429_*.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITION_LABELS = {
    "no_noise": "No noise\n(both gamma)",
    "a_only":   "A only noise\n(A=6t1c, B=gamma)",
    "b_only":   "B only noise\n(A=gamma, B=6t1c)",
    "both":     "Both noise\n(both 6t1c)",
}
CONDITION_ORDER = ["no_noise", "a_only", "b_only", "both"]
COLORS = {
    "no_noise": "#1f77b4",
    "a_only":   "#2ca02c",
    "b_only":   "#ff7f0e",
    "both":     "#d62728",
}


def main(json_path):
    data = json.loads(Path(json_path).read_text())
    conds = data["conditions"]

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(7, 5))

    positions = list(range(len(CONDITION_ORDER)))
    box_data, labels, means, stds, ns = [], [], [], [], []
    for x, key in zip(positions, CONDITION_ORDER):
        c = conds.get(key, {})
        f1s = [v for v in c.get("f1_per_seed", {}).values() if v is not None]
        box_data.append(f1s if f1s else [0])
        labels.append(CONDITION_LABELS[key])
        means.append(c.get("mean"))
        stds.append(c.get("std"))
        ns.append(c.get("n", 0))

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.5, patch_artist=True,
        showmeans=False, medianprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="x", markersize=5, markeredgecolor="0.4"),
    )
    for patch, key in zip(bp["boxes"], CONDITION_ORDER):
        patch.set_facecolor(COLORS[key])
        patch.set_alpha(0.45)

    # Overlay individual points
    for x, key, f1s in zip(positions, CONDITION_ORDER, box_data):
        if not f1s or f1s == [0]:
            continue
        rng = np.random.default_rng(seed=0)
        jitter = rng.uniform(-0.08, 0.08, size=len(f1s))
        ax.scatter(np.full(len(f1s), x) + jitter, f1s, color=COLORS[key],
                   edgecolor="black", linewidth=0.5, s=35, zorder=3)
        # Mean marker
        m = conds[key].get("mean")
        if m is not None:
            ax.scatter([x], [m], marker="D", s=70, color="white",
                       edgecolor=COLORS[key], linewidth=1.5, zorder=4)

    # Annotate mean ± std and n
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    for x, key in zip(positions, CONDITION_ORDER):
        c = conds.get(key, {})
        m, s, n = c.get("mean"), c.get("std"), c.get("n", 0)
        if m is None:
            continue
        ax.annotate(
            f"{m:.2f}±{s:.2f}\n(n={n})",
            (x, m), textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=8, color=COLORS[key],
            fontweight="bold",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 score")
    ax.set_title("Noise placement effect on LR-TT (BERT SQuAD, qkvo, 5 seeds each)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(json_path).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved: {out}")

    # Print summary table
    print("\nSummary:")
    print(f"  {'condition':<12} {'mean':>7} {'std':>6} {'n':>3}")
    for key in CONDITION_ORDER:
        c = conds.get(key, {})
        if c.get("mean") is not None:
            print(f"  {key:<12} {c['mean']:>7.3f} {c['std']:>6.3f} {c['n']:>3}")
        else:
            print(f"  {key:<12} (no data)")

    # Pairwise differences (mean of each minus no_noise)
    bn = conds.get("no_noise", {}).get("mean")
    if bn is not None:
        print(f"\nDifferences from no_noise baseline ({bn:.3f}):")
        for key in CONDITION_ORDER:
            if key == "no_noise":
                continue
            c = conds.get(key, {})
            if c.get("mean") is not None:
                delta = c["mean"] - bn
                # 95% CI of difference (two-sample, assuming equal variance)
                # SE = sqrt(s1^2/n1 + s2^2/n2)
                s1, s2 = c["std"], conds["no_noise"]["std"]
                n1, n2 = c["n"], conds["no_noise"]["n"]
                if n1 > 0 and n2 > 0:
                    se = (s1**2 / n1 + s2**2 / n2) ** 0.5
                    ci = 1.96 * se
                    sig = " *" if abs(delta) > ci else ""
                    print(f"  {key:<12} {delta:+.3f}  (95%CI ±{ci:.3f}){sig}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find latest replicate_4conditions JSON
        here = Path(__file__).parent
        candidates = sorted(here.glob("replicate_4conditions_*.json"))
        if not candidates:
            print("Usage: python replicate_4conditions_plot.py <json_path>")
            sys.exit(1)
        json_path = candidates[-1]
        print(f"Using latest: {json_path}")
    else:
        json_path = sys.argv[1]
    main(json_path)

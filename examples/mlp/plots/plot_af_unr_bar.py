#!/usr/bin/env python3
"""Bar plot of best Optuna objective along the af_ratio = unr diagonal.

Shows degradation across noise levels (af=unr ∈ {0, 1, 10, 100}) for both
gauss_a_zero and gauss_b_zero reinit modes. Saves PNG + raw-data CSV.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEVELS = [0, 1, 10, 100]
MODES = ["gauss_a_zero", "gauss_b_zero"]


def cell_log_path(results_dir: Path, reinit_mode: str, af, unr) -> Path:
    name = (
        f"optuna_mlp_mnist_lrtt_bs64_sgd_{reinit_mode}_nowd_nomom_nonest_"
        f"onehot_ascaledideal_bscaledideal_cconstantstepideal_perfect_"
        f"no-stlr_af{af:g}_unr{unr:g}_split-reset_std_linear1_30ep.log"
    )
    return results_dir / name


def best_value(log_path: Path):
    if not log_path.exists():
        return None, 0
    best, n = None, 0
    with log_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("op_code") == 6 and r.get("state") == 1:
                n += 1
                v = r["values"][0]
                if best is None or v > best:
                    best = v
    return best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = Path(__file__).parent
    mlp_dir = here.parent
    results_dir = Path(args.results_dir) if args.results_dir else mlp_dir / "results" / "optuna_mlp_mnist_lrtt"

    data = {m: [] for m in MODES}      # mode → [best_val per level]
    counts = {m: [] for m in MODES}    # mode → [n_complete per level]
    for m in MODES:
        for lvl in LEVELS:
            v, n = best_value(cell_log_path(results_dir, m, lvl, lvl))
            data[m].append(v)
            counts[m].append(n)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(LEVELS))
    width = 0.38
    colors = {"gauss_a_zero": "#3b82f6", "gauss_b_zero": "#f97316"}
    for i, m in enumerate(MODES):
        offset = (i - 0.5) * width
        vals = [v if v is not None else 0 for v in data[m]]
        bars = ax.bar(x + offset, vals, width, label=m, color=colors[m])
        for b, v, n in zip(bars, data[m], counts[m]):
            label = f"{v:.2f}" if v is not None else "—"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    label, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"af=unr={lvl}" for lvl in LEVELS])
    ax.set_ylabel("Best val acc (%)")
    ax.set_title("af×unr diagonal sweep: best val acc by reinit mode (rank=8, te=10)")
    ax.set_ylim(85, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")

    out = Path(args.out) if args.out else here / "bar_af_unr_diagonal.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Save raw data as CSV alongside the PNG
    csv_out = out.with_suffix(".csv")
    with csv_out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reinit_mode", "af_ratio", "unr", "best_val_acc", "n_complete"])
        for m in MODES:
            for lvl, v, n in zip(LEVELS, data[m], counts[m]):
                w.writerow([m, lvl, lvl,
                            f"{v:.3f}" if v is not None else "",
                            int(n)])
    print(f"Saved: {csv_out}")

    # Print table for sanity
    print("\nbest val acc (%) along af=unr diagonal:")
    print(f"{'mode':<14} " + " ".join(f"{f'af=unr={lvl}':>12}" for lvl in LEVELS))
    for m in MODES:
        row = " ".join(f"{(f'{v:.2f}' if v is not None else '—'):>12}" for v in data[m])
        print(f"{m:<14} {row}")


if __name__ == "__main__":
    main()

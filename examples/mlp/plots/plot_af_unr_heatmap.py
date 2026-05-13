#!/usr/bin/env python3
"""Plot heatmap(s) of best Optuna objective for the af_ratio × unr sweep.

By default, generates a combined figure showing both gauss_a_zero and
gauss_b_zero heatmaps side by side. Use --reinit-mode to render a single panel.

Axes:  UNR on x, γ (gamma asymmetry, af_ratio) on y.
Color: viridis with vmin=90 (purple) → vmax=98 (yellow).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AF_VALUES = [0, 1, 2, 5, 10]
UNR_VALUES = [0, 1, 3, 5, 10]


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


def collect_grid(results_dir: Path, reinit_mode: str):
    """Return grid[af_idx, unr_idx], counts[af_idx, unr_idx]."""
    grid = np.full((len(AF_VALUES), len(UNR_VALUES)), np.nan)
    counts = np.zeros_like(grid, dtype=int)
    for i, af in enumerate(AF_VALUES):
        for j, unr in enumerate(UNR_VALUES):
            v, n = best_value(cell_log_path(results_dir, reinit_mode, af, unr))
            counts[i, j] = n
            if v is not None:
                grid[i, j] = v
    return grid, counts


def render_panel(ax, grid, counts, title, vmin=90.0, vmax=98.0):
    cmap = plt.get_cmap("plasma")
    im = ax.imshow(grid, cmap=cmap, origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(UNR_VALUES)))
    ax.set_xticklabels([str(v) for v in UNR_VALUES])
    ax.set_yticks(range(len(AF_VALUES)))
    ax.set_yticklabels([str(v) for v in AF_VALUES])
    ax.set_xlabel("UNR (update noise ratio)")
    ax.set_ylabel(r"AF ($\gamma$ asymmetry)")
    ax.set_title(title)
    midpoint = (vmin + vmax) / 2
    for i in range(len(AF_VALUES)):
        for j in range(len(UNR_VALUES)):
            v = grid[i, j]
            if np.isnan(v):
                txt = "—"
                color = "white"
            else:
                txt = f"{v:.2f}"
                color = "white" if v < midpoint else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reinit-mode", default=None,
                    help="If set, render a single panel for this mode instead of both.")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vmin", type=float, default=90.0)
    ap.add_argument("--vmax", type=float, default=98.0)
    args = ap.parse_args()

    here = Path(__file__).parent           # plots/ (output dir for PNG)
    mlp_dir = here.parent                   # mlp/ (root for results/)
    results_dir = Path(args.results_dir) if args.results_dir else mlp_dir / "results" / "optuna_mlp_mnist_lrtt"

    if args.reinit_mode:
        modes = [args.reinit_mode]
    else:
        modes = ["gauss_a_zero", "gauss_b_zero"]

    # Collect all data first (for CSV dump)
    all_data = {}  # mode → (grid, counts)
    for m in modes:
        all_data[m] = collect_grid(results_dir, m)

    if len(modes) == 1:
        out = Path(args.out) if args.out else here / f"heatmap_af_unr_{modes[0]}.png"
        csv_out = out.with_suffix(".csv")
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        grid, counts = all_data[modes[0]]
        im = render_panel(ax, grid, counts,
                          f"{modes[0]} (rank=8, te=10)",
                          vmin=args.vmin, vmax=args.vmax)
        fig.colorbar(im, ax=ax, label="Best val acc (%)")
    else:
        out = Path(args.out) if args.out else here / "heatmap_af_unr_combined.png"
        csv_out = out.with_suffix(".csv")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, mode in zip(axes, modes):
            grid, counts = all_data[mode]
            im = render_panel(ax, grid, counts,
                              f"{mode} (rank=8, te=10)",
                              vmin=args.vmin, vmax=args.vmax)
        fig.colorbar(im, ax=axes, label="Best val acc (%)",
                     fraction=0.04, pad=0.02)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Save raw data as CSV alongside the PNG
    import csv
    with csv_out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reinit_mode", "af_ratio", "unr", "best_val_acc", "n_complete"])
        for m in modes:
            grid, counts = all_data[m]
            for i, af in enumerate(AF_VALUES):
                for j, unr in enumerate(UNR_VALUES):
                    v = grid[i, j]
                    n = counts[i, j]
                    w.writerow([m, af, unr,
                                f"{v:.3f}" if not np.isnan(v) else "",
                                int(n)])
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()

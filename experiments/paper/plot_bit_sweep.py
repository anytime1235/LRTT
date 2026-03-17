#!/usr/bin/env python
"""Plot bit-width sweep results: Single RPU vs Mixed Precision vs TTv1."""

import os
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["bits_fast"] = int(r["bits_fast"])
            r["bits_slow"] = int(r["bits_slow"])
            r["best_f1"] = float(r["best_f1"])
            r["final_f1"] = float(r["final_f1"])
            r["final_em"] = float(r["final_em"])
            rows.append(r)
    return rows


def plot(rows, output_dir):
    methods = {
        "single_rpu": {"label": "Single RPU", "color": "#2196F3", "marker": "o"},
        "ttv1": {"label": "TTv1 (Fast=var, Slow=10b)", "color": "#FF5722", "marker": "s"},
        "mixed_precision": {"label": "Mixed Precision (10b)", "color": "#4CAF50", "marker": "D"},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for method, style in methods.items():
        data = [r for r in rows if r["method"] == method and r["best_f1"] > 0]
        if not data:
            continue
        bits = [r["bits_fast"] for r in data]
        best_f1 = [r["best_f1"] for r in data]
        final_em = [r["final_em"] for r in data]

        ax1.plot(bits, best_f1, marker=style["marker"], color=style["color"],
                 label=style["label"], linewidth=2, markersize=8)
        ax2.plot(bits, final_em, marker=style["marker"], color=style["color"],
                 label=style["label"], linewidth=2, markersize=8)

    for ax, ylabel, title in [
        (ax1, "Best F1", "Best F1 vs Bit-Width"),
        (ax2, "Final EM", "Final EM vs Bit-Width"),
    ]:
        ax.set_xlabel("Bit-Width (Fast Tile for TTv1)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks([8, 10, 12, 14, 16])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bit_sweep_f1_em.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close()

    # --- Second plot: F1 gap from ideal (single_rpu as proxy) ---
    single_data = {r["bits_fast"]: r["best_f1"] for r in rows
                   if r["method"] == "single_rpu" and r["best_f1"] > 0}
    if single_data:
        fig2, ax3 = plt.subplots(figsize=(8, 5))
        for method, style in methods.items():
            if method == "single_rpu":
                continue
            data = [r for r in rows if r["method"] == method and r["best_f1"] > 0]
            if not data:
                continue
            bits = [r["bits_fast"] for r in data]
            gaps = [r["best_f1"] - single_data.get(r["bits_fast"], r["best_f1"])
                    for r in data]
            ax3.plot(bits, gaps, marker=style["marker"], color=style["color"],
                     label=f"{style['label']} - Single RPU", linewidth=2, markersize=8)

        ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Bit-Width", fontsize=12)
        ax3.set_ylabel("F1 Gap vs Single RPU", fontsize=12)
        ax3.set_title("F1 Gap: Method vs Single RPU Baseline", fontsize=13, fontweight="bold")
        ax3.set_xticks([8, 10, 12, 14, 16])
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()

        gap_path = os.path.join(output_dir, "bit_sweep_gap.png")
        fig2.savefig(gap_path, dpi=150, bbox_inches="tight")
        print(f"Gap plot saved: {gap_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/paper/phase3_bitsweep")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "bit_sweep_results.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        exit(1)

    rows = load_csv(csv_path)
    plot(rows, args.results_dir)

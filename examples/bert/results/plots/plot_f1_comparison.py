#!/usr/bin/env python3
"""Bar plot + CSV of best F1 across 4 conditions for the diag replication.

Reads the most-recent summary_*.json in the given diag dir (default:
diag_4conditions_multitile/) and writes f1_comparison.{png,svg,csv}.

Usage:
  python plot_f1_comparison.py [diag_dir]
"""
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIAG_DIR_DEFAULT = Path(__file__).parent / "diag_4conditions_multitile"

CONDITIONS = ["no_noise", "a_only", "b_only", "both"]
LABELS = {
    "no_noise": "No noise\n(gamma/gamma)",
    "a_only":   "A only\n(6t1c/gamma)",
    "b_only":   "B only\n(gamma/6t1c)",
    "both":     "Both\n(6t1c/6t1c)",
}
COLORS = {
    "no_noise": "#1f77b4",
    "a_only":   "#2ca02c",
    "b_only":   "#ff7f0e",
    "both":     "#d62728",
}


def latest_summary(diag_dir: Path) -> Path:
    candidates = sorted(diag_dir.glob("summary_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No summary_*.json in {diag_dir}")
    return candidates[-1]


def main(diag_dir: Path):
    sm_path = latest_summary(diag_dir)
    sm = json.loads(sm_path.read_text())
    print(f"Loading: {sm_path.name}  (stamp={sm.get('timestamp', '?')})")

    f1_by_cond = {r["tag"]: r["f1"] for r in sm.get("results", [])}
    labels = [c for c in CONDITIONS if c in f1_by_cond]
    f1s = [f1_by_cond[c] for c in labels]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    colors = [COLORS[c] for c in labels]
    bars = ax.bar(x, f1s, width=0.6, color=colors)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in labels], fontsize=9)
    ax.set_ylabel("Best F1")
    ax.set_title(f"4-condition F1 comparison  (stamp: {sm.get('timestamp', '?')})")
    ax.set_ylim(min(f1s) - 2.0, max(f1s) + 1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    out_png = diag_dir / "f1_comparison.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_png.with_suffix('.svg')}")

    # --- CSV ---
    csv_path = out_png.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "f1", "exit_code", "source_summary"])
        for r in sm.get("results", []):
            w.writerow([r["tag"], f"{r['f1']:.4f}", r.get("exit_code", ""), sm_path.name])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    diag_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DIAG_DIR_DEFAULT
    main(diag_dir)

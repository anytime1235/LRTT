#!/usr/bin/env python3
"""Bilinear instability hypothesis verification using existing diagnostic data.

Uses the seed42_lrfloor001 collapse case (deterministic L11.output collapse) and
seed42 stable case (no collapse) from diag_no_noise_variants/. Tests:

1. Bilinear amplification signature: |x|, |d| stay constant while |XB|, |DA| explode
   → A, B grow but external signals don't
2. Layer specificity: only L11.output saturates; L0.q stays bounded
3. Coherent direction emergence: dominant amplification ratio (|XB|·|d|)·(|x|·|DA|)
   should peak at L11.output near collapse

Generates plots and quantitative summary.
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "diag_no_noise_variants"
OUT_DIR = Path(__file__).parent / "bilinear_verification"
OUT_DIR.mkdir(exist_ok=True)


def load(tag):
    return json.load(open(DATA_DIR / f"diag_{tag}.json"))


def get_series(d, tile_key, field):
    return np.array([r.get(field, 0) for r in d[tile_key]["steps"]]), \
           np.array([r["step"] for r in d[tile_key]["steps"]])


def plot_run(d, tag, out_path):
    """For one run, plot all key trajectories at L0.q (first_tile) and L11.output (last_tile)."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    tiles = [("first_tile", "L0.attention.self.query"), ("last_tile", "L11.attention.output.dense")]
    fields = [
        ("norm_A", "‖A‖", "C0"),
        ("norm_B", "‖B‖", "C1"),
        ("norm_AB", "‖A·B‖", "C2"),
    ]
    abs_fields = [
        ("xa_abs_max", "|XB|_max (A tile input)", "C3"),
        ("da_abs_max", "|d|_max (A tile gradient)", "C4"),
        ("xb_abs_max", "|x|_max (B tile input)", "C5"),
        ("db_abs_max", "|DA|_max (B tile gradient)", "C6"),
    ]

    for col, (tile_key, tile_name) in enumerate(tiles):
        # Row 0: weight norms
        for field, label, color in fields:
            vals, steps = get_series(d, tile_key, field)
            axes[0, col].plot(steps, vals, color=color, label=label, linewidth=1.0)
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylabel("weight norms (log)")
        axes[0, col].set_title(f"{tile_name}")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=7, loc="upper left")

        # Row 1: external signals (|x|, |d|)
        xb_vals, steps = get_series(d, tile_key, "xb_abs_max")
        da_vals, _ = get_series(d, tile_key, "da_abs_max")
        axes[1, col].plot(steps, xb_vals, color="C5", label="|x|_max (raw input)", linewidth=1.0)
        axes[1, col].plot(steps, da_vals, color="C4", label="|d|_max (raw gradient)", linewidth=1.0)
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylabel("external signal (log)")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend(fontsize=7, loc="upper left")

        # Row 2: internal amplified signals (|XB|, |DA|)
        xa_vals, _ = get_series(d, tile_key, "xa_abs_max")
        db_vals, _ = get_series(d, tile_key, "db_abs_max")
        axes[2, col].plot(steps, xa_vals, color="C3", label="|XB|_max  (B·x, depends on ‖B‖)", linewidth=1.0)
        axes[2, col].plot(steps, db_vals, color="C6", label="|DA|_max  (A^T·d, depends on ‖A‖)", linewidth=1.0)
        axes[2, col].set_yscale("log")
        axes[2, col].set_ylabel("amplified signal (log)")
        axes[2, col].set_xlabel("step")
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Bilinear amplification trace — {tag}", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def compute_amplification_ratios(d, tile_key):
    """Compute amplification ratios at each step.

    R_A = |XB|_max / |x|_max ~ proportional to ‖B‖ (B's amplification)
    R_B = |DA|_max / |d|_max ~ proportional to ‖A‖ (A's amplification)
    """
    xa, _ = get_series(d, tile_key, "xa_abs_max")     # |XB|
    xb, _ = get_series(d, tile_key, "xb_abs_max")     # |x|
    db, _ = get_series(d, tile_key, "db_abs_max")     # |DA|
    da, _ = get_series(d, tile_key, "da_abs_max")     # |d|
    R_A = np.divide(xa, xb, out=np.zeros_like(xa), where=xb > 0)   # |XB|/|x|
    R_B = np.divide(db, da, out=np.zeros_like(db), where=da > 0)   # |DA|/|d|
    return R_A, R_B


def comparison_plot():
    """Compare collapse vs stable runs."""
    runs = {
        "stable (seed42, min_lr=0.0)": load("seed42"),
        "collapse (seed42_lrfloor001)": load("seed42_lrfloor001"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)

    for col, tile_key in enumerate(["first_tile", "last_tile"]):
        tile_name = ("L0.query" if tile_key == "first_tile"
                     else "L11.output.dense")
        for label, d in runs.items():
            R_A, R_B = compute_amplification_ratios(d, tile_key)
            steps = np.array([r["step"] for r in d[tile_key]["steps"]])
            style = "-" if "stable" in label else "-"
            color = "C0" if "stable" in label else "C3"
            axes[0, col].plot(steps, R_A, style, color=color, label=f"{label}", linewidth=1.0)
            axes[1, col].plot(steps, R_B, style, color=color, label=f"{label}", linewidth=1.0)
        axes[0, col].set_title(f"{tile_name}  —  Amplification ratio  |XB|/|x|  ∝  ‖B‖")
        axes[0, col].set_ylabel("|XB|_max / |x|_max")
        axes[0, col].set_yscale("log")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=8)

        axes[1, col].set_title(f"{tile_name}  —  Amplification ratio  |DA|/|d|  ∝  ‖A‖")
        axes[1, col].set_ylabel("|DA|_max / |d|_max")
        axes[1, col].set_xlabel("step")
        axes[1, col].set_yscale("log")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend(fontsize=8)

    fig.suptitle("Bilinear amplification: stable vs collapse run", fontsize=12, y=1.00)
    fig.tight_layout()
    out = OUT_DIR / "comparison_amplification.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print("=== Bilinear collapse hypothesis verification ===\n")

    # Per-run trajectory plots
    for tag in ["seed42", "seed42_lrfloor001"]:
        d = load(tag)
        plot_run(d, tag, OUT_DIR / f"trajectory_{tag}.png")

    # Comparison plot
    comparison_plot()

    # Quantitative summary
    print("\n=== Quantitative comparison ===")
    print(f"{'run':<35} {'tile':<25} {'peak ‖A·B‖':>12} {'peak |XB|/|x|':>15} {'peak |DA|/|d|':>15}")
    print("-" * 110)

    for tag in ["seed42", "seed42_lrfloor001"]:
        d = load(tag)
        for tile_key, tname in [("first_tile", "L0.q"), ("last_tile", "L11.out.dense")]:
            nAB, _ = get_series(d, tile_key, "norm_AB")
            R_A, R_B = compute_amplification_ratios(d, tile_key)
            print(f"{tag:<35} {tname:<25} {nAB.max():>12.2f} {R_A.max():>15.2f} {R_B.max():>15.2f}")

    # Detailed look at L11.out around collapse
    print("\n=== Detailed L11.output.dense during collapse (seed42_lrfloor001) ===")
    d_c = load("seed42_lrfloor001")
    steps = np.array([r["step"] for r in d_c["last_tile"]["steps"]])
    nA, _ = get_series(d_c, "last_tile", "norm_A")
    nB, _ = get_series(d_c, "last_tile", "norm_B")
    nAB, _ = get_series(d_c, "last_tile", "norm_AB")
    xa, _ = get_series(d_c, "last_tile", "xa_abs_max")
    xb, _ = get_series(d_c, "last_tile", "xb_abs_max")
    da, _ = get_series(d_c, "last_tile", "da_abs_max")
    db, _ = get_series(d_c, "last_tile", "db_abs_max")

    # Detect explosion onset: first step where ‖A·B‖ > 100
    explode_idx = np.where(nAB > 100)[0]
    if len(explode_idx) > 0:
        idx = explode_idx[0]
        st = steps[idx]
        print(f"Explosion onset: step {st} (|A·B| first exceeds 100)")
        print(f"\n{'step':>6} {'‖A·B‖':>10} {'|x|_max':>10} {'|d|_max':>10} {'|XB|_max':>10} {'|DA|_max':>10}")
        print("-" * 70)
        for k in range(max(0, idx - 5), min(len(steps), idx + 10)):
            print(f"  {steps[k]:>4d} {nAB[k]:>10.3f} {xb[k]:>10.4f} {da[k]:>10.5f} {xa[k]:>10.3f} {db[k]:>10.4f}")

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

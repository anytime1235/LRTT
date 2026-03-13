"""Phase 1 sweep 결과 상세 분석 및 시각화.

bit별, lr_analog/lr_digital 별 final_loss heatmap, box plot,
문제점 진단 등.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

CSV_PATH = "./main_results/weight_update/squad/single/bit_lr_sweep_summary.csv"
OUT_DIR = "./main_results/weight_update/squad/single/analysis_plots"

def load_data(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["bits"] = int(r["bits"])
        r["dw_min"] = float(r["dw_min"])
        r["lr_analog"] = float(r["lr_analog"])
        r["lr_digital"] = float(r["lr_digital"])
        r["final_loss"] = float(r["final_loss"]) if r["final_loss"] not in ("", "None", None) else None
        r["elapsed_s"] = float(r["elapsed_s"])
    return rows


def get_unique_sorted(rows, key):
    vals = sorted(set(r[key] for r in rows))
    return vals


def plot_heatmaps(rows, out_dir):
    """Per-bit heatmap: lr_analog (y) × lr_digital (x) → final_loss."""
    bits_list = sorted(set(r["bits"] for r in rows))
    n_bits = len(bits_list)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # Global loss range for consistent colorbar
    all_losses = [r["final_loss"] for r in rows if r["final_loss"] is not None]
    vmin, vmax = min(all_losses), max(all_losses)

    for ax_idx, bits in enumerate(bits_list):
        ax = axes[ax_idx]
        bit_rows = [r for r in rows if r["bits"] == bits and r["final_loss"] is not None]

        lr_a_vals = sorted(set(r["lr_analog"] for r in bit_rows))
        lr_d_vals = sorted(set(r["lr_digital"] for r in bit_rows))

        # Build matrix
        mat = np.full((len(lr_a_vals), len(lr_d_vals)), np.nan)
        for r in bit_rows:
            i = lr_a_vals.index(r["lr_analog"])
            j = lr_d_vals.index(r["lr_digital"])
            mat[i, j] = r["final_loss"]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                       origin="lower")

        # Labels
        ax.set_xticks(range(len(lr_d_vals)))
        ax.set_xticklabels([f"{v:.1e}" for v in lr_d_vals], rotation=45, fontsize=8)
        ax.set_yticks(range(len(lr_a_vals)))
        ax.set_yticklabels([f"{v:.2e}" for v in lr_a_vals], fontsize=8)

        dw_min = 2.0 / (2 ** bits)
        ax.set_title(f"{bits}-bit (dw_min={dw_min:.2e})", fontsize=11, fontweight="bold")
        ax.set_xlabel("lr_digital")
        ax.set_ylabel("lr_analog")

        # Annotate values
        for i in range(len(lr_a_vals)):
            for j in range(len(lr_d_vals)):
                if not np.isnan(mat[i, j]):
                    color = "white" if mat[i, j] > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Final Loss (100 steps)")
    fig.suptitle("Phase 1 Sweep: Final Loss Heatmap per Bit Level\n(lr_analog × lr_digital grid, 100 steps)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "heatmap_all_bits.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_heatmaps_individual_scale(rows, out_dir):
    """Per-bit heatmap with individual color scale (per-bit normalized)."""
    bits_list = sorted(set(r["bits"] for r in rows))

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for ax_idx, bits in enumerate(bits_list):
        ax = axes[ax_idx]
        bit_rows = [r for r in rows if r["bits"] == bits and r["final_loss"] is not None]

        lr_a_vals = sorted(set(r["lr_analog"] for r in bit_rows))
        lr_d_vals = sorted(set(r["lr_digital"] for r in bit_rows))

        mat = np.full((len(lr_a_vals), len(lr_d_vals)), np.nan)
        for r in bit_rows:
            i = lr_a_vals.index(r["lr_analog"])
            j = lr_d_vals.index(r["lr_digital"])
            mat[i, j] = r["final_loss"]

        bit_vmin = np.nanmin(mat)
        bit_vmax = np.nanmax(mat)

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=bit_vmin, vmax=bit_vmax,
                       origin="lower")

        ax.set_xticks(range(len(lr_d_vals)))
        ax.set_xticklabels([f"{v:.1e}" for v in lr_d_vals], rotation=45, fontsize=8)
        ax.set_yticks(range(len(lr_a_vals)))
        ax.set_yticklabels([f"{v:.2e}" for v in lr_a_vals], fontsize=8)

        dw_min = 2.0 / (2 ** bits)
        ax.set_title(f"{bits}-bit (dw_min={dw_min:.2e})\nrange: [{bit_vmin:.2f}, {bit_vmax:.2f}]",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("lr_digital")
        ax.set_ylabel("lr_analog")

        for i in range(len(lr_a_vals)):
            for j in range(len(lr_d_vals)):
                if not np.isnan(mat[i, j]):
                    norm_val = (mat[i, j] - bit_vmin) / (bit_vmax - bit_vmin + 1e-9)
                    color = "white" if norm_val > 0.5 else "black"
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Phase 1 Sweep: Final Loss Heatmap (Individual Scale per Bit)\n"
                 "Green = low loss (good), Red = high loss (bad)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "heatmap_individual_scale.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_best_loss_by_bit(rows, out_dir):
    """Bar chart: best final_loss per bit level + scatter of all trials."""
    bits_list = sorted(set(r["bits"] for r in rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: Box plot
    bit_losses = {}
    for bits in bits_list:
        losses = [r["final_loss"] for r in rows if r["bits"] == bits and r["final_loss"] is not None]
        bit_losses[bits] = losses

    positions = range(len(bits_list))
    bp = ax1.boxplot([bit_losses[b] for b in bits_list], positions=positions,
                     widths=0.6, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bits_list)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(b) for b in bits_list])
    ax1.set_xlabel("Bit Level", fontsize=12)
    ax1.set_ylabel("Final Loss", fontsize=12)
    ax1.set_title("Loss Distribution by Bit Level", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add baseline (untrained BERT loss ≈ 6.0)
    ax1.axhline(y=6.0, color="red", linestyle="--", alpha=0.5, label="Untrained baseline (~6.0)")
    ax1.legend()

    # Right: Best loss per bit (bar) + all trials scatter
    best_losses = []
    mean_losses = []
    for bits in bits_list:
        losses = bit_losses[bits]
        best_losses.append(min(losses))
        mean_losses.append(np.mean(losses))

    x = np.arange(len(bits_list))
    width = 0.35
    bars1 = ax2.bar(x - width/2, best_losses, width, label="Best loss", color="forestgreen", alpha=0.8)
    bars2 = ax2.bar(x + width/2, mean_losses, width, label="Mean loss", color="steelblue", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(b) for b in bits_list])
    ax2.set_xlabel("Bit Level", fontsize=12)
    ax2.set_ylabel("Final Loss", fontsize=12)
    ax2.set_title("Best vs Mean Loss by Bit Level", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Annotate best values
    for bar, val in zip(bars1, best_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "best_loss_by_bit.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_lr_analog_sensitivity(rows, out_dir):
    """For each bit level: loss vs lr_analog, colored by lr_digital."""
    bits_list = sorted(set(r["bits"] for r in rows))

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for ax_idx, bits in enumerate(bits_list):
        ax = axes[ax_idx]
        bit_rows = [r for r in rows if r["bits"] == bits and r["final_loss"] is not None]

        lr_d_vals = sorted(set(r["lr_digital"] for r in bit_rows))
        colors_d = plt.cm.Set1(np.linspace(0, 0.8, len(lr_d_vals)))

        for d_idx, lr_d in enumerate(lr_d_vals):
            d_rows = [r for r in bit_rows if r["lr_digital"] == lr_d]
            d_rows.sort(key=lambda r: r["lr_analog"])
            xs = [r["lr_analog"] for r in d_rows]
            ys = [r["final_loss"] for r in d_rows]
            ax.plot(xs, ys, "o-", color=colors_d[d_idx], markersize=5,
                    label=f"lr_d={lr_d:.1e}")

        ax.set_xscale("log")
        dw_min = 2.0 / (2 ** bits)
        ax.set_title(f"{bits}-bit (dw_min={dw_min:.2e})", fontsize=11, fontweight="bold")
        ax.set_xlabel("lr_analog (log)")
        ax.set_ylabel("Final Loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Mark the dead zone threshold
        # For mean grad ~3e-4: lr_a needs > dw_min / 3e-4 for updates to fire
        threshold = dw_min / 3e-4
        ax.axvline(x=threshold, color="red", linestyle=":", alpha=0.6)
        ax.text(threshold, ax.get_ylim()[1] * 0.95, f"  dw/|g|={threshold:.2e}",
                fontsize=7, color="red", va="top")

    fig.suptitle("Loss vs lr_analog (by lr_digital)\n"
                 "Red dotted: dead zone threshold (dw_min / mean_grad)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "lr_analog_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_lr_digital_sensitivity(rows, out_dir):
    """For each bit level: loss vs lr_digital, colored by lr_analog."""
    bits_list = sorted(set(r["bits"] for r in rows))

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for ax_idx, bits in enumerate(bits_list):
        ax = axes[ax_idx]
        bit_rows = [r for r in rows if r["bits"] == bits and r["final_loss"] is not None]

        lr_a_vals = sorted(set(r["lr_analog"] for r in bit_rows))
        colors_a = plt.cm.tab10(np.linspace(0, 1, len(lr_a_vals)))

        for a_idx, lr_a in enumerate(lr_a_vals):
            a_rows = [r for r in bit_rows if r["lr_analog"] == lr_a]
            a_rows.sort(key=lambda r: r["lr_digital"])
            xs = [r["lr_digital"] for r in a_rows]
            ys = [r["final_loss"] for r in a_rows]
            ax.plot(xs, ys, "o-", color=colors_a[a_idx], markersize=5,
                    label=f"lr_a={lr_a:.2e}")

        ax.set_xscale("log")
        dw_min = 2.0 / (2 ** bits)
        ax.set_title(f"{bits}-bit (dw_min={dw_min:.2e})", fontsize=11, fontweight="bold")
        ax.set_xlabel("lr_digital (log)")
        ax.set_ylabel("Final Loss")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("Loss vs lr_digital (by lr_analog)\n"
                 "Shows digital LR sensitivity for each analog LR",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "lr_digital_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_dead_zone_analysis(rows, out_dir):
    """Analyze dead zone: plot loss vs (lr_analog × mean_grad / dw_min) ratio."""
    bits_list = sorted(set(r["bits"] for r in rows))

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    mean_grad = 3e-4  # approximate BERT QKV gradient magnitude

    for ax_idx, bits in enumerate(bits_list):
        ax = axes[ax_idx]
        bit_rows = [r for r in rows if r["bits"] == bits and r["final_loss"] is not None]
        dw_min = 2.0 / (2 ** bits)

        lr_d_vals = sorted(set(r["lr_digital"] for r in bit_rows))
        colors_d = plt.cm.Set1(np.linspace(0, 0.8, len(lr_d_vals)))

        for d_idx, lr_d in enumerate(lr_d_vals):
            d_rows = [r for r in bit_rows if r["lr_digital"] == lr_d]
            # Effective update ratio: lr_a * |grad| / dw_min
            xs = [r["lr_analog"] * mean_grad / dw_min for r in d_rows]
            ys = [r["final_loss"] for r in d_rows]
            ax.scatter(xs, ys, color=colors_d[d_idx], s=40, alpha=0.7,
                      label=f"lr_d={lr_d:.1e}")

        ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="ratio=1 (dead zone boundary)")
        ax.set_xscale("log")
        ax.set_title(f"{bits}-bit (dw_min={dw_min:.2e})", fontsize=11, fontweight="bold")
        ax.set_xlabel("lr_a × |grad| / dw_min (update ratio)")
        ax.set_ylabel("Final Loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Dead Zone Analysis: Loss vs Update Ratio (lr_a × |grad| / dw_min)\n"
                 "ratio < 1 → analog updates dead, ratio > 1 → analog updates active\n"
                 "Assumes mean |grad| ≈ 3e-4 for BERT QKV layers",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "dead_zone_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_landscape_3d(rows, out_dir):
    """Combined scatter: all bits on unified log(lr_analog) vs log(lr_digital) vs loss."""
    bits_list = sorted(set(r["bits"] for r in rows))

    fig, ax = plt.subplots(figsize=(14, 8))

    colors_b = plt.cm.tab10(np.linspace(0, 0.8, len(bits_list)))
    for b_idx, bits in enumerate(bits_list):
        bit_rows = [r for r in rows if r["bits"] == bits and r["final_loss"] is not None]
        best = min(bit_rows, key=lambda r: r["final_loss"])
        losses = [r["final_loss"] for r in bit_rows]

        # Scatter: x = bit level, y = loss
        jitter = np.random.normal(0, 0.15, len(losses))
        ax.scatter([bits + j for j in jitter], losses,
                  color=colors_b[b_idx], alpha=0.4, s=20, label=f"{bits}-bit")
        ax.scatter([bits], [best["final_loss"]], color=colors_b[b_idx],
                  s=150, marker="*", edgecolors="black", zorder=5)

    ax.axhline(y=6.0, color="red", linestyle="--", alpha=0.4, label="Baseline (~6.0)")
    ax.set_xlabel("Bit Level", fontsize=12)
    ax.set_ylabel("Final Loss", fontsize=12)
    ax.set_title("All Trials: Loss Distribution per Bit Level\n"
                 "(Stars = best per bit)", fontsize=13, fontweight="bold")
    ax.set_xticks(bits_list)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "all_trials_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_convergence_quality(rows, out_dir):
    """Fraction of trials that show meaningful learning (loss < 5.5) per bit."""
    bits_list = sorted(set(r["bits"] for r in rows))

    thresholds = [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: fraction below threshold per bit
    for thresh in thresholds:
        fracs = []
        for bits in bits_list:
            losses = [r["final_loss"] for r in rows
                     if r["bits"] == bits and r["final_loss"] is not None]
            frac = sum(1 for l in losses if l < thresh) / len(losses) if losses else 0
            fracs.append(frac)
        ax1.plot(bits_list, fracs, "o-", label=f"loss < {thresh:.1f}", markersize=6)

    ax1.set_xlabel("Bit Level", fontsize=12)
    ax1.set_ylabel("Fraction of Trials", fontsize=12)
    ax1.set_title("Fraction of Trials Below Loss Threshold\n"
                  "(Higher = more configs succeed)", fontsize=13, fontweight="bold")
    ax1.set_xticks(bits_list)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: Loss variance per bit
    means = []
    stds = []
    mins = []
    maxs = []
    for bits in bits_list:
        losses = [r["final_loss"] for r in rows
                 if r["bits"] == bits and r["final_loss"] is not None]
        means.append(np.mean(losses))
        stds.append(np.std(losses))
        mins.append(min(losses))
        maxs.append(max(losses))

    x = np.arange(len(bits_list))
    ax2.bar(x, stds, color="coral", alpha=0.7, label="Std Dev")
    ax2.plot(x, [mx - mn for mx, mn in zip(maxs, mins)], "ks-",
            label="Max-Min range", markersize=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(b) for b in bits_list])
    ax2.set_xlabel("Bit Level", fontsize=12)
    ax2.set_ylabel("Loss Variation", fontsize=12)
    ax2.set_title("Loss Sensitivity to LR Choice\n"
                  "(Higher = more sensitive to hyperparameters)", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "convergence_quality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_diagnosis(rows):
    """Print detailed diagnosis text."""
    bits_list = sorted(set(r["bits"] for r in rows))
    mean_grad = 3e-4

    print("\n" + "=" * 80)
    print("PHASE 1 SWEEP DIAGNOSIS")
    print("=" * 80)

    for bits in bits_list:
        bit_rows = [r for r in rows if r["bits"] == bits and r["final_loss"] is not None]
        dw_min = 2.0 / (2 ** bits)
        losses = [r["final_loss"] for r in bit_rows]
        best = min(bit_rows, key=lambda r: r["final_loss"])
        worst = max(bit_rows, key=lambda r: r["final_loss"])

        print(f"\n{'─'*60}")
        print(f"  {bits}-bit  |  dw_min = {dw_min:.6e}")
        print(f"{'─'*60}")
        print(f"  Loss range:  [{min(losses):.4f}, {max(losses):.4f}]")
        print(f"  Mean ± std:  {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"  Best:  lr_a={best['lr_analog']:.4e}, lr_d={best['lr_digital']:.4e} → loss={best['final_loss']:.4f}")
        print(f"  Worst: lr_a={worst['lr_analog']:.4e}, lr_d={worst['lr_digital']:.4e} → loss={worst['final_loss']:.4f}")

        # Dead zone analysis
        lr_a_vals = sorted(set(r["lr_analog"] for r in bit_rows))
        threshold_lr_a = dw_min / mean_grad
        n_above = sum(1 for lr in lr_a_vals if lr * mean_grad > dw_min)
        n_total = len(lr_a_vals)
        print(f"  Dead zone threshold (lr_a > dw_min/|grad|): lr_a > {threshold_lr_a:.4e}")
        print(f"  lr_a values above threshold: {n_above}/{n_total}")

        # Check if lr_analog matters
        # Group by lr_digital and check variance across lr_analog
        lr_d_vals = sorted(set(r["lr_digital"] for r in bit_rows))
        print(f"  lr_analog sensitivity:")
        for lr_d in lr_d_vals:
            d_losses = [r["final_loss"] for r in bit_rows if r["lr_digital"] == lr_d]
            print(f"    lr_d={lr_d:.1e}: loss range [{min(d_losses):.3f}, {max(d_losses):.3f}], "
                  f"spread={max(d_losses)-min(d_losses):.3f}")

        # Diagnosis
        spread = max(losses) - min(losses)
        frac_below_5 = sum(1 for l in losses if l < 5.0) / len(losses)
        frac_below_4 = sum(1 for l in losses if l < 4.0) / len(losses)

        issues = []
        if min(losses) > 5.5:
            issues.append("CRITICAL: Best loss > 5.5 — almost no learning observed")
        if spread < 0.2:
            issues.append(f"WARNING: Very low loss spread ({spread:.3f}) — lr_analog may have no effect")
        if frac_below_5 < 0.1:
            issues.append(f"WARNING: Only {frac_below_5*100:.0f}% of trials achieve loss < 5.0")
        if n_above < n_total // 2:
            issues.append(f"WARNING: Most lr_a values ({n_total - n_above}/{n_total}) are in dead zone")

        # Check if only lr_digital matters (32-bit pattern)
        lr_a_variance = np.std([r["final_loss"] for r in bit_rows])
        for lr_d in lr_d_vals:
            d_losses = [r["final_loss"] for r in bit_rows if r["lr_digital"] == lr_d]
            if max(d_losses) - min(d_losses) < 0.01 and len(d_losses) > 1:
                issues.append(f"NOTE: For lr_d={lr_d:.1e}, lr_analog has zero effect (spread < 0.01)")

        if issues:
            print(f"  Issues:")
            for issue in issues:
                print(f"    ⚠ {issue}")
        else:
            print(f"  Status: OK — meaningful learning across multiple configs")

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Bit levels ranked by best loss:")
    bit_best = [(bits, min(r["final_loss"] for r in rows
                           if r["bits"] == bits and r["final_loss"] is not None))
                for bits in bits_list]
    bit_best.sort(key=lambda x: x[1])
    for rank, (bits, loss) in enumerate(bit_best, 1):
        status = ""
        if loss > 5.5:
            status = "  ← NOT LEARNING"
        elif loss > 4.5:
            status = "  ← BARELY LEARNING"
        print(f"    {rank}. {bits:>2}-bit: {loss:.4f}{status}")

    print(f"\n  Key observations:")
    # 4,6-bit
    low_bit_best = min(r["final_loss"] for r in rows if r["bits"] in (4, 6))
    if low_bit_best > 5.5:
        print(f"    - 4-bit and 6-bit: NO meaningful learning (best={low_bit_best:.3f})")
        print(f"      → dw_min too large relative to lr range, most updates in dead zone")
        print(f"      → Need much higher lr_analog or re-evaluate analog feasibility at these precisions")

    # 32-bit
    b32_rows = [r for r in rows if r["bits"] == 32]
    b32_lr_d_effect = {}
    for r in b32_rows:
        key = r["lr_digital"]
        if key not in b32_lr_d_effect:
            b32_lr_d_effect[key] = []
        b32_lr_d_effect[key].append(r["final_loss"])
    b32_uniform = all(max(v) - min(v) < 0.01 for v in b32_lr_d_effect.values())
    if b32_uniform:
        print(f"    - 32-bit: lr_analog has ZERO effect (dw_min={4.66e-10:.2e} is negligible)")
        print(f"      → Only lr_digital determines loss, analog updates are effectively zero")
        print(f"      → This is expected: 32-bit analog is essentially digital-only")

    # Optimal range
    good_bits = [b for b, l in bit_best if l < 3.0]
    if good_bits:
        print(f"    - Best performing bit levels: {good_bits}")
        print(f"      → Sweet spot where dw_min balances update granularity and learning rate range")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_data(CSV_PATH)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    print_diagnosis(rows)

    print(f"\nGenerating plots in {OUT_DIR}...")
    plot_heatmaps(rows, OUT_DIR)
    plot_heatmaps_individual_scale(rows, OUT_DIR)
    plot_best_loss_by_bit(rows, OUT_DIR)
    plot_lr_analog_sensitivity(rows, OUT_DIR)
    plot_lr_digital_sensitivity(rows, OUT_DIR)
    plot_dead_zone_analysis(rows, OUT_DIR)
    plot_loss_landscape_3d(rows, OUT_DIR)
    plot_convergence_quality(rows, OUT_DIR)

    print(f"\nDone! All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

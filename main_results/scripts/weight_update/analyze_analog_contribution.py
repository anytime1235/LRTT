"""Analog contribution analysis: compare each bit level to 32-bit digital-only baseline.

32-bit에서 lr_analog 효과 = 0이므로, 32-bit 결과 = digital-only baseline.
각 bit에서 같은 lr_digital일 때 loss가 32-bit보다 낮으면 → analog가 학습에 기여
각 bit에서 같은 lr_digital일 때 loss가 32-bit보다 높으면 → analog가 학습을 방해
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    return rows


def main():
    rows = load_data(CSV_PATH)
    bits_list = sorted(set(r["bits"] for r in rows))
    lr_d_vals = sorted(set(r["lr_digital"] for r in rows))

    # ── 1. Build 32-bit baseline (digital-only) ──
    # 32-bit: lr_analog has no effect, so take mean per lr_digital
    b32_baseline = {}
    for lr_d in lr_d_vals:
        losses = [r["final_loss"] for r in rows
                  if r["bits"] == 32 and r["lr_digital"] == lr_d and r["final_loss"] is not None]
        b32_baseline[lr_d] = np.mean(losses)

    print("=" * 80)
    print("32-bit DIGITAL-ONLY BASELINE (lr_analog has zero effect)")
    print("=" * 80)
    for lr_d in lr_d_vals:
        print(f"  lr_d={lr_d:.4e} → loss={b32_baseline[lr_d]:.4f}")

    # ── 2. Per-bit: compare best loss at each lr_digital to 32-bit baseline ──
    print(f"\n{'='*80}")
    print("ANALOG CONTRIBUTION: Δloss = bit_loss - baseline_32bit")
    print("  Δ < 0 → analog HELPS (lower loss = better)")
    print("  Δ > 0 → analog HURTS (higher loss = worse)")
    print("  Δ ≈ 0 → analog has no effect")
    print(f"{'='*80}")

    # Table header
    print(f"\n{'Bit':>4} | {'lr_digital':>10} | {'Best loss':>10} | {'32-bit baseline':>15} | {'Δloss':>8} | {'Verdict':>15}")
    print("-" * 80)

    contribution_data = {}  # (bits, lr_d) -> (best_loss, baseline, delta)

    for bits in bits_list:
        if bits == 32:
            continue
        for lr_d in lr_d_vals:
            bit_rows = [r for r in rows
                        if r["bits"] == bits and r["lr_digital"] == lr_d and r["final_loss"] is not None]
            if not bit_rows:
                continue
            best = min(r["final_loss"] for r in bit_rows)
            baseline = b32_baseline[lr_d]
            delta = best - baseline

            if delta < -0.5:
                verdict = "HELPS"
            elif delta < -0.1:
                verdict = "helps (small)"
            elif delta < 0.1:
                verdict = "neutral"
            elif delta < 0.5:
                verdict = "hurts (small)"
            else:
                verdict = "HURTS"

            contribution_data[(bits, lr_d)] = (best, baseline, delta, verdict)
            print(f"{bits:>4} | {lr_d:>10.4e} | {best:>10.4f} | {baseline:>15.4f} | {delta:>+8.3f} | {verdict:>15}")

    # ── 3. Summary per bit level ──
    print(f"\n{'='*80}")
    print("SUMMARY: Best overall loss per bit vs 32-bit digital-only best")
    print(f"{'='*80}")
    baseline_best = min(b32_baseline.values())
    print(f"  32-bit digital-only best: {baseline_best:.4f} (at lr_d=0.01)")
    print()

    for bits in bits_list:
        if bits == 32:
            continue
        bit_losses = [r["final_loss"] for r in rows
                      if r["bits"] == bits and r["final_loss"] is not None]
        best = min(bit_losses)
        best_row = min([r for r in rows if r["bits"] == bits], key=lambda r: r["final_loss"])
        delta = best - baseline_best

        if delta < -0.5:
            marker = "✓ ANALOG HELPS"
        elif delta < 0:
            marker = "~ marginal help"
        elif delta < 0.5:
            marker = "✗ analog neutral/hurts"
        else:
            marker = "✗✗ ANALOG HURTS"

        print(f"  {bits:>2}-bit: best={best:.4f}  Δ={delta:>+7.3f}  {marker}")
        print(f"         (lr_a={best_row['lr_analog']:.4e}, lr_d={best_row['lr_digital']:.4e})")

    # ── 4. Plot: Analog contribution visualization ──
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot A: Best loss per bit with 32-bit baseline
    ax = axes[0]
    non32_bits = [b for b in bits_list if b != 32]
    best_per_bit = []
    for bits in non32_bits:
        best = min(r["final_loss"] for r in rows
                   if r["bits"] == bits and r["final_loss"] is not None)
        best_per_bit.append(best)

    x = np.arange(len(non32_bits))
    colors = ['red' if b > baseline_best else 'green' for b in best_per_bit]
    bars = ax.bar(x, best_per_bit, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=baseline_best, color='blue', linestyle='--', linewidth=2,
               label=f'32-bit digital-only = {baseline_best:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in non32_bits])
    ax.set_xlabel("Bit Level", fontsize=12)
    ax.set_ylabel("Best Final Loss", fontsize=12)
    ax.set_title("Best Loss per Bit vs Digital-Only Baseline\n"
                 "Green = analog helps, Red = analog hurts", fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val, bits in zip(bars, best_per_bit, non32_bits):
        delta = val - baseline_best
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}\n(Δ{delta:+.2f})", ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    # Plot B: Per lr_digital comparison (grouped bar)
    ax = axes[1]
    n_lr_d = len(lr_d_vals)
    n_bits = len(non32_bits)
    width = 0.8 / (n_bits + 1)

    # 32-bit baseline bars
    x_base = np.arange(n_lr_d)
    for b_idx, bits in enumerate(non32_bits):
        best_per_lrd = []
        for lr_d in lr_d_vals:
            bit_rows = [r for r in rows
                        if r["bits"] == bits and r["lr_digital"] == lr_d and r["final_loss"] is not None]
            best_per_lrd.append(min(r["final_loss"] for r in bit_rows))
        offset = (b_idx - n_bits/2) * width
        color = plt.cm.tab10(b_idx / n_bits)
        ax.bar(x_base + offset, best_per_lrd, width, label=f'{bits}-bit', color=color, alpha=0.7)

    # 32-bit baseline
    baseline_vals = [b32_baseline[lr_d] for lr_d in lr_d_vals]
    ax.plot(x_base, baseline_vals, 'k*-', markersize=15, linewidth=2, label='32-bit (digital-only)')

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{v:.1e}" for v in lr_d_vals])
    ax.set_xlabel("lr_digital", fontsize=12)
    ax.set_ylabel("Best Loss (across lr_analog)", fontsize=12)
    ax.set_title("Best Loss per lr_digital\n(Black stars = digital-only baseline)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    # Plot C: Delta (improvement over digital-only) heatmap
    ax = axes[2]
    delta_mat = np.full((len(non32_bits), len(lr_d_vals)), np.nan)
    for b_idx, bits in enumerate(non32_bits):
        for d_idx, lr_d in enumerate(lr_d_vals):
            key = (bits, lr_d)
            if key in contribution_data:
                delta_mat[b_idx, d_idx] = contribution_data[key][2]  # delta

    im = ax.imshow(delta_mat, aspect='auto', cmap='RdYlGn_r',
                   vmin=-2.5, vmax=2.5, origin='lower')
    ax.set_xticks(range(len(lr_d_vals)))
    ax.set_xticklabels([f"{v:.1e}" for v in lr_d_vals], fontsize=9)
    ax.set_yticks(range(len(non32_bits)))
    ax.set_yticklabels([str(b) for b in non32_bits], fontsize=10)
    ax.set_xlabel("lr_digital", fontsize=12)
    ax.set_ylabel("Bit Level", fontsize=12)
    ax.set_title("Δloss vs 32-bit baseline\n(Green=analog helps, Red=analog hurts)", fontsize=12, fontweight='bold')

    for i in range(len(non32_bits)):
        for j in range(len(lr_d_vals)):
            if not np.isnan(delta_mat[i, j]):
                val = delta_mat[i, j]
                color = 'white' if abs(val) > 1.5 else 'black'
                ax.text(j, i, f"{val:+.2f}", ha='center', va='center', fontsize=9,
                        fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Δloss (negative = analog helps)')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "analog_contribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── 5. dw_min destructiveness analysis ──
    print(f"\n{'='*80}")
    print("dw_min IMPACT ANALYSIS")
    print("=" * 80)
    print("\nWhen analog update fires, it changes weight by ±dw_min.")
    print("If dw_min is too large → destructive updates on pretrained weights.")
    print("If dw_min is too small → updates negligible, no learning.\n")

    for bits in bits_list:
        dw_min = 2.0 / (2 ** bits)
        # typical pretrained BERT weight magnitude
        typical_w = 0.02  # rough estimate for QKV layers
        pct_change = (dw_min / typical_w) * 100
        status = ""
        if pct_change > 100:
            status = "DESTRUCTIVE (single update > weight magnitude)"
        elif pct_change > 10:
            status = "LARGE (may destabilize)"
        elif pct_change > 1:
            status = "MODERATE (good range)"
        elif pct_change > 0.01:
            status = "SMALL (fine-grained)"
        else:
            status = "NEGLIGIBLE (no effect)"
        print(f"  {bits:>2}-bit: dw_min={dw_min:.6e}  "
              f"|Δw/w|≈{pct_change:>8.2f}%  → {status}")

    # ── 6. Effective update count analysis ──
    print(f"\n{'='*80}")
    print("EFFECTIVE UPDATE RATE (estimated)")
    print("=" * 80)
    print("ConstantStepDevice with BL=31: expected updates per weight per step")
    print("≈ BL × P(pulse_input) × P(pulse_grad)")
    print("≈ BL × min(1, x/BL) × min(1, lr*|g|*BL/S) where S is scaling\n")

    # Simpler: for constant step, when lr*|grad| >> dw_min, ~all BL slots fire
    # when lr*|grad| << dw_min, almost no slots fire
    mean_grad = 3e-4
    max_grad = 5e-2
    bl = 31

    for bits in bits_list:
        dw_min = 2.0 / (2 ** bits)
        bit_rows = [r for r in rows if r["bits"] == bits and r["final_loss"] is not None]
        best_row = min(bit_rows, key=lambda r: r["final_loss"])
        lr_a = best_row["lr_analog"]

        # Rough estimate: fraction of weight elements that get updated
        # An element updates if lr_a * |grad_i| generates pulses
        # With BL=31, the pulse count ~ round(BL * lr_a * |grad_i| / some_scale)
        # For ConstantStep, update magnitude is always dw_min

        ratio_mean = lr_a * mean_grad / dw_min
        ratio_max = lr_a * max_grad / dw_min

        print(f"  {bits:>2}-bit: best lr_a={lr_a:.4e}, dw_min={dw_min:.2e}")
        print(f"    lr_a × mean|g| / dw_min = {ratio_mean:.4f} "
              f"({'active' if ratio_mean > 0.1 else 'mostly dead'})")
        print(f"    lr_a × max|g|  / dw_min = {ratio_max:.4f} "
              f"({'active' if ratio_max > 0.1 else 'mostly dead'})")
        print()


if __name__ == "__main__":
    main()

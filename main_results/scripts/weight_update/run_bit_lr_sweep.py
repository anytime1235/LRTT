"""Sweep: 100 runs across 3 bit levels × (lr_analog, lr_digital) grid.

BERT QKVO gradient magnitudes: mean ≈ 3e-4, max ≈ 5e-2.
For analog update to fire, need: lr × |grad| > dw_min
  → lr > dw_min / |grad|

Per-bit lr_analog ranges (for mean grad ≈ 3e-4):
  4-bit  (dw_min=0.125):    lr_needed ≈ 417   → sweep [1, 1000]
  10-bit (dw_min=0.00195):  lr_needed ≈ 6.5   → sweep [0.02, 50]
  16-bit (dw_min=3.05e-5):  lr_needed ≈ 0.1   → sweep [3e-4, 1]

Strategy: base_lr (at 10-bit ref), actual lr = base × (dw_min / ref_dw).
  This keeps dead_zone = ref_dw / base_lr constant across bits.

Factors:
  - bit levels: 4, 10, 16  (→ dw_min = 2.0 / 2^bits)
  - lr_analog: base_lr × (dw_min / ref_dw_min)
  - lr_digital: independent (classifier + LayerNorm lr)

Fixed:
  - desired_bl = 31, IO perfect, ConstantStepDevice, exclude-ffn (QKVO)
  - steps = 100, batch_size = 8, seed = 42

Usage:
  python run_bit_lr_sweep.py [--steps 100] [--dry-run]
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time

# ── Config ──────────────────────────────────────────────────────────
BITS_LIST = [4, 10, 16]
REF_DW = 2.0 / (2 ** 10)  # 10-bit reference dw_min ≈ 0.00195

# BERT QKVO per-element gradient: mean ≈ 3e-4, p99 ≈ 1e-2, max ≈ 0.05-0.1
# For update at a (i,j) position: need lr × |g_ij| > dw_min
# Most g_ij ≈ 1e-4, some outliers ≈ 1e-2.
# At 10-bit (dw_min=0.00195): lr=0.2 → only top ~5% update; lr=20 → most update but outliers saturate
# Strategy: sweep lr per bit level directly (NOT proportional scaling)
# because the gradient distribution is fixed regardless of dw_min.

# Verified: 10-bit lr_a=0.1 works (L0: 5.83→3.75, cos≈0.3, WZR≈50-70%)
#           10-bit lr_a=0.3 diverges; lr_a=7.0 saturates immediately
# Scale proportionally to dw_min for other bit levels:
#  4-bit  (64× dw_min): lr range = 10-bit range × 64
# 16-bit  (1/64× dw_min): lr range = 10-bit range / 64
LR_A_PER_BIT = {
    4:  [1.0, 3.0, 6.0, 12.0, 25.0, 50.0],
    10: [0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    16: [0.0003, 0.001, 0.002, 0.004, 0.008, 0.015],
}
LR_A_EXTRA_PER_BIT = {
    4:  [0.5, 100.0],
    10: [0.01, 2.0],
    16: [0.0001, 0.03],
}

# lr_digital: independent of bit level (drives classifier + LayerNorm)
LR_D_MAIN = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]  # 5 values
LR_D_EXTREME = [1e-4, 1e-1]

SCRIPT = os.path.join(os.path.dirname(__file__), "diag_weight_update_bert_v2.py")

# ── Build config list ───────────────────────────────────────────────

def build_configs():
    configs = []
    for bits in BITS_LIST:
        dw = 2.0 / (2 ** bits)
        # Main grid: 6 lr_a × 5 lr_d = 30 per bit
        for lr_a in LR_A_PER_BIT[bits]:
            for ld in LR_D_MAIN:
                configs.append({
                    "bits": bits, "dw_min": dw,
                    "lr_analog": lr_a,
                    "lr_digital": ld,
                })
        # Extra lr_a probes (with mid lr_d=3e-3): 2 per bit
        for lr_a in LR_A_EXTRA_PER_BIT[bits]:
            configs.append({
                "bits": bits, "dw_min": dw,
                "lr_analog": lr_a,
                "lr_digital": 3e-3,
            })

    # Extreme lr_digital probes for 4-bit and 16-bit: 2 × 2 = 4
    for bits in [4, 16]:
        dw = 2.0 / (2 ** bits)
        mid_lr_a = LR_A_PER_BIT[bits][2]  # 3rd value as mid
        for ld in LR_D_EXTREME:
            configs.append({
                "bits": bits, "dw_min": dw,
                "lr_analog": mid_lr_a,
                "lr_digital": ld,
            })

    return configs


def run_one(cfg, idx, total, steps, out_base, dry_run=False):
    bits = cfg["bits"]
    tag = f"b{bits}_lrA{cfg['lr_analog']:.0e}_lrD{cfg['lr_digital']:.0e}"
    tag = tag.replace("+", "")

    cmd = [
        sys.executable, SCRIPT,
        "--mode", "single",
        "--forward-perfect", "--backward-perfect",
        "--exclude-ffn",
        "--dw-min", f"{cfg['dw_min']:.10f}",
        "--lr", f"{cfg['lr_analog']:.10f}",
        "--digital-lr", f"{cfg['lr_digital']:.10f}",
        "--steps", str(steps),
        "--tag", tag,
        "--overwrite",
        "--no-trace",  # skip per-step weight tracing for speed
        "--eval-loss",  # track eval loss per step
    ]

    dead_zone = cfg["dw_min"] / cfg["lr_analog"]
    print(f"\n[{idx+1}/{total}] {bits}-bit | "
          f"lr_a={cfg['lr_analog']:.6e} lr_d={cfg['lr_digital']:.0e} "
          f"dead_zone={dead_zone:.4f}")

    if dry_run:
        print(f"  CMD: {' '.join(cmd)}")
        return None

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(SCRIPT) or ".")
    elapsed = time.time() - t0

    # Extract final loss from stdout
    final_loss = None
    for line in result.stdout.split("\n"):
        if "loss=" in line:
            try:
                final_loss = float(line.split("loss=")[-1].split("]")[0].split(",")[0])
            except (ValueError, IndexError):
                pass

    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"  {status} ({elapsed:.1f}s) final_loss={final_loss}")

    if result.returncode != 0:
        # Print last 10 lines of stderr
        err_lines = result.stderr.strip().split("\n")
        for line in err_lines[-10:]:
            print(f"  ERR: {line}")

    return {
        **cfg,
        "tag": tag,
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "final_loss": final_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str,
                        default="./main_results/weight_update/squad")
    args = parser.parse_args()

    configs = build_configs()
    print(f"=== Bit-LR Sweep: {len(configs)} runs, {args.steps} steps each ===")
    print(f"Fixed: desired_bl=31, IO=perfect, device=ConstantStep, QKVO only")
    for bits in BITS_LIST:
        print(f"  {bits}-bit lr_a: {LR_A_PER_BIT[bits]} + extras {LR_A_EXTRA_PER_BIT[bits]}")
    print(f"  lr_d: {LR_D_MAIN} + extras {LR_D_EXTREME}")

    results = []
    t_total = time.time()

    for i, cfg in enumerate(configs):
        row = run_one(cfg, i, len(configs), args.steps, args.out_dir, args.dry_run)
        if row:
            results.append(row)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(results)}/{len(configs)} runs in {elapsed_total/60:.1f} min")

    # Save summary CSV
    if results:
        csv_path = os.path.join(args.out_dir, "single", "bit_lr_sweep_summary.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Summary saved: {csv_path}")


if __name__ == "__main__":
    main()

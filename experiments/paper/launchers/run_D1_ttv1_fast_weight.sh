#!/bin/bash
# D1: TTv1 fast tile weight distribution measurement
# Measure actual fast tile weight range during training (128 steps)
# w_max_fast=1.0 (default) so no clipping — pure measurement
set -uo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

COMMON="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016"
DIAG="--diag-update-exact --diag-layer-set 0,5,11"
D1="$COMMON --max-steps 128 --batch-size 12 --grad-accum-steps 4 $DIAG --log-every 1"

TTv1_BASE="--method ttv1 --ttv1-mode residual_lane --n-bits 14 --n-bits-slow 10 --gamma 1.0 --with-reset-prob 1.0 --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 4"

RESULTS="results/paper/diag_wmax_sweep"
mkdir -p "$RESULTS"

echo "============================================================"
echo "  D1: TTv1 fast tile weight distribution (128 steps)"
echo "  fast_lr=0.1, transfer_every=4, reset_prob=1.0"
echo "  Start: $(date)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py \
    $D1 $TTv1_BASE \
    --output-dir "$RESULTS/D1_ttv1_fast_weight_baseline"

echo "[D1] DONE $(date)"

# Print fast tile weight stats
$PYTHON << 'PYEOF'
import json

base = "results/paper/diag_wmax_sweep/D1_ttv1_fast_weight_baseline"
d = json.load(open(f"{base}/update_diagnostics_summary.json"))

print("\n=== TTv1 Fast Tile Weight Distribution (128 steps) ===\n")
print(f"{'tile':<55} | {'fast_abs_max':>11} | {'fast_p99':>9} | {'fast_p50':>9} | {'fast_std':>9}")
print("-" * 105)
for name, t in d.items():
    print(f"{name:<55} | {t.get('max_fast_w_abs_max',0):>11.6f} | "
          f"{t.get('mean_fast_w_abs_p99',0):>9.6f} | "
          f"{t.get('mean_fast_w_abs_p50',0):>9.6f} | "
          f"{t.get('mean_fast_w_std',0):>9.6f}")

# Recommend w_max_fast
all_max = max(t.get('max_fast_w_abs_max', 0) for t in d.values())
print(f"\n>>> Max fast tile |w| across all tiles: {all_max:.6f}")
print(f">>> Suggested w_max_fast: {all_max * 5:.4f} (5x headroom)")
print(f">>> dw_min at suggested w_max_fast: {2 * all_max * 5 / (2**14):.2e}")
print(f">>> mu improvement factor: {1.0 / (all_max * 5):.1f}x")

PYEOF

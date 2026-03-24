#!/bin/bash
# D1: TTv1 fast tile sub-pulse with w_max_fast=0.25
# Sweep fast tile bits: 8, 10, 12, 14 (slow=10b fixed)
# QKV+O (attention) layers, 128 steps
set -uo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

run_one() {
    local TAG="$1"
    shift
    echo ""
    echo "[D1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py "$@"
    local RC=$?
    $PYTHON -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache()" 2>/dev/null
    if [ $RC -ne 0 ]; then
        echo "[D1] FAIL  $TAG (exit=$RC) $(date)"
    else
        echo "[D1] DONE  $TAG $(date)"
    fi
    sleep 3
}

COMMON="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016"
DIAG="--diag-update-exact --diag-layer-set 0,5,11"
D1="$COMMON --max-steps 128 --batch-size 12 --grad-accum-steps 4 $DIAG --log-every 1"

TTv1_BASE="--method ttv1 --ttv1-mode residual_lane --n-bits-slow 10 --gamma 1.0 --with-reset-prob 1.0 --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 4 --w-max-fast 0.25"

RESULTS="results/paper/diag_D1_ttv1_fast_wmax025"
mkdir -p "$RESULTS"

echo "============================================================"
echo "  D1: TTv1 fast tile sub-pulse (w_max_fast=0.25)"
echo "  fast bits: 8, 10, 12, 14 | slow: 10b | QKV+O"
echo "  dw_min_fast = 2*0.25/2^n = 0.5/2^n"
echo "  Start: $(date)"
echo "============================================================"

for FAST_BITS in 8 10 12 14; do
    run_one "ttv1_fast${FAST_BITS}b_wmax025" $D1 $TTv1_BASE \
        --n-bits $FAST_BITS \
        --output-dir "$RESULTS/ttv1_fast${FAST_BITS}b_wmax025"
done

echo ""
echo "============================================================"
echo "  D1 COMPLETE: $(date)"
echo "============================================================"

# Summary + plot
$PYTHON << 'PYEOF'
import json, os, glob
import numpy as np

base = "results/paper/diag_D1_ttv1_fast_wmax025"

print("\n=== D1 TTv1 Fast Tile Sub-Pulse Summary (w_max_fast=0.25) ===\n")
print(f"{'tag':<35} | {'dw_min':>10} | {'cos_sim':>8} | {'zero_frac':>9} | {'frac_mu<1':>9} | {'mu_p50':>10} | {'mu_mean':>10} | {'recovery':>8}")
print("-" * 120)

for d in sorted(glob.glob(f"{base}/ttv1_fast*")):
    tag = os.path.basename(d)
    spath = f"{d}/update_diagnostics_summary.json"
    cpath = f"{d}/config.json"
    if not os.path.exists(spath):
        print(f"{tag:<35} | NO SUMMARY")
        continue
    data = json.load(open(spath))
    config = json.load(open(cpath))
    tiles = list(data.values())
    if not tiles:
        continue
    avg = lambda k: sum(t.get(k, 0) for t in tiles) / len(tiles)
    dw_min = config.get("dw_min", 0)
    print(f"{tag:<35} | {dw_min:>10.2e} | {avg('mean_cosine_sim'):>8.4f} | "
          f"{avg('mean_zero_frac'):>9.4f} | {avg('mean_eff_frac_mu_lt_1'):>9.4f} | "
          f"{avg('mean_eff_mu_p50'):>10.6f} | {avg('mean_eff_mu_mean'):>10.6f} | "
          f"{avg('final_recovery_ratio'):>8.2f}")

PYEOF

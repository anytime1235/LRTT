#!/bin/bash
# D1 w_max sweep: omega=0 (no scaling), w_max={0.1, 0.01, 0.001}
# Track actual weight distribution + sub-pulse metrics
# w_min = -w_max (symmetric, already handled in code)
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

RESULTS="results/paper/diag_wmax_sweep"
mkdir -p "$RESULTS"

echo "============================================================"
echo "  D1: w_max sweep (128 steps, omega=0, bs=12 acc=4)"
echo "  w_min = -w_max (symmetric)"
echo "  Start: $(date)"
echo "============================================================"

for WMAX in 0.1 0.01 0.001; do
    run_one "single_rpu_14b_omega0_wmax${WMAX}" $D1 \
        --method single_rpu --pulse-type stochastic --n-bits 14 \
        --w-max $WMAX --omega 0.0 \
        --output-dir "$RESULTS/D1_single_rpu_14b_omega0_wmax${WMAX}"
done

echo ""
echo "============================================================"
echo "  D1 COMPLETE: $(date)"
echo "============================================================"

# Summary
$PYTHON << 'PYEOF'
import json, os, glob
import numpy as np

base = "results/paper/diag_wmax_sweep"

print("\n=== D1 Weight Distribution + Sub-Pulse Summary ===\n")
print(f"{'tag':<45} | {'w_abs_max':>9} | {'w_util':>7} | {'clip%':>7} | {'cos_sim':>8} | {'frac_mu<1':>9} | {'mu_p50':>10}")
print("-" * 110)

for d in sorted(glob.glob(f"{base}/D1_*")):
    tag = os.path.basename(d)
    spath = f"{d}/update_diagnostics_summary.json"
    if not os.path.exists(spath):
        print(f"{tag:<45} | NO SUMMARY")
        continue
    data = json.load(open(spath))
    tiles = list(data.values())
    if not tiles:
        continue
    avg = lambda k: sum(t.get(k, 0) for t in tiles) / len(tiles)
    mx = lambda k: max(t.get(k, 0) for t in tiles)
    print(f"{tag:<45} | {mx('max_w_abs_max'):>9.4f} | {avg('mean_w_utilization'):>7.4f} | "
          f"{mx('max_w_clipped_frac')*100:>6.2f}% | {avg('mean_cosine_sim'):>8.4f} | "
          f"{avg('mean_eff_frac_mu_lt_1'):>9.4f} | {avg('mean_eff_mu_p50'):>10.6f}")

PYEOF

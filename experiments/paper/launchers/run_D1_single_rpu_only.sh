#!/bin/bash
# D1 single_rpu only: 8,10,12,14,16b
# Per-element p_i/q_j histograms enabled
set -uo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

BATCH=12
GACC=4

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
DIAG_LAYERS="--diag-layer-set 0,5,11"
D1_DIAG_STEPS="1,2,4,8,16,32,64,128"
D1="$COMMON --max-steps 128 --batch-size $BATCH --grad-accum-steps $GACC \
  --diag-update-exact $DIAG_LAYERS --diag-at-steps $D1_DIAG_STEPS --log-every 8"

D1_RESULTS="results/paper/diag_D1_sweep"
mkdir -p "$D1_RESULTS"

echo "============================================================"
echo "  D1 single_rpu only (with per-element p_i/q_j histograms)"
echo "  bits: 8,10,12,14,16"
echo "  diag steps: $D1_DIAG_STEPS"
echo "  Start: $(date)"
echo "============================================================"

for BITS in 8 10 12 14 16; do
    run_one "single_rpu_${BITS}b" $D1 \
        --method single_rpu --n-bits $BITS \
        --output-dir "$D1_RESULTS/single_rpu_${BITS}b"
done

echo ""
echo "============================================================"
echo "  D1 single_rpu COMPLETE: $(date)"
echo "============================================================"

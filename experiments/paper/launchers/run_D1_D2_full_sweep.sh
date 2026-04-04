#!/bin/bash
# D1 + D2 Full Sweep: single_rpu + ttv1, bits={8,10,12,14,16}
# batch=12, grad_accum=4 (effective batch=48), MIG 20GB compatible
#
# D1: 128 steps, sparse diagnostics (8 log-spaced steps)
# D2: 1024 steps, sparse diagnostics (15 steps)
# UM-aware diagnostics: tracks actual pulse probability with update_management
set -uo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

BATCH=12
GACC=4

run_one() {
    local PHASE="$1"
    local TAG="$2"
    shift 2
    echo ""
    echo "[$PHASE] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py "$@"
    local RC=$?
    $PYTHON -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache()" 2>/dev/null
    if [ $RC -ne 0 ]; then
        echo "[$PHASE] FAIL  $TAG (exit=$RC) $(date)"
    else
        echo "[$PHASE] DONE  $TAG $(date)"
    fi
    sleep 3
}

COMMON="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016"
DIAG_LAYERS="--diag-layer-set 0,5,11"

# D1 settings: 128 steps, sparse diag (8 log-spaced steps)
D1_DIAG_STEPS="1,2,4,8,16,32,64,128"
D1="$COMMON --max-steps 128 --batch-size $BATCH --grad-accum-steps $GACC \
  --diag-update-exact $DIAG_LAYERS --diag-at-steps $D1_DIAG_STEPS --log-every 8"

# D2 settings: 1024 steps, sparse diag (15 steps)
D2_DIAG_STEPS="1,2,4,8,16,32,64,128,256,384,512,768,896,1024"
D2="$COMMON --max-steps 1024 --batch-size $BATCH --grad-accum-steps $GACC \
  --diag-carry-path --diag-update-exact \
  --diag-at-steps $D2_DIAG_STEPS \
  --diag-vrc-windows 1,16,64,256 $DIAG_LAYERS --log-every 64"

# TTv1 base (gamma=1, reset=1)
TTv1_BASE="--method ttv1 --ttv1-mode residual_lane --n-bits-slow 10 \
  --gamma 1.0 --with-reset-prob 1.0 --fast-lr 0.1 --transfer-lr 1.0 \
  --units-in-mbatch true --transfer-every 4"

D1_RESULTS="results/paper/diag_D1_sweep"
D2_RESULTS="results/paper/diag_D2_sweep"
mkdir -p "$D1_RESULTS" "$D2_RESULTS"

echo "============================================================"
echo "  D1 + D2 Full Sweep (UM-aware diagnostics)"
echo "  single_rpu: 8,10,12,14,16b (stochastic)"
echo "  ttv1:       8,10,12,14,16b (gamma=1, reset=1, slow=10b)"
echo "  batch=$BATCH, grad_accum=$GACC (eff=48)"
echo "  D1 diag steps: $D1_DIAG_STEPS"
echo "  D2 diag steps: $D2_DIAG_STEPS"
echo "  Start: $(date)"
echo "============================================================"

# ==================================================================
# Phase 1: D1 — single_rpu
# ==================================================================
echo ""
echo "==================== D1: single_rpu ========================"

for BITS in 8 10 12 14 16; do
    run_one "D1" "single_rpu_${BITS}b" $D1 \
        --method single_rpu --n-bits $BITS \
        --output-dir "$D1_RESULTS/single_rpu_${BITS}b"
done

# ==================================================================
# Phase 2: D1 — ttv1
# ==================================================================
echo ""
echo "==================== D1: ttv1 =============================="

for BITS in 8 10 12 14 16; do
    run_one "D1" "ttv1_${BITS}b" $D1 $TTv1_BASE \
        --n-bits $BITS \
        --output-dir "$D1_RESULTS/ttv1_${BITS}b"
done

echo ""
echo "============================================================"
echo "  D1 COMPLETE: $(date)"
echo "============================================================"

# ==================================================================
# Phase 3: D2 — single_rpu
# ==================================================================
echo ""
echo "==================== D2: single_rpu ========================"

for BITS in 8 10 12 14 16; do
    run_one "D2" "single_rpu_${BITS}b" $D2 \
        --method single_rpu --n-bits $BITS \
        --output-dir "$D2_RESULTS/single_rpu_${BITS}b"
done

# ==================================================================
# Phase 4: D2 — ttv1
# ==================================================================
echo ""
echo "==================== D2: ttv1 =============================="

for BITS in 8 10 12 14 16; do
    run_one "D2" "ttv1_${BITS}b" $D2 $TTv1_BASE \
        --n-bits $BITS \
        --output-dir "$D2_RESULTS/ttv1_${BITS}b"
done

echo ""
echo "============================================================"
echo "  D1 + D2 ALL COMPLETE: $(date)"
echo "============================================================"

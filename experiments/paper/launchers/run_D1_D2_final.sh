#!/bin/bash
# D1 + D2 final sweep
# bs=24 acc=2, all layers (48 tiles), warmup=0, lr fixed
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd "$SCRIPT_DIR"

run_one() {
    local PHASE="$1"
    local TAG="$2"
    shift 2
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

COMMON="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016 --diag-update-exact"

# ============================================================
# D1: Sub-pulse mapping — 128 steps
# All layers (no --diag-layer-set), bs=24 acc=2
# Diag steps: 10 points, ~evenly spaced
# ============================================================
D1_DIR="results/paper/diag_D1_subpulse"
mkdir -p "$D1_DIR"

D1_ARGS="$COMMON --max-steps 128 --batch-size 24 --grad-accum-steps 2 --diag-at-steps 1,8,16,24,32,48,64,80,96,128 --log-every 20"

echo "============================================================"
echo "  D1: Sub-pulse Mapping (128 steps, 10 diag points, all layers)"
echo "  bs=24 acc=2 (effective batch=48)"
echo "  Start: $(date)"
echo "============================================================"

# Single RPU stochastic {6,8,10,12,14}b
for BITS in 6 8 10 12 14; do
    run_one D1 "single_rpu_stoch_${BITS}b" $D1_ARGS \
        --method single_rpu --pulse-type stochastic --n-bits $BITS \
        --output-dir "$D1_DIR/single_rpu_stoch_${BITS}b"
done

# Single RPU deterministic {6,8,10,12,14}b
for BITS in 6 8 10 12 14; do
    run_one D1 "single_rpu_det_${BITS}b" $D1_ARGS \
        --method single_rpu --pulse-type deterministic --n-bits $BITS \
        --output-dir "$D1_DIR/single_rpu_det_${BITS}b"
done

echo ""
echo "=== D1 complete: $(date) ==="
echo ""

# ============================================================
# D2: Carry-path — 1024 steps
# TTv1 residual_lane s10 + single_rpu 10b (baseline)
# ============================================================
D2_DIR="results/paper/diag_D2_carrypath"
mkdir -p "$D2_DIR"

D2_DIAG="--diag-at-steps 1,16,64,128,256,384,512,640,768,896,1024 --diag-vrc-windows 1,16,64,256"

echo "============================================================"
echo "  D2: Carry-path (1024 steps, 11 diag points, all layers)"
echo "  Start: $(date)"
echo "============================================================"

# Single RPU baselines {6,8,10,12,14}b (bs=24 acc=2)
for BITS in 6 8 10 12 14; do
    run_one D2 "single_rpu_stoch_${BITS}b" $COMMON --max-steps 1024 --batch-size 24 --grad-accum-steps 2 $D2_DIAG \
        --diag-carry-path --method single_rpu --pulse-type stochastic --n-bits $BITS \
        --output-dir "$D2_DIR/single_rpu_stoch_${BITS}b"
done

# TTv1 residual_lane slow={6,8,10,12,14}b, gamma=1.0, reset=1.0
for SLOW in 6 8 10 12 14; do
    run_one D2 "ttv1_rl_g1r1_s${SLOW}" $COMMON --max-steps 1024 --batch-size 16 --grad-accum-steps 3 $D2_DIAG \
        --diag-carry-path --diag-update-exact \
        --method ttv1 --ttv1-mode residual_lane \
        --n-bits 14 --n-bits-slow $SLOW --gamma 1.0 --with-reset-prob 1.0 \
        --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 3 \
        --output-dir "$D2_DIR/ttv1_rl_g1r1_s${SLOW}"
done

# TTv1 residual_lane slow={6,8,10,12,14}b, gamma=0.0, reset=0.0
for SLOW in 6 8 10 12 14; do
    run_one D2 "ttv1_rl_g0r0_s${SLOW}" $COMMON --max-steps 1024 --batch-size 16 --grad-accum-steps 3 $D2_DIAG \
        --diag-carry-path --diag-update-exact \
        --method ttv1 --ttv1-mode residual_lane \
        --n-bits 14 --n-bits-slow $SLOW --gamma 0.0 --with-reset-prob 0.0 \
        --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 3 \
        --output-dir "$D2_DIR/ttv1_rl_g0r0_s${SLOW}"
done

echo ""
echo "============================================================"
echo "  D1 + D2 ALL COMPLETE: $(date)"
echo "============================================================"

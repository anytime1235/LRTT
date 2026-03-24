#!/bin/bash
# D2 carry-path diagnostics: bs=12 acc=4, GPU matmul
# TTv1 first, then single_rpu baselines
set -uo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

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

COMMON="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016 --diag-update-exact --diag-layer-set 0,5,11"

D2_DIR="results/paper/diag_D2_carrypath"
mkdir -p "$D2_DIR"

D2_DIAG="--diag-at-steps 1,16,64,128,256,384,512,640,768,896,1024 --diag-vrc-windows 1,16,64,256,512,1024"
D2_COMMON="$COMMON --max-steps 1024 --batch-size 12 --grad-accum-steps 4 $D2_DIAG --diag-carry-path"

echo "============================================================"
echo "  D2: Carry-path (1024 steps, bs=12 acc=4, layers 0,5,11)"
echo "  Start: $(date)"
echo "============================================================"

# --- TTv1 FIRST ---

# TTv1 residual_lane slow={8,10,12,14}b, gamma=1.0, reset=1.0
for SLOW in 8 10 12 14; do
    run_one D2 "ttv1_rl_g1r1_s${SLOW}" $D2_COMMON \
        --method ttv1 --ttv1-mode residual_lane \
        --n-bits 14 --n-bits-slow $SLOW --gamma 1.0 --with-reset-prob 1.0 \
        --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 4 \
        --output-dir "$D2_DIR/ttv1_rl_g1r1_s${SLOW}"
done

# TTv1 residual_lane slow={8,10,12,14}b, gamma=0.0, reset=0.0
for SLOW in 8 10 12 14; do
    run_one D2 "ttv1_rl_g0r0_s${SLOW}" $D2_COMMON \
        --method ttv1 --ttv1-mode residual_lane \
        --n-bits 14 --n-bits-slow $SLOW --gamma 0.0 --with-reset-prob 0.0 \
        --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 4 \
        --output-dir "$D2_DIR/ttv1_rl_g0r0_s${SLOW}"
done

# --- Single RPU baselines ---
for BITS in 8 10 12 14; do
    run_one D2 "single_rpu_stoch_${BITS}b" $D2_COMMON \
        --method single_rpu --pulse-type stochastic --n-bits $BITS \
        --output-dir "$D2_DIR/single_rpu_stoch_${BITS}b"
done

echo ""
echo "============================================================"
echo "  D2 ALL COMPLETE: $(date)"
echo "============================================================"

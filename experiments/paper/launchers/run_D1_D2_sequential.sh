#!/bin/bash
# D1 + D2 sequential on single GPU (MIG 20GB)
# Each run calls python directly — no subshells, no parallel processes.
# Settings: warmup=0, lr fixed (min_lr_rate=1.0), transfer_every=grad_accum
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

# Common: warmup=0, lr fixed (no cosine decay)
COMMON_BASE="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016 --diag-carry-path --diag-update-exact"

# ============================================================
# D1: Sub-pulse mapping — 12 runs × 128 steps
# ============================================================
D1_DIR="results/paper/diag_D1_subpulse"
mkdir -p "$D1_DIR"

echo "============================================================"
echo "  D1: Sub-pulse Mapping (128 steps, diag@1,16,32,64,128)"
echo "  Start: $(date)"
echo "============================================================"

D1_ARGS="$COMMON_BASE --max-steps 128 --batch-size 16 --grad-accum-steps 3 --diag-at-steps 1,16,32,64,128 --diag-vrc-windows 16,64 --log-every 20"

for BITS in 8 10 12 14; do
    run_one D1 "single_rpu_stoch_${BITS}b" $D1_ARGS --method single_rpu --pulse-type stochastic --n-bits $BITS --output-dir "$D1_DIR/single_rpu_stoch_${BITS}b"
done

for BITS in 8 10; do
    run_one D1 "single_rpu_det_${BITS}b" $D1_ARGS --method single_rpu --pulse-type deterministic --n-bits $BITS --output-dir "$D1_DIR/single_rpu_det_${BITS}b"
done

for BITS in 8 10; do
    run_one D1 "eco_ref_${BITS}b" $D1_ARGS --method eco_ref --n-bits $BITS --eco-rounding stochastic --output-dir "$D1_DIR/eco_ref_${BITS}b"
done

for BITS in 8 10 12 14; do
    run_one D1 "mixed_precision_${BITS}b" $D1_ARGS --method mixed_precision --n-bits $BITS --output-dir "$D1_DIR/mixed_precision_${BITS}b"
done

echo ""
echo "=== D1 complete: $(date) ==="
echo ""

# ============================================================
# D2: Carry-path comparison — 1024 steps
# ============================================================
D2_DIR="results/paper/diag_D2_carrypath"
mkdir -p "$D2_DIR"

echo "============================================================"
echo "  D2: Carry-path Comparison (1024 steps)"
echo "  Start: $(date)"
echo "============================================================"

D2_DIAG="--diag-at-steps 1,16,64,128,256,512,1024 --diag-vrc-windows 1,16,64,256 --log-every 50"

# bs16 acc3 configs
D2_BS16="$COMMON_BASE --max-steps 1024 --batch-size 16 --grad-accum-steps 3 $D2_DIAG"

# bs8 acc6 configs (TTv1, eco_ref — OOM with bs16)
D2_BS8="$COMMON_BASE --max-steps 1024 --batch-size 8 --grad-accum-steps 6 $D2_DIAG"

# TTv1 common: match original training settings, transfer_every=grad_accum
TTv1_BS16="--n-bits 14 --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 3 --with-reset-prob 1.0"
TTv1_BS8="--n-bits 14 --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 6 --with-reset-prob 1.0"

# single_rpu stoch {8,10}b
for BITS in 8 10; do
    run_one D2 "single_rpu_stoch_${BITS}b" $D2_BS16 --method single_rpu --pulse-type stochastic --n-bits $BITS --output-dir "$D2_DIR/single_rpu_stoch_${BITS}b"
done

# mixed_precision {8,10}b
for BITS in 8 10; do
    run_one D2 "mixed_precision_${BITS}b" $D2_BS16 --method mixed_precision --n-bits $BITS --output-dir "$D2_DIR/mixed_precision_${BITS}b"
done

# eco_ref {8,10}b
for BITS in 8 10; do
    run_one D2 "eco_ref_${BITS}b" $D2_BS8 --method eco_ref --n-bits $BITS --eco-rounding stochastic --output-dir "$D2_DIR/eco_ref_${BITS}b"
done

# ttv1 hidden_buffer slow={8,10}b
for SLOW in 8 10; do
    run_one D2 "ttv1_hb_s${SLOW}" $D2_BS8 --method ttv1 --ttv1-mode hidden_buffer $TTv1_BS8 --n-bits-slow $SLOW --output-dir "$D2_DIR/ttv1_hb_s${SLOW}"
done

# ttv1 residual_lane slow={8,10}b (gamma=1.0, reset=1.0)
for SLOW in 8 10; do
    run_one D2 "ttv1_rl_s${SLOW}" $D2_BS8 --method ttv1 --ttv1-mode residual_lane $TTv1_BS8 --n-bits-slow $SLOW --gamma 1.0 --output-dir "$D2_DIR/ttv1_rl_s${SLOW}"
done

# ttv1 residual_lane_noreset slow={8,10}b (gamma=1.0, reset=0.0)
for SLOW in 8 10; do
    run_one D2 "ttv1_rl_noreset_s${SLOW}" $D2_BS8 --method ttv1 --ttv1-mode residual_lane_noreset $TTv1_BS8 --n-bits-slow $SLOW --gamma 1.0 --with-reset-prob 0.0 --output-dir "$D2_DIR/ttv1_rl_noreset_s${SLOW}"
done

echo ""
echo "============================================================"
echo "  D1 + D2 ALL COMPLETE: $(date)"
echo "============================================================"

#!/bin/bash
# D2 Minimal: ttv1 14b(fast)/10b(slow) first, then single_rpu 10b
set -uo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

BATCH=12
GACC=4

run_one() {
    local TAG="$1"
    shift
    echo ""
    echo "[D2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py "$@"
    local RC=$?
    $PYTHON -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache()" 2>/dev/null
    if [ $RC -ne 0 ]; then
        echo "[D2] FAIL  $TAG (exit=$RC) $(date)"
    else
        echo "[D2] DONE  $TAG $(date)"
    fi
    sleep 3
}

COMMON="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016"
DIAG_LAYERS="--diag-layer-set 0,5,11"
D2_DIAG_STEPS="1,2,4,8,16,32,64,128,256,384,512,768,896,1024"

D2_RESULTS="results/paper/diag_D2_sweep"
mkdir -p "$D2_RESULTS"

echo "============================================================"
echo "  D2 Minimal (1024 steps)"
echo "  1) ttv1 14b(fast)/10b(slow) — with carry_path"
echo "  2) single_rpu 10b — no carry_path"
echo "  Start: $(date)"
echo "============================================================"

# 1) ttv1 14b fast / 10b slow — WITH carry_path (TTv1 first)
run_one "ttv1_14b" \
    $COMMON --max-steps 1024 --batch-size $BATCH --grad-accum-steps $GACC \
    --diag-update-exact --diag-carry-path \
    --diag-at-steps $D2_DIAG_STEPS \
    --diag-vrc-windows 1,16,64,256 $DIAG_LAYERS --log-every 64 \
    --method ttv1 --ttv1-mode residual_lane --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --with-reset-prob 1.0 --fast-lr 0.1 --transfer-lr 1.0 \
    --units-in-mbatch true --transfer-every 4 \
    --output-dir "$D2_RESULTS/ttv1_14b"

# 2) single_rpu 10b — NO carry_path (fast)
run_one "single_rpu_10b" \
    $COMMON --max-steps 1024 --batch-size $BATCH --grad-accum-steps $GACC \
    --diag-update-exact $DIAG_LAYERS --diag-at-steps $D2_DIAG_STEPS --log-every 64 \
    --method single_rpu --n-bits 10 \
    --output-dir "$D2_RESULTS/single_rpu_10b"

echo ""
echo "============================================================"
echo "  D2 Minimal COMPLETE: $(date)"
echo "============================================================"

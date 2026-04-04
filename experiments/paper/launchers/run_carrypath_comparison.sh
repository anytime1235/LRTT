#!/bin/bash
# Carry path comparison: SingleRPU vs TTv1
# Fix: ln-lr=0.003 (matches working config)
# 1) SingleRPU 10b  → baseline (no carry path)
# 2) TTv1 gamma=1, reset=1 → carry path enabled
set -euo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

COMMON="--mode fixed --seed 42 --epochs 1 --max-steps 1024 \
  --batch-size 12 --grad-accum-steps 4 \
  --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.003 \
  --warmup-ratio 0 --min-lr-rate 1.0 \
  --target-layers attention --log-every 64"

DIAG="--diag-carry-path \
  --diag-at-steps 1,2,4,8,16,32,64,128,256,384,512,768,896,1024 \
  --diag-vrc-windows 1,16,64,256 --diag-layer-set 0,5,11"

RESULTS="results/paper/diag_D2_comparison"

echo "============================================================"
echo "  Carry Path Comparison: SingleRPU vs TTv1"
echo "  ln-lr=0.003 (fixed)"
echo "  2 experiments, sequential"
echo "  Start: $(date)"
echo "============================================================"

# 1) SingleRPU 10b — no carry path baseline
echo ""
echo "[1/2] SingleRPU 10b  $(date)"
$PYTHON paper_experiment.py $COMMON $DIAG \
    --method single_rpu --n-bits 10 \
    --output-dir "$RESULTS/single_rpu_10b"
echo "[1/2] DONE $(date)"

# 2) TTv1 gamma=1, reset=1 — carry path
echo ""
echo "[2/2] TTv1 gamma=1, reset=1  $(date)"
$PYTHON paper_experiment.py $COMMON $DIAG \
    --method ttv1 \
    --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --with-reset-prob 1.0 \
    --fast-lr 0.1 --transfer-lr 1.0 \
    --units-in-mbatch true --transfer-every 4 \
    --output-dir "$RESULTS/ttv1_g1_r1"
echo "[2/2] DONE $(date)"

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"

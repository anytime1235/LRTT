#!/bin/bash
# D2 VRC 1-epoch: SingleRPU 10b → TTv1 14b/10b (gamma=1, reset=1)
# Full 1 epoch (7377 steps), extended VRC windows up to K=4096
set -euo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

COMMON="--mode fixed --seed 42 --epochs 1 --max-steps 4096 \
  --batch-size 12 --grad-accum-steps 4 \
  --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.003 \
  --warmup-ratio 0 --min-lr-rate 1.0 \
  --target-layers attention --log-every 64"

DIAG="--diag-carry-path \
  --diag-at-steps 1,2,4,8,16,32,64,128,256,512,1024,2048,3072,4096 \
  --diag-vrc-windows 1,16,64,256,512,1024,2048 --diag-layer-set 0,5,11"

RESULTS="results/paper/diag_D2_vrc_1ep"

echo "============================================================"
echo "  D2 VRC 1-epoch Experiment"
echo "  1) SingleRPU 10b"
echo "  2) TTv1 14b/10b (gamma=1, reset=1)"
echo "  Start: $(date)"
echo "============================================================"

echo ""
echo "[1/2] TTv1 14b/10b gamma=1, reset=1  $(date)"
$PYTHON paper_experiment.py $COMMON $DIAG \
    --method ttv1 \
    --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --with-reset-prob 1.0 \
    --fast-lr 0.1 --transfer-lr 1.0 \
    --units-in-mbatch true --transfer-every 4 \
    --output-dir "$RESULTS/ttv1_14b"
echo "[1/2] DONE $(date)"

echo ""
echo "[2/2] SingleRPU 10b  $(date)"
$PYTHON paper_experiment.py $COMMON $DIAG \
    --method single_rpu \
    --n-bits 10 \
    --output-dir "$RESULTS/single_rpu_10b"
echo "[2/2] DONE $(date)"

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"

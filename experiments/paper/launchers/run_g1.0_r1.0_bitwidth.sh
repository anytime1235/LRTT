#!/bin/bash
# g1.0_r1.0 with different fast tile bit widths (8, 10)
# slow tile = 10-bit, same settings as phase1c_4ep_b16

set -e
export PYTHON=/root/.venv310/bin/python
cd /root/LRTT/experiments/paper

COMMON="--mode fixed --method ttv1 --seed 42 \
  --epochs 4 --n-bits-slow 10 \
  --gamma 1.0 --units-in-mbatch true --transfer-every 3 \
  --with-reset-prob 1.0 --fast-lr 0.1 \
  --transfer-lr 1.0 --scale-transfer-lr false \
  --ln-lr 0.003 --min-lr-rate 0.05 \
  --batch-size 16 --grad-accum-steps 3 \
  --log-every 20"

echo "============================================"
echo "[1/2] g1.0_r1.0 fast=8bit slow=10bit"
echo "============================================"
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py $COMMON \
  --n-bits 8 \
  --output-dir results/paper/phase1c_4ep_b16/g1.0_r1.0_fast8b

echo "============================================"
echo "[2/2] g1.0_r1.0 fast=10bit slow=10bit"
echo "============================================"
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py $COMMON \
  --n-bits 10 \
  --output-dir results/paper/phase1c_4ep_b16/g1.0_r1.0_fast10b

echo "============================================"
echo "All done!"
echo "============================================"

#!/bin/bash
# Phase1C TTv1 Gamma×Reset sweep — first 3 experiments
# batch=16, grad_accum=3 (eff=48), transfer_every=3, min_lr_rate=0.05
# IO: perfect (default)

set -e
export PYTHON=/root/.venv310/bin/python
cd /root/LRTT/experiments/paper

COMMON="--mode fixed --method ttv1 --seed 42 \
  --epochs 4 --n-bits 14 --n-bits-slow 10 \
  --units-in-mbatch true --transfer-every 3 \
  --fast-lr 0.1 --scale-transfer-lr false \
  --ln-lr 0.003 --min-lr-rate 0.05 \
  --batch-size 16 --grad-accum-steps 3 \
  --log-every 20"

echo "============================================"
echo "[1/3] g0.1_r0  (gamma=0.1, reset=0)"
echo "============================================"
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py $COMMON \
  --gamma 0.1 --with-reset-prob 0 --transfer-lr 0.1 \
  --output-dir results/paper/phase1c_4ep_b16/g0.1_r0

echo "============================================"
echo "[2/3] g0.3_r0  (gamma=0.3, reset=0)"
echo "============================================"
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py $COMMON \
  --gamma 0.3 --with-reset-prob 0 --transfer-lr 0.3 \
  --output-dir results/paper/phase1c_4ep_b16/g0.3_r0

echo "============================================"
echo "[3/3] g0.1_r1.0  (gamma=0.1, reset=1.0)"
echo "============================================"
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py $COMMON \
  --gamma 0.1 --with-reset-prob 1.0 --transfer-lr 0.1 \
  --output-dir results/paper/phase1c_4ep_b16/g0.1_r1.0

echo "============================================"
echo "All 3 experiments done!"
echo "============================================"

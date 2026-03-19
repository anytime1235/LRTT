#!/bin/bash
# 후반 3개 TTv1 실험 순차 실행
# batch=16, grad_accum=3 (effective=48), transfer_every=3
# MIG 20GB 환경 대응

set -e
PYTHON=/root/.venv310/bin/python
cd /root/LRTT/experiments/paper
export CUDA_VISIBLE_DEVICES=0

echo "=== [1/3] g0.3_r1.0 시작: $(date) ==="
$PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 0.3 --units-in-mbatch true --transfer-every 3 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 0.3 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --batch-size 16 --grad-accum-steps 3 \
    --min-lr-rate 0.05 \
    --output-dir results/paper/phase1c_4ep/g0.3_r1.0 \
    --log-every 20
echo "=== [1/3] g0.3_r1.0 완료: $(date) ==="

echo "=== [2/3] g1.0_r0 시작: $(date) ==="
$PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 3 \
    --with-reset-prob 0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --batch-size 16 --grad-accum-steps 3 \
    --min-lr-rate 0.05 \
    --output-dir results/paper/phase1c_4ep/g1.0_r0 \
    --log-every 20
echo "=== [2/3] g1.0_r0 완료: $(date) ==="

echo "=== [3/3] g1.0_r1.0 시작: $(date) ==="
$PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 3 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --batch-size 16 --grad-accum-steps 3 \
    --min-lr-rate 0.05 \
    --output-dir results/paper/phase1c_4ep/g1.0_r1.0 \
    --log-every 20
echo "=== [3/3] g1.0_r1.0 완료: $(date) ==="

echo "=== 전체 완료: $(date) ==="

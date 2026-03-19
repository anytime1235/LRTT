#!/bin/bash
# Deterministic update bit sweep: 10, 12, 14 bit (fast + transfer)
# g1.0_r1.0 기본 설정, pulse_type만 deterministic으로 변경
# 현재 실행 중인 16bit stochastic 실험 (PID 88656) 완료 후 자동 실행

set -e
PYTHON=/root/.venv310/bin/python
cd /root/LRTT/experiments/paper
export CUDA_VISIBLE_DEVICES=0

# 16bit stochastic 실험 완료 대기
echo "=== 16bit stochastic (PID 88656) 완료 대기 중: $(date) ==="
while kill -0 88656 2>/dev/null; do
    sleep 30
done
echo "=== 이전 실험 완료 확인, deterministic bit sweep 시작: $(date) ==="

echo "=== [1/3] g1.0_r1.0 deterministic 10bit 시작: $(date) ==="
$PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 10 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 3 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --batch-size 16 --grad-accum-steps 3 \
    --min-lr-rate 0.05 \
    --ttv1-fast-pulse-type deterministic \
    --ttv1-transfer-pulse-type deterministic \
    --output-dir results/paper/phase1c_4ep/g1.0_r1.0_det_10b \
    --log-every 20
echo "=== [1/3] g1.0_r1.0 deterministic 10bit 완료: $(date) ==="

echo "=== [2/3] g1.0_r1.0 deterministic 12bit 시작: $(date) ==="
$PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 12 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 3 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --batch-size 16 --grad-accum-steps 3 \
    --min-lr-rate 0.05 \
    --ttv1-fast-pulse-type deterministic \
    --ttv1-transfer-pulse-type deterministic \
    --output-dir results/paper/phase1c_4ep/g1.0_r1.0_det_12b \
    --log-every 20
echo "=== [2/3] g1.0_r1.0 deterministic 12bit 완료: $(date) ==="

echo "=== [3/3] g1.0_r1.0 deterministic 14bit 시작: $(date) ==="
$PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 3 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --batch-size 16 --grad-accum-steps 3 \
    --min-lr-rate 0.05 \
    --ttv1-fast-pulse-type deterministic \
    --ttv1-transfer-pulse-type deterministic \
    --output-dir results/paper/phase1c_4ep/g1.0_r1.0_det_14b \
    --log-every 20
echo "=== [3/3] g1.0_r1.0 deterministic 14bit 완료: $(date) ==="

echo "=== deterministic bit sweep 전체 완료: $(date) ==="

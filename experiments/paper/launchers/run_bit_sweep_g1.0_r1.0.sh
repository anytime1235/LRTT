#!/bin/bash
# g1.0_r1.0 fast tile bit-width sweep: 12-bit, 14-bit
# 현재 실행 중인 run_remaining_3.sh 완료 후 자동 실행
# 모든 조건 동일, fast tile n-bits만 변경 (slow=10 고정)

set -e
PYTHON=/root/.venv310/bin/python
cd /root/LRTT/experiments/paper
export CUDA_VISIBLE_DEVICES=0

# run_remaining_3.sh 완료 대기
echo "=== run_remaining_3.sh (PID 14788) 완료 대기 중: $(date) ==="
while kill -0 14788 2>/dev/null; do
    sleep 30
done
echo "=== 이전 실험 완료 확인, bit sweep 시작: $(date) ==="

echo "=== [1/2] g1.0_r1.0 fast=12bit 시작: $(date) ==="
$PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 12 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 3 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --batch-size 16 --grad-accum-steps 3 \
    --min-lr-rate 0.05 \
    --output-dir results/paper/phase1c_4ep/g1.0_r1.0_fast12b \
    --log-every 20
echo "=== [1/2] g1.0_r1.0 fast=12bit 완료: $(date) ==="

echo "=== [2/2] g1.0_r1.0 fast=16bit 시작: $(date) ==="
$PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 16 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 3 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --batch-size 16 --grad-accum-steps 3 \
    --min-lr-rate 0.05 \
    --output-dir results/paper/phase1c_4ep/g1.0_r1.0_fast16b \
    --log-every 20
echo "=== [2/2] g1.0_r1.0 fast=16bit 완료: $(date) ==="

echo "=== bit sweep 전체 완료: $(date) ==="

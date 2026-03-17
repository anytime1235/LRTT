#!/bin/bash
# Remaining Mixed Precision 10-bit experiments (sequential)
# MIG 3g.20GB environment — uses inline grad_accum to avoid OOM
#
# 1) mp_10b_all        (alr=0.0357)  — all 72 layers, batch=16, accum=3
# 2) mp_10b_ffn_alr0.00357          — ffn 24 layers, batch=24, accum=2
# 3) mp_10b_all_alr0.00357          — all 72 layers, batch=16, accum=3

set -e
PYTHON=/root/.venv310/bin/python
cd /root/LRTT/experiments/paper
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "[1/3] mp_10b_all (alr=0.0357) — all layers"
echo "      batch=16, grad_accum=3, effective_batch=48"
echo "============================================================"
$PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --seed 42 \
    --epochs 4 --n-bits 10 \
    --target-layers all \
    --batch-size 16 --grad-accum-steps 3 \
    --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
    --output-dir results/paper/mixed_prec_10b/mp_10b_all \
    --log-every 20

echo ""
echo "============================================================"
echo "[2/3] mp_10b_ffn_alr0.00357 — FFN only, alr=1/10"
echo "      batch=24, grad_accum=2, effective_batch=48"
echo "============================================================"
$PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --seed 42 \
    --epochs 4 --n-bits 10 \
    --target-layers ffn \
    --batch-size 24 --grad-accum-steps 2 \
    --analog-lr 0.00357 --classifier-lr 0.00076 --ln-lr 0.00076 \
    --output-dir results/paper/mixed_prec_10b/mp_10b_ffn_alr0.00357 \
    --log-every 20

echo ""
echo "============================================================"
echo "[3/3] mp_10b_all_alr0.00357 — all layers, alr=1/10"
echo "      batch=16, grad_accum=3, effective_batch=48"
echo "============================================================"
$PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --seed 42 \
    --epochs 4 --n-bits 10 \
    --target-layers all \
    --batch-size 16 --grad-accum-steps 3 \
    --analog-lr 0.00357 --classifier-lr 0.00076 --ln-lr 0.00076 \
    --output-dir results/paper/mixed_prec_10b/mp_10b_all_alr0.00357 \
    --log-every 20

echo ""
echo "============================================================"
echo "ALL 3 EXPERIMENTS COMPLETE"
echo "============================================================"

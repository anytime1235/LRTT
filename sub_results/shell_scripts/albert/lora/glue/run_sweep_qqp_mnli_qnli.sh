#!/bin/bash
# Sequential Optuna sweep: QQP, MNLI, QNLI (all-layer, 10 trials each)
# Narrowed ranges based on SST-2 best: lr~0.001, alpha~0.02, ab_lr~0.04

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_lora.py

COMMON="--lora-target all --warm-alpha --convert-nontarget \
  --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov \
  --no-learn-out-scaling --warmup-ratio 0.1 --n-trials 30"

echo "=== Starting QQP (5ep, bs128) ==="
$PYTHON $SCRIPT --task qqp --batch-size 128 --epochs 5 $COMMON 2>&1
echo ""

echo "=== Starting MNLI (4ep, bs128) ==="
$PYTHON $SCRIPT --task mnli --batch-size 128 --epochs 4 $COMMON 2>&1
echo ""

echo "=== Starting QNLI (11ep, bs32) ==="
$PYTHON $SCRIPT --task qnli --batch-size 32 --epochs 11 $COMMON 2>&1
echo ""

echo "=== All sweeps complete ==="

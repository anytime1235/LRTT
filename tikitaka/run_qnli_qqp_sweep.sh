#!/bin/bash
# Sequential sweep: QNLI -> QQP
# QKV + classifier trainable, AnalogSGD, 3ep, 10 trials each
# Pruning disabled

export TOKENIZERS_PARALLELISM=false

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/LRTT_transformer/tikitaka/sweep_tikitaka_batch256.py
LOG_DIR=/tmp
TARGET="query key value classifier"

echo "=========================================="
echo "Starting QNLI + QQP sequential sweep"
echo "Start time: $(date)"
echo "=========================================="

echo ""
echo "[1/2] QNLI sweep (10 trials, 3 epochs)"
echo "Start: $(date)"
$PYTHON $SCRIPT --tasks qnli --n_trials 10 --target_modules $TARGET 2>&1 | tee ${LOG_DIR}/sweep_qnli_classifier.log
echo "QNLI done: $(date)"

echo ""
echo "[2/2] QQP sweep (10 trials, 3 epochs)"
echo "Start: $(date)"
$PYTHON $SCRIPT --tasks qqp --n_trials 10 --target_modules $TARGET 2>&1 | tee ${LOG_DIR}/sweep_qqp_classifier.log
echo "QQP done: $(date)"

echo ""
echo "=========================================="
echo "All sweeps complete: $(date)"
echo "=========================================="

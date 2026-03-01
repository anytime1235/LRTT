#!/bin/bash
# Sequential sweep: QNLI -> QQP -> SST-2
# QKV + classifier trainable, AnalogSGD, 3ep, 10 trials each
# LR range: 1e-3 ~ 1e-1, Pruning disabled

export TOKENIZERS_PARALLELISM=false

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/LRTT_transformer/tikitaka/sweep_tikitaka_batch256.py
LOG_DIR=/tmp
TARGET="query key value classifier"

echo "=========================================="
echo "Starting QNLI + QQP + SST-2 sequential sweep"
echo "LR range: 1e-3 ~ 1e-1"
echo "Start time: $(date)"
echo "=========================================="

echo ""
echo "[1/3] QNLI sweep (10 trials, 3 epochs)"
echo "Start: $(date)"
$PYTHON $SCRIPT --tasks qnli --n_trials 10 --target_modules $TARGET 2>&1 | tee ${LOG_DIR}/sweep_qnli_classifier.log
echo "QNLI done: $(date)"

echo ""
echo "[2/3] QQP sweep (10 trials, 3 epochs)"
echo "Start: $(date)"
$PYTHON $SCRIPT --tasks qqp --n_trials 10 --target_modules $TARGET 2>&1 | tee ${LOG_DIR}/sweep_qqp_classifier.log
echo "QQP done: $(date)"

echo ""
echo "[3/3] SST-2 sweep (10 trials, 3 epochs)"
echo "Start: $(date)"
$PYTHON $SCRIPT --tasks sst2 --n_trials 10 --target_modules $TARGET 2>&1 | tee ${LOG_DIR}/sweep_sst2_classifier.log
echo "SST-2 done: $(date)"

echo ""
echo "=========================================="
echo "All sweeps complete: $(date)"
echo "=========================================="

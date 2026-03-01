#!/bin/bash
# Sequential GLUE sweep for all tasks (except SST-2 which is already running)
# Hyperparameter ranges narrowed based on SST-2 analysis:
#   learning_rate: [0.1, 1.0], lora_alpha: [0.001, 0.05], target_ab_lr: [0.003, 0.15]
#
# Usage: nohup bash /data/run_glue_sweep_all.sh > /data/results/Analoglora_v2/sweep_all.log 2>&1 &

set -e
PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_mobilebert_glue_lora.py
COMMON_ARGS="--lora-target qkv --combined-out-scaling --warm-alpha \
  --optimizer AnalogSGD --reinit-mode decay \
  --no-wd --no-momentum --no-nesterov --batch-size 64 --epochs 5"

echo "===== GLUE Sweep Started: $(date) ====="

# --- LARGE datasets (10 trials) ---

echo ""
echo "===== [1/7] MNLI (392K samples, 10 trials) ====="
echo "Start: $(date)"
$PYTHON $SCRIPT --task mnli --n-trials 10 $COMMON_ARGS
echo "Done: $(date)"

echo ""
echo "===== [2/7] QQP (363K samples, 10 trials) ====="
echo "Start: $(date)"
$PYTHON $SCRIPT --task qqp --n-trials 10 $COMMON_ARGS
echo "Done: $(date)"

echo ""
echo "===== [3/7] QNLI (104K samples, 10 trials) ====="
echo "Start: $(date)"
$PYTHON $SCRIPT --task qnli --n-trials 10 $COMMON_ARGS
echo "Done: $(date)"

# --- SMALL datasets (30 trials) ---

echo ""
echo "===== [4/7] CoLA (8.5K samples, 30 trials) ====="
echo "Start: $(date)"
$PYTHON $SCRIPT --task cola --n-trials 30 $COMMON_ARGS
echo "Done: $(date)"

echo ""
echo "===== [5/7] STS-B (5.7K samples, 30 trials) ====="
echo "Start: $(date)"
$PYTHON $SCRIPT --task stsb --n-trials 30 $COMMON_ARGS
echo "Done: $(date)"

echo ""
echo "===== [6/7] MRPC (3.6K samples, 30 trials) ====="
echo "Start: $(date)"
$PYTHON $SCRIPT --task mrpc --n-trials 30 $COMMON_ARGS
echo "Done: $(date)"

echo ""
echo "===== [7/7] RTE (2.4K samples, 30 trials) ====="
echo "Start: $(date)"
$PYTHON $SCRIPT --task rte --n-trials 30 $COMMON_ARGS
echo "Done: $(date)"

echo ""
echo "===== All GLUE Sweeps Completed: $(date) ====="

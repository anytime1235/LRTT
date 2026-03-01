#!/usr/bin/env bash
# run_glue_optuna.sh — GLUE Optuna hyperparameter sweep with TikiTaka
#
# Runs Optuna study for BERT-base on GLUE tasks with TikiTaka analog training.
#
# Usage:
#   cd /data/main_results
#   bash run_glue_optuna.sh
#
# Customize GLUE_TASK, N_TRIALS, EPOCHS below.

set -euo pipefail

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/main_results/optuna_bert_glue_tiki.py

# Configuration
GLUE_TASK=${GLUE_TASK:-sst2}
N_TRIALS=${N_TRIALS:-10}
EPOCHS=${EPOCHS:-2}
LORA_TARGET=${LORA_TARGET:-qkv}
OPTIMIZER=${OPTIMIZER:-AnalogSGD}
OUT_DIR=${OUT_DIR:-./results/glue_optuna_${GLUE_TASK}}

echo "=== GLUE Optuna Sweep ==="
echo "  Task:      ${GLUE_TASK}"
echo "  Trials:    ${N_TRIALS}"
echo "  Epochs:    ${EPOCHS}"
echo "  LoRA:      ${LORA_TARGET}"
echo "  Optimizer: ${OPTIMIZER}"
echo "  Out dir:   ${OUT_DIR}"
echo ""

$PYTHON $SCRIPT \
    --glue-task   ${GLUE_TASK} \
    --n-trials    ${N_TRIALS} \
    --epochs      ${EPOCHS} \
    --lora-target ${LORA_TARGET} \
    --optimizer   ${OPTIMIZER} \
    --out-dir     ${OUT_DIR}

echo ""
echo "=== Optuna sweep complete ==="
echo "Results: ${OUT_DIR}"

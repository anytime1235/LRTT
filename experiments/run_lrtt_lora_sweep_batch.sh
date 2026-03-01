#!/bin/bash
#
# Automated LRTT-LoRA Hyperparameter Sweep Batch Script
#
# This script runs hyperparameter sweeps for:
# 1. SQuAD: 15 epochs, 30 trials
# 2. GLUE tasks: 3 epochs, 10 trials each (ordered by dataset size, largest first)
#
# All experiments run in 6T1C-LoRA mode with:
# - lora_alpha: [0.01, 100] log-uniform
# - learning_rate: [1e-4, 1e-1] log-uniform
#
# Results saved to: /data/results/lora_baseline/
#
# Usage:
#   bash run_lrtt_lora_sweep_batch.sh
#
# The script runs with nohup, so server disconnection won't stop execution.
#

set -e  # Exit on error

# Configuration
PYTHON=/data/venvs/aihwkit_gpu/bin/python
SCRIPT=/data/LRTT_transformer/experiments/sweep_lrtt_lora_optuna.py
MODE=sixt1c_lora
RESULTS_DIR=/data/results/lora_baseline
RANK=8
TARGET_MODULES="query key value"

# Create results directory
mkdir -p ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}/logs

# Timestamp for this batch run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG=${RESULTS_DIR}/logs/batch_run_${TIMESTAMP}.log

echo "================================================================================"
echo "LRTT-LoRA Hyperparameter Sweep Batch Execution"
echo "================================================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Mode: ${MODE}"
echo "Results directory: ${RESULTS_DIR}"
echo "Log file: ${BATCH_LOG}"
echo "================================================================================"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${BATCH_LOG}
}

log "=== BATCH RUN STARTED ==="
log "Mode: ${MODE}"
log "Python: ${PYTHON}"
log "Script: ${SCRIPT}"

# =============================================================================
# 1. SQuAD: 15 epochs, 30 trials
# =============================================================================

log ""
log "=== [1/10] Starting SQuAD sweep ==="
log "Settings: 15 epochs, 30 trials, ${MODE}"

STUDY_NAME=squad_${MODE}_${TIMESTAMP}
STORAGE=sqlite:///${RESULTS_DIR}/optuna_squad_${MODE}.db
OUTPUT_DIR=${RESULTS_DIR}/squad

nohup ${PYTHON} ${SCRIPT} \
    --task squad \
    --mode ${MODE} \
    --rank ${RANK} \
    --target_modules ${TARGET_MODULES} \
    --n_trials 30 \
    --study_name ${STUDY_NAME} \
    --storage ${STORAGE} \
    > ${RESULTS_DIR}/logs/squad_${MODE}_${TIMESTAMP}.log 2>&1 &

SQUAD_PID=$!
log "SQuAD sweep started (PID: ${SQUAD_PID})"
log "Output: ${RESULTS_DIR}/logs/squad_${MODE}_${TIMESTAMP}.log"
log "Waiting for SQuAD sweep to complete..."

# Wait for SQuAD to complete
wait ${SQUAD_PID}
SQUAD_EXIT=$?

if [ ${SQUAD_EXIT} -eq 0 ]; then
    log "✓ SQuAD sweep completed successfully"
else
    log "✗ SQuAD sweep failed with exit code ${SQUAD_EXIT}"
fi

# =============================================================================
# 2. GLUE Tasks: 3 epochs, 10 trials each
# Order: Largest datasets first (SST-2, QQP, MNLI, QNLI, MRPC, RTE, etc.)
# =============================================================================

# GLUE task list (ordered by dataset size, largest first)
declare -a GLUE_TASKS=(
    "sst2"    # 67K training samples
    "qqp"     # 364K training samples
    "mnli"    # 393K training samples
    "qnli"    # 105K training samples
    "mrpc"    # 3.7K training samples
    "rte"     # 2.5K training samples
    "cola"    # 8.5K training samples
    "stsb"    # 5.7K training samples
    "wnli"    # 634 training samples
)

TASK_NUM=2
for TASK in "${GLUE_TASKS[@]}"
do
    log ""
    log "=== [${TASK_NUM}/10] Starting GLUE ${TASK} sweep ==="
    log "Settings: 3 epochs, 10 trials, ${MODE}"

    STUDY_NAME=glue_${TASK}_${MODE}_${TIMESTAMP}
    STORAGE=sqlite:///${RESULTS_DIR}/optuna_glue_${TASK}_${MODE}.db

    nohup ${PYTHON} ${SCRIPT} \
        --task glue \
        --task_name ${TASK} \
        --mode ${MODE} \
        --rank ${RANK} \
        --target_modules ${TARGET_MODULES} \
        --n_trials 10 \
        --study_name ${STUDY_NAME} \
        --storage ${STORAGE} \
        > ${RESULTS_DIR}/logs/glue_${TASK}_${MODE}_${TIMESTAMP}.log 2>&1 &

    TASK_PID=$!
    log "${TASK} sweep started (PID: ${TASK_PID})"
    log "Output: ${RESULTS_DIR}/logs/glue_${TASK}_${MODE}_${TIMESTAMP}.log"
    log "Waiting for ${TASK} sweep to complete..."

    # Wait for task to complete
    wait ${TASK_PID}
    TASK_EXIT=$?

    if [ ${TASK_EXIT} -eq 0 ]; then
        log "✓ ${TASK} sweep completed successfully"
    else
        log "✗ ${TASK} sweep failed with exit code ${TASK_EXIT}"
    fi

    TASK_NUM=$((TASK_NUM + 1))
done

# =============================================================================
# Summary
# =============================================================================

log ""
log "=== BATCH RUN COMPLETED ==="
log "Results directory: ${RESULTS_DIR}"
log "Optuna databases: ${RESULTS_DIR}/optuna_*.db"
log "Best parameters: ${RESULTS_DIR}/**/best_params_*.json"
log "Logs: ${RESULTS_DIR}/logs/"

echo ""
echo "================================================================================"
echo "BATCH EXECUTION COMPLETE"
echo "================================================================================"
echo "Check results at: ${RESULTS_DIR}"
echo "Check logs at: ${RESULTS_DIR}/logs/"
echo "Batch log: ${BATCH_LOG}"
echo "================================================================================"

# Create summary file
SUMMARY_FILE=${RESULTS_DIR}/batch_summary_${TIMESTAMP}.txt
{
    echo "LRTT-LoRA Hyperparameter Sweep Batch Summary"
    echo "============================================="
    echo ""
    echo "Timestamp: ${TIMESTAMP}"
    echo "Mode: ${MODE}"
    echo ""
    echo "Tasks Completed:"
    echo "  1. SQuAD (15 epochs, 30 trials)"
    for TASK in "${GLUE_TASKS[@]}"
    do
        echo "  - GLUE ${TASK} (3 epochs, 10 trials)"
    done
    echo ""
    echo "Results Location: ${RESULTS_DIR}"
    echo "Databases: ${RESULTS_DIR}/optuna_*.db"
    echo "Logs: ${RESULTS_DIR}/logs/"
    echo ""
    echo "Search Space:"
    echo "  lora_alpha: [0.01, 100] (log-uniform)"
    echo "  learning_rate: [1e-4, 1e-1] (log-uniform)"
    echo "  rank: ${RANK} (fixed)"
    echo "  target_modules: ${TARGET_MODULES}"
    echo ""
    echo "Configuration:"
    echo "  Optimizer: AnalogSGD"
    echo "  Batch size: 32"
    echo "  Max seq length (GLUE): 128"
    echo "  Max seq length (SQuAD): 384"
    echo "  Doc stride (SQuAD): 128"
    echo "  Warmup: 10% linear"
    echo "  Early stopping: patience=3"
} > ${SUMMARY_FILE}

log "Summary saved to: ${SUMMARY_FILE}"

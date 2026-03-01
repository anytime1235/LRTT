#!/bin/bash
#
# Automated LRTT-LoRA Hyperparameter Sweep Batch Script (FP-LoRA Mode)
#
# Execution strategy (2-track parallel):
#   Track 1: SQuAD (15 epochs, 30 trials) - runs in background
#   Track 2: GLUE 9 tasks (3 epochs, 10 trials each) - runs sequentially
#   Both tracks run concurrently on a single GPU.
#
# All experiments run in FP-LoRA mode with:
# - lora_alpha: [0.01, 100] log-uniform
# - learning_rate: [1e-4, 1e-1] log-uniform
#
# Results saved to: /data/results/lora_fp/
#
# Usage:
#   bash run_lrtt_lora_sweep_batch_fp.sh
#

set -e  # Exit on error

# Configuration
PYTHON=/data/venvs/aihwkit_gpu/bin/python
SCRIPT=/data/LRTT_transformer/experiments/sweep_lrtt_lora_optuna.py
MODE=fp_lora
RESULTS_DIR=/data/results/lora_fp
RANK=8
TARGET_MODULES="query key value"

# Create results directory
mkdir -p ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}/logs

# Timestamp for this batch run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG=${RESULTS_DIR}/logs/batch_run_${TIMESTAMP}.log

echo "================================================================================"
echo "LRTT-LoRA Hyperparameter Sweep Batch Execution (FP-LoRA Mode)"
echo "  Track 1: SQuAD (background)"
echo "  Track 2: GLUE 9 tasks (sequential)"
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

log "=== BATCH RUN STARTED (2-TRACK PARALLEL) ==="
log "Mode: ${MODE}"
log "Python: ${PYTHON}"
log "Script: ${SCRIPT}"

# =============================================================================
# Track 1: SQuAD (background) - 15 epochs, 30 trials
# =============================================================================

log ""
log "=== [Track 1] Launching SQuAD sweep (background) ==="
log "Settings: 15 epochs, 30 trials, ${MODE}"

STUDY_NAME=squad_${MODE}_${TIMESTAMP}
STORAGE=sqlite:///${RESULTS_DIR}/optuna_squad_${MODE}.db

${PYTHON} ${SCRIPT} \
    --task squad \
    --mode ${MODE} \
    --rank ${RANK} \
    --target_modules ${TARGET_MODULES} \
    --n_trials 30 \
    --study_name ${STUDY_NAME} \
    --storage ${STORAGE} \
    > ${RESULTS_DIR}/logs/squad_${MODE}_${TIMESTAMP}.log 2>&1 &

SQUAD_PID=$!
log "SQuAD sweep launched (PID: ${SQUAD_PID}) -> logs/squad_${MODE}_${TIMESTAMP}.log"

# =============================================================================
# Track 2: GLUE Tasks (sequential) - 3 epochs, 10 trials each
# =============================================================================

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

GLUE_FAILED=0
GLUE_SUCCEEDED=0
TASK_NUM=1

for TASK in "${GLUE_TASKS[@]}"
do
    log ""
    log "=== [Track 2] GLUE ${TASK} (${TASK_NUM}/9) ==="
    log "Settings: 3 epochs, 10 trials, ${MODE}"

    STUDY_NAME=glue_${TASK}_${MODE}_${TIMESTAMP}
    STORAGE=sqlite:///${RESULTS_DIR}/optuna_glue_${TASK}_${MODE}.db

    ${PYTHON} ${SCRIPT} \
        --task glue \
        --task_name ${TASK} \
        --mode ${MODE} \
        --rank ${RANK} \
        --target_modules ${TARGET_MODULES} \
        --n_trials 10 \
        --study_name ${STUDY_NAME} \
        --storage ${STORAGE} \
        > ${RESULTS_DIR}/logs/glue_${TASK}_${MODE}_${TIMESTAMP}.log 2>&1 &

    GLUE_PID=$!
    log "${TASK} started (PID: ${GLUE_PID})"

    wait ${GLUE_PID}
    TASK_EXIT=$?

    if [ ${TASK_EXIT} -eq 0 ]; then
        log "✓ GLUE ${TASK} completed successfully"
        GLUE_SUCCEEDED=$((GLUE_SUCCEEDED + 1))
    else
        log "✗ GLUE ${TASK} failed with exit code ${TASK_EXIT}"
        GLUE_FAILED=$((GLUE_FAILED + 1))
    fi

    TASK_NUM=$((TASK_NUM + 1))
done

log ""
log "=== [Track 2] All GLUE tasks finished (${GLUE_SUCCEEDED}/9 succeeded) ==="

# =============================================================================
# Wait for Track 1 (SQuAD) if still running
# =============================================================================

if kill -0 ${SQUAD_PID} 2>/dev/null; then
    log "=== Waiting for Track 1 (SQuAD) to finish... ==="
fi

wait ${SQUAD_PID}
SQUAD_EXIT=$?

if [ ${SQUAD_EXIT} -eq 0 ]; then
    log "✓ SQuAD sweep completed successfully"
    SQUAD_STATUS="succeeded"
else
    log "✗ SQuAD sweep failed with exit code ${SQUAD_EXIT}"
    SQUAD_STATUS="failed"
fi

# =============================================================================
# Summary
# =============================================================================

TOTAL_SUCCEEDED=$((GLUE_SUCCEEDED + (SQUAD_EXIT == 0 ? 1 : 0)))
TOTAL_FAILED=$((GLUE_FAILED + (SQUAD_EXIT != 0 ? 1 : 0)))

log ""
log "=== BATCH RUN COMPLETED ==="
log "SQuAD: ${SQUAD_STATUS}"
log "GLUE: ${GLUE_SUCCEEDED}/9 succeeded, ${GLUE_FAILED}/9 failed"
log "Results directory: ${RESULTS_DIR}"

echo ""
echo "================================================================================"
echo "BATCH EXECUTION COMPLETE"
echo "================================================================================"
echo "SQuAD: ${SQUAD_STATUS}"
echo "GLUE: ${GLUE_SUCCEEDED}/9 succeeded"
echo "Check results at: ${RESULTS_DIR}"
echo "Check logs at: ${RESULTS_DIR}/logs/"
echo "Batch log: ${BATCH_LOG}"
echo "================================================================================"

# Create summary file
SUMMARY_FILE=${RESULTS_DIR}/batch_summary_${TIMESTAMP}.txt
{
    echo "LRTT-LoRA Hyperparameter Sweep Batch Summary (FP-LoRA Mode)"
    echo "=============================================================="
    echo ""
    echo "Timestamp: ${TIMESTAMP}"
    echo "Mode: ${MODE}"
    echo "Execution: 2-track parallel (SQuAD || GLUE sequential)"
    echo ""
    echo "Tasks:"
    echo "  Track 1: SQuAD (15 epochs, 30 trials) - ${SQUAD_STATUS}"
    for TASK in "${GLUE_TASKS[@]}"
    do
        echo "  Track 2: GLUE ${TASK} (3 epochs, 10 trials)"
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
    echo "  Device: FloatingPointDevice (FP-LoRA)"
    echo "  Optimizer: AnalogSGD"
    echo "  Batch size: 32"
    echo "  Max seq length (GLUE): 128"
    echo "  Max seq length (SQuAD): 384"
    echo "  Doc stride (SQuAD): 128"
    echo "  Warmup: 10% linear"
    echo "  Early stopping: patience=3"
} > ${SUMMARY_FILE}

log "Summary saved to: ${SUMMARY_FILE}"

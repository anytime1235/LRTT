#!/bin/bash
#
# PARALLEL Automated LRTT-LoRA Hyperparameter Sweep Batch Script
#
# This script runs 2 parallel tracks of hyperparameter sweeps:
# Track 1 & 2: SQuAD → GLUE tasks (순차 실행)
#
# Each track runs in 6T1C-LoRA mode with:
# - lora_alpha: [0.01, 100] log-uniform
# - learning_rate: [1e-4, 1e-2] log-uniform
#
# Results saved to: /data/results/lora_baseline/
#
# Usage:
#   bash run_lrtt_lora_sweep_batch_parallel.sh
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
BATCH_LOG=${RESULTS_DIR}/logs/batch_run_parallel_${TIMESTAMP}.log

echo "================================================================================"
echo "LRTT-LoRA PARALLEL Hyperparameter Sweep Batch Execution"
echo "================================================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Mode: ${MODE}"
echo "Parallel Tracks: 2"
echo "Results directory: ${RESULTS_DIR}"
echo "Log file: ${BATCH_LOG}"
echo "================================================================================"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${BATCH_LOG}
}

# Function to run one complete sweep track
run_sweep_track() {
    local TRACK_ID=$1
    local TRACK_TIMESTAMP="${TIMESTAMP}_track${TRACK_ID}"
    local TRACK_LOG=${RESULTS_DIR}/logs/track${TRACK_ID}_${TIMESTAMP}.log

    log "=== TRACK ${TRACK_ID} STARTED ===" >> ${TRACK_LOG}

    # SQuAD
    log "[Track ${TRACK_ID}] Starting SQuAD sweep (30 trials)" >> ${TRACK_LOG}
    STUDY_NAME=squad_${MODE}_${TRACK_TIMESTAMP}
    STORAGE=sqlite:///${RESULTS_DIR}/optuna_squad_${MODE}_track${TRACK_ID}.db

    ${PYTHON} ${SCRIPT} \
        --task squad \
        --mode ${MODE} \
        --rank ${RANK} \
        --target_modules ${TARGET_MODULES} \
        --n_trials 30 \
        --study_name ${STUDY_NAME} \
        --storage ${STORAGE} \
        > ${RESULTS_DIR}/logs/squad_${MODE}_${TRACK_TIMESTAMP}.log 2>&1

    log "[Track ${TRACK_ID}] SQuAD completed" >> ${TRACK_LOG}

    # GLUE tasks
    declare -a GLUE_TASKS=("sst2" "qqp" "mnli" "qnli" "mrpc" "rte" "cola" "stsb" "wnli")

    for TASK in "${GLUE_TASKS[@]}"
    do
        log "[Track ${TRACK_ID}] Starting GLUE ${TASK} sweep (10 trials)" >> ${TRACK_LOG}
        STUDY_NAME=glue_${TASK}_${MODE}_${TRACK_TIMESTAMP}
        STORAGE=sqlite:///${RESULTS_DIR}/optuna_glue_${TASK}_${MODE}_track${TRACK_ID}.db

        ${PYTHON} ${SCRIPT} \
            --task glue \
            --task_name ${TASK} \
            --mode ${MODE} \
            --rank ${RANK} \
            --target_modules ${TARGET_MODULES} \
            --n_trials 10 \
            --study_name ${STUDY_NAME} \
            --storage ${STORAGE} \
            > ${RESULTS_DIR}/logs/glue_${TASK}_${MODE}_${TRACK_TIMESTAMP}.log 2>&1

        log "[Track ${TRACK_ID}] ${TASK} completed" >> ${TRACK_LOG}
    done

    log "=== TRACK ${TRACK_ID} COMPLETED ===" >> ${TRACK_LOG}
}

log "=== PARALLEL BATCH RUN STARTED ==="
log "Mode: ${MODE}"
log "Python: ${PYTHON}"
log "Script: ${SCRIPT}"
log ""
log "Starting 2 parallel tracks..."

# Launch Track 1 in background
log "Launching Track 1..."
run_sweep_track 1 &
TRACK1_PID=$!
log "Track 1 started (PID: ${TRACK1_PID})"

# Launch Track 2 in background
log "Launching Track 2..."
run_sweep_track 2 &
TRACK2_PID=$!
log "Track 2 started (PID: ${TRACK2_PID})"

log ""
log "Both tracks running in parallel"
log "Track 1 PID: ${TRACK1_PID}"
log "Track 2 PID: ${TRACK2_PID}"
log "Waiting for both tracks to complete..."

# Wait for both tracks
wait ${TRACK1_PID}
TRACK1_EXIT=$?
log "Track 1 finished with exit code: ${TRACK1_EXIT}"

wait ${TRACK2_PID}
TRACK2_EXIT=$?
log "Track 2 finished with exit code: ${TRACK2_EXIT}"

# Summary
log ""
log "=== PARALLEL BATCH RUN COMPLETED ==="
log "Track 1 exit code: ${TRACK1_EXIT}"
log "Track 2 exit code: ${TRACK2_EXIT}"
log "Results directory: ${RESULTS_DIR}"
log "Logs: ${RESULTS_DIR}/logs/"

echo ""
echo "================================================================================"
echo "PARALLEL BATCH EXECUTION COMPLETE"
echo "================================================================================"
echo "Track 1 exit: ${TRACK1_EXIT}"
echo "Track 2 exit: ${TRACK2_EXIT}"
echo "Check results at: ${RESULTS_DIR}"
echo "Check logs at: ${RESULTS_DIR}/logs/"
echo "Batch log: ${BATCH_LOG}"
echo "================================================================================"

# Create summary file
SUMMARY_FILE=${RESULTS_DIR}/batch_summary_parallel_${TIMESTAMP}.txt
{
    echo "LRTT-LoRA PARALLEL Hyperparameter Sweep Batch Summary"
    echo "====================================================="
    echo ""
    echo "Timestamp: ${TIMESTAMP}"
    echo "Mode: ${MODE}"
    echo "Parallel Tracks: 2"
    echo ""
    echo "Each Track Completed:"
    echo "  1. SQuAD (3 epochs, 30 trials)"
    echo "  2. GLUE sst2 (3 epochs, 10 trials)"
    echo "  3. GLUE qqp (3 epochs, 10 trials)"
    echo "  4. GLUE mnli (3 epochs, 10 trials)"
    echo "  5. GLUE qnli (3 epochs, 10 trials)"
    echo "  6. GLUE mrpc (3 epochs, 10 trials)"
    echo "  7. GLUE rte (3 epochs, 10 trials)"
    echo "  8. GLUE cola (3 epochs, 10 trials)"
    echo "  9. GLUE stsb (3 epochs, 10 trials)"
    echo "  10. GLUE wnli (3 epochs, 10 trials)"
    echo ""
    echo "Results Location: ${RESULTS_DIR}"
    echo "Databases: ${RESULTS_DIR}/optuna_*_track1.db, optuna_*_track2.db"
    echo "Logs: ${RESULTS_DIR}/logs/"
    echo ""
    echo "Search Space:"
    echo "  lora_alpha: [0.01, 100] (log-uniform)"
    echo "  learning_rate: [1e-4, 1e-2] (log-uniform)"
    echo "  rank: ${RANK} (fixed)"
    echo "  target_modules: ${TARGET_MODULES}"
} > ${SUMMARY_FILE}

log "Summary saved to: ${SUMMARY_FILE}"

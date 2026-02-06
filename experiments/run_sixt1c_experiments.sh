#!/bin/bash
# Run sixt1c (6T1C) baseline experiments with nohup
# This script runs full training for QA and all GLUE tasks with sixt1c analog device
# Results are logged to wandb and saved to CSV

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Log directory
LOG_DIR="/data/AIMC_LoRA_logs"
mkdir -p "$LOG_DIR"

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/sixt1c_experiments_${TIMESTAMP}.log"

echo "=============================================="
echo "SIXT1C (6T1C) BASELINE EXPERIMENTS"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "=============================================="

# Change to experiments directory
cd "$SCRIPT_DIR"

# Run sixt1c experiments with nohup
nohup python run_baseline_experiments.py \
    --mode sixt1c \
    > "$LOG_FILE" 2>&1 &

# Get the PID
PID=$!
echo "Started sixt1c experiments with PID: $PID"
echo "PID $PID" > "${LOG_DIR}/sixt1c_experiments_${TIMESTAMP}.pid"

echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To stop:"
echo "  kill $PID"

#!/bin/bash
#
# Monitor LRTT-LoRA Sweep Progress
#
# Usage: bash monitor_sweep_progress.sh
#

RESULTS_DIR=/data/results/lora_baseline

echo "================================================================================"
echo "LRTT-LoRA Sweep Progress Monitor"
echo "================================================================================"
echo "Results directory: ${RESULTS_DIR}"
echo "================================================================================"

# Check if results directory exists
if [ ! -d "${RESULTS_DIR}" ]; then
    echo "⚠️  Results directory not found: ${RESULTS_DIR}"
    echo "No sweeps have been started yet."
    exit 0
fi

# Check running processes
echo ""
echo "=== Running Processes ==="
ps aux | grep "sweep_lrtt_lora_optuna.py" | grep -v grep || echo "No sweep processes running"

# Check log files
echo ""
echo "=== Recent Log Files ==="
if [ -d "${RESULTS_DIR}/logs" ]; then
    ls -lhrt ${RESULTS_DIR}/logs/*.log 2>/dev/null | tail -10 || echo "No log files found"
else
    echo "No logs directory found"
fi

# Check Optuna databases
echo ""
echo "=== Optuna Databases ==="
if ls ${RESULTS_DIR}/optuna_*.db 1> /dev/null 2>&1; then
    for db in ${RESULTS_DIR}/optuna_*.db; do
        echo "  $(basename $db)"
        # Count trials (requires sqlite3)
        if command -v sqlite3 &> /dev/null; then
            trial_count=$(sqlite3 $db "SELECT COUNT(*) FROM trials;" 2>/dev/null || echo "N/A")
            echo "    Trials: ${trial_count}"
        fi
    done
else
    echo "No Optuna databases found"
fi

# Check best parameters files
echo ""
echo "=== Best Parameters (JSON) ==="
if ls ${RESULTS_DIR}/best_params_*.json 1> /dev/null 2>&1; then
    for json in ${RESULTS_DIR}/best_params_*.json; do
        echo "  $(basename $json)"
        # Show best value if jq is available
        if command -v jq &> /dev/null; then
            best_value=$(jq -r '.best_value' $json 2>/dev/null || echo "N/A")
            echo "    Best value: ${best_value}"
        fi
    done
else
    echo "No best parameters files found yet"
fi

# Show last 20 lines of most recent log
echo ""
echo "=== Latest Log Output (last 20 lines) ==="
LATEST_LOG=$(ls -t ${RESULTS_DIR}/logs/*.log 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "From: $(basename ${LATEST_LOG})"
    echo "---"
    tail -20 "${LATEST_LOG}" 2>/dev/null || echo "Cannot read log file"
else
    echo "No log files available"
fi

echo ""
echo "================================================================================"
echo "To view full logs: tail -f ${RESULTS_DIR}/logs/<task>_*.log"
echo "To check specific task: grep '<task_name>' ${RESULTS_DIR}/logs/batch_run_*.log"
echo "================================================================================"

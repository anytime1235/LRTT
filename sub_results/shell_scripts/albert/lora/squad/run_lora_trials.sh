#!/bin/bash
# Run optuna trials one at a time in separate processes to avoid GPU memory fragmentation
# Usage: nohup bash /data/run_lora_trials.sh 26 > /data/logs/albert_squad_lora_all.log 2>&1 &

N_TRIALS=${1:-26}
PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_squad_lora.py

for i in $(seq 1 $N_TRIALS); do
    echo ""
    echo "=========================================="
    echo "[Runner] Trial $i / $N_TRIALS"
    echo "=========================================="
    $PYTHON $SCRIPT --n-trials 1 --lora-target all --epochs 5 --optimizer AnalogSGD
    EXIT_CODE=$?
    echo "[Runner] Process exited with code $EXIT_CODE"
    # Brief pause to ensure GPU memory is fully released
    sleep 3
done

echo ""
echo "[Runner] All $N_TRIALS trials completed."

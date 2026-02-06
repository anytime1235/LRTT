#!/bin/bash
# BERT-base LRTT Optuna Sweep Launcher
#
# Usage:
#   ./run_optuna_sweep.sh                    # Default: 200 trials, 1 job
#   ./run_optuna_sweep.sh 500                # 500 trials
#   ./run_optuna_sweep.sh 500 4              # 500 trials, 4 parallel jobs
#   ./run_optuna_sweep.sh 500 4 my_study     # With custom study name

N_TRIALS=${1:-200}
N_JOBS=${2:-1}
STUDY_NAME=${3:-"lrtt_bert_base_$(date +%Y%m%d_%H%M%S)"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/home/jovyan/work/ml/.venv310/bin/python"

LOG_DIR="/tmp/lrtt_optuna_logs"
mkdir -p $LOG_DIR

LOG_FILE="$LOG_DIR/optuna_${STUDY_NAME}.log"

echo "=============================================="
echo "BERT-base LRTT Optuna Sweep"
echo "=============================================="
echo "Trials: $N_TRIALS"
echo "Jobs: $N_JOBS"
echo "Study: $STUDY_NAME"
echo "Log: $LOG_FILE"
echo "=============================================="
echo ""
echo "Starting in background..."
echo "Monitor with: tail -f $LOG_FILE"
echo ""

nohup $VENV_PYTHON $SCRIPT_DIR/sweep_bert_base_optuna.py \
    --n_trials $N_TRIALS \
    --n_jobs $N_JOBS \
    --study_name $STUDY_NAME \
    > $LOG_FILE 2>&1 &

echo "PID: $!"
echo "Done. Check log: tail -f $LOG_FILE"

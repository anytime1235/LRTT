#!/bin/bash
# Sequential ALBERT GLUE sweep with --lora-target all
# Ordered by dataset size (smallest first)
# SST-2 and smaller: 30 trials, larger: 10 trials
# Sampler: TPE, learn_out_scaling=True for both TikiTaka and SingleRPU

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_tiki.py
LOGDIR=/data/results/tikitakav1
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --lora-target all --sampler tpe"

mkdir -p "$LOGDIR"

run_task() {
    local task=$1
    local trials=$2
    local logfile="$LOGDIR/albert_glue_all_${task}.log"
    echo "=============================================="
    echo "[$(date)] Starting $task ($trials trials)"
    echo "=============================================="
    $PYTHON $SCRIPT --task "$task" --n-trials "$trials" $COMMON \
        >> "$logfile" 2>&1
    echo "[$(date)] Finished $task (exit code: $?)"
    echo ""
}

# Small datasets (<=SST-2): 30 trials
run_task rte   30
run_task mrpc  30
run_task stsb  30
run_task cola  30
run_task sst2  30

# Large datasets (>SST-2): 10 trials
run_task qnli  10
run_task qqp   10
run_task mnli  10

echo "[$(date)] All tasks complete."

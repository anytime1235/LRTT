#!/bin/bash
# TikiTaka v1 (scale_transfer_lr=False) with desired_bl grid [31,60] + transfer_lr sweep [0.01,1.0]
# 50 trials each for: rte, mrpc, cola, stsb, sst2

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_tiki.py
COMMON="--no-wd --no-momentum --no-nesterov --lora-target attn --no-v2 --bl-grid 31 60 --tlr-sweep 0.01 1.0 --n-trials 50 --sampler tpe"

for TASK in rte mrpc cola stsb sst2; do
    echo ""
    echo "================================================================"
    echo "  Starting: $TASK (50 trials, v1, bl-grid=[31,60], tlr=[0.01,1.0])"
    echo "  $(date)"
    echo "================================================================"
    $PYTHON $SCRIPT --task $TASK $COMMON
    echo "  Finished: $TASK at $(date)"
done

echo ""
echo "================================================================"
echo "  ALL TASKS COMPLETE at $(date)"
echo "================================================================"

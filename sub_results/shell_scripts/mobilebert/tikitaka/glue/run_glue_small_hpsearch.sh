#!/bin/bash
# [Track 2] HP search for small GLUE tasks (CoLA -> STS-B -> MRPC -> RTE)
# 10 trials each, 5 epochs, 5% warmup, sequential execution
# Grid: lr=[0.1,0.5,1.0] x te=[1,10] x flr=[1.0,5.0] = 12 combos

set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_mobilebert_glue_tiki.py
LOGDIR=/data/results/tikitakav1

TASKS=("cola" "stsb" "mrpc" "rte")

for TASK in "${TASKS[@]}"; do
    STUDY_NAME="mobilebert_glue_tiki_${TASK}_hpsearch_10t_5ep_w5pct"
    echo "============================================================"
    echo "[Track 2] Starting: ${TASK} (study: ${STUDY_NAME})"
    echo "============================================================"
    $PYTHON $SCRIPT \
        --task "${TASK}" \
        --study-name "${STUDY_NAME}" \
        --n-trials 12 \
        --epochs 5 \
        --warmup-ratio 0.05 \
        --optimizer AnalogSGD \
        --no-wd --no-momentum --no-nesterov \
        --lora-target qkv \
        --fix-lr 0.1 0.5 1.0 \
        --fix-te 1 10 \
        --fix-flr 1.0 5.0
    echo "Finished: ${TASK}"
    echo ""
done

echo "All Track 2 tasks completed!"

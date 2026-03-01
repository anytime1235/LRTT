#!/bin/bash
# Run GLUE tasks with SQUAD best params (lr=0.1, te=1, flr=1.0)
# 5 epochs, 5% warmup, single trial per task
# Order: dataset size descending (SST-2 skipped)
# MNLI(392k) > QQP(363k) > QNLI(104k) > CoLA(8.5k) > STS-B(5.7k) > MRPC(3.6k) > RTE(2.4k)

set -e

TASKS=("mnli" "qqp" "qnli" "cola" "stsb" "mrpc" "rte")

for TASK in "${TASKS[@]}"; do
    STUDY_NAME="mobilebert_glue_tiki_${TASK}_squad_best_5ep_w5pct"
    echo "============================================================"
    echo "Starting GLUE task: ${TASK} (study: ${STUDY_NAME})"
    echo "============================================================"
    /data/venvs/lrtt/bin/python /data/optuna_mobilebert_glue_tiki.py \
        --task "${TASK}" \
        --study-name "${STUDY_NAME}" \
        --n-trials 1 \
        --epochs 5 \
        --warmup-ratio 0.05 \
        --optimizer AnalogSGD \
        --no-wd --no-momentum --no-nesterov \
        --lora-target qkv \
        --fix-lr 0.1 \
        --fix-te 1 \
        --fix-flr 1.0
    echo "Finished GLUE task: ${TASK}"
    echo ""
done

echo "All GLUE tasks completed!"

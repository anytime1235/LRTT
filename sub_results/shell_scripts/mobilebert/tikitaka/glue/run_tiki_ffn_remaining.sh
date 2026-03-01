#!/bin/bash
# TikiTaka v1 - FFN target layer, learn_out_scaling=False
# 누락된 MRPC, CoLA, STS-B 추가 실행 (각 30 trials)
set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_mobilebert_glue_tiki.py
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --lora-target ffn --sampler tpe"
LOG_DIR=/data/results/tikitakav1
LOG=${LOG_DIR}/run_ffn_tpe_remaining.log
SUFFIX="_tpe"

mkdir -p ${LOG_DIR}

echo "==========================================" | tee -a $LOG
echo "TikiTaka v1 FFN - Remaining tasks"         | tee -a $LOG
echo "MRPC(30), CoLA(30), STS-B(30)"             | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 1. MRPC - 30 trials
echo "[1/3] MRPC - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task mrpc --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_mrpc_bs64_sgd_nowd_nomom_nonest_ffn${SUFFIX} \
  2>&1 | tee -a $LOG
echo "MRPC done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 2. CoLA - 30 trials
echo "[2/3] CoLA - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task cola --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_cola_bs64_sgd_nowd_nomom_nonest_ffn${SUFFIX} \
  2>&1 | tee -a $LOG
echo "CoLA done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 3. STS-B - 30 trials
echo "[3/3] STS-B - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task stsb --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_stsb_bs64_sgd_nowd_nomom_nonest_ffn${SUFFIX} \
  2>&1 | tee -a $LOG
echo "STS-B done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

echo "==========================================" | tee -a $LOG
echo "ALL DONE at $(date)" | tee -a $LOG
echo "==========================================" | tee -a $LOG

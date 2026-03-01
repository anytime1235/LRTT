#!/bin/bash
# TikiTaka v1 - FFN target layer + learn_out_scaling=False
# AnalogSGD, no-wd, no-momentum, no-nesterov, TPE sampler
# RTE/SST-2: 30 trials, QQP/QNLI/MNLI: 10 trials
set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_mobilebert_glue_tiki.py
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --lora-target ffn --sampler tpe"
LOG_DIR=/data/results/tikitakav1
LOG=${LOG_DIR}/run_ffn_tpe_all.log
SUFFIX="_tpe"

mkdir -p ${LOG_DIR}

echo "==========================================" | tee -a $LOG
echo "TikiTaka v1 FFN + learn_out_scaling=False" | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
echo "" | tee -a $LOG
echo "Tasks: RTE(30), SST-2(30), QQP(10), QNLI(10), MNLI(10)" | tee -a $LOG
echo "Config: AnalogSGD, no-wd, no-momentum, no-nesterov, TPE, ffn" | tee -a $LOG
echo "" | tee -a $LOG

# 1. RTE - 30 trials
echo "[1/5] RTE - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task rte --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_rte_bs64_sgd_nowd_nomom_nonest_ffn${SUFFIX} \
  2>&1 | tee -a $LOG
echo "RTE done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 2. SST-2 - 30 trials
echo "[2/5] SST-2 - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task sst2 --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_sst2_bs64_sgd_nowd_nomom_nonest_ffn${SUFFIX} \
  2>&1 | tee -a $LOG
echo "SST-2 done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 3. QQP - 10 trials
echo "[3/5] QQP - 10 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task qqp --n-trials 10 $COMMON \
  --study-name mobilebert_glue_tiki_qqp_bs64_sgd_nowd_nomom_nonest_ffn${SUFFIX} \
  2>&1 | tee -a $LOG
echo "QQP done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 4. QNLI - 10 trials
echo "[4/5] QNLI - 10 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task qnli --n-trials 10 $COMMON \
  --study-name mobilebert_glue_tiki_qnli_bs64_sgd_nowd_nomom_nonest_ffn${SUFFIX} \
  2>&1 | tee -a $LOG
echo "QNLI done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 5. MNLI - 10 trials
echo "[5/5] MNLI - 10 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task mnli --n-trials 10 $COMMON \
  --study-name mobilebert_glue_tiki_mnli_bs64_sgd_nowd_nomom_nonest_ffn${SUFFIX} \
  2>&1 | tee -a $LOG
echo "MNLI done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

echo "==========================================" | tee -a $LOG
echo "ALL DONE at $(date)" | tee -a $LOG
echo "==========================================" | tee -a $LOG

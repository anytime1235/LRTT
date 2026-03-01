#!/bin/bash
# TikiTaka v1 - FFN target layer, learn_out_scaling=True
# AnalogSGD, no-wd, no-momentum, no-nesterov, TPE sampler
# Dataset size order: RTE(195) < MRPC(290) < STS-B(450) < CoLA(670) < SST-2(5265) < QNLI(8185) < QQP(28430) < MNLI(30680)
# Small tasks: 30 trials, Large tasks (QQP/QNLI/MNLI): 10 trials
set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_mobilebert_glue_tiki.py
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --lora-target ffn --learn-out-scaling --sampler tpe"
LOG_DIR=/data/results/tikitakav1
LOG=${LOG_DIR}/run_ffn_los_tpe_all.log
S="_los_tpe"

mkdir -p ${LOG_DIR}

echo "==========================================" | tee $LOG
echo "TikiTaka v1 FFN (learn_out_scaling=True)"  | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
echo "Order: RTE > MRPC > STS-B > CoLA > SST-2 > QNLI > QQP > MNLI" | tee -a $LOG
echo "" | tee -a $LOG

# 1. RTE - 30 trials (195 samples)
echo "[1/8] RTE - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task rte --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_rte_bs32_sgd_nowd_nomom_nonest_ffn${S} \
  2>&1 | tee -a $LOG
echo "RTE done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 2. MRPC - 30 trials (290 samples)
echo "[2/8] MRPC - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task mrpc --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_mrpc_bs32_sgd_nowd_nomom_nonest_ffn${S} \
  2>&1 | tee -a $LOG
echo "MRPC done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 3. STS-B - 30 trials (450 samples)
echo "[3/8] STS-B - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task stsb --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_stsb_bs16_sgd_nowd_nomom_nonest_ffn${S} \
  2>&1 | tee -a $LOG
echo "STS-B done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 4. CoLA - 30 trials (670 samples)
echo "[4/8] CoLA - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task cola --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_cola_bs16_sgd_nowd_nomom_nonest_ffn${S} \
  2>&1 | tee -a $LOG
echo "CoLA done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 5. SST-2 - 30 trials (5265 samples)
echo "[5/8] SST-2 - 30 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task sst2 --n-trials 30 $COMMON \
  --study-name mobilebert_glue_tiki_sst2_bs32_sgd_nowd_nomom_nonest_ffn${S} \
  2>&1 | tee -a $LOG
echo "SST-2 done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 6. QNLI - 10 trials (8185 samples)
echo "[6/8] QNLI - 10 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task qnli --n-trials 10 $COMMON \
  --study-name mobilebert_glue_tiki_qnli_bs32_sgd_nowd_nomom_nonest_ffn${S} \
  2>&1 | tee -a $LOG
echo "QNLI done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 7. QQP - 10 trials (28430 samples)
echo "[7/8] QQP - 10 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task qqp --n-trials 10 $COMMON \
  --study-name mobilebert_glue_tiki_qqp_bs128_sgd_nowd_nomom_nonest_ffn${S} \
  2>&1 | tee -a $LOG
echo "QQP done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# 8. MNLI - 10 trials (30680 samples)
echo "[8/8] MNLI - 10 trials ($(date))" | tee -a $LOG
$PYTHON $SCRIPT --task mnli --n-trials 10 $COMMON \
  --study-name mobilebert_glue_tiki_mnli_bs128_sgd_nowd_nomom_nonest_ffn${S} \
  2>&1 | tee -a $LOG
echo "MNLI done at $(date)" | tee -a $LOG
echo "" | tee -a $LOG

echo "==========================================" | tee -a $LOG
echo "ALL DONE at $(date)" | tee -a $LOG
echo "==========================================" | tee -a $LOG

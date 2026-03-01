#!/bin/bash
# Sequential GLUE LoRA sweep: smallest dataset first
# SST-2 이하 (N_train <= 67,349): 30 trials
# SST-2 초과: 10 trials
# Settings from Albert_setup.txt (2x epochs)

set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_lora.py
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --warm-alpha --lora-target all --warmup-ratio 0.05"
LOG_DIR=/data/results/Analoglora_all

echo "=========================================="
echo "GLUE LoRA sweep - all tasks (sequential)"
echo "=========================================="
date

# 1. RTE (N=2,490) - 30 trials
echo ""
echo "[1/8] RTE - 30 trials"
$PYTHON $SCRIPT --task rte --n-trials 30 $COMMON 2>&1 | tee ${LOG_DIR}/rte/nohup_lora_sweep.log
echo "RTE done at $(date)"

# 2. MRPC (N=3,668) - 30 trials
echo ""
echo "[2/8] MRPC - 30 trials"
$PYTHON $SCRIPT --task mrpc --n-trials 30 $COMMON 2>&1 | tee ${LOG_DIR}/mrpc/nohup_lora_sweep.log
echo "MRPC done at $(date)"

# 3. STS-B (N=5,749) - 30 trials
echo ""
echo "[3/8] STS-B - 30 trials"
$PYTHON $SCRIPT --task stsb --n-trials 30 $COMMON 2>&1 | tee ${LOG_DIR}/stsb/nohup_lora_sweep.log
echo "STS-B done at $(date)"

# 4. CoLA (N=8,551) - 30 trials
echo ""
echo "[4/8] CoLA - 30 trials"
$PYTHON $SCRIPT --task cola --n-trials 30 $COMMON 2>&1 | tee ${LOG_DIR}/cola/nohup_lora_sweep.log
echo "CoLA done at $(date)"

# 5. SST-2 (N=67,349) - 30 trials
echo ""
echo "[5/8] SST-2 - 30 trials"
$PYTHON $SCRIPT --task sst2 --n-trials 30 $COMMON 2>&1 | tee ${LOG_DIR}/sst2/nohup_lora_sweep.log
echo "SST-2 done at $(date)"

# 6. QNLI (N=104,743) - 10 trials
echo ""
echo "[6/8] QNLI - 10 trials"
$PYTHON $SCRIPT --task qnli --n-trials 10 $COMMON 2>&1 | tee ${LOG_DIR}/qnli/nohup_lora_sweep.log
echo "QNLI done at $(date)"

# 7. QQP (N=363,846) - 10 trials
echo ""
echo "[7/8] QQP - 10 trials"
$PYTHON $SCRIPT --task qqp --n-trials 10 $COMMON 2>&1 | tee ${LOG_DIR}/qqp/nohup_lora_sweep.log
echo "QQP done at $(date)"

# 8. MNLI (N=392,702) - 10 trials
echo ""
echo "[8/8] MNLI - 10 trials"
$PYTHON $SCRIPT --task mnli --n-trials 10 $COMMON 2>&1 | tee ${LOG_DIR}/mnli/nohup_lora_sweep.log
echo "MNLI done at $(date)"

echo ""
echo "=========================================="
echo "ALL GLUE TASKS COMPLETE"
echo "=========================================="
date

#!/bin/bash
# Tiki-Taka v1 rerun: single trial per task with best LR from sweep
# Purpose: verify best results are reproducible (higher than frozen baseline)
# Sequential execution.

set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_tiki.py
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --lora-target all --sampler grid --n-trials 1"
LOG_DIR=/data/results/tikitakav1

echo "=========================================="
echo "Tiki-Taka v1 best LR rerun - 5 GLUE tasks"
echo "Sequential execution"
echo "=========================================="
date

# 1. RTE (best lr=0.0625, te=16, flr=0.0264)
echo ""
echo "[1/5] RTE"
$PYTHON $SCRIPT --task rte $COMMON \
  --fix-lr 0.06251373574521749 \
  --fix-te 16 \
  --fix-flr 0.026364803038431653 \
  --study-name albert_glue_tiki_rte_bs32_sgd_nowd_nomom_nonest_all_rerun \
  2>&1 | tee ${LOG_DIR}/tiki_best_rerun_rte.log
echo "RTE done at $(date)"

# 2. MRPC (best lr=0.0625, te=16, flr=0.0264)
echo ""
echo "[2/5] MRPC"
$PYTHON $SCRIPT --task mrpc $COMMON \
  --fix-lr 0.06251373574521749 \
  --fix-te 16 \
  --fix-flr 0.026364803038431653 \
  --study-name albert_glue_tiki_mrpc_bs32_sgd_nowd_nomom_nonest_all_rerun \
  2>&1 | tee ${LOG_DIR}/tiki_best_rerun_mrpc.log
echo "MRPC done at $(date)"

# 3. STS-B (best lr=0.0870, te=50, flr=0.0124)
echo ""
echo "[3/5] STS-B"
$PYTHON $SCRIPT --task stsb $COMMON \
  --fix-lr 0.08702154746159006 \
  --fix-te 50 \
  --fix-flr 0.012376241463860878 \
  --study-name albert_glue_tiki_stsb_bs16_sgd_nowd_nomom_nonest_all_rerun \
  2>&1 | tee ${LOG_DIR}/tiki_best_rerun_stsb.log
echo "STS-B done at $(date)"

# 4. CoLA (best lr=0.0164, te=32, flr=0.1675)
echo ""
echo "[4/5] CoLA"
$PYTHON $SCRIPT --task cola $COMMON \
  --fix-lr 0.016376815637377924 \
  --fix-te 32 \
  --fix-flr 0.1674581183414637 \
  --study-name albert_glue_tiki_cola_bs16_sgd_nowd_nomom_nonest_all_rerun \
  2>&1 | tee ${LOG_DIR}/tiki_best_rerun_cola.log
echo "CoLA done at $(date)"

# 5. SST-2 (best lr=0.0233, te=79, flr=0.0346)
echo ""
echo "[5/5] SST-2"
$PYTHON $SCRIPT --task sst2 $COMMON \
  --fix-lr 0.023345864076016236 \
  --fix-te 79 \
  --fix-flr 0.034587052147518116 \
  --study-name albert_glue_tiki_sst2_bs32_sgd_nowd_nomom_nonest_all_rerun \
  2>&1 | tee ${LOG_DIR}/tiki_best_rerun_sst2.log
echo "SST-2 done at $(date)"

echo ""
echo "=========================================="
echo "ALL DONE at $(date)"
echo "=========================================="

#!/bin/bash
# HP search for MNLI, QQP, QNLI using TPE sampler
# Search ranges based on SST-2/SQuAD optimal (lr~0.1, te=1~10, fast_lr=1~5):
#   lr: [0.01, 2.0] log-uniform
#   te: [1, 5, 10, 50] categorical
#   fast_lr: [0.5, 10.0] log-uniform
# 10 trials each, 5 epochs, 5% warmup

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_mobilebert_glue_tiki.py

COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --batch-size 64 --epochs 5 --warmup-ratio 0.05 --lora-target qkv --sampler tpe --tpe-lr-range 0.01 2.0 --tpe-te-choices 1 5 10 50 --tpe-flr-range 0.5 10.0 --n-trials 10"

echo "$(date) | Starting MNLI TPE HP search (10 trials)"
$PYTHON $SCRIPT --task mnli --study-name mobilebert_glue_tiki_mnli_tpe_10t_5ep_w5pct $COMMON

echo ""
echo "$(date) | Starting QQP TPE HP search (10 trials)"
$PYTHON $SCRIPT --task qqp --study-name mobilebert_glue_tiki_qqp_tpe_10t_5ep_w5pct $COMMON

echo ""
echo "$(date) | Starting QNLI TPE HP search (10 trials)"
$PYTHON $SCRIPT --task qnli --study-name mobilebert_glue_tiki_qnli_tpe_10t_5ep_w5pct $COMMON

echo ""
echo "$(date) | All 3 tasks completed!"

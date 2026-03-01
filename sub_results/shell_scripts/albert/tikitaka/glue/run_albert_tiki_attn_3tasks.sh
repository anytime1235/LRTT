#!/bin/bash
# Sequential TikiTaka v1 HPO: QNLI -> QQP -> MNLI
# Target: attn (query/key/value/dense)
# units_in_mbatch=False, learn_out_scaling=False, AnalogSGD
#
# transfer_every choices: ~1~100 transfers/epoch per task
#   transfers/epoch = train_samples / te
#   QNLI (~104,743 samples): 1047(100x) 5237(20x) 21000(5x) 105000(1x)
#   QQP  (~363,846 samples): 3638(100x) 18192(20x) 73000(5x) 364000(1x)
#   MNLI (~392,702 samples): 3927(100x) 19635(20x) 79000(5x) 393000(1x)
#
# Usage:
#   nohup bash /data/run_albert_tiki_attn_3tasks.sh > /data/results/tikitakav1/attn_3tasks.log 2>&1 &

set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_tiki.py
N_TRIALS=30

COMMON_ARGS="--lora-target attn \
  --sampler tpe \
  --n-trials ${N_TRIALS} \
  --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov \
  --tpe-lr-range 0.005 0.15 \
  --tpe-tlr-range 0.01 1.0 \
  --tpe-flr-range 0.001 0.1"

# Per-task te choices: ~100x, ~20x, ~5x, ~1x transfers/epoch
QNLI_TE="--tpe-te-choices 1047 5237 21000 105000"
QQP_TE="--tpe-te-choices 3638 18192 73000 364000"
MNLI_TE="--tpe-te-choices 3927 19635 79000 393000"

echo "============================================"
echo "TikiTaka v1 attn HPO (units_in_mbatch=False, learn_out_scaling=False)"
echo "Tasks: QNLI -> QQP -> MNLI"
echo "Trials per task: ${N_TRIALS}"
echo "transfer_every: per-task (~1-100 transfers/epoch)"
echo "Started: $(date)"
echo "============================================"

# Enqueue seed trial: lr=0.03, te=~20 transfers/epoch, flr=0.01, tlr=0.1
echo ""
echo "[1/3] QNLI starting: $(date)"
$PYTHON $SCRIPT --task qnli $COMMON_ARGS $QNLI_TE --enqueue 0.03 5237 0.01 0.1
echo "[1/3] QNLI finished: $(date)"

echo ""
echo "[2/3] QQP starting: $(date)"
$PYTHON $SCRIPT --task qqp $COMMON_ARGS $QQP_TE --enqueue 0.03 18192 0.01 0.1
echo "[2/3] QQP finished: $(date)"

echo ""
echo "[3/3] MNLI starting: $(date)"
$PYTHON $SCRIPT --task mnli $COMMON_ARGS $MNLI_TE --enqueue 0.03 19635 0.01 0.1
echo "[3/3] MNLI finished: $(date)"

echo ""
echo "============================================"
echo "All 3 tasks completed: $(date)"
echo "============================================"

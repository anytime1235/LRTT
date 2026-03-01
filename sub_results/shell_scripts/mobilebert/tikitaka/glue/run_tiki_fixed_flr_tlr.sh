#!/bin/bash
# TikiTaka v1: fast_lr=1.0, transfer_lr=1.0 고정, lr만 탐색, 10 trials each
# Tasks: RTE, MRPC, COLA

PYTHON=/data/venvs/lrtt/bin/python
cd /data

COMMON="--tpe-flr-range 1.0 1.0 --tpe-tlr-range 1.0 1.0 --n-trials 10 --lora-target qkv --nontarget-digital --learn-out-scaling --optimizer AnalogAdam --no-wd --no-momentum --no-nesterov"

echo "=== RTE (10 trials, flr=1.0, tlr=1.0) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task rte $COMMON

echo ""
echo "=== MRPC (10 trials, flr=1.0, tlr=1.0) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task mrpc $COMMON

echo ""
echo "=== COLA (10 trials, flr=1.0, tlr=1.0) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task cola $COMMON

echo ""
echo "=== ALL DONE ==="

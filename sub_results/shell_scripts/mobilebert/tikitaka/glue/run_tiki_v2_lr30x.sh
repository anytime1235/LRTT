#!/bin/bash
# TikiTaka v2: auto_scale + use_v2, lr upper 30x, flr=1.0 tlr=1.0 fixed
# Tasks: RTE, MRPC, COLA — 20 trials each

PYTHON=/data/venvs/lrtt/bin/python
cd /data

COMMON="--tpe-flr-range 1.0 1.0 --tpe-tlr-range 1.0 1.0 --auto-scale --use-v2 --n-trials 20 --lora-target qkv --nontarget-digital --learn-out-scaling --optimizer AnalogAdam --no-wd --no-momentum --no-nesterov"

echo "=== RTE (20 trials, v2+autoscale, lr_upper=30x) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task rte $COMMON

echo ""
echo "=== MRPC (20 trials, v2+autoscale, lr_upper=30x) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task mrpc $COMMON

echo ""
echo "=== COLA (20 trials, v2+autoscale, lr_upper=30x) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task cola $COMMON

echo ""
echo "=== ALL DONE ==="

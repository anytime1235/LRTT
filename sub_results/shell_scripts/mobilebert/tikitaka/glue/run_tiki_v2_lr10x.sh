#!/bin/bash
set -e
# TikiTaka v2: auto_scale + use_v2, lr upper 10x, bl sweep 1-31, flr=1.0 tlr=1.0 fixed
# Tasks: RTE, MRPC, COLA, STSB, SST2 — 30 trials each

PYTHON=/data/venvs/lrtt/bin/python
cd /data

COMMON="--tpe-flr-range 1.0 1.0 --tpe-tlr-range 1.0 1.0 --auto-scale --use-v2 --lr-upper-mult 10 --bl-sweep 1 31 --n-trials 30 --lora-target qkv --nontarget-digital --learn-out-scaling --optimizer AnalogAdam --no-wd --no-momentum --no-nesterov"

echo "=== RTE (30 trials) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task rte $COMMON

echo ""
echo "=== MRPC (30 trials) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task mrpc $COMMON

echo ""
echo "=== COLA (30 trials) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task cola $COMMON

echo ""
echo "=== STSB (30 trials) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task stsb $COMMON

echo ""
echo "=== SST2 (30 trials) ==="
$PYTHON optuna_mobilebert_glue_tiki.py --task sst2 $COMMON

echo ""
echo "=== ALL DONE ==="

#!/bin/bash
# Sequential run: RTE -> MRPC -> CoLA -> STS-B -> SST-2
# desired_bl grid [31,60], transfer_lr log sweep [0.01,1.0], scale_transfer_lr=False
# 50 trials each

COMMON_ARGS="--tpe-flr-range 1.0 1.0 --tpe-tlr-range 0.01 1.0 --auto-scale --use-v2 --no-scale-transfer-lr --lr-upper-mult 10 --bl-grid 31 60 --n-trials 50 --lora-target qkv --nontarget-digital --learn-out-scaling --optimizer AnalogAdam --no-wd --no-momentum --no-nesterov"

cd /data

echo "=========================================="
echo "[1/5] RTE starting at $(date)"
echo "=========================================="
/data/venvs/lrtt/bin/python optuna_mobilebert_glue_tiki.py --task rte $COMMON_ARGS

echo "=========================================="
echo "[2/5] MRPC starting at $(date)"
echo "=========================================="
/data/venvs/lrtt/bin/python optuna_mobilebert_glue_tiki.py --task mrpc $COMMON_ARGS

echo "=========================================="
echo "[3/5] CoLA starting at $(date)"
echo "=========================================="
/data/venvs/lrtt/bin/python optuna_mobilebert_glue_tiki.py --task cola $COMMON_ARGS

echo "=========================================="
echo "[4/5] STS-B starting at $(date)"
echo "=========================================="
/data/venvs/lrtt/bin/python optuna_mobilebert_glue_tiki.py --task stsb $COMMON_ARGS

echo "=========================================="
echo "[5/5] SST-2 starting at $(date)"
echo "=========================================="
/data/venvs/lrtt/bin/python optuna_mobilebert_glue_tiki.py --task sst2 $COMMON_ARGS

echo "=========================================="
echo "All tasks completed at $(date)"
echo "=========================================="

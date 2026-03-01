#!/bin/bash
# Sequential TikiTaka v1 HPO: RTE -> MRPC -> STS-B -> SST-2
# Target: attn (query/key/value/dense)
# units_in_mbatch=False, learn_out_scaling=False, AnalogSGD
# te range: 100x/ep ~ 10x/ep transfer (suggest_int)
# Seed trial: ~25x/ep, LR from LoRA DB best
#
# Usage:
#   nohup bash /data/run_albert_tiki_attn_4tasks.sh > /data/results/tikitakav1/attn_4tasks.log 2>&1 &

set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_tiki.py
N_TRIALS=30

COMMON="--lora-target attn \
  --sampler tpe \
  --n-trials ${N_TRIALS} \
  --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov \
  --tpe-lr-range 0.005 0.15 \
  --tpe-tlr-range 0.01 1.0 \
  --tpe-flr-range 0.001 0.1"

# te range [100x/ep, 10x/ep] | seed ~25x/ep
# enqueue: LR TE FLR TLR

echo "============================================"
echo "TikiTaka v1 attn HPO (units_in_mbatch=False, learn_out_scaling=False)"
echo "Tasks: RTE -> MRPC -> STS-B -> SST-2"
echo "te: 100x/ep ~ 10x/ep | seed: ~25x/ep"
echo "Trials per task: ${N_TRIALS}"
echo "Started: $(date)"
echo "============================================"

echo ""
echo "[1/4] RTE starting: $(date)"
$PYTHON $SCRIPT --task rte $COMMON \
  --tpe-te-choices 25 250 \
  --enqueue 0.0625 100 0.026 0.1
echo "[1/4] RTE finished: $(date)"

echo ""
echo "[2/4] MRPC starting: $(date)"
$PYTHON $SCRIPT --task mrpc $COMMON \
  --tpe-te-choices 37 368 \
  --enqueue 0.0625 150 0.026 0.1
echo "[2/4] MRPC finished: $(date)"

echo ""
echo "[3/4] STS-B starting: $(date)"
$PYTHON $SCRIPT --task stsb $COMMON \
  --tpe-te-choices 58 576 \
  --enqueue 0.087 230 0.012 0.1
echo "[3/4] STS-B finished: $(date)"

echo ""
echo "[4/4] SST-2 starting: $(date)"
$PYTHON $SCRIPT --task sst2 $COMMON \
  --tpe-te-choices 674 6736 \
  --enqueue 0.023 2700 0.035 0.1
echo "[4/4] SST-2 finished: $(date)"

echo ""
echo "============================================"
echo "All 4 tasks completed: $(date)"
echo "============================================"

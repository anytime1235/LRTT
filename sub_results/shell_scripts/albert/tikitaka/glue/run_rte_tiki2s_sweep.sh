#!/bin/bash
# RTE 2-Stage TikiTaka sweep: te=12 fixed, fast_lr/transfer_lr TPE
# units_in_mbatch=True, FIXED_LR=0.01, OUT_SCALING_LR=0.01

set -e

# Wait for QNLI pretrain to finish
QNLI_PID=659759
if kill -0 $QNLI_PID 2>/dev/null; then
    echo "Waiting for QNLI pretrain (PID $QNLI_PID) to finish..."
    while kill -0 $QNLI_PID 2>/dev/null; do
        sleep 30
    done
    echo "QNLI pretrain finished. Starting RTE sweep..."
    sleep 5
fi

cd /data
export PATH="/data/venvs/lrtt/bin:$PATH"

echo "=== RTE 2-Stage TikiTaka Sweep ==="
echo "te=12 (fixed), fast_lr/transfer_lr TPE"
echo "units_in_mbatch=True, lr=0.01 (fixed), out_scaling_lr=0.01"
echo "Start: $(date)"

python optuna_albert_glue_tiki_2stage.py \
    --task rte \
    --te-choices 12 \
    --n-trials 30 \
    --optimizer AnalogSGD \
    --no-momentum \
    --no-nesterov \
    --pretrain-ckpt /data/classifier_ckpt/rte_adam_full/ckpt.pt \
    --lora-target attn \
    2>&1 | tee /data/rte_tiki2s_sweep.log

echo "=== Done: $(date) ==="

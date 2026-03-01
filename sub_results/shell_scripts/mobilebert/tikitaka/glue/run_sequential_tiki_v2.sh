#!/bin/bash
# Sequential run: scale_transfer_lr=True, fixed adam_lr, desired_bl=31
# transfer_lr sweep only (QQP: + adam_lr sweep)
# 10 trials each

cd /data

TASKS="rte mrpc cola stsb sst2 qnli mnli qqp"
N=10
IDX=0
TOTAL=$(echo $TASKS | wc -w)

for TASK in $TASKS; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[$IDX/$TOTAL] $TASK starting at $(date)"
    echo "=========================================="
    /data/venvs/lrtt/bin/python optuna_mobilebert_glue_tiki.py --task $TASK --n-trials $N
done

echo "=========================================="
echo "All tasks completed at $(date)"
echo "=========================================="

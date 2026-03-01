#!/bin/bash
# TikiTaka v2 attn (trainable) with fixed lr, desired_bl=31, transfer_lr=1.0, fast_lr=1.0
# use-v2, auto-scale, no-scale-transfer-lr
PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_tiki.py
COMMON="--lora-target attn --no-convert-nontarget --use-v2 --auto-scale --no-scale-transfer-lr --desired-bl 31 --fix-lr --no-wd --no-momentum --no-nesterov --n-trials 1 --sampler tpe"
LOGDIR=/data/results/tikitakav1

for NL in 3 6 9 10; do
    for TASK in rte mrpc stsb sst2 cola; do
        echo ""
        echo "================================================================"
        echo "  Starting: $TASK num_layers=$NL (TikiTaka v2 attn)"
        echo "  $(date)"
        echo "================================================================"
        $PYTHON $SCRIPT --task $TASK --num-layers $NL $COMMON 2>&1 | tee -a $LOGDIR/nohup_tikitaka_attn_nl${NL}.log
        echo "  Finished: $TASK at $(date)"
    done
done

echo ""
echo "================================================================"
echo "  ALL COMPLETE at $(date)"
echo "================================================================"

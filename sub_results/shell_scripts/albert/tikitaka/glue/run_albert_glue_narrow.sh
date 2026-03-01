#!/bin/bash
# ALBERT GLUE large-task sweep: narrowed ranges + SST-2 best seed
# SST-2 best: lr=0.023346, te=79, fast_lr=0.034587

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_tiki.py
LOGDIR=/data/results/tikitakav1
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --lora-target all --sampler tpe"
NARROW="--tpe-lr-range 0.001 0.1 --tpe-te-choices 1 100 --tpe-flr-range 0.001 0.5"
SEED="--enqueue 0.023346 79 0.034587"

mkdir -p "$LOGDIR"

echo "=============================================="
echo "[$(date)] Starting QNLI (5 trials, 11 epochs)"
echo "=============================================="
$PYTHON $SCRIPT --task qnli --n-trials 5 --epochs 11 \
    --study-name albert_glue_tiki_qnli_bs32_sgd_nowd_nomom_nonest_all_narrow \
    $COMMON $NARROW $SEED \
    >> "$LOGDIR/albert_glue_narrow_qnli.log" 2>&1
echo "[$(date)] Finished QNLI (exit code: $?)"

echo "=============================================="
echo "[$(date)] Starting QQP (5 trials, 5 epochs)"
echo "=============================================="
$PYTHON $SCRIPT --task qqp --n-trials 5 --epochs 5 \
    --study-name albert_glue_tiki_qqp_bs128_sgd_nowd_nomom_nonest_all_narrow \
    $COMMON $NARROW $SEED \
    >> "$LOGDIR/albert_glue_narrow_qqp.log" 2>&1
echo "[$(date)] Finished QQP (exit code: $?)"

echo "=============================================="
echo "[$(date)] Starting MNLI (5 trials, 4 epochs)"
echo "=============================================="
$PYTHON $SCRIPT --task mnli --n-trials 5 --epochs 4 \
    --study-name albert_glue_tiki_mnli_bs128_sgd_nowd_nomom_nonest_all_narrow \
    $COMMON $NARROW $SEED \
    >> "$LOGDIR/albert_glue_narrow_mnli.log" 2>&1
echo "[$(date)] Finished MNLI (exit code: $?)"

echo "[$(date)] All narrow tasks complete."

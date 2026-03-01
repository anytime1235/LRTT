#!/bin/bash
# =============================================================================
# ALBERT GLUE TikiTaka - te=1 fixed, transfer_lr search [0.01, 1.0]
# lr/fast_lr narrowed around previous best, 30 trials each
# =============================================================================
set -e

PYTHON="/data/venvs/lrtt/bin/python"
SCRIPT="/data/optuna_albert_glue_tiki.py"
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --lora-target all --sampler tpe --tpe-te-choices 1 1 --tpe-tlr-range 0.01 1.0 --tpe-flr-range 0.001 0.5"

# --- CoLA: best lr=0.01638, fast_lr=0.1675, te=32 ---
echo "====== CoLA (30 trials, te=1, tlr search) ======"
$PYTHON $SCRIPT --task cola --n-trials 30 $COMMON \
    --tpe-lr-range 0.005 0.05 \
    --study-name albert_glue_tiki_cola_bs16_sgd_nowd_nomom_nonest_all_te1_tlr \
    --enqueue 0.01638 1 0.1675 0.5

# --- MRPC: best lr=0.06251, fast_lr=0.02636, te=16 ---
echo "====== MRPC (30 trials, te=1, tlr search) ======"
$PYTHON $SCRIPT --task mrpc --n-trials 30 $COMMON \
    --tpe-lr-range 0.02 0.2 \
    --study-name albert_glue_tiki_mrpc_bs32_sgd_nowd_nomom_nonest_all_te1_tlr \
    --enqueue 0.06251 1 0.02636 0.5

# --- RTE: best lr=0.06251, fast_lr=0.02636, te=16 ---
echo "====== RTE (30 trials, te=1, tlr search) ======"
$PYTHON $SCRIPT --task rte --n-trials 30 $COMMON \
    --tpe-lr-range 0.02 0.2 \
    --study-name albert_glue_tiki_rte_bs32_sgd_nowd_nomom_nonest_all_te1_tlr \
    --enqueue 0.06251 1 0.02636 0.5

# --- SST-2: best lr=0.02335, fast_lr=0.03459, te=79 ---
echo "====== SST-2 (30 trials, te=1, tlr search) ======"
$PYTHON $SCRIPT --task sst2 --n-trials 30 $COMMON \
    --tpe-lr-range 0.007 0.07 \
    --study-name albert_glue_tiki_sst2_bs32_sgd_nowd_nomom_nonest_all_te1_tlr \
    --enqueue 0.02335 1 0.03459 0.5

# --- STS-B: best lr=0.08702, fast_lr=0.01238, te=50 ---
echo "====== STS-B (30 trials, te=1, tlr search) ======"
$PYTHON $SCRIPT --task stsb --n-trials 30 $COMMON \
    --tpe-lr-range 0.03 0.3 \
    --study-name albert_glue_tiki_stsb_bs16_sgd_nowd_nomom_nonest_all_te1_tlr \
    --enqueue 0.08702 1 0.01238 0.5

echo "====== ALL DONE ======"

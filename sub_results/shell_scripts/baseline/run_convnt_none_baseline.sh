#!/bin/bash
# Run convnt_none (No non-target analog conversion) baseline for STS-B, CoLA, SST-2
# Single trial each, using best LR from convnt_all experiments
# Parallel execution

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_lora.py
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --warm-alpha --lora-target all --warmup-ratio 0.05 --no-convert-nontarget"
LOG_DIR=/data/results/Analoglora_all

echo "=========================================="
echo "convnt_none baseline - STS-B, CoLA, SST-2"
echo "=========================================="
date

# Step 1: Pre-create DBs with best LR enqueued (overrides the script's default seed)
$PYTHON -c "
import optuna, os
optuna.logging.set_verbosity(optuna.logging.WARNING)

tasks = {
    'stsb': {
        'study': 'albert_stsb_lrtt_bs16_sgd_decay_nowd_nomom_nonest_combos_warmalpha_convnt_none_all',
        'dir': 'stsb',
        'params': {'learning_rate': 0.0011826316307555957, 'lora_alpha': 0.01, 'target_ab_lr': 0.03},
    },
    'cola': {
        'study': 'albert_cola_lrtt_bs16_sgd_decay_nowd_nomom_nonest_combos_warmalpha_convnt_none_all',
        'dir': 'cola',
        'params': {'learning_rate': 0.08463259666177746, 'lora_alpha': 0.01, 'target_ab_lr': 0.03},
    },
    'sst2': {
        'study': 'albert_sst2_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_warmalpha_convnt_none_all',
        'dir': 'sst2',
        'params': {'learning_rate': 0.0035498788321965025, 'lora_alpha': 0.01, 'target_ab_lr': 0.03},
    },
}

for task, cfg in tasks.items():
    results_dir = f'/data/results/Analoglora_all/{cfg[\"dir\"]}'
    os.makedirs(results_dir, exist_ok=True)
    db_path = f'sqlite:///{results_dir}/optuna_{cfg[\"study\"]}.db'
    study = optuna.create_study(
        study_name=cfg['study'], storage=db_path, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )
    # Enqueue best LR as the first trial
    study.enqueue_trial(cfg['params'])
    print(f'{task}: enqueued lr={cfg[\"params\"][\"learning_rate\"]:.6f}')
"

# Step 2: Run with --n-trials 1. The script's enqueue_trial(lr=0.5) will be queued
# AFTER our pre-enqueued best-LR trial, so trial 0 = best LR, then we stop at 1 trial.

# STS-B: best lr=0.00118 from convnt_all trial #14
echo "[1/3] STS-B (lr=0.00118)"
$PYTHON $SCRIPT --task stsb --n-trials 1 $COMMON \
  --study-name albert_stsb_lrtt_bs16_sgd_decay_nowd_nomom_nonest_combos_warmalpha_convnt_none_all \
  2>&1 | tee ${LOG_DIR}/stsb/baseline_none_stsb.log &
PID_STSB=$!

# CoLA: best lr=0.08463 from convnt_all trial #10
echo "[2/3] CoLA (lr=0.08463)"
$PYTHON $SCRIPT --task cola --n-trials 1 $COMMON \
  --study-name albert_cola_lrtt_bs16_sgd_decay_nowd_nomom_nonest_combos_warmalpha_convnt_none_all \
  2>&1 | tee ${LOG_DIR}/cola/baseline_none_cola.log &
PID_COLA=$!

# SST-2: best lr=0.00355 from convnt_all trial #6
echo "[3/3] SST-2 (lr=0.00355)"
$PYTHON $SCRIPT --task sst2 --n-trials 1 $COMMON \
  --study-name albert_sst2_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_warmalpha_convnt_none_all \
  2>&1 | tee ${LOG_DIR}/sst2/baseline_none_sst2.log &
PID_SST2=$!

echo ""
echo "PIDs: STS-B=$PID_STSB, CoLA=$PID_COLA, SST-2=$PID_SST2"
wait $PID_STSB && echo "STS-B done at $(date)" || echo "STS-B failed"
wait $PID_COLA && echo "CoLA done at $(date)" || echo "CoLA failed"
wait $PID_SST2 && echo "SST-2 done at $(date)" || echo "SST-2 failed"

echo ""
echo "ALL DONE at $(date)"

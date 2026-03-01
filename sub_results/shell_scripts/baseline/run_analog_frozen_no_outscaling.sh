#!/bin/bash
# Analog frozen baseline with learn_out_scaling=False
# Uses best LR per task (higher of convnt_all vs tiki LR)
# Purpose: check if learnable out_scaling contributes to performance

set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/optuna_albert_glue_lora.py
COMMON="--optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --warm-alpha --lora-target none --convert-nontarget --warmup-ratio 0.05 --no-learn-out-scaling"
LOG_DIR=/data/results/Analoglora_all

# Pre-create DBs with best LR enqueued
$PYTHON -c "
import optuna, os
optuna.logging.set_verbosity(optuna.logging.WARNING)

tasks = {
    'rte': {
        'study': 'albert_rte_frozen_no_outscaling',
        'params': {'learning_rate': 0.06251373574521749, 'lora_alpha': 0.01, 'target_ab_lr': 0.03},
    },
    'mrpc': {
        'study': 'albert_mrpc_frozen_no_outscaling',
        'params': {'learning_rate': 0.06251373574521749, 'lora_alpha': 0.01, 'target_ab_lr': 0.03},
    },
    'stsb': {
        'study': 'albert_stsb_frozen_no_outscaling',
        'params': {'learning_rate': 0.08702154746159006, 'lora_alpha': 0.01, 'target_ab_lr': 0.03},
    },
    'cola': {
        'study': 'albert_cola_frozen_no_outscaling',
        'params': {'learning_rate': 0.016376815637377924, 'lora_alpha': 0.01, 'target_ab_lr': 0.03},
    },
    'sst2': {
        'study': 'albert_sst2_frozen_no_outscaling',
        'params': {'learning_rate': 0.023345864076016236, 'lora_alpha': 0.01, 'target_ab_lr': 0.03},
    },
}

for task, cfg in tasks.items():
    results_dir = f'/data/results/Analoglora_all/{task}'
    os.makedirs(results_dir, exist_ok=True)
    db_path = f'sqlite:///{results_dir}/optuna_{cfg[\"study\"]}.db'
    study = optuna.create_study(
        study_name=cfg['study'], storage=db_path, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )
    study.enqueue_trial(cfg['params'])
    print(f'{task}: enqueued lr={cfg[\"params\"][\"learning_rate\"]:.6f}')
"

echo "=========================================="
echo "Frozen baseline - NO learn_out_scaling"
echo "Best LR per task, sequential execution"
echo "=========================================="
date

# 1. RTE (best=convnt_all, lr=0.0625)
echo ""
echo "[1/5] RTE"
$PYTHON $SCRIPT --task rte --n-trials 1 $COMMON \
  --study-name albert_rte_frozen_no_outscaling \
  2>&1 | tee ${LOG_DIR}/rte/frozen_no_outscaling_rte.log
echo "RTE done at $(date)"

# 2. MRPC (best=tiki, lr=0.0625)
echo ""
echo "[2/5] MRPC"
$PYTHON $SCRIPT --task mrpc --n-trials 1 $COMMON \
  --study-name albert_mrpc_frozen_no_outscaling \
  2>&1 | tee ${LOG_DIR}/mrpc/frozen_no_outscaling_mrpc.log
echo "MRPC done at $(date)"

# 3. STS-B (best=tiki, lr=0.0870)
echo ""
echo "[3/5] STS-B"
$PYTHON $SCRIPT --task stsb --n-trials 1 $COMMON \
  --study-name albert_stsb_frozen_no_outscaling \
  2>&1 | tee ${LOG_DIR}/stsb/frozen_no_outscaling_stsb.log
echo "STS-B done at $(date)"

# 4. CoLA (best=tiki, lr=0.0164)
echo ""
echo "[4/5] CoLA"
$PYTHON $SCRIPT --task cola --n-trials 1 $COMMON \
  --study-name albert_cola_frozen_no_outscaling \
  2>&1 | tee ${LOG_DIR}/cola/frozen_no_outscaling_cola.log
echo "CoLA done at $(date)"

# 5. SST-2 (best=tiki, lr=0.0233)
echo ""
echo "[5/5] SST-2"
$PYTHON $SCRIPT --task sst2 --n-trials 1 $COMMON \
  --study-name albert_sst2_frozen_no_outscaling \
  2>&1 | tee ${LOG_DIR}/sst2/frozen_no_outscaling_sst2.log
echo "SST-2 done at $(date)"

echo ""
echo "=========================================="
echo "ALL DONE at $(date)"
echo "=========================================="

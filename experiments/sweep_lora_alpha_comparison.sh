#!/bin/bash
# SST-2에서 lora_alpha 비교 실험: 0.1, 1.0, 10.0
# 각 10 trials, LR search space [1e-4, 1e-2]

set -e

PYTHON=/data/venvs/aihwkit_gpu/bin/python
SWEEP_SCRIPT=/data/LRTT_transformer/LRTT_glue/sweep_sixt1c_lora_glue_adam.py

echo "========================================================================"
echo "  SST-2 LoRA Alpha Comparison Experiment"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Task: SST-2"
echo "  Target modules: QKV (query, key, value)"
echo "  Learning rate: [1e-4, 1e-2] (log scale)"
echo "  Trials per alpha: 10"
echo "  Mode: Sixt1c (analog)"
echo "  Optimizer: AnalogSGD"
echo ""
echo "Testing lora_alpha values: 0.1, 1.0, 10.0"
echo "========================================================================"
echo ""

# Experiment 1: lora_alpha = 0.1
echo ""
echo "========================================================================"
echo "  Experiment 1/3: lora_alpha = 0.1"
echo "========================================================================"
echo ""

$PYTHON $SWEEP_SCRIPT \
    --task sst2 \
    --target QKV \
    --n_trials 10 \
    --lora_alpha 0.1

echo ""
echo "✓ Experiment 1/3 completed (alpha=0.1)"
echo ""

# Experiment 2: lora_alpha = 1.0
echo ""
echo "========================================================================"
echo "  Experiment 2/3: lora_alpha = 1.0"
echo "========================================================================"
echo ""

$PYTHON $SWEEP_SCRIPT \
    --task sst2 \
    --target QKV \
    --n_trials 10 \
    --lora_alpha 1.0

echo ""
echo "✓ Experiment 2/3 completed (alpha=1.0)"
echo ""

# Experiment 3: lora_alpha = 10.0
echo ""
echo "========================================================================"
echo "  Experiment 3/3: lora_alpha = 10.0"
echo "========================================================================"
echo ""

$PYTHON $SWEEP_SCRIPT \
    --task sst2 \
    --target QKV \
    --n_trials 10 \
    --lora_alpha 10.0

echo ""
echo "✓ Experiment 3/3 completed (alpha=10.0)"
echo ""

# Summary
echo ""
echo "========================================================================"
echo "  ALL EXPERIMENTS COMPLETED"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - Check wandb dashboard for detailed results"
echo "  - Study names contain lora_alpha values"
echo ""
echo "========================================================================"

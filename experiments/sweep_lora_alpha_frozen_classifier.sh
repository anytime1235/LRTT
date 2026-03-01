#!/bin/bash
# SST-2 LoRA Alpha Sweep with FROZEN Classifier
# lora_alpha: 0.01, 0.1, 1.0, 10.0, 100.0
# LR: [2e-4, 2e-2], 10 trials each, sequential execution

set -e

PYTHON=/data/venvs/aihwkit_gpu/bin/python
SWEEP_SCRIPT=/data/LRTT_transformer/LRTT_glue/sweep_sixt1c_lora_glue_frozen_classifier.py

echo "========================================================================"
echo "  SST-2 LoRA Alpha Sweep - FROZEN Classifier Experiment"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Task: SST-2"
echo "  Target modules: QKV (query, key, value)"
echo "  Learning rate: [2e-4, 2e-2] (log scale)"
echo "  Trials per alpha: 10"
echo "  Mode: Sixt1c (analog 6T1C)"
echo "  Optimizer: AnalogSGD"
echo "  Embeddings: FROZEN"
echo "  Classifier: FROZEN (실험 설정)"
echo ""
echo "Testing lora_alpha values: 0.1, 1.0, 10.0, 100.0 (0.01 skipped)"
echo "Total: 4 alpha values × 10 trials = 40 trials"
echo "========================================================================"
echo ""

# Experiment 1: lora_alpha = 0.1
echo ""
echo "========================================================================"
echo "  Experiment 1/4: lora_alpha = 0.1"
echo "========================================================================"
echo ""

$PYTHON $SWEEP_SCRIPT \
    --task sst2 \
    --target QKV \
    --n_trials 10 \
    --lora_alpha 0.1

echo ""
echo "✓ Experiment 1/4 completed (alpha=0.1)"
echo ""

# Experiment 2: lora_alpha = 1.0
echo ""
echo "========================================================================"
echo "  Experiment 2/4: lora_alpha = 1.0"
echo "========================================================================"
echo ""

$PYTHON $SWEEP_SCRIPT \
    --task sst2 \
    --target QKV \
    --n_trials 10 \
    --lora_alpha 1.0

echo ""
echo "✓ Experiment 2/4 completed (alpha=1.0)"
echo ""

# Experiment 3: lora_alpha = 10.0
echo ""
echo "========================================================================"
echo "  Experiment 3/4: lora_alpha = 10.0"
echo "========================================================================"
echo ""

$PYTHON $SWEEP_SCRIPT \
    --task sst2 \
    --target QKV \
    --n_trials 10 \
    --lora_alpha 10.0

echo ""
echo "✓ Experiment 3/4 completed (alpha=10.0)"
echo ""

# Experiment 4: lora_alpha = 100.0
echo ""
echo "========================================================================"
echo "  Experiment 4/4: lora_alpha = 100.0"
echo "========================================================================"
echo ""

$PYTHON $SWEEP_SCRIPT \
    --task sst2 \
    --target QKV \
    --n_trials 10 \
    --lora_alpha 100.0

echo ""
echo "✓ Experiment 4/4 completed (alpha=100.0)"
echo ""

# Summary
echo ""
echo "========================================================================"
echo "  ALL EXPERIMENTS COMPLETED"
echo "========================================================================"
echo ""
echo "Results summary:"
echo "  - alpha=0.1:   10 trials completed"
echo "  - alpha=1.0:   10 trials completed"
echo "  - alpha=10.0:  10 trials completed"
echo "  - alpha=100.0: 10 trials completed"
echo ""
echo "Total: 40 trials across 4 alpha values (0.01 skipped due to instability)"
echo ""
echo "Check wandb dashboard for detailed results."
echo "Study names contain 'frozen_classifier' and lora_alpha values."
echo ""
echo "========================================================================"

#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Uniform IO Bit Resolution Sweep (4, 6, 8, 10, 12, perfect)
# Experiment 2: Layerwise Mixed-Precision (requires code modification)
#
# All layers converted to IdealDevice (FP32 trainable analog).
# ADC/DAC bits are set identically for forward and backward IO.
#
# See: EXPERIMENT_PLAN_IO_BITS.md for full details.
# ═══════════════════════════════════════════════════════════════════════════════

PYTHON="/data/venvs/lrtt/bin/python"
SCRIPT="main_results/scripts/analysis/optuna_bert_squad_tiki.py"

# Common: IdealDevice, all sublayers, 2 epochs, single trial (fixed LR)
COMMON="--target-ideal --n-trials 1 --epochs 2 --batch-size 48 \
        --lora-target all --shared-lr --lr 2e-3"

echo "============================================================"
echo " Experiment 1: Uniform IO Bit Resolution Sweep"
echo "============================================================"

# 1a-1e: Sweep 4, 6, 8, 10, 12 bits
for BITS in 4 6 8 10 12; do
    echo ""
    echo "--- [1] Uniform ${BITS}-bit ---"
    $PYTHON $SCRIPT $COMMON --io-bits $BITS \
        --study-name "io_sweep_uniform_${BITS}b"
done

# 1f: Perfect IO (no quantization, FP32 reference)
echo ""
echo "--- [1f] Perfect IO (FP32 reference) ---"
$PYTHON $SCRIPT $COMMON \
    --study-name "io_sweep_uniform_perfect"

echo ""
echo "============================================================"
echo " Experiment 1 complete."
echo " Results in: /data/results/tikitakav1/"
echo "============================================================"

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Layerwise Mixed-Precision
#
# NOTE: This requires code modification to optuna_bert_squad_tiki.py.
#       The --io-bits flag currently sets bits uniformly for all modules.
#       To run layerwise, add --io-bits-map support first.
#
# Once implemented, run:
#
#   # Option A: Sublayer-level assignment
#   $PYTHON $SCRIPT $COMMON \
#       --io-bits-map '{"Q":8,"K":10,"V":8,"O":6,"FFN1":12,"FFN2":6}' \
#       --study-name "io_sweep_layerwise_optA"
#
#   # Comparison: Uniform 10b (same total bit-budget)
#   $PYTHON $SCRIPT $COMMON --io-bits 10 \
#       --study-name "io_sweep_layerwise_cmp_10b"
#
# ═══════════════════════════════════════════════════════════════════════════════

#!/usr/bin/env bash
# run_mitigation_plan.sh — SQuAD forward I/O mitigation screening
#
# Measures per-layer SNR improvement from each mitigation.
# Only includes mitigations that directly improve forward pass quality
# (relevant for training). ADC 10,12 excluded (saturated).
#
# Evaluation metric: per-layer / per-sublayer MAC SNR (dB)
#
# Usage:
#   cd /data/main_results
#   bash scripts/shell/run_mitigation_plan.sh

set -euo pipefail

PYTHON="/data/venvs/lrtt/bin/python"
SCRIPT="/data/main_results/scripts/diagnosis/diag_forward_io_single_rpu.py"
OUT_BASE="./results/diag_fwd_io_mitigations"
COMMON="--n-step 200 --batch-size 8 --dac-bits 7 --dw-min 0.001 --out-dir $OUT_BASE"

# 1) Baseline ADC sweep — per-layer SNR ground truth
$PYTHON $SCRIPT $COMMON --adc-bits-sweep "4,6,8" --tag baseline

# 2) Out-bound calibration — per-layer resolution improvement
$PYTHON $SCRIPT $COMMON --adc-bits-sweep "4,6,8" --tag obcal_per_module \
    --calib-out-bound --out-bound-grouping per_module --save-calib-table

# 3) Mixed precision — FFN1 bottleneck targeted (+2bit)
$PYTHON $SCRIPT $COMMON --adc-bits 6 --tag mp_base6 \
    --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1

# 4) Combined: mp + obcal
$PYTHON $SCRIPT $COMMON --adc-bits 6 --tag mp_base6_obcal \
    --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1 \
    --calib-out-bound --out-bound-grouping per_module

# 5) Plot results — compare per-layer SNR across mitigations
TAGS="baseline,obcal_per_module,mp_base6,mp_base6_obcal"
$PYTHON /data/main_results/scripts/paper/plot_mitigation_adc_io.py \
    --results-dir $OUT_BASE \
    --tags "$TAGS" \
    --out-dir ./plots/diag_fwd_io_mitigations

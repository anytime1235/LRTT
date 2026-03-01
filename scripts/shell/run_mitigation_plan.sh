#!/usr/bin/env bash
set -euo pipefail

PYTHON="/data/venvs/lrtt/bin/python"
SCRIPT="/data/main_results/scripts/diagnosis/diag_forward_io_single_rpu.py"
OUT_BASE="./results/diag_fwd_io_mitigations"
COMMON="--n-step 200 --batch-size 8 --dac-bits 7 --dw-min 0.001 --out-dir $OUT_BASE"

# 1) Baseline sweep
$PYTHON $SCRIPT $COMMON --adc-bits-sweep "4,6,8,10,12" --tag baseline

# 2) Out-bound calibration sweep (per_module)
$PYTHON $SCRIPT $COMMON --adc-bits-sweep "4,6,8,10,12" --tag obcal_per_module \
    --calib-out-bound --out-bound-grouping per_module --save-calib-table

# 3) Mixed precision base-6
$PYTHON $SCRIPT $COMMON --adc-bits 6 --tag mp_base6 \
    --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1

# 4) Mixed precision base-6 + depth boost (optional)
$PYTHON $SCRIPT $COMMON --adc-bits 6 --tag mp_base6_depth \
    --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1 \
    --depth-boost "9-11:+1"

# 5) learn_out_scaling (ADC 4,6,8)
$PYTHON $SCRIPT $COMMON --adc-bits-sweep "4,6,8" --tag los_true \
    --learn-out-scaling

# 6) learn_out_scaling + obcal (ADC 4,6,8)
$PYTHON $SCRIPT $COMMON --adc-bits-sweep "4,6,8" --tag los_obcal \
    --learn-out-scaling --calib-out-bound --out-bound-grouping per_module

# 7) Combined: mp + obcal
$PYTHON $SCRIPT $COMMON --adc-bits 6 --tag mp_base6_obcal \
    --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1 \
    --calib-out-bound

# 8) Plot all results
TAGS="baseline,obcal_per_module,mp_base6,los_true,los_obcal,mp_base6_obcal"
$PYTHON /data/main_results/scripts/paper/plot_mitigation_adc_io.py \
    --results-dir $OUT_BASE \
    --tags "$TAGS" \
    --out-dir ./plots/diag_fwd_io_mitigations

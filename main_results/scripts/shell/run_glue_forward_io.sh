#!/usr/bin/env bash
# run_glue_forward_io.sh — GLUE forward I/O diagnostic pipeline
#
# Runs four diagnostic configurations for SST-2:
#   1) Baseline ADC sweep
#   2) Out-bound calibration sweep
#   3) Mixed precision (base adc=6, FFN1+2, V+1)
#   4) Seed sweep (for mean±std error bars)
#
# Usage:
#   cd /data/main_results
#   bash run_glue_forward_io.sh
#
# Smoke test (fast):
#   bash run_glue_forward_io.sh smoke

set -euo pipefail

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=scripts/diagnosis/diag_forward_io_glue.py

# Common args for all runs
COMMON="--n-step 200 --batch-size 8 --dac-bits 7 --dw-min 0.001 \
        --glue-task sst2 --max-seq-length 128 \
        --out-dir ./results/diag_fwd_io_glue \
        --logit-eval-batches 10"

# Allow smoke test override (fast)
if [[ "${1:-}" == "smoke" ]]; then
    COMMON="--n-step 2 --batch-size 2 --dac-bits 7 --dw-min 0.001 \
            --glue-task sst2 --max-seq-length 128 \
            --out-dir /tmp/smoke_glue \
            --logit-eval-batches 2"
    echo "=== SMOKE TEST MODE ==="
fi

echo "=== 1) Baseline ADC Sweep ==="
$PYTHON $SCRIPT $COMMON \
    --adc-bits-sweep "4,6,8,10,12" \
    --tag baseline_sst2

echo ""
echo "=== 2) Out-bound Calibration Sweep ==="
$PYTHON $SCRIPT $COMMON \
    --adc-bits-sweep "4,6,8,10,12" \
    --tag obcal_sst2 \
    --calib-out-bound \
    --out-bound-grouping per_module \
    --save-calib-table

echo ""
echo "=== 3) Mixed Precision (base=6, FFN1+2, V+1) ==="
$PYTHON $SCRIPT $COMMON \
    --adc-bits 6 \
    --tag mp_base6_sst2 \
    --mixed-precision \
    --adc-base 6 \
    --ffn1-bits-plus 2 \
    --v-bits-plus 1

echo ""
echo "=== 4) Seed Sweep (seeds 42,43,44) ==="
$PYTHON $SCRIPT $COMMON \
    --adc-bits-sweep "4,6,8,10,12" \
    --tag seed_sweep_sst2 \
    --seed-sweep "42,43,44"

echo ""
echo "=== All diagnostic runs complete ==="
echo "Results: ./results/diag_fwd_io_glue/"

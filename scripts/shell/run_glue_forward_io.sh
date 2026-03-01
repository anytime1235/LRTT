#!/usr/bin/env bash
# run_glue_forward_io.sh — GLUE SST-2 forward I/O mitigation screening
#
# Measures per-layer SNR improvement from each mitigation on SST-2.
# Only includes mitigations that directly improve forward pass quality.
# ADC 10,12 excluded (saturated). Logit eval removed (inference-only artifact).
#
# Evaluation metric: per-layer / per-sublayer MAC SNR (dB)
#
# Usage:
#   cd /data/main_results
#   bash scripts/shell/run_glue_forward_io.sh
#
# Smoke test (fast):
#   bash scripts/shell/run_glue_forward_io.sh smoke

set -euo pipefail

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=scripts/diagnosis/diag_forward_io_glue.py

# Common args — no logit eval (inference-only artifact)
COMMON="--n-step 200 --batch-size 8 --dac-bits 7 --dw-min 0.001 \
        --glue-task sst2 --max-seq-length 128 \
        --out-dir ./results/diag_fwd_io_glue"

# Allow smoke test override (fast)
if [[ "${1:-}" == "smoke" ]]; then
    COMMON="--n-step 2 --batch-size 2 --dac-bits 7 --dw-min 0.001 \
            --glue-task sst2 --max-seq-length 128 \
            --out-dir /tmp/smoke_glue"
    echo "=== SMOKE TEST MODE ==="
fi

echo "=== 1) Baseline ADC Sweep ==="
$PYTHON $SCRIPT $COMMON \
    --adc-bits-sweep "4,6,8" \
    --tag baseline_sst2

echo ""
echo "=== 2) Out-bound Calibration Sweep ==="
$PYTHON $SCRIPT $COMMON \
    --adc-bits-sweep "4,6,8" \
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
echo "=== All diagnostic runs complete ==="
echo "Results: ./results/diag_fwd_io_glue/"

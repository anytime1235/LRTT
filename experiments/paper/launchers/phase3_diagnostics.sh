#!/bin/bash
# Phase 3: Mechanistic 14-bit Diagnostics
# Short-run: 100 steps, log every 1, diag-update-exact.
# Methods: 4 pulse types + MixedPrec + best TTv1
# GPUs 1,2,3 (2 each parallel).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase3}"
PYTHON="${PYTHON:-python}"
LN_LR="${BEST_LN_LR:-0.016}"

# Best TTv1 config from Phase 1 (edit after Phase 1):
BEST_GAMMA="${BEST_GAMMA:-0.0}"
BEST_UIM="${BEST_UIM:-false}"
BEST_TE="${BEST_TE:-24}"

cd "$SCRIPT_DIR"

echo "=== Phase 3: Mechanistic Diagnostics ==="
echo "Using LN LR: $LN_LR"

# Wave 1: GPU1=none_with_device, GPU2=deterministic, GPU3=mean_count
CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --seed 42 \
    --max-steps 100 --n-bits 14 \
    --pulse-type none_with_device --ln-lr $LN_LR \
    --output-dir "$RESULTS_DIR/none_with_device" \
    --log-every 1 --diag-update-exact --diag-steps 100 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --seed 42 \
    --max-steps 100 --n-bits 14 \
    --pulse-type deterministic --ln-lr $LN_LR \
    --output-dir "$RESULTS_DIR/deterministic" \
    --log-every 1 --diag-update-exact --diag-steps 100 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --seed 42 \
    --max-steps 100 --n-bits 14 \
    --pulse-type mean_count --ln-lr $LN_LR \
    --output-dir "$RESULTS_DIR/mean_count" \
    --log-every 1 --diag-update-exact --diag-steps 100 &
PID3=$!

wait $PID1 $PID2 $PID3
echo "Wave 1 complete."

# Wave 2: GPU1=stochastic, GPU2=mixed_precision, GPU3=ttv1_best
CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --seed 42 \
    --max-steps 100 --n-bits 14 \
    --pulse-type stochastic --ln-lr $LN_LR \
    --output-dir "$RESULTS_DIR/stochastic" \
    --log-every 1 --diag-update-exact --diag-steps 100 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --seed 42 \
    --max-steps 100 --n-bits 14 --ln-lr $LN_LR \
    --output-dir "$RESULTS_DIR/mixed_precision" \
    --log-every 1 --diag-update-exact --diag-steps 100 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --max-steps 100 --n-bits 14 \
    --gamma $BEST_GAMMA \
    --units-in-mbatch $BEST_UIM \
    --transfer-every $BEST_TE --ln-lr $LN_LR \
    --output-dir "$RESULTS_DIR/ttv1_best" \
    --log-every 1 --diag-update-exact --diag-steps 100 &
PID3=$!

wait $PID1 $PID2 $PID3
echo "Wave 2 complete."

echo ""
echo "=== Phase 3 Complete ==="
echo "Diagnostics CSV files:"
for TAG in none_with_device deterministic mean_count stochastic mixed_precision ttv1_best; do
    CSV="$RESULTS_DIR/$TAG/update_diagnostics.csv"
    if [ -f "$CSV" ]; then
        LINES=$(wc -l < "$CSV")
        echo "  $TAG: $CSV ($LINES lines)"
    else
        echo "  $TAG: MISSING diagnostics"
    fi
done

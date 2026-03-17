#!/bin/bash
# Phase 2: Main Paper Bit Sweep
# 10 configs = {single_rpu, mixed_precision} × bits{8,10,12,14,16}
# Seed=42 fixed. desired_bl=31. 4 epochs.
# GPUs 1,2,3 (3-4 configs each, sequential per GPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase2}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-4}"
LN_LR="${BEST_LN_LR:-0.016}"

cd "$SCRIPT_DIR"

echo "=== Phase 2: Bit Sweep ==="
echo "Using LN LR: $LN_LR"

run_exp() {
    local GPU=$1 METHOD=$2 BITS=$3
    local TAG="${METHOD}_${BITS}b"
    echo "Starting $TAG on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method $METHOD --seed 42 \
        --epochs $EPOCHS --n-bits $BITS \
        --ln-lr $LN_LR \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
}

# GPU 1: single_rpu 8,10,12,14
# GPU 2: single_rpu 16 + mixed_precision 8,10
# GPU 3: mixed_precision 12,14,16

# Wave 1 (parallel)
run_exp 1 single_rpu 8 &
run_exp 2 single_rpu 16 &
run_exp 3 mixed_precision 12 &
wait

# Wave 2 (parallel)
run_exp 1 single_rpu 10 &
run_exp 2 mixed_precision 8 &
run_exp 3 mixed_precision 14 &
wait

# Wave 3 (parallel)
run_exp 1 single_rpu 12 &
run_exp 2 mixed_precision 10 &
run_exp 3 mixed_precision 16 &
wait

# Wave 4
run_exp 1 single_rpu 14 &
wait

echo ""
echo "=== Phase 2 Complete ==="
for METHOD in single_rpu mixed_precision; do
    for BITS in 8 10 12 14 16; do
        TAG="${METHOD}_${BITS}b"
        DIR="$RESULTS_DIR/$TAG"
        if [ -f "$DIR/summary.json" ]; then
            F1=$(python -c "import json; d=json.load(open('$DIR/summary.json')); print(f'{d[\"results\"][\"best_f1\"]:.2f}')")
            echo "  $TAG: best_f1=$F1"
        else
            echo "  $TAG: MISSING"
        fi
    done
done

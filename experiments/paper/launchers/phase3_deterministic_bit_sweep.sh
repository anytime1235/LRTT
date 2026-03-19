#!/bin/bash
# Phase 3: Deterministic Pulse Bit Sweep
# Same config as g1.0_r1.0 bitwidth experiments but with deterministic pulse update
# Fast tile: 8b, 10b, 12b | Slow tile: 10b | 4 epochs
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase3_determ}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS="${EPOCHS:-4}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 3: Deterministic Pulse Bit Sweep ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local FAST_BITS=$1
    local TAG="g1.0_r1.0_determ_fast${FAST_BITS}b"
    echo ""
    echo "============================================"
    echo "  $TAG  (fast=${FAST_BITS}bit, slow=10bit, deterministic)"
    echo "============================================"
    $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS \
        --batch-size 16 --grad-accum-steps 3 \
        --n-bits $FAST_BITS --n-bits-slow 10 \
        --gamma 1.0 \
        --units-in-mbatch true \
        --transfer-every 3 \
        --with-reset-prob 1.0 \
        --fast-lr 0.1 \
        --transfer-lr 1.0 \
        --scale-transfer-lr false \
        --analog-lr 0.016 \
        --classifier-lr 0.003 \
        --ln-lr 0.003 \
        --warmup-ratio 0.05 \
        --min-lr-rate 0.05 \
        --ttv1-fast-pulse-type deterministic \
        --ttv1-transfer-pulse-type deterministic \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "DONE $TAG $(date)"
}

# Run sequentially (single GPU)
for BITS in 6 8; do
    run $BITS
done

echo ""
echo "=== All deterministic pulse experiments complete: $(date) ==="

# Print summary
for BITS in 6 8; do
    TAG="g1.0_r1.0_determ_fast${BITS}b"
    DIR="$RESULTS_DIR/$TAG"
    if [ -f "$DIR/summary.json" ]; then
        F1=$($PYTHON -c "import json; d=json.load(open('$DIR/summary.json')); print(f'best_f1={d[\"results\"][\"best_f1\"]:.2f}, final_f1={d[\"results\"][\"final_f1\"]:.2f}, em={d[\"results\"][\"final_em\"]:.2f}')")
        echo "  $TAG: $F1"
    else
        echo "  $TAG: MISSING"
    fi
done

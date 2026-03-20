#!/bin/bash
# Slow Tile Bit-Width Sweep: fast=14bit fixed, slow={6,7,8,9}bit
# Stochastic pulse (default), all other conditions identical to phase1c_4ep best config
# gamma=1.0, reset=1.0, 4 epochs, seed=42
# Note: slow=10b already exists (F1=86.17 from phase1c_4ep/g1.0_r1.0_fast14b)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/slow_tile_sweep_fast14b}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS="${EPOCHS:-4}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Slow Tile Bit Sweep (fast=14bit, slow=6,7,8,9, stochastic) ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local SLOW_BITS=$1
    local TAG="fast14b_slow${SLOW_BITS}b"
    echo ""
    echo "============================================"
    echo "  $TAG  (fast=14bit, slow=${SLOW_BITS}bit, stochastic)"
    echo "============================================"
    $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS \
        --batch-size 16 --grad-accum-steps 3 \
        --n-bits 14 --n-bits-slow $SLOW_BITS \
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
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "DONE $TAG $(date)"
}

# Run sequentially: slow=6b → 7b → 8b → 9b
for BITS in 6 7 8 9; do
    run $BITS
done

echo ""
echo "=== All slow tile sweep experiments complete: $(date) ==="

# Print summary
echo ""
echo "=== Results Summary ==="
printf "%-30s %10s %10s %10s\n" "Experiment" "Best F1" "Final F1" "Final EM"
echo "--------------------------------------------------------------"
for BITS in 6 7 8 9; do
    TAG="fast14b_slow${BITS}b"
    DIR="$RESULTS_DIR/$TAG"
    if [ -f "$DIR/summary.json" ]; then
        $PYTHON -c "
import json
d = json.load(open('$DIR/summary.json'))['results']
print(f'  fast14b_slow${BITS}b                {d[\"best_f1\"]:10.2f} {d[\"final_f1\"]:10.2f} {d[\"final_em\"]:10.2f}')
"
    else
        printf "  %-28s %10s\n" "$TAG" "MISSING"
    fi
done

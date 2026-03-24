#!/bin/bash
# Single RPU Stochastic — 10-bit 완료 대기 후 9→8→7→6 순차 실행
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/single_rpu_stoch_bit_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

# PID 16956 (10-bit) 완료 대기
echo "=== PID 16956 (10-bit) 완료 대기 중: $(date) ==="
while kill -0 16956 2>/dev/null; do
    sleep 30
done
echo "=== 10-bit 완료 확인, 나머지 실험 시작: $(date) ==="

run() {
    local BITS=$1
    local TAG="single_rpu_${BITS}b_stoch"
    echo ""
    echo "============================================"
    echo "  $TAG  (${BITS}-bit, stochastic)"
    echo "============================================"
    $PYTHON paper_experiment.py \
        --mode fixed --method single_rpu --seed 42 \
        --epochs 4 \
        --batch-size 16 --grad-accum-steps 3 \
        --n-bits $BITS \
        --pulse-type stochastic \
        --desired-bl 31 \
        --analog-lr 0.016 \
        --classifier-lr 0.003 \
        --ln-lr 0.003 \
        --warmup-ratio 0.05 \
        --min-lr-rate 0.05 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "DONE $TAG $(date)"
}

for BITS in 9 8 7 6; do
    run $BITS
done

echo ""
echo "=== All remaining experiments complete: $(date) ==="

# Print full summary (10 포함)
for BITS in 10 9 8 7 6; do
    TAG="single_rpu_${BITS}b_stoch"
    DIR="$RESULTS_DIR/$TAG"
    if [ -f "$DIR/summary.json" ]; then
        F1=$($PYTHON -c "import json; d=json.load(open('$DIR/summary.json')); print(f'best_f1={d[\"results\"][\"best_f1\"]:.2f}, final_f1={d[\"results\"][\"final_f1\"]:.2f}, em={d[\"results\"][\"final_em\"]:.2f}')")
        echo "  $TAG: $F1"
    else
        echo "  $TAG: MISSING"
    fi
done

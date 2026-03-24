#!/bin/bash
# Single RPU (Stochastic Pulse) Bit Sweep: 10 → 9 → 8 → 7 → 6
# TTv1 bit sweep과 동일한 학습 하이퍼파라미터 사용
#
# 공통 조건 (bit_sweep_summary.json과 동일):
#   model: bert-base-uncased, task: SQuAD v1.1
#   pulse_type: stochastic, epochs: 4, batch_size: 16, grad_accum_steps: 3 (eff=48)
#   analog_lr: 0.016, classifier_lr: 0.003, ln_lr: 0.003
#   warmup_ratio: 0.05, min_lr_rate: 0.05, seed: 42
#   IO: perfect, desired_bl: 31
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/single_rpu_stoch_bit_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Single RPU Stochastic Bit Sweep (10→9→8→7→6) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

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

# 순차 실행: 10 → 9 → 8 → 7 → 6
for BITS in 10 9 8 7 6; do
    run $BITS
done

echo ""
echo "=== All experiments complete: $(date) ==="

# Print summary
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

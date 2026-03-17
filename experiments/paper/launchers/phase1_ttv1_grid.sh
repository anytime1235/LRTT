#!/bin/bash
# Phase 1: TTv1 Regime Discovery — 6 experiments
# 3 transfer configs × 2 gamma values = 6. 14-bit, 2 epochs, seed=42.
#
# m_batch = batch_size × seq_len = 48 × 384 = 18432
# BERT attention in_features = 768
#
# Config A: uim=F, te=24  -> 18432/24=768 transfers/step -> full sweep every step
# Config B: uim=F, te=2400 -> 18432/2400≈7.7 -> ~8 columns/step -> ~96 steps
# Config C: uim=T, te=1   -> 1 transfer/step -> 1 column/step -> 768 steps
#
# GPU allocation:
#   GPU 1: Config A (uim=F te=24), gamma=0.0 then gamma=0.1
#   GPU 2: Config B (uim=F te=2400), gamma=0.0 then gamma=0.1
#   GPU 3: Config C (uim=T te=1), gamma=0.0 then gamma=0.1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-2}"
LN_LR="${BEST_LN_LR:-0.016}"

cd "$SCRIPT_DIR"

echo "=== Phase 1: TTv1 Regime Discovery ==="
echo "Using LN LR: $LN_LR"

run_ttv1() {
    local GPU=$1 CONFIG=$2 UIM=$3 TE=$4 GAMMA=$5
    local TAG="${CONFIG}_gamma${GAMMA}"
    echo "Starting $TAG on GPU $GPU (uim=$UIM, te=$TE, gamma=$GAMMA)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS --n-bits 14 \
        --gamma $GAMMA \
        --units-in-mbatch $UIM \
        --transfer-every $TE \
        --ln-lr $LN_LR \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
}

# First gamma=0.0 in parallel across GPUs
run_ttv1 1 configA false 24   0.0 &
run_ttv1 2 configB false 2400 0.0 &
run_ttv1 3 configC true  1    0.0 &
wait
echo "gamma=0.0 done."

# Then gamma=0.1 in parallel
run_ttv1 1 configA false 24   0.1 &
run_ttv1 2 configB false 2400 0.1 &
run_ttv1 3 configC true  1    0.1 &
wait
echo "gamma=0.1 done."

echo ""
echo "=== Phase 1 Complete ==="
for TAG in configA_gamma0.0 configA_gamma0.1 configB_gamma0.0 configB_gamma0.1 configC_gamma0.0 configC_gamma0.1; do
    DIR="$RESULTS_DIR/$TAG"
    if [ -f "$DIR/summary.json" ]; then
        F1=$(python -c "import json; d=json.load(open('$DIR/summary.json')); print(f'{d[\"results\"][\"best_f1\"]:.2f}')")
        echo "  $TAG: best_f1=$F1"
    else
        echo "  $TAG: MISSING"
    fi
done

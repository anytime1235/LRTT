#!/bin/bash
# Phase 1 Reset Sweep: TTv1 with with_reset_prob=1.0
# All: uim=false, gamma=0.0, 14-bit, 2 epochs, seed=42, ln_lr=0.003
# Variations: transfer_every={2400,18000} × {fast_lr, transfer_lr}
# GPU 1,2,3 parallel — each GPU runs sequential experiments
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1_reset}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS="${EPOCHS:-2}"

cd "$SCRIPT_DIR"

echo "=== Phase 1 Reset Sweep (with_reset_prob=1.0) ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run_ttv1() {
    local GPU=$1 TE=$2 FAST_LR=$3 TRANSFER_LR=$4 TAG=$5
    echo "[GPU $GPU] START $TAG (te=$TE, flr=$FAST_LR, tlr=$TRANSFER_LR) $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS --n-bits 14 \
        --gamma 0.0 \
        --units-in-mbatch false \
        --transfer-every $TE \
        --with-reset-prob 1.0 \
        --fast-lr $FAST_LR \
        --transfer-lr $TRANSFER_LR \
        --ln-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

# GPU 1: te=2400 baseline + te=18000 baseline + te=18000 flr=0.01
gpu1_pipeline() {
    run_ttv1 1 2400  1.0   1.0  "te2400_flr1.0_tlr1.0"
    run_ttv1 1 18000 1.0   1.0  "te18000_flr1.0_tlr1.0"
    run_ttv1 1 18000 0.01  1.0  "te18000_flr0.01_tlr1.0"
}

# GPU 2: te=2400 flr=0.1 + te=2400 flr=0.01 + te=18000 flr=0.1
gpu2_pipeline() {
    run_ttv1 2 2400  0.1   1.0  "te2400_flr0.1_tlr1.0"
    run_ttv1 2 2400  0.01  1.0  "te2400_flr0.01_tlr1.0"
    run_ttv1 2 18000 0.1   1.0  "te18000_flr0.1_tlr1.0"
}

# GPU 3: te=2400 tlr=0.1 + te=18000 tlr=0.1
gpu3_pipeline() {
    run_ttv1 3 2400  1.0   0.1  "te2400_flr1.0_tlr0.1"
    run_ttv1 3 18000 1.0   0.1  "te18000_flr1.0_tlr0.1"
}

# Launch all 3 pipelines in parallel
gpu1_pipeline 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
gpu2_pipeline 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!
gpu3_pipeline 2>&1 | tee "$RESULTS_DIR/gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3

echo ""
echo "=== All experiments complete: $(date) ==="
echo ""
echo "--- Results Summary ---"
$PYTHON -c "
import json, os
base = '$RESULTS_DIR'
tags = [
    'te2400_flr1.0_tlr1.0', 'te2400_flr0.1_tlr1.0', 'te2400_flr0.01_tlr1.0', 'te2400_flr1.0_tlr0.1',
    'te18000_flr1.0_tlr1.0', 'te18000_flr0.1_tlr1.0', 'te18000_flr0.01_tlr1.0', 'te18000_flr1.0_tlr0.1',
]
print(f'{\"Config\":<30} {\"Best F1\":>8} {\"Final F1\":>9} {\"Final EM\":>9}')
print('-' * 60)
for tag in tags:
    path = os.path.join(base, tag, 'summary.json')
    try:
        d = json.load(open(path))['results']
        print(f'{tag:<30} {d[\"best_f1\"]:8.2f} {d[\"final_f1\"]:9.2f} {d[\"final_em\"]:9.2f}')
    except:
        print(f'{tag:<30} MISSING')
"

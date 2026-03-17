#!/bin/bash
# Phase 1B: TTv1 Hyperparameter Sweep
# Fixed: uim=true, te=1, 14-bit, 2 epochs, seed=42, ln_lr=0.003, scale_transfer_lr=true(default)
# Variables: reset_prob, gamma, fast_lr, transfer_lr
# GPU 1,2,3 parallel
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1b}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS="${EPOCHS:-2}"

cd "$SCRIPT_DIR"

echo "=== Phase 1B: TTv1 Sweep (uim=true, te=1) ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local GPU=$1 TAG=$2 GAMMA=$3 RESET=$4 FLR=$5 TLR=$6
    echo "[GPU $GPU] START $TAG (g=$GAMMA, reset=$RESET, flr=$FLR, tlr=$TLR) $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS --n-bits 14 \
        --gamma $GAMMA \
        --units-in-mbatch true \
        --transfer-every 1 \
        --with-reset-prob $RESET \
        --fast-lr $FLR \
        --transfer-lr $TLR \
        --ln-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

#            GPU  TAG                          GAMMA RESET FLR   TLR
# ---------------------------------------------------------------
# GPU 1: gamma × reset_prob sweep
gpu1_pipeline() {
    run 1 "g0.0_reset1.0"            0.0  1.0  1.0   1.0
    run 1 "g0.1_reset1.0"            0.1  1.0  1.0   1.0
    run 1 "g1.0_reset1.0"            1.0  1.0  1.0   1.0
}

# GPU 2: fast_lr + transfer_lr sweep (gamma=0.0, reset=0.0)
gpu2_pipeline() {
    run 2 "g0.0_flr0.1"              0.0  0.0  0.1   1.0
    run 2 "g0.0_flr0.01"             0.0  0.0  0.01  1.0
    run 2 "g0.0_tlr0.1"              0.0  0.0  1.0   0.1
}

# GPU 3: gamma=1.0 + reset=1.0 + lr tweaks
gpu3_pipeline() {
    run 3 "g1.0_reset1.0_flr0.1"     1.0  1.0  0.1   1.0
    run 3 "g1.0_reset1.0_flr0.01"    1.0  1.0  0.01  1.0
    run 3 "g1.0_reset1.0_tlr0.1"     1.0  1.0  1.0   0.1
}

mkdir -p "$RESULTS_DIR"

gpu1_pipeline 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
gpu2_pipeline 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!
gpu3_pipeline 2>&1 | tee "$RESULTS_DIR/gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3

echo ""
echo "=== All experiments complete: $(date) ==="

# Summary
$PYTHON -c "
import json, os
base = '$RESULTS_DIR'
tags = [
    'g0.0_reset1.0', 'g0.1_reset1.0', 'g1.0_reset1.0',
    'g0.0_flr0.1', 'g0.0_flr0.01', 'g0.0_tlr0.1',
    'g1.0_reset1.0_flr0.1', 'g1.0_reset1.0_flr0.01', 'g1.0_reset1.0_tlr0.1',
]
print(f'{\"Config\":<30} {\"Best F1\":>8} {\"Final F1\":>9} {\"Final EM\":>9}')
print('-' * 60)
# Phase 1 baselines for reference
for ref, f1 in [('(ref) configC g=0.0', 77.67), ('(ref) configC g=0.1', 59.74)]:
    print(f'{ref:<30} {f1:8.2f}')
print('-' * 60)
for tag in tags:
    path = os.path.join(base, tag, 'summary.json')
    try:
        d = json.load(open(path))['results']
        print(f'{tag:<30} {d[\"best_f1\"]:8.2f} {d[\"final_f1\"]:9.2f} {d[\"final_em\"]:9.2f}')
    except:
        print(f'{tag:<30} MISSING')
"

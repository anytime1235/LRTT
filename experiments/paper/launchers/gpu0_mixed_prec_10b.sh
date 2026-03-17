#!/bin/bash
# Mixed Precision 10-bit: QKVO-only, FFN-only, ALL
# GPU 0, 4 epochs, AnalogAdam
# Best LR from QKV optuna sweep: analog_lr=0.0357, classifier_lr=0.00076
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/mixed_prec_10b}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU=0

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Mixed Precision 10-bit (GPU $GPU) ==="
echo "Start: $(date)"

run() {
    local TARGET=$1 TAG=$2
    echo "[GPU $GPU] START $TAG (target=$TARGET) $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method mixed_precision --seed 42 \
        --epochs 4 --n-bits 10 \
        --target-layers $TARGET \
        --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

run attention "mp_10b_qkvo"
run ffn       "mp_10b_ffn"
run all       "mp_10b_all"

echo ""
echo "=== analog_lr=0.0357 complete. Starting analog_lr=0.00357 (1/10): $(date) ==="

run_low() {
    local TARGET=$1 TAG=$2
    echo "[GPU $GPU] START $TAG (target=$TARGET, analog_lr=0.00357) $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method mixed_precision --seed 42 \
        --epochs 4 --n-bits 10 \
        --target-layers $TARGET \
        --analog-lr 0.00357 --classifier-lr 0.00076 --ln-lr 0.00076 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

run_low attention "mp_10b_qkvo_alr0.00357"
run_low ffn       "mp_10b_ffn_alr0.00357"
run_low all       "mp_10b_all_alr0.00357"

echo ""
echo "=== All complete: $(date) ==="

$PYTHON -c "
import json, os
base = '$RESULTS_DIR'
tags = ['mp_10b_qkvo', 'mp_10b_ffn', 'mp_10b_all',
        'mp_10b_qkvo_alr0.00357', 'mp_10b_ffn_alr0.00357', 'mp_10b_all_alr0.00357']
print(f'{\"Tag\":<30} {\"analog_lr\":>10} {\"Best F1\":>8} {\"Final F1\":>9}')
print('-' * 60)
for tag in tags:
    path = os.path.join(base, tag, 'summary.json')
    try:
        d = json.load(open(path))['results']
        c = json.load(open(os.path.join(base, tag, 'config.json')))
        print(f'{tag:<30} {c[\"analog_lr\"]:>10.5f} {d[\"best_f1\"]:>8.2f} {d[\"final_f1\"]:>9.2f}')
    except:
        print(f'{tag:<30} {\"\":>10} {\"MISSING\":>8}')
"

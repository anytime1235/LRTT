#!/bin/bash
# Phase 4: TTv1 Final Confirmation
# Best TTv1 regime from Phase 1. 14-bit, seed=42, 4 epochs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase4}"
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-4}"
LN_LR="${BEST_LN_LR:-0.016}"

# Best TTv1 config from Phase 1 (edit after Phase 1):
BEST_GAMMA="${BEST_GAMMA:-0.0}"
BEST_UIM="${BEST_UIM:-false}"
BEST_TE="${BEST_TE:-24}"

cd "$SCRIPT_DIR"

echo "=== Phase 4: TTv1 Final Confirmation ==="
echo "Config: gamma=$BEST_GAMMA, uim=$BEST_UIM, te=$BEST_TE, ln_lr=$LN_LR"

CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs $EPOCHS --n-bits 14 \
    --gamma $BEST_GAMMA \
    --units-in-mbatch $BEST_UIM \
    --transfer-every $BEST_TE \
    --ln-lr $LN_LR \
    --output-dir "$RESULTS_DIR/ttv1_final" \
    --log-every 20

echo ""
echo "=== Phase 4 Complete ==="
DIR="$RESULTS_DIR/ttv1_final"
if [ -f "$DIR/summary.json" ]; then
    python -c "
import json
d = json.load(open('$DIR/summary.json'))
r = d['results']
print(f'  best_f1={r[\"best_f1\"]:.2f}, final_f1={r[\"final_f1\"]:.2f}, em={r[\"final_em\"]:.2f}')
print(f'  steps={r[\"total_steps\"]}, wall_time={r[\"wall_time_s\"]:.0f}s')
"
fi

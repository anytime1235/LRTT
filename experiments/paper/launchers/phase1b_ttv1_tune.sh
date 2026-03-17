#!/bin/bash
# Phase 1B: TTv1 LR tuning (OPTIONAL — skip if auto_scale/scale_transfer_lr suffice)
# Uses TPE to tune analog_lr and classifier_lr for best TTv1 regime from Phase 1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1b}"
PYTHON="${PYTHON:-python}"

# Edit these based on Phase 1 results:
BEST_GAMMA="${BEST_GAMMA:-0.0}"
BEST_UIM="${BEST_UIM:-false}"
BEST_TE="${BEST_TE:-24}"

cd "$SCRIPT_DIR"

echo "=== Phase 1B: TTv1 LR Tuning (Optional) ==="
echo "Using: gamma=$BEST_GAMMA, uim=$BEST_UIM, te=$BEST_TE"

CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode tpe --method ttv1 --seed 42 \
    --epochs 2 --n-bits 14 \
    --gamma $BEST_GAMMA \
    --units-in-mbatch $BEST_UIM \
    --transfer-every $BEST_TE \
    --n-trials 20 \
    --db-dir "$RESULTS_DIR/db" \
    --output-dir "$RESULTS_DIR" \
    --log-every 50

echo "=== Phase 1B Complete ==="

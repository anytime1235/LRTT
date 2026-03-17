#!/bin/bash
# Phase 0: Smoke tests + LN LR comparison
# Part A: 5 methods × 50 steps × seed=42 → verify code paths
# Part B: LN LR ablation (ln_lr=analog_lr vs ln_lr=classifier_lr) on single_rpu, 2 epochs
# GPUs 1,2,3 (GPU 0 reserved).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase0}"
PYTHON="${PYTHON:-python}"

cd "$SCRIPT_DIR"

echo "=== Phase 0: Smoke Tests ==="
echo "Script dir: $SCRIPT_DIR"
echo "Results: $RESULTS_DIR"

# ---------------------------------------------------------------
# Part A: 5 methods × 50 steps (smoke)
# ---------------------------------------------------------------
echo ""
echo "--- Part A: Smoke (5 methods × 50 steps) ---"

# Round 1: GPU1=single_rpu, GPU2=mixed_precision, GPU3=ttv1
CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --max-steps 50 --seed 42 \
    --output-dir "$RESULTS_DIR/single_rpu" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --max-steps 50 --seed 42 \
    --output-dir "$RESULTS_DIR/mixed_precision" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --max-steps 50 --seed 42 \
    --gamma 0.0 \
    --output-dir "$RESULTS_DIR/ttv1" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID3=$!

wait $PID1 $PID2 $PID3
echo "Round 1 complete."

# Round 2: GPU1=cttv2, GPU2=ideal
CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode fixed --method cttv2 --max-steps 50 --seed 42 \
    --output-dir "$RESULTS_DIR/cttv2" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
    --mode fixed --method ideal --max-steps 50 --seed 42 \
    --output-dir "$RESULTS_DIR/ideal" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID2=$!

wait $PID1 $PID2
echo "Round 2 complete."

echo ""
echo "--- Part A Results ---"
for METHOD in single_rpu mixed_precision ttv1 cttv2 ideal; do
    DIR="$RESULTS_DIR/$METHOD"
    if [ -f "$DIR/summary.json" ]; then
        echo "  $METHOD: $(cat "$DIR/summary.json" | $PYTHON -c 'import sys,json; d=json.load(sys.stdin); print(f"steps={d[\"results\"][\"total_steps\"]}")')"
    else
        echo "  $METHOD: MISSING summary.json"
    fi
done

# ---------------------------------------------------------------
# Part B: LN LR ablation — 2 epochs, single_rpu 14-bit, seed=42
#   A: ln_lr = analog_lr (0.016)  ← current default
#   B: ln_lr = classifier_lr (0.003)
# ---------------------------------------------------------------
echo ""
echo "--- Part B: LN LR Ablation (2 epochs, single_rpu 14-bit) ---"

# GPU1: ln_lr=0.016 (=analog_lr, default)
CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --seed 42 \
    --epochs 2 --n-bits 14 \
    --ln-lr 0.016 \
    --output-dir "$RESULTS_DIR/ln_lr_ablation/ln_eq_analog" \
    --log-every 20 &
PID1=$!

# GPU2: ln_lr=0.003 (=classifier_lr)
CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --seed 42 \
    --epochs 2 --n-bits 14 \
    --ln-lr 0.003 \
    --output-dir "$RESULTS_DIR/ln_lr_ablation/ln_eq_classifier" \
    --log-every 20 &
PID2=$!

wait $PID1 $PID2
echo "LN LR ablation complete."

# Compare results
echo ""
echo "--- Part B Results ---"
$PYTHON -c "
import json, sys

results = {}
for tag, path in [('ln=analog(0.016)', '$RESULTS_DIR/ln_lr_ablation/ln_eq_analog/summary.json'),
                  ('ln=classifier(0.003)', '$RESULTS_DIR/ln_lr_ablation/ln_eq_classifier/summary.json')]:
    try:
        with open(path) as f:
            d = json.load(f)
        r = d['results']
        results[tag] = r['best_f1']
        print(f'  {tag}: best_f1={r[\"best_f1\"]:.2f}, final_f1={r[\"final_f1\"]:.2f}, em={r[\"final_em\"]:.2f}')
    except Exception as e:
        print(f'  {tag}: ERROR - {e}')
        results[tag] = -1

# Pick winner
if results:
    best_tag = max(results, key=results.get)
    best_ln_lr = '0.016' if 'analog' in best_tag else '0.003'
    print(f'')
    print(f'  >>> Winner: {best_tag}')
    print(f'  >>> BEST_LN_LR={best_ln_lr}')

    # Write best LN LR to file for downstream scripts
    with open('$RESULTS_DIR/ln_lr_ablation/best_ln_lr.txt', 'w') as f:
        f.write(best_ln_lr)
    print(f'  >>> Saved to $RESULTS_DIR/ln_lr_ablation/best_ln_lr.txt')
"

echo ""
echo "=== Phase 0 Complete ==="

#!/bin/bash
# LinearStep device gamma_up/down ratio sweep on TTv1
# Base: gamma=1.0, reset=1.0, 4ep, 14bit fast / 10bit slow (best phase1c config)
# Device: LinearStepDevice with 6T1C gamma, noise=0
# Sweep: ls_gamma_ratio = {0.5, 1.0, 2.0, 3.0} (applied to both gamma_up and gamma_down)
# Sequential execution on single GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/linearstep_gamma_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU="${GPU:-0}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================="
echo "  LinearStep Device: Gamma Ratio Sweep"
echo "  Base config: TTv1 gamma=1.0, reset=1.0, 4ep, 14b/10b"
echo "  Device: LinearStepDevice (6T1C gamma, noise-free)"
echo "  Sweep: ls_gamma_ratio = {0.5, 1.0, 2.0, 3.0}"
echo "  GPU: $GPU | Results: $RESULTS_DIR"
echo "  Start: $(date)"
echo "================================================================="

run() {
    local RATIO=$1
    local TAG="ls_gr${RATIO}"
    echo ""
    echo "--- [$TAG] ls_gamma_ratio=$RATIO (gamma_up=$(echo "-0.1678 * $RATIO" | bc), gamma_down=$(echo "0.1410 * $RATIO" | bc)) ---"
    echo "    Start: $(date)"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --gamma 1.0 \
        --units-in-mbatch true \
        --transfer-every 1 \
        --with-reset-prob 1.0 \
        --fast-lr 0.1 \
        --transfer-lr 1.0 \
        --scale-transfer-lr false \
        --ln-lr 0.003 \
        --device-type linear_step \
        --ls-gamma-up-ratio $RATIO \
        --ls-gamma-down-ratio $RATIO \
        --ls-noise-ratio 0 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 \
        2>&1 | tee "$RESULTS_DIR/${TAG}.log"

    echo "    Done: $(date)"
}

# Also run ConstantStep baseline for comparison
echo ""
echo "--- [baseline] ConstantStepDevice (no gamma_up/down) ---"
echo "    Start: $(date)"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 \
    --units-in-mbatch true \
    --transfer-every 1 \
    --with-reset-prob 1.0 \
    --fast-lr 0.1 \
    --transfer-lr 1.0 \
    --scale-transfer-lr false \
    --ln-lr 0.003 \
    --device-type constant_step \
    --output-dir "$RESULTS_DIR/baseline_cs" \
    --log-every 20 \
    2>&1 | tee "$RESULTS_DIR/baseline_cs.log"
echo "    Done: $(date)"

# Sweep LinearStep gamma ratios
for RATIO in 0.5 1.0 2.0 3.0; do
    run $RATIO
done

echo ""
echo "================================================================="
echo "  All experiments complete: $(date)"
echo "================================================================="

# Summary table
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/linearstep_gamma_sweep")
tags = [
    ("baseline_cs", "ConstantStep (baseline)"),
    ("ls_gr0.5",    "LinearStep r=0.5"),
    ("ls_gr1.0",    "LinearStep r=1.0 (6T1C)"),
    ("ls_gr2.0",    "LinearStep r=2.0"),
    ("ls_gr3.0",    "LinearStep r=3.0"),
]

print(f"\n{'Tag':<30} {'Best F1':>8} {'Final F1':>9} {'Final EM':>9}")
print("-" * 60)
for tag, label in tags:
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{label:<30} {d['best_f1']:>8.2f} {d['final_f1']:>9.2f} {d['final_em']:>9.2f}")
    except Exception:
        print(f"{label:<30} {'---':>8} {'---':>9} {'---':>9}")
PYEOF

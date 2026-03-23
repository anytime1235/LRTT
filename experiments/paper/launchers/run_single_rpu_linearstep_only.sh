#!/bin/bash
# SingleRPU LinearStep device gamma_up/down ratio sweep
# Mirrors run_linearstep_only.sh but uses single_rpu instead of ttv1
# Device: LinearStepDevice with 6T1C gamma, noise=0
# Sweep: ls_gamma_ratio = {0.5, 1.0, 2.0, 3.0}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/single_rpu_linearstep_gamma_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU="${GPU:-0}"
EPOCHS=4
BATCH_SIZE=12
GRAD_ACCUM=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================="
echo "  SingleRPU LinearStep Device: Gamma Ratio Sweep"
echo "  single_rpu, 4ep, 14b, io=8b"
echo "  batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
echo "  analog_lr=0.016, cls_lr=0.003, ln_lr=0.003, min_lr_rate=0.05"
echo "  Sweep: ls_gamma_ratio = {0.5, 1.0, 2.0, 3.0}"
echo "  GPU: $GPU | Results: $RESULTS_DIR"
echo "  Start: $(date)"
echo "================================================================="

# ConstantStep baseline for comparison
echo ""
echo "--- [1/5] [baseline_cs] ConstantStepDevice (no gamma_up/down) ---"
echo "    Start: $(date)"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --seed 42 \
    --epochs $EPOCHS --batch-size $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM \
    --n-bits 14 \
    --target-layers attention \
    --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.003 \
    --min-lr-rate 0.05 \
    --io-bits 8 \
    --device-type constant_step \
    --output-dir "$RESULTS_DIR/baseline_cs" \
    --log-every 20 \
    2>&1 | tee "$RESULTS_DIR/baseline_cs.log"
echo "    Done: $(date)"

# Sweep LinearStep gamma ratios
RUN=1
for RATIO in 0.5 1.0 2.0 3.0; do
    RUN=$((RUN + 1))
    TAG="ls_gr${RATIO}"
    echo ""
    echo "--- [$RUN/5] [$TAG] ls_gamma_ratio=$RATIO (gamma_up=$(echo "-0.1678 * $RATIO" | bc), gamma_down=$(echo "0.1410 * $RATIO" | bc)) ---"
    echo "    Start: $(date)"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method single_rpu --seed 42 \
        --epochs $EPOCHS --batch-size $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM \
        --n-bits 14 \
        --target-layers attention \
        --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.003 \
        --min-lr-rate 0.05 \
        --io-bits 8 \
        --device-type linear_step \
        --ls-gamma-up-ratio $RATIO \
        --ls-gamma-down-ratio $RATIO \
        --ls-noise-ratio 0 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 \
        2>&1 | tee "$RESULTS_DIR/${TAG}.log"

    echo "    Done: $(date)"
done

echo ""
echo "================================================================="
echo "  All 5 runs complete: $(date)"
echo "================================================================="

$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/single_rpu_linearstep_gamma_sweep")
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

#!/bin/bash
# LinearStep device noise ratio sweep on TTv1
# Base: gamma=1.0, reset=1.0, 4ep, 14bit fast / 10bit slow (best phase1c config)
# Device: LinearStepDevice with 6T1C gamma fixed (ratio=1.0), noise swept
# Sweep: ls_noise_ratio = {0.5, 1.0, 2.0, 3.0}
# Sequential execution on single GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/linearstep_noise_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU="${GPU:-0}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================="
echo "  LinearStep Device: Noise Ratio Sweep"
echo "  Base config: TTv1 gamma=1.0, reset=1.0, 4ep, 14b/10b"
echo "  Device: LinearStepDevice (6T1C gamma fixed r=1.0)"
echo "  Fixed:  gamma_up=-0.1678, gamma_down=0.1410"
echo "  Sweep:  ls_noise_ratio = {0.1, 0.3, 0.5, 1.0}"
echo "  GPU: $GPU | Results: $RESULTS_DIR"
echo "  Start: $(date)"
echo "================================================================="

run() {
    local NOISE_R=$1
    local TAG="ls_nr${NOISE_R}"
    echo ""
    echo "--- [$TAG] noise_ratio=$NOISE_R (dw_min_std=$(echo "0.3 * $NOISE_R" | bc), dw_min_dtod=$(echo "0.1 * $NOISE_R" | bc)) ---"
    echo "    Start: $(date)"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --target-layers attention \
        --batch-size 12 \
        --grad-accum-steps 4 \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --gamma 1.0 \
        --units-in-mbatch true \
        --transfer-every 4 \
        --with-reset-prob 1.0 \
        --fast-lr 0.1 \
        --transfer-lr 1.0 \
        --scale-transfer-lr false \
        --analog-lr 0.016 \
        --classifier-lr 0.003 \
        --ln-lr 0.003 \
        --warmup-ratio 0.05 \
        --min-lr-rate 0.05 \
        --io-bits 0 \
        --noise-management abs_max \
        --device-type linear_step \
        --ls-gamma-up-ratio 1.0 \
        --ls-gamma-down-ratio 1.0 \
        --ls-noise-ratio $NOISE_R \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 \
        2>&1 | tee "$RESULTS_DIR/${TAG}.log"

    echo "    Done: $(date)"
}

# Baseline: LinearStep with 6T1C gamma, noise=0 (same as gamma sweep r=1.0)
echo ""
echo "--- [baseline] LinearStep 6T1C gamma, noise=0 ---"
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
    --ls-gamma-up-ratio 1.0 \
    --ls-gamma-down-ratio 1.0 \
    --ls-noise-ratio 0 \
    --output-dir "$RESULTS_DIR/baseline_noisefree" \
    --log-every 20 \
    2>&1 | tee "$RESULTS_DIR/baseline_noisefree.log"
echo "    Done: $(date)"

# Sweep noise ratios
for NOISE_R in 0.1 0.3 0.5 1.0; do
    run $NOISE_R
done

echo ""
echo "================================================================="
echo "  All experiments complete: $(date)"
echo "================================================================="

# Summary table
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/linearstep_noise_sweep")
tags = [
    ("baseline_noisefree", "6T1C gamma, noise=0 (baseline)"),
    ("ls_nr0.1",           "6T1C gamma, noise r=0.1"),
    ("ls_nr0.3",           "6T1C gamma, noise r=0.3"),
    ("ls_nr0.5",           "6T1C gamma, noise r=0.5"),
    ("ls_nr1.0",           "6T1C gamma, noise r=1.0 (measured)"),
]

print(f"\n{'Tag':<40} {'Best F1':>8} {'Final F1':>9} {'Final EM':>9}")
print("-" * 70)
for tag, label in tags:
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{label:<40} {d['best_f1']:>8.2f} {d['final_f1']:>9.2f} {d['final_em']:>9.2f}")
    except Exception:
        print(f"{label:<40} {'---':>8} {'---':>9} {'---':>9}")

print("\nNoise parameters at each ratio:")
print(f"  {'r':>4} | {'dw_min_std':>10} | {'dw_min_dtod':>11} | {'SNR':>5}")
print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*11}-+-{'-'*5}")
for r in [0, 0.1, 0.3, 0.5, 1.0]:
    std = 0.3 * r
    dtod = 0.1 * r
    snr = f"{1/std:.1f}" if std > 0 else "inf"
    print(f"  {r:>4.1f} | {std:>10.2f} | {dtod:>11.2f} | {snr:>5}")
PYEOF

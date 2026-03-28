#!/bin/bash
# LinearStep device noise ratio sweep EXTENSION (2.0, 3.0)
# Continues from phase_linearstep_noise_sweep.sh which covered {0.1, 0.3, 0.5, 1.0}
# Base: gamma=1.0, reset=1.0, 4ep, 14bit fast / 10bit slow (best phase1c config)
# Device: LinearStepDevice with 6T1C gamma fixed (ratio=1.0), noise swept
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/linearstep_noise_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU="${GPU:-0}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================="
echo "  LinearStep Device: Noise Ratio Sweep EXTENSION"
echo "  Base config: TTv1 gamma=1.0, reset=1.0, 4ep, 14b/10b"
echo "  Device: LinearStepDevice (6T1C gamma fixed r=1.0)"
echo "  Sweep:  ls_noise_ratio = {2.0, 3.0}"
echo "  GPU: $GPU | Results: $RESULTS_DIR"
echo "  Start: $(date)"
echo "================================================================="

run() {
    local NOISE_R=$1
    local TAG="ls_nr${NOISE_R}"
    echo ""
    echo "[START] $TAG noise_ratio=$NOISE_R $(date)"

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

    echo "[DONE]  $TAG $(date)"
}

# Sweep extended noise ratios
for NOISE_R in 2.0 3.0; do
    run $NOISE_R
done

echo ""
echo "================================================================="
echo "  Noise sweep extension complete: $(date)"
echo "================================================================="

# Summary table (all noise ratios)
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/linearstep_noise_sweep")
tags = [
    ("baseline_noisefree", "noise=0 (baseline)"),
    ("ls_nr0.1",           "noise r=0.1"),
    ("ls_nr0.3",           "noise r=0.3"),
    ("ls_nr0.5",           "noise r=0.5"),
    ("ls_nr1.0",           "noise r=1.0 (6T1C)"),
    ("ls_nr2.0",           "noise r=2.0"),
    ("ls_nr3.0",           "noise r=3.0"),
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

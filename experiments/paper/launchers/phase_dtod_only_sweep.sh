#!/bin/bash
# dtod-only sweep: dw_min_std=0, noise_ratio swept to control dtod magnitude
# Base: gamma=1.0, reset=1.0, 4ep, 14bit fast / 10bit slow (best phase1c config)
# Device: LinearStepDevice with 6T1C gamma fixed (ratio=1.0)
# Already have: ls_nr1.0_no_dw_std (noise_ratio=1.0, dw_std=0)
# This script: noise_ratio = {0.1, 0.3, 0.5} with dw_std=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/linearstep_noise_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU="${GPU:-0}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================="
echo "  dtod-only Sweep (dw_min_std=0)"
echo "  Base config: TTv1 gamma=1.0, reset=1.0, 4ep, 14b/10b"
echo "  Device: LinearStepDevice (6T1C gamma fixed r=1.0)"
echo "  dw_min_std forced to 0 → only dtod noise active"
echo "  Sweep: noise_ratio = {0.1, 0.3, 0.5}"
echo "  (noise_ratio=1.0 already done as ls_nr1.0_no_dw_std)"
echo "  GPU: $GPU | Results: $RESULTS_DIR"
echo "  Start: $(date)"
echo "================================================================="

run() {
    local NOISE_R=$1
    local TAG="ls_nr${NOISE_R}_dtod_only"
    echo ""
    echo "[START] $TAG  noise_ratio=$NOISE_R, dw_min_std=0  $(date)"
    echo "        dtod params: dw_min_dtod=$(echo "0.1 * $NOISE_R" | bc), up_down_dtod=$(echo "0.01 * $NOISE_R" | bc)"

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
        --ls-dw-min-std 0 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 \
        2>&1 | tee "$RESULTS_DIR/${TAG}.log"

    echo "[DONE]  $TAG $(date)"
}

for NOISE_R in 0.1 0.3 0.5; do
    run $NOISE_R
done

echo ""
echo "================================================================="
echo "  dtod-only sweep complete: $(date)"
echo "================================================================="

# Summary table: dtod-only vs full noise vs dw_std-only
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/linearstep_noise_sweep")
tags = [
    ("baseline_noisefree",      "noise=0 (baseline)"),
    ("ls_nr0.1",                "r=0.1 full noise"),
    ("ls_nr0.1_dtod_only",      "r=0.1 dtod only"),
    ("ls_nr0.3",                "r=0.3 full noise"),
    ("ls_nr0.3_dtod_only",      "r=0.3 dtod only"),
    ("ls_nr0.5",                "r=0.5 full noise"),
    ("ls_nr0.5_dtod_only",      "r=0.5 dtod only"),
    ("ls_nr1.0",                "r=1.0 full noise"),
    ("ls_nr1.0_no_dw_std",      "r=1.0 dtod only"),
    ("ls_nr0_dw_std_only",      "r=1.0 dw_std only"),
]

print(f"\n{'Config':<30} {'Best F1':>8} {'Final F1':>9} {'Final EM':>9}")
print("-" * 60)
for tag, label in tags:
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{label:<30} {d['best_f1']:>8.2f} {d['final_f1']:>9.2f} {d['final_em']:>9.2f}")
    except Exception:
        print(f"{label:<30} {'---':>8} {'---':>9} {'---':>9}")

print("\ndtod noise at each ratio (dw_min_std=0):")
print(f"  {'r':>4} | {'dw_min_dtod':>11} | {'up_down_dtod':>12} | {'w/gamma_dtod':>12}")
print(f"  {'-'*4}-+-{'-'*11}-+-{'-'*12}-+-{'-'*12}")
for r in [0.1, 0.3, 0.5, 1.0]:
    print(f"  {r:>4.1f} | {0.1*r:>11.3f} | {0.01*r:>12.4f} | {0.05*r:>12.4f}")
PYEOF

#!/bin/bash
# Noise isolation sweep: c2c-only and d2d-only at multiple ratios
# Base: gamma=1.0, reset=1.0, 4ep, 14bit fast / 10bit slow
# Device: LinearStepDevice with 6T1C gamma fixed (ratio=1.0)
#
# Phase 1: c2c-only (d2d=0, dw_min_std = 0.3 * ratio)
# Phase 2: d2d-only (c2c=0, dtod params = baseline * ratio)
# Ratios: 0.1, 0.5, 1.0, 5.0, 10.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/linearstep_noise_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU="${GPU:-0}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

COMMON_ARGS=(
    --mode fixed --method ttv1 --seed 42
    --target-layers attention
    --batch-size 12
    --grad-accum-steps 4
    --epochs $EPOCHS --n-bits 14 --n-bits-slow 10
    --gamma 1.0
    --units-in-mbatch true
    --transfer-every 4
    --with-reset-prob 1.0
    --fast-lr 0.1
    --transfer-lr 1.0
    --scale-transfer-lr false
    --analog-lr 0.016
    --classifier-lr 0.003
    --ln-lr 0.003
    --warmup-ratio 0.05
    --min-lr-rate 0.05
    --io-bits 0
    --noise-management abs_max
    --device-type linear_step
    --ls-gamma-up-ratio 1.0
    --ls-gamma-down-ratio 1.0
    --log-every 20
)

echo "================================================================="
echo "  Noise Isolation Sweep"
echo "  Phase 1: c2c-only (d2d=0, sweep dw_min_std via ratio)"
echo "  Phase 2: d2d-only (c2c=0, sweep dtod via ratio)"
echo "  Ratios: 0.1, 0.5, 1.0, 5.0, 10.0"
echo "  GPU: $GPU | Results: $RESULTS_DIR"
echo "  Start: $(date)"
echo "================================================================="

# =====================================================================
# Phase 1: c2c-only (d2d noise = 0)
# dw_min_std = 0.3 * ratio, all dtod = 0
# =====================================================================
echo ""
echo "============== Phase 1: c2c-only sweep (d2d=0) ================"

for RATIO in 0.1 0.5 1.0 5.0 10.0; do
    TAG="ls_c2c_only_r${RATIO}"

    if [ -f "$RESULTS_DIR/$TAG/summary.json" ]; then
        echo "[SKIP] $TAG already complete"
        continue
    fi

    DW_MIN_STD=$(python3 -c "print(f'{0.3 * $RATIO:.4f}')")
    echo ""
    echo "[START] $TAG  ratio=$RATIO, dw_min_std=$DW_MIN_STD, all dtod=0  $(date)"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        "${COMMON_ARGS[@]}" \
        --ls-noise-ratio $RATIO \
        --ls-dw-min-dtod 0 \
        --output-dir "$RESULTS_DIR/$TAG" \
        2>&1 | tee "$RESULTS_DIR/${TAG}.log"

    echo "[DONE]  $TAG $(date)"
done

echo ""
echo "============== Phase 1 complete: $(date) ================"

# =====================================================================
# Phase 2: d2d-only (c2c noise = 0)
# dw_min_std = 0, dtod params = baseline * ratio
# =====================================================================
echo ""
echo "============== Phase 2: d2d-only sweep (c2c=0) ================"

for RATIO in 0.1 0.5 1.0 5.0 10.0; do
    TAG="ls_d2d_only_r${RATIO}"

    if [ -f "$RESULTS_DIR/$TAG/summary.json" ]; then
        echo "[SKIP] $TAG already complete"
        continue
    fi

    DW_MIN_DTOD=$(python3 -c "print(f'{0.1 * $RATIO:.4f}')")
    echo ""
    echo "[START] $TAG  ratio=$RATIO, dw_min_dtod=$DW_MIN_DTOD, dw_min_std=0  $(date)"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        "${COMMON_ARGS[@]}" \
        --ls-noise-ratio $RATIO \
        --ls-dw-min-std 0 \
        --output-dir "$RESULTS_DIR/$TAG" \
        2>&1 | tee "$RESULTS_DIR/${TAG}.log"

    echo "[DONE]  $TAG $(date)"
done

echo ""
echo "============== Phase 2 complete: $(date) ================"

# =====================================================================
# Summary table
# =====================================================================
echo ""
echo "================================================================="
echo "  All noise isolation sweeps complete: $(date)"
echo "================================================================="

$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/linearstep_noise_sweep")

print("\n===== c2c-only (d2d=0) =====")
print(f"{'Ratio':>6} {'dw_min_std':>10} {'Best F1':>8} {'Final F1':>9} {'Final EM':>9}")
print("-" * 50)
for r in [0.1, 0.5, 1.0, 5.0, 10.0]:
    tag = f"ls_c2c_only_r{r}"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{r:>6.1f} {0.3*r:>10.4f} {d['best_f1']:>8.2f} {d['final_f1']:>9.2f} {d['final_em']:>9.2f}")
    except Exception:
        print(f"{r:>6.1f} {0.3*r:>10.4f} {'---':>8} {'---':>9} {'---':>9}")

print("\n===== d2d-only (c2c=0) =====")
print(f"{'Ratio':>6} {'dw_min_dtod':>11} {'Best F1':>8} {'Final F1':>9} {'Final EM':>9}")
print("-" * 50)
for r in [0.1, 0.5, 1.0, 5.0, 10.0]:
    tag = f"ls_d2d_only_r{r}"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{r:>6.1f} {0.1*r:>11.4f} {d['best_f1']:>8.2f} {d['final_f1']:>9.2f} {d['final_em']:>9.2f}")
    except Exception:
        print(f"{r:>6.1f} {0.1*r:>11.4f} {'---':>8} {'---':>9} {'---':>9}")
PYEOF

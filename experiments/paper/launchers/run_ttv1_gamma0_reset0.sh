#!/bin/bash
# TTv1 with gamma=0, reset=1.0, transfer_lr=1.0
# Constant step device (no noise), same conditions as gamma sweep baseline
# 14bit fast / 10bit slow, fast_lr=0.1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/linearstep_gamma_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU="${GPU:-0}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

TAG="cs_g0.0_r1"

echo "================================================================="
echo "  TTv1: gamma=0, reset=1.0, transfer_lr=1.0"
echo "  Device: ConstantStep (noise-free), 14b fast / 10b slow"
echo "  GPU: $GPU | Output: $RESULTS_DIR/$TAG"
echo "  Start: $(date)"
echo "================================================================="

if [ -f "$RESULTS_DIR/$TAG/summary.json" ]; then
    echo "[SKIP] $TAG already complete"
    exit 0
fi

CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --target-layers attention \
    --batch-size 16 \
    --grad-accum-steps 3 \
    --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
    --gamma 0 \
    --units-in-mbatch true \
    --transfer-every 3 \
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
    --device-type constant_step \
    --output-dir "$RESULTS_DIR/$TAG" \
    --log-every 20 \
    2>&1 | tee "$RESULTS_DIR/${TAG}.log"

echo ""
echo "================================================================="
echo "  Done: $(date)"
echo "================================================================="

$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/linearstep_gamma_sweep")
tag = "cs_g0.0_r1"
path = os.path.join(base, tag, "summary.json")
try:
    d = json.load(open(path))["results"]
    print(f"\nResult: Best F1={d['best_f1']:.2f}, Final F1={d['final_f1']:.2f}, Final EM={d['final_em']:.2f}")
except Exception as e:
    print(f"\nFailed to read summary: {e}")
PYEOF

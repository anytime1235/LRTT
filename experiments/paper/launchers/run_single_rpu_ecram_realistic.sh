#!/bin/bash
# SingleRPU ECRAM & RRAM realistic device experiments (noise-free)
# Mirrors run_ecram_realistic.sh but uses single_rpu instead of ttv1
#
# Experiments:
#   1. ECRAM Li (Tang et al. IEDM 2018): LinearStepDevice, gamma_up=0.1153, gamma_down=0.5085
#   2. RRAM HfO2 (Gong & Rasch, IEDM 2022): SoftBoundsReferenceDevice (inherent nonlinearity)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/single_rpu_ecram_realistic}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
GPU="${GPU:-0}"
EPOCHS=4
BATCH_SIZE=12
GRAD_ACCUM=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "================================================================="
echo "  SingleRPU ECRAM & RRAM Realistic Device Experiments (noise-free)"
echo "  single_rpu, 4ep, 14b, io=8b"
echo "  batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
echo "  analog_lr=0.016, cls_lr=0.003, ln_lr=0.003, min_lr_rate=0.05"
echo "  GPU: $GPU | Results: $RESULTS_DIR"
echo "  Start: $(date)"
echo "================================================================="

# --- [1/2] ECRAM Li-based (Tang et al. IEDM 2018) ---
# EcRamPresetDevice: LinearStepDevice, gamma_up=0.1153, gamma_down=0.5085
TAG="ecram_li"
echo ""
echo "--- [1/2] [$TAG] EcRam Li: LinearStep gamma_up=0.1153, gamma_down=0.5085 ---"
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
    --ls-gamma-up 0.1153 \
    --ls-gamma-down 0.5085 \
    --ls-noise-ratio 0 \
    --output-dir "$RESULTS_DIR/$TAG" \
    --log-every 20 \
    2>&1 | tee "$RESULTS_DIR/${TAG}.log"
echo "    Done: $(date)"

# --- [2/2] RRAM HfO2 (Gong & Rasch, IEDM 2022) ---
# ReRamArrayHfO2PresetDevice: SoftBoundsReferenceDevice (no gamma, inherent nonlinearity)
TAG="rram_hfo2"
echo ""
echo "--- [2/2] [$TAG] RRAM HfO2: SoftBoundsRef (inherent soft bounds nonlinearity) ---"
echo "    Start: $(date)"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
    --mode fixed --method single_rpu --seed 42 \
    --epochs $EPOCHS --batch-size $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM \
    --n-bits 14 \
    --target-layers attention \
    --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.003 \
    --min-lr-rate 0.05 \
    --io-bits 8 \
    --device-type soft_bounds \
    --ls-noise-ratio 0 \
    --output-dir "$RESULTS_DIR/$TAG" \
    --log-every 20 \
    2>&1 | tee "$RESULTS_DIR/${TAG}.log"
echo "    Done: $(date)"

echo ""
echo "================================================================="
echo "  All 2 experiments complete: $(date)"
echo "================================================================="

# Summary table
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/single_rpu_ecram_realistic")
tags = [
    ("ecram_li",   "EcRam Li (LinearStep, g_up=0.115 g_dn=0.509)"),
    ("rram_hfo2",  "RRAM HfO2 (SoftBoundsRef)"),
]

print(f"\n{'Tag':<50} {'Best F1':>8} {'Final F1':>9} {'Final EM':>9}")
print("-" * 80)
for tag, label in tags:
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{label:<50} {d['best_f1']:>8.2f} {d['final_f1']:>9.2f} {d['final_em']:>9.2f}")
    except Exception:
        print(f"{label:<50} {'---':>8} {'---':>9} {'---':>9}")
PYEOF

#!/bin/bash
# Diagnostic D2b: TTv1 pulse ablation — stochastic vs deterministic fast pulse
# 2 runs × 1024 steps × seed 42 — 8-bit slow only
# Focused mechanistic claim about unbiased accumulation
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/diag_D2b_pulse_ablation}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Diagnostic D2b: Pulse Ablation (1024 steps each) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

COMMON_FLAGS="--mode fixed --seed 42 --epochs 1 --max-steps 1024 --batch-size 48 \
  --diag-carry-path --diag-steps 128 \
  --diag-checkpoint-steps 256,512,768,1024 \
  --diag-vrc-windows 1,16,64,256,768 \
  --diag-layer-set 0,5,11 \
  --log-every 20"

TTv1_COMMON="--method ttv1 --ttv1-mode residual_lane \
  --units-in-mbatch true --transfer-every 1 --n-reads-per-transfer 1 \
  --n-bits 14 --n-bits-slow 8 --gamma 1.0 --with-reset-prob 1.0"

# GPU 1: stochastic fast + stochastic transfer
run_gpu1() {
    local TAG="stoch_fast_stoch_transfer"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        $COMMON_FLAGS $TTv1_COMMON \
        --ttv1-fast-pulse-type stochastic \
        --ttv1-transfer-pulse-type stochastic \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 1] DONE  $TAG $(date)"
}

# GPU 2: deterministic fast + stochastic transfer
run_gpu2() {
    local TAG="det_fast_stoch_transfer"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS $TTv1_COMMON \
        --ttv1-fast-pulse-type deterministic \
        --ttv1-transfer-pulse-type stochastic \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 2] DONE  $TAG $(date)"
}

run_gpu1 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
run_gpu2 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!

wait $PID1 $PID2
echo ""
echo "=== D2b complete: $(date) ==="

# Summary
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/diag_D2b_pulse_ablation")
tags = ["stoch_fast_stoch_transfer", "det_fast_stoch_transfer"]

print(f"\n{'Tag':<35} | {'mean_cos':>8} | {'VRC_256':>8} | {'fast_sat':>8} | {'PMR':>8}")
print("-" * 80)
for tag in tags:
    spath = os.path.join(base, tag, "carry_path_summary.json")
    try:
        d = json.load(open(spath))
        agg = d.get("aggregate", {})
        w = d.get("windows", {}).get("256", {})
        gd = d.get("ttv1_gamma_diag", {})
        print(f"{tag:<35} | {agg.get('mean_cosine_sim', 0):>8.4f} | "
              f"{w.get('mean_VRC_K', 0):>8.4f} | "
              f"{gd.get('mean_fast_sat_ratio', 0):>8.4f} | "
              f"{gd.get('mean_pmr', 0):>8.4f}")
    except Exception as e:
        print(f"{tag:<35} | {'FAIL':>8} | {str(e)[:30]}")
PYEOF

#!/bin/bash
# ECO E0: Smoke test — 6 methods × 100 steps × seed 42
# 2 waves of 3 GPUs (GPU 0 reserved)
# Methods: eco_ref(stoch), eco_ref(rtn), mixed_precision, ttv1(hb), ttv1(rl), ttv1(rl_nr)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/eco_E0_smoke}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== ECO E0: Smoke Test (100 steps each) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

# Wave 1: eco_ref(stoch), eco_ref(rtn), mixed_precision on GPUs 1,2,3
wave1_gpu1() {
    local TAG="eco_ref_stoch"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        --mode fixed --method eco_ref --seed 42 \
        --epochs 1 --max-steps 100 --n-bits 10 \
        --eco-rounding stochastic \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 10 --diag-carry-path
    echo "[GPU 1] DONE  $TAG $(date)"
}

wave1_gpu2() {
    local TAG="eco_ref_rtn"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        --mode fixed --method eco_ref --seed 42 \
        --epochs 1 --max-steps 100 --n-bits 10 \
        --eco-rounding rtn \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 10 --diag-carry-path
    echo "[GPU 2] DONE  $TAG $(date)"
}

wave1_gpu3() {
    local TAG="mixed_precision_10b"
    echo "[GPU 3] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
        --mode fixed --method mixed_precision --seed 42 \
        --epochs 1 --max-steps 100 --n-bits 10 \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 10 --diag-carry-path
    echo "[GPU 3] DONE  $TAG $(date)"
}

wave1_gpu1 2>&1 | tee "$RESULTS_DIR/wave1_gpu1.log" &
PID1=$!
wave1_gpu2 2>&1 | tee "$RESULTS_DIR/wave1_gpu2.log" &
PID2=$!
wave1_gpu3 2>&1 | tee "$RESULTS_DIR/wave1_gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3
echo "Wave 1 complete: $(date)"

# Wave 2: ttv1 hidden_buffer, residual_lane, residual_lane_noreset
wave2_gpu1() {
    local TAG="ttv1_hidden_buffer"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs 1 --max-steps 100 \
        --n-bits 14 --n-bits-slow 10 \
        --ttv1-mode hidden_buffer \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 10 --diag-carry-path
    echo "[GPU 1] DONE  $TAG $(date)"
}

wave2_gpu2() {
    local TAG="ttv1_residual_lane"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs 1 --max-steps 100 \
        --n-bits 14 --n-bits-slow 10 \
        --ttv1-mode residual_lane \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 10 --diag-carry-path
    echo "[GPU 2] DONE  $TAG $(date)"
}

wave2_gpu3() {
    local TAG="ttv1_residual_lane_noreset"
    echo "[GPU 3] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs 1 --max-steps 100 \
        --n-bits 14 --n-bits-slow 10 \
        --ttv1-mode residual_lane_noreset \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 10 --diag-carry-path
    echo "[GPU 3] DONE  $TAG $(date)"
}

wave2_gpu1 2>&1 | tee "$RESULTS_DIR/wave2_gpu1.log" &
PID1=$!
wave2_gpu2 2>&1 | tee "$RESULTS_DIR/wave2_gpu2.log" &
PID2=$!
wave2_gpu3 2>&1 | tee "$RESULTS_DIR/wave2_gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3
echo ""
echo "=== All E0 smoke tests complete: $(date) ==="

# Summary table
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/eco_E0_smoke")
tags = [
    "eco_ref_stoch", "eco_ref_rtn", "mixed_precision_10b",
    "ttv1_hidden_buffer", "ttv1_residual_lane", "ttv1_residual_lane_noreset",
]

print(f"\n{'Method':<30} | {'Steps':>6} | {'Loss':>8}")
print("-" * 55)
for tag in tags:
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))
        steps = d["results"]["total_steps"]
        # Read last loss from training_log
        log_path = os.path.join(base, tag, "training_log.csv")
        import csv
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        last_loss = rows[-1]["loss"] if rows else "N/A"
        print(f"{tag:<30} | {steps:>6} | {last_loss:>8}")
    except Exception as e:
        print(f"{tag:<30} | {'FAIL':>6} | {str(e)[:20]}")
PYEOF

#!/bin/bash
# ECO E2: TTv1 pulse factorial — 2 modes × 2 fast_pulse × 2 transfer_pulse = 8 runs
# Modes: hidden_buffer, residual_lane
# Fast pulse: stochastic, deterministic
# Transfer pulse: stochastic, deterministic
# All at 4 epochs, seed 42, 14-bit fast / 10-bit slow
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/eco_E2_pulse}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== ECO E2: TTv1 Pulse Factorial ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local GPU=$1 MODE=$2 FP=$3 TP=$4
    local TAG="${MODE}_fp${FP}_tp${TP}"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --ttv1-mode $MODE \
        --ttv1-fast-pulse-type $FP \
        --ttv1-transfer-pulse-type $TP \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 --diag-carry-path --diag-steps 500
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

# GPU 1: hidden_buffer × 4 pulse combos
gpu1_pipeline() {
    run 1 hidden_buffer stochastic stochastic
    run 1 hidden_buffer stochastic deterministic
    run 1 hidden_buffer deterministic stochastic
}

# GPU 2: hidden_buffer(det,det) + residual_lane × 3 combos
gpu2_pipeline() {
    run 2 hidden_buffer deterministic deterministic
    run 2 residual_lane stochastic stochastic
    run 2 residual_lane stochastic deterministic
}

# GPU 3: residual_lane × 2 combos
gpu3_pipeline() {
    run 3 residual_lane deterministic stochastic
    run 3 residual_lane deterministic deterministic
}

gpu1_pipeline 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
gpu2_pipeline 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!
gpu3_pipeline 2>&1 | tee "$RESULTS_DIR/gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3

echo ""
echo "=== All E2 experiments complete: $(date) ==="

# Summary heatmap (text)
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/eco_E2_pulse")
modes = ["hidden_buffer", "residual_lane"]
fast_pulses = ["stochastic", "deterministic"]
transfer_pulses = ["stochastic", "deterministic"]

for mode in modes:
    print(f"\n{mode}:")
    print(f"  {'':>15} | {'tp=stoch':>10} | {'tp=det':>10}")
    print(f"  {'-'*40}")
    for fp in fast_pulses:
        row = []
        for tp in transfer_pulses:
            tag = f"{mode}_fp{fp}_tp{tp}"
            path = os.path.join(base, tag, "summary.json")
            try:
                d = json.load(open(path))["results"]
                row.append(f"{d['best_f1']:.2f}")
            except:
                row.append("---")
        print(f"  {'fp=' + fp:>15} | {row[0]:>10} | {row[1]:>10}")
PYEOF

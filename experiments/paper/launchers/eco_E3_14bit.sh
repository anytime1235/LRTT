#!/bin/bash
# ECO E3: 14-bit upper bound — single_rpu 14-bit stochastic + deterministic
# 2 runs, 4 epochs, seed 42
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/eco_E3_14bit}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== ECO E3: 14-bit Upper Bound ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local GPU=$1 PT=$2
    local TAG="single_rpu_14b_${PT}"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method single_rpu --seed 42 \
        --epochs $EPOCHS --n-bits 14 \
        --pulse-type $PT --desired-bl 31 \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 --diag-carry-path --diag-steps 500
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

# Run both on separate GPUs
gpu1_pipeline() {
    run 1 stochastic
}

gpu2_pipeline() {
    run 2 deterministic
}

gpu1_pipeline 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
gpu2_pipeline 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!

wait $PID1 $PID2

echo ""
echo "=== All E3 experiments complete: $(date) ==="

# Summary
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/eco_E3_14bit")
for pt in ["stochastic", "deterministic"]:
    tag = f"single_rpu_14b_{pt}"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"14-bit {pt}: best_f1={d['best_f1']:.2f}, final_f1={d['final_f1']:.2f}, em={d['final_em']:.2f}")
    except Exception as e:
        print(f"14-bit {pt}: FAIL ({e})")
PYEOF

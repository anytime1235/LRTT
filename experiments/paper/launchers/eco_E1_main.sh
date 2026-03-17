#!/bin/bash
# ECO E1: Main comparison — 8 methods × 3 seeds × 4 epochs = 24 runs
# Distributed across GPU 1,2,3 (8 runs per GPU)
# Common: attention QKVO, ln_lr=classifier_lr=0.003, analog_lr=0.016,
#         desired_bl=31, batch_size=48, warmup_ratio=0.05, min_lr_rate=0.5
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/eco_E1_main}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== ECO E1: Main Comparison (8 methods × 3 seeds × 4 epochs) ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run_single_rpu() {
    local GPU=$1 SEED=$2 PT=$3
    local TAG="single_rpu_${PT}_s${SEED}"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method single_rpu --seed $SEED \
        --epochs $EPOCHS --n-bits 10 \
        --pulse-type $PT --desired-bl 31 \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 --diag-carry-path --diag-steps 500
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

run_eco_ref() {
    local GPU=$1 SEED=$2 ROUNDING=$3
    local TAG="eco_ref_${ROUNDING}_s${SEED}"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method eco_ref --seed $SEED \
        --epochs $EPOCHS --n-bits 10 \
        --eco-rounding $ROUNDING \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 --diag-carry-path --diag-steps 500
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

run_mixed_precision() {
    local GPU=$1 SEED=$2
    local TAG="mixed_precision_s${SEED}"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method mixed_precision --seed $SEED \
        --epochs $EPOCHS --n-bits 10 \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 --diag-carry-path --diag-steps 500
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

run_ttv1() {
    local GPU=$1 SEED=$2 MODE=$3
    local TAG="ttv1_${MODE}_s${SEED}"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed $SEED \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --ttv1-mode $MODE \
        --ln-lr 0.003 --classifier-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20 --diag-carry-path --diag-steps 500
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

# GPU 1: single_rpu(det) × 3 seeds + single_rpu(stoch) × 3 seeds + eco_ref(rtn) × 2 seeds
gpu1_pipeline() {
    for SEED in 42 43 44; do
        run_single_rpu 1 $SEED deterministic
    done
    for SEED in 42 43 44; do
        run_single_rpu 1 $SEED stochastic
    done
    run_eco_ref 1 42 rtn
    run_eco_ref 1 43 rtn
}

# GPU 2: eco_ref(rtn) × 1 seed + eco_ref(stoch) × 3 seeds + mixed_precision × 3 seeds + ttv1(hb) × 1 seed
gpu2_pipeline() {
    run_eco_ref 2 44 rtn
    for SEED in 42 43 44; do
        run_eco_ref 2 $SEED stochastic
    done
    for SEED in 42 43 44; do
        run_mixed_precision 2 $SEED
    done
    run_ttv1 2 42 hidden_buffer
}

# GPU 3: ttv1(hb) × 2 seeds + ttv1(rl) × 3 seeds + ttv1(rl_nr) × 3 seeds
gpu3_pipeline() {
    for SEED in 43 44; do
        run_ttv1 3 $SEED hidden_buffer
    done
    for SEED in 42 43 44; do
        run_ttv1 3 $SEED residual_lane
    done
    for SEED in 42 43 44; do
        run_ttv1 3 $SEED residual_lane_noreset
    done
}

gpu1_pipeline 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
gpu2_pipeline 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!
gpu3_pipeline 2>&1 | tee "$RESULTS_DIR/gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3

echo ""
echo "=== All E1 experiments complete: $(date) ==="

# Summary table
$PYTHON << 'PYEOF'
import json, os, csv

base = os.environ.get("RESULTS_DIR", "results/paper/eco_E1_main")
methods = [
    ("single_rpu_deterministic", "SingleRPU (det)"),
    ("single_rpu_stochastic", "SingleRPU (stoch)"),
    ("eco_ref_rtn", "ECO ref (RTN)"),
    ("eco_ref_stochastic", "ECO ref (stoch)"),
    ("mixed_precision", "MixedPrecision"),
    ("ttv1_hidden_buffer", "TTv1 HiddenBuf"),
    ("ttv1_residual_lane", "TTv1 ResLane"),
    ("ttv1_residual_lane_noreset", "TTv1 ResLane NR"),
]
seeds = [42, 43, 44]

rows = []
print(f"\n{'Method':<25} | {'Seed 42':>8} | {'Seed 43':>8} | {'Seed 44':>8} | {'Mean±Std':>12}")
print("-" * 70)
for prefix, label in methods:
    f1s = []
    seed_strs = []
    for s in seeds:
        tag = f"{prefix}_s{s}"
        path = os.path.join(base, tag, "summary.json")
        try:
            d = json.load(open(path))["results"]
            f1 = d["best_f1"]
            f1s.append(f1)
            seed_strs.append(f"{f1:.2f}")
        except:
            seed_strs.append("---")
    if f1s:
        import numpy as np
        mean = np.mean(f1s)
        std = np.std(f1s)
        summary_str = f"{mean:.2f}±{std:.2f}"
    else:
        summary_str = "---"
    print(f"{label:<25} | {seed_strs[0]:>8} | {seed_strs[1]:>8} | {seed_strs[2]:>8} | {summary_str:>12}")
    for i, s in enumerate(seeds):
        rows.append({"method": prefix, "seed": s, "best_f1": f1s[i] if i < len(f1s) else 0})

csv_path = os.path.join(base, "eco_E1_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "seed", "best_f1"])
    w.writeheader()
    w.writerows(rows)
print(f"\nCSV saved: {csv_path}")
PYEOF

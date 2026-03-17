#!/bin/bash
# Phase 3: Bit-Width Sweep — Single RPU vs Mixed Precision vs TTv1
# 4 epochs, seed=42, warmup applied to ALL param groups
#
# A) Single RPU: bits={8,10,12,14,16}, analog_lr=0.016, cls_lr=0.003
# B) Mixed Precision: 10-bit fixed, analog_lr=0.0357, cls_lr=0.000763 (GPU0 best)
# C) TTv1: fast tile bits={8,10,12,14,16}, slow tile=10-bit fixed
#          best conditions: gamma=0.0, uim=true, te=1, fast_lr=0.1
#
# GPU 1,2,3 parallel
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase3_bitsweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 3: Bit-Width Sweep (4 epochs) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

# --- Single RPU ---
run_single_rpu() {
    local GPU=$1 BITS=$2
    local TAG="single_rpu_${BITS}b"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method single_rpu --seed 42 \
        --epochs $EPOCHS --n-bits $BITS \
        --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

# --- Mixed Precision (10-bit, GPU0 best LR) ---
run_mixed_prec() {
    local GPU=$1
    local TAG="mixed_prec_10b"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method mixed_precision --seed 42 \
        --epochs $EPOCHS --n-bits 10 \
        --analog-lr 0.0357 --classifier-lr 0.000763 --ln-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

# --- TTv1 (fast=variable bits, slow=10-bit) ---
run_ttv1() {
    local GPU=$1 BITS_FAST=$2
    local TAG="ttv1_fast${BITS_FAST}b_slow10b"
    echo "[GPU $GPU] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS --n-bits $BITS_FAST --n-bits-slow 10 \
        --gamma 0.0 \
        --units-in-mbatch true \
        --transfer-every 1 \
        --fast-lr 0.1 \
        --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

# GPU 1: single_rpu 8,10,12,14,16 (5 experiments)
gpu1_pipeline() {
    run_single_rpu 1 8
    run_single_rpu 1 10
    run_single_rpu 1 12
    run_single_rpu 1 14
    run_single_rpu 1 16
}

# GPU 2: ttv1 fast=8,10,12,14,16 slow=10 (5 experiments)
gpu2_pipeline() {
    run_ttv1 2 8
    run_ttv1 2 10
    run_ttv1 2 12
    run_ttv1 2 14
    run_ttv1 2 16
}

# GPU 3: mixed_prec 10b + help with longer runs
gpu3_pipeline() {
    run_mixed_prec 3
}

gpu1_pipeline 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
gpu2_pipeline 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!
gpu3_pipeline 2>&1 | tee "$RESULTS_DIR/gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3

echo ""
echo "=== All experiments complete: $(date) ==="

# Generate CSV summary
$PYTHON << 'PYEOF'
import json, os, csv

base = os.environ.get("RESULTS_DIR", "results/paper/phase3_bitsweep")

rows = []
# Single RPU
for bits in [8, 10, 12, 14, 16]:
    tag = f"single_rpu_{bits}b"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        rows.append({"method": "single_rpu", "bits_fast": bits, "bits_slow": bits,
                      "best_f1": d["best_f1"], "final_f1": d["final_f1"], "final_em": d["final_em"]})
    except:
        rows.append({"method": "single_rpu", "bits_fast": bits, "bits_slow": bits,
                      "best_f1": 0, "final_f1": 0, "final_em": 0})

# Mixed Precision
tag = "mixed_prec_10b"
path = os.path.join(base, tag, "summary.json")
try:
    d = json.load(open(path))["results"]
    rows.append({"method": "mixed_precision", "bits_fast": 10, "bits_slow": 10,
                  "best_f1": d["best_f1"], "final_f1": d["final_f1"], "final_em": d["final_em"]})
except:
    rows.append({"method": "mixed_precision", "bits_fast": 10, "bits_slow": 10,
                  "best_f1": 0, "final_f1": 0, "final_em": 0})

# TTv1
for bits in [8, 10, 12, 14, 16]:
    tag = f"ttv1_fast{bits}b_slow10b"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        rows.append({"method": "ttv1", "bits_fast": bits, "bits_slow": 10,
                      "best_f1": d["best_f1"], "final_f1": d["final_f1"], "final_em": d["final_em"]})
    except:
        rows.append({"method": "ttv1", "bits_fast": bits, "bits_slow": 10,
                      "best_f1": 0, "final_f1": 0, "final_em": 0})

csv_path = os.path.join(base, "bit_sweep_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "bits_fast", "bits_slow", "best_f1", "final_f1", "final_em"])
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV saved: {csv_path}")

# Print table
print(f"\n{'Method':<20} {'Bits(Fast)':<12} {'Bits(Slow)':<12} {'Best F1':>8} {'Final F1':>9} {'Final EM':>9}")
print("-" * 72)
for r in rows:
    if r["best_f1"] > 0:
        print(f"{r['method']:<20} {r['bits_fast']:<12} {r['bits_slow']:<12} {r['best_f1']:8.2f} {r['final_f1']:9.2f} {r['final_em']:9.2f}")
    else:
        print(f"{r['method']:<20} {r['bits_fast']:<12} {r['bits_slow']:<12} {'MISSING':>8}")
PYEOF

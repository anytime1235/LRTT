#!/bin/bash
# Phase 1C (4ep): Gamma × Reset Sweep with continuity-preserving transfer_lr
# Fixed: uim=true, te=1, fast_lr=0.1, scale_transfer_lr=False, transfer_lr=gamma
#        Fast=14bit, Slow=10bit, 4 epochs, seed=42, ln_lr=0.003
# Variables: gamma={0.01,0.05,0.1,0.3,0.5,1.0} × reset_prob={0,1.0}
# Total: 12 experiments, GPU 1/2/3 × 4 each
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1c_4ep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 1C (4ep): Gamma × Reset Sweep ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local GPU=$1 GAMMA=$2 RESET=$3
    local TAG="g${GAMMA}_r${RESET}"
    echo "[GPU $GPU] START $TAG (gamma=$GAMMA, reset=$RESET, tlr=$GAMMA) $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --gamma $GAMMA \
        --units-in-mbatch true \
        --transfer-every 1 \
        --with-reset-prob $RESET \
        --fast-lr 0.1 \
        --transfer-lr $GAMMA \
        --scale-transfer-lr false \
        --ln-lr 0.003 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

# GPU 1: gamma=0.01,0.05,0.1,0.3 × reset=0 (4 experiments)
gpu1_pipeline() {
    run 1 0.01 0
    run 1 0.05 0
    run 1 0.1  0
    run 1 0.3  0
}

# GPU 2: gamma=0.01,0.05,0.1,0.3 × reset=1.0 (4 experiments)
gpu2_pipeline() {
    run 2 0.01 1.0
    run 2 0.05 1.0
    run 2 0.1  1.0
    run 2 0.3  1.0
}

# GPU 3: gamma=0.5,1.0 × reset=0,1.0 (4 experiments)
gpu3_pipeline() {
    run 3 0.5 0
    run 3 0.5 1.0
    run 3 1.0 0
    run 3 1.0 1.0
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

# CSV + Table
$PYTHON << 'PYEOF'
import json, os, csv

base = os.environ.get("RESULTS_DIR", "results/paper/phase1c_4ep")
gammas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
resets = [0, 1.0]

rows = []
for g in gammas:
    for r in resets:
        tag = f"g{g}_r{r}"
        path = os.path.join(base, tag, "summary.json")
        try:
            d = json.load(open(path))["results"]
            rows.append({"gamma": g, "reset_prob": r,
                         "best_f1": d["best_f1"], "final_f1": d["final_f1"], "final_em": d["final_em"]})
        except:
            rows.append({"gamma": g, "reset_prob": r,
                         "best_f1": 0, "final_f1": 0, "final_em": 0})

csv_path = os.path.join(base, "gamma_reset_sweep_4ep.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["gamma", "reset_prob", "best_f1", "final_f1", "final_em"])
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV saved: {csv_path}")

print(f"\n{'gamma':>6} | {'reset=0':>10} | {'reset=1.0':>10}")
print("-" * 35)
for g in gammas:
    r0 = [r for r in rows if r["gamma"]==g and r["reset_prob"]==0]
    r1 = [r for r in rows if r["gamma"]==g and r["reset_prob"]==1.0]
    f0 = f"{r0[0]['best_f1']:.2f}" if r0 and r0[0]['best_f1']>0 else "---"
    f1 = f"{r1[0]['best_f1']:.2f}" if r1 and r1[0]['best_f1']>0 else "---"
    print(f"{g:>6} | {f0:>10} | {f1:>10}")
PYEOF

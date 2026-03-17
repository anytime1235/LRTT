#!/bin/bash
# Phase 1C: Gamma × Reset Prob Sweep (with continuity-preserving transfer_lr)
# Fixed: uim=true, te=1, fast_lr=0.1, scale_transfer_lr=False, transfer_lr=gamma
#        Fast=14bit, Slow=10bit, 2 epochs, seed=42, ln_lr=0.003
# Variables: gamma={0.01,0.05,0.1,0.3,0.5,1.0} × reset_prob={0,0.01,0.1,0.5,1.0}
# Total: 30 experiments, GPU 1/2/3 × 10 each
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1c}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS="${EPOCHS:-2}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 1C: Gamma × Reset Sweep (continuity-preserving) ==="
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

# GPU 1: gamma=0.01,0.05 × all reset_probs (10 experiments)
gpu1_pipeline() {
    for G in 0.01 0.05; do
        for R in 0 0.01 0.1 0.5 1.0; do
            run 1 $G $R
        done
    done
}

# GPU 2: gamma=0.1,0.3 × all reset_probs (10 experiments)
gpu2_pipeline() {
    for G in 0.1 0.3; do
        for R in 0 0.01 0.1 0.5 1.0; do
            run 2 $G $R
        done
    done
}

# GPU 3: gamma=0.5,1.0 × all reset_probs (10 experiments)
gpu3_pipeline() {
    for G in 0.5 1.0; do
        for R in 0 0.01 0.1 0.5 1.0; do
            run 3 $G $R
        done
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
echo "=== All experiments complete: $(date) ==="

# Generate CSV summary
$PYTHON << 'PYEOF'
import json, os, csv

base = os.environ.get("RESULTS_DIR", "results/paper/phase1c")
gammas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
resets = [0, 0.01, 0.1, 0.5, 1.0]

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

csv_path = os.path.join(base, "gamma_reset_sweep.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["gamma", "reset_prob", "best_f1", "final_f1", "final_em"])
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV saved: {csv_path}")

# Print table
print(f"\n{'':>12}", end="")
for r in resets:
    print(f"  reset={r:<6}", end="")
print()
print("-" * (12 + 12 * len(resets)))
for g in gammas:
    print(f"gamma={g:<5}", end="")
    for r in resets:
        match = [row for row in rows if row["gamma"] == g and row["reset_prob"] == r]
        if match and match[0]["best_f1"] > 0:
            print(f"  {match[0]['best_f1']:>8.2f}  ", end="")
        else:
            print(f"  {'---':>8}  ", end="")
    print()
PYEOF

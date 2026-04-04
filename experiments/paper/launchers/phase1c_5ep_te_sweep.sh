#!/bin/bash
# Phase 1C (5ep): Gamma=1 × transfer_every sweep
# Fixed: batch=16, gcc=3, uim=true, fast_lr=0.1, scale_transfer_lr=False,
#        transfer_lr=1.0, Fast=14bit, Slow=10bit, seed=42, ln_lr=0.003
# Variables: transfer_every = 3*61=183, 3*15=45, 3*4=12
# Total: 3 experiments, sequential
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1c_5ep_te_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=5
GPU="${GPU:-0}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

GAMMA=1.0
BATCH=16
GCC=3
BASE_TE=3

echo "=== Phase 1C (5ep): Gamma=${GAMMA} transfer_every sweep ==="
echo "Epochs: $EPOCHS | Batch: $BATCH | GCC: $GCC | GPU: $GPU"
echo "Start: $(date)"

run() {
    local TE_MULT=$1
    local TE=$((BASE_TE * TE_MULT))
    local TAG="g${GAMMA}_te${TE}"
    echo "[GPU $GPU] START $TAG (te=$TE) $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --gamma $GAMMA \
        --batch-size $BATCH \
        --grad-accum-steps $GCC \
        --units-in-mbatch true \
        --transfer-every $TE \
        --with-reset-prob 0 \
        --fast-lr 0.1 \
        --transfer-lr $GAMMA \
        --scale-transfer-lr false \
        --ln-lr 0.003 \
        --min-lr-rate 0.05 \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU $GPU] DONE  $TAG $(date)"
}

run 61   # te=183
run 15   # te=45
run 4    # te=12

echo ""
echo "=== All experiments complete: $(date) ==="

# CSV + Table
$PYTHON << 'PYEOF'
import json, os, csv

base = os.environ.get("RESULTS_DIR", "results/paper/phase1c_5ep_te_sweep")
gamma = 1.0
te_values = [183, 45, 12]

rows = []
for te in te_values:
    tag = f"g{gamma}_te{te}"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        rows.append({"gamma": gamma, "transfer_every": te,
                     "best_f1": d["best_f1"], "final_f1": d["final_f1"], "final_em": d["final_em"],
                     "f1_c_only": d.get("final_f1_c_only", 0), "em_c_only": d.get("final_em_c_only", 0)})
    except:
        rows.append({"gamma": gamma, "transfer_every": te,
                     "best_f1": 0, "final_f1": 0, "final_em": 0, "f1_c_only": 0, "em_c_only": 0})

csv_path = os.path.join(base, "gamma1_te_sweep_5ep.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["gamma", "transfer_every", "best_f1", "final_f1", "final_em", "f1_c_only", "em_c_only"])
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV saved: {csv_path}")

print(f"\n{'te':>6} | {'best_f1':>8} | {'final_f1':>8} | {'f1_c':>8} | {'delta':>7} | {'final_em':>8}")
print("-" * 58)
for r in rows:
    bf1 = f"{r['best_f1']:.2f}" if r['best_f1'] > 0 else "---"
    ff1 = f"{r['final_f1']:.2f}" if r['final_f1'] > 0 else "---"
    fc  = f"{r['f1_c_only']:.2f}" if r['f1_c_only'] > 0 else "---"
    dlt = f"{r['final_f1'] - r['f1_c_only']:+.2f}" if r['final_f1'] > 0 and r['f1_c_only'] > 0 else "---"
    fem = f"{r['final_em']:.2f}" if r['final_em'] > 0 else "---"
    print(f"{r['transfer_every']:>6} | {bf1:>8} | {ff1:>8} | {fc:>8} | {dlt:>7} | {fem:>8}")
PYEOF

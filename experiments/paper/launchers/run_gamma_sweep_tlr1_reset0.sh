#!/bin/bash
# Gamma sweep: transfer_lr=1.0 (fixed), reset=0 (fixed)
# Batch=16, grad_accum=3, transfer_every=3, fast_lr=0.1
# ConstantStep: fast=14bit, slow=10bit
# target-layers=attention (QKVO)
# analog_lr=0.016, classifier_lr=0.003, ln_lr=0.003
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/gamma_sweep_tlr1_reset0}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Gamma Sweep: transfer_lr=1.0, reset=0 ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

GAMMAS=(0.01 0.05 0.1 0.3 0.5 1.0)

for GAMMA in "${GAMMAS[@]}"; do
    TAG="g${GAMMA}_r0_tlr1"
    echo "[GPU 0] START $TAG (gamma=$GAMMA) $(date)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --epochs $EPOCHS \
        --batch-size 16 --grad-accum-steps 3 \
        --n-bits 14 --n-bits-slow 10 \
        --device-type constant_step --device-type-slow constant_step \
        --gamma $GAMMA \
        --units-in-mbatch true \
        --transfer-every 3 \
        --with-reset-prob 0 \
        --fast-lr 0.1 \
        --transfer-lr 1.0 \
        --scale-transfer-lr false \
        --analog-lr 0.016 \
        --classifier-lr 0.003 \
        --ln-lr 0.003 \
        --min-lr-rate 0.05 \
        --target-layers attention \
        --output-dir "$RESULTS_DIR/$TAG" \
        --log-every 20
    echo "[GPU 0] DONE  $TAG $(date)"
    echo ""
done

echo "=== All experiments complete: $(date) ==="

# Generate summary table
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/gamma_sweep_tlr1_reset0")
gammas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

print(f"\n{'Gamma':>8} {'Best F1':>10} {'Final F1':>10} {'Final EM':>10}")
print("-" * 42)
results = []
for g in gammas:
    tag = f"g{g}_r0_tlr1"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{g:>8} {d['best_f1']:>10.2f} {d['final_f1']:>10.2f} {d['final_em']:>10.2f}")
        results.append({"gamma": g, "best_f1": d["best_f1"], "final_f1": d["final_f1"], "final_em": d["final_em"]})
    except Exception as e:
        print(f"{g:>8} {'FAIL':>10} {'---':>10} {'---':>10}")

# Save summary JSON
summary_path = os.path.join(base, "gamma_sweep_summary.json")
json.dump(results, open(summary_path, "w"), indent=2)
print(f"\nSummary saved: {summary_path}")
PYEOF

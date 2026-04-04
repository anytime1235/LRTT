#!/bin/bash
# Phase 1C: gamma=0, transfer_lr=1, reset_prob={0, 1.0}
# Same settings as phase1c_4ep.sh except gamma=0 and transfer_lr=1
#   method=ttv1, seed=42, epochs=5, n_bits=14, n_bits_slow=10
#   batch=16, grad_accum=3, fast_lr=0.1, transfer_every=3, uim=true, scale_transfer_lr=false
#   ln_lr=0.003
# Sequential: reset_prob=0 first, then reset_prob=1.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1c_4ep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=5

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== gamma=0, transfer_lr=1: reset_prob sweep ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local GAMMA=$1 RESET=$2 TLR=$3
    local TAG="g${GAMMA}_r${RESET}"
    local OUT_DIR="$RESULTS_DIR/$TAG"

    if [ -f "$OUT_DIR/summary.json" ]; then
        echo "[SKIP] $TAG already completed"
        return 0
    fi

    echo ""
    echo "[START] $TAG (gamma=$GAMMA, reset=$RESET, transfer_lr=$TLR) $(date)"
    $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --batch-size 16 --grad-accum-steps 3 \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --gamma $GAMMA \
        --units-in-mbatch true \
        --transfer-every 3 \
        --with-reset-prob $RESET \
        --fast-lr 0.1 \
        --transfer-lr $TLR \
        --scale-transfer-lr false \
        --ln-lr 0.003 \
        --min-lr-rate 0.05 \
        --output-dir "$OUT_DIR" \
        --log-every 20
    echo "[DONE]  $TAG $(date)"
}

# Sequential: reset_prob=0, then reset_prob=1.0
run 0 0 1
run 0 1.0 1

echo ""
echo "=== gamma=0 experiments complete: $(date) ==="

# Print results
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/phase1c_4ep")
resets = [0, 1.0]

print(f"\n{'Gamma':>8} {'Reset':>8} {'TLR':>8} {'Best F1':>10} {'Final F1':>10} {'Final EM':>10}")
print("-" * 60)
for r in resets:
    tag = f"g0_r{r}"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{'0':>8} {r:>8} {'1':>8} {d['best_f1']:>10.2f} {d['final_f1']:>10.2f} {d['final_em']:>10.2f}")
    except:
        print(f"{'0':>8} {r:>8} {'1':>8} {'FAIL':>10} {'FAIL':>10} {'FAIL':>10}")
PYEOF

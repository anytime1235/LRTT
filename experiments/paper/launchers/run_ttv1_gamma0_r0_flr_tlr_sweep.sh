#!/bin/bash
# gamma=0, reset=0: fast_lr × transfer_lr sweep
# fast_lr:     [1.0, 0.1]
# transfer_lr: [0.01, 0.1]
# → 4 conditions total
#
# Base settings same as phase1c_4ep:
#   method=ttv1, seed=42, epochs=5, n_bits=14, n_bits_slow=10
#   batch=16, grad_accum=3, transfer_every=3, uim=true, scale_transfer_lr=false
#   ln_lr=0.003

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1c_4ep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=5

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== gamma=0, reset=0: fast_lr x transfer_lr sweep ==="
echo "Epochs: $EPOCHS | Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local GAMMA=$1 RESET=$2 TLR=$3 FLR=$4
    local TAG="g${GAMMA}_r${RESET}_flr${FLR}_tlr${TLR}"
    local OUT_DIR="$RESULTS_DIR/$TAG"

    if [ -f "$OUT_DIR/summary.json" ]; then
        echo "[SKIP] $TAG already completed"
        return 0
    fi

    echo ""
    echo "[START] $TAG (gamma=$GAMMA, reset=$RESET, fast_lr=$FLR, transfer_lr=$TLR) $(date)"
    $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --batch-size 16 --grad-accum-steps 3 \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --gamma $GAMMA \
        --units-in-mbatch true \
        --transfer-every 3 \
        --with-reset-prob $RESET \
        --fast-lr $FLR \
        --transfer-lr $TLR \
        --scale-transfer-lr false \
        --ln-lr 0.003 \
        --min-lr-rate 0.05 \
        --output-dir "$OUT_DIR" \
        --log-every 20
    echo "[DONE]  $TAG $(date)"
}

# 4 conditions: fast_lr x transfer_lr
run 0 0 0.01 1.0
run 0 0 0.1  1.0
run 0 0 0.01 0.1
run 0 0 0.1  0.1

echo ""
echo "=== gamma=0 r=0 fast_lr x transfer_lr sweep complete: $(date) ==="

# Print results
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/phase1c_4ep")
fast_lrs = [1.0, 0.1]
transfer_lrs = [0.01, 0.1]

print(f"\n{'Tag':>30} {'Best F1':>10} {'Final F1':>10} {'Final EM':>10}")
print("-" * 65)
for flr in fast_lrs:
    for tlr in transfer_lrs:
        tag = f"g0_r0_flr{flr}_tlr{tlr}"
        path = os.path.join(base, tag, "summary.json")
        try:
            d = json.load(open(path))["results"]
            print(f"{tag:>30} {d['best_f1']:>10.2f} {d['final_f1']:>10.2f} {d['final_em']:>10.2f}")
        except:
            print(f"{tag:>30} {'FAIL':>10} {'FAIL':>10} {'FAIL':>10}")
PYEOF

#!/bin/bash
# gamma=0, reset=0, fast_lr=0.1, epochs=4: transfer_lr sweep
# transfer_lr: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
# → 6 conditions (~4h each ≈ 24h total)
#
# Base settings:
#   method=ttv1, seed=42, n_bits=14, n_bits_slow=10
#   batch=16, grad_accum=3, transfer_every=3, uim=true, scale_transfer_lr=false
#   ln_lr=0.003, fast_lr=0.1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/g0r0_tlr_sweep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"
EPOCHS=4

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== gamma=0, reset=0, fast_lr=0.1: transfer_lr sweep (${EPOCHS}ep) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

TRANSFER_LRS=(0.0001 0.0005 0.001 0.005 0.01 0.1)
TOTAL=${#TRANSFER_LRS[@]}
COUNT=0

run() {
    local TLR=$1
    local TAG="tlr${TLR}"
    local OUT_DIR="$RESULTS_DIR/$TAG"

    COUNT=$((COUNT + 1))

    if [ -f "$OUT_DIR/summary.json" ]; then
        echo "[$COUNT/$TOTAL] [SKIP] $TAG already completed"
        return 0
    fi

    echo ""
    echo "[$COUNT/$TOTAL] [START] $TAG (transfer_lr=$TLR) $(date)"
    $PYTHON paper_experiment.py \
        --mode fixed --method ttv1 --seed 42 \
        --batch-size 16 --grad-accum-steps 3 \
        --epochs $EPOCHS --n-bits 14 --n-bits-slow 10 \
        --gamma 0 \
        --units-in-mbatch true \
        --transfer-every 3 \
        --with-reset-prob 0 \
        --fast-lr 0.1 \
        --transfer-lr $TLR \
        --scale-transfer-lr false \
        --ln-lr 0.003 \
        --min-lr-rate 0.05 \
        --output-dir "$OUT_DIR" \
        --log-every 20
    echo "[$COUNT/$TOTAL] [DONE]  $TAG $(date)"
}

for TLR in "${TRANSFER_LRS[@]}"; do
    run $TLR
done

echo ""
echo "=== Sweep complete: $(date) ==="

# Print results table
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/g0r0_tlr_sweep")
transfer_lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

print(f"\n{'transfer_lr':>12} {'Best F1':>10} {'Final F1':>10} {'Final EM':>10}")
print("-" * 48)
for tlr in transfer_lrs:
    tag = f"tlr{tlr}"
    path = os.path.join(base, tag, "summary.json")
    try:
        d = json.load(open(path))["results"]
        print(f"{tlr:>12} {d['best_f1']:>10.2f} {d['final_f1']:>10.2f} {d['final_em']:>10.2f}")
    except:
        print(f"{tlr:>12} {'---':>10} {'---':>10} {'---':>10}")
PYEOF

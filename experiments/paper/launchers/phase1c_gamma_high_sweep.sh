#!/bin/bash
# Phase 1C Extended: High Gamma Sweep (gamma=3.0, 5.0, 10.0)
# Identical settings to gamma=1.0 experiment (g1.0_r1.0 config.json)
#
# Fixed config (from g1.0_r1.0):
#   method=ttv1, target=attention, batch=16, grad_accum=3, epochs=4
#   n_bits=14, n_bits_slow=10, fast_lr=0.1, transfer_every=3
#   units_in_mbatch=true, scale_transfer_lr=false
#   analog_lr=0.016, classifier_lr=0.003, ln_lr=0.003
#   warmup_ratio=0.05, min_lr_rate=0.05, seed=42, IO=perfect
#   transfer_lr = gamma (same as gamma value)
#
# Variables: gamma={3.0, 5.0, 10.0} x reset_prob={0, 1.0}
# Total: 6 experiments (sequential, single GPU)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1c_4ep}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 1C Extended: High Gamma Sweep (3.0, 5.0, 10.0) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

run() {
    local GAMMA=$1 RESET=$2
    local TAG="g${GAMMA}_r${RESET}"
    local OUT_DIR="$RESULTS_DIR/$TAG"

    if [ -f "$OUT_DIR/summary.json" ]; then
        echo "[SKIP] $TAG already completed"
        return 0
    fi

    echo ""
    echo "[START] $TAG (gamma=$GAMMA, reset=$RESET, transfer_lr=$GAMMA) $(date)"
    $PYTHON paper_experiment.py \
        --mode fixed \
        --method ttv1 \
        --seed 42 \
        --target-layers attention \
        --batch-size 16 \
        --grad-accum-steps 3 \
        --epochs 4 \
        --n-bits 14 \
        --n-bits-slow 10 \
        --gamma $GAMMA \
        --units-in-mbatch true \
        --transfer-every 3 \
        --fast-lr 0.1 \
        --transfer-lr $GAMMA \
        --scale-transfer-lr false \
        --with-reset-prob $RESET \
        --analog-lr 0.016 \
        --classifier-lr 0.003 \
        --ln-lr 0.003 \
        --warmup-ratio 0.05 \
        --min-lr-rate 0.05 \
        --io-bits 0 \
        --noise-management abs_max \
        --output-dir "$OUT_DIR" \
        --log-every 20
    echo "[DONE]  $TAG $(date)"
}

# Run all 6 experiments sequentially
for G in 3.0 5.0 10.0; do
    for R in 0 1.0; do
        run $G $R
    done
done

echo ""
echo "=== All high-gamma experiments complete: $(date) ==="

# Append results to summary
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/phase1c_4ep")
gammas = [3.0, 5.0, 10.0]
resets = [0, 1.0]

print(f"\n{'Gamma':>8} {'Reset':>8} {'Best F1':>10} {'Final F1':>10} {'Final EM':>10}")
print("-" * 50)
for g in gammas:
    for r in resets:
        tag = f"g{g}_r{r}"
        path = os.path.join(base, tag, "summary.json")
        try:
            d = json.load(open(path))["results"]
            print(f"{g:>8} {r:>8} {d['best_f1']:>10.2f} {d['final_f1']:>10.2f} {d['final_em']:>10.2f}")
        except:
            print(f"{g:>8} {r:>8} {'FAIL':>10} {'FAIL':>10} {'FAIL':>10}")
PYEOF

#!/bin/bash
# Phase 1C Extended: High Gamma Sweep with 6T1C LinearStep Device (full noise)
# Same gamma/reset grid as noise-free sweep, but with 6T1C measured parameters:
#   gamma_up = -0.1678, gamma_down = 0.1410, noise_ratio = 1.0
#   (dw_min_std=0.3, dw_min_dtod=0.1, up_down_dtod=0.01, etc.)
#
# Variables: gamma={3.0, 5.0, 10.0} x reset_prob={0, 1.0}
# Total: 6 experiments (sequential, single GPU)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/phase1c_4ep_6t1c}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 1C Extended: High Gamma Sweep — 6T1C LinearStep (noise_ratio=1.0) ==="
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
    echo "[START] $TAG (gamma=$GAMMA, reset=$RESET, transfer_lr=$GAMMA, 6T1C noise) $(date)"
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
        --device-type linear_step \
        --ls-gamma-up-ratio 1.0 \
        --ls-gamma-down-ratio 1.0 \
        --ls-noise-ratio 1.0 \
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
echo "=== All 6T1C high-gamma experiments complete: $(date) ==="

# Summary table
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/phase1c_4ep_6t1c")
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

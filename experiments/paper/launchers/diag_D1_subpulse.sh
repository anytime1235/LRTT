#!/bin/bash
# Diagnostic D1: Sub-pulse mapping — 12 runs × 128 steps × seed 42
# Primary outputs: frac_mu_lt_1, mu_p50/p90, bl_utilization, VRC_1/VRR_1
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/diag_D1_subpulse}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Diagnostic D1: Sub-pulse Mapping (128 steps each) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

COMMON_FLAGS="--mode fixed --seed 42 --epochs 1 --max-steps 128 --batch-size 48 \
  --diag-carry-path --diag-update-exact --diag-steps 128 --log-every 1"

# GPU 1: single_rpu stochastic bits={8,10,12,14}
run_gpu1() {
    for BITS in 8 10 12 14; do
        local TAG="single_rpu_stoch_${BITS}b"
        echo "[GPU 1] START $TAG $(date)"
        CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
            $COMMON_FLAGS \
            --method single_rpu --pulse-type stochastic --n-bits $BITS \
            --output-dir "$RESULTS_DIR/$TAG"
        echo "[GPU 1] DONE  $TAG $(date)"
    done
}

# GPU 2: single_rpu deterministic bits={8,10} + eco_ref bits={8,10}
run_gpu2() {
    for BITS in 8 10; do
        local TAG="single_rpu_det_${BITS}b"
        echo "[GPU 2] START $TAG $(date)"
        CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
            $COMMON_FLAGS \
            --method single_rpu --pulse-type deterministic --n-bits $BITS \
            --output-dir "$RESULTS_DIR/$TAG"
        echo "[GPU 2] DONE  $TAG $(date)"
    done
    for BITS in 8 10; do
        local TAG="eco_ref_${BITS}b"
        echo "[GPU 2] START $TAG $(date)"
        CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
            $COMMON_FLAGS \
            --method eco_ref --n-bits $BITS --eco-rounding stochastic \
            --output-dir "$RESULTS_DIR/$TAG"
        echo "[GPU 2] DONE  $TAG $(date)"
    done
}

# GPU 3: mixed_precision bits={8,10,12,14}
run_gpu3() {
    for BITS in 8 10 12 14; do
        local TAG="mixed_precision_${BITS}b"
        echo "[GPU 3] START $TAG $(date)"
        CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
            $COMMON_FLAGS \
            --method mixed_precision --n-bits $BITS \
            --output-dir "$RESULTS_DIR/$TAG"
        echo "[GPU 3] DONE  $TAG $(date)"
    done
}

run_gpu1 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
run_gpu2 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!
run_gpu3 2>&1 | tee "$RESULTS_DIR/gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3
echo ""
echo "=== D1 complete: $(date) ==="

# Summary table
$PYTHON << 'PYEOF'
import json, os

base = os.environ.get("RESULTS_DIR", "results/paper/diag_D1_subpulse")
tags = []
for method in ["single_rpu_stoch", "single_rpu_det", "mixed_precision", "eco_ref"]:
    for bits in [8, 10, 12, 14]:
        tag = f"{method}_{bits}b"
        if os.path.isdir(os.path.join(base, tag)):
            tags.append(tag)

print(f"\n{'Tag':<35} | {'mu_p50':>8} | {'mu_p90':>8} | {'frac<1':>8} | {'BL_util':>8}")
print("-" * 80)
for tag in tags:
    spath = os.path.join(base, tag, "carry_path_summary.json")
    try:
        d = json.load(open(spath))
        pt = d.get("per_tile", {})
        if pt:
            first = list(pt.values())[0]
            print(f"{tag:<35} | {first.get('mean_mu_p50', 0):>8.4f} | "
                  f"{first.get('mean_mu_p90', 0):>8.4f} | "
                  f"{first.get('mean_frac_mu_lt_1', 0):>8.4f} | "
                  f"{first.get('mean_bl_utilization', 0) if 'mean_bl_utilization' in first else 0:>8.4f}")
        else:
            print(f"{tag:<35} | {'N/A':>8}")
    except Exception as e:
        print(f"{tag:<35} | {'FAIL':>8} | {str(e)[:30]}")
PYEOF

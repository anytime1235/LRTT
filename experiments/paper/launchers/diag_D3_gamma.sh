#!/bin/bash
# Diagnostic D3: Gamma / continuity sweep — 12 runs × 1024 steps × seed 42
# Fixed topology: units_in_mbatch=true, transfer_every=1, n_reads=1, reset=1.0
# Gamma sweep: {0.0, 0.1, 0.3, 0.5, 1.0} × slow_bits={8,10}
# Negative control: gamma=0.0, reset=0.0, slow_bits={8,10}
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/diag_D3_gamma}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Diagnostic D3: Gamma Sweep (1024 steps each) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

COMMON_FLAGS="--mode fixed --seed 42 --epochs 1 --max-steps 1024 --batch-size 48 \
  --diag-carry-path --diag-steps 128 \
  --diag-checkpoint-steps 256,512,768,1024 \
  --diag-vrc-windows 1,16,64,256,768 \
  --log-every 20"

TTv1_COMMON="--method ttv1 --ttv1-mode residual_lane \
  --units-in-mbatch true --transfer-every 1 --n-reads-per-transfer 1 \
  --n-bits 14 --with-reset-prob 1.0 --transfer-lr 1.0"

# GPU 1: gamma={0.0, 0.1, 0.3} × slow_bits=8
run_gpu1() {
    for GAMMA in 0.0 0.1 0.3; do
        local TAG="g${GAMMA}_s8b"
        echo "[GPU 1] START $TAG $(date)"
        CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
            $COMMON_FLAGS $TTv1_COMMON \
            --gamma $GAMMA --n-bits-slow 8 \
            --output-dir "$RESULTS_DIR/$TAG"
        echo "[GPU 1] DONE  $TAG $(date)"
    done
    # Negative control: gamma=0.0, reset=0.0, slow=8
    local TAG="g0.0_noreset_s8b"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode residual_lane \
        --units-in-mbatch true --transfer-every 1 --n-reads-per-transfer 1 \
        --n-bits 14 --with-reset-prob 0.0 --transfer-lr 1.0 \
        --gamma 0.0 --n-bits-slow 8 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 1] DONE  $TAG $(date)"
}

# GPU 2: gamma={0.0, 0.1, 0.3} × slow_bits=10
run_gpu2() {
    for GAMMA in 0.0 0.1 0.3; do
        local TAG="g${GAMMA}_s10b"
        echo "[GPU 2] START $TAG $(date)"
        CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
            $COMMON_FLAGS $TTv1_COMMON \
            --gamma $GAMMA --n-bits-slow 10 \
            --output-dir "$RESULTS_DIR/$TAG"
        echo "[GPU 2] DONE  $TAG $(date)"
    done
    # Negative control: gamma=0.0, reset=0.0, slow=10
    local TAG="g0.0_noreset_s10b"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode residual_lane \
        --units-in-mbatch true --transfer-every 1 --n-reads-per-transfer 1 \
        --n-bits 14 --with-reset-prob 0.0 --transfer-lr 1.0 \
        --gamma 0.0 --n-bits-slow 10 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 2] DONE  $TAG $(date)"
}

# GPU 3: gamma={0.5, 1.0} × slow_bits={8,10}
run_gpu3() {
    for GAMMA in 0.5 1.0; do
        for BITS in 8 10; do
            local TAG="g${GAMMA}_s${BITS}b"
            echo "[GPU 3] START $TAG $(date)"
            CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
                $COMMON_FLAGS $TTv1_COMMON \
                --gamma $GAMMA --n-bits-slow $BITS \
                --output-dir "$RESULTS_DIR/$TAG"
            echo "[GPU 3] DONE  $TAG $(date)"
        done
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
echo "=== D3 complete: $(date) ==="

# Summary: gamma diagnostics
$PYTHON << 'PYEOF'
import json, os, glob

base = os.environ.get("RESULTS_DIR", "results/paper/diag_D3_gamma")
dirs = sorted(glob.glob(os.path.join(base, "*/carry_path_summary.json")))

print(f"\n{'Tag':<25} | {'gamma':>6} | {'fast_sat':>8} | {'PMR':>8} | {'VRC_eff_256':>11} | {'VRC_slow_256':>12} | {'G_gamma_256':>11}")
print("-" * 100)
for sp in dirs:
    tag = os.path.basename(os.path.dirname(sp))
    try:
        d = json.load(open(sp))
        gd = d.get("ttv1_gamma_diag", {})
        w = d.get("windows", {}).get("256", {})
        print(f"{tag:<25} | {gd.get('gamma', 0):>6.2f} | "
              f"{gd.get('mean_fast_sat_ratio', 0):>8.4f} | "
              f"{gd.get('mean_pmr', 0):>8.4f} | "
              f"{w.get('mean_VRC_eff_K', 0):>11.4f} | "
              f"{w.get('mean_VRC_slow_K', 0):>12.4f} | "
              f"{w.get('mean_G_gamma_K', 0):>11.4f}")
    except Exception as e:
        print(f"{tag:<25} | {'FAIL':>6} | {str(e)[:40]}")
PYEOF

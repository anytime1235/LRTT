#!/bin/bash
# Diagnostic D2: Carry-path comparison — 14 runs × 1024 steps × seed 42
# Diagnostic steps for first 128, then checkpoints at 256,512,768,1024
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/diag_D2_carrypath}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Diagnostic D2: Carry-path Comparison (1024 steps each) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

COMMON_FLAGS="--mode fixed --seed 42 --epochs 1 --max-steps 1024 --batch-size 48 \
  --diag-carry-path --diag-steps 128 \
  --diag-checkpoint-steps 256,512,768,1024 \
  --diag-vrc-windows 1,16,64,256,768 \
  --diag-layer-set 0,5,11 \
  --log-every 20"

TTv1_COMMON="--units-in-mbatch true --transfer-every 1 --n-reads-per-transfer 1 --n-bits 14"

# GPU 1: single_rpu (stoch 8b, stoch 10b) + mixed_prec (8b, 10b) + ttv1 hb s8
run_gpu1() {
    local TAG="single_rpu_stoch_8b"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method single_rpu --pulse-type stochastic --n-bits 8 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 1] DONE  $TAG $(date)"

    TAG="single_rpu_stoch_10b"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method single_rpu --pulse-type stochastic --n-bits 10 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 1] DONE  $TAG $(date)"

    TAG="mixed_precision_8b"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method mixed_precision --n-bits 8 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 1] DONE  $TAG $(date)"

    TAG="mixed_precision_10b"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method mixed_precision --n-bits 10 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 1] DONE  $TAG $(date)"

    TAG="ttv1_hb_s8"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode hidden_buffer \
        $TTv1_COMMON --n-bits-slow 8 --with-reset-prob 1.0 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 1] DONE  $TAG $(date)"
}

# GPU 2: eco_ref (8b, 10b) + ttv1 hb s10 + ttv1 rl s8 + ttv1 rl s10
run_gpu2() {
    local TAG="eco_ref_8b"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method eco_ref --n-bits 8 --eco-rounding stochastic \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 2] DONE  $TAG $(date)"

    TAG="eco_ref_10b"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method eco_ref --n-bits 10 --eco-rounding stochastic \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 2] DONE  $TAG $(date)"

    TAG="ttv1_hb_s10"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode hidden_buffer \
        $TTv1_COMMON --n-bits-slow 10 --with-reset-prob 1.0 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 2] DONE  $TAG $(date)"

    TAG="ttv1_rl_s8"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode residual_lane \
        $TTv1_COMMON --n-bits-slow 8 --gamma 1.0 --with-reset-prob 1.0 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 2] DONE  $TAG $(date)"

    TAG="ttv1_rl_s10"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode residual_lane \
        $TTv1_COMMON --n-bits-slow 10 --gamma 1.0 --with-reset-prob 1.0 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 2] DONE  $TAG $(date)"
}

# GPU 3: ttv1 residual_lane_noreset (s8, s10) + 2 residual_lane additional
run_gpu3() {
    local TAG="ttv1_rl_noreset_s8"
    echo "[GPU 3] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode residual_lane_noreset \
        $TTv1_COMMON --n-bits-slow 8 --gamma 1.0 --with-reset-prob 0.0 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 3] DONE  $TAG $(date)"

    TAG="ttv1_rl_noreset_s10"
    echo "[GPU 3] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode residual_lane_noreset \
        $TTv1_COMMON --n-bits-slow 10 --gamma 1.0 --with-reset-prob 0.0 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 3] DONE  $TAG $(date)"
}

run_gpu1 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
run_gpu2 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!
run_gpu3 2>&1 | tee "$RESULTS_DIR/gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3
echo ""
echo "=== D2 complete: $(date) ==="

# Summary
$PYTHON << 'PYEOF'
import json, os, glob

base = os.environ.get("RESULTS_DIR", "results/paper/diag_D2_carrypath")
dirs = sorted(glob.glob(os.path.join(base, "*/carry_path_summary.json")))

print(f"\n{'Tag':<30} | {'mean_cos':>8} | {'mean_res':>8} | {'VRC_256':>8} | {'VRC_768':>8}")
print("-" * 80)
for sp in dirs:
    tag = os.path.basename(os.path.dirname(sp))
    try:
        d = json.load(open(sp))
        agg = d.get("aggregate", {})
        w = d.get("windows", {})
        vrc256 = w.get("256", {}).get("mean_VRC_K", 0)
        vrc768 = w.get("768", {}).get("mean_VRC_K", 0)
        print(f"{tag:<30} | {agg.get('mean_cosine_sim', 0):>8.4f} | "
              f"{agg.get('mean_residual_ratio', 0):>8.4f} | "
              f"{vrc256:>8.4f} | {vrc768:>8.4f}")
    except Exception as e:
        print(f"{tag:<30} | {'FAIL':>8} | {str(e)[:30]}")
PYEOF

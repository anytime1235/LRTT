#!/bin/bash
# D1+D2 w_max_fast sweep: test if reducing fast tile w_max fixes sub-pulse
#
# D1 (128 steps): Baseline weight distribution + sub-pulse metrics
#   - single_rpu 14b baseline (omega=1, w_max=1) → measure actual weight ranges
#   - single_rpu 14b with omega=0, w_max={0.5,0.3,0.1} → verify clipping
#
# D2 (1024 steps): TTv1 carry-path with w_max_fast sweep (slow=10b fixed)
#   - TTv1 w_max_fast={1.0, 0.1, 0.05, 0.01} → test carry-path improvement
#
set -uo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

run_one() {
    local PHASE="$1"
    local TAG="$2"
    shift 2
    echo ""
    echo "[$PHASE] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py "$@"
    local RC=$?
    $PYTHON -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache()" 2>/dev/null
    if [ $RC -ne 0 ]; then
        echo "[$PHASE] FAIL  $TAG (exit=$RC) $(date)"
    else
        echo "[$PHASE] DONE  $TAG $(date)"
    fi
    sleep 3
}

COMMON="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016"
DIAG="--diag-update-exact --diag-layer-set 0,5,11"

RESULTS="results/paper/diag_wmax_sweep"
mkdir -p "$RESULTS"

echo "============================================================"
echo "  D1: Weight distribution + sub-pulse (128 steps)"
echo "  Start: $(date)"
echo "============================================================"

D1_COMMON="$COMMON --max-steps 128 --batch-size 48 $DIAG --log-every 1"

# D1-a: Baseline single_rpu 14b (omega=1.0, w_max=1.0) — reference
run_one D1 "single_rpu_14b_baseline" $D1_COMMON \
    --method single_rpu --pulse-type stochastic --n-bits 14 \
    --output-dir "$RESULTS/D1_single_rpu_14b_baseline"

# D1-b: single_rpu 14b with omega=0, w_max sweep — check clipping
for WMAX in 0.5 0.3 0.1; do
    run_one D1 "single_rpu_14b_omega0_wmax${WMAX}" $D1_COMMON \
        --method single_rpu --pulse-type stochastic --n-bits 14 \
        --w-max $WMAX --omega 0.0 \
        --output-dir "$RESULTS/D1_single_rpu_14b_omega0_wmax${WMAX}"
done

echo ""
echo "============================================================"
echo "  D2: TTv1 carry-path w_max_fast sweep (1024 steps, slow=10b)"
echo "  Start: $(date)"
echo "============================================================"

D2_DIAG="--diag-at-steps 1,16,64,128,256,384,512,640,768,896,1024 --diag-vrc-windows 1,16,64,256,512,1024"
D2_COMMON="$COMMON --max-steps 1024 --batch-size 12 --grad-accum-steps 4 $DIAG $D2_DIAG --diag-carry-path --diag-update-exact"

TTv1_BASE="--method ttv1 --ttv1-mode residual_lane --n-bits 14 --n-bits-slow 10 --gamma 1.0 --with-reset-prob 1.0 --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 4"

# D2-a: TTv1 baseline (w_max_fast=default, i.e., 1.0)
run_one D2 "ttv1_wmax_fast_1.0" $D2_COMMON $TTv1_BASE \
    --output-dir "$RESULTS/D2_ttv1_s10_wmf1.0"

# D2-b: TTv1 with reduced w_max_fast
for WMF in 0.1 0.05 0.01; do
    run_one D2 "ttv1_wmax_fast_${WMF}" $D2_COMMON $TTv1_BASE \
        --w-max-fast $WMF \
        --output-dir "$RESULTS/D2_ttv1_s10_wmf${WMF}"
done

echo ""
echo "============================================================"
echo "  ALL COMPLETE: $(date)"
echo "============================================================"

# Summary
$PYTHON << 'PYEOF'
import json, os, glob

base = "results/paper/diag_wmax_sweep"

print("\n=== D1 Weight Distribution Summary ===")
for d in sorted(glob.glob(f"{base}/D1_*")):
    tag = os.path.basename(d)
    summary_path = f"{d}/update_diagnostics_summary.json"
    if not os.path.exists(summary_path):
        print(f"  {tag}: NO SUMMARY")
        continue
    data = json.load(open(summary_path))
    tiles = list(data.values())
    if not tiles:
        continue
    avg = lambda k: sum(t.get(k, 0) for t in tiles) / len(tiles)
    print(f"  {tag}:")
    print(f"    w_abs_max={avg('max_w_abs_max'):.4f}  w_util={avg('mean_w_utilization'):.4f}  "
          f"clip%={avg('max_w_clipped_frac')*100:.2f}%  "
          f"cos_sim={avg('mean_cosine_sim'):.4f}  frac_mu<1={avg('mean_eff_frac_mu_lt_1'):.4f}")

print("\n=== D2 Carry-Path Summary ===")
for d in sorted(glob.glob(f"{base}/D2_*")):
    tag = os.path.basename(d)
    cp_path = f"{d}/carry_path_summary.json"
    ud_path = f"{d}/update_diagnostics_summary.json"
    if not os.path.exists(cp_path):
        print(f"  {tag}: NO CARRY-PATH SUMMARY")
        continue
    cp = json.load(open(cp_path))
    windows = cp.get("windows", {})
    tt = cp.get("ttv1_transfer", {})
    agg = cp.get("aggregate", {})

    print(f"  {tag}:")
    for wk in ["1", "64", "256", "1024"]:
        w = windows.get(wk, {})
        print(f"    VRC_K={wk}: {w.get('mean_VRC_K', 0):.4f}")
    print(f"    E2ECos={tt.get('mean_EndToEndCos', 0):.4f}  "
          f"HandoffCos={tt.get('mean_HandoffCos', 0):.4f}  "
          f"cos_sim={agg.get('mean_cosine_sim', 0):.4f}")

PYEOF

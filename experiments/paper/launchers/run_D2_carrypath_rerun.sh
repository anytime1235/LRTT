#!/bin/bash
# Re-run D2 experiments with FIXED carry path VRC (W_eff based)
# Only carry-path diagnostics, no update_diagnostics (faster)
set -euo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

COMMON="--mode fixed --seed 42 --epochs 1 --max-steps 1024 \
  --batch-size 12 --grad-accum-steps 4 \
  --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.016 \
  --warmup-ratio 0 --min-lr-rate 1.0 \
  --target-layers attention --log-every 64"

DIAG="--diag-carry-path \
  --diag-at-steps 1,2,4,8,16,32,64,128,256,384,512,768,896,1024 \
  --diag-vrc-windows 1,16,64,256 --diag-layer-set 0,5,11"

RESULTS="results/paper/diag_D2_vrc_fixed"

echo "============================================================"
echo "  D2 Carry Path Rerun (VRC fixed: W_eff based)"
echo "  Start: $(date)"
echo "============================================================"

# 1) TTv1 14b/10b
echo ""
echo "[VRC] START ttv1_14b $(date)"
$PYTHON paper_experiment.py $COMMON $DIAG \
    --method ttv1 --ttv1-mode residual_lane \
    --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --with-reset-prob 1.0 \
    --fast-lr 0.1 --transfer-lr 1.0 \
    --units-in-mbatch true --transfer-every 4 \
    --output-dir "$RESULTS/ttv1_14b"
echo "[VRC] DONE ttv1_14b $(date)"

# 2) SingleRPU 10b (for fair comparison)
echo ""
echo "[VRC] START single_rpu_10b $(date)"
$PYTHON paper_experiment.py $COMMON $DIAG \
    --method single_rpu --n-bits 10 \
    --output-dir "$RESULTS/single_rpu_10b"
echo "[VRC] DONE single_rpu_10b $(date)"

# 3) SingleRPU 14b (same dw_min as TTv1 fast — isolate carry path effect)
echo ""
echo "[VRC] START single_rpu_14b $(date)"
$PYTHON paper_experiment.py $COMMON $DIAG \
    --method single_rpu --n-bits 14 \
    --output-dir "$RESULTS/single_rpu_14b"
echo "[VRC] DONE single_rpu_14b $(date)"

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"

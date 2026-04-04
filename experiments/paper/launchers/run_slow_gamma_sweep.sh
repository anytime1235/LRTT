#!/bin/bash
# Slow tile gamma sweep: TTv1 14b fast (ConstantStep) / 10b slow (LinearStep)
# Gamma_up = Gamma_down = {0.5, 1.0, 2.0, 5.0, 10.0}
# Noise-free, attention only (QKVO)
# Same hyperparams as D2 ttv1_14b baseline

set -euo pipefail

PYTHON="${PYTHON:-/root/.venv310/bin/python}"
cd /root/LRTT/experiments/paper

COMMON="--mode fixed --seed 42 --epochs 1 --max-steps 1024 \
  --batch-size 12 --grad-accum-steps 4 \
  --analog-lr 0.016 --classifier-lr 0.003 --ln-lr 0.016 \
  --warmup-ratio 0 --min-lr-rate 1.0 \
  --target-layers attention --log-every 64"

TTv1="--method ttv1 --ttv1-mode residual_lane \
  --n-bits 14 --n-bits-slow 10 \
  --gamma 1.0 --with-reset-prob 1.0 \
  --fast-lr 0.1 --transfer-lr 1.0 \
  --units-in-mbatch true --transfer-every 4"

DIAG="--diag-update-exact --diag-carry-path \
  --diag-at-steps 1,2,4,8,16,32,64,128,256,384,512,768,896,1024 \
  --diag-vrc-windows 1,16,64,256 --diag-layer-set 0,5,11"

RESULTS="results/paper/slow_gamma_sweep"

echo "============================================================"
echo "  Slow Tile Gamma Sweep (LinearStepDevice)"
echo "  Fast: ConstantStep 14b, Slow: LinearStep 10b"
echo "  Gamma: 0.5, 1.0, 2.0, 5.0, 10.0"
echo "  Start: $(date)"
echo "============================================================"

for GAMMA in 0.5 1.0 2.0 5.0 10.0; do
    OUTDIR="${RESULTS}/slow_gamma_${GAMMA}"
    if [ -f "${OUTDIR}/summary.json" ]; then
        echo "[SKIP] slow_gamma_${GAMMA} — already completed"
        continue
    fi

    echo ""
    echo "[SLOW_GAMMA] START gamma=${GAMMA} $(date)"
    $PYTHON paper_experiment.py $COMMON $TTv1 $DIAG \
        --device-type constant_step \
        --device-type-slow linear_step \
        --ls-gamma-up-slow ${GAMMA} \
        --ls-gamma-down-slow ${GAMMA} \
        --ls-noise-ratio-slow 0.0 \
        --output-dir "$OUTDIR"
    echo "[SLOW_GAMMA] DONE gamma=${GAMMA} $(date)"
done

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"

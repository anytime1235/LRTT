#!/bin/bash
# SALPA constrained allocation: ideal device, all layers, avg 4.0/4.5b
# FFN1 protected at b_min=6, O/FFN2 allowed down to 3b
set -e
cd "$(dirname "$0")"

PM_DIR="results/paper/salpa_lowbit"
OUT_BASE="results/paper/salpa_lowbit"
COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 4 --seed 42 --mode fixed \
  --min-lr-rate 0.05"

echo "============================================"
echo "SALPA lowbit: ideal device, all layers"
echo "  FFN1>=6b, Q/K/V>=4b, O/FFN2>=3b"
echo "============================================"

for BUDGET in 4.0 4.5; do
  JSON="${PM_DIR}/precision_map_minimax_avg${BUDGET}.json"
  DIR="${OUT_BASE}/salpa_minimax_avg${BUDGET}"

  if [ ! -f "$JSON" ]; then
    echo "[$(date +%H:%M)] SKIP avg ${BUDGET}b — no precision map"
    continue
  fi

  echo "[$(date +%H:%M)] SALPA minimax avg ${BUDGET}b → $DIR"
  mkdir -p "$DIR"
  python paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$JSON" \
    --output-dir "$DIR" \
    2>&1 | tee "${OUT_BASE}/salpa_minimax_avg${BUDGET}.log"
  echo "[$(date +%H:%M)] Done: SALPA minimax avg ${BUDGET}b"
done

echo "============================================"
echo "[$(date +%H:%M)] All runs completed"
echo "============================================"

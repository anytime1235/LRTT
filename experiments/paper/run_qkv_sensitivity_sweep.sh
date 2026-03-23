#!/bin/bash
# QKV sensitivity sweep: base=8b, QKV→{4,5,6,7}b, 1 epoch each
set -e
source ~/.venv310/bin/activate
cd /root/LRTT/experiments/paper

OUT_BASE="/root/results/sa_v4_training_sensitivity"
COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 1 --seed 42 --mode fixed \
  --min-lr-rate 0.05"

mkdir -p "$OUT_BASE"

echo "============================================"
echo "QKV Sensitivity Sweep: base=8b, QKV→{4,5,6,7}b, 1ep"
echo "============================================"

for BITS_QKV in 4 5 6 7; do
  BITS="Q=${BITS_QKV},K=${BITS_QKV},V=${BITS_QKV},O=8,FFN1=8,FFN2=8"
  DIR="${OUT_BASE}/sens_QKV_${BITS_QKV}b_1ep"
  echo "[$(date +%H:%M)] QKV=${BITS_QKV}b, rest=8b → $DIR"
  mkdir -p "$DIR"
  python paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$BITS" \
    --output-dir "$DIR" \
    2>&1 | tee "${OUT_BASE}/sens_QKV_${BITS_QKV}b_1ep.log"
  echo "[$(date +%H:%M)] Done: QKV=${BITS_QKV}b"
  echo ""
done

echo "============================================"
echo "[$(date +%H:%M)] QKV sensitivity sweep completed"
echo "============================================"

echo ""
echo "=== SUMMARY ==="
for BITS_QKV in 4 5 6 7; do
  LOG="${OUT_BASE}/sens_QKV_${BITS_QKV}b_1ep.log"
  F1=$(grep "Epoch 1 F1:" "$LOG" 2>/dev/null | tail -1)
  echo "  QKV=${BITS_QKV}b: $F1"
done

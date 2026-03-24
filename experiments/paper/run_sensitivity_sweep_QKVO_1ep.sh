#!/bin/bash
# Sensitivity sweep (1 epoch): QKVO together (4,5,6,7), FFN1=8, FFN2=8 fixed
# Also re-run FFN1(4b) and FFN2(4b) at 1ep.
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
echo "Sensitivity Sweep (1ep): QKVO(4,5,6,7), FFN1(4), FFN2(4)"
echo "Started: $(date)"
echo "============================================"

# QKVO together: 4b, 5b, 6b, 7b
for BITS_VAL in 4 5 6 7; do
  BITS="Q=${BITS_VAL},K=${BITS_VAL},V=${BITS_VAL},O=${BITS_VAL},FFN1=8,FFN2=8"
  DIR="${OUT_BASE}/sens_QKVO_${BITS_VAL}b_1ep"
  echo "[$(date +%H:%M)] QKVO=${BITS_VAL}b, FFN1=8, FFN2=8 → $DIR"
  mkdir -p "$DIR"
  python paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$BITS" \
    --output-dir "$DIR" \
    2>&1 | tee "${OUT_BASE}/sens_QKVO_${BITS_VAL}b_1ep.log"
  echo "[$(date +%H:%M)] Done: QKVO=${BITS_VAL}b"
  echo ""
done

# FFN1: 4b (re-run at 1ep)
BITS="Q=8,K=8,V=8,O=8,FFN1=4,FFN2=8"
DIR="${OUT_BASE}/sens_FFN1_4b_1ep"
echo "[$(date +%H:%M)] FFN1=4b, rest=8b → $DIR"
mkdir -p "$DIR"
python paper_experiment.py $COMMON \
  --io-bits 8 \
  --per-layer-bits "$BITS" \
  --output-dir "$DIR" \
  2>&1 | tee "${OUT_BASE}/sens_FFN1_4b_1ep.log"
echo "[$(date +%H:%M)] Done: FFN1=4b"
echo ""

# FFN2: 4b (re-run at 1ep)
BITS="Q=8,K=8,V=8,O=8,FFN1=8,FFN2=4"
DIR="${OUT_BASE}/sens_FFN2_4b_1ep"
echo "[$(date +%H:%M)] FFN2=4b, rest=8b → $DIR"
mkdir -p "$DIR"
python paper_experiment.py $COMMON \
  --io-bits 8 \
  --per-layer-bits "$BITS" \
  --output-dir "$DIR" \
  2>&1 | tee "${OUT_BASE}/sens_FFN2_4b_1ep.log"
echo "[$(date +%H:%M)] Done: FFN2=4b"
echo ""

echo "============================================"
echo "[$(date +%H:%M)] All runs completed"
echo "============================================"

# Summary
echo ""
echo "=== SUMMARY ==="
for B in 4 5 6 7; do
  LOG="${OUT_BASE}/sens_QKVO_${B}b_1ep.log"
  if [ -f "$LOG" ]; then
    RESULT=$(grep "Epoch 1 F1:" "$LOG" 2>/dev/null | tail -1)
    echo "  QKVO=${B}b: $RESULT"
  fi
done
for GROUP in FFN1 FFN2; do
  LOG="${OUT_BASE}/sens_${GROUP}_4b_1ep.log"
  if [ -f "$LOG" ]; then
    RESULT=$(grep "Epoch 1 F1:" "$LOG" 2>/dev/null | tail -1)
    echo "  ${GROUP}=4b: $RESULT"
  fi
done

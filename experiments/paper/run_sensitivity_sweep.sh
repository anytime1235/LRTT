#!/bin/bash
# Sensitivity sweep: FFN1 (5,6,7b), FFN2 (5,6,7b), O (4,5,6,7b)
# Baseline: all other sublayers = 8b
set -e
source ~/.venv310/bin/activate
cd /root/LRTT/experiments/paper

OUT_BASE="/root/results/sa_v4_training_sensitivity"
COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 4 --seed 42 --mode fixed \
  --min-lr-rate 0.05"

mkdir -p "$OUT_BASE"

echo "============================================"
echo "Sensitivity Sweep: FFN1(5,6,7), FFN2(5,6,7), O(4,5,6,7)"
echo "Started: $(date)"
echo "============================================"

# FFN1: 5b, 6b, 7b
for BITS_VAL in 5 6 7; do
  BITS="Q=8,K=8,V=8,O=8,FFN1=${BITS_VAL},FFN2=8"
  DIR="${OUT_BASE}/sens_FFN1_${BITS_VAL}b"
  echo "[$(date +%H:%M)] FFN1=${BITS_VAL}b, rest=8b → $DIR"
  mkdir -p "$DIR"
  python paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$BITS" \
    --output-dir "$DIR" \
    2>&1 | tee "${OUT_BASE}/sens_FFN1_${BITS_VAL}b.log"
  echo "[$(date +%H:%M)] Done: FFN1=${BITS_VAL}b"
  echo ""
done

# FFN2: 5b, 6b, 7b
for BITS_VAL in 5 6 7; do
  BITS="Q=8,K=8,V=8,O=8,FFN1=8,FFN2=${BITS_VAL}"
  DIR="${OUT_BASE}/sens_FFN2_${BITS_VAL}b"
  echo "[$(date +%H:%M)] FFN2=${BITS_VAL}b, rest=8b → $DIR"
  mkdir -p "$DIR"
  python paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$BITS" \
    --output-dir "$DIR" \
    2>&1 | tee "${OUT_BASE}/sens_FFN2_${BITS_VAL}b.log"
  echo "[$(date +%H:%M)] Done: FFN2=${BITS_VAL}b"
  echo ""
done

# O: 4b, 5b, 6b, 7b
for BITS_VAL in 4 5 6 7; do
  BITS="Q=8,K=8,V=8,O=${BITS_VAL},FFN1=8,FFN2=8"
  DIR="${OUT_BASE}/sens_O_${BITS_VAL}b"
  echo "[$(date +%H:%M)] O=${BITS_VAL}b, rest=8b → $DIR"
  mkdir -p "$DIR"
  python paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$BITS" \
    --output-dir "$DIR" \
    2>&1 | tee "${OUT_BASE}/sens_O_${BITS_VAL}b.log"
  echo "[$(date +%H:%M)] Done: O=${BITS_VAL}b"
  echo ""
done

echo "============================================"
echo "[$(date +%H:%M)] All sensitivity sweep runs completed"
echo "============================================"

# Summary
echo ""
echo "=== SUMMARY ==="
for GROUP in FFN1 FFN2; do
  for B in 5 6 7; do
    LOG="${OUT_BASE}/sens_${GROUP}_${B}b.log"
    RESULT=$(grep "Done:" "$LOG" 2>/dev/null | tail -1)
    echo "  ${GROUP}=${B}b: $RESULT"
  done
done
for B in 4 5 6 7; do
  LOG="${OUT_BASE}/sens_O_${B}b.log"
  RESULT=$(grep "Done:" "$LOG" 2>/dev/null | tail -1)
  echo "  O=${B}b: $RESULT"
done

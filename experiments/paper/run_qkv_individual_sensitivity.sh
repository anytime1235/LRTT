#!/bin/bash
# Q/K/V individual sensitivity sweep: base=8b, one sublayer→{4,5,6,7}b, 1ep each
# Total: 3 sublayers × 4 bit points = 12 runs
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
echo "Q/K/V Individual Sensitivity Sweep"
echo "  base=8b, one sublayer→{4,5,6,7}b, 1ep"
echo "  Total: 12 runs"
echo "============================================"

RUN=0
for SUBLAYER in Q K V; do
  for BIT in 4 5 6 7; do
    RUN=$((RUN + 1))

    case $SUBLAYER in
      Q) BITS="Q=${BIT},K=8,V=8,O=8,FFN1=8,FFN2=8" ;;
      K) BITS="Q=8,K=${BIT},V=8,O=8,FFN1=8,FFN2=8" ;;
      V) BITS="Q=8,K=8,V=${BIT},O=8,FFN1=8,FFN2=8" ;;
    esac

    DIR="${OUT_BASE}/sens_${SUBLAYER}_${BIT}b_1ep"
    echo "[$(date +%H:%M)] [${RUN}/12] ${SUBLAYER}=${BIT}b, rest=8b → $DIR"
    mkdir -p "$DIR"
    python paper_experiment.py $COMMON \
      --io-bits 8 \
      --per-layer-bits "$BITS" \
      --output-dir "$DIR" \
      2>&1 | tee "${OUT_BASE}/sens_${SUBLAYER}_${BIT}b_1ep.log"
    echo "[$(date +%H:%M)] Done: ${SUBLAYER}=${BIT}b"
    echo ""
  done
done

echo "============================================"
echo "[$(date +%H:%M)] All 12 runs completed"
echo "============================================"

echo ""
echo "=== SUMMARY ==="
for SUBLAYER in Q K V; do
  for BIT in 4 5 6 7; do
    LOG="${OUT_BASE}/sens_${SUBLAYER}_${BIT}b_1ep.log"
    F1=$(grep "Epoch 1 F1:" "$LOG" 2>/dev/null | tail -1)
    echo "  ${SUBLAYER}=${BIT}b: $F1"
  done
done

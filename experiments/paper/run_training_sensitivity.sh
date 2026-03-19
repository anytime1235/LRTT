#!/bin/bash
# Training-aware sensitivity analysis: Phase 1
# Baseline 8b, one sublayer group → 4b, 4 epochs
# Groups: QKV (in-projection), O (out-projection), FFN1, FFN2
set -e
source ~/.venv310/bin/activate
cd /root

OUT_BASE="results/sa_v4_training_sensitivity"
COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 4 --seed 42 --mode fixed \
  --min-lr-rate 0.05"

mkdir -p "$OUT_BASE"

echo "============================================"
echo "Training Sensitivity: base=8b, target=4b, 4ep"
echo "  Groups: QKV, O, FFN1, FFN2"
echo "============================================"

for GROUP in QKV O FFN1 FFN2; do
  case $GROUP in
    QKV)  BITS="Q=4,K=4,V=4,O=8,FFN1=8,FFN2=8" ;;
    O)    BITS="Q=8,K=8,V=8,O=4,FFN1=8,FFN2=8" ;;
    FFN1) BITS="Q=8,K=8,V=8,O=8,FFN1=4,FFN2=8" ;;
    FFN2) BITS="Q=8,K=8,V=8,O=8,FFN1=8,FFN2=4" ;;
  esac

  DIR="${OUT_BASE}/sens_${GROUP}_4b"
  echo "[$(date +%H:%M)] ${GROUP}=4b, rest=8b → $DIR"
  mkdir -p "$DIR"
  python paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$BITS" \
    --output-dir "$DIR" \
    2>&1 | tee "${OUT_BASE}/sens_${GROUP}_4b.log"
  echo "[$(date +%H:%M)] Done: ${GROUP}=4b"
  echo ""
done

echo "============================================"
echo "[$(date +%H:%M)] All sensitivity runs completed"
echo "============================================"

# Print summary
echo ""
echo "=== SUMMARY ==="
for GROUP in QKV O FFN1 FFN2; do
  LOG="${OUT_BASE}/sens_${GROUP}_4b.log"
  RESULT=$(grep "Done:" "$LOG" 2>/dev/null | tail -1)
  echo "  ${GROUP}=4b: $RESULT"
done

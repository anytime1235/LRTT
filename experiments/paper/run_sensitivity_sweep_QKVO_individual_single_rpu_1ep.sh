#!/bin/bash
# Sensitivity sweep (1 epoch): Q, K, V, O individually at 4,5,6,7 bits
# Device: single_rpu, ConstantStepDevice 14-bit, stochastic pulses
# Target: attention only (QKVO), FFN stays digital
# IO: 8-bit base, abs_max noise management, ITERATIVE bound management
#
# OOM-aware: tries batch=16/ga=3 → batch=12/ga=4 → batch=8/ga=6
set -e
source ~/.venv310/bin/activate
cd /root/LRTT/experiments/paper

OUT_BASE="/root/results/sa_v4_training_sensitivity_single_rpu"
DONE_MARKER="${OUT_BASE}/.completed_runs"

COMMON_FIXED="--method single_rpu --target-layers attention --noise-management abs_max \
  --n-bits 14 --pulse-type stochastic --desired-bl 31 \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --epochs 1 --seed 42 --mode fixed --min-lr-rate 0.05"

mkdir -p "$OUT_BASE"
touch "$DONE_MARKER"

# All experiment configs: SUBLAYER BITS_VAL
EXPERIMENTS=(
  "Q 4" "Q 5" "Q 6" "Q 7"
  "K 4" "K 5" "K 6" "K 7"
  "V 4" "V 5" "V 6" "V 7"
  "O 4" "O 5" "O 6" "O 7"
)

# Batch size fallback tiers
BATCH_TIERS=("16 3" "12 4" "8 6")

run_all_experiments() {
  local BS=$1
  local GA=$2

  echo "============================================"
  echo "Single RPU 14b | QKVO individual sensitivity sweep (1ep)"
  echo "Batch=$BS, GradAccum=$GA (effective=$(( BS * GA )))"
  echo "Started: $(date)"
  echo "============================================"

  for EXP in "${EXPERIMENTS[@]}"; do
    local SL=$(echo "$EXP" | awk '{print $1}')
    local BV=$(echo "$EXP" | awk '{print $2}')

    local RUN_ID="sens_${SL}_${BV}b_1ep"

    # Skip if already completed
    if grep -qx "$RUN_ID" "$DONE_MARKER" 2>/dev/null; then
      echo "[$(date +%H:%M)] SKIP (already done): $RUN_ID"
      continue
    fi

    # Build per-layer-bits string: target sublayer at BV, others at 8
    case "$SL" in
      Q) BITS="Q=${BV},K=8,V=8,O=8" ;;
      K) BITS="Q=8,K=${BV},V=8,O=8" ;;
      V) BITS="Q=8,K=8,V=${BV},O=8" ;;
      O) BITS="Q=8,K=8,V=8,O=${BV}" ;;
    esac

    local DIR="${OUT_BASE}/${RUN_ID}"
    local LOG="${OUT_BASE}/${RUN_ID}.log"
    echo "[$(date +%H:%M)] ${SL}=${BV}b (rest=8b) → $DIR"
    mkdir -p "$DIR"

    python paper_experiment.py $COMMON_FIXED \
      --batch-size "$BS" --grad-accum-steps "$GA" \
      --io-bits 8 \
      --per-layer-bits "$BITS" \
      --output-dir "$DIR" \
      2>&1 | tee "$LOG"

    local EXIT_CODE=${PIPESTATUS[0]}

    # Check for OOM
    if [ $EXIT_CODE -ne 0 ]; then
      if grep -qi "out of memory\|CUDA out of memory\|OOM\|RuntimeError.*CUDA" "$LOG" 2>/dev/null; then
        echo ""
        echo "!!! OOM detected at ${SL}=${BV}b with batch=$BS !!!"
        echo "Stopping current tier..."
        # Clean up failed run directory
        rm -rf "$DIR"
        return 1  # Signal OOM to caller
      else
        echo "!!! Non-OOM error at ${SL}=${BV}b (exit=$EXIT_CODE). Aborting. !!!"
        exit 2
      fi
    fi

    # Mark as completed
    echo "$RUN_ID" >> "$DONE_MARKER"
    echo "[$(date +%H:%M)] Done: ${SL}=${BV}b"
    echo ""
  done

  return 0  # All experiments completed successfully
}

# Main: try each batch tier
for TIER in "${BATCH_TIERS[@]}"; do
  BS=$(echo "$TIER" | awk '{print $1}')
  GA=$(echo "$TIER" | awk '{print $2}')

  if run_all_experiments "$BS" "$GA"; then
    echo "============================================"
    echo "[$(date +%H:%M)] All experiments completed successfully (batch=$BS, ga=$GA)"
    echo "============================================"

    # Print summary
    echo ""
    echo "=== SUMMARY ==="
    for EXP in "${EXPERIMENTS[@]}"; do
      SL=$(echo "$EXP" | awk '{print $1}')
      BV=$(echo "$EXP" | awk '{print $2}')
      LOG="${OUT_BASE}/sens_${SL}_${BV}b_1ep.log"
      if [ -f "$LOG" ]; then
        RESULT=$(grep "Epoch 1 F1:" "$LOG" 2>/dev/null | tail -1)
        echo "  ${SL}=${BV}b: $RESULT"
      fi
    done
    exit 0
  else
    echo ""
    echo ">>> OOM with batch=$BS. Falling back to next tier..."
    echo ""
  fi
done

echo "!!! All batch tiers exhausted. Cannot run experiments. !!!"
exit 1

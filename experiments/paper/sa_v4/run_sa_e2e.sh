#!/bin/bash
# SA mixed-precision e2e runs: avg 5,6,7,8,9 bit (highest bit first)
set -e
source ~/.venv310/bin/activate
cd /root

SA_DIR="results/sa_v4/sensitivity_allocation"
COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 4 --seed 42 --mode fixed \
  --min-lr-rate 0.05"

echo "=============================="
echo "SA Mixed-Precision E2E Runs"
echo "=============================="

for BUDGET in 9 8 7 6 5; do
  JSON="${SA_DIR}/precision_map_budget${BUDGET}.json"
  DIR="results/sa_v4/e2e_budget${BUDGET}"
  LOG="results/sa_v4/e2e_budget${BUDGET}.log"

  if [ ! -f "$JSON" ]; then
    echo "[$(date +%H:%M)] SKIP budget ${BUDGET}b — no precision map"
    continue
  fi

  echo "[$(date +%H:%M)] Budget ${BUDGET}b → $DIR"
  mkdir -p "$DIR"
  python paper_experiment.py $COMMON \
    --io-bits 8 \
    --per-layer-bits "$JSON" \
    --output-dir "$DIR" \
    2>&1 | tee "$LOG"
  echo "[$(date +%H:%M)] Done: budget ${BUDGET}b"
done

echo "=============================="
echo "[$(date +%H:%M)] All SA e2e runs completed"
echo "=============================="

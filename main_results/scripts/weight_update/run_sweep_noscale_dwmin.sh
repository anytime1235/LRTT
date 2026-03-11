#!/bin/bash
# Sweep: auto_scale=False + small dw_min + low LR
#
# Hypothesis: auto_scale=False + dw_min↓ → grad/dw_min↑ → pulse encodes gradient direction
# Also test lr=0.01, 0.001 (lower lr → smaller Δw but already small grad stays same)
#
# Conditions:
#   dw_min ∈ {0.0005(baseline), 5e-05, 5e-06, 5e-07}
#   lr ∈ {0.1(baseline), 0.01, 0.001}
#   auto_scale=False (all runs)
#   Total: 4 × 3 = 12 runs (minus baseline already exists = 11 new runs)

set -euo pipefail

PYTHON="/root/.venv310/bin/python"
SCRIPT="/root/LRTT/main_results/scripts/weight_update/diag_weight_update_bert_v2.py"
OUTDIR="/root/LRTT/main_results/results/weight_update/squad/tiki/sweep_noscale_dwmin"
LOGDIR="${OUTDIR}/logs"
mkdir -p "$LOGDIR"

# auto_scale=False → --no-auto-scale
COMMON_BASE="--mode tiki --steps 200 --transfer-every 8 \
  --transfer-desired-bl 1 --desired-bl 31 \
  --forward-perfect --backward-perfect \
  --fast-lr 1.0 --transfer-lr 1.0 --no-auto-scale \
  --no-uim --exclude-ffn --eval-loss --eval-every 10 --overwrite"

RUN=0
FAIL=0
TOTAL_RUNS=12
START_TIME=$(date +%s)

echo "============================================================"
echo "Sweep: auto_scale=False + small dw_min + low LR"
echo "  ${TOTAL_RUNS} conditions, --no-trace (eval_loss only)"
echo "  dw_min ∈ {0.0005, 5e-05, 5e-06, 5e-07}"
echo "  lr    ∈ {0.1, 0.01, 0.001}"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output: $OUTDIR"
echo "============================================================"

DW_MIN_VALUES=(0.0005 0.00005 0.000005 0.0000005)
LR_VALUES=(0.1 0.01 0.001)

for DW in "${DW_MIN_VALUES[@]}"; do
    for LR in "${LR_VALUES[@]}"; do
        RUN=$((RUN + 1))
        TAG="noscale_dw${DW}_lr${LR}"
        LOG="${LOGDIR}/${TAG}.log"

        # Expected grad/dw_min for reference
        # grad_absmean ≈ 0.0003 (at lr=0.1), scales with lr
        echo "[${RUN}/${TOTAL_RUNS}] dw_min=${DW}, lr=${LR} → ${TAG}"

        if $PYTHON $SCRIPT $COMMON_BASE \
             --dw-min "$DW" --lr "$LR" \
             --tag "$TAG" --no-trace \
             --output-dir "$OUTDIR" > "$LOG" 2>&1; then
            # Extract best and final loss
            if [ -f "$OUTDIR"/run_*"${TAG}"*/eval_loss.csv ]; then
                BEST=$(sort -t',' -k2 -n "$OUTDIR"/run_*"${TAG}"*/eval_loss.csv 2>/dev/null | grep -v step | head -1 | cut -d',' -f2)
                FINAL=$(tail -1 "$OUTDIR"/run_*"${TAG}"*/eval_loss.csv | cut -d',' -f2)
                echo "    OK  best_loss=${BEST}  final_loss=${FINAL}"
            else
                echo "    OK  (no eval_loss.csv found)"
            fi
        else
            FAIL=$((FAIL + 1))
            echo "    FAIL  (see $LOG)"
        fi
    done
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "============================================================"
echo "Phase 1 complete: ${RUN} runs, ${FAIL} failures, ${ELAPSED}s elapsed"
echo "============================================================"

# ===================== Summary =====================
echo ""
echo ">>> Results Summary <<<"
echo ""
printf "%-40s  %12s  %12s\n" "Condition" "best_loss" "final_loss"
printf "%-40s  %12s  %12s\n" "----------------------------------------" "------------" "------------"

for CSV in "$OUTDIR"/run_*/eval_loss.csv; do
    [ -f "$CSV" ] || continue
    DIR=$(dirname "$CSV")
    TAG_NAME=$(basename "$DIR")
    # Short name: extract dw and lr from tag
    SHORT=$(echo "$TAG_NAME" | grep -oP 'noscale_dw[0-9.e-]+_lr[0-9.]+')
    BEST=$(sort -t',' -k2 -n "$CSV" 2>/dev/null | grep -v step | head -1 | cut -d',' -f2)
    FINAL=$(tail -1 "$CSV" | cut -d',' -f2)
    printf "%-40s  %12s  %12s\n" "$SHORT" "$BEST" "$FINAL"
done | sort

# ===================== Phase 2: Best with trace =====================
echo ""
echo ">>> Phase 2: Re-run best condition with --trace-every 1 <<<"

BEST_TAG=""
BEST_LOSS=999999

for CSV in "$OUTDIR"/run_*/eval_loss.csv; do
    [ -f "$CSV" ] || continue
    DIR=$(dirname "$CSV")
    FINAL_LOSS=$(sort -t',' -k2 -n "$CSV" 2>/dev/null | grep -v step | head -1 | cut -d',' -f2)
    [ -z "$FINAL_LOSS" ] && continue
    IS_BETTER=$(awk "BEGIN {print ($FINAL_LOSS < $BEST_LOSS) ? 1 : 0}")
    if [ "$IS_BETTER" -eq 1 ]; then
        BEST_LOSS="$FINAL_LOSS"
        # Extract dw_min and lr from config
        BEST_DW=$(python3 -c "import json; c=json.load(open('$DIR/config_dump.json')); print(c['dw_min'])")
        BEST_LR=$(python3 -c "import json; c=json.load(open('$DIR/config_dump.json')); print(c['lr'])")
        BEST_TAG="noscale_dw${BEST_DW}_lr${BEST_LR}"
    fi
done

if [ -z "$BEST_TAG" ]; then
    echo "ERROR: No results found."
    exit 1
fi

echo "Best condition: ${BEST_TAG} (best_loss=${BEST_LOSS})"
echo "Re-running with --trace-every 1 ..."

TRACE_TAG="${BEST_TAG}_trace"
LOG="${LOGDIR}/${TRACE_TAG}.log"

$PYTHON $SCRIPT $COMMON_BASE \
    --dw-min "$BEST_DW" --lr "$BEST_LR" \
    --tag "$TRACE_TAG" --trace-every 1 \
    --output-dir "$OUTDIR" > "$LOG" 2>&1

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "============================================================"
echo "All done. Total elapsed: ${TOTAL_ELAPSED}s"
echo "Best: ${BEST_TAG} (best_loss=${BEST_LOSS})"
echo "Trace output: ${OUTDIR}/"
echo "============================================================"

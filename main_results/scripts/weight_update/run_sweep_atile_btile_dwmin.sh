#!/bin/bash
# A-tile & B-tile dw_min Sweep
#
# Baseline (sweep_24 best): lr=0.1, te=8, tbl=1, forward_perfect, backward_perfect, steps=200
# Baseline dw_min values: A=0.001981, B=0.0005 (already exists, not re-run)
#
# Phase 1: 15 runs --no-trace → eval_loss only
#   A-tile sweep (B=0.0005 fixed, noisy):  3 runs
#   B-tile sweep (A=default, noisy):       2 runs
#   Both sweep (noisy):                    2 runs
#   A noise-free (all 8 A/B combos):       8 runs
#   Baseline (A=0.001981, B=0.0005, noisy) already exists → not re-run
# Phase 2: best by sum(|delta_L_analog|) → trace_every=1 re-run

set -euo pipefail

PYTHON="/root/.venv310/bin/python"
SCRIPT="/root/LRTT/main_results/scripts/weight_update/diag_weight_update_bert_v2.py"
OUTDIR="/root/LRTT/main_results/results/weight_update/squad/tiki/sweep_dwmin_ablation"
LOGDIR="${OUTDIR}/logs"
mkdir -p "$LOGDIR"

COMMON="--mode tiki --steps 200 --lr 0.1 --transfer-every 8 \
  --transfer-desired-bl 1 --desired-bl 31 \
  --forward-perfect --backward-perfect \
  --dw-min 0.0005 --fast-lr 1.0 --transfer-lr 1.0 --auto-scale \
  --no-uim --exclude-ffn --eval-loss --eval-every 10 --overwrite"

RUN=0
FAIL=0
START_TIME=$(date +%s)

echo "============================================================"
echo "A-tile & B-tile dw_min Sweep"
echo "  Phase 1: eval_loss only (--no-trace), 15 runs"
echo "  Phase 2: best combo re-run with trace_every=1"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output: $OUTDIR"
echo "============================================================"

# ===================== Phase 1: Sweep (no-trace) =====================
echo ""
echo ">>> Phase 1: Running 15 conditions (--no-trace) <<<"
TOTAL_RUNS=15

# --- A-tile dw_min sweep (B-tile dw_min=0.0005 fixed) ---
A_DW_MIN_VALUES=(0.000198 0.0000198 0.00000198)

for A_DW in "${A_DW_MIN_VALUES[@]}"; do
    RUN=$((RUN + 1))
    TAG="Atile_dwmin_${A_DW}"
    LOG="${LOGDIR}/${TAG}.log"
    echo "[${RUN}/${TOTAL_RUNS}] A-tile dw_min=${A_DW} (B=0.0005) → ${TAG}"
    if $PYTHON $SCRIPT $COMMON --dw-min-a "$A_DW" \
         --tag "$TAG" --no-trace \
         --output-dir "$OUTDIR" > "$LOG" 2>&1; then
        LAST_LOSS=$(grep -oP 'eval_loss=\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
        echo "    OK  final_eval_loss=${LAST_LOSS}"
    else
        FAIL=$((FAIL + 1))
        echo "    FAIL  (see $LOG)"
    fi
done

# --- B-tile dw_min sweep (A-tile dw_min=default 0.001981) ---
B_DW_MIN_VALUES=(0.00005 0.000005)

for B_DW in "${B_DW_MIN_VALUES[@]}"; do
    RUN=$((RUN + 1))
    TAG="Btile_dwmin_${B_DW}"
    LOG="${LOGDIR}/${TAG}.log"
    echo "[${RUN}/${TOTAL_RUNS}] B-tile dw_min=${B_DW} (A=default) → ${TAG}"
    if $PYTHON $SCRIPT $COMMON --dw-min "$B_DW" \
         --tag "$TAG" --no-trace \
         --output-dir "$OUTDIR" > "$LOG" 2>&1; then
        LAST_LOSS=$(grep -oP 'eval_loss=\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
        echo "    OK  final_eval_loss=${LAST_LOSS}"
    else
        FAIL=$((FAIL + 1))
        echo "    FAIL  (see $LOG)"
    fi
done

# --- A-tile noise-free sweep (all A/B combos) ---
NF_COMBOS=(
    "0.001981:0.0005"       # baseline
    "0.000198:0.0005"       # A x0.1, B fixed
    "0.0000198:0.0005"      # A x0.01, B fixed
    "0.00000198:0.0005"     # A x0.001, B fixed
    "0.001981:0.00005"      # A default, B x0.1
    "0.001981:0.000005"     # A default, B x0.01
    "0.000198:0.00005"      # Both x0.1
    "0.0000198:0.000005"    # Both x0.01
)

for COMBO in "${NF_COMBOS[@]}"; do
    A_DW="${COMBO%%:*}"
    B_DW="${COMBO##*:}"
    RUN=$((RUN + 1))
    TAG="Anoisefree_A_${A_DW}_B_${B_DW}"
    LOG="${LOGDIR}/${TAG}.log"
    echo "[${RUN}/${TOTAL_RUNS}] A-tile dw_min=${A_DW} noise-free, B-tile dw_min=${B_DW} → ${TAG}"
    if $PYTHON $SCRIPT $COMMON --dw-min-a "$A_DW" --dw-min "$B_DW" --a-noise-free \
         --tag "$TAG" --no-trace \
         --output-dir "$OUTDIR" > "$LOG" 2>&1; then
        LAST_LOSS=$(grep -oP 'eval_loss=\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
        echo "    OK  final_eval_loss=${LAST_LOSS}"
    else
        FAIL=$((FAIL + 1))
        echo "    FAIL  (see $LOG)"
    fi
done

# --- Both A+B dw_min sweep ---
BOTH_COMBOS=("0.000198:0.00005" "0.0000198:0.000005")

for COMBO in "${BOTH_COMBOS[@]}"; do
    A_DW="${COMBO%%:*}"
    B_DW="${COMBO##*:}"
    RUN=$((RUN + 1))
    TAG="Both_A_${A_DW}_B_${B_DW}"
    LOG="${LOGDIR}/${TAG}.log"
    echo "[${RUN}/${TOTAL_RUNS}] A-tile dw_min=${A_DW}, B-tile dw_min=${B_DW} → ${TAG}"
    if $PYTHON $SCRIPT $COMMON --dw-min-a "$A_DW" --dw-min "$B_DW" \
         --tag "$TAG" --no-trace \
         --output-dir "$OUTDIR" > "$LOG" 2>&1; then
        LAST_LOSS=$(grep -oP 'eval_loss=\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
        echo "    OK  final_eval_loss=${LAST_LOSS}"
    else
        FAIL=$((FAIL + 1))
        echo "    FAIL  (see $LOG)"
    fi
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "============================================================"
echo "Phase 1 complete: ${RUN} runs, ${FAIL} failures, ${ELAPSED}s elapsed"
echo "============================================================"

# ===================== Phase 2: Best condition with trace =====================
echo ""
echo ">>> Phase 2: Identify best condition and re-run with --trace-every 1 <<<"
echo ""

# Collect final eval_loss from each run's eval_loss.csv
BEST_TAG=""
BEST_LOSS=999999

for CSV in "$OUTDIR"/*/eval_loss.csv; do
    [ -f "$CSV" ] || continue
    DIR=$(dirname "$CSV")
    TAG_NAME=$(basename "$DIR")
    FINAL_LOSS=$(tail -1 "$CSV" | cut -d',' -f2)
    echo "  ${TAG_NAME}: final_eval_loss=${FINAL_LOSS}"
    # Compare (use awk for float comparison)
    IS_BETTER=$(awk "BEGIN {print ($FINAL_LOSS < $BEST_LOSS) ? 1 : 0}")
    if [ "$IS_BETTER" -eq 1 ]; then
        BEST_LOSS="$FINAL_LOSS"
        BEST_TAG="$TAG_NAME"
    fi
done

if [ -z "$BEST_TAG" ]; then
    echo "ERROR: No eval_loss.csv found. Cannot determine best condition."
    exit 1
fi

echo ""
echo "Best condition: ${BEST_TAG} (eval_loss=${BEST_LOSS})"
echo "Re-running with --trace-every 1 ..."

# Parse best tag to reconstruct args
TRACE_TAG="${BEST_TAG}_trace"
LOG="${LOGDIR}/${TRACE_TAG}.log"

if [[ "$BEST_TAG" == Atile_dwmin_* ]]; then
    A_DW="${BEST_TAG#Atile_dwmin_}"
    echo "  → A-tile dw_min=${A_DW}, B-tile dw_min=0.0005"
    $PYTHON $SCRIPT $COMMON --dw-min-a "$A_DW" \
        --tag "$TRACE_TAG" --trace-every 1 \
        --output-dir "$OUTDIR" > "$LOG" 2>&1
elif [[ "$BEST_TAG" == Btile_dwmin_* ]]; then
    B_DW="${BEST_TAG#Btile_dwmin_}"
    echo "  → A-tile dw_min=default, B-tile dw_min=${B_DW}"
    $PYTHON $SCRIPT $COMMON --dw-min "$B_DW" \
        --tag "$TRACE_TAG" --trace-every 1 \
        --output-dir "$OUTDIR" > "$LOG" 2>&1
elif [[ "$BEST_TAG" == Anoisefree_A_*_B_* ]]; then
    REMAINDER="${BEST_TAG#Anoisefree_A_}"
    A_DW="${REMAINDER%%_B_*}"
    B_DW="${REMAINDER#*_B_}"
    echo "  → A-tile dw_min=${A_DW} noise-free, B-tile dw_min=${B_DW}"
    $PYTHON $SCRIPT $COMMON --dw-min-a "$A_DW" --dw-min "$B_DW" --a-noise-free \
        --tag "$TRACE_TAG" --trace-every 1 \
        --output-dir "$OUTDIR" > "$LOG" 2>&1
elif [[ "$BEST_TAG" == Both_A_*_B_* ]]; then
    REMAINDER="${BEST_TAG#Both_A_}"
    A_DW="${REMAINDER%%_B_*}"
    B_DW="${REMAINDER#*_B_}"
    echo "  → A-tile dw_min=${A_DW}, B-tile dw_min=${B_DW}"
    $PYTHON $SCRIPT $COMMON --dw-min-a "$A_DW" --dw-min "$B_DW" \
        --tag "$TRACE_TAG" --trace-every 1 \
        --output-dir "$OUTDIR" > "$LOG" 2>&1
else
    echo "ERROR: Cannot parse best tag: ${BEST_TAG}"
    exit 1
fi

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "============================================================"
echo "All done. Total elapsed: ${TOTAL_ELAPSED}s"
echo "Phase 2 trace output: ${OUTDIR}/${TRACE_TAG}/"
echo "============================================================"

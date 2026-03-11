#!/bin/bash
# Phase 1: 24 TikiTaka combinations — eval_loss ONLY (no tracing)
# Phase 2: Re-run best combo with trace_every=1 for full diagnostics
#
#   3 lr × 2 transfer_every × 2 transfer_desired_bl × 2 is_perfect = 24
# Fixed: dw_min=0.0005, desired_bl=31, fast_lr=1.0, transfer_lr=1.0, steps=200
#
# --no-trace skips all weight reads / hooks / metrics_steps.csv → ~45× faster
#
# scale_transfer_lr=True (v2 default) → actual_transfer_lr = transfer_lr × lr
#
# is_perfect dimension isolates I/O noise from weight-update quality:
#   noisy   = realistic analog fwd/bwd (default)
#   perfect = ideal fwd/bwd → pure weight-update diagnostic

set -euo pipefail

PYTHON="/root/.venv310/bin/python"
SCRIPT="/root/LRTT/main_results/scripts/weight_update/diag_weight_update_bert_v2.py"
OUTDIR="/root/LRTT/main_results/results/weight_update/squad/tiki/sweep_24_tiki"
LOGDIR="${OUTDIR}/logs"
mkdir -p "$LOGDIR"

COMMON_BASE="--mode tiki --steps 200 --dw-min 0.0005 --desired-bl 31 \
  --no-uim --exclude-ffn --eval-loss --eval-every 10 --overwrite \
  --fast-lr 1.0 --transfer-lr 1.0 --no-auto-scale"

LR_VALUES=(0.01 0.1 1.0)
TE_VALUES=(8 32)
TBL_VALUES=(1 31)
PERFECT_VALUES=("none" "both")

TOTAL=$(( ${#LR_VALUES[@]} * ${#TE_VALUES[@]} * ${#TBL_VALUES[@]} * ${#PERFECT_VALUES[@]} ))
RUN=0
FAIL=0
START_TIME=$(date +%s)

echo "============================================================"
echo "TikiTaka 24-combo sweep"
echo "  Phase 1: eval_loss only (--no-trace)"
echo "  Phase 2: best combo re-run with trace_every=1"
echo "  lr:             ${LR_VALUES[*]}"
echo "  transfer_every: ${TE_VALUES[*]}"
echo "  transfer_bl:    ${TBL_VALUES[*]}"
echo "  fwd/bwd:        noisy, perfect"
echo "  steps=200, eval_every=10"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output: $OUTDIR"
echo "============================================================"

# ===================== Phase 1: Sweep (no-trace) =====================
echo ""
echo ">>> Phase 1: Running $TOTAL combinations (--no-trace) <<<"

for LR in "${LR_VALUES[@]}"; do
    for TE in "${TE_VALUES[@]}"; do
        for TBL in "${TBL_VALUES[@]}"; do
            for PERF in "${PERFECT_VALUES[@]}"; do
                RUN=$((RUN + 1))

                PERF_FLAGS=""
                PERF_TAG="noisy"
                if [ "$PERF" = "both" ]; then
                    PERF_FLAGS="--forward-perfect --backward-perfect"
                    PERF_TAG="perfect"
                fi

                TAG="lr${LR}_te${TE}_tbl${TBL}_${PERF_TAG}"
                LOG="${LOGDIR}/${TAG}.log"

                echo ""
                echo "[$RUN/$TOTAL] lr=$LR te=$TE tbl=$TBL fwd/bwd=$PERF_TAG  ($(date '+%H:%M:%S'))"
                echo "  Log: $LOG"

                $PYTHON "$SCRIPT" $COMMON_BASE --no-trace \
                    --lr "$LR" \
                    --transfer-every "$TE" \
                    --transfer-desired-bl "$TBL" \
                    $PERF_FLAGS \
                    --output-dir "$OUTDIR" \
                    > "$LOG" 2>&1

                EXIT_CODE=$?
                if [ $EXIT_CODE -ne 0 ]; then
                    echo "  *** FAILED (exit=$EXIT_CODE) — see $LOG ***"
                    FAIL=$((FAIL + 1))
                else
                    echo "  Done."
                fi
            done
        done
    done
done

P1_END=$(date +%s)
P1_ELAPSED=$(( P1_END - START_TIME ))
P1_HOURS=$(( P1_ELAPSED / 3600 ))
P1_MINS=$(( (P1_ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "Phase 1 complete: $((TOTAL - FAIL))/$TOTAL succeeded, $FAIL failed"
echo "Elapsed: ${P1_HOURS}h ${P1_MINS}m"
echo "============================================================"

# ===================== Phase 2: Best combo with tracing =====================
echo ""
echo ">>> Phase 2: Identifying best combination by delta_L_total <<<"

# Find best combo: parse all eval_loss.csv, extract config from parent dir name
BEST_CSV=$($PYTHON -c "
import pandas as pd, glob, os
csvs = glob.glob('${OUTDIR}/run_*/eval_loss.csv')
if not csvs:
    print('NO_CSV_FOUND')
    exit(0)
results = []
for csv_path in csvs:
    try:
        df = pd.read_csv(csv_path)
        # Use last row's delta_L_total (cumulative loss change at final eval step)
        last = df.iloc[-1]
        run_dir = os.path.basename(os.path.dirname(csv_path))
        results.append((last['delta_L_total'], csv_path, run_dir))
    except Exception:
        pass
if not results:
    print('NO_VALID_RESULTS')
    exit(0)
# Best = most negative delta_L_total (biggest loss reduction)
results.sort(key=lambda x: x[0])
best_loss, best_csv, best_dir = results[0]
print(best_csv)
# Also print ranking summary
print('---RANKING---')
for i, (dl, cp, rd) in enumerate(results):
    print(f'  {i+1}. delta_L={dl:.6f}  {rd}')
")

if [ "$BEST_CSV" = "NO_CSV_FOUND" ] || [ "$BEST_CSV" = "NO_VALID_RESULTS" ]; then
    echo "  ERROR: No valid eval_loss.csv found. Skipping Phase 2."
    exit 1
fi

# Parse first line (best csv path) and ranking
BEST_PATH=$(echo "$BEST_CSV" | head -1)
BEST_DIR=$(dirname "$BEST_PATH")
BEST_RUN=$(basename "$BEST_DIR")

echo "$BEST_CSV" | tail -n +2
echo ""
echo "Best combo: $BEST_RUN"
echo "  eval_loss.csv: $BEST_PATH"

# Extract config from directory name: run_squad_seed42_uimF_te{TE}_tbl{TBL}_tcT_dwB0p0005_lrA{LR}_flr{FLR}_lrD{LRD}_{HASH}
# Parse te, tbl, lr from dir name
BEST_TE=$(echo "$BEST_RUN" | grep -oP '_te\K[0-9]+')
BEST_TBL=$(echo "$BEST_RUN" | grep -oP '_tbl\K[0-9]+')
BEST_LR=$(echo "$BEST_RUN" | grep -oP '_lrA\K[0-9p]+' | sed 's/p/./g')

# Determine if perfect: check config_dump.json
BEST_FWD_PERFECT=$($PYTHON -c "
import json
with open('${BEST_DIR}/config_dump.json') as f:
    cfg = json.load(f)
print('yes' if cfg.get('forward_perfect', False) else 'no')
")

BEST_PERF_FLAGS=""
BEST_PERF_TAG="noisy"
if [ "$BEST_FWD_PERFECT" = "yes" ]; then
    BEST_PERF_FLAGS="--forward-perfect --backward-perfect"
    BEST_PERF_TAG="perfect"
fi

echo "  Parsed: lr=$BEST_LR te=$BEST_TE tbl=$BEST_TBL fwd/bwd=$BEST_PERF_TAG"
echo ""
echo ">>> Phase 2: Re-running best combo with trace_every=1 <<<"

TRACE_TAG="lr${BEST_LR}_te${BEST_TE}_tbl${BEST_TBL}_${BEST_PERF_TAG}"
TRACE_LOG="${LOGDIR}/${TRACE_TAG}_trace.log"
TRACE_OUTDIR="${OUTDIR}_trace"

echo "  Log: $TRACE_LOG"
echo "  Output: $TRACE_OUTDIR"

P2_START=$(date +%s)

$PYTHON "$SCRIPT" $COMMON_BASE \
    --lr "$BEST_LR" \
    --transfer-every "$BEST_TE" \
    --transfer-desired-bl "$BEST_TBL" \
    $BEST_PERF_FLAGS \
    --trace-every 1 \
    --output-dir "$TRACE_OUTDIR" \
    > "$TRACE_LOG" 2>&1

P2_EXIT=$?
P2_END=$(date +%s)
P2_ELAPSED=$(( P2_END - P2_START ))
P2_HOURS=$(( P2_ELAPSED / 3600 ))
P2_MINS=$(( (P2_ELAPSED % 3600) / 60 ))

if [ $P2_EXIT -ne 0 ]; then
    echo "  *** Phase 2 FAILED (exit=$P2_EXIT) — see $TRACE_LOG ***"
else
    echo "  Phase 2 done."
fi

# ===================== Final Summary =====================
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - START_TIME ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "All done!"
echo "  Phase 1: $((TOTAL - FAIL))/$TOTAL sweep combos succeeded (${P1_HOURS}h ${P1_MINS}m)"
echo "  Phase 2: trace_every=1 on best combo (${P2_HOURS}h ${P2_MINS}m) — exit=$P2_EXIT"
echo "  Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "  Sweep results: $OUTDIR"
echo "  Trace results: $TRACE_OUTDIR"
echo "============================================================"

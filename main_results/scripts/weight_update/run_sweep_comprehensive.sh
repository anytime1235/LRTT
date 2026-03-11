#!/bin/bash
# ============================================================================
# Comprehensive Sweep: 100 runs to find working analog TikiTaka config
# ============================================================================
#
# Root cause: grad_absmean(~0.0003) < dw_min(0.0005) → pulse can't encode gradient
#
# Strategy: sweep across ALL knobs that could fix this:
#   A) auto_scale=False + small dw_min  (increase grad/dw_min ratio)
#   B) High desired_bl                   (more pulses → better direction)
#   C) High LR                           (amplify gradient magnitude)
#   D) units_in_mbatch=True              (accumulate grad within minibatch)
#   E) forget_buffer / transfer_every    (prevent noise accumulation)
#   F) Combined best ideas               (stack multiple fixes)
#   G) correct_gradient_magnitudes       (aihwkit internal correction)
#   H) digital_optimizer + digital_lr    (better digital component)
#   I) momentum                          (smooth out noise)
# ============================================================================

set -euo pipefail

PYTHON="/root/.venv310/bin/python"
SCRIPT="/root/LRTT/main_results/scripts/weight_update/diag_weight_update_bert_v2.py"
OUTDIR="/root/LRTT/main_results/results/weight_update/squad/tiki/sweep_comprehensive"
LOGDIR="${OUTDIR}/logs"
mkdir -p "$LOGDIR"

# Fixed params for all runs
FIXED="--mode tiki --steps 200 \
  --forward-perfect --backward-perfect \
  --no-uim --exclude-ffn --eval-loss --eval-every 10 --overwrite --no-trace"

RUN=0
FAIL=0
START_TIME=$(date +%s)

run_one() {
    local TAG="$1"
    shift
    RUN=$((RUN + 1))
    local LOG="${LOGDIR}/${TAG}.log"
    echo "[${RUN}] ${TAG}"
    if $PYTHON $SCRIPT $FIXED "$@" \
         --tag "$TAG" --output-dir "$OUTDIR" > "$LOG" 2>&1; then
        # Find the output dir for this run
        local CSV=$(ls "$OUTDIR"/run_*"${TAG}"*/eval_loss.csv 2>/dev/null | head -1)
        if [ -n "$CSV" ] && [ -f "$CSV" ]; then
            local BEST=$(sort -t',' -k2 -n "$CSV" | grep -v step | head -1 | cut -d',' -f2)
            local FINAL=$(tail -1 "$CSV" | cut -d',' -f2)
            echo "    OK  best=${BEST}  final=${FINAL}"
        else
            echo "    OK  (no csv)"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "    FAIL  (see $LOG)"
    fi
}

echo "============================================================"
echo "Comprehensive Sweep: ~100 runs"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output: $OUTDIR"
echo "============================================================"

# ====================================================================
# Group A: auto_scale=False + dw_min sweep × lr sweep (15 runs)
# Core hypothesis: grad/dw_min ↑ → better pulse encoding
# ====================================================================
echo ""
echo ">>> Group A: auto_scale=False + dw_min × lr (15 runs) <<<"

for DW in 0.0005 0.00005 0.000005 0.0000005 0.00000005; do
    for LR in 0.1 0.01 0.001; do
        run_one "A_noscale_dw${DW}_lr${LR}" \
            --no-auto-scale --dw-min "$DW" --lr "$LR" \
            --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
            --fast-lr 1.0 --transfer-lr 1.0
    done
done

# ====================================================================
# Group B: auto_scale=True + high desired_bl (12 runs)
# More pulses → better gradient direction resolution
# ====================================================================
echo ""
echo ">>> Group B: auto_scale=True + high desired_bl (12 runs) <<<"

for BL in 100 500 1000 2000; do
    for DW in 0.0005 0.00005 0.000005; do
        run_one "B_scale_bl${BL}_dw${DW}" \
            --auto-scale --dw-min "$DW" --lr 0.1 \
            --desired-bl "$BL" --transfer-every 8 --transfer-desired-bl 1 \
            --fast-lr 1.0 --transfer-lr 1.0
    done
done

# ====================================================================
# Group C: High LR to amplify gradient (10 runs)
# grad × lr → if lr↑, effective gradient ↑
# ====================================================================
echo ""
echo ">>> Group C: High LR (10 runs) <<<"

for LR in 0.5 1.0; do
    for DW in 0.0005 0.00005 0.000005; do
        run_one "C_highLR_lr${LR}_dw${DW}_noscale" \
            --no-auto-scale --dw-min "$DW" --lr "$LR" \
            --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
            --fast-lr 1.0 --transfer-lr 1.0
    done
    for DW in 0.0005 0.00005; do
        run_one "C_highLR_lr${LR}_dw${DW}_scale" \
            --auto-scale --dw-min "$DW" --lr "$LR" \
            --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
            --fast-lr 1.0 --transfer-lr 1.0
    done
done

# ====================================================================
# Group D: units_in_mbatch=True (grad accum within batch) (8 runs)
# ====================================================================
echo ""
echo ">>> Group D: units_in_mbatch=True (8 runs) <<<"

for DW in 0.0005 0.00005 0.000005; do
    run_one "D_uim_dw${DW}_noscale" \
        --no-auto-scale --dw-min "$DW" --lr 0.1 \
        --uim --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done
for DW in 0.0005 0.00005 0.000005; do
    run_one "D_uim_dw${DW}_scale" \
        --auto-scale --dw-min "$DW" --lr 0.1 \
        --uim --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done
# uim + high BL
run_one "D_uim_bl500_dw0.00005_scale" \
    --auto-scale --dw-min 0.00005 --lr 0.1 \
    --uim --desired-bl 500 --transfer-every 8 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0
run_one "D_uim_bl1000_dw0.000005_noscale" \
    --no-auto-scale --dw-min 0.000005 --lr 0.1 \
    --uim --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0

# ====================================================================
# Group E: transfer_every sweep (10 runs)
# Frequent transfer → less buffer accumulation
# ====================================================================
echo ""
echo ">>> Group E: transfer_every sweep (10 runs) <<<"

for TE in 1 2 4 16 32; do
    run_one "E_te${TE}_dw0.0005_noscale" \
        --no-auto-scale --dw-min 0.0005 --lr 0.1 \
        --desired-bl 31 --transfer-every "$TE" --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done
for TE in 1 2 4 16 32; do
    run_one "E_te${TE}_dw0.00005_noscale" \
        --no-auto-scale --dw-min 0.00005 --lr 0.1 \
        --desired-bl 31 --transfer-every "$TE" --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done

# ====================================================================
# Group F: forget_buffer=True (6 runs)
# Reset buffer after transfer → prevent noise accumulation
# ====================================================================
echo ""
echo ">>> Group F: forget_buffer=True (6 runs) <<<"

for DW in 0.0005 0.00005 0.000005; do
    run_one "F_forget_dw${DW}_noscale" \
        --no-auto-scale --dw-min "$DW" --lr 0.1 \
        --forget-buffer --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done
for DW in 0.0005 0.00005 0.000005; do
    run_one "F_forget_dw${DW}_scale" \
        --auto-scale --dw-min "$DW" --lr 0.1 \
        --forget-buffer --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done

# ====================================================================
# Group G: correct_gradient_magnitudes (4 runs)
# ====================================================================
echo ""
echo ">>> Group G: correct_gradient_magnitudes (4 runs) <<<"

for DW in 0.0005 0.00005; do
    run_one "G_cgm_dw${DW}_noscale" \
        --no-auto-scale --dw-min "$DW" --lr 0.1 \
        --correct-gradient-magnitudes \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
    run_one "G_cgm_dw${DW}_scale" \
        --auto-scale --dw-min "$DW" --lr 0.1 \
        --correct-gradient-magnitudes \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done

# ====================================================================
# Group H: digital_optimizer=adam + digital_lr (6 runs)
# Better digital component to compensate analog failure
# ====================================================================
echo ""
echo ">>> Group H: digital Adam optimizer (6 runs) <<<"

for DLR in 0.001 0.0001 0.00001; do
    run_one "H_adam_dlr${DLR}_dw0.0005_noscale" \
        --no-auto-scale --dw-min 0.0005 --lr 0.1 \
        --digital-optimizer adam --digital-lr "$DLR" \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done
for DLR in 0.001 0.0001 0.00001; do
    run_one "H_adam_dlr${DLR}_dw0.00005_noscale" \
        --no-auto-scale --dw-min 0.00005 --lr 0.1 \
        --digital-optimizer adam --digital-lr "$DLR" \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done

# ====================================================================
# Group I: momentum (4 runs)
# Smooth out noise in transfer
# ====================================================================
echo ""
echo ">>> Group I: momentum (4 runs) <<<"

for MOM in 0.5 0.9; do
    run_one "I_mom${MOM}_dw0.0005_noscale" \
        --no-auto-scale --dw-min 0.0005 --lr 0.1 \
        --momentum "$MOM" \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
    run_one "I_mom${MOM}_dw0.00005_noscale" \
        --no-auto-scale --dw-min 0.00005 --lr 0.1 \
        --momentum "$MOM" \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done

# ====================================================================
# Group J: Combined - stack multiple fixes (15 runs)
# Best combos from above
# ====================================================================
echo ""
echo ">>> Group J: Combined strategies (15 runs) <<<"

# Small dw_min + frequent transfer + forget buffer
for DW in 0.00005 0.000005; do
    run_one "J_combo1_dw${DW}_te1_forget" \
        --no-auto-scale --dw-min "$DW" --lr 0.1 \
        --forget-buffer --desired-bl 31 --transfer-every 1 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
    run_one "J_combo1_dw${DW}_te2_forget" \
        --no-auto-scale --dw-min "$DW" --lr 0.1 \
        --forget-buffer --desired-bl 31 --transfer-every 2 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done

# Small dw_min + UIM + forget buffer
for DW in 0.00005 0.000005; do
    run_one "J_combo2_dw${DW}_uim_forget" \
        --no-auto-scale --dw-min "$DW" --lr 0.1 \
        --uim --forget-buffer --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr 1.0
done

# High BL + small dw_min + forget buffer
run_one "J_combo3_bl500_dw5e-5_forget_scale" \
    --auto-scale --dw-min 0.00005 --lr 0.1 \
    --forget-buffer --desired-bl 500 --transfer-every 4 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0
run_one "J_combo3_bl1000_dw5e-6_forget_scale" \
    --auto-scale --dw-min 0.000005 --lr 0.1 \
    --forget-buffer --desired-bl 1000 --transfer-every 4 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0

# Small dw_min + Adam digital + forget buffer
run_one "J_combo4_dw5e-5_adam_forget" \
    --no-auto-scale --dw-min 0.00005 --lr 0.1 \
    --forget-buffer --digital-optimizer adam --digital-lr 0.001 \
    --desired-bl 31 --transfer-every 4 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0
run_one "J_combo4_dw5e-6_adam_forget" \
    --no-auto-scale --dw-min 0.000005 --lr 0.1 \
    --forget-buffer --digital-optimizer adam --digital-lr 0.001 \
    --desired-bl 31 --transfer-every 4 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0

# High LR + small dw_min + forget buffer
run_one "J_combo5_lr0.5_dw5e-5_te4_forget" \
    --no-auto-scale --dw-min 0.00005 --lr 0.5 \
    --forget-buffer --desired-bl 31 --transfer-every 4 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0
run_one "J_combo5_lr1.0_dw5e-6_te4_forget" \
    --no-auto-scale --dw-min 0.000005 --lr 1.0 \
    --forget-buffer --desired-bl 31 --transfer-every 4 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0

# UIM + high LR + small dw_min
run_one "J_combo6_uim_lr0.5_dw5e-5" \
    --no-auto-scale --dw-min 0.00005 --lr 0.5 \
    --uim --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0
run_one "J_combo6_uim_lr0.5_dw5e-6" \
    --no-auto-scale --dw-min 0.000005 --lr 0.5 \
    --uim --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
    --fast-lr 1.0 --transfer-lr 1.0

# ====================================================================
# Group K: fast_lr sweep (4 runs)
# Lower fast_lr to reduce digital noise impact
# ====================================================================
echo ""
echo ">>> Group K: fast_lr sweep (4 runs) <<<"

for FLR in 0.1 0.01; do
    run_one "K_flr${FLR}_dw0.00005_noscale" \
        --no-auto-scale --dw-min 0.00005 --lr 0.1 \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr "$FLR" --transfer-lr 1.0
    run_one "K_flr${FLR}_dw0.000005_noscale" \
        --no-auto-scale --dw-min 0.000005 --lr 0.1 \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr "$FLR" --transfer-lr 1.0
done

# ====================================================================
# Group L: transfer_lr sweep (3 runs)
# Higher transfer_lr → more aggressive transfer from A to B
# ====================================================================
echo ""
echo ">>> Group L: transfer_lr sweep (3 runs) <<<"

for TLR in 2.0 5.0 10.0; do
    run_one "L_tlr${TLR}_dw0.00005_noscale" \
        --no-auto-scale --dw-min 0.00005 --lr 0.1 \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl 1 \
        --fast-lr 1.0 --transfer-lr "$TLR"
done

# ====================================================================
# Group M: transfer_desired_bl sweep (4 runs)
# Higher transfer BL → more pulses during A→B transfer
# ====================================================================
echo ""
echo ">>> Group M: transfer_desired_bl sweep (4 runs) <<<"

for TBL in 10 31 100 500; do
    run_one "M_tbl${TBL}_dw0.00005_noscale" \
        --no-auto-scale --dw-min 0.00005 --lr 0.1 \
        --desired-bl 31 --transfer-every 8 --transfer-desired-bl "$TBL" \
        --fast-lr 1.0 --transfer-lr 1.0
done

# ====================================================================
# DONE — Summary (Total: 100 runs)
# ====================================================================
ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "============================================================"
echo "All ${RUN} runs complete. ${FAIL} failures. ${ELAPSED}s elapsed."
echo "============================================================"
echo ""
echo ">>> TOP 10 by best_loss <<<"
echo ""
printf "%-55s  %10s  %10s\n" "Tag" "best_loss" "final_loss"
printf "%-55s  %10s  %10s\n" "-------------------------------------------------------" "----------" "----------"

for CSV in "$OUTDIR"/run_*/eval_loss.csv; do
    [ -f "$CSV" ] || continue
    DIR=$(dirname "$CSV")
    NAME=$(basename "$DIR")
    # Extract tag portion
    TAG=$(echo "$NAME" | sed 's/run_squad_seed42_[^_]*_te[0-9]*_tbl[0-9]*_tc[TF]_//' | sed 's/_dw[^_]*_lrA[^_]*_flr[^_]*_lrD[^_]*_[a-f0-9]*//')
    BEST=$(sort -t',' -k2 -n "$CSV" | grep -v step | head -1 | cut -d',' -f2)
    FINAL=$(tail -1 "$CSV" | cut -d',' -f2)
    echo "${TAG}|${BEST}|${FINAL}"
done | sort -t'|' -k2 -n | head -10 | while IFS='|' read TAG BEST FINAL; do
    printf "%-55s  %10s  %10s\n" "$TAG" "$BEST" "$FINAL"
done

echo ""
echo "Baseline reference: best_loss=1.64 (auto_scale=False, dw_min=0.0005, lr=0.1)"
echo ""

# Phase 2: Re-run top-5 with trace
echo ">>> Phase 2: Re-run top-5 with --trace-every 1 <<<"

# Collect all results and sort by best_loss
python3 -c "
import csv, json, os, glob

results = []
for csv_path in glob.glob('$OUTDIR/run_*/eval_loss.csv'):
    d = os.path.dirname(csv_path)
    try:
        rows = list(csv.DictReader(open(csv_path)))
        if not rows: continue
        best = min(rows, key=lambda r: float(r['L0']))
        results.append((float(best['L0']), d))
    except:
        pass

results.sort()
for i, (loss, d) in enumerate(results[:5]):
    print(f'{loss}|{d}')
" > "${OUTDIR}/top5.txt"

TRACE_RANK=0
while IFS='|' read LOSS DIR; do
    TRACE_RANK=$((TRACE_RANK + 1))
    echo ""
    echo "[Trace ${TRACE_RANK}/5] best_loss=${LOSS} dir=$(basename $DIR)"

    TRACE_ARGS=$(python3 -c "
import json
c = json.load(open('$DIR/config_dump.json'))
args = []
args.append(f'--dw-min {c[\"dw_min\"]}')
args.append(f'--lr {c[\"lr\"]}')
args.append(f'--desired-bl {c[\"desired_bl\"]}')
args.append(f'--transfer-every {c[\"transfer_every\"]}')
tbl = c.get('transfer_desired_bl', 1) or 1
args.append(f'--transfer-desired-bl {tbl}')
args.append(f'--fast-lr {c[\"fast_lr\"]}')
args.append(f'--transfer-lr {c[\"transfer_lr\"]}')
if c.get('auto_scale', False):
    args.append('--auto-scale')
else:
    args.append('--no-auto-scale')
if c.get('units_in_mbatch', False):
    args.append('--uim')
else:
    args.append('--no-uim')
if c.get('forget_buffer', False):
    args.append('--forget-buffer')
if c.get('correct_gradient_magnitudes', False):
    args.append('--correct-gradient-magnitudes')
if c.get('momentum', 0) > 0:
    args.append(f'--momentum {c[\"momentum\"]}')
do = c.get('digital_optimizer', 'sgd')
if do != 'sgd':
    args.append(f'--digital-optimizer {do}')
dlr = c.get('digital_lr')
if dlr is not None:
    args.append(f'--digital-lr {dlr}')
if c.get('dw_min_a') is not None:
    args.append(f'--dw-min-a {c[\"dw_min_a\"]}')
if c.get('a_noise_free', False):
    args.append('--a-noise-free')
print(' '.join(args))
")

    echo "  Args: ${TRACE_ARGS}"
    TRACE_TAG="trace_top${TRACE_RANK}"
    TRACE_LOG="${LOGDIR}/${TRACE_TAG}.log"

    $PYTHON $SCRIPT --mode tiki --steps 200 \
        --forward-perfect --backward-perfect \
        --exclude-ffn --eval-loss --eval-every 10 --overwrite \
        $TRACE_ARGS \
        --trace-every 1 \
        --tag "$TRACE_TAG" --output-dir "$OUTDIR" > "$TRACE_LOG" 2>&1 && \
        echo "  OK" || echo "  FAIL (see $TRACE_LOG)"

done < "${OUTDIR}/top5.txt"

echo ""
echo "Phase 2 complete: top-5 trace runs done."

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "============================================================"
echo "GRAND TOTAL: ${TOTAL_ELAPSED}s elapsed"
echo "============================================================"

#!/bin/bash
# GPU-parallel master launcher: maximizes GPU utilization.
#
# Structure:
#   Stage 1: Phase 0 (smoke + LN ablation) — all GPUs, synchronized
#            → determines BEST_LN_LR
#   Stage 2: Phase 1-4 — GPU-independent pipelines (no idle GPUs)
#
# GPU 0: Reserved (running MP sweep experiments)
# GPU 1/2/3: Used for paper experiments
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RESULTS_DIR="${RESULTS_DIR:-results/paper}"
export PYTHON="${PYTHON:-python}"

# Phase 1 best config (UPDATE AFTER PHASE 1):
export BEST_GAMMA="${BEST_GAMMA:-0.0}"
export BEST_UIM="${BEST_UIM:-false}"
export BEST_TE="${BEST_TE:-24}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  GPU-Parallel Paper Experiment Launcher"
echo "========================================"
echo "Results root: $RESULTS_DIR"
echo ""

# Helper
run() {
    local GPU=$1; shift
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py "$@"
}

# ============================================================
# Stage 1: Phase 0 — Smoke + LN Ablation (synchronized)
# ============================================================
echo "=== Stage 1: Phase 0 (smoke + LN ablation) ==="

# Phase 0A: smoke tests — 3 GPUs parallel
echo "--- Phase 0A: Smoke (5 methods) ---"

# Round 1: 3 methods parallel
run 1 --mode fixed --method single_rpu --max-steps 50 --seed 42 \
    --output-dir "$RESULTS_DIR/phase0/single_rpu" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID1=$!
run 2 --mode fixed --method mixed_precision --max-steps 50 --seed 42 \
    --output-dir "$RESULTS_DIR/phase0/mixed_precision" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID2=$!
run 3 --mode fixed --method ttv1 --max-steps 50 --seed 42 --gamma 0.0 \
    --output-dir "$RESULTS_DIR/phase0/ttv1" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID3=$!
wait $PID1 $PID2 $PID3
echo "Round 1 done."

# Round 2: ideal only
run 1 --mode fixed --method ideal --max-steps 50 --seed 42 \
    --output-dir "$RESULTS_DIR/phase0/ideal" \
    --log-every 1 --diag-update-exact --diag-steps 50 &
PID1=$!
wait $PID1
echo "Round 2 done."

echo ""
echo "--- Phase 0A Results ---"
for METHOD in single_rpu mixed_precision ttv1 ideal; do
    DIR="$RESULTS_DIR/phase0/$METHOD"
    if [ -f "$DIR/summary.json" ]; then
        echo "  $METHOD: OK"
    else
        echo "  $METHOD: MISSING (check logs)"
    fi
done

# Phase 0B: LN ablation — GPU 1 & 2 parallel, GPU 3 idle briefly
echo ""
echo "--- Phase 0B: LN LR Ablation ---"
run 1 --mode fixed --method single_rpu --seed 42 \
    --epochs 2 --n-bits 14 --ln-lr 0.016 \
    --output-dir "$RESULTS_DIR/phase0/ln_lr_ablation/ln_eq_analog" \
    --log-every 20 &
PID1=$!
run 2 --mode fixed --method single_rpu --seed 42 \
    --epochs 2 --n-bits 14 --ln-lr 0.003 \
    --output-dir "$RESULTS_DIR/phase0/ln_lr_ablation/ln_eq_classifier" \
    --log-every 20 &
PID2=$!
wait $PID1 $PID2
echo "LN ablation complete."

# Determine best LN LR
echo ""
BEST_LN_LR=$($PYTHON -c "
import json, sys
results = {}
for tag, path in [('0.016', '$RESULTS_DIR/phase0/ln_lr_ablation/ln_eq_analog/summary.json'),
                  ('0.003', '$RESULTS_DIR/phase0/ln_lr_ablation/ln_eq_classifier/summary.json')]:
    try:
        with open(path) as f:
            d = json.load(f)
        results[tag] = d['results']['best_f1']
        print(f'  ln_lr={tag}: best_f1={d[\"results\"][\"best_f1\"]:.2f}', file=sys.stderr)
    except Exception as e:
        print(f'  ln_lr={tag}: ERROR - {e}', file=sys.stderr)
best = max(results, key=results.get) if results else '0.016'
print(best)
" 2>&1 | tee /dev/stderr | tail -1)
BEST_LN_LR="${BEST_LN_LR:-0.016}"

export BEST_LN_LR
mkdir -p "$RESULTS_DIR/phase0/ln_lr_ablation"
echo "$BEST_LN_LR" > "$RESULTS_DIR/phase0/ln_lr_ablation/best_ln_lr.txt"
echo ">>> BEST_LN_LR=$BEST_LN_LR"
echo ""

# ============================================================
# Stage 2: Phase 1-4 — GPU-independent pipelines
# ============================================================
echo "=== Stage 2: Phase 1-4 (GPU-independent pipelines) ==="
echo "Using BEST_LN_LR=$BEST_LN_LR"
echo ""

# ---- GPU 1 Pipeline ----
gpu1_pipeline() {
    local LN_LR=$BEST_LN_LR
    echo "[GPU 1] Starting pipeline (ln_lr=$LN_LR)"

    # Phase 1: TTv1 configA
    for G in 0.0 0.1; do
        echo "[GPU 1] Phase 1: configA gamma=$G"
        run 1 --mode fixed --method ttv1 --seed 42 \
            --epochs 2 --n-bits 14 --gamma $G \
            --units-in-mbatch false --transfer-every 24 --ln-lr $LN_LR \
            --output-dir "$RESULTS_DIR/phase1/configA_gamma${G}" --log-every 20
    done

    # Phase 2: single_rpu 8,10,12,14
    for BITS in 8 10 12 14; do
        echo "[GPU 1] Phase 2: single_rpu ${BITS}b"
        run 1 --mode fixed --method single_rpu --seed 42 \
            --epochs 4 --n-bits $BITS --ln-lr $LN_LR \
            --output-dir "$RESULTS_DIR/phase2/single_rpu_${BITS}b" --log-every 20
    done

    # Phase 3: diagnostics
    for PT in none_with_device stochastic; do
        echo "[GPU 1] Phase 3: $PT"
        run 1 --mode fixed --method single_rpu --seed 42 \
            --max-steps 100 --n-bits 14 --pulse-type $PT --ln-lr $LN_LR \
            --output-dir "$RESULTS_DIR/phase3/$PT" \
            --log-every 1 --diag-update-exact --diag-steps 100
    done

    echo "[GPU 1] Pipeline complete."
}

# ---- GPU 2 Pipeline ----
gpu2_pipeline() {
    local LN_LR=$BEST_LN_LR
    echo "[GPU 2] Starting pipeline (ln_lr=$LN_LR)"

    # Phase 1: TTv1 configB
    for G in 0.0 0.1; do
        echo "[GPU 2] Phase 1: configB gamma=$G"
        run 2 --mode fixed --method ttv1 --seed 42 \
            --epochs 2 --n-bits 14 --gamma $G \
            --units-in-mbatch false --transfer-every 2400 --ln-lr $LN_LR \
            --output-dir "$RESULTS_DIR/phase1/configB_gamma${G}" --log-every 20
    done

    # Phase 2: single_rpu 16, mixed_precision 8,10
    echo "[GPU 2] Phase 2: single_rpu 16b"
    run 2 --mode fixed --method single_rpu --seed 42 \
        --epochs 4 --n-bits 16 --ln-lr $LN_LR \
        --output-dir "$RESULTS_DIR/phase2/single_rpu_16b" --log-every 20

    for BITS in 8 10; do
        echo "[GPU 2] Phase 2: mixed_precision ${BITS}b"
        run 2 --mode fixed --method mixed_precision --seed 42 \
            --epochs 4 --n-bits $BITS --ln-lr $LN_LR \
            --output-dir "$RESULTS_DIR/phase2/mixed_precision_${BITS}b" --log-every 20
    done

    # Phase 3: diagnostics
    echo "[GPU 2] Phase 3: deterministic"
    run 2 --mode fixed --method single_rpu --seed 42 \
        --max-steps 100 --n-bits 14 --pulse-type deterministic --ln-lr $LN_LR \
        --output-dir "$RESULTS_DIR/phase3/deterministic" \
        --log-every 1 --diag-update-exact --diag-steps 100

    echo "[GPU 2] Phase 3: mixed_precision"
    run 2 --mode fixed --method mixed_precision --seed 42 \
        --max-steps 100 --n-bits 14 --ln-lr $LN_LR \
        --output-dir "$RESULTS_DIR/phase3/mixed_precision" \
        --log-every 1 --diag-update-exact --diag-steps 100

    echo "[GPU 2] Pipeline complete."
}

# ---- GPU 3 Pipeline ----
gpu3_pipeline() {
    local LN_LR=$BEST_LN_LR
    echo "[GPU 3] Starting pipeline (ln_lr=$LN_LR)"

    # Phase 1: TTv1 configC
    for G in 0.0 0.1; do
        echo "[GPU 3] Phase 1: configC gamma=$G"
        run 3 --mode fixed --method ttv1 --seed 42 \
            --epochs 2 --n-bits 14 --gamma $G \
            --units-in-mbatch true --transfer-every 1 --ln-lr $LN_LR \
            --output-dir "$RESULTS_DIR/phase1/configC_gamma${G}" --log-every 20
    done

    # Phase 2: mixed_precision 12,14,16
    for BITS in 12 14 16; do
        echo "[GPU 3] Phase 2: mixed_precision ${BITS}b"
        run 3 --mode fixed --method mixed_precision --seed 42 \
            --epochs 4 --n-bits $BITS --ln-lr $LN_LR \
            --output-dir "$RESULTS_DIR/phase2/mixed_precision_${BITS}b" --log-every 20
    done

    # Phase 3: diagnostics
    echo "[GPU 3] Phase 3: mean_count"
    run 3 --mode fixed --method single_rpu --seed 42 \
        --max-steps 100 --n-bits 14 --pulse-type mean_count --ln-lr $LN_LR \
        --output-dir "$RESULTS_DIR/phase3/mean_count" \
        --log-every 1 --diag-update-exact --diag-steps 100

    echo "[GPU 3] Phase 3: ttv1_best"
    run 3 --mode fixed --method ttv1 --seed 42 \
        --max-steps 100 --n-bits 14 \
        --gamma $BEST_GAMMA --units-in-mbatch $BEST_UIM --transfer-every $BEST_TE \
        --ln-lr $LN_LR \
        --output-dir "$RESULTS_DIR/phase3/ttv1_best" \
        --log-every 1 --diag-update-exact --diag-steps 100

    # Phase 4: TTv1 final (moved here for load balancing)
    echo "[GPU 3] Phase 4: TTv1 final"
    run 3 --mode fixed --method ttv1 --seed 42 \
        --epochs 4 --n-bits 14 \
        --gamma $BEST_GAMMA --units-in-mbatch $BEST_UIM --transfer-every $BEST_TE \
        --ln-lr $LN_LR \
        --output-dir "$RESULTS_DIR/phase4/ttv1_final" --log-every 20

    echo "[GPU 3] Pipeline complete."
}

# Launch Stage 2 pipelines in parallel
gpu1_pipeline > >(tee "$RESULTS_DIR/gpu1_pipeline.log") 2>&1 &
PID_GPU1=$!
gpu2_pipeline > >(tee "$RESULTS_DIR/gpu2_pipeline.log") 2>&1 &
PID_GPU2=$!
gpu3_pipeline > >(tee "$RESULTS_DIR/gpu3_pipeline.log") 2>&1 &
PID_GPU3=$!

echo "PIDs: GPU1=$PID_GPU1, GPU2=$PID_GPU2, GPU3=$PID_GPU3"
echo "Logs: $RESULTS_DIR/gpu{1,2,3}_pipeline.log"
echo ""

wait $PID_GPU1 $PID_GPU2 $PID_GPU3

echo ""
echo "========================================"
echo "  All experiments complete!"
echo "========================================"

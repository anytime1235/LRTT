#!/bin/bash
# Master launcher: ALL experiments strictly sequential per GPU.
# GPU 1,2,3 parallel (1 experiment each), phases sequential.
# Each experiment gets a full clean GPU — no memory conflicts.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHON="${PYTHON:-python}"
RESULTS="${RESULTS_DIR:-results/paper}"

echo "========================================"
echo "  Paper Experiment: Sequential Launcher"
echo "========================================"
echo "Results: $RESULTS"
echo ""

run() {
    # Usage: run GPU METHOD OUTDIR [extra args...]
    local GPU=$1 METHOD=$2 OUTDIR=$3
    shift 3
    echo "[GPU $GPU] Starting: $METHOD -> $OUTDIR"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON paper_experiment.py \
        --mode fixed --method $METHOD \
        --output-dir "$OUTDIR" "$@"
    echo "[GPU $GPU] Done: $METHOD -> $OUTDIR"
}

# ============================================================
# Phase 0A: Smoke tests (5 methods × 50 steps)
# GPU1: single_rpu, GPU2: mixed_precision, GPU3: ttv1
# then GPU1: cttv2, GPU2: ideal
# ============================================================
echo ">>> Phase 0A: Smoke ($(date))"

run 1 single_rpu "$RESULTS/phase0/single_rpu" \
    --max-steps 50 --seed 42 --log-every 1 --diag-update-exact --diag-steps 50 &
run 2 mixed_precision "$RESULTS/phase0/mixed_precision" \
    --max-steps 50 --seed 42 --log-every 1 --diag-update-exact --diag-steps 50 &
run 3 ttv1 "$RESULTS/phase0/ttv1" \
    --max-steps 50 --seed 42 --gamma 0.0 --log-every 1 --diag-update-exact --diag-steps 50 &
wait

run 1 cttv2 "$RESULTS/phase0/cttv2" \
    --max-steps 50 --seed 42 --log-every 1 --diag-update-exact --diag-steps 50 &
run 2 ideal "$RESULTS/phase0/ideal" \
    --max-steps 50 --seed 42 --log-every 1 --diag-update-exact --diag-steps 50 &
wait
echo "Phase 0A done."

# ============================================================
# Phase 0B: LN LR ablation (single_rpu 14-bit, 2 epochs)
#   GPU1: ln_lr=0.016  GPU2: ln_lr=0.003
# ============================================================
echo ""
echo ">>> Phase 0B: LN LR Ablation ($(date))"

run 1 single_rpu "$RESULTS/phase0/ln_lr_ablation/ln_eq_analog" \
    --seed 42 --epochs 2 --n-bits 14 --ln-lr 0.016 --log-every 20 &
run 2 single_rpu "$RESULTS/phase0/ln_lr_ablation/ln_eq_classifier" \
    --seed 42 --epochs 2 --n-bits 14 --ln-lr 0.003 --log-every 20 &
wait
echo "Phase 0B done."

# Pick best LN LR
$PYTHON -c "
import json
results = {}
for tag, lr, path in [('analog', '0.016', '$RESULTS/phase0/ln_lr_ablation/ln_eq_analog/summary.json'),
                       ('classifier', '0.003', '$RESULTS/phase0/ln_lr_ablation/ln_eq_classifier/summary.json')]:
    try:
        d = json.load(open(path))
        f1 = d['results']['best_f1']
        results[lr] = f1
        print(f'  ln={tag}({lr}): best_f1={f1:.2f}')
    except Exception as e:
        print(f'  ln={tag}({lr}): ERROR - {e}')
best_lr = max(results, key=results.get) if results else '0.016'
print(f'  >>> Winner: LN_LR={best_lr}')
open('$RESULTS/phase0/ln_lr_ablation/best_ln_lr.txt', 'w').write(best_lr)
"
LN_LR=$(cat "$RESULTS/phase0/ln_lr_ablation/best_ln_lr.txt" 2>/dev/null || echo "0.016")
echo "Using LN_LR=$LN_LR for all subsequent phases."

# ============================================================
# Phase 1: TTv1 Regime Discovery (6 configs: 3 transfer × 2 gamma)
# GPU1: configA (uim=F te=24)
# GPU2: configB (uim=F te=2400)
# GPU3: configC (uim=T te=1)
# Sequential: gamma=0.0 then gamma=0.1 per GPU
# ============================================================
echo ""
echo ">>> Phase 1: TTv1 Regime Discovery ($(date))"

for GAMMA in 0.0 0.1; do
    echo "--- gamma=$GAMMA ---"
    run 1 ttv1 "$RESULTS/phase1/configA_gamma${GAMMA}" \
        --seed 42 --epochs 2 --n-bits 14 --gamma $GAMMA \
        --units-in-mbatch false --transfer-every 24 --ln-lr $LN_LR --log-every 20 &
    run 2 ttv1 "$RESULTS/phase1/configB_gamma${GAMMA}" \
        --seed 42 --epochs 2 --n-bits 14 --gamma $GAMMA \
        --units-in-mbatch false --transfer-every 2400 --ln-lr $LN_LR --log-every 20 &
    run 3 ttv1 "$RESULTS/phase1/configC_gamma${GAMMA}" \
        --seed 42 --epochs 2 --n-bits 14 --gamma $GAMMA \
        --units-in-mbatch true --transfer-every 1 --ln-lr $LN_LR --log-every 20 &
    wait
done
echo "Phase 1 done."

# Print Phase 1 results
for TAG in configA_gamma0.0 configA_gamma0.1 configB_gamma0.0 configB_gamma0.1 configC_gamma0.0 configC_gamma0.1; do
    DIR="$RESULTS/phase1/$TAG"
    if [ -f "$DIR/summary.json" ]; then
        $PYTHON -c "import json; d=json.load(open('$DIR/summary.json')); print(f'  $TAG: best_f1={d[\"results\"][\"best_f1\"]:.2f}')"
    else
        echo "  $TAG: MISSING"
    fi
done

# ============================================================
# Phase 2: Bit Sweep (10 configs: {single_rpu,mixed_precision} × 5 bits)
# GPU1,2,3 — 3-4 configs each, wave-parallel
# ============================================================
echo ""
echo ">>> Phase 2: Bit Sweep ($(date))"

# Wave 1
run 1 single_rpu "$RESULTS/phase2/single_rpu_8b" \
    --seed 42 --epochs 4 --n-bits 8 --ln-lr $LN_LR --log-every 20 &
run 2 single_rpu "$RESULTS/phase2/single_rpu_10b" \
    --seed 42 --epochs 4 --n-bits 10 --ln-lr $LN_LR --log-every 20 &
run 3 single_rpu "$RESULTS/phase2/single_rpu_12b" \
    --seed 42 --epochs 4 --n-bits 12 --ln-lr $LN_LR --log-every 20 &
wait

# Wave 2
run 1 single_rpu "$RESULTS/phase2/single_rpu_14b" \
    --seed 42 --epochs 4 --n-bits 14 --ln-lr $LN_LR --log-every 20 &
run 2 single_rpu "$RESULTS/phase2/single_rpu_16b" \
    --seed 42 --epochs 4 --n-bits 16 --ln-lr $LN_LR --log-every 20 &
run 3 mixed_precision "$RESULTS/phase2/mixed_precision_8b" \
    --seed 42 --epochs 4 --n-bits 8 --ln-lr $LN_LR --log-every 20 &
wait

# Wave 3
run 1 mixed_precision "$RESULTS/phase2/mixed_precision_10b" \
    --seed 42 --epochs 4 --n-bits 10 --ln-lr $LN_LR --log-every 20 &
run 2 mixed_precision "$RESULTS/phase2/mixed_precision_12b" \
    --seed 42 --epochs 4 --n-bits 12 --ln-lr $LN_LR --log-every 20 &
run 3 mixed_precision "$RESULTS/phase2/mixed_precision_14b" \
    --seed 42 --epochs 4 --n-bits 14 --ln-lr $LN_LR --log-every 20 &
wait

# Wave 4
run 1 mixed_precision "$RESULTS/phase2/mixed_precision_16b" \
    --seed 42 --epochs 4 --n-bits 16 --ln-lr $LN_LR --log-every 20 &
wait
echo "Phase 2 done."

# ============================================================
# Phase 3: Diagnostics (6 configs × 100 steps)
# ============================================================
echo ""
echo ">>> Phase 3: Diagnostics ($(date))"

# Wave 1
run 1 single_rpu "$RESULTS/phase3/none_with_device" \
    --seed 42 --max-steps 100 --n-bits 14 --pulse-type none_with_device \
    --ln-lr $LN_LR --log-every 1 --diag-update-exact --diag-steps 100 &
run 2 single_rpu "$RESULTS/phase3/deterministic" \
    --seed 42 --max-steps 100 --n-bits 14 --pulse-type deterministic \
    --ln-lr $LN_LR --log-every 1 --diag-update-exact --diag-steps 100 &
run 3 single_rpu "$RESULTS/phase3/mean_count" \
    --seed 42 --max-steps 100 --n-bits 14 --pulse-type mean_count \
    --ln-lr $LN_LR --log-every 1 --diag-update-exact --diag-steps 100 &
wait

# Wave 2
run 1 single_rpu "$RESULTS/phase3/stochastic" \
    --seed 42 --max-steps 100 --n-bits 14 --pulse-type stochastic \
    --ln-lr $LN_LR --log-every 1 --diag-update-exact --diag-steps 100 &
run 2 mixed_precision "$RESULTS/phase3/mixed_precision" \
    --seed 42 --max-steps 100 --n-bits 14 \
    --ln-lr $LN_LR --log-every 1 --diag-update-exact --diag-steps 100 &
run 3 ttv1 "$RESULTS/phase3/ttv1_best" \
    --seed 42 --max-steps 100 --n-bits 14 --gamma 0.0 \
    --ln-lr $LN_LR --log-every 1 --diag-update-exact --diag-steps 100 &
wait
echo "Phase 3 done."

# ============================================================
# Phase 4: TTv1 Final (1 config, 4 epochs)
# ============================================================
echo ""
echo ">>> Phase 4: TTv1 Final ($(date))"

run 1 ttv1 "$RESULTS/phase4/ttv1_final" \
    --seed 42 --epochs 4 --n-bits 14 --gamma 0.0 \
    --ln-lr $LN_LR --log-every 20 &
wait
echo "Phase 4 done."

echo ""
echo "========================================"
echo "  ALL PHASES COMPLETE ($(date))"
echo "========================================"

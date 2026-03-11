#!/usr/bin/env bash
set -euo pipefail
cd /data/main_results

PYTHON="/data/venvs/lrtt/bin/python"
SQUAD_SCRIPT="scripts/diagnosis/diag_forward_io_single_rpu.py"
GLUE_SCRIPT="scripts/diagnosis/diag_forward_io_glue.py"

SQUAD_OUT="./results/diag_fwd_io_mitigations"
GLUE_OUT="./results/diag_fwd_io_glue"

SQUAD_COMMON="--n-step 200 --batch-size 8 --dac-bits 7 --dw-min 0.001 --out-dir $SQUAD_OUT"
GLUE_COMMON="--n-step 200 --batch-size 8 --dac-bits 7 --dw-min 0.001 \
             --glue-task sst2 --max-seq-length 128 \
             --out-dir $GLUE_OUT --logit-eval-batches 10"

has_summary() {
    local dir="$1" tag="$2"
    local count
    count=$(find "$dir/$tag" -name '*summary*' 2>/dev/null | wc -l)
    [ "$count" -gt 0 ]
}

# ── S1: SQuAD Mixed Precision base-6 ──
if has_summary "$SQUAD_OUT" "mp_base6"; then
    echo "[SKIP] S1: mp_base6 already has summary files"
else
    echo "========================================"
    echo "S1: SQuAD Mixed Precision base-6"
    echo "========================================"
    $PYTHON $SQUAD_SCRIPT $SQUAD_COMMON --adc-bits-sweep "6" --tag mp_base6 \
        --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1
fi

# ── S2: SQuAD Mixed Precision + Depth Boost ──
if has_summary "$SQUAD_OUT" "mp_base6_depth"; then
    echo "[SKIP] S2: mp_base6_depth already has summary files"
else
    echo "========================================"
    echo "S2: SQuAD Mixed Precision + Depth Boost"
    echo "========================================"
    $PYTHON $SQUAD_SCRIPT $SQUAD_COMMON --adc-bits-sweep "6" --tag mp_base6_depth \
        --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1 \
        --depth-boost "9-11:+1"
fi

# ── S3: SQuAD Stochastic Rounding Baseline ──
if has_summary "$SQUAD_OUT" "baseline_sr"; then
    echo "[SKIP] S3: baseline_sr already has summary files"
else
    echo "========================================"
    echo "S3: SQuAD Stochastic Rounding Baseline"
    echo "========================================"
    $PYTHON $SQUAD_SCRIPT $SQUAD_COMMON --adc-bits-sweep "4,6" --tag baseline_sr --sto-round
fi

# ── S4: SQuAD Stochastic Rounding + OBCAL ──
if has_summary "$SQUAD_OUT" "obcal_per_module_sr"; then
    echo "[SKIP] S4: obcal_per_module_sr already has summary files"
else
    echo "========================================"
    echo "S4: SQuAD Stochastic Rounding + OBCAL"
    echo "========================================"
    $PYTHON $SQUAD_SCRIPT $SQUAD_COMMON --adc-bits-sweep "4,6" --tag obcal_per_module_sr \
        --calib-out-bound --sto-round
fi

# ── S5: SQuAD Combined MP + OBCAL ──
if has_summary "$SQUAD_OUT" "mp_base6_obcal"; then
    echo "[SKIP] S5: mp_base6_obcal already has summary files"
else
    echo "========================================"
    echo "S5: SQuAD Combined MP + OBCAL"
    echo "========================================"
    $PYTHON $SQUAD_SCRIPT $SQUAD_COMMON --adc-bits-sweep "6" --tag mp_base6_obcal \
        --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1 \
        --calib-out-bound
fi

# ── G1: SST-2 OBCAL Sweep ──
if has_summary "$GLUE_OUT" "obcal_sst2"; then
    echo "[SKIP] G1: obcal_sst2 already has summary files"
else
    echo "========================================"
    echo "G1: SST-2 OBCAL Sweep"
    echo "========================================"
    $PYTHON $GLUE_SCRIPT $GLUE_COMMON \
        --adc-bits-sweep "4,6,8,10,12" --tag obcal_sst2 \
        --calib-out-bound --out-bound-grouping per_module --save-calib-table
fi

# ── G2: SST-2 Mixed Precision base-6 ──
if has_summary "$GLUE_OUT" "mp_base6_sst2"; then
    echo "[SKIP] G2: mp_base6_sst2 already has summary files"
else
    echo "========================================"
    echo "G2: SST-2 Mixed Precision base-6"
    echo "========================================"
    $PYTHON $GLUE_SCRIPT $GLUE_COMMON \
        --adc-bits-sweep "6" --tag mp_base6_sst2 \
        --mixed-precision --adc-base 6 --ffn1-bits-plus 2 --v-bits-plus 1
fi

# ── G3: SST-2 Seed Sweep ──
if has_summary "$GLUE_OUT" "seed_sweep_sst2"; then
    echo "[SKIP] G3: seed_sweep_sst2 already has summary files"
else
    echo "========================================"
    echo "G3: SST-2 Seed Sweep"
    echo "========================================"
    $PYTHON $GLUE_SCRIPT $GLUE_COMMON \
        --adc-bits-sweep "4,6,8,10,12" --tag seed_sweep_sst2 \
        --seed-sweep "42,43,44"
fi

echo "========================================"
echo "ALL REMAINING MITIGATIONS COMPLETE"
echo "========================================"

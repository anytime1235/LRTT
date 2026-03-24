#!/bin/bash
# Diagnostic D0: Smoke test — 4 methods × 32 steps × seed 42
# Verifies all new CSV columns exist and have sane values
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-results/paper/diag_D0_smoke}"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Diagnostic D0: Smoke Test (32 steps each) ==="
echo "Results: $RESULTS_DIR"
echo "Start: $(date)"

COMMON_FLAGS="--mode fixed --seed 42 --epochs 1 --max-steps 32 --batch-size 8 \
  --log-every 10 \
  --diag-carry-path --diag-update-exact --diag-steps 32 \
  --diag-vrc-windows 16,32 --diag-layer-set 0,5,11"

# GPU 1: single_rpu 8-bit stochastic
run_gpu1() {
    local TAG="single_rpu_8b_stoch"
    echo "[GPU 1] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method single_rpu --n-bits 8 --pulse-type stochastic \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 1] DONE  $TAG $(date)"
}

# GPU 2: mixed_precision 8-bit
run_gpu2() {
    local TAG="mixed_precision_8b"
    echo "[GPU 2] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method mixed_precision --n-bits 8 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 2] DONE  $TAG $(date)"

    # eco_ref 8-bit (stochastic rounding) after mixed_precision
    local TAG2="eco_ref_8b"
    echo "[GPU 2] START $TAG2 $(date)"
    CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method eco_ref --n-bits 8 --eco-rounding stochastic \
        --output-dir "$RESULTS_DIR/$TAG2"
    echo "[GPU 2] DONE  $TAG2 $(date)"
}

# GPU 3: ttv1 hidden_buffer (slow=8, fast=14)
run_gpu3() {
    local TAG="ttv1_hb_s8_f14"
    echo "[GPU 3] START $TAG $(date)"
    CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
        $COMMON_FLAGS \
        --method ttv1 --ttv1-mode hidden_buffer \
        --n-bits 14 --n-bits-slow 8 \
        --output-dir "$RESULTS_DIR/$TAG"
    echo "[GPU 3] DONE  $TAG $(date)"
}

run_gpu1 2>&1 | tee "$RESULTS_DIR/gpu1.log" &
PID1=$!
run_gpu2 2>&1 | tee "$RESULTS_DIR/gpu2.log" &
PID2=$!
run_gpu3 2>&1 | tee "$RESULTS_DIR/gpu3.log" &
PID3=$!

wait $PID1 $PID2 $PID3
echo ""
echo "=== D0 runs complete: $(date) ==="

# Post-run validation: check CSV columns exist and have sane values
$PYTHON << 'PYEOF'
import csv, json, os, sys

base = os.environ.get("RESULTS_DIR", "results/paper/diag_D0_smoke")
tags = ["single_rpu_8b_stoch", "mixed_precision_8b", "eco_ref_8b", "ttv1_hb_s8_f14"]

errors = []
for tag in tags:
    d = os.path.join(base, tag)

    # Check update_diagnostics.csv (non-eco only)
    if tag != "eco_ref_8b":
        upath = os.path.join(d, "update_diagnostics.csv")
        if os.path.exists(upath):
            with open(upath) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for col in ["frac_mu_lt_1", "frac_mu_lt_0p25", "mu_p50", "mu_p90", "bl_utilization"]:
                if col not in rows[0]:
                    errors.append(f"{tag}: update_diagnostics.csv missing column '{col}'")
            # Sanity: frac_mu_lt_1 should be in [0, 1]
            vals = [float(r["frac_mu_lt_1"]) for r in rows]
            if not all(0.0 <= v <= 1.0 for v in vals):
                errors.append(f"{tag}: frac_mu_lt_1 out of [0,1] range")
        else:
            errors.append(f"{tag}: update_diagnostics.csv not found")

    # Check carry_path_step.csv
    cpath = os.path.join(d, "carry_path_step.csv")
    if os.path.exists(cpath):
        with open(cpath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for col in ["fast_sat_ratio", "pmr", "delta_slow_norm",
                     "frac_mu_lt_1", "frac_mu_lt_0p25", "mu_p50", "mu_p90", "bl_utilization"]:
            if col not in rows[0]:
                errors.append(f"{tag}: carry_path_step.csv missing column '{col}'")
    else:
        errors.append(f"{tag}: carry_path_step.csv not found")

    # Check carry_path_window.csv
    wpath = os.path.join(d, "carry_path_window.csv")
    if os.path.exists(wpath):
        with open(wpath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            for col in ["VRC_K", "VRR_K", "VRC_eff_K", "VRR_eff_K",
                         "VRC_slow_K", "VRR_slow_K", "G_gamma_K"]:
                if col not in rows[0]:
                    errors.append(f"{tag}: carry_path_window.csv missing column '{col}'")

    # Check carry_path_summary.json
    spath = os.path.join(d, "carry_path_summary.json")
    if os.path.exists(spath):
        with open(spath) as f:
            summary = json.load(f)
        if "ttv1" in tag:
            if "ttv1_gamma_diag" not in summary:
                errors.append(f"{tag}: carry_path_summary.json missing 'ttv1_gamma_diag'")
    else:
        errors.append(f"{tag}: carry_path_summary.json not found")

if errors:
    print(f"\nVALIDATION FAILED ({len(errors)} errors):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"\nVALIDATION PASSED: All {len(tags)} methods produced correct output columns.")
PYEOF

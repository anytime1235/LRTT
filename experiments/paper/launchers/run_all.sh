#!/bin/bash
# Master launcher: runs all experiment phases in sequence.
# Phase 0 determines BEST_LN_LR automatically.
# Edit BEST_* variables after Phase 1 completes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON="${PYTHON:-python}"
export RESULTS_DIR="${RESULTS_DIR:-results/paper}"

# Phase 1 best config (UPDATE AFTER PHASE 1):
export BEST_GAMMA="${BEST_GAMMA:-0.0}"
export BEST_UIM="${BEST_UIM:-false}"
export BEST_TE="${BEST_TE:-24}"

echo "========================================"
echo "  Paper Experiment: Master Launcher"
echo "========================================"
echo "Results root: $RESULTS_DIR"
echo ""

# Phase 0: Smoke + LN LR ablation
echo ">>> Phase 0: Smoke Tests + LN LR Ablation"
RESULTS_DIR="$RESULTS_DIR/phase0" bash "$SCRIPT_DIR/phase0_smoke.sh"
echo ""

# Read best LN LR from Phase 0 ablation
LN_LR_FILE="$RESULTS_DIR/phase0/ln_lr_ablation/best_ln_lr.txt"
if [ -f "$LN_LR_FILE" ]; then
    export BEST_LN_LR="$(cat "$LN_LR_FILE")"
    echo ">>> Phase 0 result: BEST_LN_LR=$BEST_LN_LR"
else
    export BEST_LN_LR="0.016"
    echo ">>> Phase 0 LN LR file not found, defaulting to BEST_LN_LR=0.016"
fi
echo ""

# Phase 1: TTv1 Regime Discovery
echo ">>> Phase 1: TTv1 Regime Discovery"
RESULTS_DIR="$RESULTS_DIR/phase1" bash "$SCRIPT_DIR/phase1_ttv1_grid.sh"
echo ""
echo "!!! REVIEW Phase 1 results and update BEST_GAMMA, BEST_UIM, BEST_TE !!!"
echo "!!! Then re-run this script or continue with phases 2-4 manually.     !!!"
echo ""

# Phase 2: Bit Sweep
echo ">>> Phase 2: Bit Sweep"
RESULTS_DIR="$RESULTS_DIR/phase2" bash "$SCRIPT_DIR/phase2_bit_sweep.sh"
echo ""

# Phase 3: Diagnostics
echo ">>> Phase 3: Diagnostics"
RESULTS_DIR="$RESULTS_DIR/phase3" bash "$SCRIPT_DIR/phase3_diagnostics.sh"
echo ""

# Phase 4: TTv1 Final
echo ">>> Phase 4: TTv1 Final"
RESULTS_DIR="$RESULTS_DIR/phase4" bash "$SCRIPT_DIR/phase4_ttv1_final.sh"
echo ""

echo "========================================"
echo "  All phases complete!"
echo "========================================"

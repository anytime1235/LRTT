#!/usr/bin/env bash
# Orchestrates 4 sequential rank phases for the c-device sweep on a given
# REINIT_MODE. Edits the rank_exp literal in optuna_mlp_mnist_lrtt.py between
# phases and launches sweep_c_device.sh for each.
#
# Run after phase 1 (rank=1) is already running, OR set START_FROM_RANK_EXP=0
# to run all 4 phases.

set -u
cd "$(dirname "$0")/.."   # plots/ → mlp/

REINIT_MODE=${REINIT_MODE:-gauss_b_zero}
SCRIPT=optuna_mlp_mnist_lrtt.py
ORCH_LOG=logs/cdev_sweep/orchestrator_${REINIT_MODE}.log
mkdir -p "$(dirname "$ORCH_LOG")"

START_FROM_RANK_EXP=${START_FROM_RANK_EXP:-2}   # next phase to launch (skip 0; phase 1 already running)

wait_for_phase_done() {
    while [[ $(pgrep -af "$SCRIPT.*--c-device" 2>/dev/null | grep -v /bin/bash | wc -l) -gt 0 ]]; do
        sleep 60
    done
}

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$ORCH_LOG"; }

log "Orchestrator started (REINIT_MODE=$REINIT_MODE, start from rank_exp=$START_FROM_RANK_EXP)"

# Wait for any currently-running phase to complete
log "Waiting for current phase to finish..."
wait_for_phase_done
log "Current phase done"

# Determine remaining phases
ALL_RANKS=(0 2 4 6)
for rank_exp in "${ALL_RANKS[@]}"; do
    if [[ "$rank_exp" -lt "$START_FROM_RANK_EXP" ]]; then
        continue
    fi

    # Edit literal in optuna script
    sed -i -E "s/rank_exp = trial\.suggest_int\('rank_exp', [0-9]+, [0-9]+\)/rank_exp = trial.suggest_int('rank_exp', $rank_exp, $rank_exp)/" "$SCRIPT"
    new_lit=$(grep -oP "trial\.suggest_int\('rank_exp', \K[0-9]+, [0-9]+" "$SCRIPT" | head -1)
    log "Edited script literal → ($new_lit)"

    rank=$((2 ** rank_exp))
    log "Launching phase rank=$rank (rank_exp=$rank_exp)..."
    RANK_EXP=$rank_exp REINIT_MODE=$REINIT_MODE \
      DEVICES="${DEVICES:-}" N_TRIALS=${N_TRIALS:-} TOP_N=${TOP_N:-} \
      ./plots/sweep_c_device.sh >> "$ORCH_LOG" 2>&1
    log "All cells of rank=$rank launched. Waiting for completion..."
    wait_for_phase_done
    log "Phase rank=$rank complete"
done

log "All 4 phases complete for $REINIT_MODE"

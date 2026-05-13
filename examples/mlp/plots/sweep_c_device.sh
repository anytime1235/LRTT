#!/usr/bin/env bash
# C-device sweep: 3 cells (idealizedpreset, reramespreset, ecrampreset) for ONE
# reinit-mode at the rank phase set by RANK_EXP env var.
#
# Workflow per reinit-mode (4 phases):
#   1. Edit optuna_mlp_mnist_lrtt.py: rank_exp = trial.suggest_int('rank_exp', X, X)
#   2. RANK_EXP=X REINIT_MODE=gauss_b_zero ./sweep_c_device.sh
#   3. Wait until done
#   4. Repeat for X in {0, 2, 4, 6}
#
# After all 4 phases of gauss_b_zero done, manually swap a/b reset_std ranges
# in optuna_mlp_mnist_lrtt.py for gauss_a_zero, then run again with
# REINIT_MODE=gauss_a_zero.

set -u
cd "$(dirname "$0")/.."   # plots/ → mlp/
mkdir -p logs/cdev_sweep

GPUS=(0 1 2 3)
N_GPU=${#GPUS[@]}
# Override DEVICES env var with space-separated list, e.g. DEVICES=reramespreset
if [[ -n "${DEVICES:-}" ]]; then
  IFS=' ' read -ra DEVICES <<< "$DEVICES"
else
  DEVICES=(idealizedpreset reramespreset ecrampreset)
fi
REINIT_MODE=${REINIT_MODE:-gauss_b_zero}
RANK_EXP=${RANK_EXP:?must set RANK_EXP env var (0/2/4/6 → rank 1/4/16/64)}
N_TRIALS=${N_TRIALS:-60}
TOP_N=${TOP_N:-10}
RANK=$((2 ** RANK_EXP))

# Sanity check: optuna script literal must match RANK_EXP env var
SCRIPT_RANK=$(grep -oP "trial\.suggest_int\('rank_exp', \K[0-9]+" optuna_mlp_mnist_lrtt.py | head -1)
if [[ "$SCRIPT_RANK" != "$RANK_EXP" ]]; then
  echo "ERROR: optuna_mlp_mnist_lrtt.py has rank_exp literal=$SCRIPT_RANK but RANK_EXP=$RANK_EXP" >&2
  echo "       Edit line 1175 to: rank_exp = trial.suggest_int('rank_exp', $RANK_EXP, $RANK_EXP)" >&2
  exit 1
fi

echo "[c-device sweep] reinit=$REINIT_MODE  rank=$RANK (rank_exp=$RANK_EXP)  devices=${DEVICES[*]}  n_trials=$N_TRIALS  top_n=$TOP_N"
if [[ "$TOP_N" -gt 0 ]]; then
  echo "[warm-start] Enqueuing top-$TOP_N from $REINIT_MODE prior log into ${#DEVICES[@]} cells..."
  for dev in "${DEVICES[@]}"; do
    python plots/enqueue_warmstart_c_device.py \
      --reinit-mode "$REINIT_MODE" \
      --c-device "$dev" \
      --rank-exp "$RANK_EXP" \
      --top-n "$TOP_N"
  done
  echo "[warm-start] Done."
else
  echo "[warm-start] Skipped (TOP_N=0)."
fi
echo

i=0
for dev in "${DEVICES[@]}"; do
  gpu=${GPUS[$((i % N_GPU))]}
  tag="${REINIT_MODE}_${dev}_rk${RANK}"
  log="logs/cdev_sweep/${tag}.out"
  CUDA_VISIBLE_DEVICES=$gpu nohup python optuna_mlp_mnist_lrtt.py \
    --n-trials "$N_TRIALS" \
    --reinit-mode "$REINIT_MODE" \
    --optimizer AnalogSGD \
    --no-wd --no-momentum --no-nesterov \
    --no-scale-transfer-lr \
    --batch-size 64 --epochs 30 \
    --ab-device constantstepideal --c-device "$dev" \
    --is-perfect --transfer-method onehot --lora-target linear1 \
    > "$log" 2>&1 &
  echo "[launch] $tag → GPU $gpu  pid=$!  log=$log"
  i=$((i + 1))
  sleep 0.3
done

echo
echo "Launched $i cells. Monitor: tail -f logs/cdev_sweep/*.out"
echo "When done, edit rank_exp literal in optuna_mlp_mnist_lrtt.py and rerun with next RANK_EXP."

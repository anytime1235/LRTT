#!/usr/bin/env bash
# Launch 25-cell sweep: af_ratio × unr (5×5) for gauss_a_zero, rank=8, te=10.
# Each cell runs an independent Optuna study (50 trials) on a single worker.
# Cells are distributed across GPUs round-robin.

set -u
cd "$(dirname "$0")/.."   # plots/ → mlp/ (so optuna/results paths resolve)
mkdir -p logs/af_unr_sweep

GPUS=(0 1 2 3)
N_GPU=${#GPUS[@]}
AF_VALUES=(0 1 2 5 10)
UNR_VALUES=(0 1 3 5 10)
# Total per cell: TOP_N enqueued (warm-start) + remaining fresh-sampled = N_TRIALS.
N_TRIALS=${N_TRIALS:-60}
TOP_N=${TOP_N:-10}
REINIT_MODE=${REINIT_MODE:-gauss_a_zero}

# Pre-fill each cell with the top TOP_N from the prior (constantstepideal) study
echo "[warm-start] Enqueuing top-${TOP_N} from ${REINIT_MODE} prior log into 25 cells..."
python plots/enqueue_warmstart.py --reinit-mode "$REINIT_MODE" --top-n "$TOP_N"
echo "[warm-start] Done."
echo

i=0
for af in "${AF_VALUES[@]}"; do
  for unr in "${UNR_VALUES[@]}"; do
    gpu=${GPUS[$((i % N_GPU))]}
    tag="af${af}_unr${unr}"
    log="logs/af_unr_sweep/${tag}.out"
    CUDA_VISIBLE_DEVICES=$gpu nohup python optuna_mlp_mnist_lrtt.py \
      --n-trials "$N_TRIALS" \
      --reinit-mode "$REINIT_MODE" \
      --optimizer AnalogSGD \
      --no-wd --no-momentum --no-nesterov \
      --no-scale-transfer-lr \
      --batch-size 64 --epochs 30 \
      --a-device scaledideal --b-device scaledideal --c-device constantstepideal \
      --is-perfect --transfer-method onehot --lora-target linear1 \
      --af-ratio "$af" --unr "$unr" \
      > "$log" 2>&1 &
    echo "[launch] $tag → GPU $gpu  pid=$!  log=$log"
    i=$((i + 1))
    sleep 0.3   # small stagger
  done
done

echo
echo "Launched $i cells. Monitor: tail -f logs/af_unr_sweep/*.out"
echo "GPU util: watch -n2 nvidia-smi"

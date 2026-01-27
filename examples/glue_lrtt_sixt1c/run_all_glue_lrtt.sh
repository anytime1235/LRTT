#!/bin/bash
# Run all GLUE tasks with LRTT 6T1C
# Settings: transfer_every=1000, transfer_lr(analog_lr)=0.01

export WANDB_PROJECT="lrtt_glue"
export MODEL_NAME=bert-base-uncased

cd /root/LRTT/examples/glue_lrtt_sixt1c

# Common parameters
COMMON_ARGS="--model_name_or_path $MODEL_NAME \
  --report_to wandb \
  --logging_steps 50 \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --save_strategy no \
  --lrtt_rank 8 \
  --transfer_every 1000 \
  --lora_alpha 32.0 \
  --forward_inject False \
  --reinit_mode standard \
  --update_mode lora \
  --analog_lr 0.01 \
  --analog_momentum 0.9 \
  --use_wandb True \
  --wandb_project lrtt_glue"

# Task-specific epochs
# Smaller datasets need more epochs
TASKS=("sst2" "cola" "mrpc" "rte" "stsb" "wnli" "qnli" "qqp" "mnli")
EPOCHS=(3 10 10 10 10 10 3 3 3)

mkdir -p logs

echo "========================================"
echo "Starting All GLUE Tasks with LRTT"
echo "Model: $MODEL_NAME"
echo "Transfer Every: 1000"
echo "Analog LR: 0.01"
echo "Tasks: ${TASKS[*]}"
echo "========================================"

for i in "${!TASKS[@]}"; do
  TASK=${TASKS[$i]}
  NUM_EPOCHS=${EPOCHS[$i]}

  echo ""
  echo "========================================"
  echo "Starting Task: $TASK (Epochs: $NUM_EPOCHS)"
  echo "Time: $(date)"
  echo "========================================"

  python run_glue_lrtt.py \
    $COMMON_ARGS \
    --task_name $TASK \
    --num_train_epochs $NUM_EPOCHS \
    --output_dir ./results/$TASK/lrtt_bert_${TASK}_te1000 \
    2>&1 | tee ./logs/${TASK}_lrtt_training.log

  echo "========================================"
  echo "Completed Task: $TASK"
  echo "Time: $(date)"
  echo "========================================"
done

echo ""
echo "========================================"
echo "All LRTT GLUE Tasks Completed!"
echo "Time: $(date)"
echo "========================================"

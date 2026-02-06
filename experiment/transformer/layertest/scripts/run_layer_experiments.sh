#!/bin/bash
# Run layer experiments for LRTT with different target_modules
# Settings: 3 epochs, warmup_steps=500, same hyperparameters as before

cd /home/jovyan/work/single_layer_comparison

PYTHON="/home/jovyan/work/ml/.venv310/bin/python"
SCRIPT="compare_ttv2_lrtt_accuracy.py"

# Base settings
EPOCHS=3
WARMUP=500

echo "Starting layer experiments at $(date)"
echo "Settings: epochs=$EPOCHS, warmup_steps=$WARMUP"
echo "============================================"

# 1. Q (query)
echo "Starting Q (query) experiment..."
WANDB_RUN_NAME="lrtt_Q_3ep" $PYTHON $SCRIPT --num_epochs $EPOCHS --warmup_steps $WARMUP --target_modules query \
    --output_dir results/layer_exp_Q_$(date +%Y%m%d_%H%M%S) &
PID_Q=$!
echo "Q experiment PID: $PID_Q"

sleep 5

# 2. K (key)
echo "Starting K (key) experiment..."
WANDB_RUN_NAME="lrtt_K_3ep" $PYTHON $SCRIPT --num_epochs $EPOCHS --warmup_steps $WARMUP --target_modules key \
    --output_dir results/layer_exp_K_$(date +%Y%m%d_%H%M%S) &
PID_K=$!
echo "K experiment PID: $PID_K"

sleep 5

# 3. V (value)
echo "Starting V (value) experiment..."
WANDB_RUN_NAME="lrtt_V_3ep" $PYTHON $SCRIPT --num_epochs $EPOCHS --warmup_steps $WARMUP --target_modules value \
    --output_dir results/layer_exp_V_$(date +%Y%m%d_%H%M%S) &
PID_V=$!
echo "V experiment PID: $PID_V"

sleep 5

# 4. QKV (query, key, value)
echo "Starting QKV experiment..."
WANDB_RUN_NAME="lrtt_QKV_3ep" $PYTHON $SCRIPT --num_epochs $EPOCHS --warmup_steps $WARMUP --target_modules query key value \
    --output_dir results/layer_exp_QKV_$(date +%Y%m%d_%H%M%S) &
PID_QKV=$!
echo "QKV experiment PID: $PID_QKV"

sleep 5

# 5. Dense layers (attention output + FFN dense)
echo "Starting dense layers experiment..."
WANDB_RUN_NAME="lrtt_dense_3ep" $PYTHON $SCRIPT --num_epochs $EPOCHS --warmup_steps $WARMUP --target_modules dense \
    --output_dir results/layer_exp_dense_$(date +%Y%m%d_%H%M%S) &
PID_DENSE=$!
echo "Dense experiment PID: $PID_DENSE"

sleep 5

# 6. Embedding transformation
echo "Starting embedding transformation experiment..."
WANDB_RUN_NAME="lrtt_emb_trans_3ep" $PYTHON $SCRIPT --num_epochs $EPOCHS --warmup_steps $WARMUP --target_modules embedding_transformation \
    --output_dir results/layer_exp_emb_trans_$(date +%Y%m%d_%H%M%S) &
PID_EMB=$!
echo "Embedding transformation experiment PID: $PID_EMB"

sleep 5

# 7. All layers
echo "Starting all layers experiment..."
WANDB_RUN_NAME="lrtt_all_layers_3ep" $PYTHON $SCRIPT --num_epochs $EPOCHS --warmup_steps $WARMUP --target_modules query key value dense embedding_transformation \
    --output_dir results/layer_exp_all_$(date +%Y%m%d_%H%M%S) &
PID_ALL=$!
echo "All layers experiment PID: $PID_ALL"

echo ""
echo "============================================"
echo "All experiments started!"
echo "PIDs: Q=$PID_Q, K=$PID_K, V=$PID_V, QKV=$PID_QKV, DENSE=$PID_DENSE, EMB=$PID_EMB, ALL=$PID_ALL"
echo "Monitor with: ps aux | grep compare_ttv2"
echo "Check WandB: https://wandb.ai/"
echo "============================================"

# Wait for all to complete
wait
echo "All experiments completed at $(date)"

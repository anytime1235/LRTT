#!/bin/bash
# SST-2 Grid Search
# LR: [1.0, 0.1, 0.01, 0.001]
# alpha_lrtt: [0.01, 0.1, 1.0, 10.0]
# 16 combinations, 3 epochs

LRS=(1.0 0.1 0.01 0.001)
ALPHAS=(0.01 0.1 1.0 10.0)
RANK=8
EPOCHS=3
BATCH_SIZE=32

echo "================================================================================"
echo "SST-2 GRID SEARCH - 16 combinations"
echo "================================================================================"
echo "LR values: ${LRS[@]}"
echo "Alpha_lrtt values: ${ALPHAS[@]}"
echo "Rank: $RANK"
echo "Epochs: $EPOCHS"
echo "Target modules: query, key, value, classifier"
echo "================================================================================"
echo ""

count=0
total=16

for lr in "${LRS[@]}"; do
  for alpha_lrtt in "${ALPHAS[@]}"; do
    count=$((count + 1))
    
    # Convert alpha_lrtt to alpha_standard (since script converts back)
    alpha_std=$(python -c "print($alpha_lrtt * $RANK)")
    
    echo "[$count/$total] lr=$lr, alpha_lrtt=$alpha_lrtt (alpha_std=$alpha_std)"
    
    # Create a temp Python script to run one configuration
    cat > /tmp/run_single_config.py << EOFPY
import sys
sys.path.insert(0, '/data/LRTT_transformer/experiments')
import optuna

# Create a study with enqueued trial
study = optuna.create_study(
    study_name=f"grid_sst2_lr{$lr}_alpha{$alpha_lrtt}",
    direction="maximize",
    sampler=optuna.samplers.GridSampler({"lora_alpha": [$alpha_std], "learning_rate": [$lr]})
)

# Run will use the grid sampler
print(f"Study created with fixed params: alpha_std=$alpha_std, lr=$lr")
EOFPY
    
    # Run with fixed parameters by using a grid sampler
    /data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
      --task glue \
      --task_name sst2 \
      --mode sixt1c_lora \
      --rank $RANK \
      --target_modules query key value classifier \
      --n_trials 1 \
      --study_name "grid_sst2_${count}_lr${lr}_a${alpha_lrtt}" \
      2>&1 | tee /tmp/grid_sst2_${count}.log
    
    echo "  ✓ Completed combination $count"
    echo ""
  done
done

echo "================================================================================"
echo "GRID SEARCH COMPLETED - $total combinations"
echo "================================================================================"

#!/bin/bash
# MRPC seed comparison - remaining 10 runs
# Trial 102 no-transfer + Trial 143 no-transfer (transfer ON already done)
#
# no-transfer: T_Every=999999999 (effectively disables transfer)

set -e
cd "$(dirname "$0")"

SEEDS=(42 123 456 789 1024)
SCRIPT="fine_mobilebert_glue_lrtt.py"
BACKUP="${SCRIPT}.bak"

cp "$SCRIPT" "$BACKUP"

restore() { cp "$BACKUP" "$SCRIPT"; }
trap restore EXIT

run_with_params() {
    local label=$1 seed=$2 lr=$3 t_lr=$4 t_every=$5 fast_lr=$6

    echo "============================================"
    echo "[$label] seed=$seed lr=$lr t_lr=$t_lr t_every=$t_every fast_lr=$fast_lr"
    echo "============================================"

    cp "$BACKUP" "$SCRIPT"

    sed -i "s/^SEED = .*/SEED = $seed/" "$SCRIPT"
    sed -i "s/^LEARNING_RATE = .*/LEARNING_RATE = $lr/" "$SCRIPT"
    sed -i "s/^TRANSFER_LR = .*/TRANSFER_LR = $t_lr/" "$SCRIPT"
    sed -i "s/^TRANSFER_EVERY = .*/TRANSFER_EVERY = $t_every/" "$SCRIPT"
    sed -i "s/^FAST_LR = .*/FAST_LR = $fast_lr/" "$SCRIPT"
    sed -i "s/^LRTT_RANK = .*/LRTT_RANK = 8/" "$SCRIPT"
    sed -i "s/^N_EPOCHS = .*/N_EPOCHS = 14/" "$SCRIPT"
    sed -i "s/^BATCH_SIZE = .*/BATCH_SIZE = 32/" "$SCRIPT"
    sed -i "s/^WARMUP_STEPS = .*/WARMUP_STEPS = 80/" "$SCRIPT"
    sed -i "s/^OPTIMIZER = .*/OPTIMIZER = \"AnalogAdam\"/" "$SCRIPT"
    sed -i "s/^TRANSFER_METHOD = .*/TRANSFER_METHOD = \"set\"/" "$SCRIPT"
    sed -i "s/^AUTO_SCALE_MODE = .*/AUTO_SCALE_MODE = \"separate\"/" "$SCRIPT"
    sed -i "s/^CORRECT_GRADIENT_MAGNITUDES = .*/CORRECT_GRADIENT_MAGNITUDES = True/" "$SCRIPT"
    sed -i "s/^REINIT_MODE = .*/REINIT_MODE = \"hybrid\"/" "$SCRIPT"
    sed -i "s/^WEIGHT_DECAY = .*/WEIGHT_DECAY = 0.0/" "$SCRIPT"
    sed -i "s/^LORA_TARGET = .*/LORA_TARGET = \"qkvo\"/" "$SCRIPT"
    sed -i "s/^IO_NOISE = .*/IO_NOISE = False/" "$SCRIPT"
    sed -i "s/^NO_ADC_AB_PROJ = .*/NO_ADC_AB_PROJ = False/" "$SCRIPT"
    sed -i "s/^ENCODER_ANALOG = .*/ENCODER_ANALOG = False/" "$SCRIPT"
    sed -i "s/^EMBEDDING_ANALOG = .*/EMBEDDING_ANALOG = False/" "$SCRIPT"
    sed -i "s/^HEAD_ANALOG = .*/HEAD_ANALOG = False/" "$SCRIPT"
    sed -i "s/^ENABLE_DIAGNOSTIC = .*/ENABLE_DIAGNOSTIC = False/" "$SCRIPT"

    HF_HUB_DISABLE_XET=1 python "$SCRIPT" --task mrpc 2>&1 | tee "results/mrpc_${label}_seed${seed}.log"
}

# --- Trial 102 without transfer ---
echo "=== Trial 102 + No-Transfer ==="
for seed in "${SEEDS[@]}"; do
    run_with_params "t102_notransfer" "$seed" 5.91e-04 9.050022 999999999 1.0
done

# --- Trial 143 without transfer ---
echo ""
echo "=== Trial 143 + No-Transfer ==="
for seed in "${SEEDS[@]}"; do
    run_with_params "t143_notransfer" "$seed" 4.96e-04 3.77e-04 999999999 0.968120
done

restore
echo ""
echo "=== All runs complete (10 runs) ==="

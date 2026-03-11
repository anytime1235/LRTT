#!/bin/bash
# 18 combinations: 3 (lr, fast_lr) × 3 transfer_every × 2 transfer_desired_bl
# All share: dw_min=0.0005, desired_bl=31, no-uim, exclude-ffn, steps=100, eval-loss

SCRIPT="diag_weight_update_bert_v2.py"
COMMON="--mode tiki --steps 100 --dw-min 0.0005 --desired-bl 31 \
  --no-uim --exclude-ffn --eval-loss --eval-every 1 --overwrite"

# (lr, fast_lr, transfer_lr) combinations — all have lr*fast_lr = 0.01
LR_CONFIGS=(
    "0.01 1.0 1.0"
    "0.1 0.1 10.0"
    "1.0 0.01 100.0"
)

TE_VALUES=(8 80 800)
TBL_VALUES=(1 31)

for lr_cfg in "${LR_CONFIGS[@]}"; do
    read -r LR FAST_LR TRANSFER_LR <<< "$lr_cfg"
    for TE in "${TE_VALUES[@]}"; do
        for TBL in "${TBL_VALUES[@]}"; do
            echo "=== lr=$LR fast_lr=$FAST_LR te=$TE tbl=$TBL ==="
            python "$SCRIPT" $COMMON \
                --lr "$LR" --fast-lr "$FAST_LR" --transfer-lr "$TRANSFER_LR" \
                --transfer-every "$TE" --transfer-desired-bl "$TBL"
            echo ""
        done
    done
done

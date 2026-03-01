#!/bin/bash
# Wait for sweep (PID 424238) to finish, then run full epoch training

echo "Waiting for MNLI Adam sweep (PID 424238) to finish..."
while kill -0 424238 2>/dev/null; do
    sleep 60
done
echo "Sweep finished at $(date)"

# Check if sweep checkpoint exists
CKPT="/data/classifier_ckpt/mnli_adam/ckpt.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Sweep checkpoint not found at $CKPT"
    exit 1
fi

echo "Starting full epoch training..."
/data/venvs/lrtt/bin/python /data/pretrain_classifier_full.py \
    --task mnli \
    --optimizer adam \
    --epochs 4

echo "Full epoch training done at $(date)"

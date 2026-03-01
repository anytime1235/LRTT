#!/bin/bash
# Wait for QQP+QNLI sweep (PID 558130) to finish, then run full epoch training sequentially

SWEEP_PID=558130

echo "Waiting for QQP+QNLI Adam sweep (PID $SWEEP_PID) to finish..."
while kill -0 $SWEEP_PID 2>/dev/null; do
    sleep 60
done
echo "Sweep finished at $(date)"

# --- QQP full epoch training ---
CKPT_QQP="/data/classifier_ckpt/qqp_adam/ckpt.pt"
if [ ! -f "$CKPT_QQP" ]; then
    echo "ERROR: QQP sweep checkpoint not found at $CKPT_QQP"
    exit 1
fi
echo ""
echo "========== Starting QQP full epoch training (5 epochs) =========="
/data/venvs/lrtt/bin/python /data/pretrain_classifier_full.py \
    --task qqp \
    --optimizer adam \
    --epochs 5
echo "QQP full epoch training done at $(date)"

# --- QNLI full epoch training ---
CKPT_QNLI="/data/classifier_ckpt/qnli_adam/ckpt.pt"
if [ ! -f "$CKPT_QNLI" ]; then
    echo "ERROR: QNLI sweep checkpoint not found at $CKPT_QNLI"
    exit 1
fi
echo ""
echo "========== Starting QNLI full epoch training (11 epochs) =========="
/data/venvs/lrtt/bin/python /data/pretrain_classifier_full.py \
    --task qnli \
    --optimizer adam \
    --epochs 11
echo "QNLI full epoch training done at $(date)"

echo ""
echo "All done at $(date)"

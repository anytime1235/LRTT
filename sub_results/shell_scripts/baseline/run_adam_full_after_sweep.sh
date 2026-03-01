#!/bin/bash
# Wait for Adam sweep to finish, then run full epoch training for all tasks
# Tasks: RTE -> CoLA -> STS-B -> MRPC -> SST-2 (sequential)
#
# Usage:
#   nohup bash /data/run_adam_full_after_sweep.sh > /data/results/tikitakav1/adam_full_all.log 2>&1 &

set -e

PYTHON=/data/venvs/lrtt/bin/python
SCRIPT=/data/pretrain_classifier_full.py
SWEEP_PID=1156678

echo "============================================"
echo "Stage 0-Full: Adam full epoch training"
echo "Tasks: RTE -> CoLA -> STS-B -> MRPC -> SST-2"
echo "Waiting for sweep PID ${SWEEP_PID} to finish..."
echo "Started: $(date)"
echo "============================================"

# Wait for sweep to complete
while kill -0 ${SWEEP_PID} 2>/dev/null; do
    sleep 60
done
echo "Sweep finished at $(date)"
echo ""

# Verify all sweep checkpoints exist
for TASK in rte cola stsb mrpc sst2; do
    CKPT="/data/classifier_ckpt/${TASK}_adam/ckpt.pt"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Sweep checkpoint not found at $CKPT"
        exit 1
    fi
    echo "OK: $CKPT"
done
echo ""

echo "[1/5] RTE starting: $(date)"
$PYTHON $SCRIPT --task rte --optimizer adam
echo "[1/5] RTE finished: $(date)"
echo ""

echo "[2/5] CoLA starting: $(date)"
$PYTHON $SCRIPT --task cola --optimizer adam
echo "[2/5] CoLA finished: $(date)"
echo ""

echo "[3/5] STS-B starting: $(date)"
$PYTHON $SCRIPT --task stsb --optimizer adam
echo "[3/5] STS-B finished: $(date)"
echo ""

echo "[4/5] MRPC starting: $(date)"
$PYTHON $SCRIPT --task mrpc --optimizer adam
echo "[4/5] MRPC finished: $(date)"
echo ""

echo "[5/5] SST-2 starting: $(date)"
$PYTHON $SCRIPT --task sst2 --optimizer adam
echo "[5/5] SST-2 finished: $(date)"
echo ""

echo "============================================"
echo "All 5 tasks completed: $(date)"
echo "============================================"

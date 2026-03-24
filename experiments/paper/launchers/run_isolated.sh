#!/bin/bash
# Run a single paper_experiment.py in isolation.
# Usage: run_isolated.sh <TAG> <PHASE> <FLAGS...>
set -uo pipefail

TAG="$1"
PHASE="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/.venv310/bin/python}"

echo "[$PHASE] START $TAG $(date)"

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py "$@"
RUN_EXIT=$?

# Force cleanup
$PYTHON -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache()" 2>/dev/null

if [ $RUN_EXIT -ne 0 ]; then
    echo "[$PHASE] FAIL  $TAG (exit=$RUN_EXIT) $(date)"
else
    echo "[$PHASE] DONE  $TAG $(date)"
fi

sleep 3

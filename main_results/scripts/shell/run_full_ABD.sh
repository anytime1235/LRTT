#!/bin/bash
# Sequential full run: 1) paper_figures.py (SQuAD) 2) paper_figures_glue.py (GLUE)
#
# Output structure (no overwrites — skip logic for existing CSVs):
#   SQuAD -> /data/main_results/results/squad/seed_{42,43,44}/
#   GLUE  -> /data/main_results/results/glue/{task}/seed_{42,43,44}/
#   Aggregates -> .../squad/aggregate/  and  .../glue/aggregate/
#
# Figure C is deprecated — only A/B/D are generated.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUT_BASE="/data/main_results/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${OUT_BASE}/run_full_ABD_${TIMESTAMP}.log"

echo "========================================" | tee "$LOG"
echo " Full Run: SQuAD (ABD) + GLUE (ABDEF)" | tee -a "$LOG"
echo " Seeds: 42,43,44" | tee -a "$LOG"
echo " OUT_DIR: ${OUT_BASE}" | tee -a "$LOG"
echo " Started: $(date)" | tee -a "$LOG"
echo " Log: $LOG" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# -------------------------------------------------------
# 1) paper_figures.py (SQuAD, seeds=42,43,44, Figures ABD)
# -------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[1/2] paper_figures.py --figures ABD --seeds 42,43,44" | tee -a "$LOG"
echo "  Out: ${OUT_BASE}/squad/seed_{42,43,44}/" | tee -a "$LOG"
echo "  Start: $(date)" | tee -a "$LOG"

/data/venvs/lrtt/bin/python paper_figures.py \
  --figures ABD \
  --seeds 42,43,44 \
  --run-tag v3_ABD \
  --out-dir "$OUT_BASE" \
  2>&1 | tee -a "$LOG"

echo "[1/2] SQuAD Done: $(date)" | tee -a "$LOG"

# -------------------------------------------------------
# 2) paper_figures_glue.py (GLUE 7 tasks x 3 seeds, ABD+EF)
# -------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[2/2] paper_figures_glue.py --figures ABDEF --seeds 42,43,44" | tee -a "$LOG"
echo "  Out: ${OUT_BASE}/glue/{task}/seed_{seed}/" | tee -a "$LOG"
echo "  Start: $(date)" | tee -a "$LOG"

/data/venvs/lrtt/bin/python paper_figures_glue.py \
  --tasks cola,rte,mrpc,mnli,sst2,stsb,qqp,qnli \
  --seeds 42,43,44 \
  --figures ABDEF \
  --run-tag glue_v1 \
  --out-dir "$OUT_BASE" \
  2>&1 | tee -a "$LOG"

echo "[2/2] GLUE Done: $(date)" | tee -a "$LOG"

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo " All done: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- SQuAD per-seed ---" | tee -a "$LOG"
for d in ${OUT_BASE}/squad/seed_*/; do
  seed=$(basename "$d")
  n_csv=$(find "$d" -name "*.csv" | wc -l)
  n_png=$(find "$d" -name "*.png" | wc -l)
  n_npz=$(find "$d" -name "*.npz" | wc -l)
  echo "  $seed: ${n_csv} CSVs, ${n_png} PNGs, ${n_npz} NPZs" | tee -a "$LOG"
done

echo "--- SQuAD aggregate ---" | tee -a "$LOG"
ls ${OUT_BASE}/squad/aggregate/ 2>/dev/null | tee -a "$LOG"

echo "--- GLUE per-task ---" | tee -a "$LOG"
for d in ${OUT_BASE}/glue/*/; do
  task=$(basename "$d")
  if [ "$task" != "aggregate" ]; then
    n_seeds=$(find "$d" -maxdepth 1 -type d -name "seed_*" | wc -l)
    n_files=$(find "$d" -type f | wc -l)
    echo "  $task: ${n_seeds} seeds, ${n_files} files" | tee -a "$LOG"
  fi
done

echo "--- GLUE aggregate ---" | tee -a "$LOG"
ls ${OUT_BASE}/glue/aggregate/ 2>/dev/null | tee -a "$LOG"

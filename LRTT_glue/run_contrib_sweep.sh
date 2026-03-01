#!/bin/bash
# Run each config as separate process to isolate GPU memory
set -e
cd /data/LRTT_transformer/LRTT_glue

OUTDIR="/tmp/contrib_sweep"
mkdir -p $OUTDIR

# Configs: ab_type alpha lr
# Key: lr_eff = lr * alpha. Previous best: sixt1c a=0.02 lr=0.002 (lr_eff=4e-5)
CONFIGS=(
    # FP A/B: alpha sweep, lr adjusted to keep lr_eff=4e-5
    "fp 0.1  0.0004"
    "fp 1.0  0.00004"
    "fp 8.0  0.000005"
    # sixt1c A/B: same alpha/lr for direct comparison
    "sixt1c 0.02 0.002"
    "sixt1c 0.1  0.0004"
    "sixt1c 1.0  0.00004"
    "sixt1c 8.0  0.000005"
)

echo "========================================"
echo "  FP vs sixt1c Contribution Sweep"
echo "  All configs have lr_eff ≈ 4e-5"
echo "========================================"

for cfg in "${CONFIGS[@]}"; do
    read -r ab alpha lr <<< "$cfg"
    tag="${ab}_a${alpha}_lr${lr}"
    outf="${OUTDIR}/${tag}.json"
    echo ""
    echo ">>> Running: $ab alpha=$alpha lr=$lr (lr_eff=$(python3 -c "print(f'{$alpha*$lr:.6f}')") )"
    python run_single_contrib.py "$ab" "$alpha" "$lr" "$outf" 2>&1 | tail -10
    echo "--- saved to $outf"
done

# Summary
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
python3 -c "
import json, glob, os
files = sorted(glob.glob('${OUTDIR}/*.json'))
print(f\"  {'Type':>7} {'Alpha':>6} {'LR':>10} {'lr_eff':>10} | {'F1':>7} {'Stat':>8} | {'InitR':>10} {'Ep1R':>10} {'Ep2R':>10} {'Ep3R':>10} | {'FinalC':>8} {'FinalAB':>8}\")
print(f\"  {'-'*115}\")
for f in files:
    with open(f) as fh: r = json.load(fh)
    eps = r.get('epochs', [])
    rats = [e['ratio'] for e in eps] if eps else [0]*4
    while len(rats) < 4: rats.append(rats[-1] if rats else 0)
    cn = eps[-1]['c_norm'] if eps else 0
    an = eps[-1]['ab_norm'] if eps else 0
    print(f\"  {r['ab_type']:>7} {r['alpha']:>6.2f} {r['lr']:>10.6f} {r['lr_eff']:>10.6f} | {r.get('best_f1',0):>7.4f} {r['status']:>8} | {rats[0]:>10.6f} {rats[1]:>10.6f} {rats[2]:>10.6f} {rats[3]:>10.6f} | {cn:>8.2f} {an:>8.4f}\")
"

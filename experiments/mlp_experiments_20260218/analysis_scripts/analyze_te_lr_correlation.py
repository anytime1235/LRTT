#!/usr/bin/env python3
"""Analyze correlation between TE and lr/tlr across all ranks."""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Load best configs
with open('/root/LRTT/experiments/mnist_sweep_analysis/lrtt_sweep_best_configs.json', 'r') as f:
    configs = json.load(f)

print("=" * 80)
print("TE vs LR/TLR CORRELATION ANALYSIS - ALL RANKS")
print("=" * 80)

def analyze_mode(mode_name, mode_data):
    """Analyze TE vs lr/tlr correlation for a given mode."""
    print(f"\n{'=' * 80}")
    print(f"{mode_name.upper()} MODE ANALYSIS")
    print(f"{'=' * 80}")

    all_tes = []
    all_lrs = []
    all_tlrs = []
    all_accs = []
    all_ranks = []

    for rank_str, rank_data in mode_data['by_rank_te'].items():
        rank = int(rank_str)

        print(f"\n--- Rank {rank} ---")
        print(f"{'TE':<6} {'LR':<10} {'TLR':<10} {'Acc':<8} {'LR/TLR ratio':<12}")
        print("-" * 60)

        rank_tes = []
        rank_lrs = []
        rank_tlrs = []
        rank_accs = []

        for te_str, data in sorted(rank_data.items(), key=lambda x: int(x[0])):
            te = int(te_str)

            if isinstance(data, dict):
                lr = data.get('lr', 0)
                tlr = data.get('tlr', 0)
                acc = data.get('best_acc', 0)
            else:
                acc = data
                # Try to get lr/tlr from HYPERPARAMETERS in decay mode
                lr = tlr = 0

            if lr > 0 and tlr > 0:
                ratio = lr / tlr
                rank_tes.append(te)
                rank_lrs.append(lr)
                rank_tlrs.append(tlr)
                rank_accs.append(acc)

                all_tes.append(te)
                all_lrs.append(lr)
                all_tlrs.append(tlr)
                all_accs.append(acc)
                all_ranks.append(rank)

                print(f"{te:<6} {lr:<10.5f} {tlr:<10.5f} {acc:<8.2f} {ratio:<12.3f}")

        if len(rank_tes) >= 3:
            # Calculate correlations for this rank
            corr_te_lr, _ = pearsonr(rank_tes, rank_lrs)
            corr_te_tlr, _ = pearsonr(rank_tes, rank_tlrs)
            corr_lr_tlr, _ = pearsonr(rank_lrs, rank_tlrs)

            print(f"\nRank {rank} Correlations:")
            print(f"  TE  vs LR:  {corr_te_lr:+.3f}")
            print(f"  TE  vs TLR: {corr_te_tlr:+.3f}")
            print(f"  LR  vs TLR: {corr_lr_tlr:+.3f}")

    # Overall correlations
    if len(all_tes) >= 3:
        print(f"\n{'=' * 80}")
        print(f"OVERALL {mode_name.upper()} CORRELATIONS (all ranks combined)")
        print(f"{'=' * 80}")

        corr_te_lr, p_te_lr = pearsonr(all_tes, all_lrs)
        corr_te_tlr, p_te_tlr = pearsonr(all_tes, all_tlrs)
        corr_lr_tlr, p_lr_tlr = pearsonr(all_lrs, all_tlrs)

        print(f"TE  vs LR:  {corr_te_lr:+.3f} (p={p_te_lr:.4f})")
        print(f"TE  vs TLR: {corr_te_tlr:+.3f} (p={p_te_tlr:.4f})")
        print(f"LR  vs TLR: {corr_lr_tlr:+.3f} (p={p_lr_tlr:.4f})")

        # Spearman correlation (rank-based, better for non-linear relationships)
        sp_te_lr, _ = spearmanr(all_tes, all_lrs)
        sp_te_tlr, _ = spearmanr(all_tes, all_tlrs)
        sp_lr_tlr, _ = spearmanr(all_lrs, all_tlrs)

        print(f"\nSpearman (rank-based) correlations:")
        print(f"TE  vs LR:  {sp_te_lr:+.3f}")
        print(f"TE  vs TLR: {sp_te_tlr:+.3f}")
        print(f"LR  vs TLR: {sp_lr_tlr:+.3f}")

        # Statistical summary
        print(f"\nValue ranges:")
        print(f"  TE:  {min(all_tes)} - {max(all_tes)}")
        print(f"  LR:  {min(all_lrs):.5f} - {max(all_lrs):.5f} (span: {max(all_lrs)/min(all_lrs):.1f}x)")
        print(f"  TLR: {min(all_tlrs):.5f} - {max(all_tlrs):.5f} (span: {max(all_tlrs)/min(all_tlrs):.1f}x)")

        return {
            'tes': all_tes,
            'lrs': all_lrs,
            'tlrs': all_tlrs,
            'accs': all_accs,
            'ranks': all_ranks,
            'corr_te_lr': corr_te_lr,
            'corr_te_tlr': corr_te_tlr,
            'corr_lr_tlr': corr_lr_tlr
        }

    return None

# Analyze both modes
hybrid_results = analyze_mode("Hybrid", configs['hybrid_sweep'])
decay_results = analyze_mode("Decay", configs['decay_sweep'])

# Pattern analysis
print(f"\n{'=' * 80}")
print("PATTERN ANALYSIS & RECOMMENDATIONS")
print(f"{'=' * 80}")

print("\n1. TE vs LR relationship:")
if hybrid_results and decay_results:
    if abs(hybrid_results['corr_te_lr']) < 0.3 and abs(decay_results['corr_te_lr']) < 0.3:
        print("   ✗ WEAK correlation - LR does NOT strongly depend on TE")
        print("   → Current interpolation may not be optimal")
    else:
        print(f"   ✓ Moderate correlation found")
        print(f"     Hybrid: {hybrid_results['corr_te_lr']:+.3f}")
        print(f"     Decay:  {decay_results['corr_te_lr']:+.3f}")

print("\n2. TE vs TLR relationship:")
if hybrid_results and decay_results:
    if abs(hybrid_results['corr_te_tlr']) < 0.3 and abs(decay_results['corr_te_tlr']) < 0.3:
        print("   ✗ WEAK correlation - TLR does NOT strongly depend on TE")
        print("   → TLR may need different strategy")
    else:
        print(f"   ✓ Moderate correlation found")
        print(f"     Hybrid: {hybrid_results['corr_te_tlr']:+.3f}")
        print(f"     Decay:  {decay_results['corr_te_tlr']:+.3f}")

print("\n3. LR vs TLR relationship:")
if hybrid_results and decay_results:
    print(f"   Hybrid: {hybrid_results['corr_lr_tlr']:+.3f}")
    print(f"   Decay:  {decay_results['corr_lr_tlr']:+.3f}")
    if abs(hybrid_results['corr_lr_tlr']) < 0.3 and abs(decay_results['corr_lr_tlr']) < 0.3:
        print("   → LR and TLR are largely independent")
    else:
        print("   → Some coupling between LR and TLR exists")

print("\n" + "=" * 80)
print("RECOMMENDATION FOR IMPROVED LR/TLR SELECTION")
print("=" * 80)
print("""
Based on the correlation analysis:

1. **If TE vs LR/TLR correlation is weak:**
   - Instead of interpolating, use a fixed set of "good" values
   - Or use a heuristic: smaller LR for larger TE (due to more frequent updates)
   - Example: LR ~ 1/sqrt(TE) or LR ~ constant for TE ranges

2. **If you want TE-dependent LR/TLR:**
   - Observe the pattern: does LR increase or decrease with TE?
   - Use piecewise functions or binning instead of smooth interpolation
   - Consider rank-dependent formulas

3. **Current approach (interpolation) works IF:**
   - Data shows clear trends (correlation > 0.5)
   - You have enough data points across TE range
   - The relationship is approximately log-linear

4. **Alternative: Grid search around empirical best**
   - For each TE, use the closest tested TE's hyperparameters
   - Add small perturbations (±20%) as trials
""")

# Save analysis results
output = {
    'hybrid': hybrid_results if hybrid_results else {},
    'decay': decay_results if decay_results else {}
}

with open('/root/te_lr_correlation_analysis.json', 'w') as f:
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    json.dump(convert(output), f, indent=2)

print(f"\nAnalysis saved to: /root/te_lr_correlation_analysis.json")

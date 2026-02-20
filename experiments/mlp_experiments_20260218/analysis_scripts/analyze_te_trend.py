#!/usr/bin/env python3
"""Analyze TE (Transfer Every) trend from rank8_nn_hybrid results"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('/root/results/rank8_nn_hybrid/results_final.json', 'r') as f:
    results = json.load(f)

# Extract TE and best_acc
te_values = []
best_accs = []

for result in results:
    te_values.append(result['te'])
    best_accs.append(result['best_acc'])

# Sort by TE
sorted_indices = np.argsort(te_values)
te_sorted = np.array(te_values)[sorted_indices]
acc_sorted = np.array(best_accs)[sorted_indices]

# Analyze trends
print("=" * 70)
print("TRANSFER EVERY (TE) TREND ANALYSIS")
print("=" * 70)
print()

# Find ranges
very_small_te = [(te, acc) for te, acc in zip(te_sorted, acc_sorted) if te < 40]
optimal_te = [(te, acc) for te, acc in zip(te_sorted, acc_sorted) if 40 <= te <= 680]
large_te = [(te, acc) for te, acc in zip(te_sorted, acc_sorted) if te > 680]

print("1. VERY FREQUENT TRANSFER (TE < 40):")
print(f"   Range: TE={very_small_te[0][0]}-{very_small_te[-1][0]}")
print(f"   Accuracy: {min([a for _, a in very_small_te]):.2f}% - {max([a for _, a in very_small_te]):.2f}%")
print(f"   Average: {np.mean([a for _, a in very_small_te]):.2f}%")
print(f"   Stability: High variance (some failures at TE=27: 12.19%)")
print()

print("2. OPTIMAL RANGE (40 ≤ TE ≤ 680): ⭐")
print(f"   Range: TE={optimal_te[0][0]}-{optimal_te[-1][0]}")
print(f"   Accuracy: {min([a for _, a in optimal_te]):.2f}% - {max([a for _, a in optimal_te]):.2f}%")
print(f"   Average: {np.mean([a for _, a in optimal_te]):.2f}%")
print(f"   Top performers:")
for te, acc in sorted(optimal_te, key=lambda x: x[1], reverse=True)[:5]:
    print(f"      TE={te:4d}: {acc:.2f}%")
print()

print("3. INFREQUENT TRANSFER (TE > 680):")
print(f"   Range: TE={large_te[0][0]}-{large_te[-1][0]}")
print(f"   Accuracy: {min([a for _, a in large_te]):.2f}% - {max([a for _, a in large_te]):.2f}%")
print(f"   Average: {np.mean([a for _, a in large_te]):.2f}%")
print(f"   Trend: Performance degradation")
print()

# Find global best
best_idx = np.argmax(acc_sorted)
print(f"🏆 GLOBAL BEST: TE={te_sorted[best_idx]}, Accuracy={acc_sorted[best_idx]:.2f}%")
print()

# Statistical analysis
print("=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)
print()
print(f"Overall range: {min(acc_sorted):.2f}% - {max(acc_sorted):.2f}%")
print(f"Overall mean: {np.mean(acc_sorted):.2f}%")
print(f"Overall std: {np.std(acc_sorted):.2f}%")
print()

# Create visualization
plt.figure(figsize=(14, 8))

# Plot 1: TE vs Accuracy (linear scale)
plt.subplot(2, 1, 1)
plt.plot(te_sorted, acc_sorted, 'o-', linewidth=2, markersize=6)
plt.axhline(y=95, color='r', linestyle='--', alpha=0.3, label='95% threshold')
plt.axvline(x=40, color='g', linestyle='--', alpha=0.3, label='Optimal range start')
plt.axvline(x=680, color='g', linestyle='--', alpha=0.3, label='Optimal range end')

# Highlight best
plt.scatter([te_sorted[best_idx]], [acc_sorted[best_idx]],
           color='red', s=200, marker='*', zorder=5, label=f'Best: TE={te_sorted[best_idx]}')

plt.xlabel('Transfer Every (TE)', fontsize=12)
plt.ylabel('Best Accuracy (%)', fontsize=12)
plt.title('Transfer Frequency (TE) vs Accuracy - Rank 8 NN Hybrid', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: TE vs Accuracy (log scale for better visualization)
plt.subplot(2, 1, 2)
plt.semilogx(te_sorted, acc_sorted, 'o-', linewidth=2, markersize=6)
plt.axhline(y=95, color='r', linestyle='--', alpha=0.3, label='95% threshold')
plt.axvspan(40, 680, alpha=0.2, color='green', label='Optimal range')

# Highlight best
plt.scatter([te_sorted[best_idx]], [acc_sorted[best_idx]],
           color='red', s=200, marker='*', zorder=5, label=f'Best: TE={te_sorted[best_idx]}')

plt.xlabel('Transfer Every (TE) [log scale]', fontsize=12)
plt.ylabel('Best Accuracy (%)', fontsize=12)
plt.title('Transfer Frequency (TE) vs Accuracy - Log Scale', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('/root/results/te_trend_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved: /root/results/te_trend_analysis.png")
print()

# Key findings
print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print()
print("1. SWEET SPOT: TE ∈ [40, 680]")
print("   - Consistent high performance (95%+)")
print("   - Best results at TE=60, 120")
print()
print("2. TOO FREQUENT (TE < 40):")
print("   - Moderate performance (92-94%)")
print("   - Risk of instability (TE=27 failed)")
print("   - Hypothesis: AB doesn't learn enough between transfers")
print()
print("3. TOO INFREQUENT (TE > 680):")
print("   - Performance degradation (90-93%)")
print("   - Hypothesis: AB accumulates too much, C-AB imbalance")
print()
print("4. RECOMMENDATION:")
print("   Use TE ∈ [50, 200] for optimal performance")
print("   Specifically: TE=60, 120 are excellent choices")
print()

"""Figure 6(b): Rank sweep — onehot vs set, with trainable parameter ratio."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# BERT-base: 12 layers, each attention has Q/K/V/O (768x768)
# LRTT trainable params per layer: A(768,r) + B(r,768) = 2*768*r
# QKVO = 4 sublayers * 12 layers = 48 analog layers
# + LayerNorm params: 38,400 (always trainable)
# Full trainable (digital fine-tuning): ~109M total, but let's use actual number
# From diagnostic: Total params=80,581,778, Trainable=40,082 at rank=32
# Trainable = 48 * 2 * 768 * r + 38400 (LayerNorm)
# At r=32: 48*2*768*32 + 38400 = 2,359,296 + 38,400 = 2,397,696... but diagnostic says 40,082
# Wait, that's the number of trainable parameters in the optimizer
# Actually LRTT A/B tiles are not standard trainable params — they're analog
# The 40,082 is LayerNorm + qa_outputs (2*768+2 = 1538) + embedding layernorm
# So "trainable" in standard sense is just the digital params

# For the ratio, we compare LRTT effective trainable rank against full-rank:
# QKVO trainable in full fine-tuning: 48 * 768 * 768 + LayerNorm(38400) + qa_outputs(1538) = 28,351,490
# LRTT: 48 * 2 * 768 * r + LayerNorm(38400) + qa_outputs(1538)
DIGITAL_PARAMS = 39938  # LayerNorm + qa_outputs (shared between full FT and LR-TT)
FULL_PARAMS = 48 * 768 * 768 + DIGITAL_PARAMS  # 28,351,490
def lrtt_params(r):
    return 48 * 2 * 768 * r + DIGITAL_PARAMS

ranks = [1, 2, 4, 8, 16, 32, 64]

onehot_f1 = {1: 82.76, 2: 82.92, 4: 83.81, 8: 84.19, 16: 84.70, 32: 84.98, 64: 84.94}
set_f1 =    {1: 82.39, 2: 83.23, 4: 83.45, 8: 83.90, 16: 84.97, 32: 84.79, 64: 84.56}
param_ratio = {r: lrtt_params(r) / FULL_PARAMS * 100 for r in ranks}

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(5.5, 3.5))

ax.plot(ranks, [onehot_f1[r] for r in ranks], 'o-', color='#1f77b4', linewidth=1.6,
        markersize=6, label='LR-TT (onehot)', zorder=3)
ax.plot(ranks, [set_f1[r] for r in ranks], 's--', color='#ff7f0e', linewidth=1.6,
        markersize=6, label='LR-TT (set)', zorder=3)

for r in ranks:
    ax.annotate(f'{onehot_f1[r]:.2f}', (r, onehot_f1[r]),
                textcoords='offset points', xytext=(0, 7),
                fontsize=7, ha='center', color='#1f77b4')
    oy = -12 if r != 16 else -18
    ax.annotate(f'{set_f1[r]:.2f}', (r, set_f1[r]),
                textcoords='offset points', xytext=(0, oy),
                fontsize=7, ha='center', color='#ff7f0e')

ax.set_xscale('log', base=2)
ax.set_xticks(ranks)
ax.set_xticklabels([str(r) for r in ranks])
ax.minorticks_off()

# Parameter ratio on secondary x-axis
ax2 = ax.twiny()
ax2.set_xscale('log', base=2)
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(ranks)
ax2.set_xticklabels([f'{param_ratio[r]:.1f}%' for r in ranks], fontsize=8)
ax2.minorticks_off()
ax2.set_xlabel('Trainable params (% of full-rank)', fontsize=9)
ax.set_xlabel('Rank')
ax.set_ylabel('F1 score')
ax.set_ylim(76, 87)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9, edgecolor='0.7')
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/fig6b_rank_sweep.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')
for r in ranks:
    print(f"  rank={r}: params={lrtt_params(r):,} ({param_ratio[r]:.2f}% of full)")

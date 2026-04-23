"""Figure S10: Rank sweep — AF ratio = 1 (6T1C-gamma) condition, with trainable parameter ratio."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIGITAL_PARAMS = 39938
FULL_PARAMS = 48 * 768 * 768 + DIGITAL_PARAMS

def lrtt_params(r):
    return 48 * 2 * 768 * r + DIGITAL_PARAMS

ranks = [1, 2, 4, 8, 16, 32, 64]
onehot_f1 = {1: 82.02, 2: 81.97, 4: 82.41, 8: 82.76, 16: 83.42, 32: 84.06, 64: 84.49}
param_ratio = {r: lrtt_params(r) / FULL_PARAMS * 100 for r in ranks}

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(5.5, 3.5))

ax.plot(ranks, [onehot_f1[r] for r in ranks], 'o-', color='#1f77b4', linewidth=1.6,
        markersize=6, label='LR-TT (opt. at AF ratio = 1, onehot)', zorder=3)

for r in ranks:
    ax.annotate(f'{onehot_f1[r]:.2f}', (r, onehot_f1[r]),
                textcoords='offset points', xytext=(0, 7),
                fontsize=7, ha='center', color='#1f77b4')

ax.set_xscale('log', base=2)
ax.set_xticks(ranks)
ax.set_xticklabels([str(r) for r in ranks])
ax.minorticks_off()

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

OUT = '/root/LRTT/examples/bert/results/plots/figS10_rank_sweep_af1.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')
for r in ranks:
    print(f"  rank={r}: params={lrtt_params(r):,} ({param_ratio[r]:.2f}% of full)")

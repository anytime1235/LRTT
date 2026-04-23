"""Figure S6: A/B (auxiliary) tile bit sweep — AF ratio = 1 (6T1C-gamma), per-bit optimized."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = [
    {'bits': 6,  'f1': 78.00, 'trial': 'T588', 'trials_run': 12},
    {'bits': 8,  'f1': 79.50, 'trial': 'T601', 'trials_run': 11},
    {'bits': 10, 'f1': 83.21, 'trial': 'T419', 'trials_run': 1},
    {'bits': 12, 'f1': 84.06, 'trial': 'T249', 'trials_run': 92},
    {'bits': 14, 'f1': 83.63, 'trial': 'T31',  'trials_run': 10},
]

bits = [d['bits'] for d in data]
f1s  = [d['f1']  for d in data]

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(bits, f1s, 'o-', color='#1f77b4', linewidth=1.6, markersize=6, zorder=3,
        label='LR-TT (opt. at AF ratio = 1, core tile 10-bit)')

for b, f in zip(bits, f1s):
    ax.annotate(f'{f:.2f}', (b, f), textcoords='offset points', xytext=(0, 7),
                fontsize=8, ha='center')

ax.set_xlabel('Auxiliary tile weight bits')
ax.set_ylabel('F1 score')
ax.set_xticks(bits)
ax.set_xlim(min(bits) - 1, max(bits) + 1)
ax.set_ylim(0, 90)
ax.legend(loc='lower right', fontsize=8, framealpha=0.9, edgecolor='0.7')
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/figS6_ab_bit_sweep_af1.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

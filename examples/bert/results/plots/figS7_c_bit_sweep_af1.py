"""Figure S7: Core tile (C) bit sweep — AF ratio = 1 (6T1C-gamma), fixed condition."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = [
    {'bits': 6,  'f1': 80.33},
    {'bits': 7,  'f1': 80.78},
    {'bits': 8,  'f1': 83.31},
    {'bits': 9,  'f1': 83.47},
    {'bits': 10, 'f1': 84.06},
]

bits = [d['bits'] for d in data]
f1s  = [d['f1']  for d in data]

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(bits, f1s, 's-', color='#ff7f0e', linewidth=1.6, markersize=6, zorder=3,
        label='LR-TT (opt. at AF ratio = 1, aux. tile 12-bit)')

for b, f in zip(bits, f1s):
    ax.annotate(f'{f:.2f}', (b, f), textcoords='offset points', xytext=(0, 7),
                fontsize=8, ha='center')

ax.set_xlabel('Core tile weight bits')
ax.set_ylabel('F1 score')
ax.set_xticks(bits)
ax.set_xlim(min(bits) - 0.5, max(bits) + 0.5)
ax.set_ylim(76, 87)
ax.legend(loc='lower right', fontsize=8, framealpha=0.9, edgecolor='0.7')
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/figS7_c_bit_sweep_af1.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

"""Figure 5(h): A/B (auxiliary) tile bit sweep — per-bit optimized, constantstepideal + Tiki-Taka baseline."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = [
    {'bits': 6,  'f1': 83.63, 'trial': 'T322'},
    {'bits': 8,  'f1': 84.98, 'trial': 'T290'},
    {'bits': 10, 'f1': 84.98, 'trial': 'T267'},
    {'bits': 12, 'f1': 82.16, 'trial': 'T242'},
    {'bits': 14, 'f1': 78.72, 'trial': 'T347'},
]

tikitaka = [
    {'bits': 6,  'f1': 83.4},
    {'bits': 8,  'f1': 86.2},
    {'bits': 10, 'f1': 86.9},
    {'bits': 12, 'f1': 86.0},
    {'bits': 14, 'f1': 85.1},
]

bits = [d['bits'] for d in data]
f1s  = [d['f1']   for d in data]
tt_bits = [d['bits'] for d in tikitaka]
tt_f1s  = [d['f1']   for d in tikitaka]

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(tt_bits, tt_f1s, 'D-', color='#9467bd', linewidth=1.6, markersize=6, zorder=3,
        label='Tiki-Taka')
ax.plot(bits, f1s, 'o-', color='#1f77b4', linewidth=1.6, markersize=6, zorder=3,
        label='LR-TT (core tile 10-bit)')

for b, f in zip(tt_bits, tt_f1s):
    ax.annotate(f'{f:.1f}', (b, f),
                textcoords='offset points', xytext=(0, 7),
                fontsize=8, ha='center', color='#9467bd')

for b, f in zip(bits, f1s):
    ax.annotate(f'{f:.1f}', (b, f),
                textcoords='offset points', xytext=(0, -14),
                fontsize=8, ha='center', color='#1f77b4')

ax.legend(loc='lower left', fontsize=8, framealpha=0.9, edgecolor='0.7')
ax.set_xlabel('Auxiliary tile weight bits')
ax.set_ylabel('F1 score')
ax.set_xticks(bits)
ax.set_xlim(bits[0] - 0.5, bits[-1] + 0.5)
ax.set_ylim(76, 90)
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/fig5h_ab_bit_sweep.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

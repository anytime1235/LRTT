"""Figure 5(g): Core tile (C) bit sweep — fixed condition, constantstepideal + Single RPU + Tiki-Taka baseline."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = [
    {'bits': 6,  'f1': 82.7, 'trial': 'T359'},
    {'bits': 7,  'f1': 83.4, 'trial': 'T360'},
    {'bits': 8,  'f1': 84.8, 'trial': 'T474'},
    {'bits': 9,  'f1': 84.8, 'trial': 'T475'},
    {'bits': 10, 'f1': 85.0, 'trial': 'T267'},
]

single_rpu = [
    {'bits': 6,  'f1': 11.6},
    {'bits': 7,  'f1': 17.4},
    {'bits': 8,  'f1': 58.2},
    {'bits': 9,  'f1': 75.9},
    {'bits': 10, 'f1': 82.6},
]

tikitaka = [
    {'bits': 6,  'f1': 85.2},
    {'bits': 7,  'f1': 85.9},
    {'bits': 8,  'f1': 86.3},
    {'bits': 9,  'f1': 87.0},
    {'bits': 10, 'f1': 86.9},
]

bits = [d['bits'] for d in data]
f1s  = [d['f1']   for d in data]
sr_bits = [d['bits'] for d in single_rpu]
sr_f1s  = [d['f1']   for d in single_rpu]
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
ax.plot(bits, f1s, 's-', color='#ff7f0e', linewidth=1.6, markersize=6, zorder=3,
        label='LR-TT (aux. tile 10-bit)')
ax.plot(sr_bits, sr_f1s, '^--', color='#d62728', linewidth=1.6, markersize=6, zorder=3,
        label='Single RPU')

for b, f in zip(tt_bits, tt_f1s):
    ax.annotate(f'{f:.1f}', (b, f),
                textcoords='offset points', xytext=(0, 7),
                fontsize=8, ha='center', color='#9467bd')

for b, f in zip(bits, f1s):
    ax.annotate(f'{f:.1f}', (b, f),
                textcoords='offset points', xytext=(0, -14),
                fontsize=8, ha='center', color='#ff7f0e')

for b, f in zip(sr_bits, sr_f1s):
    oy = -14
    ax.annotate(f'{f:.1f}', (b, f),
                textcoords='offset points', xytext=(0, oy),
                fontsize=8, ha='center', color='#d62728')

ax.legend(loc='lower right', fontsize=8, framealpha=0.9, edgecolor='0.7')
ax.set_xlabel('Core tile weight bits')
ax.set_ylabel('F1 score')
all_bits = sorted(set(bits + sr_bits))
ax.set_xticks(all_bits)
ax.set_xlim(min(all_bits) - 0.5, max(all_bits) + 0.5)
ax.set_ylim(0, 90)
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/fig5g_c_bit_sweep.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

"""Figure 6(c): Noise ratio and asymmetry (gamma) ratio sweeps.
Two device conditions compared: 6T1C-gamma (T249) and multi-level ideal (T267).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# T267: constantstepideal + abml=10 (ideal symmetric, multi-level)
noise_t267 = [
    {'ratio': 0.0, 'f1': 78.9},
    {'ratio': 0.1, 'f1': 79.1},
    {'ratio': 0.3, 'f1': 78.8},
    {'ratio': 0.5, 'f1': 78.8},
    {'ratio': 0.7, 'f1': 78.7},
    {'ratio': 1.0, 'f1': 79.1},
]
gamma_t267 = [
    {'ratio': 0.0, 'f1': 85.0},
    {'ratio': 0.5, 'f1': 80.0},
    {'ratio': 1.0, 'f1': 78.9},
    {'ratio': 2.0, 'f1': 78.4},
    {'ratio': 3.0, 'f1': 78.5},
    {'ratio': 5.0, 'f1': 78.5},
    {'ratio': 10.0, 'f1': 78.4},
]

# T249: 6T1C-gamma device (inherent asymmetry, AF ratio=1 is native)
noise_t249 = [
    {'ratio': 0.0, 'f1': 84.1},
    {'ratio': 0.1, 'f1': 83.8},
    {'ratio': 0.3, 'f1': 84.2},
    {'ratio': 0.5, 'f1': 82.9},
    {'ratio': 0.7, 'f1': 80.7},
    {'ratio': 1.0, 'f1': 78.1},
]
gamma_t249 = [
    {'ratio': 0.0, 'f1': 7.3},
    {'ratio': 0.5, 'f1': 7.3},
    {'ratio': 1.0, 'f1': 84.1},
    {'ratio': 2.0, 'f1': 82.1},
    {'ratio': 3.0, 'f1': 81.4},
    {'ratio': 5.0, 'f1': 80.3},
    {'ratio': 10.0, 'f1': 79.4},
]

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

# --- Left: Noise ratio sweep (AF=1 optimized only) ---
nr = [d['ratio'] for d in noise_t249]
nf = [d['f1'] for d in noise_t249]
ax1.plot(nr, nf, 's-', color='#d62728', linewidth=1.6, markersize=6, zorder=3,
         label='LR-TT')
for r, f in zip(nr, nf):
    ax1.annotate(f'{f:.1f}', (r, f), textcoords='offset points', xytext=(0, 7),
                 fontsize=7, ha='center', color='#d62728')

ax1.set_xlabel('Noise scaling ratio')
ax1.set_ylabel('F1 score')
ax1.set_ylim(76, 87)
ax1.text(0.97, 0.03, 'Fixed: AF ratio = 1', transform=ax1.transAxes,
         fontsize=7.5, ha='right', va='bottom', color='0.4')
ax1.legend(fontsize=8, loc='lower left', framealpha=0.9, edgecolor='0.7')
ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

# --- Right: Gamma (AF) ratio sweep ---
for data, color, marker, label in [
    (gamma_t249, '#d62728', 's', 'LR-TT (opt. at AF ratio = 1)'),
    (gamma_t267, '#1f77b4', 'o', 'LR-TT (opt. at AF ratio = 0)'),
]:
    x = [d['ratio'] for d in data]
    y = [d['f1'] for d in data]
    ax2.plot(x, y, f'{marker}-', color=color, linewidth=1.6, markersize=6, zorder=3,
             label=label)
    for r, f in zip(x, y):
        if f < 10:
            ax2.annotate(f'{f:.1f}', (r, f), textcoords='offset points', xytext=(0, 7),
                         fontsize=7, ha='center', color=color)
        else:
            oy = 7 if label.endswith('ratio = 1)') else -12
            if label.endswith('ratio = 0)') and r == 0.5:
                oy = 7
            ax2.annotate(f'{f:.1f}', (r, f), textcoords='offset points', xytext=(0, oy),
                         fontsize=7, ha='center', color=color)

ax2.set_xlabel('Asymmetry factor (AF) ratio')
ax2.set_ylabel('F1 score')
ax2.set_ylim(0, 90)
ax2.text(0.97, 0.12, 'Fixed: noise ratio = 0.0', transform=ax2.transAxes,
         fontsize=7.5, ha='right', va='bottom', color='0.4')
ax2.legend(fontsize=8, loc='center right', framealpha=0.9, edgecolor='0.7')
ax2.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/fig6c_noise_asymmetry.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

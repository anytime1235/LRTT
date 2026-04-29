"""Figure 6(c): Noise ratio and asymmetry (gamma) ratio sweeps.
Two device conditions compared: 6T1C-gamma (T249) and multi-level ideal (T267).
Left panel additionally compares two AF=1 optimization conditions:
  - T249 (optimized at noise ratio = 0)
  - T98  (optimized at noise ratio = 1, the full 6T1C operating point)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Common plot points (option A): match gamma sweep ratios for visual alignment.
NOISE_PLOT_RATIOS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

# T267: constantstepideal + abml=10 (ideal symmetric, multi-level) — full data preserved in JSON
noise_t267 = [
    {'ratio': 0.0, 'f1': 78.91},
    {'ratio': 0.1, 'f1': 79.08},
    {'ratio': 0.3, 'f1': 78.79},
    {'ratio': 0.5, 'f1': 78.83},
    {'ratio': 0.7, 'f1': 78.71},
    {'ratio': 1.0, 'f1': 79.11},
]
gamma_t267 = [
    {'ratio': 0.0, 'f1': 84.98},
    {'ratio': 0.5, 'f1': 80.03},
    {'ratio': 1.0, 'f1': 78.91},
    {'ratio': 2.0, 'f1': 78.44},
    {'ratio': 3.0, 'f1': 78.51},
    {'ratio': 5.0, 'f1': 78.48},
    {'ratio': 10.0, 'f1': 78.36},
]

# T249: 6T1C-gamma base, optimized at noise_ratio=0 (clean device)
noise_t249 = [
    {'ratio': 0.0, 'f1': 84.06},
    {'ratio': 0.1, 'f1': 83.76},
    {'ratio': 0.3, 'f1': 84.20},
    {'ratio': 0.5, 'f1': 82.90},
    {'ratio': 0.7, 'f1': 80.74},
    {'ratio': 1.0, 'f1': 78.12},
    {'ratio': 2.0, 'f1': 7.27},
    {'ratio': 3.0, 'f1': 7.27},
    {'ratio': 5.0, 'f1': 7.27},
    {'ratio': 10.0, 'f1': 7.27},
]
gamma_t249 = [
    {'ratio': 0.0, 'f1': 7.27},
    {'ratio': 0.5, 'f1': 7.27},
    {'ratio': 1.0, 'f1': 84.06},
    {'ratio': 2.0, 'f1': 82.11},
    {'ratio': 3.0, 'f1': 81.43},
    {'ratio': 5.0, 'f1': 80.26},
    {'ratio': 10.0, 'f1': 79.37},
]

# T98: 6T1C base, optimized at noise_ratio=1 (full 6T1C operating point)
noise_t98 = [
    {'ratio': 0.0,  'f1': 82.13},
    {'ratio': 0.1,  'f1': 81.93},
    {'ratio': 0.3,  'f1': 81.87},
    {'ratio': 0.5,  'f1': 82.26},
    {'ratio': 0.7,  'f1': 82.25},
    {'ratio': 1.0,  'f1': 82.24},  # native training condition
    {'ratio': 2.0,  'f1': 76.82},
    {'ratio': 3.0,  'f1': 73.44},
    {'ratio': 5.0,  'f1': 9.50},
    {'ratio': 10.0, 'f1': 7.27},
]


def _select(data, ratios):
    """Return list of (ratio, f1) filtered to the requested ratios."""
    table = {d['ratio']: d['f1'] for d in data}
    return [(r, table[r]) for r in ratios if r in table]


plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

# --- Left: Noise ratio sweep (two AF=1 optimization conditions) ---
for data, color, marker, label in [
    (noise_t98,  '#1f77b4', 'o', 'LR-TT (opt. at noise ratio = 1)'),
    (noise_t249, '#d62728', 's', 'LR-TT (opt. at noise ratio = 0)'),
]:
    pts = _select(data, NOISE_PLOT_RATIOS)
    x = [r for r, _ in pts]
    y = [f for _, f in pts]
    ax1.plot(x, y, f'{marker}-', color=color, linewidth=1.6, markersize=6, zorder=3,
             label=label)
    for r, f in zip(x, y):
        oy = 7 if label.endswith('= 1)') else -12
        ax1.annotate(f'{f:.1f}', (r, f), textcoords='offset points', xytext=(0, oy),
                     fontsize=7, ha='center', color=color)

ax1.set_xlabel('Noise scaling ratio')
ax1.set_ylabel('F1 score')
ax1.set_ylim(0, 90)
ax1.set_xscale('symlog', linthresh=0.5, linscale=0.5)
_xt = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
ax1.set_xticks(_xt)
ax1.set_xticklabels([str(t) for t in _xt])
ax1.set_xlim(-0.05, 11.0)
ax1.minorticks_off()
ax1.text(0.97, 0.12, 'Fixed: AF ratio = 1', transform=ax1.transAxes,
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
ax2.set_xscale('symlog', linthresh=0.5, linscale=0.5)
_xt2 = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
ax2.set_xticks(_xt2)
ax2.set_xticklabels([str(t) for t in _xt2])
ax2.set_xlim(-0.05, 11.0)
ax2.minorticks_off()
ax2.text(0.97, 0.12, 'Fixed: noise ratio = 0.0', transform=ax2.transAxes,
         fontsize=7.5, ha='right', va='bottom', color='0.4')
ax2.legend(fontsize=8, loc='center right', framealpha=0.9, edgecolor='0.7')
ax2.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)

fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/fig6c_noise_asymmetry.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

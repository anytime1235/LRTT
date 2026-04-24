"""Figure S9: Gamma ratio × w_max 2D sweep heatmaps — AF ratio = 0 vs AF ratio = 1."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gamma_ratios = [0.0, 0.5, 1.0, 2.0, 5.0]
w_max_vals = [0.0312, 0.0624, 0.125, 0.25, 0.5, 1.0]

# T267: opt. at AF ratio = 0 (constantstepideal, abml=10)
f1_t267 = np.array([
    [84.42, 84.98, 80.00, 57.76,  8.99,  7.27],
    [79.07, 79.99, 81.16, 82.62, 83.55, 75.10],
    [78.46, 79.09, 79.89, 80.99, 82.59, 82.38],
    [78.59, 78.53, 78.77, 79.51, 80.88, 82.37],
    [78.24, 78.41, 78.45, 78.49, 79.23, 80.62],
])

# T249: opt. at AF ratio = 1 (6T1C-gamma, abml=12)
f1_t249 = np.array([
    [67.00, 10.56,  7.27,  7.27,  7.30,  7.36],
    [68.25, 70.82, 70.97,  7.27,  7.27,  7.66],
    [72.23, 61.22, 73.24, 70.01, 78.37, 84.06],
    [77.69, 75.17, 78.83, 79.77, 80.80, 82.25],
    [78.49, 78.36, 78.42, 78.77, 79.13, 80.35],
])

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.0))

vmin, vmax = 0, 90

opt_cells = {
    'af0': (0, 1),   # T267: gamma=0.0, w_max=0.0624
    'af1': (2, 5),   # T249: gamma=1.0, w_max=1.0
}

for ax, data, title, opt_key in [
    (ax1, f1_t267, '(a) Opt. at AF ratio = 0\n(constantstepideal, abml=10)', 'af0'),
    (ax2, f1_t249, '(b) Opt. at AF ratio = 1\n(6T1C-gamma, abml=12)', 'af1'),
]:
    im = ax.imshow(data, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(w_max_vals)))
    ax.set_xticklabels([str(w) for w in w_max_vals])
    ax.set_yticks(range(len(gamma_ratios)))
    ax.set_yticklabels([str(g) for g in gamma_ratios])
    ax.set_xlabel('$w_{max}$')
    ax.set_ylabel('AF ratio')
    ax.set_title(title, fontsize=9.5)

    for i in range(len(gamma_ratios)):
        for j in range(len(w_max_vals)):
            v = data[i, j]
            color = 'white' if v < 40 else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='bold')

    oi, oj = opt_cells[opt_key]
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((oj - 0.5, oi - 0.5), 1, 1,
                           linewidth=2.5, edgecolor='blue', facecolor='none', zorder=5))

fig.subplots_adjust(left=0.06, right=0.88, wspace=0.3)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('F1 score', fontsize=10)

OUT = '/root/LRTT/examples/bert/results/plots/figS9_gamma_wmax_2d.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

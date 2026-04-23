"""Figure S9: F1 comparison across LoRA targets — AF ratio = 1 (6T1C-gamma) condition."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

methods = ['LR-TT\n(opt. at AF ratio = 1)']

data = {
    'LR-TT\n(opt. at AF ratio = 1)': [84.61, 82.04, 83.55],
}

targets = ['QKVO', 'FFN', 'ALL']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

n_methods = len(methods)
n_targets = len(targets)
bar_width = 0.22
x = np.arange(n_methods)

fig, ax = plt.subplots(figsize=(4.0, 3.5))

for j, (target, color) in enumerate(zip(targets, colors)):
    offset = (j - (n_targets - 1) / 2) * bar_width
    vals = [data[m][j] for m in methods]
    bars = ax.bar(x + offset, vals, bar_width, label=target, color=color, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)

ax.set_ylabel('F1 score')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(76, 87)
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='0.7')
ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/figS9_target_comparison_af1.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

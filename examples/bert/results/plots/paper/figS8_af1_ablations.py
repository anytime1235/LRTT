"""Figure S8: AF ratio = 1 ablations — (a) bit sweeps, (b) rank sweep, (c) target comparison."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

# --- (a) Bit sweeps: AB and C ---
ax = axes[0]
ab_data = [
    {'bits': 6,  'f1': 78.00, 'trial': 'T588'},
    {'bits': 8,  'f1': 79.50, 'trial': 'T601'},
    {'bits': 10, 'f1': 83.21, 'trial': 'T419'},
    {'bits': 12, 'f1': 84.06, 'trial': 'T249'},
    {'bits': 14, 'f1': 84.11, 'trial': 'T638'},
]
c_data = [
    {'bits': 6,  'f1': 80.33, 'trial': 'T436'},
    {'bits': 7,  'f1': 80.78, 'trial': 'T425'},
    {'bits': 8,  'f1': 83.31, 'trial': 'T426'},
    {'bits': 9,  'f1': 83.47, 'trial': 'T427'},
    {'bits': 10, 'f1': 84.06, 'trial': 'T249'},
]

ab_bits = [d['bits'] for d in ab_data]
ab_f1 = [d['f1'] for d in ab_data]
c_bits = [d['bits'] for d in c_data]
c_f1 = [d['f1'] for d in c_data]

ax.plot(ab_bits, ab_f1, 'o-', color='#1f77b4', linewidth=1.6, markersize=6, zorder=3,
        label='Aux. tile sweep\n(core 10-bit fixed)')
ax.plot(c_bits, c_f1, 's-', color='#ff7f0e', linewidth=1.6, markersize=6, zorder=3,
        label='Core tile sweep\n(aux. 12-bit fixed)')

for b, f in zip(ab_bits, ab_f1):
    ax.annotate(f'{f:.1f}', (b, f), textcoords='offset points', xytext=(0, 7),
                fontsize=7, ha='center', color='#1f77b4')
for b, f in zip(c_bits, c_f1):
    ax.annotate(f'{f:.1f}', (b, f), textcoords='offset points', xytext=(0, -12),
                fontsize=7, ha='center', color='#ff7f0e')

ax.set_xlabel('Weight bits')
ax.set_ylabel('F1 score')
ax.set_ylim(76, 87)
ax.legend(fontsize=7, loc='lower right', framealpha=0.9, edgecolor='0.7')
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
ax.set_title('(a) Bit sweep', fontsize=10)

# --- (b) Rank sweep ---
ax = axes[1]
rank_data = [
    {'rank': 1,  'f1': 82.02, 'trial': 'T256'},
    {'rank': 2,  'f1': 81.97, 'trial': 'T229'},
    {'rank': 4,  'f1': 82.42, 'trial': 'T284'},
    {'rank': 8,  'f1': 82.76, 'trial': 'T316'},
    {'rank': 16, 'f1': 83.42, 'trial': 'T289'},
    {'rank': 32, 'f1': 84.06, 'trial': 'T249'},
    {'rank': 64, 'f1': 84.49, 'trial': 'T292'},
]
ranks = [d['rank'] for d in rank_data]
rank_f1 = [d['f1'] for d in rank_data]

ax.plot(ranks, rank_f1, 'o-', color='#1f77b4', linewidth=1.6, markersize=6, zorder=3,
        label='LR-TT (opt. at AF ratio = 1, rank-wise)')
for r, f in zip(ranks, rank_f1):
    ax.annotate(f'{f:.1f}', (r, f), textcoords='offset points', xytext=(0, 7),
                fontsize=7, ha='center')

ax.set_xscale('log', base=2)
ax.set_xticks(ranks)
ax.set_xticklabels([str(r) for r in ranks])
ax.minorticks_off()
ax.set_xlabel('Rank')
ax.set_ylabel('F1 score')
ax.set_ylim(76, 87)
ax.legend(fontsize=7, loc='lower right', framealpha=0.9, edgecolor='0.7')
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
ax.set_title('(b) Rank sweep', fontsize=10)

# --- (c) Target comparison ---
ax = axes[2]
targets = ['QKVO', 'FFN', 'ALL']
f1_values = [84.61, 82.51, 83.55]
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']
bar_width = 0.22
x = np.arange(1)

for j, (target, color, f1) in enumerate(zip(targets, colors_bar, f1_values)):
    offset = (j - 1) * bar_width
    bars = ax.bar(x + offset, [f1], bar_width, label=target, color=color, zorder=3)
    ax.text(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height() + 0.3,
            f'{f1:.1f}', ha='center', va='bottom', fontsize=7.5)

ax.set_ylabel('F1 score')
ax.set_xticks(x)
ax.set_xticklabels(['LR-TT\n(opt. at AF ratio = 1)'])
ax.set_ylim(76, 87)
ax.legend(fontsize=8, framealpha=0.9, edgecolor='0.7')
ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
ax.set_title('(c) Target comparison', fontsize=10)

fig.tight_layout(pad=0.8)

OUT = '/root/LRTT/examples/bert/results/plots/figS8_af1_ablations.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

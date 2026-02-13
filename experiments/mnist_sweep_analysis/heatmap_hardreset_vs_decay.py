import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

# Load data
with open('/data/results/mnist/lrtt_sweep_best_configs.json') as f:
    data = json.load(f)

ranks = [1, 4, 8, 16, 32, 64]
tes = [1, 10, 50, 100, 500, 1000]

# Extract hybrid (hard reset) accuracy matrix
hybrid_by_rank_te = data['hybrid_sweep']['by_rank_te']
hybrid_matrix = np.zeros((len(ranks), len(tes)))
for i, r in enumerate(ranks):
    for j, te in enumerate(tes):
        entry = hybrid_by_rank_te[str(r)][str(te)]
        if isinstance(entry, dict):
            hybrid_matrix[i, j] = entry['best_acc']
        else:
            hybrid_matrix[i, j] = entry

# Extract decay accuracy matrix
decay_by_rank_te = data['decay_sweep']['by_rank_te']
decay_matrix = np.zeros((len(ranks), len(tes)))
for i, r in enumerate(ranks):
    for j, te in enumerate(tes):
        entry = decay_by_rank_te[str(r)][str(te)]
        if isinstance(entry, dict):
            decay_matrix[i, j] = entry['best_acc']
        else:
            decay_matrix[i, j] = entry

# Common color range
vmin = min(hybrid_matrix.min(), decay_matrix.min())
vmax = max(hybrid_matrix.max(), decay_matrix.max())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

te_labels = [str(t) for t in tes]
rank_labels = [str(r) for r in ranks]

for ax, matrix, title in zip(axes, [hybrid_matrix, decay_matrix],
                              ['Hard Reset (A=0)', 'Decay (A & B decay)']):
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(tes)))
    ax.set_xticklabels(te_labels)
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels(rank_labels)
    ax.set_xlabel('TE (Training Epochs)', fontsize=13)
    ax.set_ylabel('Rank', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Annotate cells
    for i in range(len(ranks)):
        for j in range(len(tes)):
            val = matrix[i, j]
            color = 'white' if val < 92 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

fig.suptitle('MNIST LRTT Sweep: Best Accuracy by Rank × TE', fontsize=15, fontweight='bold', y=1.02)
cbar = fig.colorbar(im, ax=axes, shrink=0.8, label='Best Accuracy (%)')

plt.tight_layout()
plt.savefig('/data/results/mnist/heatmap_hardreset_vs_decay.png', dpi=150, bbox_inches='tight')
plt.savefig('/data/results/mnist/heatmap_hardreset_vs_decay.pdf', bbox_inches='tight')
print("Saved: heatmap_hardreset_vs_decay.png / .pdf")

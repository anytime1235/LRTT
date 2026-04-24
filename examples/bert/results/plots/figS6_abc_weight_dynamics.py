"""Figure S6: A, B, C tile individual cell weight dynamics (10 cells each).

From T267 diagnostic (constantstepideal, abml=10, rank=32).
First tile = Layer 0, query. Last tile = Layer 11, attention output.
NOTE: code A = paper B, code B = paper A.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIAG_PATH = '/root/LRTT/examples/bert/results/BERT_SQUAD_LRTT_FINE/squad_diagnostic_log_te2_r32_onehot.json'
with open(DIAG_PATH) as f:
    d = json.load(f)

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
})

tile_configs = [
    ('first_tile', 'Layer 0, query'),
    ('last_tile', 'Layer 11, attention output'),
]

# code A = paper B, code B = paper A
cell_configs = [
    ('A_cells', 'Tile B (aux.)'),
    ('B_cells', 'Tile A (aux.)'),
    ('C_cells', 'Tile C (core)'),
]

fig, axes = plt.subplots(3, 2, figsize=(10, 8))

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for col, (tile_key, tile_label) in enumerate(tile_configs):
    steps_data = d[tile_key]['steps']
    all_steps = [s['step'] for s in steps_data]
    stride = max(1, len(all_steps) // 1000)

    for row, (cell_key, cell_label) in enumerate(cell_configs):
        ax = axes[row, col]
        cells = np.array([s[cell_key] for s in steps_data])

        for i in range(cells.shape[1]):
            ax.plot(all_steps[::stride], cells[::stride, i],
                    color=colors[i], linewidth=0.8, alpha=0.8)

        ax.set_ylabel(cell_label)
        if row == 0:
            ax.set_title(tile_label)
        if row == 2:
            ax.set_xlabel('Training step')
        ax.set_xlim(all_steps[0], all_steps[-1])
        ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))


fig.tight_layout(pad=0.8)

OUT = '/root/LRTT/examples/bert/results/plots/figS6_abc_weight_dynamics.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

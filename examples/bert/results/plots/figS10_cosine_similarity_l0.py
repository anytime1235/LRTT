"""Figure S10: Cosine similarity — Layer 0 query (supplementary)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIAG_PATH = '/root/LRTT/examples/bert/results/BERT_SQUAD_LRTT_FINE/squad_diagnostic_log_te2_r32_onehot.json'
with open(DIAG_PATH) as f:
    d = json.load(f)

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

tiles = [
    ('first_tile', 'Layer 0, query'),
]

win = 200

fig, axes = plt.subplots(1, 1, figsize=(5.0, 3.2))
axes = [axes]

for ax, (tile_key, tile_label) in zip(axes, tiles):
    tile = d[tile_key]

    # cos(τ·AB, -G) every step (scale-invariant, same as cos(AB,-G))
    all_steps = [r['step'] for r in tile['steps'] if 'cos_AB_G' in r]
    cos_tlrAB_nG = [-r['cos_AB_G'] for r in tile['steps'] if 'cos_AB_G' in r]

    # Transfer-step only metrics
    xfer = [r for r in tile['steps'] if r.get('is_transfer')]
    xfer_steps = [r['step'] for r in xfer]
    cos_dC_tlr = [r.get('cos_dC_tlrAB', 0) for r in xfer]

    # Scatter raw (faint)
    ax.scatter(all_steps, cos_tlrAB_nG, s=1, alpha=0.06, color='#2ca02c', zorder=1)
    ax.scatter(xfer_steps, cos_dC_tlr, s=1, alpha=0.06, color='#9467bd', zorder=1)

    # Moving averages
    if len(cos_tlrAB_nG) >= win:
        ma = np.convolve(cos_tlrAB_nG, np.ones(win)/win, mode='valid')
        ax.plot(all_steps[win-1:], ma, color='#2ca02c', linewidth=1.8,
                label=r'$\cos(\tau \cdot AB,\, -G)$', zorder=3)

    if len(cos_dC_tlr) >= win:
        ma = np.convolve(cos_dC_tlr, np.ones(win)/win, mode='valid')
        ax.plot(xfer_steps[win-1:], ma, color='#9467bd', linewidth=1.8,
                label=r'$\cos(\Delta C,\, \tau \cdot AB)$', zorder=3)

    # Loss on twin axis (moving average)
    loss_steps = [r['step'] for r in tile['steps']]
    losses = [r.get('loss', None) for r in tile['steps']]
    losses = [l if l is not None else np.nan for l in losses]
    ax2 = ax.twinx()
    ax2.scatter(loss_steps, losses, s=0.3, alpha=0.08, color='gray', zorder=0)
    if len(losses) >= win:
        ma_loss = np.convolve(losses, np.ones(win)/win, mode='valid')
        ax2.plot(loss_steps[win-1:], ma_loss, color='gray', alpha=0.5, linewidth=1.4, zorder=0)
    ax2.set_ylabel('Training loss', color='gray', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)

    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Training step')
    ax.set_title(tile_label)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

axes[0].set_ylabel('Cosine similarity')
handles, labels = axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
handles.append(Line2D([0], [0], color='gray', alpha=0.4, linewidth=1))
labels.append('Training loss')
axes[0].legend(handles, labels, fontsize=7.5, loc='upper right', framealpha=0.9, edgecolor='0.7')

fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/figS10_cosine_similarity_l0.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')

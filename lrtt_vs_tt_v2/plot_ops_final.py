#!/usr/bin/env python3
"""OPs breakdown: grouped bars (not stacked), TT as reference line.
Focused y-axis on LR-TT range, TT annotated separately.
"""

import os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
COST_MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "cost_model")
sys.path.insert(0, COST_MODEL_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    RANKS, TARGETS, BATCH_SIZE, S_PAD_DEFAULT,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7,
    'axes.linewidth': 0.5,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 200,
})

C_PROJ   = '#c0392b'
C_UPDATE = '#e67e22'
C_VIS    = '#2980b9'


def compute_ops(targeted, rank, gamma):
    lr = {'proj': 0, 'update': 0, 'visible': 0}
    tt = {'proj': 0, 'update': 0, 'visible': 0}
    for l in targeted:
        M, N = l.M, l.N
        lr['proj']   += BT * 2 * (rank * N + M * rank)
        lr['update'] += BT * 2 * (M * rank + rank * N)
        if gamma == 1:
            lr['visible'] += BT * 2 * (rank * N + M * rank)
        tt['update'] += BT * 2 * M * N
        if gamma == 1:
            tt['visible'] += BT * 2 * M * N
    return lr, tt


def main():
    inventory = build_layer_inventory()

    for gamma in [0, 1]:
        if gamma == 0:
            comp_keys = ['proj', 'update']
            comp_names = ['Projection', 'Update']
            comp_colors = [C_PROJ, C_UPDATE]
        else:
            comp_keys = ['proj', 'update', 'visible']
            comp_names = ['Projection', 'Update', 'Visible fwd']
            comp_colors = [C_PROJ, C_UPDATE, C_VIS]

        n_comps = len(comp_keys)
        n_ranks = len(RANKS)
        x = np.arange(n_ranks)
        total_w = 0.82
        bw = total_w / n_comps

        fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))

        for col, target in enumerate(TARGETS):
            ax = axes[col]
            targeted = get_targeted_layers(inventory, target)

            # TT ops
            _, tt = compute_ops(targeted, 8, gamma)
            tt_total = sum(tt[k] / 1e12 for k in comp_keys)  # TOps

            # TT reference line
            ax.axhline(tt_total, color='#e67e22', ls='--', lw=1.2, alpha=0.7, zorder=2)

            # TT annotation
            tt_parts = [f'{cn}: {tt[k]/1e12:.2f}T'
                        for k, cn in zip(comp_keys, comp_names) if tt[k] > 0]
            ax.annotate(f'TT total: {tt_total:.2f}T\n' + '\n'.join(tt_parts),
                        xy=(n_ranks - 0.5, tt_total),
                        xytext=(n_ranks - 0.5, tt_total * 1.4),
                        fontsize=5.5, ha='right', va='bottom',
                        color='#333', style='italic',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#fff8e1', edgecolor='#e67e22',
                                  alpha=0.9, lw=0.5), zorder=8)

            # LR-TT grouped bars
            lr_list = []
            for rank in RANKS:
                lr, _ = compute_ops(targeted, rank, gamma)
                lr_list.append(lr)

            for ci, (k, cn, cc) in enumerate(zip(comp_keys, comp_names, comp_colors)):
                offset = -total_w / 2 + (ci + 0.5) * bw
                vals = [lr_list[i][k] / 1e12 for i in range(n_ranks)]  # TOps
                ax.bar(x + offset, vals, bw * 0.92, color=cc,
                       edgecolor='white', lw=0.4, zorder=3,
                       label=cn if col == 0 else '')

            # Annotations
            for i in range(n_ranks):
                total = sum(lr_list[i][k] / 1e12 for k in comp_keys)
                bar_top = max(lr_list[i][k] / 1e12 for k in comp_keys)
                ratio = tt_total / total if total > 0 else 0
                ax.text(x[i], bar_top * 1.15, f'{total:.2f}T\n{ratio:.0f}x',
                        ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                        color='#333',
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

            # Axis
            ax.set_yscale('log')
            all_lr = [lr_list[i][k] / 1e12 for i in range(n_ranks)
                      for k in comp_keys if lr_list[i][k] > 0]
            ax.set_ylim(min(all_lr) * 0.3, tt_total * 3)
            ax.set_xticks(x)
            ax.set_xticklabels([f'r={r}' for r in RANKS], fontsize=7)
            ax.set_title(target, fontsize=9, fontweight='bold', pad=4)
            if col == 0:
                ax.set_ylabel('$\\Delta$Ops [TOps]', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.08, which='major', zorder=0)

        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='#e67e22', ls='--', lw=1.2, alpha=0.7))
        labels.append('TT total')
        fig.legend(handles, labels, loc='lower center', ncol=n_comps + 1, fontsize=7,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
                   handlelength=1.5, handletextpad=0.4, columnspacing=1.0)

        fig.suptitle(f'Adapter ops per training step  ($\\gamma$={gamma})\n'
                     f'LR-TT bars vs TikiTaka (dashed)  |  BS={BATCH_SIZE}, S={S_PAD_DEFAULT}',
                     fontsize=9.5, fontweight='bold', y=1.01)
        plt.tight_layout(rect=[0, 0.07, 1, 0.94], w_pad=1.5)
        path = os.path.join(OUT, f"LRTT_VS_TT_OPS_FINAL_G{gamma}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

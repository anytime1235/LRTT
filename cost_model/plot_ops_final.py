#!/usr/bin/env python3
"""Single plot: TOPS per step, γ=0 and γ=1 side-by-side per method."""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from extract_layer_inventory import build_layer_inventory, get_targeted_layers

OUT = "/root/paper_figures_anchored"
inventory = build_layer_inventory()
BS, S, BT = 48, 384, 48*384
RANKS = [1, 4, 8, 16, 32, 64]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
})


def count_ops(layers, rank, gamma):
    lr_proj = lr_upd = lr_vis = tt_upd = tt_vis = 0
    for l in layers:
        M, N = l.M, l.N
        eA, eB = M*rank, rank*N
        lr_proj += BT * (eA + eB)
        lr_upd += BT * (eA + eB)
        if gamma == 1: lr_vis += BT * (eA + eB)
        tt_upd += BT * M * N
        if gamma == 1: tt_vis += BT * M * N
    return lr_proj, lr_upd, lr_vis, tt_upd, tt_vis


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for col, target in enumerate(['attention', 'ffn', 'all']):
        ax = axes[col]
        layers = get_targeted_layers(inventory, target)

        # Build grouped bars: for each method, γ=0 (left) and γ=1 (right)
        methods = ['TikiTaka'] + [f'r={r}' for r in RANKS]
        n_methods = len(methods)
        x = np.arange(n_methods)
        w = 0.35  # bar width

        # γ=0 bars (left)
        g0_upd = []; g0_proj = []; g0_vis = []
        # γ=1 bars (right)
        g1_upd = []; g1_proj = []; g1_vis = []

        for i, method in enumerate(methods):
            if i == 0:  # TikiTaka
                for gamma, upd_list, proj_list, vis_list in [
                    (0, g0_upd, g0_proj, g0_vis), (1, g1_upd, g1_proj, g1_vis)]:
                    _, _, _, tu, tv = count_ops(layers, 8, gamma)
                    upd_list.append(tu / 1e12)
                    proj_list.append(0)
                    vis_list.append(tv / 1e12)
            else:  # LR-TT
                r = RANKS[i-1]
                for gamma, upd_list, proj_list, vis_list in [
                    (0, g0_upd, g0_proj, g0_vis), (1, g1_upd, g1_proj, g1_vis)]:
                    lp, lu, lv, _, _ = count_ops(layers, r, gamma)
                    upd_list.append(lu / 1e12)
                    proj_list.append(lp / 1e12)
                    vis_list.append(lv / 1e12)

        # Plot γ=0 (left bars)
        ax.bar(x - w/2, g0_upd, w, color='#FFA726', edgecolor='#333', lw=0.4)
        ax.bar(x - w/2, g0_proj, w, bottom=g0_upd, color='#EF5350', edgecolor='#333', lw=0.4)
        bot0 = [a+b for a, b in zip(g0_upd, g0_proj)]
        ax.bar(x - w/2, g0_vis, w, bottom=bot0, color='#42A5F5', edgecolor='#333', lw=0.4)

        # Plot γ=1 (right bars)
        ax.bar(x + w/2, g1_upd, w, color='#FFA726', edgecolor='#333', lw=0.4, alpha=0.6, hatch='///')
        ax.bar(x + w/2, g1_proj, w, bottom=g1_upd, color='#EF5350', edgecolor='#333', lw=0.4, alpha=0.6, hatch='///')
        bot1 = [a+b for a, b in zip(g1_upd, g1_proj)]
        ax.bar(x + w/2, g1_vis, w, bottom=bot1, color='#42A5F5', edgecolor='#333', lw=0.4, alpha=0.6, hatch='///')

        # Total labels
        totals_g0 = [a+b+c for a,b,c in zip(g0_upd, g0_proj, [0]*len(g0_vis))]  # γ=0 has no vis
        totals_g0 = [a+b for a,b in zip(g0_upd, g0_proj)]
        totals_g1 = [a+b+c for a,b,c in zip(g1_upd, g1_proj, g1_vis)]

        ymax = max(max(totals_g0), max(totals_g1))
        for i in range(n_methods):
            if totals_g0[i] > ymax * 0.01:
                ax.text(x[i] - w/2, totals_g0[i] + ymax*0.01, f'{totals_g0[i]:.2f}',
                        ha='center', fontsize=6, fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            if totals_g1[i] > ymax * 0.01:
                ax.text(x[i] + w/2, totals_g1[i] + ymax*0.01, f'{totals_g1[i]:.2f}',
                        ha='center', fontsize=6, fontweight='bold', color='#555',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        if col == 0:
            ax.set_ylabel('TOPS per training step (×10¹²)', fontsize=10)
        ax.set_title(f'{target}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.08)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFA726', edgecolor='#333', label='Pulsed update'),
        Patch(facecolor='#EF5350', edgecolor='#333', label='Projection MVM'),
        Patch(facecolor='#42A5F5', edgecolor='#333', label='Visible forward MVM'),
        Patch(facecolor='white', edgecolor='#333', label='γ=0 (solid fill)'),
        Patch(facecolor='white', edgecolor='#333', hatch='///', label='γ=1 (hatched)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc')

    fig.suptitle('ACIM Operations per Training Step: TikiTaka vs LR-TT\n'
                 'Same hardware → fewer TOPS = less latency & energy  |  '
                 'Left bar = γ=0, Right bar (hatched) = γ=1',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(f'{OUT}/ops_tops_gamma01.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ops_tops_gamma01.png")


if __name__ == "__main__":
    main()
    plt.rcParams.update(plt.rcParamsDefault)

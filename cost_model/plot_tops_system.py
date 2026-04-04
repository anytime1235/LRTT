#!/usr/bin/env python3
"""System-level TOPS: base + adapter, TikiTaka vs LR-TT."""

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


def system_ops(target, rank, gamma):
    """Full system ops per step: base + adapter."""
    all_layers = list(inventory)  # all 72 for base
    tgt_layers = get_targeted_layers(inventory, target)

    # Base: forward + backward over ALL 72 layers
    base_fwd = sum(BT * l.M * l.N for l in all_layers)
    base_bwd = base_fwd  # same MAC count
    base = base_fwd + base_bwd

    # TikiTaka adapter
    tt_upd = sum(BT * l.M * l.N for l in tgt_layers)
    tt_vis = tt_upd if gamma == 1 else 0
    tt_total = base + tt_upd + tt_vis

    # LR-TT adapter
    lr_proj = sum(BT * (l.M*rank + rank*l.N) for l in tgt_layers)
    lr_upd = lr_proj
    lr_vis = lr_proj if gamma == 1 else 0
    lr_total = base + lr_proj + lr_upd + lr_vis

    return {
        'base': base,
        'tt_upd': tt_upd, 'tt_vis': tt_vis, 'tt_total': tt_total,
        'lr_proj': lr_proj, 'lr_upd': lr_upd, 'lr_vis': lr_vis, 'lr_total': lr_total,
    }


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))

    for col, target in enumerate(['attention', 'ffn', 'all']):
        ax = axes[col]

        methods = ['TikiTaka'] + [f'r={r}' for r in RANKS]
        n = len(methods)
        x = np.arange(n)
        w = 0.35

        for gamma, offset, alpha_bar, hatch in [(0, -w/2, 1.0, ''), (1, +w/2, 0.7, '///')]:
            bases = []; adapter_upds = []; adapter_projs = []; adapter_viss = []

            for i, method in enumerate(methods):
                if i == 0:  # TikiTaka
                    ops = system_ops(target, 8, gamma)
                    bases.append(ops['base'] / 1e12)
                    adapter_upds.append(ops['tt_upd'] / 1e12)
                    adapter_projs.append(0)
                    adapter_viss.append(ops['tt_vis'] / 1e12)
                else:
                    r = RANKS[i-1]
                    ops = system_ops(target, r, gamma)
                    bases.append(ops['base'] / 1e12)
                    adapter_upds.append(ops['lr_upd'] / 1e12)
                    adapter_projs.append(ops['lr_proj'] / 1e12)
                    adapter_viss.append(ops['lr_vis'] / 1e12)

            # Stack: base (gray) + update (orange) + proj (red) + vis (blue)
            ax.bar(x + offset, bases, w, color='#BDBDBD', edgecolor='#666', lw=0.3,
                   alpha=alpha_bar, hatch=hatch)
            ax.bar(x + offset, adapter_upds, w, bottom=bases, color='#FFA726',
                   edgecolor='#333', lw=0.3, alpha=alpha_bar, hatch=hatch)
            bot1 = [a+b for a, b in zip(bases, adapter_upds)]
            ax.bar(x + offset, adapter_projs, w, bottom=bot1, color='#EF5350',
                   edgecolor='#333', lw=0.3, alpha=alpha_bar, hatch=hatch)
            bot2 = [a+b for a, b in zip(bot1, adapter_projs)]
            ax.bar(x + offset, adapter_viss, w, bottom=bot2, color='#42A5F5',
                   edgecolor='#333', lw=0.3, alpha=alpha_bar, hatch=hatch)

            # Total label
            totals = [a+b+c+d for a,b,c,d in zip(bases, adapter_upds, adapter_projs, adapter_viss)]
            for i, t in enumerate(totals):
                fontsize = 6 if gamma == 1 else 7
                color_t = '#555' if gamma == 1 else 'black'
                ax.text(x[i] + offset, t + max(totals)*0.01, f'{t:.1f}',
                        ha='center', fontsize=fontsize, fontweight='bold', color=color_t,
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        if col == 0:
            ax.set_ylabel('Total Ops per Step (TOPS, ×10¹²)', fontsize=10)
        ax.set_title(f'{target}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.08)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#BDBDBD', edgecolor='#666', label='Base path (common)'),
        Patch(facecolor='#FFA726', edgecolor='#333', label='Pulsed update'),
        Patch(facecolor='#EF5350', edgecolor='#333', label='Projection MVM'),
        Patch(facecolor='#42A5F5', edgecolor='#333', label='Visible forward'),
        Patch(facecolor='white', edgecolor='#333', label='γ=0 (solid)'),
        Patch(facecolor='white', edgecolor='#333', hatch='///', label='γ=1 (hatched)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc')

    fig.suptitle('System-Level TOPS per Training Step: TikiTaka vs LR-TT\n'
                 'Gray = common base (identical) | Colors = adapter-specific (method difference)\n'
                 'Left bar = γ=0 | Right hatched = γ=1  |  BS=48, S=384',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    fig.savefig(f'{OUT}/tops_system_level.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  tops_system_level.png")


if __name__ == "__main__":
    main()
    plt.rcParams.update(plt.rcParamsDefault)

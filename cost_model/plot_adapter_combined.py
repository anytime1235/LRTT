#!/usr/bin/env python3
"""Adapter-only TOPS: γ=0 and γ=1 grouped, log scale."""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

from extract_layer_inventory import build_layer_inventory, get_targeted_layers

OUT = "/root/paper_figures_anchored"
inventory = build_layer_inventory()
BS, S, BT = 48, 384, 48*384
RANKS = [1, 4, 8, 16, 32, 64]
TE = 4

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
})


def adapter_tops(target, rank, gamma):
    tgt = get_targeted_layers(inventory, target)
    lr_vis = lr_proj = lr_upd = lr_tr = 0
    tt_upd = tt_vis = tt_tr = 0
    for l in tgt:
        M, N = l.M, l.N
        if gamma == 1:
            lr_vis += BT * (rank*N + M*rank)
        lr_proj += BT * (rank*N + M*rank)
        lr_upd += BT * (M*rank + rank*N)
        lr_tr += (rank*(M+N)/TE + rank*M*N/TE)
        tt_upd += BT * M * N
        if gamma == 1:
            tt_vis += BT * M * N
        tt_tr += 2*M
    return {
        'lr_vis': lr_vis/1e12, 'lr_proj': lr_proj/1e12,
        'lr_upd': lr_upd/1e12, 'lr_tr': lr_tr/1e12,
        'tt_upd': tt_upd/1e12, 'tt_vis': tt_vis/1e12, 'tt_tr': tt_tr/1e12,
    }


def main():
    fig, ax = plt.subplots(figsize=(14, 7))

    methods = ['TikiTaka'] + [f'r={r}' for r in RANKS]
    n = len(methods)
    x = np.arange(n) * 1.2
    w = 0.25

    colors_g0 = {'tt': '#E65100', 'lr': '#1565C0'}
    colors_g1 = {'tt': '#FF8A65', 'lr': '#64B5F6'}

    for gamma, offset, edge_w in [(0, -w/2-0.02, 1.0), (1, +w/2+0.02, 0.6)]:
        for i, method in enumerate(methods):
            if i == 0:
                ops = adapter_tops('all', 8, gamma)
                total = ops['tt_upd'] + ops['tt_vis'] + ops['tt_tr']
                upd_frac = ops['tt_upd'] / total if total > 0 else 0
                vis_frac = ops['tt_vis'] / total if total > 0 else 0
                color = colors_g0['tt'] if gamma == 0 else colors_g1['tt']
            else:
                r = RANKS[i-1]
                ops = adapter_tops('all', r, gamma)
                total = ops['lr_vis'] + ops['lr_proj'] + ops['lr_upd'] + ops['lr_tr']
                color = colors_g0['lr'] if gamma == 0 else colors_g1['lr']

            if total > 0:
                hatch = '' if gamma == 0 else '///'
                ax.bar(x[i] + offset, total, w, color=color, edgecolor='#333',
                       lw=0.4, hatch=hatch, alpha=0.85)

                # Label
                label_y = total * 1.15
                ax.text(x[i] + offset, label_y, f'{total:.3f}',
                        ha='center', fontsize=7, fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_yscale('log')
    ax.set_ylim(1e-3, 10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Adapter ΔOps per Step (TOPS, log scale)', fontsize=11)
    ax.grid(axis='y', alpha=0.15, which='both')

    # Horizontal reference lines
    ax.axhline(1.0, color='#999', ls=':', lw=0.5)
    ax.axhline(0.1, color='#999', ls=':', lw=0.3)
    ax.axhline(0.01, color='#999', ls=':', lw=0.3)

    legend_elements = [
        Patch(facecolor=colors_g0['tt'], edgecolor='#333', label='TikiTaka γ=0'),
        Patch(facecolor=colors_g1['tt'], edgecolor='#333', hatch='///', label='TikiTaka γ=1'),
        Patch(facecolor=colors_g0['lr'], edgecolor='#333', label='LR-TT γ=0'),
        Patch(facecolor=colors_g1['lr'], edgecolor='#333', hatch='///', label='LR-TT γ=1'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right', framealpha=0.95)

    # Annotate the gap
    tt_g0 = adapter_tops('all', 8, 0)
    lr8_g0 = adapter_tops('all', 8, 0)
    tt_val = tt_g0['tt_upd']
    lr_val = lr8_g0['lr_proj'] + lr8_g0['lr_upd']
    ratio = tt_val / lr_val if lr_val > 0 else 0
    ax.annotate(f'{ratio:.0f}×\ngap', xy=(x[3]-w/2-0.02, lr_val), xytext=(x[3]+0.5, 0.3),
                fontsize=10, fontweight='bold', color='#C62828',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_title('Adapter-Only TOPS per Training Step (log scale)\n'
                 'Same ACIM hardware → fewer TOPS = less latency & energy  |  target=all',
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{OUT}/tile_ops_adapter_combined.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  tile_ops_adapter_combined.png")


if __name__ == "__main__":
    main()
    plt.rcParams.update(plt.rcParamsDefault)

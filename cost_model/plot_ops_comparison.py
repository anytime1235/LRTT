#!/usr/bin/env python3
"""Absolute Ops comparison: TikiTaka vs LR-TT on same ACIM hardware.

This is NOT algebra — it's counting actual hardware operations per training step.
Same hardware → same cost per op → lower ops = less latency AND less energy.
"""

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
    """Count absolute element-level operations per step."""
    lr_proj_ops = 0; lr_upd_ops = 0; lr_vis_ops = 0
    tt_upd_ops = 0; tt_vis_ops = 0

    for l in layers:
        M, N = l.M, l.N
        eA, eB = M*rank, rank*N

        lr_proj_ops += BT * (eA + eB)     # projection MACs
        lr_upd_ops += BT * (eA + eB)      # pulsed update element ops
        if gamma == 1:
            lr_vis_ops += BT * (eA + eB)  # visible forward MACs

        tt_upd_ops += BT * M * N           # full-rank update ops
        if gamma == 1:
            tt_vis_ops += BT * M * N       # visible forward MACs

    return {
        'lr_proj': lr_proj_ops, 'lr_upd': lr_upd_ops, 'lr_vis': lr_vis_ops,
        'lr_total': lr_proj_ops + lr_upd_ops + lr_vis_ops,
        'tt_upd': tt_upd_ops, 'tt_vis': tt_vis_ops,
        'tt_total': tt_upd_ops + tt_vis_ops,
    }


def main():
    # ═══════════════════════════════════════════════
    # MAIN FIGURE: Absolute Ops per Step (γ=0 and γ=1)
    # ═══════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row, gamma in enumerate([0, 1]):
        for col, target in enumerate(['attention', 'ffn', 'all']):
            ax = axes[row, col]
            layers = get_targeted_layers(inventory, target)

            # TikiTaka (single bar)
            ops_tt = count_ops(layers, 8, gamma)  # rank irrelevant for TT

            # Bar data: TT + LR-TT per rank
            labels = ['TikiTaka'] + [f'LR-TT\nr={r}' for r in RANKS]
            n = len(labels)
            x = np.arange(n)

            # Stacked bars: update (orange) + projection (red) + visible (blue)
            tt_upd_T = ops_tt['tt_upd'] / 1e12
            tt_vis_T = ops_tt['tt_vis'] / 1e12

            lr_upds_T = []; lr_projs_T = []; lr_viss_T = []
            for r in RANKS:
                ops_lr = count_ops(layers, r, gamma)
                lr_upds_T.append(ops_lr['lr_upd'] / 1e12)
                lr_projs_T.append(ops_lr['lr_proj'] / 1e12)
                lr_viss_T.append(ops_lr['lr_vis'] / 1e12)

            upd_vals = [tt_upd_T] + lr_upds_T
            proj_vals = [0] + lr_projs_T          # TT has no projection
            vis_vals = [tt_vis_T] + lr_viss_T

            w = 0.6
            bars_upd = ax.bar(x, upd_vals, w, color='#FFA726', edgecolor='#333', lw=0.4, label='Pulsed update')
            bars_proj = ax.bar(x, proj_vals, w, bottom=upd_vals, color='#EF5350', edgecolor='#333', lw=0.4, label='Projection MVM')
            bot2 = [a+b for a,b in zip(upd_vals, proj_vals)]
            bars_vis = ax.bar(x, vis_vals, w, bottom=bot2, color='#42A5F5', edgecolor='#333', lw=0.4, label='Visible fwd MVM')

            # Total label on top
            totals = [a+b+c for a,b,c in zip(upd_vals, proj_vals, vis_vals)]
            for i, t in enumerate(totals):
                if t > 0:
                    ax.text(i, t + max(totals)*0.02, f'{t:.1f}T', ha='center', fontsize=7, fontweight='bold',
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

            # Reduction annotation
            if len(totals) > 1 and totals[0] > 0:
                for i in range(1, len(totals)):
                    reduction = (1 - totals[i]/totals[0]) * 100
                    if reduction > 10:
                        ax.text(i, totals[i] * 0.5, f'−{reduction:.0f}%', ha='center',
                                fontsize=7, color='white', fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            if col == 0:
                ax.set_ylabel(f'γ={gamma}\nOps per step (×10¹²)', fontsize=10)
            ax.set_title(f'{target}', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.08)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Absolute ACIM Operations per Training Step\n'
                 'Same hardware → same cost per op → lower = cheaper\n'
                 'BS=48, S=384, target: attention / FFN / all',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f'{OUT}/ops_absolute_comparison.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ops_absolute_comparison.png")

    # ═══════════════════════════════════════════════
    # SINGLE SUMMARY: target=all, both γ side-by-side
    # ═══════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    layers = get_targeted_layers(inventory, 'all')

    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]
        ops_tt = count_ops(layers, 8, gamma)

        labels = ['TikiTaka'] + [f'r={r}' for r in RANKS]
        x = np.arange(len(labels))

        tt_upd_T = ops_tt['tt_upd'] / 1e12
        tt_vis_T = ops_tt['tt_vis'] / 1e12

        lr_upds = []; lr_projs = []; lr_viss = []
        for r in RANKS:
            ops_lr = count_ops(layers, r, gamma)
            lr_upds.append(ops_lr['lr_upd'] / 1e12)
            lr_projs.append(ops_lr['lr_proj'] / 1e12)
            lr_viss.append(ops_lr['lr_vis'] / 1e12)

        upd = [tt_upd_T] + lr_upds
        proj = [0] + lr_projs
        vis = [tt_vis_T] + lr_viss

        colors_bar = ['#4CAF50'] + ['#880E4F','#E91E63','#E53935','#FF9800','#FFC107','#CDDC39']

        # Total ops as single bar with color coding
        totals = [a+b+c for a,b,c in zip(upd, proj, vis)]

        ax.bar(x, upd, 0.6, color='#FFA726', edgecolor='#333', lw=0.4, label='Update')
        ax.bar(x, proj, 0.6, bottom=upd, color='#EF5350', edgecolor='#333', lw=0.4, label='Projection')
        bot2 = [a+b for a,b in zip(upd, proj)]
        ax.bar(x, vis, 0.6, bottom=bot2, color='#42A5F5', edgecolor='#333', lw=0.4, label='Visible fwd')

        for i, t in enumerate(totals):
            ax.text(i, t+max(totals)*0.01, f'{t:.1f}T', ha='center', fontsize=8, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        # TT line for reference
        ax.axhline(totals[0], color='#4CAF50', ls=':', lw=1, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        if ax_i == 0:
            ax.set_ylabel('Ops per step (TOPS, ×10¹²)', fontsize=10)
        ax.set_title(f'γ={gamma}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.08)
        if ax_i == 0:
            ax.legend(fontsize=8)

    fig.suptitle('ACIM Ops per Training Step: TikiTaka vs LR-TT  (target=all)\n'
                 'Same ACIM hardware → fewer ops = less latency & energy',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f'{OUT}/ops_summary.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ops_summary.png")


if __name__ == "__main__":
    main()
    plt.rcParams.update(plt.rcParamsDefault)

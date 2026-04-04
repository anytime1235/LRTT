#!/usr/bin/env python3
"""Per-tile-operation TOPS breakdown for LR-TT and TikiTaka."""

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
TE = 4  # transfer_every for LR-TT

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
})


def compute_all_ops(target, rank, gamma):
    all_layers = list(inventory)
    tgt = get_targeted_layers(inventory, target)

    # Common base (all 72 layers)
    base_fwd = sum(BT * l.M * l.N for l in all_layers)
    base_bwd = base_fwd

    # LR-TT adapter ops (targeted layers only)
    lr_vis_b = lr_vis_a = 0
    lr_proj_b = lr_proj_a = 0
    lr_upd_a = lr_upd_b = 0
    lr_tr_read_a = lr_tr_read_b = 0
    lr_tr_upd_c = 0

    # TikiTaka adapter ops
    tt_upd = tt_vis = 0
    tt_tr_read = tt_tr_write = 0

    for l in tgt:
        M, N = l.M, l.N

        # LR-TT
        if gamma == 1:
            lr_vis_b += BT * rank * N       # F2: B.forward(x) visible
            lr_vis_a += BT * M * rank       # F3: A.forward(Bx) visible
        lr_proj_b += BT * rank * N          # U1: B.forward(x) projection
        lr_proj_a += BT * M * rank          # U2: A.backward(d) projection
        lr_upd_a += BT * M * rank           # U3: A.update(XB, d)
        lr_upd_b += BT * rank * N           # U4: B.update(x, DA)
        lr_tr_read_a += rank * M / TE       # T1: A.forward(one-hot) amortized
        lr_tr_read_b += rank * N / TE       # T2: B.backward(one-hot) amortized
        lr_tr_upd_c += rank * M * N / TE    # T3: C.update(a_k, b_k) amortized

        # TikiTaka
        tt_upd += BT * M * N
        if gamma == 1:
            tt_vis += BT * M * N
        tt_tr_read += M / 1     # column read, te=1
        tt_tr_write += M / 1    # column write, te=1

    return {
        'base_fwd': base_fwd, 'base_bwd': base_bwd,
        # LR-TT
        'lr_vis_b': lr_vis_b, 'lr_vis_a': lr_vis_a,
        'lr_proj_b': lr_proj_b, 'lr_proj_a': lr_proj_a,
        'lr_upd_a': lr_upd_a, 'lr_upd_b': lr_upd_b,
        'lr_tr_read_a': lr_tr_read_a, 'lr_tr_read_b': lr_tr_read_b,
        'lr_tr_upd_c': lr_tr_upd_c,
        # TikiTaka
        'tt_upd': tt_upd, 'tt_vis': tt_vis,
        'tt_tr_read': tt_tr_read, 'tt_tr_write': tt_tr_write,
    }


def main():
    # ═══════════════════════════════════════
    # FIGURE 1: Stacked bar — all operations
    # ═══════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]

        methods = ['TikiTaka'] + [f'LR-TT r={r}' for r in RANKS]
        n = len(methods)
        x = np.arange(n)
        w = 0.55

        # Operation categories (order: bottom to top)
        op_names = [
            'F1: C.forward (base)',
            'B1: C.backward (base)',
            'F2: B.forward (visible)',
            'F3: A.forward (visible)',
            'U1: B.forward (projection)',
            'U2: A.backward (projection)',
            'U3: A.update (pulsed)',
            'U4: B.update (pulsed)',
            'T: Transfer (amortized)',
            'TT: U full-rank update',
            'TT: U visible forward',
        ]
        op_colors = [
            '#BDBDBD',   # base fwd
            '#9E9E9E',   # base bwd
            '#42A5F5',   # vis B
            '#1E88E5',   # vis A
            '#EF5350',   # proj B
            '#C62828',   # proj A
            '#FFA726',   # upd A
            '#F57C00',   # upd B
            '#CE93D8',   # transfer
            '#FF8A65',   # TT update
            '#64B5F6',   # TT visible
        ]

        all_bars = []
        for i, method in enumerate(methods):
            if i == 0:  # TikiTaka
                ops = compute_all_ops('all', 8, gamma)
                vals = [
                    ops['base_fwd'], ops['base_bwd'],
                    0, 0, 0, 0, 0, 0, ops['tt_tr_read'] + ops['tt_tr_write'],
                    ops['tt_upd'], ops['tt_vis'],
                ]
            else:
                r = RANKS[i-1]
                ops = compute_all_ops('all', r, gamma)
                vals = [
                    ops['base_fwd'], ops['base_bwd'],
                    ops['lr_vis_b'], ops['lr_vis_a'],
                    ops['lr_proj_b'], ops['lr_proj_a'],
                    ops['lr_upd_a'], ops['lr_upd_b'],
                    ops['lr_tr_read_a'] + ops['lr_tr_read_b'] + ops['lr_tr_upd_c'],
                    0, 0,
                ]
            all_bars.append([v / 1e12 for v in vals])  # Convert to TOPS

        # Stack bars
        bottom = np.zeros(n)
        for j, (name, color) in enumerate(zip(op_names, op_colors)):
            vals = [all_bars[i][j] for i in range(n)]
            if max(vals) > 0:  # skip empty ops
                ax.bar(x, vals, w, bottom=bottom, color=color, edgecolor='#333',
                       lw=0.3, label=name)
                bottom += np.array(vals)

        # Total labels
        for i in range(n):
            total = sum(all_bars[i])
            ax.text(i, total + max(bottom)*0.01, f'{total:.2f}T',
                    ha='center', fontsize=7, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel(f'γ={gamma}\nTotal Ops per Step (TOPS)', fontsize=10)
        ax.grid(axis='y', alpha=0.08)
        if ax_i == 0:
            ax.legend(fontsize=6, ncol=3, loc='upper right', framealpha=0.9)

    fig.suptitle('Per-Tile-Operation TOPS Breakdown: TikiTaka vs LR-TT  (target=all)\n'
                 'Gray = common base | Colors = adapter operations | Same ACIM hardware',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/tile_ops_breakdown_full.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  tile_ops_breakdown_full.png")

    # ═══════════════════════════════════════
    # FIGURE 2: Adapter-only (zoom in, no base)
    # ═══════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]

        methods = ['TikiTaka'] + [f'r={r}' for r in RANKS]
        n = len(methods)
        x = np.arange(n)

        adapter_ops_names = [
            'Visible B.fwd', 'Visible A.fwd',
            'Proj B.fwd', 'Proj A.bwd',
            'Update A', 'Update B',
            'Transfer',
            'TT Update', 'TT Visible',
        ]
        adapter_colors = [
            '#42A5F5', '#1E88E5',
            '#EF5350', '#C62828',
            '#FFA726', '#F57C00',
            '#CE93D8',
            '#FF8A65', '#64B5F6',
        ]

        all_bars = []
        for i, method in enumerate(methods):
            if i == 0:
                ops = compute_all_ops('all', 8, gamma)
                vals = [0, 0, 0, 0, 0, 0,
                        ops['tt_tr_read']+ops['tt_tr_write'],
                        ops['tt_upd'], ops['tt_vis']]
            else:
                r = RANKS[i-1]
                ops = compute_all_ops('all', r, gamma)
                vals = [ops['lr_vis_b'], ops['lr_vis_a'],
                        ops['lr_proj_b'], ops['lr_proj_a'],
                        ops['lr_upd_a'], ops['lr_upd_b'],
                        ops['lr_tr_read_a']+ops['lr_tr_read_b']+ops['lr_tr_upd_c'],
                        0, 0]
            all_bars.append([v / 1e12 for v in vals])

        bottom = np.zeros(n)
        for j, (name, color) in enumerate(zip(adapter_ops_names, adapter_colors)):
            vals = [all_bars[i][j] for i in range(n)]
            if max(vals) > 0:
                ax.bar(x, vals, 0.6, bottom=bottom, color=color, edgecolor='#333',
                       lw=0.3, label=name)
                bottom += np.array(vals)

        for i in range(n):
            total = sum(all_bars[i])
            if total > 0:
                ax.text(i, total + max(bottom)*0.01, f'{total:.3f}T',
                        ha='center', fontsize=7, fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        if ax_i == 0:
            ax.set_ylabel('Adapter ΔOps (TOPS)', fontsize=10)
        ax.set_title(f'γ={gamma}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.08)
        if ax_i == 0:
            ax.legend(fontsize=6, ncol=3, loc='upper right', framealpha=0.9)

    fig.suptitle('Adapter-Only TOPS Breakdown (base removed)\n'
                 'TikiTaka: orange=full-rank update, blue=visible | LR-TT: decomposed by tile operation',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f'{OUT}/tile_ops_breakdown_adapter.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  tile_ops_breakdown_adapter.png")


if __name__ == "__main__":
    main()
    plt.rcParams.update(plt.rcParamsDefault)

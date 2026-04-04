#!/usr/bin/env python3
"""B, C, D with distinct color palettes per metric.
ACS/NPG style. Separate from generate_all.py.

B (Ops):   vermillion TT / indigo LR-TT
C (Tiles): vermillion TT / teal LR-TT
D (Energy): vermillion TT / amber LR-TT
"""

import os, sys, math
import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'lrtt_vs_tt_v2'))
COST_MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'cost_model')
sys.path.insert(0, COST_MODEL_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    BATCH_SIZE, S_PAD_DEFAULT, EPS_ACIM_PJ,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
T = 512
KAPPA_E = 0.5
RANKS = [1, 2, 4, 8, 16, 32, 64]
TARGETS = ['attention', 'ffn', 'all']

ACS_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
}

# Per-metric color pairs (TT, LR-TT)
METRIC_COLORS = {
    'ops':    ('#E64B35', '#3C5488'),   # vermillion / indigo
    'tiles':  ('#E64B35', '#00A087'),   # vermillion / teal
    'energy': ('#E64B35', '#F39B7F'),   # vermillion / light coral
}


def compute_ops(targeted, rank, gamma=0):
    lr_ops = 0; tt_ops = 0
    for l in targeted:
        M, N = l.M, l.N
        lr_ops += BT * 2 * (rank*N + M*rank) + BT * 2 * (M*rank + rank*N)
        tt_ops += BT * 2 * M * N
        if gamma == 1:
            lr_ops += BT * 2 * (rank*N + M*rank)
            tt_ops += BT * 2 * M * N
    return tt_ops, lr_ops


def compute_tiles(targeted, rank):
    shape_counts = Counter((l.M, l.N) for l in targeted)
    apt = max(1, T // rank) if rank < T else 1
    lr_t = 0
    for (M_s, N_s), cnt in shape_counts.items():
        lr_t += math.ceil(cnt / apt) * math.ceil(M_s / T)
        lr_t += math.ceil(cnt / apt) * math.ceil(N_s / T)
    tt_t = sum(math.ceil(l.M / T) * math.ceil(l.N / T) for l in targeted)
    return tt_t, lr_t


def compute_energy(targeted, rank, gamma=0):
    lr_e = 0.0; tt_e = 0.0
    for l in targeted:
        M, N = l.M, l.N
        proj = BT * 2 * (rank*N + M*rank)
        upd_lr = BT * 2 * (M*rank + rank*N)
        upd_tt = BT * 2 * M * N
        vis_lr = BT * 2 * (rank*N + M*rank) if gamma == 1 else 0
        vis_tt = BT * 2 * M * N if gamma == 1 else 0
        lr_e += EPS_ACIM_PJ * (proj + vis_lr) / 1e6
        lr_e += KAPPA_E * EPS_ACIM_PJ * upd_lr / 1e6
        tt_e += EPS_ACIM_PJ * vis_tt / 1e6
        tt_e += KAPPA_E * EPS_ACIM_PJ * upd_tt / 1e6
    return tt_e, lr_e


def draw_metric(inventory, metric_fn, ylabel, title, fname, c_tt, c_lr, transform=None):
    plt.rcdefaults()
    plt.rcParams.update(ACS_RC)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for col, target in enumerate(TARGETS):
        ax = axes[col]
        targeted = get_targeted_layers(inventory, target)
        x_labels = ['TT'] + [f'r={r}' for r in RANKS]
        xp = np.arange(len(x_labels))

        tt_raw, _ = metric_fn(targeted, 8)
        tt_val = transform(tt_raw) if transform else tt_raw
        vals = [tt_val]
        for rank in RANKS:
            _, lr_raw = metric_fn(targeted, rank)
            vals.append(transform(lr_raw) if transform else lr_raw)

        colors = [c_tt] + [c_lr] * len(RANKS)
        ax.bar(xp, vals, 0.6, color=colors, alpha=0.92, edgecolor='none', zorder=3)

        # Annotations
        for i in range(len(xp)):
            r = vals[0] / vals[i] if vals[i] > 0 else 0
            if i == 0:
                txt = f'{vals[i]:.2f}' if vals[i] < 10 else f'{vals[i]:.0f}'
                c = c_tt
            else:
                txt = f'{r:.0f}×'
                c = '#333'
            ax.text(xp[i], vals[i] * 1.12, txt, ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=c)

        ax.set_yscale('log')
        ax.set_xticks(xp)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(target, fontweight='bold')
        if col == 0:
            ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.2, axis='y')

    h = [Patch(facecolor=c_tt, alpha=0.92, label='TikiTaka'),
         Patch(facecolor=c_lr, alpha=0.92, label='LR-TT')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='none')
    fig.suptitle(title, fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(f'{OUT}/{fname}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  {fname} saved')


def main():
    inventory = build_layer_inventory()
    print(f'Layers: {len(inventory)}')

    c = METRIC_COLORS

    # B: Ops (indigo)
    draw_metric(inventory,
                lambda t, r: compute_ops(t, r, 0),
                'Ops [TOps]',
                '(b)  Active operations per training step, $\\gamma$ = 0',
                'B_ops_colored.png',
                c['ops'][0], c['ops'][1],
                transform=lambda v: v / 1e12)

    # C: Tiles (teal)
    draw_metric(inventory,
                compute_tiles,
                'Tiles (packed)',
                '(c)  Physical tile count (multi-layer column-packing)',
                'C_tiles_colored.png',
                c['tiles'][0], c['tiles'][1])

    # D: Energy (coral)
    draw_metric(inventory,
                lambda t, r: compute_energy(t, r, 0),
                'Energy [$\\mu$J]',
                f'(d)  Adapter energy, $\\gamma$ = 0, $\\kappa_e$ = {KAPPA_E}',
                'D_energy_colored.png',
                c['energy'][0], c['energy'][1])

    print('Done.')


if __name__ == '__main__':
    main()

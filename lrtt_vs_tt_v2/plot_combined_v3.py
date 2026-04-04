#!/usr/bin/env python3
"""Combined figure (2 rows):
  Top: Latency — per-element (optimistic) vs hybrid (conservative) side by side
  Bottom: Ops total (single bar)
  γ=1, attention for latency, attention/ffn/all for ops
"""

import os, sys, math
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
    RANKS, TARGETS, BATCH_SIZE, S_PAD_DEFAULT, TAU_ACIM,
    ALPINE_TILE_MVM_NS, ALPINE_TILE_SIZE,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
TILE = ALPINE_TILE_SIZE

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

C_ELEM = '#2980b9'     # per-element (optimistic) - blue
C_HYBRID = '#e74c3c'   # hybrid (conservative) - red
C_TT_ELEM = '#f39c12'  # TT per-element
C_TT_HYBRID = '#c0392b' # TT hybrid
C_OPS_TT = '#e67e22'
C_OPS_LR = '#2980b9'

COMP_KEYS = ['proj', 'update', 'visible']

ALPHA_LIT = [
    (0.25, 'Gokmen 2016\n(RPU 4096)'),
    (1.4,  'Rasch 2024\n(TTv2)'),
    (2.0,  'Gokmen 2016\n(RPU 512)'),
]


def compute_ops(targeted, rank, gamma=1):
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


def latency_elem(ops, alpha_t):
    """Per-element latency total [ms]."""
    t = TAU_ACIM * ops['proj'] + alpha_t * TAU_ACIM * ops['update'] + TAU_ACIM * ops['visible']
    return t / 1e6


def latency_hybrid(ops, targeted, rank_or_none, alpha_t, is_tt=False):
    """Hybrid: MVM per-element, Update per-tile. Total [ms]."""
    t_mvm = TAU_ACIM * (ops['proj'] + ops['visible']) / 1e6
    # Update per-tile
    n_tiles = 0
    for l in targeted:
        M, N = l.M, l.N
        if is_tt:
            n_tiles += math.ceil(M / TILE) * math.ceil(N / TILE)
        else:
            n_tiles += math.ceil(M / TILE) * math.ceil(rank_or_none / TILE) + \
                       math.ceil(rank_or_none / TILE) * math.ceil(N / TILE)
    t_upd = alpha_t * ALPINE_TILE_MVM_NS * BT * n_tiles / 1e6
    return t_mvm + t_upd


def main():
    inventory = build_layer_inventory()
    gamma = 1
    targeted_attn = get_targeted_layers(inventory, 'attention')

    fig, axes = plt.subplots(2, 3, figsize=(10, 7.5),
                             gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35})

    # ── Top row: Latency (per-element vs hybrid), attention, 3 α_t ──
    for pi, (alpha_t, cite) in enumerate(ALPHA_LIT):
        ax = axes[0, pi]
        x_labels = ['TT'] + [f'r={r}' for r in RANKS]
        n_x = len(x_labels)
        xp = np.arange(n_x)
        w = 0.35

        _, tt = compute_ops(targeted_attn, 8, gamma)
        tt_elem = latency_elem(tt, alpha_t)
        tt_hyb = latency_hybrid(tt, targeted_attn, None, alpha_t, is_tt=True)

        # TT bars
        ax.bar(xp[0] - w/2, tt_elem, w * 0.9, color=C_ELEM, edgecolor='white',
               lw=0.4, zorder=3, alpha=0.5)
        ax.bar(xp[0] + w/2, tt_hyb, w * 0.9, color=C_HYBRID, edgecolor='white',
               lw=0.4, zorder=3, alpha=0.5)
        tt_max = max(tt_elem, tt_hyb)
        ax.text(xp[0], tt_max * 1.15, f'{tt_elem:.0f} | {tt_hyb:.0f}',
                ha='center', va='bottom', fontsize=5, fontweight='bold', color='#e67e22',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        # LR-TT bars
        for ri, rank in enumerate(RANKS):
            lr, _ = compute_ops(targeted_attn, rank, gamma)
            lr_elem = latency_elem(lr, alpha_t)
            lr_hyb = latency_hybrid(lr, targeted_attn, rank, alpha_t)

            ax.bar(xp[1 + ri] - w/2, lr_elem, w * 0.9, color=C_ELEM,
                   edgecolor='white', lw=0.4, zorder=3)
            ax.bar(xp[1 + ri] + w/2, lr_hyb, w * 0.9, color=C_HYBRID,
                   edgecolor='white', lw=0.4, zorder=3)

            # Ratio labels
            r_elem = tt_elem / lr_elem if lr_elem > 0 else 0
            r_hyb = tt_hyb / lr_hyb if lr_hyb > 0 else 0
            bar_top = max(lr_elem, lr_hyb)
            ax.text(xp[1 + ri], bar_top * 1.15,
                    f'{r_elem:.0f}x|{r_hyb:.0f}x',
                    ha='center', va='bottom', fontsize=5, fontweight='bold', color='#555',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        ax.set_yscale('log')
        ax.set_xticks(xp)
        ax.set_xticklabels(x_labels, fontsize=6.5)
        tl = ax.get_xticklabels()
        tl[0].set_fontweight('bold')
        tl[0].set_color('#e67e22')
        ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
        ax.set_title(f'attention  |  $\\alpha_t$={alpha_t}\n{cite}',
                     fontsize=8, fontweight='bold', pad=6)
        if pi == 0:
            ax.set_ylabel('$\\Delta T_{step}$ [ms]', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.08, which='major', zorder=0)

    # ── Bottom row: OPs total, attention/ffn/all ──
    for col, target in enumerate(TARGETS):
        ax = axes[1, col]
        targeted = get_targeted_layers(inventory, target)
        x_labels_o = ['TT'] + [f'r={r}' for r in RANKS]
        xo = np.arange(len(x_labels_o))
        w_bar = 0.55

        _, tt = compute_ops(targeted, 8, gamma)
        tt_total = sum(tt[k] / 1e12 for k in COMP_KEYS)

        ax.bar(xo[0], tt_total, w_bar, color=C_OPS_TT, edgecolor='white',
               lw=0.4, zorder=3, alpha=0.7)
        ax.text(xo[0], tt_total * 1.08, f'{tt_total:.2f}T',
                ha='center', va='bottom', fontsize=6, fontweight='bold', color='#e67e22',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        for ri, rank in enumerate(RANKS):
            lr, _ = compute_ops(targeted, rank, gamma)
            lr_total = sum(lr[k] / 1e12 for k in COMP_KEYS)
            ax.bar(xo[1 + ri], lr_total, w_bar, color=C_OPS_LR, edgecolor='white',
                   lw=0.4, zorder=3)
            ratio = tt_total / lr_total if lr_total > 0 else 0
            ax.text(xo[1 + ri], lr_total * 1.08, f'{ratio:.0f}x',
                    ha='center', va='bottom', fontsize=6, fontweight='bold', color='#555',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        ax.set_yscale('log')
        ax.set_xticks(xo)
        ax.set_xticklabels(x_labels_o, fontsize=6.5)
        tl = ax.get_xticklabels()
        tl[0].set_fontweight('bold')
        tl[0].set_color('#e67e22')
        ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
        if col == 0:
            ax.set_ylabel('$\\Delta$Ops [TOps]', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.08, which='major', zorder=0)
        ax.set_title(target, fontsize=9, fontweight='bold', pad=4)

    # Row labels
    axes[0, 0].text(-0.25, 0.5, 'Latency',
                    transform=axes[0, 0].transAxes, fontsize=7, fontweight='bold',
                    va='center', ha='center', rotation=90, color='#333')
    axes[1, 0].text(-0.25, 0.5, 'Ops count',
                    transform=axes[1, 0].transAxes, fontsize=7, fontweight='bold',
                    va='center', ha='center', rotation=90, color='#333')

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_ELEM),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_HYBRID),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_OPS_TT, alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_OPS_LR),
    ]
    labels = [
        'Per-element (optimistic)',
        'Hybrid: MVM elem + Update tile (conservative)',
        'TT', 'LR-TT',
    ]
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
               handlelength=1.5, handletextpad=0.4, columnspacing=1.0)

    fig.suptitle('$\\gamma$=1  |  Top: Latency (blue=optimistic, red=conservative)  |  '
                 'Bottom: Active ops  |  Nx = TT/LR-TT',
                 fontsize=9, fontweight='bold', y=1.0)

    path = os.path.join(OUT, "LRTT_VS_TT_COMBINED_V3_G1.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combined figure:
  Top row: Latency (per-element), attention only, 3 α_t columns
  Bottom row: OPs, attention/ffn/all 3 columns
  γ=1
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
import math

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

COMP_KEYS   = ['proj', 'update', 'visible']
COMP_NAMES  = ['Projection', 'Update', 'Visible fwd']
COMP_COLORS = [C_PROJ, C_UPDATE, C_VIS]

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


def draw_panel(ax, targeted, val_fn_lr, val_fn_tt, ylabel, unit_fmt,
               show_legend=False, gamma=1):
    n_c = len(COMP_KEYS)
    # x: TT at 0, then LR-TT ranks at 1..
    x_labels = ['TT'] + [f'r={r}' for r in RANKS]
    n_x = len(x_labels)
    x = np.arange(n_x)
    tw = 0.82
    bw = tw / n_c

    # TT ops
    _, tt = compute_ops(targeted, 8, gamma)
    tt_total = sum(val_fn_tt(tt, k) for k in COMP_KEYS)

    # TT bars at x=0
    for ci, (k, cn, cc) in enumerate(zip(COMP_KEYS, COMP_NAMES, COMP_COLORS)):
        offset = -tw / 2 + (ci + 0.5) * bw
        v = val_fn_tt(tt, k)
        if v > 0:
            ax.bar(x[0] + offset, v, bw * 0.92, color=cc, edgecolor='white', lw=0.4,
                   zorder=3, alpha=0.6)

    # TT total label
    ax.text(x[0], tt_total * 1.12, unit_fmt(tt_total),
            ha='center', va='bottom', fontsize=6, fontweight='bold', color='#e67e22',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    # LR-TT bars at x=1..
    lr_list = []
    for rank in RANKS:
        lr, _ = compute_ops(targeted, rank, gamma)
        lr_list.append(lr)

    for ci, (k, cn, cc) in enumerate(zip(COMP_KEYS, COMP_NAMES, COMP_COLORS)):
        offset = -tw / 2 + (ci + 0.5) * bw
        vals = [val_fn_lr(lr_list[i], k) for i in range(len(RANKS))]
        ax.bar(x[1:] + offset, vals, bw * 0.92, color=cc, edgecolor='white', lw=0.4,
               zorder=3, label=cn if show_legend else '')

    # Ratio annotations on LR-TT bars
    for i in range(len(RANKS)):
        total = sum(val_fn_lr(lr_list[i], k) for k in COMP_KEYS)
        bar_top = max(val_fn_lr(lr_list[i], k) for k in COMP_KEYS
                      if val_fn_lr(lr_list[i], k) > 0)
        ratio = tt_total / total if total > 0 else 0
        ax.text(x[1 + i], bar_top * 1.12, f'{ratio:.0f}x',
                ha='center', va='bottom', fontsize=6, fontweight='bold', color='#555',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    ax.set_yscale('log')
    all_v = [val_fn_lr(lr_list[i], k) for i in range(len(RANKS))
             for k in COMP_KEYS if val_fn_lr(lr_list[i], k) > 0]
    all_v += [val_fn_tt(tt, k) for k in COMP_KEYS if val_fn_tt(tt, k) > 0]
    if all_v:
        ax.set_ylim(min(all_v) * 0.3, max(all_v) * 3.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6.5)
    tl = ax.get_xticklabels()
    tl[0].set_fontweight('bold')
    tl[0].set_color('#e67e22')
    ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.08, which='major', zorder=0)


def main():
    inventory = build_layer_inventory()
    gamma = 1
    targeted_attn = get_targeted_layers(inventory, 'attention')

    fig, axes = plt.subplots(3, 3, figsize=(10, 11),
                             gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.40})

    # ── Top row: Latency (per-element), attention, 3 α_t ──
    for pi, (alpha_t, cite) in enumerate(ALPHA_LIT):
        lat_w = {
            'proj':    TAU_ACIM,
            'update':  alpha_t * TAU_ACIM,
            'visible': TAU_ACIM,
        }

        draw_panel(
            axes[0, pi], targeted_attn,
            val_fn_lr=lambda lr, k, lw=lat_w: lr[k] * lw[k] / 1e6,
            val_fn_tt=lambda tt, k, lw=lat_w: tt[k] * lw[k] / 1e6,
            ylabel='$\\Delta T_{step}$ [ms]' if pi == 0 else '',
            unit_fmt=lambda v: f'{v:.0f}ms',
            show_legend=(pi == 0),
        )
        axes[0, pi].set_title(f'attention  |  $\\alpha_t$={alpha_t}\n{cite}',
                              fontsize=8, fontweight='bold', pad=6)

    # ── Middle row: Hybrid Latency (MVM per-element + Update per-tile), attention, 3 α_t ──
    # Single bar per method (total only, no component split)
    C_HYBRID_TT = '#e67e22'
    C_HYBRID_LR = '#2980b9'

    for pi, (alpha_t, cite) in enumerate(ALPHA_LIT):
        ax = axes[1, pi]
        x_labels_h = ['TT'] + [f'r={r}' for r in RANKS]
        n_x = len(x_labels_h)
        xh = np.arange(n_x)
        w_bar = 0.55

        _, tt = compute_ops(targeted_attn, 8, gamma)

        # TT hybrid total
        tt_mvm = (tt['proj'] + tt['visible']) * TAU_ACIM / 1e6
        tt_tiles = sum(math.ceil(l.M / 256) * math.ceil(l.N / 256) for l in targeted_attn)
        tt_upd = alpha_t * ALPINE_TILE_MVM_NS * BT * tt_tiles / 1e6
        tt_total = tt_mvm + tt_upd

        ax.bar(xh[0], tt_total, w_bar, color=C_HYBRID_TT, edgecolor='white',
               lw=0.4, zorder=3, alpha=0.7)
        ax.text(xh[0], tt_total * 1.08, f'{tt_total:.0f}ms',
                ha='center', va='bottom', fontsize=6, fontweight='bold', color='#e67e22',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        # LR-TT hybrid totals
        for ri, rank in enumerate(RANKS):
            lr, _ = compute_ops(targeted_attn, rank, gamma)
            lr_mvm = (lr['proj'] + lr['visible']) * TAU_ACIM / 1e6
            lr_tiles = sum(
                math.ceil(l.M / 256) * math.ceil(rank / 256) +
                math.ceil(rank / 256) * math.ceil(l.N / 256)
                for l in targeted_attn)
            lr_upd = alpha_t * ALPINE_TILE_MVM_NS * BT * lr_tiles / 1e6
            lr_total = lr_mvm + lr_upd

            ax.bar(xh[1 + ri], lr_total, w_bar, color=C_HYBRID_LR, edgecolor='white',
                   lw=0.4, zorder=3)
            ratio = tt_total / lr_total if lr_total > 0 else 0
            ax.text(xh[1 + ri], lr_total * 1.08, f'{ratio:.0f}x',
                    ha='center', va='bottom', fontsize=6, fontweight='bold', color='#555',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        ax.set_yscale('log')
        ax.set_xticks(xh)
        ax.set_xticklabels(x_labels_h, fontsize=6.5)
        tl = ax.get_xticklabels()
        tl[0].set_fontweight('bold')
        tl[0].set_color('#e67e22')
        ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
        if pi == 0:
            ax.set_ylabel('$\\Delta T_{step}$ [ms]', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.08, which='major', zorder=0)
        axes[1, pi].set_title(f'attention  |  $\\alpha_t$={alpha_t}\n{cite}',
                              fontsize=8, fontweight='bold', pad=6)

    # ── Bottom row: OPs (total, single bar), attention/ffn/all ──
    C_OPS_TT = '#e67e22'
    C_OPS_LR = '#2980b9'

    for col, target in enumerate(TARGETS):
        ax = axes[2, col]
        targeted = get_targeted_layers(inventory, target)

        x_labels_o = ['TT'] + [f'r={r}' for r in RANKS]
        n_x = len(x_labels_o)
        xo = np.arange(n_x)
        w_bar = 0.55

        # TT total ops
        _, tt = compute_ops(targeted, 8, gamma)
        tt_total = sum(tt[k] / 1e12 for k in COMP_KEYS)

        ax.bar(xo[0], tt_total, w_bar, color=C_OPS_TT, edgecolor='white',
               lw=0.4, zorder=3, alpha=0.7)
        ax.text(xo[0], tt_total * 1.08, f'{tt_total:.2f}T',
                ha='center', va='bottom', fontsize=6, fontweight='bold', color='#e67e22',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        # LR-TT total ops
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
    axes[0, 0].text(-0.28, 0.5, 'Latency\n(per-element)\noptimistic',
                    transform=axes[0, 0].transAxes, fontsize=6.5, fontweight='bold',
                    va='center', ha='center', rotation=90, color='#333')
    axes[1, 0].text(-0.28, 0.5, 'Latency\n(hybrid)\nconservative',
                    transform=axes[1, 0].transAxes, fontsize=6.5, fontweight='bold',
                    va='center', ha='center', rotation=90, color='#c0392b')
    axes[2, 0].text(-0.28, 0.5, 'Ops\ncount',
                    transform=axes[2, 0].transAxes, fontsize=6.5, fontweight='bold',
                    va='center', ha='center', rotation=90, color='#333')

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='white')
               for c in COMP_COLORS]
    handles += [
        plt.Rectangle((0, 0), 1, 1, facecolor='#e67e22', edgecolor='white', alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor='#2980b9', edgecolor='white'),
    ]
    labels = list(COMP_NAMES) + ['TT (hybrid total)', 'LR-TT (hybrid total)']
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
               handlelength=1.5, handletextpad=0.4, columnspacing=1.2)

    fig.suptitle('$\\gamma$=1  |  Top: Latency per-element (optimistic)  |  '
                 'Mid: Latency hybrid (MVM per-elem + Update per-tile, conservative)  |  '
                 'Bottom: Ops',
                 fontsize=8, fontweight='bold', y=1.0)

    path = os.path.join(OUT, "LRTT_VS_TT_COMBINED_ELEM_G1.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

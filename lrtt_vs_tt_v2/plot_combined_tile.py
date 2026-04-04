#!/usr/bin/env python3
"""Combined: top=Latency (per-tile update), bottom=OPs. γ=1, target=all.
3 α_t literature values as columns.
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


def compute_all(targeted, rank, gamma=1):
    """Return ops dict and tile counts for LR-TT and TT."""
    lr_ops = {'proj': 0, 'update': 0, 'visible': 0}
    tt_ops = {'proj': 0, 'update': 0, 'visible': 0}
    lr_tiles = 0
    tt_tiles = 0

    for l in targeted:
        M, N = l.M, l.N
        lr_ops['proj']   += BT * 2 * (rank * N + M * rank)
        lr_ops['update'] += BT * 2 * (M * rank + rank * N)
        if gamma == 1:
            lr_ops['visible'] += BT * 2 * (rank * N + M * rank)

        tt_ops['update'] += BT * 2 * M * N
        if gamma == 1:
            tt_ops['visible'] += BT * 2 * M * N

        lr_tiles += math.ceil(M / TILE) * math.ceil(rank / TILE) + \
                    math.ceil(rank / TILE) * math.ceil(N / TILE)
        tt_tiles += math.ceil(M / TILE) * math.ceil(N / TILE)

    return lr_ops, tt_ops, lr_tiles, tt_tiles


def latency_pertile(ops, n_tiles, alpha_t):
    """Compute latency: MVM=per-element, Update=per-tile."""
    t_proj = TAU_ACIM * ops['proj']
    t_vis  = TAU_ACIM * ops['visible']
    t_upd  = alpha_t * ALPINE_TILE_MVM_NS * BT * n_tiles
    return {'proj': t_proj / 1e6, 'update': t_upd / 1e6, 'visible': t_vis / 1e6}


def draw_panel(ax, targeted, comp_keys, comp_names, comp_colors,
               val_fn_lr, val_fn_tt, ylabel, unit_fmt, show_legend=False):
    """Grouped bars for LR-TT, TT as dashed line."""
    n_c = len(comp_keys)
    n_r = len(RANKS)
    x = np.arange(n_r)
    tw = 0.82
    bw = tw / n_c

    tt_total = sum(val_fn_tt(k) for k in comp_keys)
    ax.axhline(tt_total, color='#e67e22', ls='--', lw=1.2, alpha=0.7, zorder=2)

    # TT annotation
    tt_parts = [f'{cn.split(" ")[0]}: {unit_fmt(val_fn_tt(k))}'
                for k, cn in zip(comp_keys, comp_names) if val_fn_tt(k) > 0]
    ax.annotate(f'TT: {unit_fmt(tt_total)}\n' + '\n'.join(tt_parts),
                xy=(n_r - 0.5, tt_total),
                xytext=(n_r - 0.5, tt_total * 1.5),
                fontsize=5, ha='right', va='bottom', color='#333', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff8e1',
                          edgecolor='#e67e22', alpha=0.9, lw=0.5), zorder=8)

    lr_vals_all = []
    for rank in RANKS:
        lr_ops, _, lr_tiles, _ = compute_all(targeted, rank)
        lr_vals_all.append((lr_ops, lr_tiles))

    for ci, (k, cn, cc) in enumerate(zip(comp_keys, comp_names, comp_colors)):
        offset = -tw / 2 + (ci + 0.5) * bw
        vals = [val_fn_lr(lr_vals_all[i], k) for i in range(n_r)]
        ax.bar(x + offset, vals, bw * 0.92, color=cc, edgecolor='white', lw=0.4,
               zorder=3, label=cn if show_legend else '')

    for i in range(n_r):
        total = sum(val_fn_lr(lr_vals_all[i], k) for k in comp_keys)
        bar_top = max(val_fn_lr(lr_vals_all[i], k) for k in comp_keys
                      if val_fn_lr(lr_vals_all[i], k) > 0)
        ratio = tt_total / total if total > 0 else 0
        ax.text(x[i], bar_top * 1.12, f'{ratio:.0f}x',
                ha='center', va='bottom', fontsize=6, fontweight='bold', color='#555',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    ax.set_yscale('log')
    all_v = [val_fn_lr(lr_vals_all[i], k) for i in range(n_r)
             for k in comp_keys if val_fn_lr(lr_vals_all[i], k) > 0]
    if all_v:
        ax.set_ylim(min(all_v) * 0.3, tt_total * 3.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in RANKS], fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.08, which='major', zorder=0)


def main():
    inventory = build_layer_inventory()
    gamma = 1

    for target in TARGETS:
        targeted = get_targeted_layers(inventory, target)

        fig, axes = plt.subplots(2, 3, figsize=(10, 7.5),
                                 gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.30})

        # Precompute TT (fixed across α_t for ops, varies for latency)
        _, tt_ops, _, tt_tiles = compute_all(targeted, 8, gamma)

        for pi, (alpha_t, cite) in enumerate(ALPHA_LIT):
            # --- Top: Latency (per-tile update) ---
            tt_lat = latency_pertile(tt_ops, tt_tiles, alpha_t)

            draw_panel(
                axes[0, pi], targeted, COMP_KEYS, COMP_NAMES, COMP_COLORS,
                val_fn_lr=lambda lr_pair, k, at=alpha_t: latency_pertile(
                    lr_pair[0], lr_pair[1], at)[k],
                val_fn_tt=lambda k, tl=tt_lat: tl[k],
                ylabel='$\\Delta T_{step}$ [ms]' if pi == 0 else '',
                unit_fmt=lambda v: f'{v:.0f}ms',
                show_legend=(pi == 0),
            )
            axes[0, pi].set_title(f'$\\alpha_t$={alpha_t}\n{cite}',
                                  fontsize=8, fontweight='bold', pad=6)

            # --- Bottom: OPs (same for all α_t) ---
            draw_panel(
                axes[1, pi], targeted, COMP_KEYS, COMP_NAMES, COMP_COLORS,
                val_fn_lr=lambda lr_pair, k: lr_pair[0][k] / 1e12,
                val_fn_tt=lambda k, to=tt_ops: to[k] / 1e12,
                ylabel='$\\Delta$Ops [TOps]' if pi == 0 else '',
                unit_fmt=lambda v: f'{v:.2f}T',
            )

        # Row labels
        axes[0, 0].text(-0.3, 0.5, 'Latency\n(per-tile\nupdate)',
                        transform=axes[0, 0].transAxes, fontsize=7, fontweight='bold',
                        va='center', ha='center', rotation=90, color='#333')
        axes[1, 0].text(-0.3, 0.5, 'Ops\ncount',
                        transform=axes[1, 0].transAxes, fontsize=7, fontweight='bold',
                        va='center', ha='center', rotation=90, color='#333')

        # Legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='white')
            for c in COMP_COLORS
        ] + [Line2D([0], [0], color='#e67e22', ls='--', lw=1.2, alpha=0.7)]
        labels = COMP_NAMES + ['TT total']
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
                   handlelength=1.5, handletextpad=0.4, columnspacing=1.2)

        fig.suptitle(f'target={target}, $\\gamma$=1  |  '
                     f'Top: Latency (MVM per-elem, Update per-tile)  |  '
                     f'Bottom: Active ops',
                     fontsize=9.5, fontweight='bold', y=1.0)

        path = os.path.join(OUT, f"LRTT_VS_TT_COMBINED_TILE_{target}_G1.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combined figure: top=Latency (per-element), bottom=OPs breakdown.
Both γ=1, 3 panels (attention/ffn/all), TT as reference line.
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
    RANKS, TARGETS, BATCH_SIZE, S_PAD_DEFAULT, TAU_ACIM,
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

ALPHA_LITERATURE = [
    (0.25, 'Gokmen 2016 (RPU 4096)'),
    (1.4,  'Rasch 2024 (TTv2)'),
    (2.0,  'Gokmen 2016 (RPU 512)'),
]

COMP_KEYS   = ['proj', 'update', 'visible']
COMP_NAMES  = ['Projection', 'Update', 'Visible fwd']
COMP_COLORS = [C_PROJ, C_UPDATE, C_VIS]


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


def draw_grouped_panel(ax, targeted, comp_keys, comp_names, comp_colors,
                       value_fn_lr, value_fn_tt, ylabel, unit, gamma=1):
    """Draw grouped bars for LR-TT + TT reference line."""
    n_comps = len(comp_keys)
    n_ranks = len(RANKS)
    x = np.arange(n_ranks)
    total_w = 0.82
    bw = total_w / n_comps

    # TT
    _, tt = compute_ops(targeted, 8, gamma)
    tt_total = sum(value_fn_tt(tt, k) for k in comp_keys)

    # TT reference line
    ax.axhline(tt_total, color='#e67e22', ls='--', lw=1.2, alpha=0.7, zorder=2)

    # TT annotation
    tt_parts = [f'{cn.split(" ")[0]}: {value_fn_tt(tt, k):.2f}{unit}'
                for k, cn in zip(comp_keys, comp_names) if value_fn_tt(tt, k) > 0]
    ax.annotate(f'TT: {tt_total:.2f}{unit}\n' + '\n'.join(tt_parts),
                xy=(n_ranks - 0.5, tt_total),
                xytext=(n_ranks - 0.5, tt_total * 1.5),
                fontsize=5, ha='right', va='bottom',
                color='#333', style='italic',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='#fff8e1', edgecolor='#e67e22',
                          alpha=0.9, lw=0.5), zorder=8)

    # LR-TT grouped bars
    lr_list = []
    for rank in RANKS:
        lr, _ = compute_ops(targeted, rank, gamma)
        lr_list.append(lr)

    for ci, (k, cn, cc) in enumerate(zip(comp_keys, comp_names, comp_colors)):
        offset = -total_w / 2 + (ci + 0.5) * bw
        vals = [value_fn_lr(lr_list[i], k) for i in range(n_ranks)]
        ax.bar(x + offset, vals, bw * 0.92, color=cc,
               edgecolor='white', lw=0.4, zorder=3,
               label=cn)

    # Ratio annotations
    for i in range(n_ranks):
        total = sum(value_fn_lr(lr_list[i], k) for k in comp_keys)
        bar_top = max(value_fn_lr(lr_list[i], k) for k in comp_keys
                      if value_fn_lr(lr_list[i], k) > 0)
        ratio = tt_total / total if total > 0 else 0
        ax.text(x[i], bar_top * 1.12, f'{ratio:.0f}x',
                ha='center', va='bottom', fontsize=6, fontweight='bold',
                color='#555',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    # Axis
    ax.set_yscale('log')
    all_lr = [value_fn_lr(lr_list[i], k) for i in range(n_ranks)
              for k in comp_keys if value_fn_lr(lr_list[i], k) > 0]
    ax.set_ylim(min(all_lr) * 0.3, tt_total * 3.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in RANKS], fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.08, which='major', zorder=0)


def main():
    inventory = build_layer_inventory()
    gamma = 1

    for alpha_t in [1.4]:  # Main literature value
        fig, axes = plt.subplots(2, 3, figsize=(10, 7.5),
                                 gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35})

        lat_w = {
            'proj':    TAU_ACIM,
            'update':  alpha_t * TAU_ACIM,
            'visible': TAU_ACIM,
        }

        for col, target in enumerate(TARGETS):
            targeted = get_targeted_layers(inventory, target)

            # --- Top row: Latency (per-element, Uelem) ---
            ax_lat = axes[0, col]
            draw_grouped_panel(
                ax_lat, targeted, COMP_KEYS, COMP_NAMES, COMP_COLORS,
                value_fn_lr=lambda lr, k: lr[k] * lat_w[k] / 1e6,
                value_fn_tt=lambda tt, k: tt[k] * lat_w[k] / 1e6,
                ylabel='$\\Delta T_{step}$ [ms]' if col == 0 else '',
                unit='ms', gamma=gamma,
            )
            ax_lat.set_title(target, fontsize=9, fontweight='bold', pad=4)
            # Remove duplicate legends
            if col > 0:
                ax_lat.get_legend_handles_labels()
                for h in ax_lat.get_children():
                    pass  # keep all

            # --- Bottom row: OPs ---
            ax_ops = axes[1, col]
            draw_grouped_panel(
                ax_ops, targeted, COMP_KEYS, COMP_NAMES, COMP_COLORS,
                value_fn_lr=lambda lr, k: lr[k] / 1e12,
                value_fn_tt=lambda tt, k: tt[k] / 1e12,
                ylabel='$\\Delta$Ops [TOps]' if col == 0 else '',
                unit='T', gamma=gamma,
            )

        # Row labels
        axes[0, 0].text(-0.25, 0.5, 'Latency\n(per-element)', transform=axes[0, 0].transAxes,
                        fontsize=8, fontweight='bold', va='center', ha='center', rotation=90,
                        color='#333')
        axes[1, 0].text(-0.25, 0.5, 'Ops count', transform=axes[1, 0].transAxes,
                        fontsize=8, fontweight='bold', va='center', ha='center', rotation=90,
                        color='#333')

        # Single legend at bottom
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=C_PROJ, edgecolor='white'),
            plt.Rectangle((0, 0), 1, 1, facecolor=C_UPDATE, edgecolor='white'),
            plt.Rectangle((0, 0), 1, 1, facecolor=C_VIS, edgecolor='white'),
            Line2D([0], [0], color='#e67e22', ls='--', lw=1.2, alpha=0.7),
        ]
        labels = ['Projection', 'Update', 'Visible fwd', 'TT total']
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
                   handlelength=1.5, handletextpad=0.4, columnspacing=1.2)

        fig.suptitle(f'Adapter cost breakdown ($\\gamma$=1, $\\alpha_t$={alpha_t})\n'
                     f'Top: Latency [ms]  |  Bottom: Active ops [TOps]  |  '
                     f'BS={BATCH_SIZE}, S={S_PAD_DEFAULT}',
                     fontsize=10, fontweight='bold', y=1.0)

        path = os.path.join(OUT, f"LRTT_VS_TT_COMBINED_G1_A{str(alpha_t).replace('.','p')}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Adapter-only ops breakdown: TikiTaka vs LR-TT, ALPINE-calibrated.
Nature-style, log y-axis, stacked bars. Taller figure for visible segment clarity.
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

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    RANKS, TARGETS, BATCH_SIZE, S_PAD_DEFAULT, TAU_ACIM,
    LRTT_TRANSFER_EVERY, TIKITAKA_TRANSFER_EVERY,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
TE_LR = LRTT_TRANSFER_EVERY
TE_TT = TIKITAKA_TRANSFER_EVERY

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
    'legend.fontsize': 7,
    'legend.frameon': True,
    'legend.edgecolor': '#bbb',
    'legend.fancybox': False,
    'figure.dpi': 200,
})

C_PROJ   = '#c0392b'
C_UPDATE = '#e67e22'
C_VIS    = '#2980b9'
C_TR     = '#8e44ad'

COMP_KEYS   = ['proj', 'update', 'visible', 'transfer']
COMP_NAMES  = ['Projection', 'Update', 'Visible fwd', 'Transfer']
COMP_COLORS = [C_PROJ, C_UPDATE, C_VIS, C_TR]


def compute_adapter_ops(targeted, rank, gamma):
    lr = {'proj': 0, 'update': 0, 'visible': 0, 'transfer': 0}
    tt = {'proj': 0, 'update': 0, 'visible': 0, 'transfer': 0}
    for l in targeted:
        M, N = l.M, l.N
        lr['proj']     += BT * 2 * (rank * N + M * rank)
        lr['update']   += BT * 2 * (M * rank + rank * N)
        if gamma == 1:
            lr['visible'] += BT * 2 * (rank * N + M * rank)
        lr['transfer'] += (rank * M + rank * N + rank * M * N) * 2 / TE_LR
        tt['update']   += BT * 2 * M * N
        if gamma == 1:
            tt['visible'] += BT * 2 * M * N
        tt['transfer'] += 2 * M / TE_TT
    return lr, tt


def _draw_panel(ax, x, all_bars, w, col, ylabel, fmt_fn, y_bottom):
    """Log-scale stacked bars with ratio annotations."""
    n = len(x)
    tt_total = sum(all_bars[0])

    # Stack
    bottom = np.full(n, y_bottom)  # start from y_bottom (not 0) for log
    for j, (cname, ccolor) in enumerate(zip(COMP_NAMES, COMP_COLORS)):
        vals = np.array([all_bars[i][j] for i in range(n)])
        if vals.max() > 0:
            ax.bar(x, vals, w, bottom=bottom, color=ccolor,
                   edgecolor='#444', lw=0.25, zorder=3,
                   label=cname if col == 0 else '')
            bottom = bottom + vals

    ax.set_yscale('log')
    ax.set_ylim(bottom=y_bottom * 0.5)

    # Annotations
    for i in range(n):
        total = sum(all_bars[i])
        top_y = total + y_bottom  # actual bar top
        txt = fmt_fn(total)
        if i > 0 and total > 0:
            ratio = tt_total / total
            txt += f'\n{ratio:.0f}x'
        ax.text(x[i], top_y * 1.15, txt,
                ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                color='#333',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    # Style
    xlabels = ['TT'] + [f'r={r}' for r in RANKS]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6.5)
    tl = ax.get_xticklabels()
    tl[0].set_fontweight('bold')
    tl[0].set_color(C_UPDATE)
    ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
    if col == 0:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.10, which='major', zorder=0)
    ax.grid(axis='y', alpha=0.04, which='minor', zorder=0)


def main():
    inventory = build_layer_inventory()
    print(f"Layers: {len(inventory)}")

    n = 1 + len(RANKS)
    x = np.arange(n)
    w = 0.62

    # ==== Ops breakdown (TOps) ====
    for gamma in [0, 1]:
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.5))

        for col, target in enumerate(TARGETS):
            ax = axes[col]
            targeted = get_targeted_layers(inventory, target)

            all_bars = []
            _, tt = compute_adapter_ops(targeted, 8, gamma)
            all_bars.append([tt[k] / 1e12 for k in COMP_KEYS])
            for rank in RANKS:
                lr, _ = compute_adapter_ops(targeted, rank, gamma)
                all_bars.append([lr[k] / 1e12 for k in COMP_KEYS])

            _draw_panel(ax, x, all_bars, w, col, '$\\Delta$Ops [TOps]',
                        lambda v: f'{v:.2f}T', y_bottom=1e-4)
            ax.set_title(target, fontsize=9, fontweight='bold', pad=4)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
                   handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
        fig.suptitle(f'Adapter ops per training step  ($\\gamma$={gamma})',
                     fontsize=9.5, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95], w_pad=1.2)
        path = os.path.join(OUT, f"LRTT_VS_TT_OPS_BREAKDOWN_G{gamma}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")

    # ==== Latency breakdown (ms) ====
    for alpha_t in [1.0, 4.15]:
        for gamma in [0, 1]:
            fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.5))

            lat_w = {
                'proj': TAU_ACIM,
                'update': alpha_t * TAU_ACIM,
                'visible': TAU_ACIM,
                'transfer': TAU_ACIM,
            }

            for col, target in enumerate(TARGETS):
                ax = axes[col]
                targeted = get_targeted_layers(inventory, target)

                all_bars = []
                _, tt = compute_adapter_ops(targeted, 8, gamma)
                all_bars.append([tt[k] * lat_w[k] / 1e6 for k in COMP_KEYS])
                for rank in RANKS:
                    lr, _ = compute_adapter_ops(targeted, rank, gamma)
                    all_bars.append([lr[k] * lat_w[k] / 1e6 for k in COMP_KEYS])

                _draw_panel(ax, x, all_bars, w, col, '$\\Delta T_{step}$ [ms]',
                            lambda v: f'{v:.0f}ms', y_bottom=0.1)
                ax.set_title(target, fontsize=9, fontweight='bold', pad=4)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7,
                       bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
                       handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
            fig.suptitle(f'Adapter latency per training step  '
                         f'($\\gamma$={gamma},  $\\alpha_t$={alpha_t})',
                         fontsize=9.5, fontweight='bold', y=0.99)
            plt.tight_layout(rect=[0, 0.06, 1, 0.95], w_pad=1.2)
            a_str = str(alpha_t).replace('.', 'p')
            path = os.path.join(OUT,
                                f"LRTT_VS_TT_LATENCY_BREAKDOWN_G{gamma}_A{a_str}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

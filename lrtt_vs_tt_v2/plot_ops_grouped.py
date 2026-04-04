#!/usr/bin/env python3
"""Grouped bar breakdown: TikiTaka vs LR-TT, ALPINE-calibrated.
TT shown as annotated reference line; LR-TT bars zoomed in.
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


def main():
    inventory = build_layer_inventory()
    print(f"Layers: {len(inventory)}")

    n_ranks = len(RANKS)
    x = np.arange(n_ranks)
    x_labels = [f'r={r}' for r in RANKS]

    for gamma in [0, 1]:
        # Determine active components
        active = []
        for j, k in enumerate(COMP_KEYS):
            if k == 'visible' and gamma == 0:
                continue
            if k == 'transfer':
                continue
            active.append((j, k, COMP_NAMES[j], COMP_COLORS[j]))

        n_comps = len(active)
        total_w = 0.82
        bw = total_w / n_comps

        for mode in ['ops', 'latency']:
            for alpha_t in ([1.0] if mode == 'ops' else [0.25, 1.0, 1.4, 4.15]):
                lat_w = {
                    'proj': TAU_ACIM,
                    'update': alpha_t * TAU_ACIM,
                    'visible': TAU_ACIM,
                    'transfer': TAU_ACIM,
                }

                fig, axes = plt.subplots(1, 3, figsize=(10, 4.0))

                for col, target in enumerate(TARGETS):
                    ax = axes[col]
                    targeted = get_targeted_layers(inventory, target)

                    # Compute TT
                    _, tt = compute_adapter_ops(targeted, 8, gamma)

                    # Compute LR-TT per rank
                    lr_data = []
                    for rank in RANKS:
                        lr, _ = compute_adapter_ops(targeted, rank, gamma)
                        lr_data.append(lr)

                    # TT total for reference line
                    if mode == 'ops':
                        tt_vals = {k: tt[k] / 1e12 for k in COMP_KEYS}
                    else:
                        tt_vals = {k: tt[k] * lat_w[k] / 1e6 for k in COMP_KEYS}
                    tt_total = sum(tt_vals[k] for _, k, _, _ in active)

                    # TT component breakdown text
                    tt_parts = []
                    for _, k, cname, _ in active:
                        v = tt_vals[k]
                        if v > 0:
                            if mode == 'ops':
                                tt_parts.append(f'{cname}: {v:.2f}T')
                            else:
                                tt_parts.append(f'{cname}: {v:.0f}ms')

                    # Draw TT as horizontal reference lines per component
                    for _, k, cname, ccolor in active:
                        v = tt_vals[k]
                        if v > 0:
                            ax.axhline(v, color=ccolor, ls='--', lw=1.0, alpha=0.6, zorder=2)

                    # TT total reference line (bold)
                    ax.axhline(tt_total, color='#333', ls='-', lw=0.8, alpha=0.4, zorder=2)

                    # TT annotation box
                    if mode == 'ops':
                        tt_label = f'TT total: {tt_total:.2f}T'
                    else:
                        tt_label = f'TT total: {tt_total:.0f}ms'
                    tt_detail = '\n'.join(tt_parts)

                    ax.annotate(f'{tt_label}\n{tt_detail}',
                                xy=(n_ranks - 0.5, tt_total),
                                xytext=(n_ranks - 0.5, tt_total * 1.3),
                                fontsize=5.5, ha='right', va='bottom',
                                color='#333', style='italic',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='#fff8e1', edgecolor='#e67e22',
                                          alpha=0.9, lw=0.5),
                                zorder=8)

                    # Draw LR-TT grouped bars
                    for ci, (j, k, cname, ccolor) in enumerate(active):
                        offset = -total_w / 2 + (ci + 0.5) * bw

                        if mode == 'ops':
                            vals = [lr_data[i][k] / 1e12 for i in range(n_ranks)]
                        else:
                            vals = [lr_data[i][k] * lat_w[k] / 1e6 for i in range(n_ranks)]

                        ax.bar(x + offset, vals, bw * 0.92, color=ccolor,
                               edgecolor='white', lw=0.4, zorder=3,
                               label=cname if col == 0 else '')

                    # Annotate LR-TT totals + ratio
                    for i in range(n_ranks):
                        if mode == 'ops':
                            total = sum(lr_data[i][k] / 1e12 for _, k, _, _ in active)
                            bar_top = max(lr_data[i][k] / 1e12 for _, k, _, _ in active)
                        else:
                            total = sum(lr_data[i][k] * lat_w[k] / 1e6 for _, k, _, _ in active)
                            bar_top = max(lr_data[i][k] * lat_w[k] / 1e6 for _, k, _, _ in active)

                        ratio = tt_total / total if total > 0 else 0
                        if mode == 'ops':
                            txt = f'{total:.2f}T\n{ratio:.0f}x'
                        else:
                            txt = f'{total:.0f}ms\n{ratio:.0f}x'

                        ax.text(x[i], bar_top * 1.15, txt,
                                ha='center', va='bottom', fontsize=5.5,
                                fontweight='bold', color='#333',
                                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

                    # Axis styling
                    ax.set_yscale('log')
                    # Tight ylim focused on LR-TT range
                    all_lr_vals = []
                    for i in range(n_ranks):
                        for _, k, _, _ in active:
                            if mode == 'ops':
                                v = lr_data[i][k] / 1e12
                            else:
                                v = lr_data[i][k] * lat_w[k] / 1e6
                            if v > 0:
                                all_lr_vals.append(v)
                    if all_lr_vals:
                        y_lo = min(all_lr_vals) * 0.4
                        y_hi = tt_total * 2.5  # show TT line with some room
                        ax.set_ylim(y_lo, y_hi)

                    ax.set_xticks(x)
                    ax.set_xticklabels(x_labels, fontsize=7)
                    ax.set_title(target, fontsize=9, fontweight='bold', pad=4)
                    if col == 0:
                        if mode == 'ops':
                            ax.set_ylabel('$\\Delta$Ops [TOps]', fontsize=8)
                        else:
                            ax.set_ylabel('$\\Delta T_{step}$ [ms]', fontsize=8)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='y', alpha=0.08, which='major', zorder=0)

                handles, labels = axes[0].get_legend_handles_labels()
                # Add TT reference to legend
                from matplotlib.lines import Line2D
                handles.append(Line2D([0], [0], color='#333', ls='--', lw=1.0, alpha=0.6))
                labels.append('TT component (dashed line)')
                fig.legend(handles, labels, loc='lower center',
                           ncol=n_comps + 1, fontsize=6.5,
                           bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
                           handlelength=1.5, handletextpad=0.4, columnspacing=1.0)

                if mode == 'ops':
                    title = f'Adapter ops per training step  ($\\gamma$={gamma})'
                else:
                    title = (f'Adapter latency per training step  '
                             f'($\\gamma$={gamma},  $\\alpha_t$={alpha_t})')
                fig.suptitle(title, fontsize=9.5, fontweight='bold', y=0.99)
                plt.tight_layout(rect=[0, 0.06, 1, 0.95], w_pad=1.5)

                if mode == 'ops':
                    fname = f"LRTT_VS_TT_OPS_GROUPED_G{gamma}.png"
                else:
                    a_str = str(alpha_t).replace('.', 'p')
                    fname = f"LRTT_VS_TT_LATENCY_GROUPED_G{gamma}_A{a_str}.png"

                path = os.path.join(OUT, fname)
                fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

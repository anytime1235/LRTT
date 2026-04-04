#!/usr/bin/env python3
"""3-panel figure: attention/ffn/all, γ=0 and γ=1, α_t from 3 literature sources.
Two update models: per-element (optimistic) and per-tile (conservative).
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
    LRTT_TRANSFER_EVERY, TIKITAKA_TRANSFER_EVERY,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
TE_LR = LRTT_TRANSFER_EVERY
TE_TT = TIKITAKA_TRANSFER_EVERY
TILE = ALPINE_TILE_SIZE  # 256

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
    (0.25, 'Gokmen 2016\n(RPU 4096)'),
    (1.4,  'Rasch 2024\n(TTv2)'),
    (2.0,  'Gokmen 2016\n(RPU 512)'),
]


def compute_latency(targeted, rank, gamma, alpha_t, update_model='per_element'):
    """Compute adapter latency components.

    update_model:
      'per_element': MVM per-element, Update per-element (optimistic)
      'per_tile':    MVM per-element, Update per-tile (conservative)
    """
    lr = {'proj': 0.0, 'update': 0.0, 'visible': 0.0}
    tt = {'proj': 0.0, 'update': 0.0, 'visible': 0.0}

    for l in targeted:
        M, N = l.M, l.N

        # --- MVM: always per-element (column-serial, physically correct) ---
        # Projection
        lr['proj'] += TAU_ACIM * BT * 2 * (rank * N + M * rank)
        # Visible (γ=1)
        if gamma == 1:
            lr['visible'] += TAU_ACIM * BT * 2 * (rank * N + M * rank)
            tt['visible'] += TAU_ACIM * BT * 2 * M * N

        # --- Update: depends on model ---
        if update_model == 'per_element':
            lr['update'] += alpha_t * TAU_ACIM * BT * 2 * (M * rank + rank * N)
            tt['update'] += alpha_t * TAU_ACIM * BT * 2 * M * N
        else:  # per_tile
            # t_upd_per_tile = alpha_t * t_tile_mvm (scaled from ALPINE anchor)
            t_upd_tile = alpha_t * ALPINE_TILE_MVM_NS  # ns per tile per sample

            # LR-TT: A [M×r] + B [r×N] tile counts
            n_tiles_A = math.ceil(M / TILE) * math.ceil(rank / TILE)
            n_tiles_B = math.ceil(rank / TILE) * math.ceil(N / TILE)
            lr['update'] += t_upd_tile * BT * (n_tiles_A + n_tiles_B)

            # TT: U [M×N] tile count
            n_tiles_U = math.ceil(M / TILE) * math.ceil(N / TILE)
            tt['update'] += t_upd_tile * BT * n_tiles_U

    # Convert to ms
    lr_ms = {k: v / 1e6 for k, v in lr.items()}
    tt_ms = {k: v / 1e6 for k, v in tt.items()}
    return lr_ms, tt_ms


def make_plot(gamma, target, update_model):
    inventory = build_layer_inventory()
    targeted = get_targeted_layers(inventory, target)
    n_ranks = len(RANKS)
    x = np.arange(n_ranks)
    x_labels = [f'r={r}' for r in RANKS]

    if gamma == 0:
        comp_keys = ['proj', 'update']
        comp_names = ['Projection (MVM)', 'Update (pulsed)']
        comp_colors = [C_PROJ, C_UPDATE]
    else:
        comp_keys = ['proj', 'update', 'visible']
        comp_names = ['Projection (MVM)', 'Update (pulsed)', 'Visible fwd (MVM)']
        comp_colors = [C_PROJ, C_UPDATE, C_VIS]

    n_comps = len(comp_keys)
    total_w = 0.82
    bw = total_w / n_comps

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))

    for pi, (alpha_t, cite) in enumerate(ALPHA_LITERATURE):
        ax = axes[pi]

        # TT
        _, tt_ms = compute_latency(targeted, 8, gamma, alpha_t, update_model)
        tt_total = sum(tt_ms[k] for k in comp_keys)

        # TT reference line
        ax.axhline(tt_total, color='#e67e22', ls='--', lw=1.2, alpha=0.7, zorder=2)

        # TT annotation
        tt_parts = [f'{cn.split(" (")[0]}: {tt_ms[k]:.0f}ms'
                    for k, cn in zip(comp_keys, comp_names) if tt_ms[k] > 0]
        ax.annotate(f'TT total: {tt_total:.0f}ms\n' + '\n'.join(tt_parts),
                    xy=(n_ranks - 0.5, tt_total),
                    xytext=(n_ranks - 0.5, tt_total * 1.4),
                    fontsize=5.5, ha='right', va='bottom',
                    color='#333', style='italic',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#fff8e1', edgecolor='#e67e22',
                              alpha=0.9, lw=0.5), zorder=8)

        # LR-TT grouped bars
        lr_list = []
        for rank in RANKS:
            lr_ms, _ = compute_latency(targeted, rank, gamma, alpha_t, update_model)
            lr_list.append(lr_ms)

        for ci, (k, cn, cc) in enumerate(zip(comp_keys, comp_names, comp_colors)):
            offset = -total_w / 2 + (ci + 0.5) * bw
            vals = [lr_list[i][k] for i in range(n_ranks)]
            ax.bar(x + offset, vals, bw * 0.92, color=cc,
                   edgecolor='white', lw=0.4, zorder=3,
                   label=cn if pi == 0 else '')

        # Annotations
        for i in range(n_ranks):
            total = sum(lr_list[i][k] for k in comp_keys)
            bar_top = max(lr_list[i][k] for k in comp_keys if lr_list[i][k] > 0)
            ratio = tt_total / total if total > 0 else 0
            ax.text(x[i], bar_top * 1.15, f'{total:.0f}ms\n{ratio:.0f}x',
                    ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                    color='#333',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        # Axis
        ax.set_yscale('log')
        all_lr = [lr_list[i][k] for i in range(n_ranks) for k in comp_keys
                  if lr_list[i][k] > 0]
        if all_lr:
            ax.set_ylim(min(all_lr) * 0.3, tt_total * 3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_title(f'$\\alpha_t$={alpha_t}\n{cite}', fontsize=8, fontweight='bold', pad=6)
        if pi == 0:
            ax.set_ylabel('$\\Delta T_{step}$ [ms]', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.08, which='major', zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#e67e22', ls='--', lw=1.2, alpha=0.7))
    labels.append('TT total $\\Delta T$')
    fig.legend(handles, labels, loc='lower center', ncol=n_comps + 1, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
               handlelength=1.5, handletextpad=0.4, columnspacing=1.0)

    model_label = 'per-element (optimistic)' if update_model == 'per_element' else 'per-tile (conservative)'
    fig.suptitle(f'target={target},  $\\gamma$={gamma}  |  Update model: {model_label}\n'
                 f'MVM=per-element, Update={model_label.split(" ")[0]}  |  '
                 f'BS={BATCH_SIZE}, S={S_PAD_DEFAULT}',
                 fontsize=9, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.07, 1, 0.93], w_pad=1.5)

    um = 'elem' if update_model == 'per_element' else 'tile'
    path = os.path.join(OUT, f"LRTT_VS_TT_ALPHA_{target}_G{gamma}_U{um}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    for target in TARGETS:
        for gamma in [0, 1]:
            for um in ['per_element', 'per_tile']:
                make_plot(gamma, target, um)

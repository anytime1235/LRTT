#!/usr/bin/env python3
"""TOPS comparison: TikiTaka vs LR-TT across attention/ffn/all.
Both per-element and per-tile update models.
TOPS = Total active ops / DeltaT
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
import matplotlib.patches as mpatches

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    RANKS, TARGETS, BATCH_SIZE, S_PAD_DEFAULT, TAU_ACIM,
    ALPINE_TILE_MVM_NS, ALPINE_TILE_SIZE,
    LRTT_TRANSFER_EVERY, TIKITAKA_TRANSFER_EVERY,
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

C_TT = '#e67e22'
C_LR = '#2980b9'


def compute_ops_and_latency(targeted, rank, gamma, alpha_t, update_model):
    """Returns (total_ops, DeltaT_ns) for LR-TT and TT."""

    lr_ops = 0
    tt_ops = 0
    lr_dt = 0.0
    tt_dt = 0.0

    for l in targeted:
        M, N = l.M, l.N

        # --- Ops (always per-element, = actual work done) ---
        lr_proj_ops = BT * 2 * (rank * N + M * rank)
        lr_upd_ops  = BT * 2 * (M * rank + rank * N)
        lr_vis_ops  = BT * 2 * (rank * N + M * rank) if gamma == 1 else 0

        tt_upd_ops = BT * 2 * M * N
        tt_vis_ops = BT * 2 * M * N if gamma == 1 else 0

        lr_ops += lr_proj_ops + lr_upd_ops + lr_vis_ops
        tt_ops += tt_upd_ops + tt_vis_ops

        # --- Latency ---
        # MVM: always per-element
        lr_dt += TAU_ACIM * lr_proj_ops
        if gamma == 1:
            lr_dt += TAU_ACIM * lr_vis_ops
            tt_dt += TAU_ACIM * tt_vis_ops

        # Update: per-element or per-tile
        if update_model == 'per_element':
            lr_dt += alpha_t * TAU_ACIM * lr_upd_ops
            tt_dt += alpha_t * TAU_ACIM * tt_upd_ops
        else:
            t_upd_tile = alpha_t * ALPINE_TILE_MVM_NS
            n_A = math.ceil(M / TILE) * math.ceil(rank / TILE)
            n_B = math.ceil(rank / TILE) * math.ceil(N / TILE)
            n_U = math.ceil(M / TILE) * math.ceil(N / TILE)
            lr_dt += t_upd_tile * BT * (n_A + n_B)
            tt_dt += t_upd_tile * BT * n_U

    return lr_ops, lr_dt, tt_ops, tt_dt


def main():
    inventory = build_layer_inventory()
    print(f"Layers: {len(inventory)}")

    x_labels = ['TT'] + [f'r={r}' for r in RANKS]
    n = len(x_labels)
    x = np.arange(n)

    for alpha_t in [0.25, 1.0, 1.4, 2.0]:
        for gamma in [0, 1]:
            for update_model in ['per_element', 'per_tile']:
                fig, axes = plt.subplots(1, 3, figsize=(10, 4.0))

                for col, target in enumerate(TARGETS):
                    ax = axes[col]
                    targeted = get_targeted_layers(inventory, target)

                    tops_vals = []

                    # TT
                    _, _, tt_ops, tt_dt = compute_ops_and_latency(
                        targeted, 8, gamma, alpha_t, update_model)
                    tt_tops = tt_ops / tt_dt * 1e-3 if tt_dt > 0 else 0  # TOp/s
                    tops_vals.append(tt_tops)

                    # LR-TT per rank
                    for rank in RANKS:
                        lr_ops, lr_dt, _, _ = compute_ops_and_latency(
                            targeted, rank, gamma, alpha_t, update_model)
                        lr_tops = lr_ops / lr_dt * 1e-3 if lr_dt > 0 else 0
                        tops_vals.append(lr_tops)

                    # Colors
                    colors = [C_TT] + [C_LR] * len(RANKS)

                    bars = ax.bar(x, tops_vals, 0.6, color=colors,
                                  edgecolor='white', lw=0.4, zorder=3)

                    # Labels
                    for i in range(n):
                        ax.text(x[i], tops_vals[i] * 1.05, f'{tops_vals[i]:.2f}',
                                ha='center', va='bottom', fontsize=5.5,
                                fontweight='bold', color='#333',
                                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

                    ax.set_xticks(x)
                    ax.set_xticklabels(x_labels, fontsize=7)
                    tl = ax.get_xticklabels()
                    tl[0].set_fontweight('bold')
                    tl[0].set_color(C_TT)
                    ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
                    ax.set_title(target, fontsize=9, fontweight='bold', pad=4)
                    if col == 0:
                        ax.set_ylabel('TOPS [TOp/s]', fontsize=8)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='y', alpha=0.08, zorder=0)
                    ax.set_ylim(bottom=0)

                handles = [
                    mpatches.Patch(facecolor=C_TT, label='TikiTaka'),
                    mpatches.Patch(facecolor=C_LR, label='LR-TT'),
                ]
                fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7,
                           bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc')

                um_label = 'per-element' if update_model == 'per_element' else 'per-tile'
                fig.suptitle(f'Adapter TOPS (= Ops / $\\Delta T$)  |  '
                             f'$\\gamma$={gamma},  $\\alpha_t$={alpha_t}\n'
                             f'Update model: {um_label}  |  '
                             f'BS={BATCH_SIZE}, S={S_PAD_DEFAULT}',
                             fontsize=9, fontweight='bold', y=1.01)
                plt.tight_layout(rect=[0, 0.06, 1, 0.94], w_pad=1.5)

                um = 'elem' if update_model == 'per_element' else 'tile'
                a_str = str(alpha_t).replace('.', 'p')
                path = os.path.join(OUT, f"LRTT_VS_TT_TOPS_G{gamma}_A{a_str}_U{um}.png")
                fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper-ready figure: 3 rows (Ops, Tiles, Energy).
Single x-axis: TT_attn | r=1..64_attn | TT_ffn | r=1..64_ffn | TT_all | r=1..64_all
γ=0 main, tile=512×512, κ_e=0.5
"""

import os, sys, math
import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
COST_MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "cost_model")
sys.path.insert(0, COST_MODEL_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    BATCH_SIZE, S_PAD_DEFAULT, EPS_ACIM_PJ,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
T = 512
KAPPA_E = 0.5
RANKS = [1, 8, 16, 32, 64]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.linewidth': 0.5, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.major.width': 0.4, 'ytick.major.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 200,
})

C_TT = '#e67e22'
C_LR = '#2980b9'
TARGETS = ['attention', 'ffn', 'all']


def compute_metrics(targeted, rank, gamma):
    n_layers = len(targeted)
    ops_lr = 0; ops_tt = 0
    energy_lr = 0.0; energy_tt = 0.0

    for l in targeted:
        M, N = l.M, l.N
        proj = BT * 2 * (rank * N + M * rank)
        upd_lr = BT * 2 * (M * rank + rank * N)
        upd_tt = BT * 2 * M * N
        vis_lr = BT * 2 * (rank * N + M * rank) if gamma == 1 else 0
        vis_tt = BT * 2 * M * N if gamma == 1 else 0

        ops_lr += proj + upd_lr + vis_lr
        ops_tt += upd_tt + vis_tt

        energy_lr += EPS_ACIM_PJ * (proj + vis_lr) / 1e6
        energy_lr += KAPPA_E * EPS_ACIM_PJ * upd_lr / 1e6
        energy_tt += EPS_ACIM_PJ * vis_tt / 1e6
        energy_tt += KAPPA_E * EPS_ACIM_PJ * upd_tt / 1e6

    # Area: shape-grouped packing
    shape_counts = Counter((l.M, l.N) for l in targeted)
    adapters_per_tile = max(1, T // rank) if rank < T else 1
    area_lr = 0
    for (M_s, N_s), count in shape_counts.items():
        a_rows = math.ceil(M_s / T)
        b_cols = math.ceil(N_s / T)
        a_packed = math.ceil(count / adapters_per_tile) * a_rows
        b_packed = math.ceil(count / adapters_per_tile) * b_cols
        area_lr += a_packed + b_packed
    area_tt = sum(math.ceil(l.M / T) * math.ceil(l.N / T) for l in targeted)

    return ops_lr, ops_tt, area_lr, area_tt, energy_lr, energy_tt


def main():
    inventory = build_layer_inventory()

    # Build unified x-axis: [TT, r1..r64] × 3 targets, with gaps
    n_per_group = 1 + len(RANKS)  # TT + ranks
    gap = 1.5  # gap between target groups
    group_width = n_per_group

    x_positions = []
    x_tick_labels = []
    group_centers = []
    group_boundaries = []

    for gi, target in enumerate(TARGETS):
        offset = gi * (group_width + gap)
        group_x = [offset + i for i in range(n_per_group)]
        x_positions.extend(group_x)
        x_tick_labels.extend(['TT'] + [f'r={r}' for r in RANKS])
        group_centers.append(offset + n_per_group / 2 - 0.5)
        if gi > 0:
            group_boundaries.append(offset - gap / 2)

    x_positions = np.array(x_positions)
    n_total = len(x_positions)

    for gamma in [0, 1]:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.subplots_adjust(hspace=0.15)

        # Compute all data
        all_ops = []; all_tiles = []; all_energy = []
        all_colors = []
        tt_vals = {}

        for gi, target in enumerate(TARGETS):
            targeted = get_targeted_layers(inventory, target)
            _, ops_tt, _, area_tt, _, energy_tt = compute_metrics(targeted, 8, gamma)
            tt_vals[target] = {'ops': ops_tt, 'area': area_tt, 'energy': energy_tt}

            all_ops.append(ops_tt / 1e12)
            all_tiles.append(area_tt)
            all_energy.append(energy_tt)
            all_colors.append(C_TT)

            for rank in RANKS:
                o_lr, _, a_lr, _, e_lr, _ = compute_metrics(targeted, rank, gamma)
                all_ops.append(o_lr / 1e12)
                all_tiles.append(a_lr)
                all_energy.append(e_lr)
                all_colors.append(C_LR)

        # Row data
        row_data = [
            (all_ops, 'Ops [TOps]', '(a) Active Operations'),
            (all_tiles, 'Tiles (packed)', '(b) Physical Tile Count'),
            (all_energy, 'Energy [$\\mu$J]', '(c) Energy'),
        ]

        for row, (vals, ylabel, title) in enumerate(row_data):
            ax = axes[row]
            ax.bar(x_positions, vals, 0.7, color=all_colors, edgecolor='white',
                   lw=0.4, zorder=3)

            # Ratio annotations
            idx = 0
            for gi, target in enumerate(TARGETS):
                tt_v = vals[idx]
                # TT label
                if row == 0:
                    txt = f'{tt_v:.2f}T'
                elif row == 1:
                    txt = f'{tt_v:.0f}'
                else:
                    txt = f'{tt_v:.0f}'
                ax.text(x_positions[idx], tt_v * 1.12, txt,
                        ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                        color='#e67e22',
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                idx += 1

                # LR-TT labels
                for ri, rank in enumerate(RANKS):
                    lr_v = vals[idx]
                    ratio = tt_v / lr_v if lr_v > 0 else 0
                    ax.text(x_positions[idx], lr_v * 1.12, f'{ratio:.0f}x',
                            ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                            color='#555',
                            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                    idx += 1

            ax.set_yscale('log')
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=9, fontweight='bold', pad=4, loc='left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.08, which='major', zorder=0)

            # Group separators
            for bx in group_boundaries:
                ax.axvline(bx, color='#ccc', ls='-', lw=0.8, zorder=0)

            # Target labels (top row only)
            if row == 0:
                for gi, target in enumerate(TARGETS):
                    ax.text(group_centers[gi], ax.get_ylim()[1] * 0.7,
                            target, ha='center', va='top',
                            fontsize=10, fontweight='bold', color='#333',
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        # X-axis labels (bottom only)
        axes[-1].set_xticks(x_positions)
        axes[-1].set_xticklabels(x_tick_labels, fontsize=6, rotation=0)
        for i in range(n_total):
            tl = axes[-1].get_xticklabels()
            if x_tick_labels[i] == 'TT':
                tl[i].set_fontweight('bold')
                tl[i].set_color(C_TT)

        # Legend
        handles = [
            mpatches.Patch(facecolor=C_TT, label='TikiTaka (full-rank)'),
            mpatches.Patch(facecolor=C_LR, label='LR-TT (low-rank)'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc')

        fig.suptitle(f'LR-TT vs TikiTaka  |  $\\gamma$={gamma}  |  '
                     f'Tile={T}×{T}  |  $\\kappa_e$={KAPPA_E} (Gokmen 2016)\n'
                     f'Tiles: multi-layer column-packing  |  '
                     f'Energy: active-cell  |  '
                     f'Nx = TT/LR-TT ratio',
                     fontsize=9, fontweight='bold', y=1.0)

        path = os.path.join(OUT, f"PAPER_FINAL_G{gamma}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

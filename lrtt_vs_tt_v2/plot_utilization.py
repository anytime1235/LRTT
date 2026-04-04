#!/usr/bin/env python3
"""Per-tile utilization model + TOPS.
Top row: Latency (tile util 100%, 75%, 50%)
Bottom row: TOPS = Ops / Latency (at each util level)
α_t = 0.625, tile = 512×512
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

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    RANKS, TARGETS, BATCH_SIZE, S_PAD_DEFAULT,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
T = 512
t_tile = 400.0  # ns per full 512×512 tile MVM (ALPINE-scaled)
ALPHA_T = 0.625

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.linewidth': 0.5, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.major.width': 0.4, 'ytick.major.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 200,
})

UTIL_SCENARIOS = [1.00, 0.75, 0.50]
UTIL_COLORS = ['#2980b9', '#27ae60', '#8e44ad']
UTIL_LABELS = ['100%', '75%', '50%']
C_TT = '#e67e22'


def compute_latency_and_ops(targeted, rank, gamma, util_factor=1.0):
    """Returns (latency_ms, total_ops) for LR-TT and TT."""
    lr_ms = 0.0
    tt_ms = 0.0
    lr_ops = 0
    tt_ops = 0

    for l in targeted:
        M, N = l.M, l.N

        # TT
        tt_tiles = math.ceil(M / T) * math.ceil(N / T)
        tt_nat_util = (M * N) / (tt_tiles * T * T)
        tt_ms += ALPHA_T * t_tile * tt_nat_util * tt_tiles * BT / 1e6
        tt_ops += BT * 2 * M * N
        if gamma == 1:
            tt_ms += t_tile * tt_nat_util * tt_tiles * BT / 1e6
            tt_ops += BT * 2 * M * N

        # LR-TT
        # opt_util = r/T = fraction of tile columns used with perfect packing
        # util_factor = packing efficiency (1.0=perfect, 0.5=half columns wasted)
        # effective tile fraction = opt_util / util_factor
        #   (lower efficiency → need more tile time → larger fraction)
        # Capped at 1.0 (can't exceed full tile)
        opt_util = min(rank, T) / T
        eff_frac = min(opt_util / util_factor, 1.0)
        a_tiles = math.ceil(M / T)
        b_tiles = math.ceil(N / T)

        # Projection
        lr_ms += t_tile * eff_frac * (a_tiles + b_tiles) * BT / 1e6
        lr_ops += BT * 2 * (rank * N + M * rank)
        # Update
        lr_ms += ALPHA_T * t_tile * eff_frac * (a_tiles + b_tiles) * BT / 1e6
        lr_ops += BT * 2 * (M * rank + rank * N)
        # Visible
        if gamma == 1:
            lr_ms += t_tile * eff_frac * (a_tiles + b_tiles) * BT / 1e6
            lr_ops += BT * 2 * (rank * N + M * rank)

    return lr_ms, tt_ms, lr_ops, tt_ops


def main():
    inventory = build_layer_inventory()

    x_labels = ['TT'] + [f'r={r}' for r in RANKS]
    n_x = len(x_labels)
    x = np.arange(n_x)

    for gamma in [0, 1]:
        fig, axes = plt.subplots(2, 3, figsize=(10, 8),
                                 gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35},
                                 sharey='row')

        for col, target in enumerate(TARGETS):
            ax_lat = axes[0, col]
            ax_tops = axes[1, col]
            targeted = get_targeted_layers(inventory, target)

            # TT (same for all util)
            _, tt_ms, _, tt_ops = compute_latency_and_ops(targeted, 8, gamma, 1.0)
            tt_tops = tt_ops / (tt_ms * 1e6) * 1e-3 if tt_ms > 0 else 0  # TOp/s

            n_util = len(UTIL_SCENARIOS)
            total_w = 0.75
            bw_group = total_w / n_util

            # TT bar (latency)
            ax_lat.bar(x[0], tt_ms, 0.5, color=C_TT, edgecolor='white', lw=0.4, zorder=3)
            ax_lat.text(x[0], tt_ms * 1.08, f'{tt_ms:.0f}ms',
                        ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                        color='#e67e22',
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

            # TT bar (TOPS)
            ax_tops.bar(x[0], tt_tops, 0.5, color=C_TT, edgecolor='white', lw=0.4, zorder=3)
            ax_tops.text(x[0], tt_tops * 1.08, f'{tt_tops:.2f}',
                         ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                         color='#e67e22',
                         path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

            # LR-TT per util
            for ui, (util, color, ulabel) in enumerate(zip(UTIL_SCENARIOS, UTIL_COLORS, UTIL_LABELS)):
                offset = -total_w / 2 + (ui + 0.5) * bw_group

                for ri, rank in enumerate(RANKS):
                    lr_ms, _, lr_ops, _ = compute_latency_and_ops(targeted, rank, gamma, util)
                    lr_tops = lr_ops / (lr_ms * 1e6) * 1e-3 if lr_ms > 0 else 0

                    # Latency bar
                    ax_lat.bar(x[1 + ri] + offset, lr_ms, bw_group * 0.9,
                               color=color, edgecolor='white', lw=0.3, zorder=3,
                               label=f'{ulabel} util.' if (col == 0 and ri == 0) else '')

                    # TOPS bar
                    ax_tops.bar(x[1 + ri] + offset, lr_tops, bw_group * 0.9,
                                color=color, edgecolor='white', lw=0.3, zorder=3)

                # Ratio label (100% util only, to avoid clutter)
                if ui == 0:
                    for ri, rank in enumerate(RANKS):
                        lr_ms_100, _, _, _ = compute_latency_and_ops(targeted, rank, gamma, 1.0)
                        ratio = tt_ms / lr_ms_100 if lr_ms_100 > 0 else 0
                        bar_top = max(
                            compute_latency_and_ops(targeted, rank, gamma, u)[0]
                            for u in UTIL_SCENARIOS)
                        ax_lat.text(x[1 + ri], bar_top * 1.12, f'{ratio:.0f}x',
                                    ha='center', va='bottom', fontsize=5.5,
                                    fontweight='bold', color='#555',
                                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

            # Style latency
            ax_lat.set_yscale('log')
            ax_lat.set_xticks(x)
            ax_lat.set_xticklabels(x_labels, fontsize=6.5)
            tl = ax_lat.get_xticklabels()
            tl[0].set_fontweight('bold')
            tl[0].set_color(C_TT)
            ax_lat.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
            ax_lat.set_title(target, fontsize=9, fontweight='bold', pad=4)
            if col == 0:
                ax_lat.set_ylabel('$\\Delta T_{step}$ [ms]', fontsize=8)
            ax_lat.spines['top'].set_visible(False)
            ax_lat.spines['right'].set_visible(False)
            ax_lat.grid(axis='y', alpha=0.08, which='major', zorder=0)

            # Style TOPS
            ax_tops.set_xticks(x)
            ax_tops.set_xticklabels(x_labels, fontsize=6.5)
            tl = ax_tops.get_xticklabels()
            tl[0].set_fontweight('bold')
            tl[0].set_color(C_TT)
            ax_tops.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
            if col == 0:
                ax_tops.set_ylabel('TOPS [TOp/s]', fontsize=8)
            ax_tops.spines['top'].set_visible(False)
            ax_tops.spines['right'].set_visible(False)
            ax_tops.grid(axis='y', alpha=0.08, zorder=0)
            ax_tops.set_ylim(bottom=0)

        # Row labels
        axes[0, 0].text(-0.25, 0.5, 'Latency',
                        transform=axes[0, 0].transAxes, fontsize=7, fontweight='bold',
                        va='center', ha='center', rotation=90, color='#333')
        axes[1, 0].text(-0.25, 0.5, 'TOPS',
                        transform=axes[1, 0].transAxes, fontsize=7, fontweight='bold',
                        va='center', ha='center', rotation=90, color='#333')

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C_TT)]
        labels_leg = ['TikiTaka']
        for color, ulabel in zip(UTIL_COLORS, UTIL_LABELS):
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color))
            labels_leg.append(f'LR-TT {ulabel} util.')
        fig.legend(handles, labels_leg, loc='lower center', ncol=4, fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc',
                   handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

        fig.suptitle(f'Per-tile utilization model  |  $\\gamma$={gamma},  '
                     f'$\\alpha_t$={ALPHA_T}\n'
                     f'Tile={T}×{T}  |  Top: Latency  |  Bottom: TOPS (=Ops/$\\Delta T$)',
                     fontsize=9, fontweight='bold', y=1.0)
        plt.tight_layout(rect=[0, 0.06, 1, 0.94], w_pad=1.5)
        path = os.path.join(OUT, f"LRTT_VS_TT_UTIL_G{gamma}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

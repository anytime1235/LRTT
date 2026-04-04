#!/usr/bin/env python3
"""4 metrics with physical tile packing across layers.
Ops, Area(packed tiles), Latency(hybrid), Energy(per-element).
Attention only, 2×2 layout.
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
    RANKS, BATCH_SIZE, S_PAD_DEFAULT,
    TAU_ACIM, EPS_ACIM_PJ,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
T = 512
t_tile = 400.0       # ns per 512×512 MVM
ALPHA_T = 0.625      # pure pulse ratio
KAPPA_E = 1.0        # energy ratio

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


def compute_packed_metrics(targeted, rank, gamma):
    """Compute all metrics with physical tile packing across layers.

    Packing: multiple layers' adapters share one tile column-wise.
    - adapters_per_tile = T // r (how many fit in one tile)
    - Physical tiles = ceil(n_layers / adapters_per_tile) × tile_rows

    Latency hybrid:
    - MVM: per-element (column-serial)
    - Update: per physical packed tile (one pulse cycle per tile, updates all packed adapters simultaneously?
      NO — different gradients → sequential. But the tile is already packed, so
      update events = n_layers (not n_physical_tiles). Each event = one pulse cycle on one tile.
      So update latency = n_layers × t_tile_update × tile_rows_per_layer × BT

    Actually for update:
    - Each layer's adapter needs its own pulse cycle per tile-row
    - n_update_events = n_layers × tile_rows_per_layer (for A) + n_layers × tile_cols_per_layer (for B)
    - Each event = ALPHA_T × t_tile
    - Packing doesn't reduce update events (different gradients per layer)

    Area: physical packed tiles (packing DOES reduce this)
    """
    n_layers = len(targeted)
    adapters_per_tile = max(1, T // rank) if rank < T else 1

    # Per-layer tile geometry
    # Collect unique (M, N) shapes
    ops_total = {'lr': 0, 'tt': 0}
    energy = {'lr': 0.0, 'tt': 0.0}
    update_events_lr = 0
    update_events_tt = 0

    # Physical packed tiles (across all layers)
    # Group layers by M (row dimension) for A, by N for B
    # For simplicity, sum tile-rows and tile-cols per layer
    a_tile_rows_total = 0
    b_tile_cols_total = 0
    tt_tiles_total = 0

    for l in targeted:
        M, N = l.M, l.N

        # Ops (per-element, always)
        lr_ops = BT * 2 * (rank * N + M * rank)     # proj
        lr_ops += BT * 2 * (M * rank + rank * N)    # update
        tt_ops = BT * 2 * M * N                      # update
        if gamma == 1:
            lr_ops += BT * 2 * (rank * N + M * rank)  # visible
            tt_ops += BT * 2 * M * N
        ops_total['lr'] += lr_ops
        ops_total['tt'] += tt_ops

        # Energy (per active element)
        lr_e = EPS_ACIM_PJ * BT * 2 * (rank * N + M * rank) / 1e6   # proj MVM
        lr_e += KAPPA_E * EPS_ACIM_PJ * BT * 2 * (M * rank + rank * N) / 1e6  # update
        tt_e = KAPPA_E * EPS_ACIM_PJ * BT * 2 * M * N / 1e6
        if gamma == 1:
            lr_e += EPS_ACIM_PJ * BT * 2 * (rank * N + M * rank) / 1e6
            tt_e += EPS_ACIM_PJ * BT * 2 * M * N / 1e6
        energy['lr'] += lr_e
        energy['tt'] += tt_e

        # Update events per layer (per-tile, packing doesn't reduce)
        a_rows = math.ceil(M / T)
        b_cols = math.ceil(N / T)
        update_events_lr += a_rows + b_cols  # A tiles + B tiles per layer
        update_events_tt += math.ceil(M / T) * math.ceil(N / T)

        a_tile_rows_total += a_rows
        b_tile_cols_total += b_cols
        tt_tiles_total += math.ceil(M / T) * math.ceil(N / T)

    # Area: physical packed tiles
    # A matrices: all layers packed column-wise
    # For each unique tile-row, pack adapters_per_tile layers
    a_packed = math.ceil(n_layers / adapters_per_tile) * (a_tile_rows_total // n_layers)
    b_packed = math.ceil(n_layers / adapters_per_tile) * (b_tile_cols_total // n_layers)
    lr_phys_tiles = a_packed + b_packed
    tt_phys_tiles = tt_tiles_total  # TT can't pack (full-rank)

    # Latency (hybrid: MVM per-element + Update per-tile)
    # MVM per-element
    lr_lat_mvm = 0.0
    tt_lat_mvm = 0.0
    for l in targeted:
        M, N = l.M, l.N
        lr_lat_mvm += TAU_ACIM * BT * 2 * (rank * N + M * rank) / 1e6  # proj
        if gamma == 1:
            lr_lat_mvm += TAU_ACIM * BT * 2 * (rank * N + M * rank) / 1e6
            tt_lat_mvm += TAU_ACIM * BT * 2 * M * N / 1e6

    # Update per-tile: each layer needs its own pulse cycle per tile
    lr_lat_upd = ALPHA_T * t_tile * update_events_lr * BT / 1e6
    tt_lat_upd = ALPHA_T * t_tile * update_events_tt * BT / 1e6

    lr_lat = lr_lat_mvm + lr_lat_upd
    tt_lat = tt_lat_mvm + tt_lat_upd

    return {
        'lr_ops': ops_total['lr'],
        'tt_ops': ops_total['tt'],
        'lr_tiles': lr_phys_tiles,
        'tt_tiles': tt_phys_tiles,
        'lr_lat': lr_lat,
        'tt_lat': tt_lat,
        'lr_energy': energy['lr'],
        'tt_energy': energy['tt'],
    }


def main():
    inventory = build_layer_inventory()

    x_labels = ['TT'] + [f'r={r}' for r in RANKS]
    n_x = len(x_labels)
    x = np.arange(n_x)
    w = 0.55

    metrics = [
        ('ops',    '$\\Delta$Ops [TOps]',    'lr_ops',    'tt_ops',    lambda v: v / 1e12, 'log'),
        ('tiles',  'Physical tiles\n(packed)', 'lr_tiles', 'tt_tiles', lambda v: v,        'log'),
        ('lat',    '$\\Delta T_{step}$ [ms]', 'lr_lat',   'tt_lat',   lambda v: v,        'log'),
        ('energy', 'Energy [$\\mu$J]',        'lr_energy','tt_energy', lambda v: v,        'log'),
    ]

    for gamma in [0, 1]:
        fig, axes_2d = plt.subplots(2, 2, figsize=(8, 7))
        axes = axes_2d.flatten()

        target = 'attention'
        targeted = get_targeted_layers(inventory, target)

        for mi, (key, ylabel, lr_key, tt_key, transform, scale) in enumerate(metrics):
            ax = axes[mi]

            # Compute for each rank
            data_tt = compute_packed_metrics(targeted, 8, gamma)
            tt_val = transform(data_tt[tt_key])

            vals = [tt_val]
            for rank in RANKS:
                data = compute_packed_metrics(targeted, rank, gamma)
                vals.append(transform(data[lr_key]))

            colors = [C_TT] + [C_LR] * len(RANKS)
            ax.bar(x, vals, w, color=colors, edgecolor='white', lw=0.4, zorder=3)

            # Annotations
            for i in range(n_x):
                ratio = vals[0] / vals[i] if vals[i] > 0 else 0
                if i == 0:
                    if key == 'ops':
                        txt = f'{vals[i]:.2f}T'
                    elif key == 'tiles':
                        txt = f'{vals[i]:.0f}'
                    elif key == 'lat':
                        txt = f'{vals[i]:.0f}ms'
                    else:
                        txt = f'{vals[i]:.0f}'
                    color_txt = '#e67e22'
                else:
                    txt = f'{ratio:.0f}x'
                    if key == 'tiles':
                        txt = f'{vals[i]:.0f}\n{ratio:.0f}x'
                    color_txt = '#555'

                ax.text(x[i], vals[i] * 1.1, txt,
                        ha='center', va='bottom', fontsize=6,
                        fontweight='bold', color=color_txt,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

            if scale == 'log':
                ax.set_yscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=6.5)
            tl = ax.get_xticklabels()
            tl[0].set_fontweight('bold')
            tl[0].set_color(C_TT)
            ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
            ax.set_title(ylabel, fontsize=9, fontweight='bold', pad=4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.08, which='major', zorder=0)

        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=C_TT),
            plt.Rectangle((0, 0), 1, 1, facecolor=C_LR),
        ]
        fig.legend(handles, ['TikiTaka', 'LR-TT'], loc='lower center', ncol=2,
                   fontsize=8, bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc')

        fig.suptitle(f'LR-TT vs TikiTaka  |  target=attention,  $\\gamma$={gamma}\n'
                     f'$\\alpha_t$={ALPHA_T}  |  Tile={T}  |  '
                     f'Area=packed tiles  |  Latency=hybrid  |  Energy=per-element',
                     fontsize=9, fontweight='bold', y=1.01)
        plt.tight_layout(rect=[0, 0.05, 1, 0.93], w_pad=1.5, h_pad=2.0)
        path = os.path.join(OUT, f"LRTT_VS_TT_4METRICS_G{gamma}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper-ready figure: 4 metrics, 3×3 grid.
Row 1: Latency per-element (attention, 3 α_t) — optimistic
Row 2: Latency hybrid (attention, 3 α_t) — conservative
Row 3: Ops / Area(packed) / Energy (attention, ffn, all)
γ=1, tile=512×512
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
    RANKS, TARGETS, BATCH_SIZE, S_PAD_DEFAULT,
    TAU_ACIM, EPS_ACIM_PJ,
    ALPINE_TILE_MVM_NS,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
T = 512
t_tile = 400.0
ALPHA_T_PURE = 0.625
KAPPA_E = 1.0

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.linewidth': 0.5, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.major.width': 0.4, 'ytick.major.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 200,
})

C_TT = '#e67e22'
C_LR = '#2980b9'
C_PROJ = '#c0392b'
C_UPD = '#e67e22'
C_VIS = '#2980b9'

ALPHA_LIT = [
    (0.25, '$\\alpha_t$=0.25\nGokmen 2016'),
    (0.625, '$\\alpha_t$=0.625\nRasch 2024'),
    (2.0, '$\\alpha_t$=2.0\nGokmen 2016'),
]

x_labels = ['TT'] + [f'r={r}' for r in RANKS]
n_x = len(x_labels)
xp = np.arange(n_x)
w_single = 0.55
w_dual = 0.35


def compute_ops(targeted, rank, gamma=1):
    lr = {'proj': 0, 'update': 0, 'visible': 0}
    tt = {'proj': 0, 'update': 0, 'visible': 0}
    for l in targeted:
        M, N = l.M, l.N
        lr['proj'] += BT * 2 * (rank * N + M * rank)
        lr['update'] += BT * 2 * (M * rank + rank * N)
        if gamma == 1:
            lr['visible'] += BT * 2 * (rank * N + M * rank)
        tt['update'] += BT * 2 * M * N
        if gamma == 1:
            tt['visible'] += BT * 2 * M * N
    return lr, tt


def latency_elem(targeted, rank, gamma, alpha_t):
    lr_ms = 0; tt_ms = 0
    lat_w = {'proj': TAU_ACIM, 'update': alpha_t * TAU_ACIM, 'visible': TAU_ACIM}
    for l in targeted:
        M, N = l.M, l.N
        lr_ms += lat_w['proj'] * BT * 2 * (rank * N + M * rank) / 1e6
        lr_ms += lat_w['update'] * BT * 2 * (M * rank + rank * N) / 1e6
        tt_ms += lat_w['update'] * BT * 2 * M * N / 1e6
        if gamma == 1:
            lr_ms += lat_w['visible'] * BT * 2 * (rank * N + M * rank) / 1e6
            tt_ms += lat_w['visible'] * BT * 2 * M * N / 1e6
    return lr_ms, tt_ms


def latency_hybrid(targeted, rank, gamma, alpha_t):
    lr_ms = 0; tt_ms = 0
    # MVM per-element
    for l in targeted:
        M, N = l.M, l.N
        lr_ms += TAU_ACIM * BT * 2 * (rank * N + M * rank) / 1e6
        if gamma == 1:
            lr_ms += TAU_ACIM * BT * 2 * (rank * N + M * rank) / 1e6
            tt_ms += TAU_ACIM * BT * 2 * M * N / 1e6
    # Update per-tile
    for l in targeted:
        M, N = l.M, l.N
        lr_tiles = math.ceil(M/T) * math.ceil(rank/T) + math.ceil(rank/T) * math.ceil(N/T)
        tt_tiles = math.ceil(M/T) * math.ceil(N/T)
        lr_ms += alpha_t * t_tile * lr_tiles * BT / 1e6
        tt_ms += alpha_t * t_tile * tt_tiles * BT / 1e6
    return lr_ms, tt_ms


def packed_tiles(targeted, rank):
    n_layers = len(targeted)
    adapters_per_tile = max(1, T // rank) if rank < T else 1
    a_rows = math.ceil(targeted[0].M / T) if targeted else 1
    b_cols = math.ceil(targeted[0].N / T) if targeted else 1
    a_packed = math.ceil(n_layers / adapters_per_tile) * a_rows
    b_packed = math.ceil(n_layers / adapters_per_tile) * b_cols
    return a_packed + b_packed


def energy_elem(targeted, rank, gamma):
    lr_uj = 0; tt_uj = 0
    for l in targeted:
        M, N = l.M, l.N
        lr_uj += EPS_ACIM_PJ * BT * 2 * (rank * N + M * rank) / 1e6
        lr_uj += KAPPA_E * EPS_ACIM_PJ * BT * 2 * (M * rank + rank * N) / 1e6
        tt_uj += KAPPA_E * EPS_ACIM_PJ * BT * 2 * M * N / 1e6
        if gamma == 1:
            lr_uj += EPS_ACIM_PJ * BT * 2 * (rank * N + M * rank) / 1e6
            tt_uj += EPS_ACIM_PJ * BT * 2 * M * N / 1e6
    return lr_uj, tt_uj


def draw_single_bar(ax, vals, ylabel):
    """Single bar per x-position (TT + LR-TT ranks)."""
    colors = [C_TT] + [C_LR] * len(RANKS)
    ax.bar(xp, vals, w_single, color=colors, edgecolor='white', lw=0.4, zorder=3)
    tt_val = vals[0]
    for i in range(n_x):
        ratio = tt_val / vals[i] if vals[i] > 0 else 0
        if i == 0:
            txt = f'{vals[i]:.0f}' if vals[i] >= 1 else f'{vals[i]:.2f}'
            c = '#e67e22'
        else:
            txt = f'{ratio:.0f}x'
            c = '#555'
        ax.text(xp[i], vals[i] * 1.1, txt, ha='center', va='bottom',
                fontsize=5.5, fontweight='bold', color=c,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
    ax.set_yscale('log')
    ax.set_xticks(xp)
    ax.set_xticklabels(x_labels, fontsize=6)
    tl = ax.get_xticklabels()
    tl[0].set_fontweight('bold')
    tl[0].set_color(C_TT)
    ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.08, which='major', zorder=0)


def draw_dual_bar(ax, vals_a, vals_b, ylabel):
    """Two bars per x-position (optimistic vs conservative)."""
    colors_a = [C_TT] + [C_LR] * len(RANKS)
    colors_b = [C_TT] + [C_LR] * len(RANKS)
    for i in range(n_x):
        ax.bar(xp[i] - w_dual/2, vals_a[i], w_dual * 0.9,
               color=colors_a[i], edgecolor='white', lw=0.3, zorder=3,
               alpha=1.0 if i > 0 else 0.5)
        ax.bar(xp[i] + w_dual/2, vals_b[i], w_dual * 0.9,
               color=colors_b[i], edgecolor='white', lw=0.3, zorder=3,
               alpha=0.5 if i > 0 else 1.0, hatch='///' if i > 0 else '')

    tt_a = vals_a[0]; tt_b = vals_b[0]
    for i in range(n_x):
        if i == 0:
            txt = f'{vals_a[i]:.0f}|{vals_b[i]:.0f}'
            c = '#e67e22'
            y = max(vals_a[i], vals_b[i])
        else:
            r_a = tt_a / vals_a[i] if vals_a[i] > 0 else 0
            r_b = tt_b / vals_b[i] if vals_b[i] > 0 else 0
            txt = f'{r_a:.0f}x|{r_b:.0f}x'
            c = '#555'
            y = max(vals_a[i], vals_b[i])
        ax.text(xp[i], y * 1.12, txt, ha='center', va='bottom',
                fontsize=5, fontweight='bold', color=c,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    ax.set_yscale('log')
    ax.set_xticks(xp)
    ax.set_xticklabels(x_labels, fontsize=6)
    tl = ax.get_xticklabels()
    tl[0].set_fontweight('bold')
    tl[0].set_color(C_TT)
    ax.axvline(0.5, color='#eee', ls='-', lw=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.08, which='major', zorder=0)


def main():
    inventory = build_layer_inventory()
    gamma = 1
    targeted_attn = get_targeted_layers(inventory, 'attention')

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                          height_ratios=[1, 1, 1])

    # ═══ Row 1: Latency — elem vs hybrid side by side, 3 α_t ═══
    for pi, (alpha_t, cite) in enumerate(ALPHA_LIT):
        ax = fig.add_subplot(gs[0, pi])

        vals_elem = []
        vals_hybrid = []

        # TT
        _, tt_e = latency_elem(targeted_attn, 8, gamma, alpha_t)
        _, tt_h = latency_hybrid(targeted_attn, 8, gamma, alpha_t)
        vals_elem.append(tt_e)
        vals_hybrid.append(tt_h)

        for rank in RANKS:
            lr_e, _ = latency_elem(targeted_attn, rank, gamma, alpha_t)
            lr_h, _ = latency_hybrid(targeted_attn, rank, gamma, alpha_t)
            vals_elem.append(lr_e)
            vals_hybrid.append(lr_h)

        draw_dual_bar(ax, vals_elem, vals_hybrid, '')
        ax.set_title(f'{cite}', fontsize=7.5, fontweight='bold', pad=4)
        if pi == 0:
            ax.set_ylabel('$\\Delta T_{step}$ [ms]', fontsize=8)

    # ═══ Row 2: Ops / Area / Energy — attention, ffn, all ═══
    row2_configs = [
        ('Ops [TOps]', 'attention'),
        ('Ops [TOps]', 'ffn'),
        ('Ops [TOps]', 'all'),
    ]
    for ci, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[1, ci])
        targeted = get_targeted_layers(inventory, target)
        _, tt = compute_ops(targeted, 8, gamma)
        tt_total = sum(tt[k] / 1e12 for k in ['proj', 'update', 'visible'])
        vals = [tt_total]
        for rank in RANKS:
            lr, _ = compute_ops(targeted, rank, gamma)
            vals.append(sum(lr[k] / 1e12 for k in ['proj', 'update', 'visible']))
        draw_single_bar(ax, vals, '')
        ax.set_title(f'Ops — {target}', fontsize=7.5, fontweight='bold', pad=4)
        if ci == 0:
            ax.set_ylabel('$\\Delta$Ops [TOps]', fontsize=8)

    # ═══ Row 3: Area / Energy — attention, ffn, all ═══
    for ci, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[2, ci])
        targeted = get_targeted_layers(inventory, target)

        # Area (packed tiles) + Energy side by side
        n_layers = len(targeted)
        tt_tiles = sum(math.ceil(l.M/T) * math.ceil(l.N/T) for l in targeted)

        vals_area = [tt_tiles]
        vals_energy = []
        _, tt_uj = energy_elem(targeted, 8, gamma)
        vals_energy.append(tt_uj)

        for rank in RANKS:
            vals_area.append(packed_tiles(targeted, rank))
            lr_uj, _ = energy_elem(targeted, rank, gamma)
            vals_energy.append(lr_uj)

        # Draw area and energy as dual bars
        draw_dual_bar(ax, vals_area, vals_energy, '')
        ax.set_title(f'Area|Energy — {target}', fontsize=7.5, fontweight='bold', pad=4)
        if ci == 0:
            ax.set_ylabel('Tiles | $\\mu$J', fontsize=8)

    # ═══ Row labels ═══
    fig.text(0.02, 0.83, 'Latency\n(solid=elem\nhatch=hybrid)',
             fontsize=7, fontweight='bold', va='center', ha='center',
             rotation=90, color='#333')
    fig.text(0.02, 0.50, 'Ops count',
             fontsize=7, fontweight='bold', va='center', ha='center',
             rotation=90, color='#333')
    fig.text(0.02, 0.17, 'Area | Energy\n(solid=tiles\nhatch=µJ)',
             fontsize=7, fontweight='bold', va='center', ha='center',
             rotation=90, color='#333')

    # ═══ Legend ═══
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_TT, alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_LR),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_LR, alpha=0.5, hatch='///'),
    ]
    labels = ['TikiTaka', 'LR-TT (solid)', 'LR-TT (hatched)']
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.005), frameon=True, edgecolor='#ccc')

    fig.suptitle('LR-TT vs TikiTaka  |  $\\gamma$=1  |  '
                 'Tile=512  |  Nx = TT/LR-TT ratio\n'
                 'Row 1: Latency (solid=per-element, hatch=hybrid)  |  '
                 'Row 2: Ops  |  Row 3: Area & Energy (solid=tiles, hatch=$\\mu$J)',
                 fontsize=8.5, fontweight='bold', y=1.01)

    path = os.path.join(OUT, "LRTT_VS_TT_PAPER_FIGURE_G1.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

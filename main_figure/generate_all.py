#!/usr/bin/env python3
"""Main figure sub-plots in ACS/NPG style.
Unified: Tile=512, BT=48×384, ALPINE-calibrated.

A: Latency hybrid (grouped Proj/Upd/Vis, 3 α_t, target=all, γ=1)
B: Ops total (single bar, attention/ffn/all, γ=0)
C: Tile count (packed, attention/ffn/all, γ=0)
D: Energy (per-element, κ_e=0.5, attention/ffn/all, γ=0)
E: Model size scaling (BERT → LLaMA)
F: Sequence length sweep (r=8, target=all, hybrid)
"""

import os, sys, math
import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'lrtt_vs_tt_v2'))
COST_MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'cost_model')
sys.path.insert(0, COST_MODEL_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    BATCH_SIZE, S_PAD_DEFAULT, TAU_ACIM, EPS_ACIM_PJ,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
T = 512
t_tile = 400.0
KAPPA_E = 0.5
RANKS = [1, 2, 4, 8, 16, 32, 64]
TARGETS = ['attention', 'ffn', 'all']

# ─── ACS / NPG Style ───
ACS_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
}

# NPG palette
C_TT = '#E64B35'      # vermillion (TikiTaka)
C_LR = '#4DBBD5'      # sky cyan (LR-TT)
C_PROJ = '#E64B35'     # vermillion
C_UPD = '#F39B7F'      # light coral
C_VIS = '#4DBBD5'      # sky cyan
C_ACCENT = '#00A087'   # teal

ALPHA_LIT = [
    (0.25, 'Gokmen 2016'),
    (0.625, 'Rasch 2024'),
    (2.0, 'Gokmen 2016'),
]


def compute_ops_components(targeted, rank, gamma):
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


def compute_packed_tiles(targeted, rank):
    shape_counts = Counter((l.M, l.N) for l in targeted)
    apt = max(1, T // rank) if rank < T else 1
    lr_tiles = 0
    for (M_s, N_s), cnt in shape_counts.items():
        lr_tiles += math.ceil(cnt / apt) * math.ceil(M_s / T)
        lr_tiles += math.ceil(cnt / apt) * math.ceil(N_s / T)
    tt_tiles = sum(math.ceil(l.M / T) * math.ceil(l.N / T) for l in targeted)
    return lr_tiles, tt_tiles


def compute_energy(targeted, rank, gamma):
    lr_e = 0.0; tt_e = 0.0
    for l in targeted:
        M, N = l.M, l.N
        proj = BT * 2 * (rank * N + M * rank)
        upd_lr = BT * 2 * (M * rank + rank * N)
        upd_tt = BT * 2 * M * N
        vis_lr = BT * 2 * (rank * N + M * rank) if gamma == 1 else 0
        vis_tt = BT * 2 * M * N if gamma == 1 else 0
        lr_e += EPS_ACIM_PJ * (proj + vis_lr) / 1e6
        lr_e += KAPPA_E * EPS_ACIM_PJ * upd_lr / 1e6
        tt_e += EPS_ACIM_PJ * vis_tt / 1e6
        tt_e += KAPPA_E * EPS_ACIM_PJ * upd_tt / 1e6
    return lr_e, tt_e


def _anno(ax, xp, vals, is_tt_first=True):
    """Annotate bars: TT value label, LR-TT ratio."""
    for i in range(len(xp)):
        r = vals[0] / vals[i] if vals[i] > 0 else 0
        if i == 0 and is_tt_first:
            txt = f'{vals[i]:.2f}' if vals[i] < 10 else f'{vals[i]:.0f}'
            c = C_TT
        else:
            txt = f'{r:.0f}×'
            c = '#333'
        ax.text(xp[i], vals[i] * 1.12, txt, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=c)


def _style_ax(ax, xp, x_labels, ylabel):
    ax.set_yscale('log')
    ax.set_xticks(xp)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.2, axis='y')


# ═══════════════════════════════════════
# A: Latency hybrid (grouped, 3 α_t, γ=1)
# ═══════════════════════════════════════
def plot_A(inventory):
    plt.rcParams.update(ACS_RC)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    targeted = get_targeted_layers(inventory, 'all')
    n_ranks = len(RANKS)
    x = np.arange(n_ranks)
    comp_keys = ['proj', 'update', 'visible']
    comp_names = ['Projection (MVM)', 'Update (pulsed)', 'Visible fwd (MVM)']
    comp_colors = [C_PROJ, C_UPD, C_VIS]
    n_c = len(comp_keys)
    tw = 0.82; bw = tw / n_c

    for pi, (alpha_t, cite) in enumerate(ALPHA_LIT):
        ax = axes[pi]

        # TT total
        tt_lat = 0.0
        for l in targeted:
            M, N = l.M, l.N
            tt_lat += TAU_ACIM * BT * 2 * M * N / 1e6
            tt_lat += alpha_t * t_tile * math.ceil(M/T) * math.ceil(N/T) * BT / 1e6

        ax.axhline(tt_lat, color=C_TT, ls='--', lw=1.5, alpha=0.7, zorder=2)
        ax.text(n_ranks - 0.3, tt_lat * 1.2, f'TT: {tt_lat:.0f} ms',
                ha='right', va='bottom', fontsize=9, color=C_TT, fontweight='bold')

        # LR-TT grouped
        lr_data = []
        for rank in RANKS:
            lc = {'proj': 0.0, 'update': 0.0, 'visible': 0.0}
            for l in targeted:
                M, N = l.M, l.N
                lc['proj'] += TAU_ACIM * BT * 2 * (rank * N + M * rank) / 1e6
                lr_t = math.ceil(M/T)*math.ceil(rank/T) + math.ceil(rank/T)*math.ceil(N/T)
                lc['update'] += alpha_t * t_tile * lr_t * BT / 1e6
                lc['visible'] += TAU_ACIM * BT * 2 * (rank * N + M * rank) / 1e6
            lr_data.append(lc)

        for ci, (k, cn, cc) in enumerate(zip(comp_keys, comp_names, comp_colors)):
            offset = -tw/2 + (ci+0.5)*bw
            vals = [lr_data[i][k] for i in range(n_ranks)]
            ax.bar(x + offset, vals, bw*0.88, color=cc, alpha=0.92,
                   edgecolor='none', zorder=3, label=cn if pi == 0 else '')

        for i in range(n_ranks):
            total = sum(lr_data[i][k] for k in comp_keys)
            bar_top = max(lr_data[i][k] for k in comp_keys if lr_data[i][k] > 0)
            ratio = tt_lat / total if total > 0 else 0
            ax.text(x[i], bar_top * 1.15, f'{ratio:.1f}×',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')

        ax.set_yscale('log')
        all_lr = [lr_data[i][k] for i in range(n_ranks) for k in comp_keys if lr_data[i][k] > 0]
        ax.set_ylim(min(all_lr)*0.3, tt_lat*3)
        ax.set_xticks(x); ax.set_xticklabels([f'r={r}' for r in RANKS], fontsize=9)
        ax.set_title(f'$\\alpha_t$={alpha_t}\n({cite})', fontweight='bold')
        if pi == 0: ax.set_ylabel('$\\Delta T_{step}$ [ms]', fontweight='bold')
        ax.set_axisbelow(True); ax.grid(True, alpha=0.2, axis='y')

    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([0],[0], color=C_TT, ls='--', lw=1.5))
    l.append('TT total')
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, edgecolor='none')
    fig.suptitle('(a)  Adapter latency — hybrid model, target = all, $\\gamma$ = 1',
                 fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(f'{OUT}/A_latency_hybrid.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print("  A saved")


# ═══════════════════════════════════════
# B, C, D: Single-bar 3-panel (shared structure)
# ═══════════════════════════════════════
def _plot_3panel(inventory, metric_fn, ylabel, title, fname, gamma=0):
    plt.rcParams.update(ACS_RC)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for col, target in enumerate(TARGETS):
        ax = axes[col]
        targeted = get_targeted_layers(inventory, target)
        x_labels = ['TT'] + [f'r={r}' for r in RANKS]
        xp = np.arange(len(x_labels))
        tt_val, lr_vals = metric_fn(targeted, gamma)
        vals = [tt_val] + lr_vals
        colors = [C_TT] + [C_LR] * len(RANKS)
        ax.bar(xp, vals, 0.6, color=colors, alpha=0.92, edgecolor='none', zorder=3)
        _anno(ax, xp, vals)
        _style_ax(ax, xp, x_labels, ylabel if col == 0 else '')
        ax.set_title(target, fontweight='bold')

    h = [Patch(facecolor=C_TT, alpha=0.92, label='TikiTaka'),
         Patch(facecolor=C_LR, alpha=0.92, label='LR-TT')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='none')
    fig.suptitle(title, fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(f'{OUT}/{fname}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  {fname.split('.')[0]} saved")


def plot_B(inventory):
    def fn(targeted, gamma):
        _, tt = compute_ops_components(targeted, 8, gamma)
        tt_val = sum(tt[k]/1e12 for k in ['proj','update','visible'])
        lr_vals = []
        for rank in RANKS:
            lr, _ = compute_ops_components(targeted, rank, gamma)
            lr_vals.append(sum(lr[k]/1e12 for k in ['proj','update','visible']))
        return tt_val, lr_vals
    _plot_3panel(inventory, fn, 'Ops [TOps]',
                 '(b)  Active operations per training step, $\\gamma$ = 0',
                 'B_ops_total.png', gamma=0)


def plot_C(inventory):
    def fn(targeted, gamma):
        _, tt_t = compute_packed_tiles(targeted, 8)
        lr_vals = [compute_packed_tiles(targeted, r)[0] for r in RANKS]
        return tt_t, lr_vals
    _plot_3panel(inventory, fn, 'Tiles (packed)',
                 '(c)  Physical tile count (multi-layer column-packing)',
                 'C_tile_count.png', gamma=0)


def plot_D(inventory):
    def fn(targeted, gamma):
        _, tt_e = compute_energy(targeted, 8, gamma)
        lr_vals = [compute_energy(targeted, r, gamma)[0] for r in RANKS]
        return tt_e, lr_vals
    _plot_3panel(inventory, fn, 'Energy [$\\mu$J]',
                 f'(d)  Adapter energy, $\\gamma$ = 0, $\\kappa_e$ = {KAPPA_E}',
                 'D_energy.png', gamma=0)


# ═══════════════════════════════════════
# E: Model size scaling
# ═══════════════════════════════════════
def plot_E(inventory):
    plt.rcParams.update(ACS_RC)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    dims = [768, 1024, 2048, 4096, 8192]
    labels = ['BERT\n768', 'GPT-2S\n1024', 'GPT-2M\n2048', 'GPT-2L\n4096', 'LLaMA\n8192']
    rank_colors = {4: '#E64B35', 8: '#F39B7F', 16: '#4DBBD5', 32: '#00A087'}
    for rk, color in rank_colors.items():
        ratios = [d**2 / (rk * 2 * d) for d in dims]
        ax.plot(range(len(dims)), ratios, '-o', color=color, label=f'r = {rk}',
                lw=2.5, markersize=7, alpha=0.92)
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('TT / LR-TT ops ratio', fontweight='bold')
    ax.set_title('(e)  Model-size scaling: ratio ≈ d / (2r)', fontweight='bold')
    ax.set_yscale('log'); ax.legend(fontsize=10, framealpha=0.9, edgecolor='none')
    ax.set_axisbelow(True); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    fig.savefig(f'{OUT}/E_model_scaling.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print("  E saved")


# ═══════════════════════════════════════
# F: Sequence length sweep
# ═══════════════════════════════════════
def plot_F(inventory):
    plt.rcParams.update(ACS_RC)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    targeted = get_targeted_layers(inventory, 'all')
    seq_lens = [64, 128, 256, 352, 384]
    rank = 8; alpha_t = 0.625

    styles = [
        ('tt', 0, C_TT, '-', '^', 'TikiTaka $\\gamma$=0'),
        ('lr', 0, C_LR, '-', 'o', 'LR-TT r=8 $\\gamma$=0'),
        ('lr', 1, C_LR, '--', 's', 'LR-TT r=8 $\\gamma$=1'),
    ]
    for method, gamma, color, ls, marker, label in styles:
        ys = []
        for S in seq_lens:
            bt = BATCH_SIZE * S
            lr_ms = 0.0; tt_ms = 0.0
            for l in targeted:
                M, N = l.M, l.N
                proj = bt * 2 * (rank * N + M * rank)
                vis_lr = bt * 2 * (rank * N + M * rank) if gamma == 1 else 0
                vis_tt = bt * 2 * M * N if gamma == 1 else 0
                lr_ms += TAU_ACIM * (proj + vis_lr) / 1e6
                tt_ms += TAU_ACIM * vis_tt / 1e6
                lr_t = math.ceil(M/T)*math.ceil(rank/T) + math.ceil(rank/T)*math.ceil(N/T)
                tt_t = math.ceil(M/T)*math.ceil(N/T)
                lr_ms += alpha_t * t_tile * lr_t * bt / 1e6
                tt_ms += alpha_t * t_tile * tt_t * bt / 1e6
            ys.append(tt_ms if method == 'tt' else lr_ms)
        ax.plot(seq_lens, ys, color=color, ls=ls, marker=marker, markersize=7,
                lw=2.5, label=label, alpha=0.92)

    ax.axvline(352, color='gray', ls=':', lw=1.0, alpha=0.5)
    ax.text(355, ax.get_ylim()[0]*1.5 if ax.get_ylim()[0]>0 else 50, 'dyn mean',
            fontsize=9, color='gray', va='bottom')
    ax.set_xlabel('Sequence length $S$', fontweight='bold')
    ax.set_ylabel('$\\Delta T_{step}$ [ms]', fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('(f)  Sequence-length sweep (r = 8, target = all, hybrid)',
                 fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9, edgecolor='none')
    ax.set_axisbelow(True); ax.grid(True, alpha=0.2, which='both')
    plt.tight_layout()
    fig.savefig(f'{OUT}/F_sequence_sweep.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print("  F saved")


if __name__ == '__main__':
    plt.rcdefaults()
    inventory = build_layer_inventory()
    print(f'Layers: {len(inventory)}')
    print(f'Generating ACS-style sub-plots in {OUT}/')
    plot_A(inventory)
    plot_B(inventory)
    plot_C(inventory)
    plot_D(inventory)
    plot_E(inventory)
    plot_F(inventory)
    print('Done.')

#!/usr/bin/env python3
"""DL vs LR-TT break-even: θ* per rank.
θ = t_digital_per_flop / τ_acim (digital/analog speed ratio)

When θ > θ*: LR-TT wins (digital is too slow)
When θ < θ*: DL wins (digital is fast enough)

Same AIMC chip, same base weights on ACIM tiles.
Only adapter path differs: DL on PMCA vs LR-TT on ACIM.
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
    BATCH_SIZE, S_PAD_DEFAULT, TAU_ACIM,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
T = 512
t_tile = 400.0  # ns per 512×512 tile
ALPHA_T = 0.625  # update/MVM ratio

RANKS = [1, 2, 4, 8, 16, 32, 64]
TARGETS = ['attention', 'ffn', 'all']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.linewidth': 0.5, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.major.width': 0.4, 'ytick.major.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 200,
})


def compute_breakeven_theta(targeted, rank, gamma, alpha_t):
    """Compute θ* where ΔT_DL = ΔT_LRTT.

    ΔT_DL = θ × (fwd_flops + bwd_flops + adam_flops)
      where θ = t_digital_per_flop [ns/flop]

    ΔT_LRTT (hybrid: MVM per-element + Update per-tile):
      = τ_acim × (proj_ops + vis_ops)  [MVM per-element]
      + α_t × t_tile × update_tiles × BT  [Update per-tile]

    θ* = ΔT_LRTT / DL_total_flops
    """
    # LR-TT adapter latency (hybrid model)
    lrtt_mvm_ns = 0.0
    lrtt_upd_ns = 0.0

    # DL adapter FLOPs
    dl_flops = 0

    for l in targeted:
        M, N = l.M, l.N

        # LR-TT MVM (per-element)
        proj = BT * 2 * (rank * N + M * rank)
        vis = BT * 2 * (rank * N + M * rank) if gamma == 1 else 0
        lrtt_mvm_ns += TAU_ACIM * (proj + vis)

        # LR-TT Update (per-tile)
        lr_tiles = math.ceil(M / T) * math.ceil(rank / T) + \
                   math.ceil(rank / T) * math.ceil(N / T)
        lrtt_upd_ns += alpha_t * t_tile * lr_tiles * BT

        # DL FLOPs
        # Forward: 2 GEMMs
        fwd = 2 * BT * N * rank + 2 * BT * rank * M
        # Backward: 4 GEMMs (dA, dZ, dB, dX)
        bwd = 2 * BT * M * rank + 2 * BT * M * rank + \
              2 * BT * rank * N + 2 * BT * rank * N
        # Optimizer: Adam 5 ops per param
        opt = (M * rank + rank * N) * 5
        dl_flops += fwd + bwd + opt

    lrtt_total_ns = lrtt_mvm_ns + lrtt_upd_ns

    # θ* = ΔT_LRTT / DL_flops  [ns per flop]
    theta_star = lrtt_total_ns / dl_flops if dl_flops > 0 else 0

    return theta_star, lrtt_total_ns, dl_flops


def main():
    inventory = build_layer_inventory()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))

    colors = {'attention': '#c0392b', 'ffn': '#2980b9', 'all': '#2c3e50'}

    for col, target in enumerate(TARGETS):
        ax = axes[col]
        targeted = get_targeted_layers(inventory, target)

        for gamma in [0]:  # γ=0 only (main operating mode)
            thetas = []
            for rank in RANKS:
                theta, lrtt_ns, dl_flops = compute_breakeven_theta(
                    targeted, rank, gamma, ALPHA_T)
                thetas.append(theta)

            ax.plot(RANKS, thetas, '-', color=colors[target], lw=2.5,
                    marker='o', markersize=5)

            # Annotate values
            for i, (r, t) in enumerate(zip(RANKS, thetas)):
                ax.text(r, t * 1.2, f'{t:.3f}',
                        ha='center', va='bottom', fontsize=6,
                        color=colors[target], fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        # 3 reference lines: measured on-chip digital accelerators
        ref_points = [
            (0.017, 'RedMulE 58.5G\n(Tortorella 2023, 22nm)', '#e53935', 's'),
            (0.051, 'Maestro 19.8G\n(Montagna 2025, 65nm)', '#e67e22', 'D'),
            (0.1,   'GAP9 ~10G\n(GreenWaves, commercial)', '#8e44ad', '^'),
        ]
        for theta_ref, label, color, marker in ref_points:
            ax.axhline(theta_ref, color=color, ls=':', lw=1.0, alpha=0.7)
            # marker on the line at rank=1 position
            ax.plot(RANKS[0] * 0.85, theta_ref, marker, color=color,
                    markersize=6, markeredgecolor='white', markeredgewidth=0.5, zorder=6)
            if col == 2:
                ax.text(RANKS[-1] * 1.05, theta_ref, label,
                        ha='left', va='center', fontsize=5, color=color,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor=color, alpha=0.9, lw=0.5))

        # Light fill regions
        ax.fill_between(RANKS, [t for t in thetas], [100]*len(RANKS),
                         alpha=0.04, color='#2980b9')
        ax.fill_between(RANKS, [0.0001]*len(RANKS), [t for t in thetas],
                         alpha=0.04, color='#27ae60')

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS], fontsize=7)
        ax.set_xlabel('Rank $r$', fontsize=8)
        if col == 0:
            ax.set_ylabel('$\\theta^*$ [ns/FLOP]', fontsize=8)
        ax.set_title(target, fontsize=10, fontweight='bold', pad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='both', alpha=0.08, which='major')
        # no legend needed (single line)

    fig.suptitle('Digital LoRA vs LR-TT Break-Even:  $\\theta^*$ = $\\Delta T_{LRTT}$ / DL FLOPs\n'
                 f'$\\alpha_t$={ALPHA_T}  |  Tile={T}  |  '
                 'LR-TT: hybrid (MVM per-elem + Update per-tile)  |  '
                 'θ > θ* → LR-TT favored',
                 fontsize=8.5, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(OUT, "PAPER_DL_BREAKEVEN.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TOPS/W bar chart: TikiTaka vs LR-TT by rank, for attention/FFN/all."""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from extract_layer_inventory import build_layer_inventory, get_targeted_layers

OUT = "/root/paper_figures_anchored"
inventory = build_layer_inventory()
BS, S, BT = 48, 384, 48*384
RANKS = [1, 4, 8, 16, 32, 64]

# ALPINE anchor
T_ACIM = 200e-9       # 200 ns → seconds
E_ACIM_PJ = 50        # pJ per tile-element-equivalent MVM
ALPHA = 0.3            # t_upd / t_acim


def compute_topsw(layers, rank, gamma):
    """Compute TOPS/W for a method.

    TOPS = total useful operations / latency (seconds)
    W = total energy / latency (seconds)
    TOPS/W = total useful ops / total energy

    For same-ACIM: energy ∝ element_weighted_cost × e_per_element
    Useful ops = total weight elements updated per step (= adapter footprint × BT)

    We define:
      useful_ops = BT × adapter_params (multiply-accumulate equivalents per step)
      energy = element_weighted_cost × e_per_element_op

    TOPS/W = useful_ops / energy = useful_ops / (elem_weighted_cost × e_per_op)
    """
    # LR-TT
    lr_ops = 0; lr_energy_elem = 0
    for l in layers:
        M, N = l.M, l.N
        eA, eB = M*rank, rank*N
        lr_ops += BT * (eA + eB)  # useful update ops

        proj_elem = BT * (eA + eB)
        upd_elem = BT * (eA + eB)
        vis_elem = BT * (eA + eB) if gamma == 1 else 0
        # Energy: MVM costs 1 unit, update costs α units
        lr_energy_elem += proj_elem * 1.0 + upd_elem * ALPHA + vis_elem * 1.0

    # TikiTaka
    tt_ops = 0; tt_energy_elem = 0
    for l in layers:
        M, N = l.M, l.N
        tt_ops += BT * M * N  # useful update ops (full rank)

        upd_elem = BT * M * N
        vis_elem = BT * M * N if gamma == 1 else 0
        tt_energy_elem += upd_elem * ALPHA + vis_elem * 1.0

    # TOPS/W = ops / energy (both in element units, e_per_op cancels in ratio)
    # But for absolute TOPS/W we need concrete energy
    # Energy (pJ) = energy_elem × e_per_element_op
    # e_per_element_op ≈ E_ACIM_PJ / (512*512) per element ≈ 0.00019 pJ
    # This is very small, so let's use per-tile energy instead

    # Actually: useful_ops in TOPS = lr_ops / 1e12
    # energy in W = energy_elem × (E_ACIM_PJ × 1e-12) / (latency in seconds)
    # latency = energy_elem × T_ACIM / (512*512) ... this gets complicated

    # SIMPLER: TOPS/W = useful_ops / energy_in_joules
    # where energy_in_joules = energy_elem_count × energy_per_element_joule
    # energy_per_element ≈ E_ACIM_PJ / (512*512) per element = 50e-12 / 262144 ≈ 1.9e-16 J

    e_per_elem_j = E_ACIM_PJ * 1e-12 / (512*512)  # J per element op

    lr_energy_j = lr_energy_elem * e_per_elem_j
    tt_energy_j = tt_energy_elem * e_per_elem_j

    lr_topsw = (lr_ops / 1e12) / lr_energy_j if lr_energy_j > 0 else 0
    tt_topsw = (tt_ops / 1e12) / tt_energy_j if tt_energy_j > 0 else 0

    return lr_topsw, tt_topsw


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax_i, target in enumerate(['attention', 'ffn', 'all']):
        ax = axes[ax_i]
        layers = get_targeted_layers(inventory, target)

        gamma = 0  # γ=0 main comparison

        # Compute TOPS/W
        tt_topsw = compute_topsw(layers, 8, gamma)[1]  # TT doesn't depend on rank
        lr_topsws = []
        for r in RANKS:
            lr_tw, _ = compute_topsw(layers, r, gamma)
            lr_topsws.append(lr_tw)

        # Bar positions: TikiTaka + LR-TT per rank
        n = len(RANKS) + 1  # TT + 6 ranks
        x = np.arange(n)
        labels = ['TikiTaka'] + [f'LR-TT\nr={r}' for r in RANKS]
        values = [tt_topsw] + lr_topsws

        colors = ['#4CAF50'] + ['#880E4F', '#E91E63', '#E53935', '#FF9800', '#FFC107', '#CDDC39']

        bars = ax.bar(x, values, 0.65, color=colors, edgecolor='#333', linewidth=0.5)

        # Value labels
        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.02, f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        if ax_i == 0:
            ax.set_ylabel('TOPS/W (element-level proxy)', fontsize=11)
        ax.set_title(f'{target}  (γ=0, α=0.3)', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.1)

    fig.suptitle('Training Efficiency: TikiTaka vs LR-TT by Rank\n'
                 'TOPS/W proxy = useful_update_ops / energy  |  Same ACIM family, E$_{acim}$=50pJ/tile',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f'{OUT}/topsw_bar_gamma0.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  topsw_bar_gamma0.png")

    # γ=1 version
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax_i, target in enumerate(['attention', 'ffn', 'all']):
        ax = axes[ax_i]
        layers = get_targeted_layers(inventory, target)
        gamma = 1

        tt_topsw = compute_topsw(layers, 8, gamma)[1]
        lr_topsws = []
        for r in RANKS:
            lr_tw, _ = compute_topsw(layers, r, gamma)
            lr_topsws.append(lr_tw)

        n = len(RANKS) + 1
        x = np.arange(n)
        labels = ['TikiTaka'] + [f'LR-TT\nr={r}' for r in RANKS]
        values = [tt_topsw] + lr_topsws
        colors = ['#4CAF50'] + ['#880E4F', '#E91E63', '#E53935', '#FF9800', '#FFC107', '#CDDC39']

        bars = ax.bar(x, values, 0.65, color=colors, edgecolor='#333', linewidth=0.5)

        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.02, f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        if ax_i == 0:
            ax.set_ylabel('TOPS/W (element-level proxy)', fontsize=11)
        ax.set_title(f'{target}  (γ=1, α=0.3)', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.1)

    fig.suptitle('Training Efficiency: TikiTaka vs LR-TT by Rank  (γ=1)\n'
                 'TOPS/W proxy = useful_update_ops / energy  |  Same ACIM family',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f'{OUT}/topsw_bar_gamma1.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  topsw_bar_gamma1.png")


if __name__ == "__main__":
    main()

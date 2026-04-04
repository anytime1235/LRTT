#!/usr/bin/env python3
"""Absolute TOPS and W bar charts — TikiTaka vs LR-TT by rank."""

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
ALPHA = 0.3
E_PER_ELEM_PJ = 50 / (512*512)  # pJ per element op ≈ 0.00019 pJ
T_PER_ELEM_NS = 200 / (512*512)  # ns per element op (parallel within tile)

# But actually for absolute time, use tile-level: t_acim per tile MVM
# For absolute energy: e_acim per tile MVM = 50 pJ
# Per element: 50/(512*512) pJ ≈ too small

# BETTER: Use total energy per step as proxy
# Energy_total = elem_weighted_count × e_per_element
# Latency_total = depends on parallelism (tile-level or element-level interpretation)

# For THIS plot: show absolute element-weighted cost (proportional to energy)
# and absolute useful ops (proportional to throughput)
# Then TOPS = useful_ops / latency, W = energy / latency
# Since latency cancels: just show ops and energy side by side


def compute(layers, rank, gamma):
    lr_useful = 0; lr_energy = 0
    tt_useful = 0; tt_energy = 0

    for l in layers:
        M, N = l.M, l.N
        eA, eB = M*rank, rank*N

        # LR-TT
        lr_useful += BT * (eA + eB)
        lr_energy += BT*(eA+eB)*1.0 + BT*(eA+eB)*ALPHA + (BT*(eA+eB)*1.0 if gamma==1 else 0)

        # TikiTaka
        tt_useful += BT * M * N
        tt_energy += BT*M*N*ALPHA + (BT*M*N*1.0 if gamma==1 else 0)

    return lr_useful, lr_energy, tt_useful, tt_energy


def main():
    for gamma in [0, 1]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for col, target in enumerate(['attention', 'ffn', 'all']):
            layers = get_targeted_layers(inventory, target)

            # Compute for all ranks
            tt_useful, tt_energy = compute(layers, 8, gamma)[2:]  # TT independent of rank
            lr_usefuls = []; lr_energies = []
            for r in RANKS:
                lu, le, _, _ = compute(layers, r, gamma)
                lr_usefuls.append(lu)
                lr_energies.append(le)

            # Top row: useful ops (throughput proxy)
            ax = axes[0, col]
            n = len(RANKS) + 1
            x = np.arange(n)
            labels = ['TT'] + [f'r={r}' for r in RANKS]
            ops = [tt_useful/1e12] + [u/1e12 for u in lr_usefuls]
            colors = ['#4CAF50'] + ['#880E4F','#E91E63','#E53935','#FF9800','#FFC107','#CDDC39']

            ax.bar(x, ops, 0.65, color=colors, edgecolor='#333', lw=0.5)
            for i, v in enumerate(ops):
                ax.text(i, v+max(ops)*0.02, f'{v:.1f}T', ha='center', fontsize=7, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
            if col==0: ax.set_ylabel('Useful update ops (TOPS)', fontsize=10)
            ax.set_title(f'{target}', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.1)
            ax.set_yscale('log')

            # Bottom row: energy (element-weighted cost proxy)
            ax = axes[1, col]
            energy = [tt_energy/1e12] + [e/1e12 for e in lr_energies]
            ax.bar(x, energy, 0.65, color=colors, edgecolor='#333', lw=0.5)
            for i, v in enumerate(energy):
                ax.text(i, v+max(energy)*0.02, f'{v:.1f}T', ha='center', fontsize=7, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
            if col==0: ax.set_ylabel('Energy proxy (elem-weighted, ×10¹²)', fontsize=10)
            ax.grid(axis='y', alpha=0.1)
            ax.set_yscale('log')

        fig.suptitle(f'TikiTaka vs LR-TT: Throughput (top) and Energy (bottom)  |  γ={gamma}, α={ALPHA}\n'
                     f'LR-TT: lower rank → less energy but also fewer useful ops  |  TT: high ops + high energy',
                     fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0,0,1,0.90])
        fig.savefig(f'{OUT}/topsw_absolute_gamma{gamma}.png', dpi=250, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  topsw_absolute_gamma{gamma}.png")


if __name__ == "__main__":
    main()

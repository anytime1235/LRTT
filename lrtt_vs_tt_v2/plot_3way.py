#!/usr/bin/env python3
"""3-way comparison: TikiTaka vs LR-TT vs Digital LoRA.
3 rows (Ops, Tiles, Energy) × 3 columns (attention, ffn, all).
γ=0 main. Tile=512×512.

Digital LoRA:
  - Base W on ACIM tile (same as TT/LR-TT)
  - Adapter A[M×r], B[r×N] on digital SRAM/PMCA
  - Forward: 2 digital GEMMs per layer
  - Backward: 4 digital GEMMs per layer
  - Optimizer: Adam (5 ops per param)
  - No analog tile needed for adapter → 0 adapter tiles
  - Ops = digital FLOPs (forward + backward + optimizer)
  - Energy = digital FLOPs × e_digital_per_flop
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

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    BATCH_SIZE, S_PAD_DEFAULT, EPS_ACIM_PJ,
)

OUT = SCRIPT_DIR
BT = BATCH_SIZE * S_PAD_DEFAULT
T = 512
KAPPA_E = 0.5    # analog update energy ratio (Gokmen 2016)
RANKS = [1, 8, 16, 32, 64]
TARGETS = ['attention', 'ffn', 'all']

# Digital LoRA energy: assume SRAM-CIM at ~10 fJ/op (65nm SRAM-CIM literature)
# vs ACIM eps_acim = 78 fJ/op (ALPINE)
# Digital is more energy-efficient for compute but needs separate SRAM
E_DIGITAL_PJ = 0.010  # 10 fJ/op = 0.010 pJ/op (SRAM-CIM)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.linewidth': 0.5, 'axes.labelsize': 7.5, 'axes.titlesize': 9,
    'xtick.major.width': 0.4, 'ytick.major.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 200,
})

C_TT = '#e67e22'   # orange
C_LR = '#2980b9'   # blue
C_DL = '#27ae60'   # green


def compute_3way(targeted, rank, gamma):
    """Compute Ops, Tiles, Energy for all 3 methods."""
    n_layers = len(targeted)

    # ── TikiTaka ──
    tt_ops = 0; tt_energy = 0.0
    for l in targeted:
        M, N = l.M, l.N
        upd = BT * 2 * M * N
        vis = BT * 2 * M * N if gamma == 1 else 0
        tt_ops += upd + vis
        tt_energy += KAPPA_E * EPS_ACIM_PJ * upd / 1e6
        tt_energy += EPS_ACIM_PJ * vis / 1e6
    tt_tiles = sum(math.ceil(l.M / T) * math.ceil(l.N / T) for l in targeted)

    # ── LR-TT ──
    lr_ops = 0; lr_energy = 0.0
    for l in targeted:
        M, N = l.M, l.N
        proj = BT * 2 * (rank * N + M * rank)
        upd = BT * 2 * (M * rank + rank * N)
        vis = BT * 2 * (rank * N + M * rank) if gamma == 1 else 0
        lr_ops += proj + upd + vis
        lr_energy += EPS_ACIM_PJ * (proj + vis) / 1e6
        lr_energy += KAPPA_E * EPS_ACIM_PJ * upd / 1e6

    # LR-TT tiles (packed)
    shape_counts = Counter((l.M, l.N) for l in targeted)
    apt = max(1, T // rank) if rank < T else 1
    lr_tiles = 0
    for (M_s, N_s), cnt in shape_counts.items():
        a_packed = math.ceil(cnt / apt) * math.ceil(M_s / T)
        b_packed = math.ceil(cnt / apt) * math.ceil(N_s / T)
        lr_tiles += a_packed + b_packed

    # ── Digital LoRA ──
    # Forward: 2 GEMMs: X@B^T [BT,N]×[N,r] + result@A^T [BT,r]×[r,M]
    # Backward: 4 GEMMs: dA, dZ, dB, dX_LoRA
    # Optimizer: Adam 5 ops/param for A[M×r] + B[r×N]
    dl_ops = 0; dl_energy = 0.0
    for l in targeted:
        M, N = l.M, l.N
        fwd = 2 * BT * N * rank + 2 * BT * rank * M         # 2 GEMMs
        bwd = 2 * BT * M * rank * 2 + 2 * BT * rank * N * 2  # 4 GEMMs
        opt = (M * rank + rank * N) * 5                        # Adam
        dl_ops += fwd + bwd + opt
        dl_energy += E_DIGITAL_PJ * (fwd + bwd + opt) / 1e6

    # DL tiles: adapter on SRAM (0 analog tiles for adapter)
    # But we count "equivalent SRAM area" as parameters / T²
    dl_params = sum(l.M * rank + rank * l.N for l in targeted)
    # SRAM area in "tile equivalents": each tile = T×T = 262144 cells
    # LoRA params are much smaller
    dl_tiles = max(1, math.ceil(dl_params / (T * T)))

    return {
        'tt': {'ops': tt_ops, 'tiles': tt_tiles, 'energy': tt_energy},
        'lr': {'ops': lr_ops, 'tiles': lr_tiles, 'energy': lr_energy},
        'dl': {'ops': dl_ops, 'tiles': dl_tiles, 'energy': dl_energy},
    }


def main():
    inventory = build_layer_inventory()

    # X-axis: for each target group, show TT + LR-TT ranks + DL ranks
    # Simplified: TT | LR r=1,8,16,32,64 | DL r=1,8,16,32,64
    methods_per_group = ['TT'] + [f'LR r={r}' for r in RANKS] + [f'DL r={r}' for r in RANKS]
    n_per_group = len(methods_per_group)
    gap = 2.0

    for gamma in [0, 1]:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.subplots_adjust(hspace=0.15)

        # Build x positions
        all_x = []
        all_labels = []
        all_colors = []
        all_vals = {'ops': [], 'tiles': [], 'energy': []}
        group_centers = []
        group_boundaries = []

        for gi, target in enumerate(TARGETS):
            targeted = get_targeted_layers(inventory, target)
            offset = gi * (n_per_group + gap)
            group_x = [offset + i for i in range(n_per_group)]
            all_x.extend(group_x)
            all_labels.extend(methods_per_group)
            group_centers.append(offset + n_per_group / 2 - 0.5)
            if gi > 0:
                group_boundaries.append(offset - gap / 2)

            # TT
            data = compute_3way(targeted, 8, gamma)
            all_vals['ops'].append(data['tt']['ops'] / 1e12)
            all_vals['tiles'].append(data['tt']['tiles'])
            all_vals['energy'].append(data['tt']['energy'])
            all_colors.append(C_TT)

            # LR-TT per rank
            for rank in RANKS:
                data = compute_3way(targeted, rank, gamma)
                all_vals['ops'].append(data['lr']['ops'] / 1e12)
                all_vals['tiles'].append(data['lr']['tiles'])
                all_vals['energy'].append(data['lr']['energy'])
                all_colors.append(C_LR)

            # DL per rank
            for rank in RANKS:
                data = compute_3way(targeted, rank, gamma)
                all_vals['ops'].append(data['dl']['ops'] / 1e12)
                all_vals['tiles'].append(data['dl']['tiles'])
                all_vals['energy'].append(data['dl']['energy'])
                all_colors.append(C_DL)

        all_x = np.array(all_x)

        row_configs = [
            ('ops', 'Ops [TOps]', '(a) Active Operations'),
            ('tiles', 'Tiles', '(b) Tile / SRAM Count'),
            ('energy', 'Energy [$\\mu$J]', '(c) Energy'),
        ]

        for row, (key, ylabel, title) in enumerate(row_configs):
            ax = axes[row]
            vals = all_vals[key]
            ax.bar(all_x, vals, 0.7, color=all_colors, edgecolor='white', lw=0.3, zorder=3)

            # Ratio annotations (vs TT)
            idx = 0
            for gi, target in enumerate(TARGETS):
                tt_v = vals[idx]
                # TT label
                if key == 'ops':
                    ax.text(all_x[idx], tt_v * 1.12, f'{tt_v:.2f}T',
                            ha='center', va='bottom', fontsize=5, fontweight='bold',
                            color='#e67e22',
                            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                else:
                    ax.text(all_x[idx], tt_v * 1.12, f'{tt_v:.0f}',
                            ha='center', va='bottom', fontsize=5, fontweight='bold',
                            color='#e67e22',
                            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                idx += 1

                # LR-TT + DL ratios
                for i in range(len(RANKS) * 2):
                    v = vals[idx]
                    ratio = tt_v / v if v > 0 else 0
                    if ratio >= 1:
                        txt = f'{ratio:.0f}x'
                    else:
                        txt = f'{1/ratio:.0f}x↑'
                    ax.text(all_x[idx], v * 1.12, txt,
                            ha='center', va='bottom', fontsize=4.5, fontweight='bold',
                            color='#555',
                            path_effects=[pe.withStroke(linewidth=1.2, foreground='white')])
                    idx += 1

            ax.set_yscale('log')
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=9, fontweight='bold', pad=4, loc='left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.08, which='major', zorder=0)

            for bx in group_boundaries:
                ax.axvline(bx, color='#ccc', ls='-', lw=0.8, zorder=0)

            # Target labels
            if row == 0:
                for gi, target in enumerate(TARGETS):
                    ax.text(group_centers[gi], ax.get_ylim()[1] * 0.5,
                            target, ha='center', va='top',
                            fontsize=10, fontweight='bold', color='#333',
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

            # LR-TT / DL separators within each group
            for gi in range(len(TARGETS)):
                offset = gi * (n_per_group + gap)
                # After TT
                ax.axvline(offset + 0.5, color='#eee', ls='-', lw=0.5, zorder=0)
                # Between LR-TT and DL
                ax.axvline(offset + len(RANKS) + 0.5, color='#ddd', ls='--', lw=0.5, zorder=0)

        # X-axis labels
        axes[-1].set_xticks(all_x)
        short_labels = []
        for l in all_labels:
            if l == 'TT':
                short_labels.append('TT')
            elif l.startswith('LR'):
                short_labels.append(l.replace('LR ', ''))
            else:
                short_labels.append(l.replace('DL ', ''))
        axes[-1].set_xticklabels(short_labels, fontsize=5, rotation=45, ha='right')

        # Legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=C_TT),
            plt.Rectangle((0, 0), 1, 1, facecolor=C_LR),
            plt.Rectangle((0, 0), 1, 1, facecolor=C_DL),
        ]
        fig.legend(handles, ['TikiTaka (full-rank analog)', 'LR-TT (low-rank analog)',
                             'Digital LoRA (SRAM)'],
                   loc='lower center', ncol=3, fontsize=8,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc')

        fig.suptitle(f'3-Way Comparison: TikiTaka vs LR-TT vs Digital LoRA  |  $\\gamma$={gamma}\n'
                     f'Tile={T}×{T}  |  $\\kappa_e$={KAPPA_E}  |  '
                     f'DL energy: {E_DIGITAL_PJ*1000:.0f} fJ/op (SRAM-CIM)',
                     fontsize=9, fontweight='bold', y=1.0)

        path = os.path.join(OUT, f"PAPER_3WAY_G{gamma}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

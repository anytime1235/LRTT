#!/usr/bin/env python3
"""Clean, publication-quality unified break-even plot."""

import os, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe

OUT = "/root/paper_figures_anchored"
os.makedirs(OUT, exist_ok=True)


def main():
    fig, ax = plt.subplots(figsize=(8, 6.5))

    phis = np.linspace(0, 3.0, 600)
    betas = np.linspace(0, 2.5, 600)
    PHI, BETA = np.meshgrid(phis, betas)

    # ── Background: continuous heatmap showing "which α is the break-even at this point" ──
    # At point (φ, β), the γ=1 break-even α satisfies: 3β = 2 + α(1-φ)
    # → α_be = (3β - 2) / (1 - φ)  when φ ≠ 1
    # If β > break-even at given α, LR-TT wins.
    # We shade by: for α=0.3 reference, how deep in LR-TT vs DL territory

    alpha_ref = 0.3
    # "advantage" = LR-TT cost - DL cost (negative = LR-TT wins)
    # DL = 3β + φα, LRTT_g1 = 2 + α → advantage = (2+α) - (3β + φα) = 2 + α(1-φ) - 3β
    advantage_g1 = 2 + alpha_ref * (1 - PHI) - 3 * BETA
    advantage_g0 = 1 + alpha_ref * (1 - PHI) - 3 * BETA

    # Custom diverging colormap: blue (DL wins) → white (break-even) → green (LR-TT wins)
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {
        'red':   [(0, 0.26, 0.26), (0.45, 0.85, 0.85), (0.5, 1.0, 1.0), (0.55, 0.95, 0.95), (1, 0.17, 0.17)],
        'green': [(0, 0.52, 0.52), (0.45, 0.92, 0.92), (0.5, 1.0, 1.0), (0.55, 0.90, 0.90), (1, 0.63, 0.63)],
        'blue':  [(0, 0.96, 0.96), (0.45, 0.97, 0.97), (0.5, 1.0, 1.0), (0.55, 0.85, 0.85), (1, 0.17, 0.17)],
    }
    cmap = LinearSegmentedColormap('custom', cdict)

    im = ax.contourf(PHI, BETA, advantage_g1, levels=np.linspace(-2, 2, 40),
                      cmap=cmap, alpha=0.5, extend='both')

    # ── Break-even lines for selected α values ──
    alpha_vals = [0.01, 0.1, 0.3, 1.0, 3.0]
    colors_a = ['#FFD600', '#FF9800', '#E53935', '#7B1FA2', '#283593']

    for alpha, color in zip(alpha_vals, colors_a):
        bg1 = (2 + alpha * (1 - phis)) / 3
        bg0 = (1 + alpha * (1 - phis)) / 3

        mask1 = (bg1 > 0) & (bg1 < 2.5)
        mask0 = (bg0 > 0) & (bg0 < 2.5)

        line1, = ax.plot(phis[mask1], bg1[mask1], '-', color=color, linewidth=2.2, alpha=0.9)
        line1.set_path_effects([pe.Stroke(linewidth=3.5, foreground='white', alpha=0.6), pe.Normal()])

        line0, = ax.plot(phis[mask0], bg0[mask0], '--', color=color, linewidth=1.8, alpha=0.7)
        line0.set_path_effects([pe.Stroke(linewidth=3, foreground='white', alpha=0.5), pe.Normal()])

        # Label at left edge
        y1_left = (2 + alpha) / 3
        y0_left = (1 + alpha) / 3
        if 0.1 < y1_left < 2.4:
            ax.text(-0.08, y1_left, f'α={alpha}', fontsize=8, color=color,
                    fontweight='bold', va='center', ha='right',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Reference lines ──
    ax.axhline(1.0, color='#666', ls=':', lw=0.7, alpha=0.5)
    ax.axvline(1.0, color='#666', ls=':', lw=0.7, alpha=0.5)

    # ── Literature anchor ──
    ax.plot(0.3, 1.0, '*', color='#D32F2F', markersize=16, markeredgecolor='white',
            markeredgewidth=1.5, zorder=10)
    ax.annotate('ALPINE-like', xy=(0.3, 1.0), xytext=(0.75, 1.35),
                fontsize=9, fontweight='bold', color='#D32F2F',
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5,
                                connectionstyle='arc3,rad=0.2'))

    ax.plot(0.15, 0.35, 'D', color='#1565C0', markersize=9, markeredgecolor='white',
            markeredgewidth=1, zorder=10)
    ax.annotate('Fast DCIM', xy=(0.15, 0.35), xytext=(0.55, 0.15),
                fontsize=8, fontweight='bold', color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2))

    # ── Region labels ──
    ax.text(0.2, 2.2, 'LR-TT wins\n(both γ)', fontsize=13, color='#1B5E20',
            fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax.text(2.3, 0.18, 'Digital LoRA\nwins', fontsize=13, color='#0D47A1',
            fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax.text(2.0, 0.65, 'γ=0\nexclusive', fontsize=10, color='#BF360C',
            fontweight='bold', ha='center', style='italic',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Axis labels ──
    ax.set_xlabel('φ = t_dcim_write / t_acim_upd\n(Update comparison: Adam vs Pulsed)', fontsize=11)
    ax.set_ylabel('β = t_dcim / t_acim\n(MVM comparison: DCIM vs ACIM)', fontsize=11)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.5)

    # ── Legend ──
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, ls='-', label='γ=1 boundary (solid)'),
        Line2D([0], [0], color='black', lw=1.5, ls='--', label='γ=0 boundary (dashed)'),
        Line2D([0], [0], marker='*', color='#D32F2F', ls='', markersize=10, label='ALPINE reference'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right',
              framealpha=0.9, edgecolor='#ccc')

    ax.set_title('Digital LoRA vs LR-TT: Latency Break-Even\n'
                 'Background shaded for α=0.3 | Lines show α ∈ {0.01, 0.1, 0.3, 1.0, 3.0}',
                 fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()
    p = f'{OUT}/fig1_unified_break_even.png'
    fig.savefig(p, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  {p}")


if __name__ == "__main__":
    main()

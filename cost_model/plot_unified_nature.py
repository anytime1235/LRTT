#!/usr/bin/env python3
"""Unified break-even in Nature-style: clean, minimal, grayscale-friendly."""

import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

OUT = "/root/paper_figures_anchored"
PHIS = np.linspace(0, 3.0, 600)
ALPHAS = [0.01, 0.1, 0.3, 1.0, 3.0]


def bg1(a, p): return (2 + a*(1-p)) / 3
def bg0(a, p): return (1 + a*(1-p)) / 3


def main():
    # Nature style: thin serif-like feel, no heavy gridlines, muted palette
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    fig, ax = plt.subplots(figsize=(7, 5.8))
    ax.set_facecolor('white')

    # ── Soft background regions for α=0.3 ──
    a_ref = 0.3
    g1r = np.clip(bg1(a_ref, PHIS), 0, 2.5)
    g0r = np.clip(bg0(a_ref, PHIS), 0, 2.5)
    ax.fill_between(PHIS, 2.5, g1r, color='#D7ECD9', alpha=0.5)     # green tint
    ax.fill_between(PHIS, g1r, g0r, color='#FFF0D4', alpha=0.5)      # amber tint
    ax.fill_between(PHIS, g0r, 0, color='#D6E6F5', alpha=0.5)        # blue tint

    # ── α sweep lines: muted academic colors ──
    palette = {
        0.01: ('#B8860B', 0.45),   # dark goldenrod, light
        0.1:  ('#CC6600', 0.55),   # burnt orange
        0.3:  ('#C62828', 1.0),    # main red (bold)
        1.0:  ('#6A1B9A', 0.55),   # purple
        3.0:  ('#1A237E', 0.45),   # navy
    }

    for alpha in ALPHAS:
        color, alpha_line = palette[alpha]
        lw = 2.2 if alpha == 0.3 else 1.2
        g1 = bg1(alpha, PHIS); g0 = bg0(alpha, PHIS)
        m1 = (g1 > -0.1) & (g1 < 2.6)
        m0 = (g0 > -0.1) & (g0 < 2.6)

        l1, = ax.plot(PHIS[m1], g1[m1], '-', color=color, lw=lw, alpha=alpha_line)
        l0, = ax.plot(PHIS[m0], g0[m0], '--', color=color, lw=lw*0.75, alpha=alpha_line*0.7)

        if alpha == 0.3:
            l1.set_path_effects([pe.Stroke(linewidth=lw+1.5, foreground='white', alpha=0.7), pe.Normal()])
            l0.set_path_effects([pe.Stroke(linewidth=lw+1, foreground='white', alpha=0.6), pe.Normal()])

        # Right-edge label
        y1_end = bg1(alpha, 2.95)
        if 0.05 < y1_end < 2.45:
            ax.text(3.03, y1_end, f'α={alpha}', fontsize=7, color=color,
                    fontweight='bold' if alpha == 0.3 else 'normal', va='center',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Reference lines (very subtle) ──
    ax.axhline(1.0, color='#AAAAAA', ls=':', lw=0.4)
    ax.axvline(1.0, color='#AAAAAA', ls=':', lw=0.4)

    # ── Literature anchor ──
    ax.plot(0.3, 1.0, 'o', color='#C62828', markersize=7, markeredgecolor='white',
            markeredgewidth=1.2, zorder=10)
    ax.annotate('ALPINE-like', xy=(0.3, 1.0), xytext=(0.85, 1.45),
                fontsize=8, color='#C62828',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1, connectionstyle='arc3,rad=0.15'),
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Region labels (small, italic) ──
    ax.text(0.15, 2.2, 'LR-TT wins', fontsize=10.5, color='#2E7D32', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    ax.text(2.2, 0.1, 'Digital LoRA wins', fontsize=10.5, color='#1565C0', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    ax.text(1.6, 0.58, 'γ=0 exclusive', fontsize=8.5, color='#BF360C', style='italic',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Axes ──
    ax.set_xlabel('φ = t$_{dcim,write}$ / t$_{acim,upd}$    (Update speed comparison)', fontsize=10)
    ax.set_ylabel('β = t$_{dcim}$ / t$_{acim}$    (MVM speed comparison)', fontsize=10)
    ax.set_xlim(0, 3.0); ax.set_ylim(0, 2.5)
    ax.tick_params(direction='in')

    # ── Legend ──
    handles = [
        Line2D([0], [0], color='#333', lw=1.5, ls='-', label='γ=1 boundary'),
        Line2D([0], [0], color='#333', lw=1, ls='--', label='γ=0 boundary'),
        Line2D([0], [0], marker='o', color='#C62828', ls='', markersize=6,
               markeredgecolor='white', label='ALPINE ref.'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper right', frameon=True,
              edgecolor='#DDD', fancybox=False, framealpha=0.95)

    # ── Title (subtle) ──
    ax.set_title('Digital LoRA vs LR-TT: latency break-even\n'
                 'Shading for α = 0.3;  lines for α ∈ {0.01, 0.1, 0.3, 1.0, 3.0}',
                 fontsize=10, pad=10)

    plt.tight_layout()
    p = f'{OUT}/fig1_unified_break_even.png'
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  {p}")

    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == "__main__":
    main()

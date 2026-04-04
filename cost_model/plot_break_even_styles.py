#!/usr/bin/env python3
"""Break-even plot in 6 different visual styles."""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

OUT = "/root/paper_figures_anchored/styles"
os.makedirs(OUT, exist_ok=True)

ALPHAS = [0.01, 0.1, 0.3, 1.0, 3.0]
COLORS_A = ['#FFD600', '#FF9800', '#E53935', '#7B1FA2', '#283593']
PHIS = np.linspace(0, 3.0, 600)


def bg1(alpha, phi): return (2 + alpha*(1-phi)) / 3
def bg0(alpha, phi): return (1 + alpha*(1-phi)) / 3


# ═══════════════════════════════════════════════════
# Style 1: Minimal lines only (Nature style)
# ═══════════════════════════════════════════════════
def style_1():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_facecolor('#FAFAFA')

    for alpha, color in zip(ALPHAS, COLORS_A):
        g1 = bg1(alpha, PHIS); g0 = bg0(alpha, PHIS)
        m1 = (g1 > 0) & (g1 < 2.5); m0 = (g0 > 0) & (g0 < 2.5)
        ax.plot(PHIS[m1], g1[m1], '-', color=color, lw=2)
        ax.plot(PHIS[m0], g0[m0], '--', color=color, lw=1.5, alpha=0.6)
        y = bg1(alpha, 0)
        if 0.1 < y < 2.4:
            ax.text(-0.05, y, f'{alpha}', fontsize=7, color=color, fontweight='bold', va='center', ha='right')

    ax.plot(0.3, 1.0, '*', color='red', markersize=12, markeredgecolor='white', zorder=10)
    ax.axhline(1, color='#ccc', ls=':', lw=0.5)
    ax.axvline(1, color='#ccc', ls=':', lw=0.5)
    ax.set_xlabel('φ (Update: Adam / Pulsed)', fontsize=10)
    ax.set_ylabel('β (MVM: DCIM / ACIM)', fontsize=10)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.5)
    ax.set_title('Style 1: Minimal Lines', fontsize=11, fontweight='bold')
    ax.text(0.3, 2.2, 'LR-TT', fontsize=11, color='#2E7D32', fontweight='bold')
    ax.text(2.5, 0.15, 'DL', fontsize=11, color='#1565C0', fontweight='bold')
    fig.savefig(f'{OUT}/style1_minimal.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(); print("  style1")


# ═══════════════════════════════════════════════════
# Style 2: Filled regions with α=0.3 focus
# ═══════════════════════════════════════════════════
def style_2():
    fig, ax = plt.subplots(figsize=(7, 5.5))

    a = 0.3
    g1 = bg1(a, PHIS); g0 = bg0(a, PHIS)
    ax.fill_between(PHIS, 2.5, np.clip(g1, 0, 2.5), color='#C8E6C9', alpha=0.7, label='LR-TT wins (both γ)')
    ax.fill_between(PHIS, np.clip(g1, 0, 2.5), np.clip(g0, 0, 2.5), color='#FFE0B2', alpha=0.7, label='γ=0 exclusive')
    ax.fill_between(PHIS, 0, np.clip(g0, 0, 2.5), color='#BBDEFB', alpha=0.7, label='Digital LoRA wins')
    ax.plot(PHIS, g1, '-', color='#2E7D32', lw=2.5)
    ax.plot(PHIS, g0, '--', color='#E65100', lw=2.5)

    # Other α as thin lines
    for alpha, color in zip(ALPHAS, COLORS_A):
        if alpha == 0.3: continue
        ax.plot(PHIS, np.clip(bg1(alpha, PHIS), 0, 2.5), '-', color=color, lw=1, alpha=0.4)

    ax.plot(0.3, 1.0, '*', color='#D32F2F', markersize=14, markeredgecolor='black', zorder=10)
    ax.set_xlabel('φ (Update: Adam / Pulsed)', fontsize=10)
    ax.set_ylabel('β (MVM: DCIM / ACIM)', fontsize=10)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Style 2: Filled Regions (α=0.3 focus)', fontsize=11, fontweight='bold')
    fig.savefig(f'{OUT}/style2_filled.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(); print("  style2")


# ═══════════════════════════════════════════════════
# Style 3: Heatmap with contour overlay
# ═══════════════════════════════════════════════════
def style_3():
    fig, ax = plt.subplots(figsize=(7, 5.5))

    PHI, BETA = np.meshgrid(np.linspace(0, 3, 300), np.linspace(0, 2.5, 300))
    a = 0.3
    # "advantage" for LR-TT γ=1: positive = LR-TT wins
    adv = (2 + a*(1-PHI)) / 3 - BETA

    cmap = LinearSegmentedColormap.from_list('rg', ['#1565C0', '#E3F2FD', 'white', '#E8F5E9', '#2E7D32'])
    im = ax.contourf(PHI, BETA, adv, levels=np.linspace(-1.5, 1.5, 30), cmap=cmap, alpha=0.8)
    ax.contour(PHI, BETA, adv, levels=[0], colors='black', linewidths=2.5)  # γ=1 boundary

    adv0 = (1 + a*(1-PHI)) / 3 - BETA
    ax.contour(PHI, BETA, adv0, levels=[0], colors='#E65100', linewidths=2, linestyles='--')  # γ=0

    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('LR-TT advantage (α=0.3)', fontsize=9)

    ax.plot(0.3, 1.0, '*', color='red', markersize=14, markeredgecolor='white', zorder=10)
    ax.set_xlabel('φ (Update: Adam / Pulsed)', fontsize=10)
    ax.set_ylabel('β (MVM: DCIM / ACIM)', fontsize=10)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.5)
    ax.set_title('Style 3: Heatmap + Contour', fontsize=11, fontweight='bold')
    fig.savefig(f'{OUT}/style3_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(); print("  style3")


# ═══════════════════════════════════════════════════
# Style 4: Multi-α fan with gradient bands
# ═══════════════════════════════════════════════════
def style_4():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_facecolor('#F5F5F5')

    for alpha, color in zip(ALPHAS, COLORS_A):
        g1 = np.clip(bg1(alpha, PHIS), 0, 2.5)
        g0 = np.clip(bg0(alpha, PHIS), 0, 2.5)
        ax.fill_between(PHIS, g0, g1, color=color, alpha=0.12)
        ax.plot(PHIS, g1, '-', color=color, lw=2, label=f'α={alpha} (γ=1)')
        ax.plot(PHIS, g0, '--', color=color, lw=1.2, alpha=0.5)

    ax.plot(0.3, 1.0, '*', color='red', markersize=14, markeredgecolor='white', zorder=10)
    ax.annotate('★ ALPINE', xy=(0.3, 1.0), xytext=(0.7, 1.4), fontsize=9,
                fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    ax.axhline(1, color='#999', ls=':', lw=0.5)
    ax.set_xlabel('φ (Update: Adam / Pulsed)', fontsize=10)
    ax.set_ylabel('β (MVM: DCIM / ACIM)', fontsize=10)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.5)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.set_title('Style 4: Fan Bands (γ=0 zone as colored bands)', fontsize=11, fontweight='bold')
    fig.savefig(f'{OUT}/style4_fan.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(); print("  style4")


# ═══════════════════════════════════════════════════
# Style 5: Dark theme
# ═══════════════════════════════════════════════════
def style_5():
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(7, 5.5))

        a = 0.3
        g1 = bg1(a, PHIS); g0 = bg0(a, PHIS)
        ax.fill_between(PHIS, 2.5, np.clip(g1, 0, 2.5), color='#1B5E20', alpha=0.4)
        ax.fill_between(PHIS, np.clip(g1, 0, 2.5), np.clip(g0, 0, 2.5), color='#E65100', alpha=0.3)
        ax.fill_between(PHIS, 0, np.clip(g0, 0, 2.5), color='#0D47A1', alpha=0.4)
        ax.plot(PHIS, g1, '-', color='#69F0AE', lw=2.5)
        ax.plot(PHIS, g0, '--', color='#FFD54F', lw=2)

        for alpha, color in zip([0.01, 0.1, 1.0, 3.0], ['#FFF176', '#FFB74D', '#CE93D8', '#90CAF9']):
            ax.plot(PHIS, np.clip(bg1(alpha, PHIS), 0, 2.5), '-', color=color, lw=0.8, alpha=0.5)

        ax.plot(0.3, 1.0, '*', color='#FF5252', markersize=14, zorder=10)
        ax.text(0.5, 1.2, 'ALPINE', color='#FF5252', fontsize=9, fontweight='bold')
        ax.text(0.3, 2.2, 'LR-TT wins', color='#69F0AE', fontsize=12, fontweight='bold')
        ax.text(2.3, 0.15, 'DL wins', color='#82B1FF', fontsize=12, fontweight='bold')
        ax.text(1.8, 0.65, 'γ=0 only', color='#FFD54F', fontsize=10, fontweight='bold', style='italic')

        ax.set_xlabel('φ (Update: Adam / Pulsed)', fontsize=10, color='white')
        ax.set_ylabel('β (MVM: DCIM / ACIM)', fontsize=10, color='white')
        ax.set_xlim(0, 3); ax.set_ylim(0, 2.5)
        ax.set_title('Style 5: Dark Theme (α=0.3)', fontsize=11, fontweight='bold')
        fig.savefig(f'{OUT}/style5_dark.png', dpi=200, bbox_inches='tight')
        plt.close(); print("  style5")


# ═══════════════════════════════════════════════════
# Style 6: Publication clean (two-panel: α=0.3 + α sweep)
# ═══════════════════════════════════════════════════
def style_6():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: clean filled α=0.3
    ax = axes[0]
    a = 0.3
    g1 = bg1(a, PHIS); g0 = bg0(a, PHIS)
    ax.fill_between(PHIS, 2.5, np.clip(g1, 0, 2.5), color='#A5D6A7', alpha=0.6)
    ax.fill_between(PHIS, np.clip(g1, 0, 2.5), np.clip(g0, 0, 2.5), color='#FFCC80', alpha=0.6)
    ax.fill_between(PHIS, 0, np.clip(g0, 0, 2.5), color='#90CAF9', alpha=0.6)
    ax.plot(PHIS, g1, '-', color='#2E7D32', lw=2.5, label='γ=1 boundary')
    ax.plot(PHIS, g0, '--', color='#E65100', lw=2, label='γ=0 boundary')

    ax.plot(0.3, 1.0, '*', color='#D32F2F', markersize=14, markeredgecolor='black', zorder=10, label='ALPINE-like')

    ax.text(0.2, 2.1, 'LR-TT wins', fontsize=11, color='#1B5E20', fontweight='bold')
    ax.text(2.2, 0.15, 'DL wins', fontsize=11, color='#0D47A1', fontweight='bold')
    ax.text(1.5, 0.6, 'γ=0 excl.', fontsize=9, color='#BF360C', fontweight='bold')

    ax.axhline(1, color='#999', ls=':', lw=0.5)
    ax.set_xlabel('φ = t_dcim_write / t_acim_upd', fontsize=10)
    ax.set_ylabel('β = t_dcim / t_acim', fontsize=10)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('(a) α=0.3 (6T1C reference)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.1)

    # Right: α sweep (lines only)
    ax = axes[1]
    ax.set_facecolor('#FAFAFA')

    for alpha, color in zip(ALPHAS, COLORS_A):
        g1c = np.clip(bg1(alpha, PHIS), 0, 2.5)
        g0c = np.clip(bg0(alpha, PHIS), 0, 2.5)
        lw = 2.5 if alpha == 0.3 else 1.5
        ax.plot(PHIS, g1c, '-', color=color, lw=lw)
        ax.plot(PHIS, g0c, '--', color=color, lw=lw*0.7, alpha=0.6)

    # Arrow showing α direction
    ax.annotate('', xy=(0.05, 0.35), xytext=(0.05, 0.95),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.15, 0.6, 'α↓\nLR-TT\nbetter', fontsize=8, fontweight='bold', ha='left')

    ax.plot(0.3, 1.0, '*', color='#D32F2F', markersize=12, markeredgecolor='black', zorder=10)

    # Legend for α values
    handles = [Line2D([0],[0], color=c, lw=2, label=f'α={a}') for a, c in zip(ALPHAS, COLORS_A)]
    handles.append(Line2D([0],[0], color='black', lw=1.5, ls='-', label='γ=1 (solid)'))
    handles.append(Line2D([0],[0], color='black', lw=1, ls='--', label='γ=0 (dashed)'))
    ax.legend(handles=handles, fontsize=7, ncol=2, loc='upper right')

    ax.axhline(1, color='#999', ls=':', lw=0.5)
    ax.set_xlabel('φ = t_dcim_write / t_acim_upd', fontsize=10)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2.5)
    ax.set_title('(b) α sweep: above line = LR-TT wins', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.1)

    fig.suptitle('Digital LoRA vs LR-TT: Latency Break-Even', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT}/style6_publication.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(); print("  style6")


if __name__ == "__main__":
    print(f"Generating 6 break-even styles in {OUT}/\n")
    style_1(); style_2(); style_3(); style_4(); style_5(); style_6()
    print(f"\nDone.")

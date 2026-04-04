#!/usr/bin/env python3
"""ρ_proj sweep at fixed rank=32: latency crossover between tile-level and element-level."""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from extract_layer_inventory import build_layer_inventory, get_targeted_layers, compute_adapter_tile_counts

OUT = "/root/paper_figures_anchored"
inventory = build_layer_inventory()
tgt = get_targeted_layers(inventory, 'all')
BT = 48 * 384
R = 32

T_ACIM = 200   # ns
T_UPD = 60     # ns (α=0.3)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
})


def compute_latency(rho_proj, gamma):
    """Compute adapter latency in ms for LR-TT and TikiTaka at given ρ_proj."""

    # ── LR-TT ──
    lr_proj_ns = 0; lr_upd_ns = 0; lr_vis_ns = 0
    for l in tgt:
        atc = compute_adapter_tile_counts(l, R)
        nA, nB = atc['n_tiles_A'], atc['n_tiles_B']

        # Projection: tile-level count × ρ_proj
        lr_proj_ns += rho_proj * BT * (atc['n_tiles_B_cols'] + atc['n_tiles_A_rows']) * T_ACIM

        # Update: tile-level (ρ_proj doesn't apply to updates)
        lr_upd_ns += BT * (nA + nB) * T_UPD

        # Visible (γ=1): also scaled by ρ_proj
        if gamma == 1:
            lr_vis_ns += rho_proj * BT * (atc['n_tiles_B_cols'] + atc['n_tiles_A_cols']) * T_ACIM

    lr_adapter_ms = (lr_proj_ns + lr_upd_ns + lr_vis_ns) / 1e6

    # ── TikiTaka ──
    tt_upd_ns = sum(BT * l.n_tiles * T_UPD for l in tgt)
    tt_vis_ns = sum(BT * l.n_tile_cols * T_ACIM for l in tgt) if gamma == 1 else 0
    tt_adapter_ms = (tt_upd_ns + tt_vis_ns) / 1e6

    # ── Base (common) ──
    base_fwd = sum(BT * l.n_tile_cols * T_ACIM for l in inventory)
    base_ms = base_fwd * (1 + 2.5) / 1e6  # fwd + k_bwd*fwd

    return {
        'lr_adapter': lr_adapter_ms,
        'lr_total': base_ms + lr_adapter_ms,
        'lr_proj': lr_proj_ns / 1e6,
        'lr_upd': lr_upd_ns / 1e6,
        'lr_vis': lr_vis_ns / 1e6,
        'tt_adapter': tt_adapter_ms,
        'tt_total': base_ms + tt_adapter_ms,
        'base': base_ms,
    }


def main():
    rhos = np.linspace(0, 1.0, 500)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ─────────────────────────────────────
    # Top left: Total step latency vs ρ_proj
    # ─────────────────────────────────────
    ax = axes[0, 0]
    for gamma, ls, alpha in [(0, '-', 1.0), (1, '--', 0.7)]:
        lr_totals = [compute_latency(rp, gamma)['lr_total'] for rp in rhos]
        tt_total = compute_latency(0, gamma)['tt_total']  # TT doesn't depend on ρ

        ax.plot(rhos, lr_totals, ls, color='#1565C0', lw=2, alpha=alpha, label=f'LR-TT r=32 γ={gamma}')
        ax.axhline(tt_total, color='#E65100', ls=ls, lw=2, alpha=alpha, label=f'TikiTaka γ={gamma}')

    ax.set_xlabel('ρ_proj', fontsize=10)
    ax.set_ylabel('Total step latency (ms)', fontsize=10)
    ax.set_title('(a) Total step latency', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.1)

    # ─────────────────────────────────────
    # Top right: Adapter-only latency vs ρ_proj
    # ─────────────────────────────────────
    ax = axes[0, 1]
    for gamma, ls, alpha in [(0, '-', 1.0), (1, '--', 0.7)]:
        lr_adapters = [compute_latency(rp, gamma)['lr_adapter'] for rp in rhos]
        tt_adapter = compute_latency(0, gamma)['tt_adapter']

        ax.plot(rhos, lr_adapters, ls, color='#1565C0', lw=2, alpha=alpha, label=f'LR-TT r=32 γ={gamma}')
        ax.axhline(tt_adapter, color='#E65100', ls=ls, lw=2, alpha=alpha, label=f'TikiTaka γ={gamma}')

        # Find crossover
        for i, rp in enumerate(rhos):
            if lr_adapters[i] >= tt_adapter and i > 0:
                rp_cross = rhos[i]
                ax.axvline(rp_cross, color='gray', ls=':', lw=0.8)
                ax.text(rp_cross + 0.02, tt_adapter * 0.5,
                        f'ρ*={rp_cross:.2f}\n(γ={gamma})', fontsize=8, color='#C62828',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
                break

    ax.set_xlabel('ρ_proj', fontsize=10)
    ax.set_ylabel('Adapter latency (ms)', fontsize=10)
    ax.set_title('(b) Adapter-only latency', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.1)

    # ─────────────────────────────────────
    # Bottom left: LR-TT adapter decomposition vs ρ_proj
    # ─────────────────────────────────────
    ax = axes[1, 0]
    gamma = 0
    projs = [compute_latency(rp, gamma)['lr_proj'] for rp in rhos]
    upds = [compute_latency(rp, gamma)['lr_upd'] for rp in rhos]
    viss = [compute_latency(rp, gamma)['lr_vis'] for rp in rhos]

    ax.fill_between(rhos, 0, upds, color='#FFA726', alpha=0.7, label='Update (fixed)')
    ax.fill_between(rhos, upds, [u+p for u,p in zip(upds, projs)], color='#EF5350', alpha=0.7, label='Projection (∝ ρ)')
    ax.axhline(compute_latency(0, 0)['tt_adapter'], color='#E65100', lw=2.5, ls='--', label='TikiTaka γ=0')

    ax.set_xlabel('ρ_proj', fontsize=10)
    ax.set_ylabel('LR-TT adapter latency (ms), γ=0', fontsize=10)
    ax.set_title('(c) LR-TT decomposition (γ=0)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.1)

    # ─────────────────────────────────────
    # Bottom right: Speedup (TT/LR-TT total latency)
    # ─────────────────────────────────────
    ax = axes[1, 1]
    for gamma, color, ls in [(0, '#1565C0', '-'), (1, '#E53935', '--')]:
        speedups = []
        for rp in rhos:
            c = compute_latency(rp, gamma)
            speedups.append(c['tt_total'] / c['lr_total'])
        ax.plot(rhos, speedups, ls, color=color, lw=2, label=f'γ={gamma}')

    ax.axhline(1.0, color='#999', ls=':', lw=0.8)
    ax.fill_between(rhos, 1.0, [max(s, 1) for s in speedups], alpha=0.05, color='green')
    ax.fill_between(rhos, [min(s, 1) for s in speedups], 1.0, alpha=0.05, color='red')
    ax.text(0.05, 1.05, 'LR-TT faster', fontsize=9, color='#2E7D32', fontweight='bold')
    ax.text(0.7, 0.92, 'TikiTaka faster', fontsize=9, color='#C62828', fontweight='bold')

    # Physical reference: ρ ≈ r/512
    rho_phys = R / 512
    ax.axvline(rho_phys, color='#9C27B0', ls='--', lw=1.5)
    ax.text(rho_phys + 0.01, ax.get_ylim()[1]*0.95, f'ρ≈r/512\n={rho_phys:.3f}',
            fontsize=8, color='#9C27B0', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_xlabel('ρ_proj', fontsize=10)
    ax.set_ylabel('Speedup (TT_total / LR-TT_total)', fontsize=10)
    ax.set_title('(d) Total step speedup', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.1)

    fig.suptitle(f'ρ_proj Sweep at rank={R}: Latency Crossover  |  t_acim=200ns, t_upd=60ns, target=all\n'
                 f'ρ=0 → element-level (LR-TT wins)  |  ρ=1 → tile-level (TikiTaka wins)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f'{OUT}/rho_sweep_r32.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  rho_sweep_r32.png")


if __name__ == "__main__":
    main()
    plt.rcParams.update(plt.rcParamsDefault)

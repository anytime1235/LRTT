#!/usr/bin/env python3
"""Paper-ready figures: DL vs LR-TT (3 main + 2 supplementary)."""

import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from extract_layer_inventory import build_layer_inventory, get_targeted_layers, compute_adapter_tile_counts

OUT = "/root/paper_figures"
os.makedirs(OUT, exist_ok=True)

inventory = build_layer_inventory()
tgt = get_targeted_layers(inventory, 'all')
BS, S, BT = 48, 384, 48*384
r = 8

# ── Compute K = BT × Σ(nA+nB) ──
K = sum(BT * (compute_adapter_tile_counts(l, r)['n_tiles_A'] +
              compute_adapter_tile_counts(l, r)['n_tiles_B']) for l in tgt)

# ── Adapter footprint ──
adapter_params = sum(l.M*r + r*l.N for l in tgt)
full_params = sum(l.M*l.N for l in tgt)

print(f"K = {K:,}  |  adapter_params = {adapter_params:,}  |  full_params = {full_params:,}")
print(f"Layers = {len(tgt)}")


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE 1: Latency Break-Even Heatmap (β vs α)
# ═══════════════════════════════════════════════════════════════

def fig_main_1():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    alphas = np.linspace(0.01, 4.0, 400)
    betas = np.linspace(0.01, 2.5, 400)
    A, B = np.meshgrid(alphas, betas)

    for ax_i, (delta, delta_label) in enumerate([(0, 'δ=0 (Adam free)'),
                                                   (0.2, 'δ=0.2'),
                                                   (0.5, 'δ=0.5 (Adam costly)')]):
        ax = axes[ax_i]

        # Break-even boundaries
        # γ=1: β* = (2 + α − δ) / 3
        # γ=0: β* = (1 + α − δ) / 3
        beta_star_g1 = (2 + A - delta) / 3
        beta_star_g0 = (1 + A - delta) / 3

        # Regions:
        # B < β*(γ=0): DL wins always
        # β*(γ=0) < B < β*(γ=1): γ=0-exclusive LR-TT win
        # B > β*(γ=1): LR-TT wins always

        region = np.zeros_like(A)
        region[B < beta_star_g0] = 0     # DL wins
        region[(B >= beta_star_g0) & (B < beta_star_g1)] = 1  # γ=0 exclusive
        region[B >= beta_star_g1] = 2    # LR-TT wins

        colors = ['#BBDEFB', '#FFE0B2', '#C8E6C9']  # blue, orange, green
        cmap = LinearSegmentedColormap.from_list('regions', colors, N=3)

        ax.contourf(A, B, region, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.7)
        ax.contour(A, B, B - beta_star_g1, levels=[0], colors='#2E7D32', linewidths=2.5, linestyles='-')
        ax.contour(A, B, B - beta_star_g0, levels=[0], colors='#E65100', linewidths=2.5, linestyles='--')

        # Labels in regions
        ax.text(3.0, 0.3, 'Digital LoRA\nwins', fontsize=10, ha='center', color='#1565C0', fontweight='bold')
        ax.text(0.5, 2.0, 'LR-TT\nwins\n(γ=0,1)', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')

        # Find center of γ=0 exclusive zone
        mid_alpha = 2.0
        mid_g0 = (1 + mid_alpha - delta) / 3
        mid_g1 = (2 + mid_alpha - delta) / 3
        mid_beta = (mid_g0 + mid_g1) / 2
        if mid_beta > 0.3 and mid_beta < 2.3:
            ax.text(mid_alpha, mid_beta, 'γ=0\nexclusive', fontsize=9,
                    ha='center', color='#E65100', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#E65100', alpha=0.9))

        # β=1 reference line
        ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
        ax.text(0.15, 1.05, 'β=1\n(same speed)', fontsize=7, color='gray')

        ax.set_xlabel('α = t_acim_upd / t_acim', fontsize=11)
        if ax_i == 0:
            ax.set_ylabel('β = t_dcim / t_acim', fontsize=11)
        ax.set_title(delta_label, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 4); ax.set_ylim(0, 2.5)
        ax.grid(alpha=0.15)

    # Legend
    handles = [
        mpatches.Patch(color='#BBDEFB', alpha=0.7, label='Digital LoRA wins (both γ)'),
        mpatches.Patch(color='#FFE0B2', alpha=0.7, label='γ=0 exclusive: LR-TT wins only with hidden-carry'),
        mpatches.Patch(color='#C8E6C9', alpha=0.7, label='LR-TT wins (both γ)'),
        plt.Line2D([0],[0], color='#2E7D32', lw=2.5, ls='-', label='γ=1 boundary'),
        plt.Line2D([0],[0], color='#E65100', lw=2.5, ls='--', label='γ=0 boundary'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle('Digital LoRA (always γ=1) vs LR-TT: Latency Break-Even Region\n'
                 'β* = (2+α−δ)/3 [γ=1],  β* = (1+α−δ)/3 [γ=0]   |   target=all, r=8',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    p = f'{OUT}/main1_break_even_region.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {p}")


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE 2: Adapter-Path Decomposition Bars
# ═══════════════════════════════════════════════════════════════

def fig_main_2():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Components in units of K
    methods = ['Digital LoRA\n(always γ=1)', 'LR-TT\n(γ=1)', 'LR-TT\n(γ=0)']
    x = np.arange(len(methods))
    w = 0.5

    # DL: 3K total = 1K fwd_mvm + 1K fwd_mvm (part of fwd is 2 ops using nA+nB)
    # Actually: fwd=1K(nB)+1K(nA)? No, let's use the per-coefficient structure
    # DL: forward=BT×(nA+nB)=K, backward=2K → 3K MVMs + Adam
    # LRTT γ=1: proj=K, vis=K, upd=K → 2K MVMs + K updates
    # LRTT γ=0: proj=K, upd=K → K MVMs + K updates

    components = {
        'Forward DCIM-MVM': [1, 0, 0],
        'Backward DCIM-MVM': [2, 0, 0],
        'Adam write-back': [0.2, 0, 0],  # symbolic δ=0.2
        'Projection ACIM-MVM': [0, 1, 1],
        'Visible ACIM-MVM': [0, 1, 0],
        'Pulsed update': [0, 1, 1],
        'Transfer': [0, 0.01, 0.01],
    }

    colors = {
        'Forward DCIM-MVM': '#42A5F5',
        'Backward DCIM-MVM': '#1976D2',
        'Adam write-back': '#90CAF9',
        'Projection ACIM-MVM': '#EF5350',
        'Visible ACIM-MVM': '#FF7043',
        'Pulsed update': '#FFA726',
        'Transfer': '#CE93D8',
    }

    bottom = np.zeros(3)
    for comp, vals in components.items():
        bars = ax.bar(x, vals, w, bottom=bottom, label=comp, color=colors[comp], edgecolor='#333', linewidth=0.5)
        bottom += np.array(vals)

    # Annotate totals
    totals = [3.2, 3.01, 2.01]
    for i, t in enumerate(totals):
        ax.text(i, t + 0.08, f'{t:.1f}K', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Normalized latency (units of K = BT×Σ(nA+nB))', fontsize=11)
    ax.set_title('Adapter-Path Latency Decomposition\n'
                 f'K = {K:,} tile-vector events  |  target=all, r=8, δ=0.2',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 4)

    # Annotation: γ=0 advantage
    ax.annotate('', xy=(2, 2.01), xytext=(1, 3.01),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=2))
    ax.text(1.7, 2.7, '−1K\n(no visible\npath)', fontsize=9, color='#E65100',
            fontweight='bold', ha='center')

    plt.tight_layout()
    p = f'{OUT}/main2_adapter_decomposition.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {p}")


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE 3: Memory / State Footprint
# ═══════════════════════════════════════════════════════════════

def fig_main_3():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Digital LoRA vs LR-TT state footprint ──
    ax = axes[0]
    methods = ['Digital LoRA', 'LR-TT']

    # DL: A,B weights + Adam m + Adam v = 3× adapter params (FP16 = 2 bytes each)
    dl_weights = adapter_params * 2 / 1e6   # MB
    dl_adam_m = adapter_params * 2 / 1e6
    dl_adam_v = adapter_params * 2 / 1e6
    dl_buffer = adapter_params * 2 / 1e6    # gradient buffer

    # LR-TT: A,B on analog tiles (no digital storage needed for weights)
    lr_analog = adapter_params  # elements (on-tile, not SRAM)
    lr_digital = 0  # no Adam, no digital weight copy

    x = np.arange(2)
    w = 0.5

    bars_w = ax.bar(x, [dl_weights, 0], w, label='A,B weights (SRAM)', color='#42A5F5', edgecolor='#333')
    bars_m = ax.bar(x, [dl_adam_m, 0], w, bottom=[dl_weights, 0], label='Adam m (SRAM)', color='#1976D2', edgecolor='#333')
    bars_v = ax.bar(x, [dl_adam_v, 0], w, bottom=[dl_weights+dl_adam_m, 0], label='Adam v (SRAM)', color='#0D47A1', edgecolor='#333')
    bars_b = ax.bar(x, [dl_buffer, 0], w, bottom=[dl_weights+dl_adam_m+dl_adam_v, 0], label='Grad buffer (SRAM)', color='#90CAF9', edgecolor='#333')

    # LR-TT analog state
    ax.bar(x[1], lr_analog/1e6*2, w, label='A,B on analog tiles\n(no SRAM)', color='#FFA726', edgecolor='#333', hatch='///')

    dl_total = dl_weights + dl_adam_m + dl_adam_v + dl_buffer
    lr_total_sram = 0

    ax.text(0, dl_total + 0.3, f'{dl_total:.1f} MB\nSRAM', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
    ax.text(1, lr_analog/1e6*2 + 0.3, f'0 MB SRAM\n({adapter_params:,} elements\non analog tile)', ha='center', fontsize=9, fontweight='bold', color='#E65100')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Digital SRAM requirement (MB)', fontsize=11)
    ax.set_title('Training State Footprint (r=8, target=all)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # ── Right: Rank sweep — SRAM requirement ──
    ax = axes[1]
    ranks = [4, 8, 16, 32]

    for target, color, marker in [('attention', '#42A5F5', 'o'), ('ffn', '#EF5350', 's'), ('all', '#AB47BC', '^')]:
        tgt_t = get_targeted_layers(inventory, target)
        dl_srams = []
        for rk in ranks:
            p = sum(l.M*rk + rk*l.N for l in tgt_t)
            dl_srams.append(p * 4 * 2 / 1e6)  # 4× (w, m, v, buf) × 2 bytes
        ax.plot(ranks, dl_srams, f'-{marker}', color=color, label=f'DL {target}', linewidth=2, markersize=8)

    ax.axhline(0, color='#FFA726', lw=3, ls='--', label='LR-TT (0 MB SRAM, any rank)')
    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Digital LoRA SRAM (MB)', fontsize=11)
    ax.set_title('SRAM Requirement vs Rank', fontsize=12, fontweight='bold')
    ax.set_xticks(ranks)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Memory / State Footprint: Digital LoRA requires SRAM, LR-TT is fully on-tile',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = f'{OUT}/main3_state_footprint.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {p}")


# ═══════════════════════════════════════════════════════════════
# SUPPLEMENTARY 1: Sequence-Length Sensitivity
# ═══════════════════════════════════════════════════════════════

def fig_supp_1():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    S_vals = [64, 128, 256, 384]

    # Left: break-even β* vs S
    ax = axes[0]
    for alpha, color, ls in [(0.5, '#4CAF50', '-'), (1.0, '#FF9800', '--'), (2.0, '#F44336', ':')]:
        for gamma, marker, lbl in [(1, 'o', 'γ=1'), (0, 's', 'γ=0')]:
            betas = []
            for s in S_vals:
                # β* doesn't actually depend on S (it cancels out in the ratio)
                # But the COMMON path changes with S, affecting delta/total ratio
                if gamma == 1:
                    beta_star = (2 + alpha) / 3
                else:
                    beta_star = (1 + alpha) / 3
                betas.append(beta_star)
            if gamma == 1:
                ax.plot(S_vals, betas, f'{ls}{marker}', color=color,
                        label=f'α={alpha}, {lbl}', linewidth=2, markersize=7)
            else:
                ax.plot(S_vals, betas, f'{ls}{marker}', color=color,
                        linewidth=2, markersize=7, alpha=0.5)

    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel('Sequence length S', fontsize=11)
    ax.set_ylabel('Break-even β*', fontsize=11)
    ax.set_title('β* vs S (δ=0)\n(β* is S-independent: cancels in ratio)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(S_vals)

    # Right: ΔT contribution vs S (adapter delta grows linearly with S)
    ax = axes[1]
    K_vals = [BS * s * 384 / S * 384 / 384 for s in S_vals]  # actually K ∝ S

    # Common path also ∝ S, but common path tiles are base (480) vs adapter (384)
    base_tiles = sum(l.n_tiles for l in inventory)  # 480
    adapter_tiles = sum(compute_adapter_tile_counts(l, r)['n_tiles_A'] +
                        compute_adapter_tile_counts(l, r)['n_tiles_B'] for l in tgt)

    for s in S_vals:
        bt = BS * s
        common_events = bt * base_tiles * 3.5  # fwd + k_bwd*fwd (k=2.5)
        dl_events = bt * adapter_tiles * 3     # 3K
        lr_events_g1 = bt * adapter_tiles * 3  # 2K MVM + 1K upd
        lr_events_g0 = bt * adapter_tiles * 2  # 1K MVM + 1K upd

    # Plot delta/total ratio
    for method, color, label in [('DL', '#2196F3', 'Digital LoRA'),
                                   ('LR-TT γ=1', '#F44336', 'LR-TT γ=1'),
                                   ('LR-TT γ=0', '#FF9800', 'LR-TT γ=0')]:
        ratios = []
        for s in S_vals:
            bt = BS * s
            common = bt * base_tiles * 3.5
            if method == 'DL':
                delta = bt * adapter_tiles * 3.2  # 3K + δ
            elif method == 'LR-TT γ=1':
                delta = bt * adapter_tiles * 3.0
            else:
                delta = bt * adapter_tiles * 2.0
            ratios.append(delta / (common + delta) * 100)
        ax.plot(S_vals, ratios, '-o', color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel('Sequence length S', fontsize=11)
    ax.set_ylabel('ΔT / T_total (%)', fontsize=11)
    ax.set_title('Adapter overhead as % of total step cost', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(S_vals)

    fig.suptitle('Supplementary: Sequence-Length Sensitivity', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = f'{OUT}/supp1_seq_length_sensitivity.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {p}")


# ═══════════════════════════════════════════════════════════════
# SUPPLEMENTARY 2: LR-TT vs TikiTaka (same-ACIM ratio)
# ═══════════════════════════════════════════════════════════════

def fig_supp_2():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    RANKS = [4, 8, 16, 32]

    # Left: element-level update ratio vs rank (per target)
    ax = axes[0]
    for target, color, marker in [('attention', '#42A5F5', 'o'), ('ffn', '#EF5350', 's'), ('all', '#AB47BC', '^')]:
        tgt_t = get_targeted_layers(inventory, target)
        tt_upd = sum(BT * l.M * l.N for l in tgt_t)
        ratios = []
        for rk in RANKS:
            lr_upd = sum(BT * (l.M*rk + rk*l.N) for l in tgt_t)
            ratios.append(tt_upd / lr_upd)
        ax.plot(RANKS, ratios, f'-{marker}', color=color, label=target, linewidth=2, markersize=8)

    ax.set_xlabel('LR-TT rank', fontsize=11)
    ax.set_ylabel('TT/LR-TT update element ratio', fontsize=11)
    ax.set_title('Update Footprint Ratio (same ACIM family)\nγ-independent', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(RANKS)
    ax.set_yscale('log')

    # Right: model size scaling
    ax = axes[1]
    dims = [768, 1024, 2048, 4096, 8192]
    labels = ['BERT\n768', 'GPT-2S\n1024', 'GPT-2M\n2048', 'GPT-2L\n4096', 'LLaMA\n8192']
    for rk, color, marker in [(4, '#E91E63', 'o'), (8, '#F44336', 's'), (16, '#FF9800', '^'), (32, '#FFC107', 'D')]:
        ratios = [d**2 / (rk * 2 * d) for d in dims]
        ax.plot(range(len(dims)), ratios, f'-{marker}', color=color, label=f'r={rk}', linewidth=2, markersize=8)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('TT/LR-TT ratio', fontsize=11)
    ax.set_title('Model Size Scaling: ratio ≈ d/(2r)', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle('Supplementary: LR-TT vs TikiTaka (Same-ACIM Element-Level Comparison)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = f'{OUT}/supp2_lrtt_vs_tikitaka.png'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {p}")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating paper figures in {OUT}/\n")
    fig_main_1()
    fig_main_2()
    fig_main_3()
    fig_supp_1()
    fig_supp_2()
    print(f"\nAll figures saved to {OUT}/")

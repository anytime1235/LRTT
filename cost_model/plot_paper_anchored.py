#!/usr/bin/env python3
"""Paper figures with literature-anchored hardware values.

Anchors:
  ALPINE:  t_tile=100ns (256×256) → t_acim≈200ns (512×512, conservative 2×)
  ALBERT chip: analog tile ~20 TOPS/W, digital aux ~5-7 TOPS/W
  TTv2 training: update time 56.3-62.1 ns (per-tile)
  SRAM-CIM: t_dcim ≈ 50-200ns range (literature)
"""

import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from extract_layer_inventory import build_layer_inventory, get_targeted_layers, compute_adapter_tile_counts

OUT = "/root/paper_figures_anchored"
os.makedirs(OUT, exist_ok=True)

inventory = build_layer_inventory()
tgt_all = get_targeted_layers(inventory, 'all')
BS, S, BT = 48, 384, 48*384
RANKS = [4, 8, 16, 32]

# ── Literature anchors ──
T_ACIM_NS = 200    # ALPINE 100ns (256×256) → 200ns (512×512)
T_ACIM_RANGE = [100, 200, 400]  # sensitivity
T_DCIM_RANGE = [50, 100, 200]   # SRAM-CIM literature
T_UPD_RANGE = [60, 200, 500, 1000]  # TTv2: 56-62ns, higher BL → more

# ── Compute K per rank ──
def get_K(rank):
    return sum(BT * (compute_adapter_tile_counts(l, rank)['n_tiles_A'] +
                     compute_adapter_tile_counts(l, rank)['n_tiles_B']) for l in tgt_all)

def get_adapter_params(rank):
    return sum(l.M*rank + rank*l.N for l in tgt_all)

K8 = get_K(8)
print(f"K(r=8) = {K8:,} | adapter_params(r=8) = {get_adapter_params(8):,}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Break-Even with Literature Overlay
# ═══════════════════════════════════════════════════════════════

def fig1_break_even_anchored():
    """Break-even with proper axes: β (MVM comparison) vs φ (Update comparison).

    β = t_dcim / t_acim        : who has faster MVM?
    φ = t_dcim_write / t_acim_upd : who has faster update?

    Break-even (per K unit):
      DL:       3 × t_dcim + t_dcim_write
      LRTT γ=1: 2 × t_acim + t_acim_upd
      LRTT γ=0: 1 × t_acim + t_acim_upd

    Normalize by t_acim, let α = t_acim_upd/t_acim (device constant):
      DL:       3β + φα
      LRTT γ=1: 2 + α
      LRTT γ=0: 1 + α

    Break-even γ=1: β* = [2 + α(1−φ)] / 3
    Break-even γ=0: β* = [1 + α(1−φ)] / 3
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    phis = np.linspace(0.01, 3.0, 500)   # φ = t_dcim_write / t_acim_upd
    betas = np.linspace(0.01, 3.0, 500)  # β = t_dcim / t_acim
    PHI, BETA = np.meshgrid(phis, betas)

    for ax_i, (alpha, alpha_label) in enumerate([
        (0.3, 'α=0.3 (6T1C, TTv2-like)'),
        (1.0, 'α=1.0 (update = read)'),
        (2.0, 'α=2.0 (slow update device)'),
    ]):
        ax = axes[ax_i]

        beta_g1 = (2 + alpha * (1 - PHI)) / 3
        beta_g0 = (1 + alpha * (1 - PHI)) / 3

        region = np.zeros_like(PHI)
        region[BETA < beta_g0] = 0     # DL wins
        region[(BETA >= beta_g0) & (BETA < beta_g1)] = 1  # γ=0 exclusive
        region[BETA >= beta_g1] = 2    # LR-TT wins

        colors = ['#BBDEFB', '#FFE0B2', '#C8E6C9']
        ax.contourf(PHI, BETA, region, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.6)
        ax.contour(PHI, BETA, BETA - beta_g1, levels=[0], colors='#2E7D32', linewidths=2.5)
        ax.contour(PHI, BETA, BETA - beta_g0, levels=[0], colors='#E65100', linewidths=2.5, linestyles='--')

        # Region labels
        ax.text(2.3, 0.3, 'DL wins', fontsize=10, ha='center', color='#1565C0', fontweight='bold')
        ax.text(0.3, 2.5, 'LR-TT\nwins', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')

        # γ=0 exclusive zone label (find a good spot)
        mid_phi = 1.5
        mid_g0 = (1 + alpha*(1-mid_phi)) / 3
        mid_g1 = (2 + alpha*(1-mid_phi)) / 3
        mid_b = (mid_g0 + mid_g1) / 2
        if 0.3 < mid_b < 2.5:
            ax.text(mid_phi, mid_b, 'γ=0\nexcl.', fontsize=9, ha='center', color='#E65100',
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#FFF3E0', ec='#E65100', alpha=0.9))

        # Reference lines
        ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.4)
        ax.axvline(1.0, color='gray', ls=':', lw=0.8, alpha=0.4)
        ax.text(0.05, 1.05, 'same MVM speed', fontsize=7, color='gray')
        ax.text(1.05, 0.05, 'same update speed', fontsize=7, color='gray', rotation=90)

        # Literature anchor points
        if alpha == 0.3:
            # ALPINE-like: β≈1 (same MVM), φ≈0.3 (Adam fast)
            ax.plot(0.3, 1.0, '*', color='#D32F2F', markersize=16, markeredgecolor='black', markeredgewidth=1, zorder=10)
            ax.annotate('ALPINE-like', xy=(0.3, 1.0), xytext=(0.8, 0.5),
                        fontsize=8, fontweight='bold', color='#D32F2F',
                        arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.2))

            # Fast DCIM + fast Adam
            ax.plot(0.2, 0.5, 'D', color='#1565C0', markersize=10, markeredgecolor='black', zorder=10)
            ax.annotate('Fast DCIM\n+Fast Adam', xy=(0.2, 0.5), xytext=(0.8, 0.2),
                        fontsize=7, color='#1565C0', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1))

        ax.set_xlabel('φ = t_dcim_write / t_acim_upd\n(update speed comparison)', fontsize=10)
        if ax_i == 0:
            ax.set_ylabel('β = t_dcim / t_acim\n(MVM speed comparison)', fontsize=10)
        ax.set_title(alpha_label, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 3); ax.set_ylim(0, 3)
        ax.grid(alpha=0.15)

    handles = [
        mpatches.Patch(color='#BBDEFB', alpha=0.6, label='Digital LoRA wins'),
        mpatches.Patch(color='#FFE0B2', alpha=0.6, label='γ=0 exclusive (hidden-carry only)'),
        mpatches.Patch(color='#C8E6C9', alpha=0.6, label='LR-TT wins (both γ)'),
        plt.Line2D([0],[0], color='#2E7D32', lw=2.5, label='γ=1 boundary: β*=(2+α(1−φ))/3'),
        plt.Line2D([0],[0], color='#E65100', lw=2.5, ls='--', label='γ=0 boundary: β*=(1+α(1−φ))/3'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle('Digital LoRA vs LR-TT: Latency Break-Even\n'
                 'β = t_dcim/t_acim (MVM comparison)  |  φ = t_dcim_write/t_acim_upd (Update comparison)\n'
                 'α = t_acim_upd/t_acim is device constant (panel)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    fig.savefig(f'{OUT}/fig1b_break_even_phi_vs_beta.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig1b_break_even_phi_vs_beta.png")


def fig1a_break_even_alpha():
    """Original axes: β (MVM comparison) vs α (ACIM internal update/read ratio).

    β = t_dcim / t_acim
    α = t_acim_upd / t_acim

    Break-even:
      γ=1: β* = (2 + α − δ) / 3
      γ=0: β* = (1 + α − δ) / 3
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    alphas = np.linspace(0.01, 5.0, 500)
    betas = np.linspace(0.01, 3.0, 500)
    A, B = np.meshgrid(alphas, betas)

    for ax_i, (delta, delta_label) in enumerate([
        (0, 'δ=0 (Adam free)'),
        (0.2, 'δ=0.2'),
        (0.5, 'δ=0.5 (Adam costly)'),
    ]):
        ax = axes[ax_i]

        beta_g1 = (2 + A - delta) / 3
        beta_g0 = (1 + A - delta) / 3

        region = np.zeros_like(A)
        region[B < beta_g0] = 0
        region[(B >= beta_g0) & (B < beta_g1)] = 1
        region[B >= beta_g1] = 2

        colors = ['#BBDEFB', '#FFE0B2', '#C8E6C9']
        ax.contourf(A, B, region, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors, alpha=0.6)
        ax.contour(A, B, B - beta_g1, levels=[0], colors='#2E7D32', linewidths=2.5)
        ax.contour(A, B, B - beta_g0, levels=[0], colors='#E65100', linewidths=2.5, linestyles='--')

        ax.text(4.0, 0.4, 'DL wins', fontsize=10, ha='center', color='#1565C0', fontweight='bold')
        ax.text(0.8, 2.5, 'LR-TT\nwins', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')
        mid_a = 2.5
        mid_g0 = (1 + mid_a - delta) / 3
        mid_g1 = (2 + mid_a - delta) / 3
        mid_b = (mid_g0 + mid_g1) / 2
        if 0.3 < mid_b < 2.5:
            ax.text(mid_a, mid_b, 'γ=0\nexcl.', fontsize=9, ha='center', color='#E65100',
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='#FFF3E0', ec='#E65100', alpha=0.9))

        ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.4)

        if ax_i == 0:
            ax.plot(0.3, 1.0, '*', color='#D32F2F', markersize=16, markeredgecolor='black', markeredgewidth=1, zorder=10)
            ax.annotate('ALPINE-like\n(α=0.3, β=1)', xy=(0.3, 1.0), xytext=(1.2, 0.4),
                        fontsize=8, fontweight='bold', color='#D32F2F',
                        arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.2))

        ax.set_xlabel('α = t_acim_upd / t_acim\n(ACIM internal: update vs read)', fontsize=10)
        if ax_i == 0:
            ax.set_ylabel('β = t_dcim / t_acim\n(MVM speed comparison)', fontsize=10)
        ax.set_title(delta_label, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 5); ax.set_ylim(0, 3)
        ax.grid(alpha=0.15)

    handles = [
        mpatches.Patch(color='#BBDEFB', alpha=0.6, label='Digital LoRA wins'),
        mpatches.Patch(color='#FFE0B2', alpha=0.6, label='γ=0 exclusive (hidden-carry only)'),
        mpatches.Patch(color='#C8E6C9', alpha=0.6, label='LR-TT wins (both γ)'),
        plt.Line2D([0],[0], color='#2E7D32', lw=2.5, label='γ=1 boundary: β*=(2+α−δ)/3'),
        plt.Line2D([0],[0], color='#E65100', lw=2.5, ls='--', label='γ=0 boundary: β*=(1+α−δ)/3'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle('Digital LoRA vs LR-TT: Latency Break-Even (ACIM-internal view)\n'
                 'β = t_dcim/t_acim  |  α = t_acim_upd/t_acim  |  δ = t_adam/t_acim (panel)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    fig.savefig(f'{OUT}/fig1a_break_even_alpha_vs_beta.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig1a_break_even_alpha_vs_beta.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Absolute Adapter Latency (ns) with t_acim anchor
# ═══════════════════════════════════════════════════════════════

def fig2_absolute_latency():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    r = 8
    K = get_K(r)

    # Left: bar chart at t_acim=200ns, t_upd=200ns, t_dcim=200ns
    ax = axes[0]
    t_acim = 200; t_upd = 200; t_dcim = 200; t_adam = 20  # ns

    dl_fwd = K * t_dcim / 1e9     # seconds
    dl_bwd = 2 * K * t_dcim / 1e9
    dl_adam = K * t_adam / 1e9 * 0.2  # δ=0.2 equivalent

    lr1_proj = K * t_acim / 1e9
    lr1_vis = K * t_acim / 1e9
    lr1_upd = K * t_upd / 1e9

    lr0_proj = K * t_acim / 1e9
    lr0_upd = K * t_upd / 1e9

    methods = ['Digital LoRA\n(always γ=1)', 'LR-TT (γ=1)', 'LR-TT (γ=0)']
    x = np.arange(3)
    w = 0.5

    # Stack bars
    c1 = ['#42A5F5', '#EF5350', '#EF5350']
    c2 = ['#1976D2', '#FF7043', '#FF7043']
    c3 = ['#90CAF9', '#FFA726', '#FFA726']

    b1 = [dl_fwd, lr1_vis, 0]
    b2 = [dl_bwd, lr1_proj, lr0_proj]
    b3 = [dl_adam, lr1_upd, lr0_upd]

    labels1 = ['Fwd DCIM-MVM', 'Visible ACIM-MVM', '']
    labels2 = ['Bwd DCIM-MVM', 'Projection ACIM-MVM', 'Projection ACIM-MVM']
    labels3 = ['Adam write', 'Pulsed update', 'Pulsed update']

    ax.bar(x, b1, w, color=[c1[0], c1[1], c1[2]], edgecolor='#333', linewidth=0.5, label='Forward/Visible MVM')
    ax.bar(x, b2, w, bottom=b1, color=[c2[0], c2[1], c2[2]], edgecolor='#333', linewidth=0.5, label='Backward/Projection MVM')
    bot2 = [a+b for a,b in zip(b1, b2)]
    ax.bar(x, b3, w, bottom=bot2, color=[c3[0], c3[1], c3[2]], edgecolor='#333', linewidth=0.5, label='Adam/Pulsed update')

    totals = [sum(v) for v in zip(b1, b2, b3)]
    for i, t in enumerate(totals):
        ax.text(i, t + 0.05, f'{t:.2f}s', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Adapter ΔT per step (seconds)', fontsize=11)
    ax.set_title(f'Absolute Latency\nt_acim={t_acim}ns, t_upd={t_upd}ns, t_dcim={t_dcim}ns', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Annotate γ=0 savings
    ax.annotate('', xy=(2, totals[2]), xytext=(1, totals[1]),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=2))
    saving_pct = (1 - totals[2]/totals[1]) * 100
    ax.text(1.7, (totals[1]+totals[2])/2, f'−{saving_pct:.0f}%\n(γ=0)', fontsize=10,
            color='#E65100', fontweight='bold', ha='center')

    # Right: sensitivity across t_acim range
    ax = axes[1]
    t_acims = np.linspace(50, 500, 100)

    for t_upd_val, color, ls in [(60, '#4CAF50', '-'), (200, '#FF9800', '--'), (500, '#F44336', ':')]:
        dl_total = 3 * K * 200 / 1e9 + K * 20 * 0.2 / 1e9  # DL at fixed t_dcim=200
        lr1_vals = [(2*K*ta + K*t_upd_val) / 1e9 for ta in t_acims]
        lr0_vals = [(K*ta + K*t_upd_val) / 1e9 for ta in t_acims]

        ax.plot(t_acims, lr1_vals, ls, color=color, linewidth=2, label=f'LR-TT γ=1, t_upd={t_upd_val}ns')
        ax.plot(t_acims, lr0_vals, ls, color=color, linewidth=2, alpha=0.5)

    ax.axhline(dl_total, color='#1976D2', lw=3, label=f'DL (t_dcim=200ns)')
    ax.fill_between(t_acims, 0, dl_total, alpha=0.05, color='blue')

    ax.set_xlabel('t_acim (ns)', fontsize=11)
    ax.set_ylabel('Adapter ΔT per step (seconds)', fontsize=11)
    ax.set_title('Latency Sensitivity\n(solid=γ=1, faded=γ=0, blue line=DL)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # ALPINE anchor
    ax.axvline(200, color='red', ls=':', lw=1.5, alpha=0.5)
    ax.text(210, ax.get_ylim()[1]*0.9, 'ALPINE\nanchor', fontsize=8, color='red')

    fig.suptitle(f'Adapter-Path Absolute Latency (target=all, r=8, K={K:,} tile events)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig2_absolute_latency.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig2_absolute_latency.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Per-Layer Overhead % (AHWA-LoRA style)
# ═══════════════════════════════════════════════════════════════

def fig3_per_layer_overhead():
    fig, ax = plt.subplots(figsize=(14, 5))

    t_acim = 200; t_upd = 200; t_dcim = 200; k_bwd = 2.5

    layer_names = []
    dl_pcts = []
    lr1_pcts = []
    lr0_pcts = []

    for i, l in enumerate(tgt_all):
        atc = compute_adapter_tile_counts(l, 8)
        nA, nB = atc['n_tiles_A'], atc['n_tiles_B']
        nC = l.n_tiles

        # Common layer latency (base fwd + bwd)
        T_common = BT * l.n_tile_cols * t_acim * (1 + k_bwd)  # ns

        # DL adapter delta
        T_dl = BT * (3*nA + 3*nB) * t_dcim  # 6 matmuls

        # LR-TT γ=1
        T_lr1 = BT * (nA+nB) * t_acim * 2 + BT * (nA+nB) * t_upd  # proj+vis + upd

        # LR-TT γ=0
        T_lr0 = BT * (nA+nB) * t_acim + BT * (nA+nB) * t_upd  # proj + upd

        dl_pcts.append(T_dl / T_common * 100)
        lr1_pcts.append(T_lr1 / T_common * 100)
        lr0_pcts.append(T_lr0 / T_common * 100)

        short = l.sub_name.split('.')[-1][:3]
        layer_names.append(f'L{l.layer_idx}.{short}')

    x = np.arange(len(tgt_all))

    ax.plot(x, dl_pcts, '-', color='#1976D2', linewidth=1.5, alpha=0.8, label='Digital LoRA')
    ax.plot(x, lr1_pcts, '-', color='#F44336', linewidth=1.5, alpha=0.8, label='LR-TT (γ=1)')
    ax.plot(x, lr0_pcts, '-', color='#FF9800', linewidth=1.5, alpha=0.8, label='LR-TT (γ=0)')

    # Shade attention vs FFN regions
    for i, l in enumerate(tgt_all):
        if l.group == 'ffn':
            ax.axvspan(i-0.5, i+0.5, alpha=0.05, color='purple')

    ax.set_xlabel('Layer index', fontsize=11)
    ax.set_ylabel('Adapter ΔT / Base Layer T (%)', fontsize=11)
    ax.set_title(f'Per-Layer Adapter Overhead (AHWA-LoRA style)\nt_acim={t_acim}ns, t_dcim={t_dcim}ns, t_upd={t_upd}ns, r=8',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Mark attention vs FFN
    ax.text(12, max(dl_pcts)*0.95, 'attention\n[768×768]', fontsize=9, ha='center', color='gray')
    ax.text(54, max(dl_pcts)*0.95, 'FFN\n[3072×768]', fontsize=9, ha='center', color='purple')

    # Set x ticks sparse
    tick_idx = list(range(0, len(tgt_all), 6))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([layer_names[i] for i in tick_idx], fontsize=7, rotation=45)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig3_per_layer_overhead.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig3_per_layer_overhead.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: State Footprint (concrete MB)
# ═══════════════════════════════════════════════════════════════

def fig4_state_footprint():
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Digital LoRA', 'LR-TT']
    x = np.arange(2)
    w = 0.4

    r = 8
    ap = get_adapter_params(r)
    proj_buf = 2 * BT * r * 2 / 1e6  # MB (shared by both)

    # DL: weights + m + v + grad_buf + proj_buf
    dl_w = ap * 2 / 1e6
    dl_m = ap * 2 / 1e6
    dl_v = ap * 2 / 1e6
    dl_g = ap * 2 / 1e6
    dl_pb = proj_buf

    # LR-TT: proj_buf only (weights on analog tile)
    lr_pb = proj_buf

    ax.bar(0, dl_w, w, label='A,B weights', color='#42A5F5', edgecolor='#333')
    ax.bar(0, dl_m, w, bottom=dl_w, label='Adam m', color='#1976D2', edgecolor='#333')
    ax.bar(0, dl_v, w, bottom=dl_w+dl_m, label='Adam v', color='#0D47A1', edgecolor='#333')
    ax.bar(0, dl_g, w, bottom=dl_w+dl_m+dl_v, label='Gradient buffer', color='#90CAF9', edgecolor='#333')
    ax.bar(0, dl_pb, w, bottom=dl_w+dl_m+dl_v+dl_g, label='Projection buffer\n(shared)', color='#E0E0E0', edgecolor='#333')

    ax.bar(1, lr_pb, w, color='#E0E0E0', edgecolor='#333')

    dl_total = dl_w + dl_m + dl_v + dl_g + dl_pb
    lr_total = lr_pb

    ax.text(0, dl_total + 0.3, f'{dl_total:.1f} MB\ntotal SRAM', ha='center', fontsize=11, fontweight='bold', color='#1565C0')
    ax.text(1, lr_total + 0.3, f'{lr_total:.1f} MB\n(projection buffer only)', ha='center', fontsize=11, fontweight='bold', color='#E65100')

    # Savings arrow
    ax.annotate('', xy=(1, lr_total), xytext=(0, dl_total),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5))
    saving = dl_total - lr_total
    ax.text(0.5, (dl_total+lr_total)/2 + 0.5, f'−{saving:.1f} MB\n({saving/dl_total*100:.0f}% savings)',
            ha='center', fontsize=11, fontweight='bold', color='#2E7D32')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel('Digital SRAM (MB)', fontsize=11)
    ax.set_title(f'Training State Footprint\nr=8, target=all, adapter params={ap:,}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig4_state_footprint.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig4_state_footprint.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5 (Supp): LR-TT vs TikiTaka — Absolute latency at anchor
# ═══════════════════════════════════════════════════════════════

def fig5_lrtt_vs_tt_anchored():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    t_acim = 200

    # Left: absolute update latency vs rank at different t_upd
    ax = axes[0]
    for t_upd, color, ls in [(60, '#4CAF50', '-'), (200, '#FF9800', '--'), (500, '#F44336', ':')]:
        tt_vals = []
        lr_vals_g0 = []
        for r in RANKS:
            tt_upd = sum(BT * l.M * l.N for l in tgt_all) * t_upd / 1e9
            lr_proj = sum(BT * (l.M*r + r*l.N) for l in tgt_all) * t_acim / 1e9
            lr_upd = sum(BT * (l.M*r + r*l.N) for l in tgt_all) * t_upd / 1e9
            tt_vals.append(tt_upd)
            lr_vals_g0.append(lr_proj + lr_upd)

        ax.plot(RANKS, lr_vals_g0, f'{ls}o', color=color, linewidth=2, markersize=7,
                label=f'LR-TT γ=0, t_upd={t_upd}ns')
        if t_upd == 200:
            ax.axhline(tt_vals[0], color=color, lw=3, alpha=0.3, label=f'TikiTaka, t_upd={t_upd}ns')

    ax.set_xlabel('LR-TT rank', fontsize=11)
    ax.set_ylabel('ΔT adapter (seconds/step)', fontsize=11)
    ax.set_title(f'LR-TT vs TikiTaka: Absolute Latency\nt_acim={t_acim}ns', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(RANKS)
    ax.set_yscale('log')

    # Right: TT/LRTT ratio at different t_upd (element-level, not tile)
    ax = axes[1]
    for t_upd, color, ls in [(60, '#4CAF50', '-'), (200, '#FF9800', '--'), (500, '#F44336', ':')]:
        ratios = []
        for r in RANKS:
            tt_cost = sum(BT * l.M * l.N * t_upd for l in tgt_all)
            lr_cost = sum(BT * (l.M*r+r*l.N) * t_acim + BT * (l.M*r+r*l.N) * t_upd for l in tgt_all)
            ratios.append(tt_cost / lr_cost if lr_cost > 0 else 1)
        ax.plot(RANKS, ratios, f'{ls}o', color=color, linewidth=2, markersize=7,
                label=f't_upd={t_upd}ns')

    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel('LR-TT rank', fontsize=11)
    ax.set_ylabel('TT/LR-TT cost ratio', fontsize=11)
    ax.set_title(f'TT/LR-TT Ratio at t_acim={t_acim}ns\n(above 1 = LR-TT wins)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(RANKS)

    fig.suptitle('LR-TT vs TikiTaka with Literature Anchor (same ACIM)', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig5_lrtt_vs_tt_anchored.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig5_lrtt_vs_tt_anchored.png")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating anchored paper figures in {OUT}/\n")
    fig1a_break_even_alpha()        # α vs β (ACIM-internal view)
    fig1_break_even_anchored()      # φ vs β (cross-method view)
    fig2_absolute_latency()
    fig3_per_layer_overhead()
    fig4_state_footprint()
    fig5_lrtt_vs_tt_anchored()
    print(f"\nAll figures saved to {OUT}/")

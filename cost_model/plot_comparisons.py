#!/usr/bin/env python3
"""Generate all comparison plots for Digital LoRA vs LR-TT and LR-TT vs TikiTaka."""

import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT = "/root/comparison_plots"
os.makedirs(OUT, exist_ok=True)

# ─── Constants ───
BT = 48 * 384
RANKS = [4, 8, 16, 32]
shapes = {
    'attention': [(768,768)] * 48,
    'ffn': [(3072,768)] * 12 + [(768,3072)] * 12,
    'all': [(768,768)] * 48 + [(3072,768)] * 12 + [(768,3072)] * 12,
}

def dl_f_eff(target, r):
    return sum(2*BT*r*(N+M)+2*BT*r*(2*M+2*N) for M,N in shapes[target]) + 5*sum(M*r+r*N for M,N in shapes[target])

def lr_elem(target, r, gamma):
    upd = sum(BT*(M*r+r*N) for M,N in shapes[target])
    proj = upd
    vis = proj if gamma == 1 else 0
    return upd, proj, vis

def tt_elem(target, gamma):
    upd = sum(BT*M*N for M,N in shapes[target])
    vis = upd if gamma == 1 else 0
    return upd, vis

C = {
    'dl_blue': '#2196F3', 'lr_red': '#F44336', 'lr_orange': '#FF9800',
    'tt_green': '#4CAF50', 'tt_purple': '#9C27B0',
    'r4': '#E91E63', 'r8': '#F44336', 'r16': '#FF9800', 'r32': '#FFC107',
}

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Digital LoRA vs LR-TT — Operation Count Comparison
# ═══════════════════════════════════════════════════════════════
def fig1_op_count():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_i, target in enumerate(['attention', 'ffn', 'all']):
        ax = axes[ax_i]
        x = np.arange(len(RANKS))
        w = 0.35

        dl_vals = [dl_f_eff(target, r)/1e9 for r in RANKS]
        lr_vals = [sum(lr_elem(target, r, 1))/1e9 for r in RANKS]

        bars1 = ax.bar(x - w/2, dl_vals, w, label='Digital LoRA (F_eff)', color=C['dl_blue'], edgecolor='#333')
        bars2 = ax.bar(x + w/2, lr_vals, w, label='LR-TT (elem, γ=1)', color=C['lr_red'], edgecolor='#333')

        for i, (d, l) in enumerate(zip(dl_vals, lr_vals)):
            ax.text(i, max(d, l)*1.05, f'{d/l:.1f}×', ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'r={r}' for r in RANKS])
        ax.set_ylabel('Operations (billions)' if ax_i == 0 else '', fontsize=11)
        ax.set_title(f'target={target}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Fig 1: Operation Count — Digital LoRA vs LR-TT (DL always 2.0× more ops)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig1_dl_vs_lrtt_op_count.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig1_dl_vs_lrtt_op_count.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Digital LoRA vs LR-TT — Break-Even Surface
# ═══════════════════════════════════════════════════════════════
def fig2_break_even():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    r = 8; target = 'all'
    F = dl_f_eff(target, r)
    upd, proj, vis = lr_elem(target, r, 1)

    # Left: θ_gemm* vs α at different ρ_proj
    ax = axes[0]
    alphas = np.linspace(0.01, 5.0, 200)
    for rp, color, ls in [(0.0, '#4CAF50', '-'), (0.1, '#8BC34A', '--'),
                           (0.5, '#FF9800', '-.'), (1.0, '#F44336', ':')]:
        thresholds = []
        for a in alphas:
            num = rp * proj + a * upd + vis
            theta_star = num / F * 256  # ns/FLOP at θ_mvm=256
            throughput_mflops = 1e3 / theta_star if theta_star > 0 else 1e6
            thresholds.append(throughput_mflops)
        ax.plot(alphas, thresholds, ls, color=color, label=f'ρ_proj={rp}', linewidth=2)

    ax.set_xlabel('α = θ_upd / θ_mvm', fontsize=11)
    ax.set_ylabel('Break-even digital throughput (MFLOPS)', fontsize=11)
    ax.set_title('Digital LoRA wins above line', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.axhspan(1e3, 1e5, alpha=0.08, color='blue', label='DPU range (1-100 GFLOPS)')
    ax.axhspan(1, 1e3, alpha=0.08, color='red', label='PMCA range (1-1000 MFLOPS)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim(1, 1e5)

    # Right: 2D heatmap α vs ρ_proj → throughput
    ax = axes[1]
    alphas_2d = np.linspace(0.01, 5.0, 100)
    rhos_2d = np.linspace(0.0, 1.0, 100)
    A, R = np.meshgrid(alphas_2d, rhos_2d)
    Z = np.zeros_like(A)
    for i in range(len(rhos_2d)):
        for j in range(len(alphas_2d)):
            num = rhos_2d[i]*proj + alphas_2d[j]*upd + vis
            theta_star = num / F * 256
            Z[i, j] = 1e3 / theta_star if theta_star > 0 else 1e6

    cs = ax.contourf(A, R, np.log10(Z), levels=20, cmap='RdYlGn_r')
    cb = plt.colorbar(cs, ax=ax)
    cb.set_label('log₁₀(MFLOPS needed)')
    ax.contour(A, R, Z, levels=[10, 100, 1000, 10000], colors='black', linewidths=1)
    ax.set_xlabel('α = θ_upd / θ_mvm', fontsize=11)
    ax.set_ylabel('ρ_proj', fontsize=11)
    ax.set_title('Break-even surface (r=8, γ=1, all)', fontsize=12, fontweight='bold')

    fig.suptitle('Fig 2: Digital LoRA vs LR-TT Break-Even (θ_mvm=256ns)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig2_dl_vs_lrtt_break_even.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig2_dl_vs_lrtt_break_even.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: LR-TT vs TikiTaka — Update Element Ratio vs Rank
# ═══════════════════════════════════════════════════════════════
def fig3_update_ratio():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_i, target in enumerate(['attention', 'ffn', 'all']):
        ax = axes[ax_i]
        tt_upd_val, _ = tt_elem(target, 0)

        ratios_g0 = []
        ratios_g1 = []
        for r in RANKS:
            lr_upd, lr_proj, lr_vis = lr_elem(target, r, 0)
            ratios_g0.append(tt_upd_val / lr_upd)
            lr_upd1, lr_proj1, lr_vis1 = lr_elem(target, r, 1)
            tt_upd1, tt_vis1 = tt_elem(target, 1)
            ratios_g1.append((tt_upd1+tt_vis1) / (lr_upd1+lr_proj1+lr_vis1))

        ax.bar(np.arange(len(RANKS))-0.18, ratios_g0, 0.35,
               label='Update-only (γ=0, ρ=0)', color=C['tt_green'], edgecolor='#333')
        ax.bar(np.arange(len(RANKS))+0.18, ratios_g1, 0.35,
               label='Full cost (γ=1, ρ=1.0)', color=C['tt_purple'], edgecolor='#333')

        for i, (g0, g1) in enumerate(zip(ratios_g0, ratios_g1)):
            ax.text(i-0.18, g0+2, f'{g0:.0f}×', ha='center', fontsize=8, fontweight='bold')
            ax.text(i+0.18, g1+2, f'{g1:.0f}×', ha='center', fontsize=8, fontweight='bold')

        ax.axhline(1.0, color='gray', ls=':', lw=1)
        ax.set_xticks(range(len(RANKS)))
        ax.set_xticklabels([f'r={r}' for r in RANKS])
        ax.set_ylabel('TT/LR-TT ratio' if ax_i == 0 else '', fontsize=11)
        ax.set_title(f'target={target}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Fig 3: LR-TT vs TikiTaka Element-Level Ratio (above 1 = LR-TT wins)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig3_lrtt_vs_tt_update_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig3_lrtt_vs_tt_update_ratio.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: LR-TT vs TikiTaka — ρ_proj Sensitivity (Element)
# ═══════════════════════════════════════════════════════════════
def fig4_rho_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rhos = np.linspace(0, 2.0, 200)

    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]
        target = 'all'
        tt_upd_val, tt_vis_val = tt_elem(target, gamma)
        tt_total = tt_upd_val + tt_vis_val

        for r, color, lw in [(4, C['r4'], 1.5), (8, C['r8'], 2.5),
                               (16, C['r16'], 1.5), (32, C['r32'], 1.5)]:
            lr_upd, lr_proj, lr_vis = lr_elem(target, r, gamma)
            vals = []
            for rp in rhos:
                lr_cost = rp * lr_proj + lr_upd + lr_vis
                vals.append(tt_total / lr_cost if lr_cost > 0 else 1)
            ax.plot(rhos, vals, '-', color=color, label=f'r={r}', linewidth=lw)

        ax.axhline(1.0, color='gray', ls=':', lw=1)
        ax.fill_between(rhos, 1.0, 200, alpha=0.05, color='green')
        ax.fill_between(rhos, 0.5, 1.0, alpha=0.05, color='red')
        ax.set_xlabel('ρ_proj (projection cost / update cost)', fontsize=11)
        ax.set_ylabel('C_TT / C_LRTT (element-level)', fontsize=11)
        ax.set_title(f'γ={gamma}, target=all', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
        ax.set_ylim(0.8, 200)

    fig.suptitle('Fig 4: Element-Level Ratio vs ρ_proj (above 1 = LR-TT wins)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig4_lrtt_vs_tt_rho_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig4_lrtt_vs_tt_rho_sensitivity.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Model Size Scaling — Future Projection
# ═══════════════════════════════════════════════════════════════
def fig5_model_scaling():
    fig, ax = plt.subplots(figsize=(10, 6))

    dims = [768, 1024, 2048, 4096, 8192, 16384]
    labels = ['BERT\n768', 'GPT-2S\n1024', 'GPT-2M\n2048', 'GPT-2L\n4096', 'LLaMA-7B\n8192', 'LLaMA-70B\n16384']

    for r, color, marker in [(4, C['r4'], 'o'), (8, C['r8'], 's'),
                               (16, C['r16'], '^'), (32, C['r32'], 'D')]:
        ratios = [d*d / (r*(d+d)) for d in dims]  # M=N=d case
        ax.plot(range(len(dims)), ratios, f'-{marker}', color=color, label=f'r={r}',
                linewidth=2, markersize=8)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('TT/LR-TT update ratio (element-level)', fontsize=11)
    ax.set_title('Fig 5: LR-TT Advantage Scales with Model Size', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Annotate key points
    r8_bert = 768**2 / (8*2*768)
    r8_llama = 8192**2 / (8*2*8192)
    ax.annotate(f'{r8_bert:.0f}×', (0, r8_bert), textcoords="offset points",
                xytext=(15, -5), fontsize=9, color=C['r8'], fontweight='bold')
    ax.annotate(f'{r8_llama:.0f}×', (4, r8_llama), textcoords="offset points",
                xytext=(15, -5), fontsize=9, color=C['r8'], fontweight='bold')

    fig.savefig(f'{OUT}/fig5_model_size_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig5_model_size_scaling.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Adapter Footprint Comparison
# ═══════════════════════════════════════════════════════════════
def fig6_footprint():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_i, target in enumerate(['attention', 'ffn', 'all']):
        ax = axes[ax_i]
        tt_fp = sum(M*N for M,N in shapes[target])
        lr_fps = [sum(M*r+r*N for M,N in shapes[target]) for r in RANKS]

        x = np.arange(len(RANKS) + 1)
        heights = lr_fps + [tt_fp]
        colors_bar = [C['lr_red']]*len(RANKS) + [C['tt_green']]
        labels_bar = [f'LR-TT\nr={r}' for r in RANKS] + ['TikiTaka\nfull-rank']

        bars = ax.bar(x, [h/1e6 for h in heights], 0.6, color=colors_bar, edgecolor='#333')

        for i, h in enumerate(heights):
            if i < len(RANKS):
                ratio = tt_fp / h
                ax.text(i, h/1e6 * 1.1, f'{ratio:.0f}×\nsmaller', ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, fontsize=8)
        ax.set_ylabel('Adapter parameters (millions)' if ax_i == 0 else '', fontsize=10)
        ax.set_title(f'target={target}', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Fig 6: Adapter Footprint — LR-TT vs TikiTaka',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig6_adapter_footprint.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig6_adapter_footprint.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 7: Combined Summary — Both Comparisons
# ═══════════════════════════════════════════════════════════════
def fig7_summary():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: DL vs LR-TT — who wins at what digital throughput
    ax = axes[0]
    target = 'all'
    alphas_x = np.linspace(0.01, 5.0, 200)
    for r, color in [(4, C['r4']), (8, C['r8']), (16, C['r16']), (32, C['r32'])]:
        F = dl_f_eff(target, r)
        upd, proj, vis = lr_elem(target, r, 1)
        rp = 0.1
        vals = []
        for a in alphas_x:
            num = rp * proj + a * upd + vis
            theta_star = num / F * 256
            vals.append(1e3 / theta_star if theta_star > 0 else 1e6)
        ax.plot(alphas_x, vals, '-', color=color, label=f'r={r}', linewidth=2)

    ax.axhspan(1e3, 1e5, alpha=0.08, color='blue')
    ax.axhspan(10, 1e3, alpha=0.08, color='orange')
    ax.axhspan(0.1, 10, alpha=0.08, color='red')
    ax.text(0.5, 5e3, 'DPU: DL wins', fontsize=10, color='blue', fontweight='bold')
    ax.text(0.5, 100, 'PMCA: depends on α', fontsize=10, color='orange')
    ax.text(0.5, 1, 'MCU: LR-TT wins', fontsize=10, color='red', fontweight='bold')

    ax.set_xlabel('α = θ_upd / θ_mvm', fontsize=11)
    ax.set_ylabel('Break-even MFLOPS', fontsize=11)
    ax.set_title('Digital LoRA vs LR-TT', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: LR-TT vs TT — element-level ratio
    ax = axes[1]
    for target, color, ls in [('attention', '#42A5F5', '-'), ('ffn', '#EF5350', '--'), ('all', '#AB47BC', '-.')]:
        tt_upd_val, _ = tt_elem(target, 0)
        ratios = [tt_upd_val / lr_elem(target, r, 0)[0] for r in RANKS]
        ax.plot(RANKS, ratios, f'{ls}o', color=color, label=f'{target}', linewidth=2, markersize=8)

    ax.set_xlabel('LR-TT rank', fontsize=11)
    ax.set_ylabel('TT/LR-TT update ratio', fontsize=11)
    ax.set_title('LR-TT vs TikiTaka (element-level)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(RANKS)

    fig.suptitle('Fig 7: Summary — Two Comparisons', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig7_combined_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig7_combined_summary.png")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating comparison plots in {OUT}/")
    fig1_op_count()
    fig2_break_even()
    fig3_update_ratio()
    fig4_rho_sensitivity()
    fig5_model_scaling()
    fig6_footprint()
    fig7_summary()
    print(f"\nAll 7 figures saved to {OUT}/")

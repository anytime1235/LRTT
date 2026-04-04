#!/usr/bin/env python3
"""Main paper figures: non-algebraic, context-rich plots.

Fig A: Total Step Time (T_common + DeltaT) — stacked bar showing adapter overhead in context
Fig B: Cost Breakdown — proj / upd / vis / transfer components, across alpha_t
Fig C: Per-layer group contribution — attention vs FFN breakdown
"""

import os, sys, math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
COST_MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "cost_model")
sys.path.insert(0, COST_MODEL_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from alpine_calibrated_model import (
    build_layer_inventory, get_targeted_layers,
    compute_lrtt_element_counts, compute_tikitaka_element_counts,
    compute_model_a,
    ALPHA_T_SCENARIOS, RANKS, TARGETS,
    BATCH_SIZE, S_PAD_DEFAULT, TAU_ACIM,
)

OUT = SCRIPT_DIR

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

# Colors
C_COMMON = '#BDBDBD'
C_PROJ = '#1565C0'
C_UPD_LR = '#42A5F5'
C_VIS_LR = '#90CAF9'
C_UPD_TT = '#EF5350'
C_VIS_TT = '#EF9A9A'
C_TR = '#FFA726'
C_ATTN = '#7E57C2'
C_FFN = '#26A69A'


def compute_T_common(inventory, BT, k_bwd=2.5):
    """T_common = base fwd + bwd, ALPINE-calibrated."""
    fwd_ops = sum(BT * 2 * l.M * l.N for l in inventory)
    T_fwd = TAU_ACIM * fwd_ops
    T_bwd = k_bwd * T_fwd
    return T_fwd, T_bwd, T_fwd + T_bwd


# =============================================================================
# Fig A: Total Step Time — T_common + DeltaT stacked
# =============================================================================

def plot_total_step_time(inventory):
    """Stacked bar: gray=T_common (identical), color=DeltaT (method-specific)."""
    BT = BATCH_SIZE * S_PAD_DEFAULT
    _, _, T_common = compute_T_common(inventory, BT)
    T_common_ms = T_common / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    x_labels = ['TT\n$\\gamma$=0', 'TT\n$\\gamma$=1'] + [f'r={r}' for r in RANKS]
    n = len(x_labels)
    x = np.arange(n)

    for col, target in enumerate(TARGETS):
        ax = axes[col]
        targeted = get_targeted_layers(inventory, target)

        deltas = []
        colors = []
        labels_short = []

        # TikiTaka gamma=0, gamma=1
        for gamma in [0, 1]:
            ec = compute_tikitaka_element_counts(targeted, gamma, BT)
            r = compute_model_a(ec, alpha_t=1.0)
            deltas.append(r.DeltaT_total_ns / 1e6)
            colors.append(C_UPD_TT if gamma == 0 else C_VIS_TT)

        # LR-TT gamma=0 for each rank
        for rank in RANKS:
            ec = compute_lrtt_element_counts(targeted, rank, 0, BT)
            r = compute_model_a(ec, alpha_t=1.0)
            deltas.append(r.DeltaT_total_ns / 1e6)
            colors.append(C_UPD_LR)

        # Stacked bars
        ax.bar(x, T_common_ms, 0.6, color=C_COMMON, edgecolor='#999', lw=0.3,
               label='$T_{common}$ (base fwd+bwd)', zorder=2)
        for i in range(n):
            ax.bar(x[i], deltas[i], 0.6, bottom=T_common_ms, color=colors[i],
                   edgecolor='#333', lw=0.3, zorder=3)

        # Percentage labels
        for i in range(n):
            total = T_common_ms + deltas[i]
            pct = deltas[i] / total * 100
            if pct >= 0.3:
                ax.text(x[i], total + 50, f'{pct:.1f}%',
                        ha='center', va='bottom', fontsize=7, fontweight='bold',
                        color='#333',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7.5)
        ax.axvline(1.5, color='#ccc', ls='-', lw=0.8, zorder=0)
        ax.set_title(f'target = {target}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('$T_{step}$ [ms]', fontsize=10)
        ax.grid(axis='y', alpha=0.1, zorder=0)

    handles = [
        mpatches.Patch(facecolor=C_COMMON, edgecolor='#999', label='$T_{common}$ (base fwd + bwd)'),
        mpatches.Patch(facecolor=C_UPD_TT, edgecolor='#333', label='$\\Delta T$ TikiTaka'),
        mpatches.Patch(facecolor=C_UPD_LR, edgecolor='#333', label='$\\Delta T$ LR-TT'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='#ccc')
    fig.suptitle('Total Training Step Time: $T_{step} = T_{common} + \\Delta T$\n'
                 '$T_{common}$ is identical across all methods (gray)  |  '
                 'Percentages = adapter overhead  |  '
                 f'$\\alpha_t$=1.0, BS={BATCH_SIZE}, S={S_PAD_DEFAULT}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    path = os.path.join(OUT, "LRTT_VS_TT_TOTAL_STEP_TIME.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Fig B: Cost Breakdown — stacked components across alpha_t
# =============================================================================

def plot_cost_breakdown(inventory):
    """Stacked bars showing proj/upd/vis/transfer breakdown at different alpha_t."""
    BT = BATCH_SIZE * S_PAD_DEFAULT
    target = "all"
    targeted = get_targeted_layers(inventory, target)
    rank = 8

    alpha_set = [1.0, 1.55, 4.15, 9.3]
    methods = ['TT $\\gamma$=0', 'TT $\\gamma$=1',
               'LR-TT $\\gamma$=0', 'LR-TT $\\gamma$=1']

    fig, axes = plt.subplots(1, len(alpha_set), figsize=(18, 6), sharey=True)

    for ai, alpha_t in enumerate(alpha_set):
        ax = axes[ai]
        x = np.arange(len(methods))

        # Compute each method
        results = []
        for method_label in methods:
            if 'TT' in method_label and 'LR' not in method_label:
                gamma = 1 if '$\\gamma$=1' in method_label else 0
                ec = compute_tikitaka_element_counts(targeted, gamma, BT)
                r = compute_model_a(ec, alpha_t)
            else:
                gamma = 1 if '$\\gamma$=1' in method_label else 0
                ec = compute_lrtt_element_counts(targeted, rank, gamma, BT)
                r = compute_model_a(ec, alpha_t)
            results.append(r)

        # Stacked bars
        proj_vals = [r.DeltaT_proj_ns / 1e6 for r in results]
        upd_vals = [r.DeltaT_upd_ns / 1e6 for r in results]
        vis_vals = [r.DeltaT_vis_ns / 1e6 for r in results]
        tr_vals = [r.DeltaT_tr_ns / 1e6 for r in results]

        bar_colors_proj = [C_COMMON, C_COMMON, C_PROJ, C_PROJ]
        bar_colors_upd = [C_UPD_TT, C_UPD_TT, C_UPD_LR, C_UPD_LR]
        bar_colors_vis = [C_VIS_TT, C_VIS_TT, C_VIS_LR, C_VIS_LR]

        w = 0.6
        for i in range(len(methods)):
            bottom = 0
            # Projection (LR-TT only)
            if proj_vals[i] > 0:
                ax.bar(x[i], proj_vals[i], w, bottom=bottom, color=C_PROJ,
                       edgecolor='#333', lw=0.3, zorder=3)
                bottom += proj_vals[i]
            # Update
            ax.bar(x[i], upd_vals[i], w, bottom=bottom, color=bar_colors_upd[i],
                   edgecolor='#333', lw=0.3, zorder=3)
            bottom += upd_vals[i]
            # Visible
            if vis_vals[i] > 0:
                ax.bar(x[i], vis_vals[i], w, bottom=bottom, color=bar_colors_vis[i],
                       edgecolor='#333', lw=0.3, zorder=3)
                bottom += vis_vals[i]
            # Transfer (tiny)
            if tr_vals[i] > 0.01:
                ax.bar(x[i], tr_vals[i], w, bottom=bottom, color=C_TR,
                       edgecolor='#333', lw=0.3, zorder=3)
                bottom += tr_vals[i]

            # Total label
            ax.text(x[i], bottom * 1.02, f'{bottom:.0f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.set_xticks(x)
        ax.set_xticklabels(['TT\n$\\gamma$=0', 'TT\n$\\gamma$=1',
                            'LRTT\n$\\gamma$=0', 'LRTT\n$\\gamma$=1'],
                           fontsize=7.5)
        ax.set_title(f'$\\alpha_t$ = {alpha_t}', fontsize=11, fontweight='bold')
        if ai == 0:
            ax.set_ylabel('$\\Delta T_{step}$ [ms]', fontsize=10)
        ax.grid(axis='y', alpha=0.1, zorder=0)
        ax.axvline(1.5, color='#ccc', ls='-', lw=0.8, zorder=0)

    handles = [
        mpatches.Patch(facecolor=C_PROJ, edgecolor='#333', label='Projection MVM (LR-TT)'),
        mpatches.Patch(facecolor=C_UPD_TT, edgecolor='#333', label='Pulsed Update (TT)'),
        mpatches.Patch(facecolor=C_UPD_LR, edgecolor='#333', label='Pulsed Update (LR-TT)'),
        mpatches.Patch(facecolor=C_VIS_TT, edgecolor='#333', label='Visible Fwd (TT)'),
        mpatches.Patch(facecolor=C_VIS_LR, edgecolor='#333', label='Visible Fwd (LR-TT)'),
        mpatches.Patch(facecolor=C_TR, edgecolor='#333', label='Transfer'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.03), frameon=True, edgecolor='#ccc')
    fig.suptitle(f'Adapter Cost Breakdown: TikiTaka vs LR-TT (r={rank}, target={target})\n'
                 'How cost distributes across projection, update, visible forward, and transfer\n'
                 f'BS={BATCH_SIZE}, S={S_PAD_DEFAULT}  |  '
                 'Numbers on top = total $\\Delta T$ [ms]',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.87])
    path = os.path.join(OUT, "LRTT_VS_TT_COST_BREAKDOWN.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Fig C: Per-layer group contribution (attention vs FFN)
# =============================================================================

def plot_per_layer_contribution(inventory):
    """Stacked bars: attention vs FFN contribution to DeltaT."""
    BT = BATCH_SIZE * S_PAD_DEFAULT
    targeted_attn = get_targeted_layers(inventory, 'attention')
    targeted_ffn = get_targeted_layers(inventory, 'ffn')

    x_labels = ['TT\n$\\gamma$=0', 'TT\n$\\gamma$=1'] + [f'r={r}' for r in RANKS]
    n = len(x_labels)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for panel, alpha_t in enumerate([1.0, 4.15]):
        ax = axes[panel]

        attn_vals = []
        ffn_vals = []

        # TikiTaka gamma=0, 1
        for gamma in [0, 1]:
            ec_a = compute_tikitaka_element_counts(targeted_attn, gamma, BT)
            ec_f = compute_tikitaka_element_counts(targeted_ffn, gamma, BT)
            r_a = compute_model_a(ec_a, alpha_t)
            r_f = compute_model_a(ec_f, alpha_t)
            attn_vals.append(r_a.DeltaT_total_ns / 1e6)
            ffn_vals.append(r_f.DeltaT_total_ns / 1e6)

        # LR-TT gamma=0
        for rank in RANKS:
            ec_a = compute_lrtt_element_counts(targeted_attn, rank, 0, BT)
            ec_f = compute_lrtt_element_counts(targeted_ffn, rank, 0, BT)
            r_a = compute_model_a(ec_a, alpha_t)
            r_f = compute_model_a(ec_f, alpha_t)
            attn_vals.append(r_a.DeltaT_total_ns / 1e6)
            ffn_vals.append(r_f.DeltaT_total_ns / 1e6)

        w = 0.6
        ax.bar(x, attn_vals, w, color=C_ATTN, edgecolor='#333', lw=0.3,
               label='Attention (Q,K,V,O)', zorder=3)
        ax.bar(x, ffn_vals, w, bottom=attn_vals, color=C_FFN, edgecolor='#333', lw=0.3,
               label='FFN (intermediate, output)', zorder=3)

        # Percentage labels
        for i in range(n):
            total = attn_vals[i] + ffn_vals[i]
            if total > 0:
                pct_ffn = ffn_vals[i] / total * 100
                ax.text(x[i], total * 1.02, f'{total:.0f}',
                        ha='center', va='bottom', fontsize=6.5, fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
                # Show attn/ffn split inside bar
                if attn_vals[i] > total * 0.08:
                    ax.text(x[i], attn_vals[i] / 2, f'{100-pct_ffn:.0f}%',
                            ha='center', va='center', fontsize=6, color='white', fontweight='bold')
                if ffn_vals[i] > total * 0.08:
                    ax.text(x[i], attn_vals[i] + ffn_vals[i] / 2, f'{pct_ffn:.0f}%',
                            ha='center', va='center', fontsize=6, color='white', fontweight='bold')

        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7.5)
        ax.axvline(1.5, color='#ccc', ls='-', lw=0.8, zorder=0)
        ax.set_title(f'$\\alpha_t$ = {alpha_t}', fontsize=11, fontweight='bold')
        if panel == 0:
            ax.set_ylabel('$\\Delta T_{step}$ [ms]  (log scale)', fontsize=10)
        ax.grid(axis='y', alpha=0.1, which='both', zorder=0)

    handles = [
        mpatches.Patch(facecolor=C_ATTN, edgecolor='#333', label='Attention layers (48 layers: Q,K,V,$W_O$)'),
        mpatches.Patch(facecolor=C_FFN, edgecolor='#333', label='FFN layers (24 layers: intermediate, output)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='#ccc')
    fig.suptitle('Per-Layer Group Contribution to $\\Delta T_{step}$: Attention vs FFN\n'
                 'LR-TT uses $\\gamma$=0  |  Numbers on top = total $\\Delta T$ [ms]  |  '
                 f'BS={BATCH_SIZE}, S={S_PAD_DEFAULT}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    path = os.path.join(OUT, "LRTT_VS_TT_PER_LAYER_CONTRIBUTION.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Generating main paper figures (non-algebraic)")
    print("=" * 60)
    inventory = build_layer_inventory()
    print(f"  Layers: {len(inventory)}")

    plot_total_step_time(inventory)
    plot_cost_breakdown(inventory)
    plot_per_layer_contribution(inventory)

    print("\nDone.")


if __name__ == "__main__":
    main()

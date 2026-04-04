#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restructured V2: Dual-Model AIMC Training Cost Study.

Fixes the v1 artifacts:
  1) Tile-granularity collapse: rank 4/8/16/32 looked identical → fixed by element-level model
  2) TikiTaka update=0 artifact: null θ_upd made TT look free → fixed by symbolic/ratio treatment
  3) Digital LoRA=0 artifact: null θ_gemm made DL look free → fixed by break-even analysis
  4) Projection over-penalty: ρ_proj=1 too pessimistic → parameterized sensitivity

Two complementary models:
  Model 1 (Latency Upper Bound): tile-oriented, conservative, with ρ_proj_lat sensitivity
  Model 2 (Utilization-Aware):   element-level, exposes true rank-proportional advantage
"""

import os, sys, csv, math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from extract_layer_inventory import (
    build_layer_inventory, get_targeted_layers, summarize_inventory,
    compute_adapter_tile_counts, LayerSpec, DEFAULT_TILE_SIZE,
)

OUT = os.path.join(SCRIPT_DIR, "restructured_v2")
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

# ─── Constants ──────────────────────────────────────────────────────
BS = 48
S_DEFAULT = 384
RANKS = [4, 8, 16, 32]
TARGETS = ["attention", "ffn", "all"]
LRTT_TE = 4
TT_TE = 1
NUM_READS = 1
RHO_PROJ_SWEEP = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


# ═══════════════════════════════════════════════════════════════════════
# Census Functions (both tile-level and element-level)
# ═══════════════════════════════════════════════════════════════════════

def census_layer(layer, rank, gamma, method, S=S_DEFAULT):
    """Compute both tile-level and element-level counts for one layer.

    Returns dict with tile_* and elem_* prefixed counts.
    """
    BT = BS * S
    M, N = layer.M, layer.N
    atc = compute_adapter_tile_counts(layer, rank) if rank > 0 else {}
    nA = atc.get('n_tiles_A', 0)
    nB = atc.get('n_tiles_B', 0)
    nC = layer.n_tiles
    nU = layer.n_tiles

    d = {
        'layer_idx': layer.layer_idx, 'sub_name': layer.sub_name,
        'group': layer.group, 'M': M, 'N': N, 'rank': rank,
        'gamma': gamma, 'method': method,
    }

    if method == 'lrtt':
        # ── Tile-level ──
        d['tile_proj']  = BT * (nB + nA)                          # B.fwd + A.bwd
        d['tile_upd']   = BT * (nA + nB)                          # A.upd + B.upd
        d['tile_tr_src']= rank * (nA + nB) * NUM_READS / LRTT_TE  # onehot reads
        d['tile_tr_dst']= rank * nC / LRTT_TE                     # pulsed writes to C
        d['tile_vis']   = BT * (nB + nA) if gamma == 1 else 0     # visible fwd

        # ── Element-level ──
        elem_A = M * rank
        elem_B = rank * N
        d['elem_proj']  = BT * (elem_B + elem_A)                  # MACs for projections
        d['elem_upd']   = BT * (elem_A + elem_B)                  # weight elements updated
        d['elem_tr_src']= rank * (M + N) / LRTT_TE                # source elements read per step
        d['elem_tr_dst']= rank * (M * N) / LRTT_TE                # destination elements written (rank-1 outer products)
        d['elem_vis']   = BT * (elem_A + elem_B) if gamma == 1 else 0
        d['elem_total_adapter'] = elem_A + elem_B                  # adapter footprint
        d['elem_total_base']    = M * N                            # base weight footprint

    elif method == 'tikitaka':
        # ── Tile-level ──
        d['tile_proj']  = 0
        d['tile_upd']   = BT * nU
        tr = layer.n_tile_rows
        d['tile_tr_src']= tr / TT_TE
        d['tile_tr_dst']= tr / TT_TE
        d['tile_vis']   = BT * nU if gamma == 1 else 0

        # ── Element-level ──
        d['elem_proj']  = 0
        d['elem_upd']   = BT * M * N                              # full-rank element updates
        d['elem_tr_src']= M / TT_TE                               # one column = M elements
        d['elem_tr_dst']= M / TT_TE                               # one column write
        d['elem_vis']   = BT * M * N if gamma == 1 else 0
        d['elem_total_adapter'] = M * N
        d['elem_total_base']    = M * N

    return d


def census_digital_lora(targeted, rank, S=S_DEFAULT):
    """Digital LoRA FLOP census."""
    BT = BS * S
    F_fwd = F_bwd = P = B_mem = 0
    for l in targeted:
        M, N = l.M, l.N
        F_fwd += 2 * BT * rank * (N + M)
        F_bwd += 2 * BT * rank * (2*M + 2*N)
        P += M*rank + rank*N
        B_mem += 2 * (M*rank + rank*N)
    return {'F_fwd': F_fwd, 'F_bwd': F_bwd, 'F_total': F_fwd + F_bwd,
            'P_opt': P, 'B_mem': B_mem, 'B_adam': B_mem * 4}


def aggregate_census(inventory, target, rank, gamma, method, S=S_DEFAULT):
    """Aggregate tile-level and element-level counts across targeted layers."""
    tgt = get_targeted_layers(inventory, target)
    totals = {}
    for l in tgt:
        d = census_layer(l, rank, gamma, method, S)
        for k, v in d.items():
            if k.startswith('tile_') or k.startswith('elem_'):
                totals[k] = totals.get(k, 0) + v
    totals['n_layers'] = len(tgt)
    return totals


# ═══════════════════════════════════════════════════════════════════════
# COST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def cost_latency(counts, rho_proj, rho_vis=1.0, tau=1.0):
    """Conservative latency upper-bound (tile-oriented)."""
    return (rho_proj * counts.get('tile_proj', 0)
            + counts.get('tile_upd', 0)
            + tau * (counts.get('tile_tr_src', 0) + counts.get('tile_tr_dst', 0))
            + rho_vis * counts.get('tile_vis', 0))


def cost_utilization(counts, rho_proj=1.0, rho_vis=1.0, tau_src=1.0, tau_dst=1.0):
    """Utilization-aware element-level cost."""
    return (rho_proj * counts.get('elem_proj', 0)
            + counts.get('elem_upd', 0)
            + tau_src * counts.get('elem_tr_src', 0)
            + tau_dst * counts.get('elem_tr_dst', 0)
            + rho_vis * counts.get('elem_vis', 0))


# ═══════════════════════════════════════════════════════════════════════
# STRUCTURAL COMPARISON V2
# ═══════════════════════════════════════════════════════════════════════

def write_structural(inventory):
    print("\n" + "="*70 + "\nSTRUCTURAL COMPARISON V2\n" + "="*70)

    L = []
    L.append("# Structural Comparison V2: Tile-Level vs Element-Level Census")
    L.append("")
    L.append("## Why Two Counting Models")
    L.append("")
    L.append("| Model | Unit | Rank-sensitive? | Purpose |")
    L.append("|-------|------|-----------------|---------|")
    L.append("| Tile-level | 512×512 tile events | NO (rank≤32 → ceil(r/512)=1) | Conservative latency upper bound |")
    L.append("| Element-level | Weight elements (M×r, r×N, M×N) | YES (∝ rank) | Utilization-aware energy/footprint |")
    L.append("")

    BT = BS * S_DEFAULT
    L.append(f"## Per-Step Counts (BS={BS}, S={S_DEFAULT}, BT={BT:,})")
    L.append("")

    # ── Table: Tile-level ──
    L.append("### Tile-Level Event Counts")
    L.append("")
    L.append("| Target | Method | γ | r | tile_proj | tile_upd | tile_tr | tile_vis |")
    L.append("|--------|--------|---|---|----------|---------|---------|---------|")

    for target in TARGETS:
        for gamma in [0, 1]:
            for r in RANKS:
                lr = aggregate_census(inventory, target, r, gamma, 'lrtt')
                L.append(f"| {target} | LR-TT | {gamma} | {r} | {lr['tile_proj']:,} | {lr['tile_upd']:,} | {lr['tile_tr_src']+lr['tile_tr_dst']:.0f} | {lr['tile_vis']:,} |")
            tt = aggregate_census(inventory, target, 0, gamma, 'tikitaka')
            L.append(f"| {target} | TikiTaka | {gamma} | full | 0 | {tt['tile_upd']:,} | {tt['tile_tr_src']+tt['tile_tr_dst']:.0f} | {tt['tile_vis']:,} |")
    L.append("")

    # ── Table: Element-level ──
    L.append("### Element-Level Counts (rank-sensitive)")
    L.append("")
    L.append("| Target | Method | γ | r | elem_proj | elem_upd | elem_tr_src | elem_vis | Adapter footprint | Base footprint | Ratio base/adapter |")
    L.append("|--------|--------|---|---|----------|---------|------------|---------|-------------------|----------------|-------------------|")

    rows_csv = []
    for target in TARGETS:
        for gamma in [0, 1]:
            for r in RANKS:
                lr = aggregate_census(inventory, target, r, gamma, 'lrtt')
                ratio = lr['elem_total_base'] / lr['elem_total_adapter'] if lr['elem_total_adapter'] > 0 else 0
                L.append(f"| {target} | LR-TT | {gamma} | {r} | {lr['elem_proj']:,} | {lr['elem_upd']:,} | {lr['elem_tr_src']:.0f} | {lr['elem_vis']:,} | {lr['elem_total_adapter']:,} | {lr['elem_total_base']:,} | {ratio:.1f}× |")
                rows_csv.append({**lr, 'target': target, 'gamma': gamma, 'rank': r, 'method': 'lrtt', 'ratio_base_over_adapter': ratio})
            tt = aggregate_census(inventory, target, 0, gamma, 'tikitaka')
            L.append(f"| {target} | TikiTaka | {gamma} | full | 0 | {tt['elem_upd']:,} | {tt['elem_tr_src']:.0f} | {tt['elem_vis']:,} | {tt['elem_total_adapter']:,} | {tt['elem_total_base']:,} | 1.0× |")
            rows_csv.append({**tt, 'target': target, 'gamma': gamma, 'rank': 0, 'method': 'tikitaka', 'ratio_base_over_adapter': 1.0})
    L.append("")

    L.append("### Key Observation: Rank Sensitivity")
    L.append("")
    L.append("| Target | r=4 adapter elems | r=32 adapter elems | TT full elems | Ratio TT/LR-TT r=4 | Ratio TT/LR-TT r=32 |")
    L.append("|--------|------------------|--------------------|---------------|--------------------|--------------------|")
    for target in TARGETS:
        lr4 = aggregate_census(inventory, target, 4, 0, 'lrtt')
        lr32 = aggregate_census(inventory, target, 32, 0, 'lrtt')
        tt = aggregate_census(inventory, target, 0, 0, 'tikitaka')
        r4 = tt['elem_total_base'] / lr4['elem_total_adapter']
        r32 = tt['elem_total_base'] / lr32['elem_total_adapter']
        L.append(f"| {target} | {lr4['elem_total_adapter']:,} | {lr32['elem_total_adapter']:,} | {tt['elem_total_base']:,} | {r4:.1f}× | {r32:.1f}× |")
    L.append("")
    L.append("**Element-level model correctly shows rank-proportional scaling.**")
    L.append("Tile-level model collapses all ranks to the same tile count.")

    # Digital LoRA census
    L.append("")
    L.append("### Digital LoRA FLOP Census")
    L.append("")
    L.append("| Target | r | F_fwd | F_bwd (4-GEMM) | F_total | P_opt | Bwd/Fwd |")
    L.append("|--------|---|-------|----------------|---------|-------|---------|")
    dl_data = {}
    for target in TARGETS:
        tgt = get_targeted_layers(inventory, target)
        for r in RANKS:
            c = census_digital_lora(tgt, r)
            dl_data[(target, r)] = c
            L.append(f"| {target} | {r} | {c['F_fwd']:,} | {c['F_bwd']:,} | {c['F_total']:,} | {c['P_opt']:,} | {c['F_bwd']/c['F_fwd']:.1f}× |")

    path = os.path.join(OUT, "STRUCTURAL_COMPARISON_V2.md")
    with open(path, 'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")

    csv_path = os.path.join(OUT, "utilization_aware_counts.csv")
    keys = [k for k in rows_csv[0].keys() if not k.startswith('tile_') and not k.startswith('elem_total')]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()), extrasaction='ignore')
        w.writeheader(); w.writerows(rows_csv)

    return dl_data


# ═══════════════════════════════════════════════════════════════════════
# LR-TT vs TikiTaka: UTILIZATION-AWARE (element-level)
# ═══════════════════════════════════════════════════════════════════════

def write_utilization_aware(inventory):
    print("\n" + "="*70 + "\nLR-TT vs TikiTaka: UTILIZATION-AWARE MODEL\n" + "="*70)

    L = []
    L.append("# LR-TT vs TikiTaka: Utilization-Aware (Element-Level) Comparison")
    L.append("")
    L.append("This model uses **weight element counts** instead of tile events,")
    L.append("correctly exposing LR-TT's rank-proportional advantage.")
    L.append("")
    L.append("```")
    L.append("C_elem = ρ_proj × elem_proj + elem_upd + τ_src × elem_tr_src + τ_dst × elem_tr_dst + ρ_vis × elem_vis")
    L.append("```")
    L.append("")

    # ── Update-only ratio (pure rank advantage) ──
    L.append("## 1. Update-Only Ratio (ρ_proj=0, τ=0: pure rank advantage)")
    L.append("")
    L.append("| Target | rank | γ | LR-TT elem_upd | TT elem_upd | **Ratio TT/LR** |")
    L.append("|--------|------|---|----------------|-------------|:----------------|")

    ratio_rows = []
    for target in TARGETS:
        for gamma in [0, 1]:
            tt = aggregate_census(inventory, target, 0, gamma, 'tikitaka')
            for r in RANKS:
                lr = aggregate_census(inventory, target, r, gamma, 'lrtt')
                ratio_upd = tt['elem_upd'] / lr['elem_upd'] if lr['elem_upd'] > 0 else 0
                L.append(f"| {target} | {r} | {gamma} | {lr['elem_upd']:,} | {tt['elem_upd']:,} | **{ratio_upd:.1f}×** |")
                ratio_rows.append({
                    'target': target, 'rank': r, 'gamma': gamma,
                    'lr_elem_upd': lr['elem_upd'], 'tt_elem_upd': tt['elem_upd'],
                    'ratio_upd_only': ratio_upd,
                })
    L.append("")
    L.append("**Rank-proportional scaling confirmed:** ratio = M×N / (M×r + r×N) ≈ min(M,N)/(2r)")
    L.append("")

    # ── Full ratio with ρ_proj sweep ──
    L.append("## 2. Full Ratio with Projection Cost (ρ_proj sweep)")
    L.append("")
    L.append("| Target | rank | γ | ρ=0 | ρ=0.01 | ρ=0.05 | ρ=0.1 | ρ=0.25 | ρ=0.5 | ρ=1.0 |")
    L.append("|--------|------|---|-----|--------|--------|-------|--------|-------|-------|")

    for target in TARGETS:
        for gamma in [0, 1]:
            tt = aggregate_census(inventory, target, 0, gamma, 'tikitaka')
            for r in RANKS:
                lr = aggregate_census(inventory, target, r, gamma, 'lrtt')
                parts = []
                for rp in RHO_PROJ_SWEEP:
                    c_lr = cost_utilization(lr, rho_proj=rp, rho_vis=1.0, tau_src=1.0, tau_dst=0.0)
                    c_tt = cost_utilization(tt, rho_proj=rp, rho_vis=1.0, tau_src=1.0, tau_dst=0.0)
                    ratio = c_tt / c_lr if c_lr > 0 else 0
                    parts.append(f"{ratio:.1f}×")

                    # save to csv
                    for rr in ratio_rows:
                        if rr['target']==target and rr['rank']==r and rr['gamma']==gamma:
                            rr[f'ratio_rho_{rp}'] = ratio

                L.append(f"| {target} | {r} | {gamma} | {' | '.join(parts)} |")
    L.append("")
    L.append("**LR-TT wins (ratio>1) across the entire ρ_proj range in the element-level model,**")
    L.append("because the rank-proportional update savings (~48× for r=8) far exceeds the projection overhead.")

    # ── Plot 1: Rank sensitivity (element-level) ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_i, target in enumerate(TARGETS):
        ax = axes[ax_i]
        tt_g0 = aggregate_census(inventory, target, 0, 0, 'tikitaka')
        tt_g1 = aggregate_census(inventory, target, 0, 1, 'tikitaka')

        for rp, color, ls in [(0, '#4CAF50', '-'), (0.1, '#8BC34A', '--'), (1.0, '#FF9800', ':')]:
            vals_g1 = []
            for r in RANKS:
                lr = aggregate_census(inventory, target, r, 1, 'lrtt')
                c_lr = cost_utilization(lr, rho_proj=rp, rho_vis=1.0)
                c_tt = cost_utilization(tt_g1, rho_proj=rp, rho_vis=1.0)
                vals_g1.append(c_tt / c_lr if c_lr > 0 else 1)
            ax.plot(RANKS, vals_g1, f'{ls}o', color=color, label=f'ρ_proj={rp}', linewidth=2, markersize=7)

        ax.axhline(1.0, color='gray', ls=':', lw=1)
        ax.set_xlabel('LR-TT rank', fontsize=11)
        if ax_i == 0: ax.set_ylabel('C_TT / C_LRTT (element-level)', fontsize=11)
        ax.set_title(f'target={target} (γ=1)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(RANKS)
        ax.set_yscale('log')

    fig.suptitle('Utilization-Aware Ratio: TT/LR-TT vs Rank (above 1 = LR-TT wins)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(PLOTS, "rank_sensitivity_utilization_v2.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")

    # ── Plot 2: ρ_proj sensitivity (element-level) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rhos = np.linspace(0, 2.0, 200)

    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]
        tt = aggregate_census(inventory, 'all', 0, gamma, 'tikitaka')
        for r, color in [(4, '#E91E63'), (8, '#F44336'), (16, '#FF9800'), (32, '#FFC107')]:
            lr = aggregate_census(inventory, 'all', r, gamma, 'lrtt')
            vals = []
            for rp in rhos:
                c_lr = cost_utilization(lr, rho_proj=rp, rho_vis=1.0)
                c_tt = cost_utilization(tt, rho_proj=rp, rho_vis=1.0)
                vals.append(c_tt / c_lr if c_lr > 0 else 1)
            ax.plot(rhos, vals, '-', color=color, label=f'r={r}', linewidth=2)

        ax.axhline(1.0, color='gray', ls=':', lw=1)
        ax.set_xlabel('ρ_proj (projection cost / update cost)', fontsize=11)
        ax.set_ylabel('C_TT / C_LRTT', fontsize=11)
        ax.set_title(f'γ={gamma}, target=all (element-level)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
        ax.set_ylim(0.8, 200)

    fig.suptitle('Utilization-Aware: ρ_proj Sensitivity (above 1 = LR-TT wins)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(PLOTS, "rho_proj_utilization_v2.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")

    # CSV
    csv_path = os.path.join(OUT, "lrtt_vs_tikitaka_utilization_ratio.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ratio_rows[0].keys())
        w.writeheader(); w.writerows(ratio_rows)

    path = os.path.join(OUT, "LRTT_VS_TIKITAKA_UTILIZATION_AWARE.md")
    with open(path, 'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════
# LR-TT vs TikiTaka: LATENCY UPPER BOUND (tile-level)
# ═══════════════════════════════════════════════════════════════════════

def write_latency_upper_bound(inventory):
    print("\n" + "="*70 + "\nLR-TT vs TikiTaka: LATENCY UPPER BOUND\n" + "="*70)

    L = []
    L.append("# LR-TT vs TikiTaka: Conservative Latency Upper Bound (Tile-Level)")
    L.append("")
    L.append("**Warning:** This model uses 512×512 tile-event counting.")
    L.append("It is a **conservative upper bound** that likely over-penalizes LR-TT")
    L.append("projection reads on sub-tile adapter arrays ([M,r] with r≪512).")
    L.append("")
    L.append("```")
    L.append("C_lat = ρ_proj_lat × tile_proj + tile_upd + τ × tile_tr + ρ_vis_lat × tile_vis")
    L.append("```")
    L.append("")
    L.append("ρ_proj_lat sweep: " + str(RHO_PROJ_SWEEP))
    L.append("")

    L.append("## Ratio TT/LR-TT (tile-level)")
    L.append("")
    hdr = "| Target | rank | γ | " + " | ".join(f"ρ={rp}" for rp in RHO_PROJ_SWEEP) + " |"
    L.append(hdr)
    L.append("|" + "---|"*(len(RHO_PROJ_SWEEP)+3))

    lat_rows = []
    for target in TARGETS:
        for gamma in [0, 1]:
            tt = aggregate_census(inventory, target, 0, gamma, 'tikitaka')
            for r in RANKS:
                lr = aggregate_census(inventory, target, r, gamma, 'lrtt')
                parts = []
                row_d = {'target': target, 'rank': r, 'gamma': gamma}
                for rp in RHO_PROJ_SWEEP:
                    c_lr = cost_latency(lr, rho_proj=rp, rho_vis=1.0)
                    c_tt = cost_latency(tt, rho_proj=rp, rho_vis=1.0)
                    ratio = c_tt / c_lr if c_lr > 0 else 0
                    parts.append(f"{ratio:.3f}")
                    row_d[f'ratio_rho_{rp}'] = ratio
                L.append(f"| {target} | {r} | {gamma} | " + " | ".join(parts) + " |")
                lat_rows.append(row_d)
    L.append("")
    L.append("**Note:** Ratio>1 = LR-TT wins. At ρ_proj_lat=1.0, the tile model")
    L.append("over-penalizes LR-TT because a [768,8] tile MVM is counted the same")
    L.append("as a [512,512] tile MVM. In reality, sub-tile projections are faster.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rhos = np.linspace(0, 1.5, 200)
    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]
        tt = aggregate_census(inventory, 'all', 0, gamma, 'tikitaka')
        for r, color in [(4, '#E91E63'), (8, '#F44336'), (16, '#FF9800'), (32, '#FFC107')]:
            lr = aggregate_census(inventory, 'all', r, gamma, 'lrtt')
            vals = [cost_latency(tt, rp) / cost_latency(lr, rp) if cost_latency(lr, rp) > 0 else 1
                    for rp in rhos]
            ax.plot(rhos, vals, '-', color=color, label=f'r={r}', linewidth=2)
        ax.axhline(1.0, color='gray', ls=':', lw=1)
        ax.fill_between(rhos, 1.0, max(max(vals),2), alpha=0.06, color='green')
        ax.fill_between(rhos, min(min(vals),0.3), 1.0, alpha=0.06, color='red')
        ax.set_xlabel('ρ_proj_lat', fontsize=11)
        ax.set_ylabel('C_TT / C_LRTT (tile-level)', fontsize=11)
        ax.set_title(f'γ={gamma}, target=all', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('Latency Upper Bound: TT/LR-TT vs ρ_proj_lat (above 1 = LR-TT wins)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(PLOTS, "rho_proj_latency_sensitivity_v2.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")

    csv_path = os.path.join(OUT, "lrtt_vs_tikitaka_latency_ratio.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=lat_rows[0].keys())
        w.writeheader(); w.writerows(lat_rows)

    path = os.path.join(OUT, "LRTT_VS_TIKITAKA_LATENCY_UPPER_BOUND.md")
    with open(path, 'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════
# Digital LoRA vs LR-TT: BREAK-EVEN V2
# ═══════════════════════════════════════════════════════════════════════

def write_break_even(inventory, dl_data):
    print("\n" + "="*70 + "\nDigital LoRA vs LR-TT: BREAK-EVEN V2\n" + "="*70)

    L = []
    L.append("# Digital LoRA vs LR-TT: Break-Even Analysis V2")
    L.append("")
    L.append("## Cost Formulas")
    L.append("")
    L.append("```")
    L.append("ΔT_dig  = θ_gemm × F_total + θ_opt × 5 × P_opt")
    L.append("")
    L.append("ΔT_lrtt = ρ_proj_lat × θ_mvm × N_proj_tile")
    L.append("        + θ_upd × N_upd_elem      ← element-level update")
    L.append("        + θ_tr  × N_tr")
    L.append("        + 1_{γ=1} × θ_mvm × N_vis_tile")
    L.append("```")
    L.append("")

    L.append("## Break-Even: θ_gemm* at which ΔT_dig = ΔT_lrtt")
    L.append("")
    L.append("Assuming θ_opt ≈ 5×θ_gemm and θ_upd/θ_mvm = α:")
    L.append("")
    L.append("```")
    L.append("θ_gemm* = θ_mvm × [ρ_proj × N_proj_tile + α × N_upd_elem + N_tr + 1_{γ=1} × N_vis_tile]")
    L.append("          / (F_total + 5 × P_opt)")
    L.append("```")
    L.append("")

    # Table: θ_gemm* at various (ρ_proj, α) for target=all, r=8, γ=1
    L.append("## θ_gemm* / θ_mvm Values (target=all, r=8, γ=1)")
    L.append("")
    L.append("| ρ_proj | α=0.5 | α=1.0 | α=2.0 | α=5.0 |")
    L.append("|--------|-------|-------|-------|-------|")

    be_rows = []
    dl = dl_data[('all', 8)]
    lr = aggregate_census(inventory, 'all', 8, 1, 'lrtt')
    F_eff = dl['F_total'] + 5 * dl['P_opt']

    for rp in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        parts = []
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            # N_proj uses tile, N_upd uses element
            num = (rp * lr['tile_proj']
                   + alpha * lr['elem_upd']
                   + lr['tile_tr_src'] + lr['tile_tr_dst']
                   + lr['tile_vis'])
            ratio = num / F_eff if F_eff > 0 else 0
            parts.append(f"{ratio:.6f}")
            be_rows.append({'rho_proj': rp, 'alpha_upd_mvm': alpha,
                           'theta_gemm_star_over_theta_mvm': ratio,
                           'at_256ns_gflops': 1e9/(ratio*256)/1e9 if ratio > 0 else float('inf')})
        L.append(f"| {rp} | " + " | ".join(parts) + " |")

    L.append("")
    L.append("### Reading the Table")
    L.append("")
    L.append("- If θ_mvm = 256ns and ratio = 0.05 → θ_gemm* = 12.8 ns/FLOP → 78 MFLOPS break-even")
    L.append("- Any digital core faster than this → Digital LoRA wins")
    L.append("- Any digital core slower → LR-TT wins")
    L.append("")
    L.append("**α = θ_upd/θ_mvm is the update-to-read cost ratio.**")
    L.append("Higher α means pulsed updates are expensive relative to reads,")
    L.append("which increases LR-TT's total cost and makes Digital LoRA more competitive.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: θ_gemm* vs ρ_proj for various α
    ax = axes[0]
    for alpha, color in [(0.5, '#4CAF50'), (1.0, '#FF9800'), (2.0, '#F44336'), (5.0, '#9C27B0')]:
        vals = []
        for rp in np.linspace(0, 1, 100):
            num = rp * lr['tile_proj'] + alpha * lr['elem_upd'] + lr['tile_tr_src'] + lr['tile_tr_dst'] + lr['tile_vis']
            vals.append(num / F_eff * 256)  # ns/FLOP at θ_mvm=256
        ax.plot(np.linspace(0, 1, 100), vals, '-', color=color, label=f'α={alpha}', lw=2)
    ax.set_xlabel('ρ_proj_lat', fontsize=11)
    ax.set_ylabel('θ_gemm* (ns/FLOP) at θ_mvm=256ns', fontsize=11)
    ax.set_title('Break-even: DL wins below, LR-TT wins above', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Right: 2D region
    ax = axes[1]
    alphas = np.linspace(0.1, 5.0, 100)
    rhos = np.linspace(0, 1.0, 100)
    A, R = np.meshgrid(alphas, rhos)
    Z = np.zeros_like(A)
    for i in range(len(rhos)):
        for j in range(len(alphas)):
            num = rhos[i]*lr['tile_proj'] + alphas[j]*lr['elem_upd'] + lr['tile_tr_src'] + lr['tile_tr_dst'] + lr['tile_vis']
            Z[i,j] = num / F_eff * 256
    cs = ax.contourf(A, R, Z, levels=20, cmap='RdYlGn_r')
    plt.colorbar(cs, ax=ax, label='θ_gemm* (ns/FLOP)')
    ax.set_xlabel('α = θ_upd / θ_mvm', fontsize=11)
    ax.set_ylabel('ρ_proj_lat', fontsize=11)
    ax.set_title('Break-even surface (r=8, γ=1, target=all)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(PLOTS, "dig_vs_lrtt_break_even_region_v2.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")

    csv_path = os.path.join(OUT, "break_even_dig_vs_lrtt_v2.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=be_rows[0].keys())
        w.writeheader(); w.writerows(be_rows)

    path = os.path.join(OUT, "DIGITAL_LORA_VS_LRTT_BREAK_EVEN_V2.md")
    with open(path, 'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════
# COMBINED REGION PLOT (LR-TT vs TikiTaka)
# ═══════════════════════════════════════════════════════════════════════

def write_combined_region(inventory):
    print("\n" + "="*70 + "\nCOMBINED REGION PLOT\n" + "="*70)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for row, (model_name, cost_fn) in enumerate([
        ('Tile-level (latency)', cost_latency),
        ('Element-level (utilization)', cost_utilization),
    ]):
        for col, r in enumerate(RANKS):
            ax = axes[row, col]
            rhos = np.linspace(0, 1.5, 150)

            tt_g0 = aggregate_census(inventory, 'all', 0, 0, 'tikitaka')
            tt_g1 = aggregate_census(inventory, 'all', 0, 1, 'tikitaka')
            lr_g0 = aggregate_census(inventory, 'all', r, 0, 'lrtt')
            lr_g1 = aggregate_census(inventory, 'all', r, 1, 'lrtt')

            for gamma, tt, lr, color, ls in [(0, tt_g0, lr_g0, '#4CAF50', '--'), (1, tt_g1, lr_g1, '#9C27B0', '-')]:
                vals = []
                for rp in rhos:
                    c_lr = cost_fn(lr, rho_proj=rp, rho_vis=1.0)
                    c_tt = cost_fn(tt, rho_proj=rp, rho_vis=1.0)
                    vals.append(c_tt / c_lr if c_lr > 0 else 1)
                ax.plot(rhos, vals, ls, color=color, label=f'γ={gamma}', linewidth=2)

            ax.axhline(1.0, color='gray', ls=':', lw=1)
            ax.set_title(f'r={r}', fontsize=11, fontweight='bold')
            if col == 0: ax.set_ylabel(f'{model_name}\nC_TT / C_LRTT', fontsize=10)
            if row == 1: ax.set_xlabel('ρ_proj', fontsize=10)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            if row == 1:  # element-level: use log scale
                ax.set_yscale('log'); ax.set_ylim(0.5, 200)
            else:
                ax.set_ylim(0.3, 2.5)

    fig.suptitle('LR-TT vs TikiTaka: Tile-Level (top) vs Element-Level (bottom), target=all',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(PLOTS, "lrtt_vs_tikitaka_region_plot_v2.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════
# INVALIDATION NOTE
# ═══════════════════════════════════════════════════════════════════════

def write_invalidation_note():
    L = []
    L.append("# Cost Model Invalidation Note")
    L.append("")
    L.append("## Previous Absolute Ranking (INVALID)")
    L.append("")
    L.append("```")
    L.append("INVALID: LR-TT (6,229ms) >> TikiTaka (4,077ms) >> Digital LoRA (3,171ms)")
    L.append("```")
    L.append("")
    L.append("## Why It Was Invalid")
    L.append("")
    L.append("| Method | What was reflected | What was missing (null→0) |")
    L.append("|--------|-------------------|--------------------------|")
    L.append("| Digital LoRA | Nothing (DeltaT=0) | θ_gemm (146B FLOPs), θ_opt (1.2M params) |")
    L.append("| LR-TT | Projection MVM (t_tile_ns) | θ_upd (pulsed update on A,B) |")
    L.append("| TikiTaka | γ=1 visible MVM only | θ_upd (full-rank U update: 8.8M tile events) |")
    L.append("")
    L.append("**Only LR-TT had its main adapter cost instantiated.** The others were")
    L.append("artificially zero, making LR-TT appear uniquely expensive.")
    L.append("")
    L.append("## Additional V1 Artifact: Flat Rank Sensitivity")
    L.append("")
    L.append("The V1 tile-level model used ceil(rank/512) for adapter tile counts.")
    L.append("Since rank ≤ 32 < 512, all ranks mapped to 1 tile → no rank differentiation.")
    L.append("This destroyed LR-TT's core value proposition (rank-proportional cost).")
    L.append("")
    L.append("## How V2 Fixes These Artifacts")
    L.append("")
    L.append("1. **Dual-model approach**: tile-level (latency UB) + element-level (utilization)")
    L.append("2. **Element-level model**: counts actual weight elements (M×r+r×N vs M×N)")
    L.append("   → rank sensitivity correctly shows ~48× advantage at r=8")
    L.append("3. **Symbolic primitives**: no null→0; ratios remain symbolic until mapped")
    L.append("4. **ρ_proj separation**: projection cost parameterized, not assumed equal to full tile")
    L.append("5. **Break-even analysis**: no absolute winner claimed without primitive mapping")
    L.append("")
    L.append("## Remaining Primitive Mappings Needed")
    L.append("")
    L.append("| Primitive | Symbol | For |")
    L.append("|-----------|--------|-----|")
    L.append("| Pulsed update per element | θ_upd | LR-TT + TikiTaka absolute cost |")
    L.append("| Digital GEMM throughput | θ_gemm | Digital LoRA absolute cost |")
    L.append("| Projection read relative cost | ρ_proj | LR-TT latency calibration |")
    L.append("| Tile energy | e_tile | All energy metrics |")

    path = os.path.join(OUT, "COST_MODEL_INVALIDATION_NOTE.md")
    with open(path, 'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("RESTRUCTURED V2: Dual-Model AIMC Training Cost Study")
    print("="*70)

    inventory = build_layer_inventory()
    print(f"Inventory: {len(inventory)} layers, {sum(l.n_tiles for l in inventory)} tiles")

    dl_data = write_structural(inventory)
    write_utilization_aware(inventory)
    write_latency_upper_bound(inventory)
    write_break_even(inventory, dl_data)
    write_combined_region(inventory)
    write_invalidation_note()

    print("\n" + "="*70)
    print("ALL V2 OUTPUTS:")
    for root, dirs, files in os.walk(OUT):
        for f in sorted(files):
            print(f"  {os.path.join(root, f)}")
    print("="*70)


if __name__ == "__main__":
    main()

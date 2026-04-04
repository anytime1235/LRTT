#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 AIMC Training Cost Study — Dual-Model with DCIM-aware Digital LoRA.

Fixes all prior artifacts:
  - Constant 2.0× DL/LR ratio (meaningless cross-unit comparison) → removed
  - Tile-granularity rank collapse → element-level model
  - TikiTaka update=0 artifact → symbolic/ratio treatment
  - Digital LoRA=0 artifact → DCIM event counting + break-even
  - Projection over-penalty → ρ_proj parameterized

Two models:
  Model 1 (Latency UB):    tile-oriented, conservative, ρ_proj_lat sensitivity
  Model 2 (Utilization):   element-level, rank-proportional, energy/footprint proxy

Digital LoRA: Dual framing
  Option A: Traditional digital (θ_gemm per FLOP)
  Option B: DCIM CIM (θ_dcim per tile-MVM event) — same unit family as ACIM
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

OUT = os.path.join(SCRIPT_DIR, "v3")
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

BS = 48; S = 384; BT = BS * S
RANKS = [4, 8, 16, 32]
TARGETS = ["attention", "ffn", "all"]
LRTT_TE = 4; TT_TE = 1; NUM_READS = 1
RHO_SWEEP = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


# ═══════════════════════════════════════════════════════════════
# Census: tile-level + element-level + DCIM events
# ═══════════════════════════════════════════════════════════════

def census_all(inventory, target, rank, gamma):
    """Compute all event counts for LR-TT, TikiTaka, and Digital LoRA."""
    tgt = get_targeted_layers(inventory, target)
    lr_tile = lr_elem = tt_tile = tt_elem = dl_dcim = 0
    lr_t = {'proj':0,'upd':0,'vis':0,'tr_src':0,'tr_dst':0}
    lr_e = {'proj':0,'upd':0,'vis':0,'tr_src':0,'tr_dst':0,'footprint':0}
    tt_t = {'upd':0,'vis':0,'tr_src':0,'tr_dst':0}
    tt_e = {'upd':0,'vis':0,'tr_src':0,'tr_dst':0,'footprint':0}
    dl = {'fwd_mvm':0,'bwd_mvm':0,'opt_write':0,'F_fwd':0,'F_bwd':0,'P_opt':0}

    for l in tgt:
        M, N = l.M, l.N
        atc = compute_adapter_tile_counts(l, rank)
        nA, nB, nC = atc['n_tiles_A'], atc['n_tiles_B'], l.n_tiles

        # ── LR-TT tile-level ──
        lr_t['proj'] += BT * (nB + nA)
        lr_t['upd']  += BT * (nA + nB)
        lr_t['tr_src'] += rank * (nA + nB) * NUM_READS / LRTT_TE
        lr_t['tr_dst'] += rank * nC / LRTT_TE
        if gamma == 1:
            lr_t['vis'] += BT * (nB + nA)

        # ── LR-TT element-level ──
        eA, eB = M*rank, rank*N
        lr_e['proj'] += BT * (eB + eA)
        lr_e['upd']  += BT * (eA + eB)
        lr_e['tr_src'] += rank * (M + N) / LRTT_TE
        lr_e['tr_dst'] += rank * M * N / LRTT_TE  # rank-1 outer products to C
        lr_e['footprint'] += eA + eB
        if gamma == 1:
            lr_e['vis'] += BT * (eA + eB)

        # ── TikiTaka tile-level ──
        tt_t['upd'] += BT * l.n_tiles
        tt_t['tr_src'] += l.n_tile_rows / TT_TE
        tt_t['tr_dst'] += l.n_tile_rows / TT_TE
        if gamma == 1:
            tt_t['vis'] += BT * l.n_tiles

        # ── TikiTaka element-level ──
        tt_e['upd'] += BT * M * N
        tt_e['tr_src'] += M / TT_TE  # one column read
        tt_e['tr_dst'] += M / TT_TE  # one column write
        tt_e['footprint'] += M * N
        if gamma == 1:
            tt_e['vis'] += BT * M * N

        # ── Digital LoRA: DCIM MVM event counting ──
        # Forward: 2 DCIM-MVMs (X@B^T, result@A^T) on rank-r tiles
        dl['fwd_mvm'] += 2  # 2 tile-array MVMs per layer
        # Backward: 4 DCIM-MVMs (dA, dZ, dB, dX_LoRA)
        dl['bwd_mvm'] += 4
        # Optimizer: write-back updated A,B weights
        dl['opt_write'] += 1  # 1 write event per layer
        # FLOP census (for traditional digital option)
        dl['F_fwd'] += 2 * BT * rank * (N + M)
        dl['F_bwd'] += 2 * BT * rank * (2*M + 2*N)
        dl['P_opt'] += M*rank + rank*N

    dl['F_total'] = dl['F_fwd'] + dl['F_bwd']
    dl['total_mvm'] = dl['fwd_mvm'] + dl['bwd_mvm']  # per step
    dl['footprint'] = dl['P_opt']  # same adapter footprint as LR-TT

    return lr_t, lr_e, tt_t, tt_e, dl


# ═══════════════════════════════════════════════════════════════
# Structural Comparison
# ═══════════════════════════════════════════════════════════════

def write_structural(inventory):
    print("\n" + "="*70 + "\nSTRUCTURAL COMPARISON V3\n" + "="*70)
    L = []
    L.append("# V3 Structural Comparison")
    L.append("")
    inv = summarize_inventory(inventory)
    L.append(f"BERT-base: {inv['total_layers']} encoder linears, {inv['total_tiles']} base tiles (512×512)")
    L.append(f"Attention: {inv['attention_layers']} layers ({inv['attention_tiles']} tiles) | FFN: {inv['ffn_layers']} layers ({inv['ffn_tiles']} tiles)")
    L.append("")

    # Tile vs Element comparison table
    L.append("## Tile-Level vs Element-Level Counts (target=all, γ=1)")
    L.append("")
    L.append("| Model | rank | LR-TT proj | LR-TT upd | LR-TT vis | TT upd | TT vis |")
    L.append("|-------|------|-----------|-----------|----------|--------|--------|")

    all_data = {}
    for r in RANKS:
        lr_t, lr_e, tt_t, tt_e, dl = census_all(inventory, 'all', r, 1)
        all_data[('all', r)] = (lr_t, lr_e, tt_t, tt_e, dl)
        L.append(f"| Tile | {r} | {lr_t['proj']:,} | {lr_t['upd']:,} | {lr_t['vis']:,} | {tt_t['upd']:,} | {tt_t['vis']:,} |")
        L.append(f"| Elem | {r} | {lr_e['proj']:,} | {lr_e['upd']:,} | {lr_e['vis']:,} | {tt_e['upd']:,} | {tt_e['vis']:,} |")
    L.append("")
    L.append("**Tile-level:** rank 4/8/16/32 collapse to same count (ceil(r/512)=1). Artifact.")
    L.append("**Element-level:** rank-proportional. LR-TT scales as r×(M+N), TikiTaka as M×N.")
    L.append("")

    # Digital LoRA: dual framing
    L.append("## Digital LoRA Census (target=all, r=8)")
    L.append("")
    _, _, _, _, dl8 = census_all(inventory, 'all', 8, 1)
    L.append(f"- DCIM framing: {dl8['total_mvm']} DCIM-MVM events/step ({dl8['fwd_mvm']} fwd + {dl8['bwd_mvm']} bwd) + {dl8['opt_write']} opt writes")
    L.append(f"- FLOP framing: {dl8['F_total']:,} FLOPs/step ({dl8['F_fwd']:,} fwd + {dl8['F_bwd']:,} bwd)")
    L.append(f"- Adapter footprint: {dl8['P_opt']:,} params (same as LR-TT r=8)")
    L.append("")
    L.append("**Note:** DCIM framing counts per-layer tile-array operations.")
    L.append("FLOP framing counts individual multiply-adds. These are different units.")

    path = os.path.join(OUT, "STRUCTURAL_COMPARISON_V3.md")
    with open(path, 'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")
    return all_data


# ═══════════════════════════════════════════════════════════════
# LR-TT vs TikiTaka: Utilization-Aware (element-level)
# ═══════════════════════════════════════════════════════════════

def write_lrtt_vs_tt_util(inventory):
    print("\n" + "="*70 + "\nLR-TT vs TikiTaka: UTILIZATION-AWARE\n" + "="*70)
    L = []
    L.append("# LR-TT vs TikiTaka: Utilization-Aware (Element-Level)")
    L.append("")
    L.append("Same ACIM primitive family → element-level ratio is valid.")
    L.append("")
    L.append("```")
    L.append("C_elem = ρ_proj × elem_proj + elem_upd + τ_src × elem_tr_src + ρ_vis × elem_vis")
    L.append("```")
    L.append("")

    # Update-only ratio
    L.append("## Update-Only Ratio (ρ_proj=0: pure rank advantage)")
    L.append("")
    L.append("| Target | rank | LR-TT elem_upd | TT elem_upd | Ratio TT/LR |")
    L.append("|--------|------|----------------|-------------|-------------|")

    csv_rows = []
    for target in TARGETS:
        for r in RANKS:
            lr_t, lr_e, tt_t, tt_e, _ = census_all(inventory, target, r, 0)
            ratio = tt_e['upd'] / lr_e['upd'] if lr_e['upd'] > 0 else 0
            L.append(f"| {target} | {r} | {lr_e['upd']:,} | {tt_e['upd']:,} | **{ratio:.1f}×** |")
            csv_rows.append({'target':target,'rank':r,'gamma':0,
                            'lr_elem_upd':lr_e['upd'],'tt_elem_upd':tt_e['upd'],
                            'ratio_upd_only':ratio})
    L.append("")

    # Full ratio with ρ_proj sweep
    L.append("## Full Ratio (γ=1, ρ_vis=1)")
    L.append("")
    hdr = "| Target | rank | " + " | ".join(f"ρ={rp}" for rp in RHO_SWEEP) + " |"
    L.append(hdr)
    L.append("|" + "---|"*(len(RHO_SWEEP)+2))

    for target in TARGETS:
        tt_t0, tt_e0, _, _, _ = census_all(inventory, target, 0, 1)  # dummy rank for TT
        # Need TT census - rank doesn't matter for TT
        _, _, tt_t, tt_e, _ = census_all(inventory, target, 4, 1)
        for r in RANKS:
            lr_t, lr_e, _, _, _ = census_all(inventory, target, r, 1)
            parts = []
            for rp in RHO_SWEEP:
                c_lr = rp*lr_e['proj'] + lr_e['upd'] + lr_e['vis']
                c_tt = tt_e['upd'] + tt_e['vis']  # TT has no proj
                ratio = c_tt / c_lr if c_lr > 0 else 0
                parts.append(f"{ratio:.1f}×")
                # update csv
                for row in csv_rows:
                    if row['target']==target and row['rank']==r and row['gamma']==0:
                        row[f'ratio_g1_rho_{rp}'] = c_tt / (rp*lr_e['proj']+lr_e['upd']+lr_e['vis']) if (rp*lr_e['proj']+lr_e['upd']+lr_e['vis'])>0 else 0
            L.append(f"| {target} | {r} | " + " | ".join(parts) + " |")
    L.append("")
    L.append("**LR-TT wins across entire ρ_proj range** in element-level model.")

    # ── Plot: rank sensitivity (γ=0 and γ=1, 2 rows × 3 targets) ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, gamma in enumerate([0, 1]):
        for col, target in enumerate(TARGETS):
            ax = axes[row, col]
            _, _, _, tt_e, _ = census_all(inventory, target, 4, gamma)
            c_tt = tt_e['upd'] + tt_e['vis']
            for rp, color, ls in [(0,'#4CAF50','-'),(0.1,'#8BC34A','--'),(1.0,'#FF9800',':')]:
                vals = []
                for r in RANKS:
                    lr_t, lr_e, _, _, _ = census_all(inventory, target, r, gamma)
                    c_lr = rp*lr_e['proj'] + lr_e['upd'] + lr_e['vis']
                    vals.append(c_tt/c_lr if c_lr>0 else 1)
                ax.plot(RANKS, vals, f'{ls}o', color=color, label=f'ρ={rp}', linewidth=2, markersize=7)
            ax.axhline(1, color='gray', ls=':', lw=1)
            ax.set_xlabel('LR-TT rank'); ax.set_xticks(RANKS)
            if col==0: ax.set_ylabel(f'C_TT / C_LRTT (γ={gamma})')
            ax.set_title(f'{target} (γ={gamma})', fontweight='bold')
            ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_yscale('log')
    fig.suptitle('Element-Level: TT/LR-TT vs Rank — γ=0 (top) vs γ=1 (bottom)\nabove 1 = LR-TT wins',
                 fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0,0,1,0.92])
    p = os.path.join(PLOTS,"rank_sensitivity_utilization_v2.png")
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  → {p}")

    # CSV
    cp = os.path.join(OUT,"lrtt_vs_tikitaka_utilization_ratio.csv")
    with open(cp,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)

    path = os.path.join(OUT, "LRTT_VS_TIKITAKA_UTILIZATION_AWARE.md")
    with open(path,'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════
# LR-TT vs TikiTaka: Latency Upper Bound (tile-level)
# ═══════════════════════════════════════════════════════════════

def write_lrtt_vs_tt_lat(inventory):
    print("\n" + "="*70 + "\nLR-TT vs TikiTaka: LATENCY UPPER BOUND\n" + "="*70)
    L = []
    L.append("# LR-TT vs TikiTaka: Conservative Latency Upper Bound")
    L.append("")
    L.append("**Warning:** Tile-level model. Rank insensitive (ceil(r/512)=1). Pessimistic for sub-tile projections.")
    L.append("")

    csv_rows = []
    L.append("| Target | rank | γ | " + " | ".join(f"ρ={rp}" for rp in RHO_SWEEP) + " |")
    L.append("|" + "---|"*(len(RHO_SWEEP)+3))

    for target in TARGETS:
        for gamma in [0, 1]:
            _, _, tt_t, _, _ = census_all(inventory, target, 4, gamma)
            c_tt = tt_t['upd'] + tt_t['vis'] + tt_t['tr_src'] + tt_t['tr_dst']
            for r in RANKS:
                lr_t, _, _, _, _ = census_all(inventory, target, r, gamma)
                parts = []; row = {'target':target,'rank':r,'gamma':gamma}
                for rp in RHO_SWEEP:
                    c_lr = rp*lr_t['proj'] + lr_t['upd'] + rp*lr_t['vis'] + lr_t['tr_src']+lr_t['tr_dst']
                    ratio = c_tt/c_lr if c_lr>0 else 0
                    parts.append(f"{ratio:.3f}")
                    row[f'ratio_rho_{rp}'] = ratio
                L.append(f"| {target} | {r} | {gamma} | " + " | ".join(parts) + " |")
                csv_rows.append(row)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rhos = np.linspace(0, 1.5, 200)
    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]
        _, _, tt_t, _, _ = census_all(inventory, 'all', 4, gamma)
        c_tt = tt_t['upd'] + tt_t['vis'] + tt_t['tr_src'] + tt_t['tr_dst']
        for r, color in [(4,'#E91E63'),(8,'#F44336'),(16,'#FF9800'),(32,'#FFC107')]:
            lr_t, _, _, _, _ = census_all(inventory, 'all', r, gamma)
            vals = []
            for rp in rhos:
                c_lr = rp*lr_t['proj'] + lr_t['upd'] + rp*lr_t['vis'] + lr_t['tr_src']+lr_t['tr_dst']
                vals.append(c_tt/c_lr if c_lr>0 else 1)
            ax.plot(rhos, vals, '-', color=color, label=f'r={r}', linewidth=2)
        ax.axhline(1, color='gray', ls=':', lw=1)
        ax.fill_between(rhos, 1, 3, alpha=0.05, color='green')
        ax.fill_between(rhos, 0.3, 1, alpha=0.05, color='red')
        ax.set_xlabel('ρ_proj_lat'); ax.set_ylabel('C_TT/C_LRTT')
        ax.set_title(f'γ={gamma}, target=all', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0.3, 2.5)
    fig.suptitle('Latency UB: TT/LR-TT vs ρ_proj (above 1 = LR-TT wins)', fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.93])
    p = os.path.join(PLOTS,"rho_proj_latency_sensitivity_v2.png")
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  → {p}")

    cp = os.path.join(OUT,"lrtt_vs_tikitaka_latency_ratio.csv")
    with open(cp,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)
    path = os.path.join(OUT, "LRTT_VS_TIKITAKA_LATENCY_UPPER_BOUND.md")
    with open(path,'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════
# Digital LoRA vs LR-TT: DCIM-aware Break-Even
# ═══════════════════════════════════════════════════════════════

def write_dl_vs_lrtt(inventory):
    print("\n" + "="*70 + "\nDigital LoRA vs LR-TT: DCIM-AWARE BREAK-EVEN\n" + "="*70)
    L = []
    L.append("# Digital LoRA vs LR-TT: Break-Even Analysis V3")
    L.append("")
    L.append("## Two Framings for Digital LoRA")
    L.append("")
    L.append("| Framing | Cost unit | Digital LoRA events/step | LR-TT events/step | Comparable? |")
    L.append("|---------|-----------|:-----------------------:|:------------------:|:-----------:|")

    _, lr_e, _, _, dl = census_all(inventory, 'all', 8, 1)
    L.append(f"| **DCIM** (CIM-based) | tile-MVM event | {dl['total_mvm']} DCIM-MVMs + {dl['opt_write']} writes | 2~3 ACIM-MVMs + updates | ✓ (same tile-event unit) |")
    L.append(f"| Traditional digital | FLOP | {dl['F_total']:,} FLOPs | N/A (different unit) | ✗ (cross-unit) |")
    L.append("")

    L.append("## DCIM Framing: Per-Layer Event Comparison")
    L.append("")
    L.append("| Operation | Digital LoRA (DCIM) | LR-TT (ACIM) |")
    L.append("|-----------|:-------------------:|:-------------:|")
    L.append("| Forward MVM | 2 DCIM-MVM (X@B^T, Z@A^T) | — |")
    L.append("| Visible (γ=1) | — | 2 ACIM-MVM (B.fwd, A.fwd) |")
    L.append("| Projection | — | 2 ACIM-MVM (B.fwd, A.bwd) |")
    L.append("| Backward | 4 DCIM-MVM (dA,dZ,dB,dX) | — |")
    L.append("| Update | 1 DCIM-write (Adam → A,B) | 1 ACIM-pulsed (A.upd + B.upd) |")
    L.append(f"| **Total per layer** | **6 DCIM-MVM + 1 write** | **2~4 ACIM-MVM + 1 update** |")
    L.append("")

    n_layers = len(get_targeted_layers(inventory, 'all'))
    dl_events = dl['total_mvm']  # per step, all layers
    L.append(f"Across {n_layers} layers: DL = {dl_events} DCIM-MVMs/step, LR-TT = variable ACIM events")
    L.append("")

    L.append("## Break-Even: t_dcim / t_acim Ratio")
    L.append("")
    L.append("When DCIM and ACIM tiles have the same size (512×512, 8-bit I/O),")
    L.append("the break-even depends on the per-tile MVM speed ratio:")
    L.append("")
    L.append("```")
    L.append("ΔT_dl  = t_dcim × N_dcim_mvm + t_dcim_write × N_dcim_write")
    L.append("ΔT_lrtt = t_acim × (ρ_proj × N_proj + N_vis) + t_acim_upd × N_upd + t_acim_tr × N_tr")
    L.append("")
    L.append("Break-even: t_dcim / t_acim = (ρ × N_proj + N_vis + α × N_upd) / N_dcim_mvm")
    L.append("  where α = t_acim_upd / t_acim")
    L.append("```")
    L.append("")

    # Break-even table (tile-level for DCIM comparison)
    L.append("## γ=0 System Advantage: LR-TT Can Hide Adapter from Forward")
    L.append("")
    L.append("Digital LoRA is **always γ=1** — adapter A,B appear in every forward pass.")
    L.append("LR-TT can operate at **γ=0** — forward uses only base C, adapter A,B are hidden.")
    L.append("")
    L.append("| | Digital LoRA (γ=1 only) | LR-TT γ=1 | LR-TT γ=0 |")
    L.append("|---|:---:|:---:|:---:|")

    lr_t1, lr_e1, _, _, dl1 = census_all(inventory, 'all', 8, 1)
    lr_t0, lr_e0, _, _, dl0 = census_all(inventory, 'all', 8, 0)
    L.append(f"| Forward MVM events | {dl1['fwd_mvm']} DCIM | 2 ACIM/layer (vis) | **0** (C-path only) |")
    L.append(f"| Backward MVM events | {dl1['bwd_mvm']} DCIM | — | — |")
    L.append(f"| Projection events | — | {lr_t1['proj']:,} | {lr_t0['proj']:,} |")
    L.append(f"| Visible path events | included in fwd | {lr_t1['vis']:,} | **0** |")
    L.append(f"| Update events | {dl1['opt_write']} writes | {lr_t1['upd']:,} | {lr_t0['upd']:,} |")
    L.append("")
    L.append("**LR-TT γ=0 eliminates all visible-path overhead** while still training A,B.")
    L.append("Digital LoRA cannot do this — removing A,B from forward means no gradient for A,B.")
    L.append("This is a **structural system-level advantage** of the analog hidden-carry architecture.")
    L.append("")

    # Break-even tables for both γ
    be_rows = []
    for gamma_case in [1, 0]:
        lr_t, lr_e, _, _, dl = census_all(inventory, 'all', 8, gamma_case)
        N_dcim = dl['total_mvm']

        L.append(f"## Break-Even t_dcim*/t_acim (target=all, r=8, **γ={gamma_case}**)")
        if gamma_case == 0:
            L.append("*(DL always γ=1 vs LR-TT γ=0: LR-TT has no visible path overhead)*")
        L.append("")
        L.append("| ρ_proj | α=0.1 | α=0.5 | α=1.0 | α=2.0 |")
        L.append("|--------|-------|-------|-------|-------|")

        for rp in RHO_SWEEP:
            parts = []
            for alpha in [0.1, 0.5, 1.0, 2.0]:
                num = rp*lr_t['proj'] + lr_t['vis'] + alpha*lr_t['upd']
                ratio = num / N_dcim if N_dcim > 0 else 0
                parts.append(f"{ratio:.1f}")
                be_rows.append({'gamma':gamma_case,'rho_proj':rp,'alpha':alpha,'tdcim_over_tacim':ratio})
            L.append(f"| {rp} | " + " | ".join(parts) + " |")
        L.append("")

    L.append("")
    L.append("### Reading the table")
    L.append("")
    L.append("- If t_dcim/t_acim < table value → **Digital LoRA wins** (DCIM is fast enough)")
    L.append("- If t_dcim/t_acim > table value → **LR-TT wins** (ACIM advantage holds)")
    L.append("")
    L.append("Example: ρ=0.1, α=1.0 → ratio ≈ threshold")
    L.append("- If DCIM tile is 2× faster than ACIM → t_dcim/t_acim = 0.5 → check vs threshold")
    L.append("- If DCIM tile is same speed → t_dcim/t_acim = 1.0 → check vs threshold")
    L.append("")

    # Plot: break-even — γ=0 vs γ=1 overlaid
    _, _, _, _, dl_always = census_all(inventory, 'all', 8, 1)
    N_dl = dl_always['total_mvm']
    alphas_x = np.linspace(0.01, 3.0, 200)

    # ── Plot A: γ=0 vs γ=1 at fixed ρ_proj, sweep α ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: overlaid γ=0 (solid) vs γ=1 (dashed) at ρ=0.1
    ax = axes[0]
    for gamma_case, ls, lbl_g in [(0, '-', 'γ=0 (hidden carry)'), (1, '--', 'γ=1 (visible)')]:
        lr_tc, _, _, _, _ = census_all(inventory, 'all', 8, gamma_case)
        for rp, color in [(0.0, '#4CAF50'), (0.1, '#8BC34A'), (0.5, '#FF9800'), (1.0, '#F44336')]:
            vals = [(rp*lr_tc['proj']+lr_tc['vis']+a*lr_tc['upd'])/N_dl for a in alphas_x]
            label = f'ρ={rp} {lbl_g}' if rp in [0.0, 1.0] else None
            ax.plot(alphas_x, vals, ls, color=color, linewidth=2, alpha=0.9 if gamma_case==0 else 0.5, label=label)
    ax.axhline(1.0, color='black', ls=':', lw=1.5, label='t_dcim = t_acim')
    ax.set_xlabel('α = t_acim_upd / t_acim', fontsize=11)
    ax.set_ylabel('Break-even t_dcim / t_acim', fontsize=11)
    ax.set_title('γ=0 (solid) vs γ=1 (dashed)\nLR-TT wins above line', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='upper left'); ax.grid(alpha=0.3)

    # Right: γ=0 advantage ratio = threshold_g1 / threshold_g0
    ax = axes[1]
    for rp, color in [(0.0, '#4CAF50'), (0.1, '#8BC34A'), (0.5, '#FF9800'), (1.0, '#F44336')]:
        lr_g0, _, _, _, _ = census_all(inventory, 'all', 8, 0)
        lr_g1, _, _, _, _ = census_all(inventory, 'all', 8, 1)
        ratio_vals = []
        for a in alphas_x:
            th_g1 = (rp*lr_g1['proj']+lr_g1['vis']+a*lr_g1['upd'])/N_dl
            th_g0 = (rp*lr_g0['proj']+lr_g0['vis']+a*lr_g0['upd'])/N_dl
            ratio_vals.append(th_g1/th_g0 if th_g0>0 else 1)
        ax.plot(alphas_x, ratio_vals, '-', color=color, label=f'ρ={rp}', linewidth=2)
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel('α = t_acim_upd / t_acim', fontsize=11)
    ax.set_ylabel('threshold(γ=1) / threshold(γ=0)', fontsize=11)
    ax.set_title('γ=0 advantage: how much easier\nfor LR-TT to beat DL', fontweight='bold', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Digital LoRA (always γ=1) vs LR-TT: γ=0 System Advantage', fontweight='bold', fontsize=13)
    plt.tight_layout(rect=[0,0,1,0.93])
    p = os.path.join(PLOTS,"dig_vs_lrtt_break_even_region_v2.png")
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  → {p}")

    cp = os.path.join(OUT,"break_even_dig_vs_lrtt_v2.csv")
    with open(cp,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=be_rows[0].keys()); w.writeheader(); w.writerows(be_rows)
    path = os.path.join(OUT, "DIGITAL_LORA_VS_LRTT_BREAK_EVEN_V3.md")
    with open(path,'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════
# Combined Region Plot
# ═══════════════════════════════════════════════════════════════

def write_combined_region(inventory):
    print("\n" + "="*70 + "\nCOMBINED REGION PLOT\n" + "="*70)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    rhos = np.linspace(0, 1.5, 150)

    for row, (model_label, use_elem) in enumerate([('Tile-Level (Latency UB)', False),
                                                     ('Element-Level (Utilization)', True)]):
        for col, r in enumerate(RANKS):
            ax = axes[row, col]
            for gamma, color, ls in [(0,'#4CAF50','--'),(1,'#9C27B0','-')]:
                lr_t, lr_e, tt_t, tt_e, _ = census_all(inventory, 'all', r, gamma)
                vals = []
                for rp in rhos:
                    if use_elem:
                        c_lr = rp*lr_e['proj'] + lr_e['upd'] + lr_e['vis']
                        c_tt = tt_e['upd'] + tt_e['vis']
                    else:
                        c_lr = rp*lr_t['proj'] + lr_t['upd'] + rp*lr_t['vis'] + lr_t['tr_src']+lr_t['tr_dst']
                        c_tt = tt_t['upd'] + tt_t['vis'] + tt_t['tr_src']+tt_t['tr_dst']
                    vals.append(c_tt/c_lr if c_lr>0 else 1)
                ax.plot(rhos, vals, ls, color=color, label=f'γ={gamma}', linewidth=2)
            ax.axhline(1, color='gray', ls=':', lw=1)
            ax.set_title(f'r={r}', fontweight='bold')
            if col==0: ax.set_ylabel(f'{model_label}\nC_TT/C_LRTT', fontsize=9)
            if row==1: ax.set_xlabel('ρ_proj')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            if use_elem: ax.set_yscale('log'); ax.set_ylim(0.8, 200)
            else: ax.set_ylim(0.3, 2.5)

    fig.suptitle('LR-TT vs TikiTaka: Tile (top) vs Element (bottom), target=all', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    p = os.path.join(PLOTS,"lrtt_vs_tikitaka_region_plot_v2.png")
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f"  → {p}")


# ═══════════════════════════════════════════════════════════════
# Invalidation Note
# ═══════════════════════════════════════════════════════════════

def write_invalidation():
    L = []
    L.append("# Cost Model Invalidation Note V3")
    L.append("")
    L.append("## Previous Invalid Rankings")
    L.append("")
    L.append("```")
    L.append("V0: LR-TT (6229ms) >> TikiTaka (4077ms) >> Digital LoRA (3171ms)  ← INVALID")
    L.append("V1: 'DL has 2.0× more ops than LR-TT'                             ← MEANINGLESS (different units)")
    L.append("V2: Fig 1 bar chart showing constant 2.0× ratio                    ← TRIVIAL IDENTITY")
    L.append("```")
    L.append("")
    L.append("## Root Causes")
    L.append("")
    L.append("| Version | Artifact | Cause |")
    L.append("|---------|----------|-------|")
    L.append("| V0 | LR-TT too expensive | Only LR-TT had adapter cost instantiated (t_tile_ns) |")
    L.append("| V0 | TikiTaka too cheap | θ_upd=null → update=0 |")
    L.append("| V0 | Digital LoRA too cheap | θ_gemm=null → delta=0 |")
    L.append("| V1 | Flat rank sensitivity | Tile granularity: ceil(r/512)=1 for all r≤32 |")
    L.append("| V2 | Constant 2.0× ratio | DL FLOPs / LR-TT elements = 6k/3k = 2.0 (algebraic identity) |")
    L.append("| V2 | Fig 1 meaningless | Different units on same axis (digital FLOP ≠ analog element op) |")
    L.append("")
    L.append("## What V3 Fixes")
    L.append("")
    L.append("1. **DL vs LR-TT**: Reframed as DCIM vs ACIM tile-event break-even (same unit)")
    L.append("2. **LR-TT vs TT**: Dual model — tile (latency UB) + element (utilization)")
    L.append("3. **Rank sensitivity**: Element-level shows true proportional scaling")
    L.append("4. **No fabricated absolute rankings**: All comparisons are symbolic or ratio-based")
    L.append("")
    L.append("## Still Needed for Absolute Comparison")
    L.append("")
    L.append("| Primitive | For | Source |")
    L.append("|-----------|-----|--------|")
    L.append("| t_acim (AIMC tile MVM) | All ACIM methods | AIMC chip literature |")
    L.append("| t_acim_upd (pulsed update) | LR-TT, TikiTaka | AIMC chip measurement |")
    L.append("| t_dcim (DCIM tile MVM) | Digital LoRA break-even | DCIM/SRAM-CIM literature |")
    L.append("| e_acim, e_dcim (energy) | Energy comparison | Chip-level power measurement |")

    path = os.path.join(OUT, "COST_MODEL_INVALIDATION_NOTE.md")
    with open(path,'w') as f: f.write('\n'.join(L))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════
def main():
    print("="*70)
    print("V3 AIMC Training Cost Study — DCIM-aware Dual Model")
    print("="*70)

    inventory = build_layer_inventory()
    print(f"Inventory: {len(inventory)} layers, {sum(l.n_tiles for l in inventory)} tiles")

    write_structural(inventory)
    write_lrtt_vs_tt_util(inventory)
    write_lrtt_vs_tt_lat(inventory)
    write_dl_vs_lrtt(inventory)
    write_combined_region(inventory)
    write_invalidation()

    print("\n" + "="*70 + "\nALL V3 OUTPUTS:")
    for root, dirs, files in os.walk(OUT):
        for f in sorted(files): print(f"  {os.path.join(root, f)}")
    print("="*70)

if __name__ == "__main__":
    main()

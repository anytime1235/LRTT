#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restructured AIMC Training Cost Study: Three-Layer Analysis.

Layer A: Structural / hardware-independent event census
Layer B: Digital LoRA vs LR-TT break-even analysis (symbolic)
Layer C: LR-TT vs TikiTaka normalized ACIM ratio
Optional: Common-path proxy sensitivity

This script replaces the previous absolute-ranking approach, which was invalid
because missing primitive mappings made different methods' costs non-comparable.
"""

import os, sys, csv, math, itertools
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from extract_layer_inventory import (
    build_layer_inventory, get_targeted_layers, summarize_inventory,
    compute_adapter_tile_counts, tile_count, LayerSpec, DEFAULT_TILE_SIZE,
)

OUT = os.path.join(SCRIPT_DIR, "restructured")
os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────
BS = 48
S_DEFAULT = 384
RANKS = [4, 8, 16, 32]
TARGETS = ["attention", "ffn", "all"]
LRTT_TE = 4        # transfer_every for LR-TT (steps)
TT_TE = 1           # transfer_every for TikiTaka (steps)
NUM_READS = 1


# ═══════════════════════════════════════════════════════════════════════════
# LAYER A: Structural Event Census
# ═══════════════════════════════════════════════════════════════════════════

def census_digital_lora(targeted, rank, S=S_DEFAULT):
    """Digital LoRA FLOP / parameter / memory census (hardware-independent)."""
    BT = BS * S
    total_fwd = 0; total_bwd = 0; total_params = 0; total_mem = 0
    for l in targeted:
        M, N = l.M, l.N
        fwd = 2 * BT * rank * (N + M)
        bwd = 2 * BT * rank * (2*M + 2*N)  # 4-GEMM: dA+dZ+dB+dX
        total_fwd += fwd; total_bwd += bwd
        total_params += M*rank + rank*N
        total_mem += 2 * (M*rank + rank*N)  # FP16 weight bytes
    return {
        'F_fwd': total_fwd, 'F_bwd': total_bwd, 'F_total': total_fwd + total_bwd,
        'P_opt': total_params,
        'B_mem_bytes': total_mem,
        'B_adam_bytes': total_mem * 4,  # m, v, m_hat, v_hat (FP16 each → ×4)
    }


def census_lrtt(targeted, rank, gamma, S=S_DEFAULT):
    """LR-TT tile-event census (hardware-independent)."""
    BT = BS * S
    N_proj = 0; N_upd = 0; N_tr_read = 0; N_tr_write = 0; N_vis = 0
    for l in targeted:
        atc = compute_adapter_tile_counts(l, rank)
        nA, nB, nC = atc['n_tiles_A'], atc['n_tiles_B'], l.n_tiles
        # Projection MVM: B.forward + A.backward
        N_proj += BT * (nB + nA)
        # Pulsed update: A.update + B.update
        N_upd += BT * (nA + nB)
        # Transfer (amortized): onehot reads + pulsed writes to C
        N_tr_read += rank * (nA + nB) * NUM_READS / LRTT_TE
        N_tr_write += rank * nC / LRTT_TE
        # Visible forward (gamma=1)
        if gamma == 1:
            N_vis += BT * (nB + nA)
    return {
        'N_proj': int(N_proj), 'N_upd': int(N_upd),
        'N_tr_read': N_tr_read, 'N_tr_write': N_tr_write,
        'N_tr_total': N_tr_read + N_tr_write,
        'N_vis': int(N_vis),
    }


def census_tikitaka(targeted, gamma, S=S_DEFAULT):
    """TikiTaka tile-event census (hardware-independent)."""
    BT = BS * S
    N_upd = 0; N_tr_read = 0; N_tr_write = 0; N_vis = 0
    for l in targeted:
        nU = l.n_tiles
        tr = l.n_tile_rows
        # Full-rank pulsed update
        N_upd += BT * nU
        # Column transfer (transfer_columns=True, n_reads_per_transfer=1)
        N_tr_read += tr / TT_TE
        N_tr_write += tr / TT_TE
        # Visible forward (gamma=1)
        if gamma == 1:
            N_vis += BT * nU
    return {
        'N_proj': 0,  # TikiTaka has no projection
        'N_upd': int(N_upd),
        'N_tr_read': N_tr_read, 'N_tr_write': N_tr_write,
        'N_tr_total': N_tr_read + N_tr_write,
        'N_vis': int(N_vis),
    }


def run_layer_a(inventory):
    """Run Layer A: structural census."""
    print("\n" + "="*70)
    print("LAYER A: Structural / Hardware-Independent Census")
    print("="*70)

    lines = []
    L = lines.append
    L("# Layer A: Structural Event Census (Hardware-Independent)")
    L("")
    L("All values below are **exact architectural facts** derived from the BERT-base")
    L("model structure and training configuration. They require NO hardware primitive")
    L("mapping and are valid regardless of AIMC implementation specifics.")
    L("")

    inv_sum = summarize_inventory(inventory)
    L("## 1. BERT-base Layer Inventory")
    L("")
    L(f"| Item | Value |")
    L(f"|------|-------|")
    L(f"| Encoder linear layers | {inv_sum['total_layers']} |")
    L(f"| Base tiles (512×512) | {inv_sum['total_tiles']} |")
    L(f"| Attention layers | {inv_sum['attention_layers']} ({inv_sum['attention_tiles']} tiles) |")
    L(f"| FFN layers | {inv_sum['ffn_layers']} ({inv_sum['ffn_tiles']} tiles) |")
    L(f"| Weight parameters | {inv_sum['total_params']:,} |")
    L("")

    # Adapter tile counts
    L("## 2. Adapter Tile Counts (per target, per rank)")
    L("")
    L("| Target | Layers | Base Tiles | rank=4 A+B tiles | rank=8 | rank=16 | rank=32 | TikiTaka U tiles |")
    L("|--------|--------|------------|------------------|--------|---------|---------|------------------|")
    for target in TARGETS:
        tgt = get_targeted_layers(inventory, target)
        bt = sum(l.n_tiles for l in tgt)
        ut = bt  # TikiTaka U is same size as base
        parts = []
        for r in RANKS:
            ab = sum(compute_adapter_tile_counts(l, r)['n_tiles_A'] +
                     compute_adapter_tile_counts(l, r)['n_tiles_B'] for l in tgt)
            parts.append(f"{ab}")
        L(f"| {target} | {len(tgt)} | {bt} | {' | '.join(parts)} | {ut} |")
    L("")

    # Event census
    BT = BS * S_DEFAULT
    L(f"## 3. Per-Step Event Census (BS={BS}, S={S_DEFAULT}, BT={BT:,})")
    L("")
    L("### 3.1 Digital LoRA FLOP Census")
    L("")
    L("| Target | rank | F_fwd | F_bwd (4-GEMM) | F_total | P_opt (params) | Bwd/Fwd |")
    L("|--------|------|-------|----------------|---------|----------------|---------|")
    dl_data = {}
    for target in TARGETS:
        tgt = get_targeted_layers(inventory, target)
        for r in RANKS:
            c = census_digital_lora(tgt, r)
            dl_data[(target, r)] = c
            L(f"| {target} | {r} | {c['F_fwd']:,} | {c['F_bwd']:,} | {c['F_total']:,} | {c['P_opt']:,} | {c['F_bwd']/c['F_fwd']:.1f}× |")
    L("")

    L("### 3.2 LR-TT Event Census")
    L("")
    L("| Target | rank | γ | N_proj (MVM) | N_upd (pulsed) | N_tr (amort) | N_vis (γ=1) |")
    L("|--------|------|---|-------------|----------------|-------------|-------------|")
    lrtt_data = {}
    for target in TARGETS:
        tgt = get_targeted_layers(inventory, target)
        for r in RANKS:
            for gamma in [0, 1]:
                c = census_lrtt(tgt, r, gamma)
                lrtt_data[(target, r, gamma)] = c
                L(f"| {target} | {r} | {gamma} | {c['N_proj']:,} | {c['N_upd']:,} | {c['N_tr_total']:.0f} | {c['N_vis']:,} |")
    L("")

    L("### 3.3 TikiTaka Event Census")
    L("")
    L("| Target | γ | N_upd (full-rank) | N_tr (column) | N_vis (γ=1) |")
    L("|--------|---|-------------------|---------------|-------------|")
    tt_data = {}
    for target in TARGETS:
        tgt = get_targeted_layers(inventory, target)
        for gamma in [0, 1]:
            c = census_tikitaka(tgt, gamma)
            tt_data[(target, gamma)] = c
            L(f"| {target} | {gamma} | {c['N_upd']:,} | {c['N_tr_total']:.0f} | {c['N_vis']:,} |")
    L("")

    L("### 3.4 Key Structural Ratios")
    L("")
    L("| Comparison | Metric | Value |")
    L("|-----------|--------|-------|")
    # TikiTaka vs LR-TT update events
    for target in ['all']:
        for r in [8]:
            lr = lrtt_data[(target, r, 0)]
            tt = tt_data[(target, 0)]
            L(f"| TikiTaka/LR-TT update events | target={target}, r={r} | {tt['N_upd']/lr['N_upd']:.2f}× |")
            L(f"| TikiTaka/LR-TT total analog events (γ=0) | target={target} | N_upd ratio dominates |")
            L(f"| LR-TT projection overhead / base MVM | target={target}, r={r} | {lr['N_proj']/(BT*inv_sum['total_tiles']):.2f}× |")
    L("")

    path = os.path.join(OUT, "STRUCTURAL_COMPARISON.md")
    with open(path, 'w') as f: f.write('\n'.join(lines))
    print(f"  → {path}")
    return dl_data, lrtt_data, tt_data


# ═══════════════════════════════════════════════════════════════════════════
# LAYER B: Digital LoRA vs LR-TT Break-Even
# ═══════════════════════════════════════════════════════════════════════════

def run_layer_b(inventory, dl_data, lrtt_data):
    """Layer B: Digital LoRA vs LR-TT break-even analysis."""
    print("\n" + "="*70)
    print("LAYER B: Digital LoRA vs LR-TT Break-Even Analysis")
    print("="*70)

    lines = []
    L = lines.append
    L("# Layer B: Digital LoRA vs LR-TT Break-Even Analysis")
    L("")
    L("## Problem Statement")
    L("")
    L("Digital LoRA and LR-TT operate on **different primitive families**:")
    L("- Digital LoRA: digital GEMM + optimizer + memory traffic (PMCA/DPU)")
    L("- LR-TT: analog MVM read + pulsed update + one-hot transfer (AIMC tiles)")
    L("")
    L("Without mapping both families to the same latency/energy units, **no valid")
    L("absolute ranking exists**. Instead, we derive **break-even thresholds**.")
    L("")

    L("## Cost Formulas")
    L("")
    L("```")
    L("ΔT_dig  = θ_gemm × F_dig  +  θ_opt × P_dig")
    L("ΔT_lrtt = θ_mvm  × N_proj +  θ_upd × N_upd  +  θ_tr × N_tr  +  1_{γ=1} × θ_mvm × N_vis")
    L("```")
    L("")
    L("Where:")
    L("- θ_gemm: digital GEMM latency per FLOP (ns/FLOP)")
    L("- θ_opt:  optimizer update latency per parameter (ns/param)")
    L("- θ_mvm:  AIMC tile MVM latency per event (ns/tile-MVM)")
    L("- θ_upd:  AIMC pulsed update latency per event (ns/tile-update)")
    L("- θ_tr:   transfer event latency (ns/event)")
    L("")

    L("## Break-Even Threshold: θ_gemm*")
    L("")
    L("Setting ΔT_dig = ΔT_lrtt and solving for θ_gemm:")
    L("")
    L("```")
    L("θ_gemm* = (θ_mvm × N_proj + θ_upd × N_upd + θ_tr × N_tr + 1_{γ=1} × θ_mvm × N_vis − θ_opt × P_dig) / F_dig")
    L("```")
    L("")
    L("- If the actual θ_gemm > θ_gemm* → **LR-TT wins** (digital core is too slow)")
    L("- If the actual θ_gemm < θ_gemm* → **Digital LoRA wins** (digital core is fast enough)")
    L("")

    # Compute break-even for various scenarios
    L("## Break-Even Values")
    L("")
    L("Assuming θ_opt ≈ 5 × θ_gemm (Adam ≈ 5 FLOPs/param), θ_tr ≈ θ_mvm:")
    L("")
    L("| Target | rank | γ | θ_mvm=θ_upd condition | θ_gemm* (ns/FLOP) |")
    L("|--------|------|---|-----------------------|-------------------|")

    be_results = []
    for target in TARGETS:
        for r in RANKS:
            dl = dl_data[(target, r)]
            for gamma in [0, 1]:
                lr = lrtt_data[(target, r, gamma)]
                F = dl['F_total']
                P = dl['P_opt']
                N_proj = lr['N_proj']
                N_upd = lr['N_upd']
                N_tr = lr['N_tr_total']
                N_vis = lr['N_vis']

                # Parametric: θ_mvm = θ_upd = θ_tr (same tile), θ_opt = 5*θ_gemm
                # θ_gemm* = θ_mvm * (N_proj + N_upd + N_tr + N_vis) / (F + 5*P)
                N_analog_total = N_proj + N_upd + N_tr + N_vis
                F_effective = F + 5 * P

                # Express as ratio: θ_gemm* / θ_mvm = N_analog / F_effective
                ratio = N_analog_total / F_effective if F_effective > 0 else 0

                be_results.append({
                    'target': target, 'rank': r, 'gamma': gamma,
                    'F_total': F, 'P_opt': P,
                    'N_proj': N_proj, 'N_upd': N_upd, 'N_tr': N_tr, 'N_vis': N_vis,
                    'N_analog_total': N_analog_total,
                    'ratio_gemm_over_mvm': ratio,
                })

                L(f"| {target} | {r} | {gamma} | θ_mvm = θ_upd | θ_gemm* = {ratio:.6f} × θ_mvm |")

    L("")
    L("### Interpretation")
    L("")
    L("For target=all, rank=8, γ=1, with θ_mvm = θ_upd:")
    be_all_8_1 = next(b for b in be_results if b['target']=='all' and b['rank']==8 and b['gamma']==1)
    ratio_val = be_all_8_1['ratio_gemm_over_mvm']
    L(f"- θ_gemm* / θ_mvm = {ratio_val:.6f}")
    L(f"- N_analog_total = {be_all_8_1['N_analog_total']:,}")
    L(f"- F_effective (FLOPs + 5×params) = {be_all_8_1['F_total'] + 5*be_all_8_1['P_opt']:,}")
    L("")
    L(f"If θ_mvm = 256 ns → θ_gemm* = {ratio_val * 256:.4f} ns/FLOP")
    L(f"  → Equivalent to {1e9 / (ratio_val * 256) / 1e9:.2f} GFLOPS digital throughput")
    L(f"  → If your digital core exceeds this, Digital LoRA wins")
    L(f"  → If your digital core is slower, LR-TT wins")
    L("")

    # ── Break-even region plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: θ_gemm* vs rank for different γ
    ax = axes[0]
    for gamma, color, ls in [(0, '#FF9800', '--'), (1, '#F44336', '-')]:
        vals = []
        for r in RANKS:
            b = next(x for x in be_results if x['target']=='all' and x['rank']==r and x['gamma']==gamma)
            vals.append(b['ratio_gemm_over_mvm'] * 256)  # θ_mvm=256 proxy
        ax.plot(RANKS, vals, f'{ls}o', color=color, label=f'γ={gamma}', linewidth=2, markersize=8)
    ax.set_xlabel('LR-TT rank (r)', fontsize=11)
    ax.set_ylabel('θ_gemm* (ns/FLOP) at θ_mvm=256ns', fontsize=11)
    ax.set_title('Break-Even: Digital LoRA wins below line', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(RANKS)

    # Right: 2D region plot — θ_gemm vs θ_upd/θ_mvm
    ax = axes[1]
    upd_ratios = np.linspace(0.1, 5.0, 100)  # θ_upd / θ_mvm
    gemm_thresholds = []

    b = next(x for x in be_results if x['target']=='all' and x['rank']==8 and x['gamma']==1)
    for ur in upd_ratios:
        # θ_gemm* = (θ_mvm * (N_proj + N_vis) + θ_upd * N_upd + θ_mvm * N_tr − 5*θ_gemm*P) / F
        # θ_gemm* = θ_mvm * (N_proj + N_vis + N_tr + ur * N_upd) / (F + 5*P)
        num = b['N_proj'] + b['N_vis'] + b['N_tr'] + ur * b['N_upd']
        den = b['F_total'] + 5 * b['P_opt']
        gemm_thresholds.append(num / den * 256)  # at θ_mvm=256

    ax.plot(upd_ratios, gemm_thresholds, '-', color='#F44336', linewidth=2)
    ax.fill_between(upd_ratios, gemm_thresholds, max(gemm_thresholds)*1.2,
                     alpha=0.15, color='#F44336', label='LR-TT wins')
    ax.fill_between(upd_ratios, 0, gemm_thresholds,
                     alpha=0.15, color='#2196F3', label='Digital LoRA wins')
    ax.set_xlabel('θ_upd / θ_mvm (update-to-read cost ratio)', fontsize=11)
    ax.set_ylabel('θ_gemm* (ns/FLOP) at θ_mvm=256ns', fontsize=11)
    ax.set_title('Break-Even Region (target=all, r=8, γ=1)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "plots", "break_even_dig_vs_lrtt.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")

    # Write CSV
    csv_path = os.path.join(OUT, "break_even_dig_vs_lrtt.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=be_results[0].keys())
        w.writeheader(); w.writerows(be_results)

    L(f"## Break-Even Region Plot")
    L(f"")
    L(f"![Break-Even](plots/break_even_dig_vs_lrtt.png)")
    L("")
    L("**Left**: θ_gemm* vs rank. Digital LoRA wins if its digital core throughput")
    L("exceeds the threshold. **Right**: 2D region showing how the break-even shifts")
    L("as the update-to-read cost ratio θ_upd/θ_mvm varies.")

    path = os.path.join(OUT, "DIGITAL_LORA_VS_LRTT_BREAK_EVEN.md")
    with open(path, 'w') as f: f.write('\n'.join(lines))
    print(f"  → {path}")
    return be_results


# ═══════════════════════════════════════════════════════════════════════════
# LAYER C: LR-TT vs TikiTaka Normalized ACIM Ratio
# ═══════════════════════════════════════════════════════════════════════════

def compute_acim_cost(events, rho_proj, rho_vis, tau):
    """Compute normalized ACIM cost with separate projection weight.

    C = ρ_proj × N_proj + N_upd + τ × N_tr + ρ_vis × N_vis

    ρ_proj: projection MVM cost relative to pulsed update.
            For rank-r adapter tiles (size [M,r] or [r,N]):
            - Physical array is MUCH smaller than 512×512
            - ρ_proj << 1 if adapter tile MVM is fast or overlapped
            - ρ_proj = 0 if projections overlap with base tile MVM (free)
    ρ_vis:  visible forward MVM cost relative to pulsed update (gamma=1 path)
    τ:      transfer cost relative to pulsed update
    """
    return (rho_proj * events['N_proj']
          + events['N_upd']
          + tau * events['N_tr_total']
          + rho_vis * events['N_vis'])


def run_layer_c(inventory, lrtt_data, tt_data):
    """Layer C: LR-TT vs TikiTaka normalized ACIM ratio.

    Key insight: LR-TT projection MVMs use rank-r adapter tiles that are
    physically MUCH smaller than 512×512 base tiles. Their cost should be
    weighted by ρ_proj (which can be << 1) separately from ρ_vis.
    """
    print("\n" + "="*70)
    print("LAYER C: LR-TT vs TikiTaka Normalized ACIM Ratio")
    print("="*70)

    lines = []
    L = lines.append
    L("# Layer C: LR-TT vs TikiTaka Normalized ACIM Ratio")
    L("")
    L("## Rationale")
    L("")
    L("Both LR-TT and TikiTaka use the **same ACIM primitive family**. The meaningful")
    L("comparison is a **primitive-weighted cost ratio**.")
    L("")
    L("## Normalized ACIM Cost (4-term)")
    L("")
    L("```")
    L("C_ACIM = ρ_proj × N_proj  +  N_upd  +  τ × N_tr  +  ρ_vis × N_vis")
    L("```")
    L("")
    L("**ρ_proj** is the key parameter. LR-TT projection MVMs use rank-r adapter")
    L("tiles ([M,r] or [r,N]) that are physically much smaller than 512×512.")
    L("- ρ_proj = 1: projection MVM costs same as a pulsed update (conservative)")
    L("- ρ_proj ≈ 0: projections are near-free (overlap with base MVM or tiny tile)")
    L("- ρ_proj = 0: projections perfectly hidden (optimistic)")
    L("")
    L("**ρ_vis** = visible forward MVM cost / update cost (for γ=1 path)")
    L("")
    L("**TikiTaka has N_proj = 0** — no projection overhead at all.")
    L("")

    # ── Update-only ratio (the structural advantage of rank) ──
    L("## 1. Update-Only Ratio (ρ_proj=0: projections free)")
    L("")
    L("This isolates LR-TT's pure rank-proportional advantage in pulsed updates:")
    L("")
    L("| Target | rank | γ | N_upd_LRTT | N_upd_TT | TT/LRTT | LR-TT saves |")
    L("|--------|------|---|-----------|---------|---------|-------------|")

    ratio_results = []
    for target in TARGETS:
        for gamma in [0, 1]:
            tt = tt_data[(target, gamma)]
            for r in RANKS:
                lr = lrtt_data[(target, r, gamma)]
                # ρ_proj=0: only update matters
                c_lr_upd = lr['N_upd']
                c_tt_upd = tt['N_upd']
                ratio_upd = c_tt_upd / c_lr_upd if c_lr_upd > 0 else 0
                save = (1 - c_lr_upd / c_tt_upd) * 100 if c_tt_upd > 0 else 0
                L(f"| {target} | {r} | {gamma} | {c_lr_upd:,} | {c_tt_upd:,} | {ratio_upd:.3f} | {save:.1f}% |")

                # Full ratio at various ρ_proj
                for rp in [0, 0.1, 0.5, 1.0]:
                    c_lr = compute_acim_cost(lr, rp, 1.0, 1.0)
                    c_tt = compute_acim_cost(tt, rp, 1.0, 1.0)
                    ratio = c_tt / c_lr if c_lr > 0 else 0
                    ratio_results.append({
                        'target': target, 'rank': r, 'gamma': gamma,
                        'rho_proj': rp, 'C_lrtt': c_lr, 'C_tt': c_tt,
                        'ratio_tt_over_lrtt': ratio,
                    })
    L("")

    L("**At ρ_proj=0 (projections free), LR-TT ALWAYS wins** because its update")
    L("tile count (384 for target=all) < TikiTaka's (480). Savings = 20%.")
    L("")

    # ── Full ratio table at different ρ_proj ──
    L("## 2. Full Ratio at Different ρ_proj Values (target=all, τ=1, ρ_vis=1)")
    L("")
    L("| rank | γ | ρ_proj=0 | ρ_proj=0.1 | ρ_proj=0.5 | ρ_proj=1.0 |")
    L("|------|---|----------|-----------|-----------|-----------|")
    for r in RANKS:
        for gamma in [0, 1]:
            parts = []
            for rp in [0, 0.1, 0.5, 1.0]:
                rr = next(x for x in ratio_results
                          if x['target']=='all' and x['rank']==r
                          and x['gamma']==gamma and x['rho_proj']==rp)
                parts.append(f"{rr['ratio_tt_over_lrtt']:.3f}")
            L(f"| {r} | {gamma} | {' | '.join(parts)} |")
    L("")

    # ── Plot 1: Ratio vs ρ_proj (the main plot) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rhos_proj = np.linspace(0, 2.0, 200)

    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]
        target = 'all'
        for r, color, lw in [(4, '#E91E63', 1.5), (8, '#F44336', 2.5), (16, '#FF9800', 1.5), (32, '#FFC107', 1.5)]:
            lr = lrtt_data[(target, r, gamma)]
            tt = tt_data[(target, gamma)]
            ratios = []
            for rp in rhos_proj:
                c_lr = compute_acim_cost(lr, rp, 1.0, 1.0)
                c_tt = compute_acim_cost(tt, rp, 1.0, 1.0)
                ratios.append(c_tt / c_lr if c_lr > 0 else 1.0)
            ax.plot(rhos_proj, ratios, '-', color=color, label=f'r={r}', linewidth=lw)

        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='break-even')
        ax.fill_between(rhos_proj, 1.0, 2.5, alpha=0.08, color='green')
        ax.fill_between(rhos_proj, 0.3, 1.0, alpha=0.08, color='red')
        ax.text(0.05, 1.8, 'LR-TT wins', fontsize=11, color='green', fontweight='bold')
        ax.text(1.2, 0.5, 'TikiTaka wins', fontsize=11, color='red', fontweight='bold')
        ax.set_xlabel('ρ_proj (projection MVM cost / update cost)', fontsize=11)
        ax.set_ylabel('C_TT / C_LRTT', fontsize=11)
        ax.set_title(f'γ={gamma}, target=all', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.3, 2.5)
        ax.set_xlim(0, 2.0)

    fig.suptitle('Cost Ratio vs Projection Cost Weight — above 1 = LR-TT wins',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT, "plots", "ratio_vs_rho_proj.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")

    # ── Plot 2: 2D heatmap ρ_proj vs ρ_vis ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rp_2d = np.linspace(0, 2.0, 100)
    rv_2d = np.linspace(0, 2.0, 100)
    RP, RV = np.meshgrid(rp_2d, rv_2d)

    for ax_i, gamma in enumerate([0, 1]):
        ax = axes[ax_i]
        target, r = 'all', 8
        lr = lrtt_data[(target, r, gamma)]
        tt = tt_data[(target, gamma)]

        Z = np.zeros_like(RP)
        for i in range(len(rv_2d)):
            for j in range(len(rp_2d)):
                c_lr = compute_acim_cost(lr, rp_2d[j], rv_2d[i], 1.0)
                c_tt = compute_acim_cost(tt, rp_2d[j], rv_2d[i], 1.0)
                Z[i, j] = c_tt / c_lr if c_lr > 0 else 1.0

        cs = ax.contourf(RP, RV, Z, levels=np.linspace(0.5, 2.0, 16), cmap='RdYlGn')
        ax.contour(RP, RV, Z, levels=[1.0], colors='black', linewidths=2)
        plt.colorbar(cs, ax=ax, label='C_TT / C_LRTT')
        ax.set_xlabel('ρ_proj (projection cost / update cost)', fontsize=11)
        ax.set_ylabel('ρ_vis (visible fwd cost / update cost)', fontsize=11)
        ax.set_title(f'γ={gamma}, r=8', fontsize=12, fontweight='bold')

    fig.suptitle('Cost Ratio: ρ_proj vs ρ_vis (green=LR-TT wins, red=TikiTaka wins)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT, "plots", "ratio_heatmap_rho_proj_vis.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")

    # ── Plot 3: Ratio vs rank at key ρ_proj values ──
    fig, ax = plt.subplots(figsize=(9, 6))
    for rp, color, ls in [(0, '#4CAF50', '-'), (0.1, '#8BC34A', '--'),
                            (0.5, '#FF9800', '-.'), (1.0, '#F44336', ':')]:
        vals = []
        for r in RANKS:
            lr = lrtt_data[('all', r, 1)]
            tt = tt_data[('all', 1)]
            c_lr = compute_acim_cost(lr, rp, 1.0, 1.0)
            c_tt = compute_acim_cost(tt, rp, 1.0, 1.0)
            vals.append(c_tt / c_lr if c_lr > 0 else 1.0)
        ax.plot(RANKS, vals, f'{ls}o', color=color, label=f'ρ_proj={rp}', linewidth=2, markersize=8)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('LR-TT rank', fontsize=11)
    ax.set_ylabel('C_TT / C_LRTT', fontsize=11)
    ax.set_title('Cost Ratio vs Rank at Different ρ_proj (γ=1, target=all)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(RANKS)
    plt.tight_layout()
    path = os.path.join(OUT, "plots", "ratio_vs_rank_rho_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {path}")

    L("## Plots")
    L("")
    L("### Ratio vs ρ_proj (Main Plot)")
    L("![Ratio vs ρ_proj](plots/ratio_vs_rho_proj.png)")
    L("")
    L("### 2D Heatmap: ρ_proj vs ρ_vis")
    L("![Heatmap](plots/ratio_heatmap_rho_proj_vis.png)")
    L("")
    L("### Ratio vs Rank at Different ρ_proj")
    L("![Ratio vs Rank](plots/ratio_vs_rank_rho_sweep.png)")
    L("")

    L("## Key Findings")
    L("")
    L("1. **At ρ_proj=0 (projections free/overlapped)**: LR-TT ALWAYS wins.")
    L("   Update-only ratio = 480/384 = 1.25× (LR-TT 20% cheaper)")
    L("2. **At ρ_proj≤0.25 for γ=0**: LR-TT wins. Projection overhead is small")
    L("   enough that rank-proportional update savings dominate.")
    L("3. **At ρ_proj=1 (projection = update cost)**: TikiTaka wins because")
    L("   LR-TT's dual projection overhead (7.1M events) exceeds TikiTaka's")
    L("   extra update events (only 1.8M more than LR-TT).")
    L("4. **Physical argument for low ρ_proj**: LR-TT adapter tiles are [M,r]")
    L("   or [r,N] with r≤32, meaning the crossbar is 512×1 or 1×512 (single")
    L("   tile column). MVM through such a thin tile is architecturally fast")
    L("   and can potentially overlap with base tile operations.")
    L("")
    L("## Conclusion")
    L("")
    L("The LR-TT vs TikiTaka comparison hinges critically on **how expensive")
    L("rank-r projection MVMs are relative to full-rank pulsed updates**.")
    L("If projections can be made cheap (ρ_proj < 0.25), LR-TT wins cleanly.")
    L("The architectural case for cheap projections is strong: adapter tiles")
    L("are physically small and can share peripherals with the base path.")

    csv_path = os.path.join(OUT, "normalized_acim_ratio.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ratio_results[0].keys())
        w.writeheader(); w.writerows(ratio_results)

    path = os.path.join(OUT, "LRTT_VS_TIKITAKA_NORMALIZED_ACIM.md")
    with open(path, 'w') as f: f.write('\n'.join(lines))
    print(f"  → {path}")
    return ratio_results


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL: Common Path Proxy Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def run_common_path_sensitivity(inventory):
    """Common-path latency sensitivity (literature proxy only)."""
    print("\n" + "="*70)
    print("OPTIONAL: Common Path Proxy Sensitivity")
    print("="*70)

    lines = []
    L = lines.append
    L("# Optional: Common Path Proxy Sensitivity")
    L("")
    L("These results use **literature-calibrated tile timing proxies** for the")
    L("common base path only. They are clearly labeled as proxy values.")
    L("")

    T_TILES = [100, 128, 256, 512]
    K_BWDS = [2.0, 2.5, 3.0]
    S_VALS = [64, 128, 256, 384]

    L("## T_common vs t_tile and k_bwd (S=384)")
    L("")
    L("| t_tile (ns) | k_bwd | T_base_fwd (ms) | T_base_bwd (ms) | T_common (ms) |")
    L("|-------------|-------|-----------------|-----------------|---------------|")

    results = []
    for t in T_TILES:
        for k in K_BWDS:
            BT = BS * S_DEFAULT
            t_fwd = sum(BT * l.n_tile_cols * t for l in inventory) / 1e6
            t_bwd = k * t_fwd
            t_common = t_fwd + t_bwd
            L(f"| {t} | {k} | {t_fwd:,.2f} | {t_bwd:,.2f} | {t_common:,.2f} |")
            results.append({'t_tile': t, 'k_bwd': k, 'S': S_DEFAULT,
                           'T_fwd_ms': t_fwd, 'T_bwd_ms': t_bwd, 'T_common_ms': t_common})
    L("")

    L("## T_common vs Sequence Length (t_tile=256, k_bwd=2.5)")
    L("")
    L("| S | T_common (ms) | vs S=384 |")
    L("|---|---------------|----------|")
    base_384 = None
    for S in S_VALS:
        BT = BS * S
        t_fwd = sum(BT * l.n_tile_cols * 256 for l in inventory) / 1e6
        t_common = t_fwd * (1 + 2.5)
        if S == 384: base_384 = t_common
        ratio = t_common / base_384 if base_384 else 1.0
        L(f"| {S} | {t_common:,.2f} | {ratio:.2f}× |")
    L("")

    L("## Key Observation")
    L("")
    L("T_common is **identical across all three methods**. It scales linearly with")
    L("S×t_tile×(1+k_bwd). The method-specific ΔT is always additive on top of this")
    L("shared base. For large models, T_common dominates T_step for all methods.")

    csv_path = os.path.join(OUT, "common_path_sensitivity.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)

    path = os.path.join(OUT, "OPTIONAL_COMMON_PATH_PROXY_SENSITIVITY.md")
    with open(path, 'w') as f: f.write('\n'.join(lines))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# LIMITATIONS REPORT
# ═══════════════════════════════════════════════════════════════════════════

def write_limitations():
    lines = []
    L = lines.append
    L("# Cost Model Limitations and Missing Primitives")
    L("")
    L("## 1. What is proven structurally (no hardware mapping needed)")
    L("")
    L("- [x] BERT-base has 72 encoder linear layers mapped to 480 base tiles")
    L("- [x] LR-TT adapter tiles scale as ceil(M/512)+ceil(N/512) per layer (rank<512)")
    L("- [x] TikiTaka full-rank U tiles = base tile count (480 for target=all)")
    L("- [x] LR-TT transfer cost is linear in rank")
    L("- [x] Digital LoRA backward requires 4 GEMMs (not 3): dA, dZ, dB, dX_LoRA")
    L("- [x] Digital LoRA Bwd/Fwd FLOP ratio = 2.0×")
    L("- [x] T_common is bit-identical across all three methods")
    L("- [x] All costs scale linearly with sequence length S")
    L("- [x] Dynamic padding mean S_pad=352, waste=51% for SQuAD v1.1")
    L("")
    L("## 2. What requires primitive mapping")
    L("")
    L("| Primitive | Symbol | Needed for | Source |")
    L("|-----------|--------|-----------|--------|")
    L("| Digital GEMM latency | θ_gemm | Digital LoRA ΔT | PMCA/DPU spec or chip measurement |")
    L("| Optimizer latency | θ_opt | Digital LoRA ΔT | PMCA/DPU spec |")
    L("| AIMC pulsed update | θ_upd | LR-TT & TikiTaka ΔT | AIMC chip measurement or literature |")
    L("| AIMC transfer event | θ_tr | LR-TT & TikiTaka ΔT | AIMC chip measurement |")
    L("| Tile MVM energy | e_tile | All energy metrics | AIMC chip power measurement |")
    L("| Digital GEMM energy | e_gemm | Digital LoRA energy | PMCA/DPU power spec |")
    L("")
    L("## 3. Prior absolute rankings that are INVALID")
    L("")
    L("The following absolute ranking from the previous study is **scientifically invalid**:")
    L("")
    L("```")
    L("INVALID: LR-TT (6229ms) >> TikiTaka (4077ms) >> Digital LoRA (3171ms)")
    L("```")
    L("")
    L("Reasons:")
    L("- Digital LoRA ΔT = 0 because θ_gemm is unmapped (not because it's actually free)")
    L("- TikiTaka ΔT_update = 0 because θ_upd is unmapped (not because update is free)")
    L("- Only LR-TT has its projection MVM cost instantiated (via θ_mvm = t_tile_ns)")
    L("- This creates an artificial bias making LR-TT appear most expensive")
    L("")
    L("## 4. What would unlock a full absolute comparison")
    L("")
    L("| Priority | Parameter | Impact |")
    L("|----------|-----------|--------|")
    L("| Critical | θ_upd (pulsed update latency) | Determines TT vs LR-TT absolute cost |")
    L("| Critical | θ_gemm (digital GEMM throughput) | Determines Digital LoRA absolute cost |")
    L("| High | e_tile, e_upd (energy per event) | Enables TOPS/W comparison |")
    L("| Medium | θ_opt (optimizer throughput) | Refines Digital LoRA ΔT |")
    L("| Low | θ_tr (transfer event cost) | Transfer is small fraction of total |")
    L("")
    L("## 5. Recommended path forward")
    L("")
    L("1. Survey AIMC chip literature for θ_upd range (expected: 100ns–10μs/tile/sample)")
    L("2. Obtain PMCA/DPU throughput spec for θ_gemm (expected: 0.001–0.1 ns/FLOP)")
    L("3. Run break-even analysis at those ranges to determine which method wins")
    L("4. Only claim absolute ranking when all critical primitives are mapped")

    path = os.path.join(OUT, "COST_MODEL_LIMITATIONS_AND_MISSING_PRIMITIVES.md")
    with open(path, 'w') as f: f.write('\n'.join(lines))
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("RESTRUCTURED AIMC TRAINING COST STUDY")
    print("Three-Layer Analysis: Structural → Break-Even → Normalized Ratio")
    print("="*70)

    inventory = build_layer_inventory()
    print(f"\nInventory: {len(inventory)} layers, {sum(l.n_tiles for l in inventory)} tiles")

    dl_data, lrtt_data, tt_data = run_layer_a(inventory)
    run_layer_b(inventory, dl_data, lrtt_data)
    run_layer_c(inventory, lrtt_data, tt_data)
    run_common_path_sensitivity(inventory)
    write_limitations()

    print("\n" + "="*70)
    print("ALL OUTPUTS:")
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.md') or f.endswith('.csv'):
            print(f"  {OUT}/{f}")
    for f in sorted(os.listdir(os.path.join(OUT, "plots"))):
        print(f"  {OUT}/plots/{f}")
    print("="*70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate all paper-ready plots for LR-TT vs TikiTaka ALPINE-calibrated study.

Plots:
  1. LRTT_VS_TT_LATENCY_INTERVAL.png
  2. LRTT_VS_TT_ETOPS_INTERVAL.png
  3. LRTT_VS_TT_ENERGY_INTERVAL.png
  4. LRTT_VS_TT_ETOPSW_INTERVAL.png
  5. LRTT_VS_TT_SEQUENCE_SWEEP.png
  6. LRTT_VS_TT_TILE_UPPER_BOUND_SUPP.png
"""

import os
import sys
import math
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
    run_full_sweep, run_sequence_sweep, compute_model_a, compute_model_b,
    compute_lrtt_element_counts, compute_tikitaka_element_counts,
    get_dynamic_S_pad_mean, build_layer_inventory, get_targeted_layers,
    ALPHA_T_SCENARIOS, KAPPA_E_SCENARIOS, RANKS, TARGETS,
    BATCH_SIZE, S_PAD_DEFAULT, FIXED_SEQ_LENGTHS,
    TAU_ACIM, EPS_ACIM_PJ, ALPINE_TILE_MVM_NS, ALPINE_TILE_SIZE,
)

# x-axis: TikiTaka as first bar, then LR-TT per rank
X_LABELS = ['TT'] + [f'r={r}' for r in RANKS]
N_X = len(X_LABELS)  # 1 + len(RANKS)

OUT = SCRIPT_DIR

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 150,
})

# Colors
C_LRTT_G0 = '#1976D2'    # Blue
C_LRTT_G1 = '#64B5F6'    # Light blue
C_TT_G0 = '#E53935'      # Red
C_TT_G1 = '#EF9A9A'      # Light red
C_LRTT_PROJ = '#1565C0'
C_LRTT_UPD = '#42A5F5'
C_LRTT_VIS = '#90CAF9'
C_TT_UPD = '#EF5350'
C_TT_VIS = '#EF9A9A'


def build_data():
    """Build all data needed for plots."""
    print("Building layer inventory...")
    inventory = build_layer_inventory()
    trace_path = os.path.join(COST_MODEL_DIR, "batch_trace.csv")
    dyn_mean = get_dynamic_S_pad_mean(trace_path)
    print(f"  Layers: {len(inventory)}, Dynamic S_pad mean: {dyn_mean}")

    print("Running full sweep...")
    sweep = run_full_sweep(inventory)

    print("Running sequence sweep...")
    seq_results = run_sequence_sweep(
        inventory, rank=8, target="all", alpha_t=1.0,
        dynamic_S_pad_mean=dyn_mean,
    )

    return inventory, sweep, seq_results, dyn_mean


def _get_interval_data(model_a_results, method, target, gamma, rank=None,
                       vary='alpha_t', fixed_kappa=1.0, fixed_alpha=1.0):
    """Extract min/max/nominal interval for a given scenario."""
    filtered = [r for r in model_a_results
                if r.method == method and r.target == target and r.gamma == gamma]
    if rank is not None:
        filtered = [r for r in filtered if r.rank == rank]

    if vary == 'alpha_t':
        filtered = [r for r in filtered if r.kappa_e == fixed_kappa]
        nominal = [r for r in filtered if r.alpha_t == 1.0]
    elif vary == 'kappa_e':
        filtered = [r for r in filtered if r.alpha_t == fixed_alpha]
        nominal = [r for r in filtered if r.kappa_e == 1.0]
    else:
        nominal = filtered

    return filtered, nominal


# =============================================================================
# Plot 1: Latency Interval
# =============================================================================

def _draw_interval_bars(ax, xpos, w, data_list, color, whisker_hw=0.06):
    """Draw bars at explicit x positions with interval whiskers.
    data_list: [(x, min, nom, max), ...]
    """
    for (xi, mn, nom, mx) in data_list:
        ax.bar(xi, nom, w, color=color, edgecolor='#333', lw=0.4, zorder=3)
        ax.plot([xi, xi], [mn, mx], color='#333', lw=1.5, zorder=4)
        ax.plot([xi-whisker_hw, xi+whisker_hw], [mn, mn], color='#333', lw=1.2, zorder=4)
        ax.plot([xi-whisker_hw, xi+whisker_hw], [mx, mx], color='#333', lw=1.2, zorder=4)


def _build_unified_data(model_a_results, target, extract_fn, vary='alpha_t', fixed_kappa=1.0, fixed_alpha=1.0):
    """Build unified x-axis data: TT at x=0, then LR-TT at x=1..N_RANKS.

    Returns dict keyed by (method_label, gamma) -> list of (x, min, nom, max).
    extract_fn: callable(result) -> float (the y-value to extract)
    """
    x = np.arange(N_X)  # 0=TT, 1..7=ranks
    data = {}

    for gamma in [0, 1]:
        # TikiTaka at x=0
        filt = [r for r in model_a_results
                if r.method == "tikitaka_6t1c" and r.target == target and r.gamma == gamma]
        if vary == 'alpha_t':
            filt = [r for r in filt if r.kappa_e == fixed_kappa]
            nom_list = [r for r in filt if r.alpha_t == 1.0]
        elif vary == 'kappa_e':
            filt = [r for r in filt if r.alpha_t == fixed_alpha]
            nom_list = [r for r in filt if r.kappa_e == 1.0]
        else:
            nom_list = filt

        vals = [extract_fn(r) for r in filt]
        nom_v = extract_fn(nom_list[0]) if nom_list else 0
        tt_key = ('TT', gamma)
        data[tt_key] = [(x[0], min(vals) if vals else 0, nom_v, max(vals) if vals else 0)]

        # LR-TT at x=1..
        lr_key = ('LRTT', gamma)
        lr_list = []
        for ri, rank in enumerate(RANKS):
            filt = [r for r in model_a_results
                    if r.method == "lrtt_6t1c" and r.target == target
                    and r.gamma == gamma and r.rank == rank]
            if vary == 'alpha_t':
                filt = [r for r in filt if r.kappa_e == fixed_kappa]
                nom_list = [r for r in filt if r.alpha_t == 1.0]
            elif vary == 'kappa_e':
                filt = [r for r in filt if r.alpha_t == fixed_alpha]
                nom_list = [r for r in filt if r.kappa_e == 1.0]
            else:
                nom_list = filt
            vals = [extract_fn(r) for r in filt]
            nom_v = extract_fn(nom_list[0]) if nom_list else 0
            lr_list.append((x[1 + ri], min(vals) if vals else 0, nom_v, max(vals) if vals else 0))
        data[lr_key] = lr_list

    return data


def _setup_unified_xaxis(ax, add_separator=True):
    """Configure x-axis for unified TT + LR-TT layout."""
    x = np.arange(N_X)
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=8)
    # Bold the TT label
    labels = ax.get_xticklabels()
    labels[0].set_fontweight('bold')
    labels[0].set_color(C_TT_G0)
    ax.set_xlim(-0.6, N_X - 0.4)
    if add_separator:
        ax.axvline(0.5, color='#ccc', ls='-', lw=0.8, zorder=0)


# =============================================================================
# Plot 1: Latency Interval
# =============================================================================

def plot_latency_interval(model_a_results):
    """LRTT_VS_TT_LATENCY_INTERVAL.png
    TikiTaka as bar at x=0, LR-TT bars at x=1..7.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    w = 0.35

    for col, target in enumerate(TARGETS):
        ax = axes[col]

        data = _build_unified_data(model_a_results, target,
                                   lambda r: r.DeltaT_total_ns / 1e6)

        # gamma=0: left half of each bar position
        tt_g0 = data[('TT', 0)]
        lr_g0 = data[('LRTT', 0)]
        _draw_interval_bars(ax, -w/2, w*0.9, tt_g0, C_TT_G0)
        _draw_interval_bars(ax, -w/2, w*0.9, lr_g0, C_LRTT_G0)

        # gamma=1: right half
        tt_g1 = data[('TT', 1)]
        lr_g1 = data[('LRTT', 1)]
        _draw_interval_bars(ax, +w/2, w*0.9, tt_g1, C_TT_G1)
        _draw_interval_bars(ax, +w/2, w*0.9, lr_g1, C_LRTT_G1)

        # Ratio annotations (gamma=0 LR-TT vs TT)
        tt_nom = tt_g0[0][2]  # TikiTaka nominal
        for (xi, _, nom, mx) in lr_g0:
            if nom > 0 and tt_nom > 0:
                ratio = tt_nom / nom
                ax.annotate(f'{ratio:.0f}x',
                            xy=(xi - w/2, mx),
                            xytext=(0, 6), textcoords='offset points',
                            fontsize=6, ha='center', color='#444', fontweight='bold',
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.set_yscale('log')
        _setup_unified_xaxis(ax)
        ax.set_title(f'target = {target}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('$\\Delta T_{step}$ [ms]  (log scale)', fontsize=10)
        ax.grid(axis='y', alpha=0.15, which='both', zorder=0)

    handles = [
        mpatches.Patch(facecolor=C_TT_G0, edgecolor='#333', label='TikiTaka $\\gamma$=0'),
        mpatches.Patch(facecolor=C_TT_G1, edgecolor='#333', label='TikiTaka $\\gamma$=1'),
        mpatches.Patch(facecolor=C_LRTT_G0, edgecolor='#333', label='LR-TT $\\gamma$=0'),
        mpatches.Patch(facecolor=C_LRTT_G1, edgecolor='#333', label='LR-TT $\\gamma$=1'),
        Line2D([0],[0], color='#333', lw=1.5, label='$\\alpha_t$ scenario range'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='#ccc')
    fig.suptitle('Adapter-Path Latency $\\Delta T_{step}$: TikiTaka (full-rank) vs LR-TT (low-rank)\n'
                 'Model A: ALPINE-throughput-calibrated  |  '
                 f'BS={BATCH_SIZE}, S={S_PAD_DEFAULT}  |  '
                 f'$\\alpha_t \\in$ {ALPHA_T_SCENARIOS}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    path = os.path.join(OUT, "LRTT_VS_TT_LATENCY_INTERVAL.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Plot 2: Reduction Factor + EqOps
# =============================================================================

def plot_etops_interval(model_a_results):
    """LRTT_VS_TT_ETOPS_INTERVAL.png
    Top: Latency Reduction Factor (TT/LRTT)
    Bottom: Absolute EqOps per step
    TikiTaka shown as bar at x=0 (ratio=1 by definition; absolute ops shown).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), height_ratios=[1, 1])
    w = 0.30

    for col, target in enumerate(TARGETS):
        # ---- Row 0: Latency Reduction Factor ----
        ax = axes[0, col]
        x = np.arange(N_X)

        for gi, (gamma, color_lr, color_tt) in enumerate([
            (0, C_LRTT_G0, C_TT_G0), (1, C_LRTT_G1, C_TT_G1)
        ]):
            offset = (-0.5 + gi) * w
            # TT baseline
            filt_tt, nom_tt = _get_interval_data(model_a_results, "tikitaka_6t1c", target, gamma, rank=0)
            tt_by_alpha = {r.alpha_t: r.DeltaT_total_ns for r in filt_tt}
            tt_nom = nom_tt[0].DeltaT_total_ns if nom_tt else 1

            # TT at x=0: ratio = 1 (reference)
            ax.bar(x[0] + offset, 1.0, w*0.9, color=color_tt, edgecolor='#333', lw=0.4, zorder=3)

            # LR-TT ratios at x=1..
            for ri, rank in enumerate(RANKS):
                filt_lr, nom_lr = _get_interval_data(model_a_results, "lrtt_6t1c", target, gamma, rank)
                ratios = [tt_by_alpha.get(r.alpha_t, 1) / r.DeltaT_total_ns for r in filt_lr]
                nom_ratio = tt_nom / nom_lr[0].DeltaT_total_ns if nom_lr else 1
                xi = x[1 + ri] + offset
                ax.bar(xi, nom_ratio, w*0.9, color=color_lr, edgecolor='#333', lw=0.4, zorder=3)
                ax.plot([xi, xi], [min(ratios), max(ratios)], color='#333', lw=1.5, zorder=4)
                ax.plot([xi-0.05, xi+0.05], [min(ratios), min(ratios)], color='#333', lw=1.0, zorder=4)
                ax.plot([xi-0.05, xi+0.05], [max(ratios), max(ratios)], color='#333', lw=1.0, zorder=4)
                # Label
                if gi == 0:
                    ax.text(xi, max(ratios)*1.03, f'{nom_ratio:.0f}x', ha='center', va='bottom',
                            fontsize=6, fontweight='bold', color='#444',
                            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        ax.axhline(1, color='#999', ls=':', lw=0.8, zorder=1)
        _setup_unified_xaxis(ax)
        ax.set_title(f'target = {target}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Latency Reduction Factor\n$\\Delta T_{TT} / \\Delta T_{LRTT}$', fontsize=9)
        ax.grid(axis='y', alpha=0.15, zorder=0)
        ax.set_ylim(bottom=0)

        # ---- Row 1: Absolute EqOps ----
        ax2 = axes[1, col]

        data_ops = _build_unified_data(model_a_results, target,
                                        lambda r: r.EqOps / 1e9, vary='none')

        for gi, (gamma, color_lr, color_tt) in enumerate([
            (0, C_LRTT_G0, C_TT_G0), (1, C_LRTT_G1, C_TT_G1)
        ]):
            offset = (-0.5 + gi) * w
            tt_data = data_ops[('TT', gamma)]
            lr_data = data_ops[('LRTT', gamma)]
            for (xi, _, nom, _) in tt_data:
                ax2.bar(xi + offset, nom, w*0.9, color=color_tt, edgecolor='#333', lw=0.4, zorder=3)
            for (xi, _, nom, _) in lr_data:
                ax2.bar(xi + offset, nom, w*0.9, color=color_lr, edgecolor='#333', lw=0.4, zorder=3)

        ax2.set_yscale('log')
        _setup_unified_xaxis(ax2)
        if col == 0:
            ax2.set_ylabel('Active Ops per Step [GOps]\n(log scale)', fontsize=9)
        ax2.grid(axis='y', alpha=0.15, which='both', zorder=0)

    handles = [
        mpatches.Patch(facecolor=C_TT_G0, edgecolor='#333', label='TikiTaka $\\gamma$=0'),
        mpatches.Patch(facecolor=C_TT_G1, edgecolor='#333', label='TikiTaka $\\gamma$=1'),
        mpatches.Patch(facecolor=C_LRTT_G0, edgecolor='#333', label='LR-TT $\\gamma$=0'),
        mpatches.Patch(facecolor=C_LRTT_G1, edgecolor='#333', label='LR-TT $\\gamma$=1'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='#ccc')
    fig.suptitle('Latency Reduction Factor & Active Operation Count\n'
                 'Top: how many times faster LR-TT is vs TikiTaka  |  '
                 'Bottom: total active ops per step (log scale)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    path = os.path.join(OUT, "LRTT_VS_TT_ETOPS_INTERVAL.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Plot 3: Energy Interval
# =============================================================================

def plot_energy_interval(model_a_results):
    """LRTT_VS_TT_ENERGY_INTERVAL.png"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    w = 0.35

    for col, target in enumerate(TARGETS):
        ax = axes[col]
        data = _build_unified_data(model_a_results, target,
                                   lambda r: r.DeltaE_total_pj / 1e6,
                                   vary='kappa_e')

        for gi, (gamma, color_lr, color_tt) in enumerate([
            (0, C_LRTT_G0, C_TT_G0), (1, C_LRTT_G1, C_TT_G1)
        ]):
            offset = (-0.5 + gi) * w
            _draw_interval_bars(ax, offset, w*0.9, data[('TT', gamma)], color_tt)
            _draw_interval_bars(ax, offset, w*0.9, data[('LRTT', gamma)], color_lr)

        ax.set_yscale('log')
        _setup_unified_xaxis(ax)
        ax.set_title(f'target = {target}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('$\\Delta E_{step}$ [$\\mu$J]  (log scale)', fontsize=10)
        ax.grid(axis='y', alpha=0.15, which='both', zorder=0)

    handles = [
        mpatches.Patch(facecolor=C_TT_G0, edgecolor='#333', label='TikiTaka $\\gamma$=0'),
        mpatches.Patch(facecolor=C_TT_G1, edgecolor='#333', label='TikiTaka $\\gamma$=1'),
        mpatches.Patch(facecolor=C_LRTT_G0, edgecolor='#333', label='LR-TT $\\gamma$=0'),
        mpatches.Patch(facecolor=C_LRTT_G1, edgecolor='#333', label='LR-TT $\\gamma$=1'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='#ccc')
    fig.suptitle('Adapter-Path Energy $\\Delta E_{step}$: TikiTaka vs LR-TT\n'
                 f'$\\kappa_e \\in$ {KAPPA_E_SCENARIOS}  |  Sensitivity-based (update energy unknown)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    path = os.path.join(OUT, "LRTT_VS_TT_ENERGY_INTERVAL.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Plot 4: Energy Reduction Factor
# =============================================================================

def plot_etopsw_interval(model_a_results):
    """LRTT_VS_TT_ETOPSW_INTERVAL.png — Energy Reduction Factor."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    kappa_colors = {1: '#1565C0', 2: '#42A5F5', 5: '#90CAF9', 10: '#BBDEFB'}
    n_kappa = len(KAPPA_E_SCENARIOS)

    for col, target in enumerate(TARGETS):
        ax = axes[col]
        x = np.arange(N_X)
        total_w = 0.7
        w = total_w / n_kappa

        for ki, kappa_e in enumerate(KAPPA_E_SCENARIOS):
            offset = -total_w/2 + (ki + 0.5) * w
            color = kappa_colors[kappa_e]

            # TT baseline energy
            filt_tt = [r for r in model_a_results
                       if r.method == "tikitaka_6t1c" and r.target == target
                       and r.gamma == 0 and r.alpha_t == 1.0 and r.kappa_e == kappa_e]
            tt_e = filt_tt[0].DeltaE_total_pj if filt_tt else 1

            # TT at x=0: ratio = 1
            ax.bar(x[0] + offset, 1.0, w*0.9, color=color, edgecolor='#333', lw=0.3, zorder=3,
                   label=f'$\\kappa_e$={kappa_e}' if col == 0 else '')

            # LR-TT at x=1..
            for ri, rank in enumerate(RANKS):
                filt_lr = [r for r in model_a_results
                           if r.method == "lrtt_6t1c" and r.target == target
                           and r.gamma == 0 and r.rank == rank
                           and r.alpha_t == 1.0 and r.kappa_e == kappa_e]
                lr_e = filt_lr[0].DeltaE_total_pj if filt_lr else 1
                ratio = tt_e / lr_e
                xi = x[1 + ri] + offset
                ax.bar(xi, ratio, w*0.9, color=color, edgecolor='#333', lw=0.3, zorder=3)
                if ki == 0:  # label only first kappa
                    ax.text(xi, ratio + 1.5, f'{ratio:.0f}x', ha='center', va='bottom',
                            fontsize=5.5, color='#333',
                            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        ax.axhline(1, color='#999', ls=':', lw=0.8)
        _setup_unified_xaxis(ax)
        ax.set_title(f'target = {target}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Energy Reduction Factor\n$\\Delta E_{TT} / \\Delta E_{LRTT}$  ($\\gamma$=0)',
                          fontsize=9)
        ax.grid(axis='y', alpha=0.15, zorder=0)
        ax.set_ylim(bottom=0)

    handles = [mpatches.Patch(facecolor=kappa_colors[k], edgecolor='#333',
               label=f'$\\kappa_e$ = {k}') for k in KAPPA_E_SCENARIOS]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='#ccc')
    fig.suptitle('Energy Reduction Factor: TikiTaka / LR-TT  ($\\gamma$=0)\n'
                 'TT bar = 1x (reference)  |  LR-TT bars = how many times less energy  |  Supplementary',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    path = os.path.join(OUT, "LRTT_VS_TT_ETOPSW_INTERVAL.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Plot 5: Sequence Sweep
# =============================================================================

def plot_sequence_sweep(inventory, dyn_mean):
    """LRTT_VS_TT_SEQUENCE_SWEEP.png"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    target = "all"
    rank = 8
    targeted = get_targeted_layers(inventory, target)

    seq_lens = list(FIXED_SEQ_LENGTHS)
    if dyn_mean is not None:
        dyn_int = int(round(dyn_mean))
        if dyn_int not in seq_lens:
            seq_lens.append(dyn_int)
    seq_lens.sort()

    styles = [
        ("tikitaka_6t1c", 0, C_TT_G0, '-', '^', 'TikiTaka $\\gamma$=0'),
        ("tikitaka_6t1c", 1, C_TT_G1, ':', 'v', 'TikiTaka $\\gamma$=1'),
        ("lrtt_6t1c", 0, C_LRTT_G0, '-', 'o', 'LR-TT r=8 $\\gamma$=0'),
        ("lrtt_6t1c", 1, C_LRTT_G1, '--', 's', 'LR-TT r=8 $\\gamma$=1'),
    ]

    for method, gamma, color, ls, marker, label in styles:
        ys = []
        for S in seq_lens:
            BT = BATCH_SIZE * S
            if method == "lrtt_6t1c":
                ec = compute_lrtt_element_counts(targeted, rank, gamma, BT)
            else:
                ec = compute_tikitaka_element_counts(targeted, gamma, BT)
            ec.target = target
            r = compute_model_a(ec, alpha_t=1.0, S_pad=S)
            ys.append(r.DeltaT_total_ns / 1e6)

        ax.plot(seq_lens, ys, color=color, ls=ls, marker=marker, markersize=6,
                lw=1.8, label=label, zorder=3)

    if dyn_mean is not None:
        dyn_int = int(round(dyn_mean))
        ax.axvline(dyn_int, color='gray', ls=':', lw=1.0, alpha=0.7)
        ax.text(dyn_int + 3, ax.get_ylim()[1] * 0.95,
                f'dyn mean\n{dyn_mean:.0f}', fontsize=7, color='gray', va='top')

    ax.set_xlabel('Sequence Length $S$', fontsize=10)
    ax.set_ylabel('$\\Delta T_{step}$ [ms]  (log scale)', fontsize=10)
    ax.set_yscale('log')
    ax.set_title(f'Sequence Length Sweep  |  target={target}, r={rank}, $\\alpha_t$=1.0\n'
                 f'BS={BATCH_SIZE}  |  Model A (ALPINE-throughput-calibrated)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, edgecolor='#ccc')
    ax.grid(alpha=0.15, which='both')

    plt.tight_layout()
    path = os.path.join(OUT, "LRTT_VS_TT_SEQUENCE_SWEEP.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Plot 6: Tile Upper Bound (Supplementary)
# =============================================================================

def plot_tile_upper_bound(inventory):
    """LRTT_VS_TT_TILE_UPPER_BOUND_SUPP.png — TT as bar + LR-TT bars."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    BT = BATCH_SIZE * S_PAD_DEFAULT
    w = 0.35

    for col, target in enumerate(TARGETS):
        ax = axes[col]
        targeted = get_targeted_layers(inventory, target)
        x = np.arange(N_X)

        for gi, (gamma, color_lr, color_tt, hatch) in enumerate([
            (0, C_LRTT_G0, C_TT_G0, ''),
            (1, C_LRTT_G1, C_TT_G1, '///')
        ]):
            offset = (-0.5 + gi) * w

            # TT at x=0
            rb_tt = compute_model_b(targeted, "tikitaka_6t1c", 0, gamma, BT)
            ax.bar(x[0] + offset, rb_tt.DeltaT_total_ns / 1e6, w*0.9,
                   color=color_tt, edgecolor='#333', lw=0.4, hatch=hatch, zorder=3)

            # LR-TT at x=1..
            for ri, rank in enumerate(RANKS):
                rb = compute_model_b(targeted, "lrtt_6t1c", rank, gamma, BT)
                ax.bar(x[1+ri] + offset, rb.DeltaT_total_ns / 1e6, w*0.9,
                       color=color_lr, edgecolor='#333', lw=0.4, hatch=hatch, zorder=3)

        _setup_unified_xaxis(ax)
        ax.set_title(f'target = {target}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('$\\Delta T_{step}$ [ms] (fixed-tile model)', fontsize=10)
        ax.grid(axis='y', alpha=0.15, zorder=0)

    handles = [
        mpatches.Patch(facecolor=C_TT_G0, edgecolor='#333', label='TikiTaka $\\gamma$=0'),
        mpatches.Patch(facecolor=C_TT_G1, edgecolor='#333', hatch='///', label='TikiTaka $\\gamma$=1'),
        mpatches.Patch(facecolor=C_LRTT_G0, edgecolor='#333', label='LR-TT $\\gamma$=0'),
        mpatches.Patch(facecolor=C_LRTT_G1, edgecolor='#333', hatch='///', label='LR-TT $\\gamma$=1'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor='#ccc')
    fig.suptitle('Model B: Fixed-Tile ALPINE Upper Bound (Supplementary)\n'
                 '256x256 tile, 100 ns/tile  |  Conservative: rank sensitivity suppressed (ceil(r/256)=1)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    path = os.path.join(OUT, "LRTT_VS_TT_TILE_UPPER_BOUND_SUPP.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Generating LR-TT vs TikiTaka ALPINE-calibrated plots")
    print("=" * 70)

    inventory, sweep, seq_results, dyn_mean = build_data()
    model_a = sweep['model_a']

    print("\n--- Generating plots ---")
    plot_latency_interval(model_a)
    plot_etops_interval(model_a)
    plot_energy_interval(model_a)
    plot_etopsw_interval(model_a)
    plot_sequence_sweep(inventory, dyn_mean)
    plot_tile_upper_bound(inventory)

    print("\n--- All plots generated ---")


if __name__ == "__main__":
    main()

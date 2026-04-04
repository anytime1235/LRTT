#!/usr/bin/env python3
"""Generate all cost model comparison plots."""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results_training_step.csv")
FWD_CSV = os.path.join(SCRIPT_DIR, "results_forward_only.csv")
TRACE_CSV = os.path.join(SCRIPT_DIR, "batch_trace.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Colors
C_DL = '#2196F3'   # Digital LoRA - blue
C_LR0 = '#FF9800'  # LRTT gamma=0 - orange
C_LR1 = '#F44336'  # LRTT gamma=1 - red
C_TT0 = '#4CAF50'  # TikiTaka gamma=0 - green
C_TT1 = '#9C27B0'  # TikiTaka gamma=1 - purple

def load_results():
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))

def load_trace():
    with open(TRACE_CSV) as f:
        return list(csv.DictReader(f))

def get(rows, method, target, gamma, rank, t_tile, k_bwd, S):
    for r in rows:
        if (r['method'] == method and r['target'] == target
            and r['gamma'] == str(gamma) and r['rank'] == str(rank)
            and r['t_tile_ns'] == str(t_tile) and r['k_bwd'] == str(k_bwd)
            and r['S_or_dynamic'] == str(S)):
            return r
    return None

# ============================================================
# Plot 1: Main 3-method comparison bar chart
# ============================================================
def plot_main_comparison(rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    t_tile, k_bwd, S = 256, 2.5, 384

    for ax_i, target in enumerate(['attention', 'ffn', 'all']):
        methods = []
        t_common = []
        delta_t = []
        colors = []
        labels = []

        # Digital LoRA rank=8
        r = get(rows, 'digital_lora', target, 1, 8, t_tile, k_bwd, S)
        if r:
            methods.append('Digital\nLoRA r=8')
            t_common.append(float(r['T_common_ns'])/1e6)
            delta_t.append(float(r['DeltaT_ns'])/1e6)
            colors.append(C_DL)

        # LRTT gamma=0 rank=8
        r = get(rows, 'lrtt_6t1c', target, 0, 8, t_tile, k_bwd, S)
        if r:
            methods.append('LR-TT\nγ=0 r=8')
            t_common.append(float(r['T_common_ns'])/1e6)
            delta_t.append(float(r['DeltaT_ns'])/1e6)
            colors.append(C_LR0)

        # LRTT gamma=1 rank=8
        r = get(rows, 'lrtt_6t1c', target, 1, 8, t_tile, k_bwd, S)
        if r:
            methods.append('LR-TT\nγ=1 r=8')
            t_common.append(float(r['T_common_ns'])/1e6)
            delta_t.append(float(r['DeltaT_ns'])/1e6)
            colors.append(C_LR1)

        # TikiTaka gamma=0
        r = get(rows, 'tikitaka_6t1c', target, 0, 0, t_tile, k_bwd, S)
        if r:
            methods.append('TikiTaka\nγ=0')
            t_common.append(float(r['T_common_ns'])/1e6)
            delta_t.append(float(r['DeltaT_ns'])/1e6)
            colors.append(C_TT0)

        # TikiTaka gamma=1
        r = get(rows, 'tikitaka_6t1c', target, 1, 0, t_tile, k_bwd, S)
        if r:
            methods.append('TikiTaka\nγ=1')
            t_common.append(float(r['T_common_ns'])/1e6)
            delta_t.append(float(r['DeltaT_ns'])/1e6)
            colors.append(C_TT1)

        x = np.arange(len(methods))
        ax = axes[ax_i]
        bars_c = ax.bar(x, t_common, 0.6, label='T_common', color='#E0E0E0', edgecolor='#999')
        bars_d = ax.bar(x, delta_t, 0.6, bottom=t_common, color=colors, edgecolor='#333', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_title(f'target = {target}', fontsize=12, fontweight='bold')
        if ax_i == 0:
            ax.set_ylabel('T_step (ms)', fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Training Step Latency Comparison (t_tile={t_tile}ns, k_bwd={k_bwd}, S={S})',
                 fontsize=13, fontweight='bold')
    fig.legend(['T_common (shared)', 'ΔT (method-specific)'],
               loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, "01_main_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path}")

# ============================================================
# Plot 2: Sequence length scaling
# ============================================================
def plot_seq_scaling(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    S_vals = [64, 128, 256, 384]
    t_tile, k_bwd = 256, 2.5

    configs = [
        ('digital_lora', 1, 8, 'Digital LoRA r=8', C_DL, '-o'),
        ('lrtt_6t1c', 0, 8, 'LR-TT γ=0 r=8', C_LR0, '--s'),
        ('lrtt_6t1c', 1, 8, 'LR-TT γ=1 r=8', C_LR1, '-s'),
        ('tikitaka_6t1c', 0, 0, 'TikiTaka γ=0', C_TT0, '--^'),
        ('tikitaka_6t1c', 1, 0, 'TikiTaka γ=1', C_TT1, '-^'),
    ]

    for method, gamma, rank, label, color, marker in configs:
        vals = []
        for S in S_vals:
            r = get(rows, method, 'all', gamma, rank, t_tile, k_bwd, S)
            vals.append(float(r['T_step_ns'])/1e6 if r else 0)
        ax.plot(S_vals, vals, marker, color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel('Sequence Length (S)', fontsize=11)
    ax.set_ylabel('T_step (ms)', fontsize=11)
    ax.set_title('Step Latency vs Sequence Length (target=all)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xticks(S_vals)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_seq_length_scaling.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path}")

# ============================================================
# Plot 3: Tile latency sensitivity
# ============================================================
def plot_tile_sensitivity(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    t_tiles = [100, 128, 256, 512]
    k_bwd, S = 2.5, 384

    configs = [
        ('digital_lora', 1, 8, 'Digital LoRA r=8', C_DL, '-o'),
        ('lrtt_6t1c', 1, 8, 'LR-TT γ=1 r=8', C_LR1, '-s'),
        ('tikitaka_6t1c', 1, 0, 'TikiTaka γ=1', C_TT1, '-^'),
    ]

    for method, gamma, rank, label, color, marker in configs:
        vals = []
        for t in t_tiles:
            r = get(rows, method, 'all', gamma, rank, t, k_bwd, S)
            vals.append(float(r['T_step_ns'])/1e6 if r else 0)
        ax.plot(t_tiles, vals, marker, color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel('t_tile (ns)', fontsize=11)
    ax.set_ylabel('T_step (ms)', fontsize=11)
    ax.set_title('Step Latency vs Tile MVM Latency (target=all, S=384)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(t_tiles)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_tile_latency_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path}")

# ============================================================
# Plot 4: k_common_bwd sensitivity
# ============================================================
def plot_kbwd_sensitivity(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    k_vals = [2.0, 2.5, 3.0]
    t_tile, S = 256, 384

    configs = [
        ('digital_lora', 1, 8, 'Digital LoRA r=8', C_DL, '-o'),
        ('lrtt_6t1c', 1, 8, 'LR-TT γ=1 r=8', C_LR1, '-s'),
        ('tikitaka_6t1c', 1, 0, 'TikiTaka γ=1', C_TT1, '-^'),
    ]

    for method, gamma, rank, label, color, marker in configs:
        vals = []
        for k in k_vals:
            r = get(rows, method, 'all', gamma, rank, t_tile, k, S)
            vals.append(float(r['T_step_ns'])/1e6 if r else 0)
        ax.plot(k_vals, vals, marker, color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel('k_common_bwd', fontsize=11)
    ax.set_ylabel('T_step (ms)', fontsize=11)
    ax.set_title('Step Latency vs Backward Multiplier (target=all, S=384)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(k_vals)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_kbwd_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path}")

# ============================================================
# Plot 5: Rank sensitivity for LR-TT
# ============================================================
def plot_rank_sensitivity(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ranks = [4, 8, 16, 32]
    t_tile, k_bwd, S = 256, 2.5, 384

    # Left: DeltaT vs rank
    ax = axes[0]
    for gamma, color, ls in [(0, C_LR0, '--'), (1, C_LR1, '-')]:
        vals = []
        for r in ranks:
            row = get(rows, 'lrtt_6t1c', 'all', gamma, r, t_tile, k_bwd, S)
            vals.append(float(row['DeltaT_ns'])/1e6 if row else 0)
        ax.plot(ranks, vals, f'{ls}s', color=color, label=f'LR-TT γ={gamma}', linewidth=2, markersize=7)

    ax.set_xlabel('Rank (r)', fontsize=11)
    ax.set_ylabel('ΔT (ms)', fontsize=11)
    ax.set_title('LR-TT Adapter Delta vs Rank', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(ranks)

    # Right: Transfer cost vs rank
    ax = axes[1]
    for gamma, color, ls in [(0, C_LR0, '--'), (1, C_LR1, '-')]:
        vals = []
        for r in ranks:
            row = get(rows, 'lrtt_6t1c', 'all', gamma, r, t_tile, k_bwd, S)
            vals.append(float(row['DeltaT_transfer_ns'])/1e3 if row else 0)  # microseconds
        ax.plot(ranks, vals, f'{ls}s', color=color, label=f'LR-TT γ={gamma}', linewidth=2, markersize=7)

    ax.set_xlabel('Rank (r)', fontsize=11)
    ax.set_ylabel('ΔT_transfer (μs)', fontsize=11)
    ax.set_title('LR-TT Transfer Cost vs Rank (te=4)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(ranks)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_rank_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path}")

# ============================================================
# Plot 6: Cost decomposition stacked bar
# ============================================================
def plot_cost_decomposition(rows):
    fig, ax = plt.subplots(figsize=(12, 6))
    t_tile, k_bwd, S = 256, 2.5, 384

    configs = [
        ('Digital LoRA\nr=8', 'digital_lora', 1, 8),
        ('LR-TT γ=0\nr=8', 'lrtt_6t1c', 0, 8),
        ('LR-TT γ=1\nr=8', 'lrtt_6t1c', 1, 8),
        ('TikiTaka\nγ=0', 'tikitaka_6t1c', 0, 0),
        ('TikiTaka\nγ=1', 'tikitaka_6t1c', 1, 0),
    ]

    labels_list = []
    base_fwd = []
    base_bwd = []
    delta_fwd = []
    delta_bwd = []
    delta_xfer = []

    for label, method, gamma, rank in configs:
        r = get(rows, method, 'all', gamma, rank, t_tile, k_bwd, S)
        if r:
            labels_list.append(label)
            base_fwd.append(float(r['T_base_fwd_ns'])/1e6)
            base_bwd.append(float(r['T_base_bwd_ns'])/1e6)
            delta_fwd.append(float(r['DeltaT_adapter_fwd_ns'])/1e6)
            delta_bwd.append(float(r['DeltaT_adapter_bwd_ns'])/1e6)
            delta_xfer.append(float(r['DeltaT_transfer_ns'])/1e6)

    x = np.arange(len(labels_list))
    w = 0.5

    b1 = ax.bar(x, base_fwd, w, label='Base Forward (AIMC)', color='#90CAF9', edgecolor='#333')
    b2 = ax.bar(x, base_bwd, w, bottom=base_fwd, label='Base Backward (k×fwd)', color='#42A5F5', edgecolor='#333')
    bottom2 = [a+b for a,b in zip(base_fwd, base_bwd)]
    b3 = ax.bar(x, delta_fwd, w, bottom=bottom2, label='Δ Adapter Forward', color='#FFA726', edgecolor='#333')
    bottom3 = [a+b for a,b in zip(bottom2, delta_fwd)]
    b4 = ax.bar(x, delta_bwd, w, bottom=bottom3, label='Δ Adapter Bwd/Update', color='#EF5350', edgecolor='#333')
    bottom4 = [a+b for a,b in zip(bottom3, delta_bwd)]
    b5 = ax.bar(x, delta_xfer, w, bottom=bottom4, label='Δ Transfer', color='#AB47BC', edgecolor='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=9)
    ax.set_ylabel('T_step (ms)', fontsize=11)
    ax.set_title('Training Step Cost Decomposition (target=all, S=384)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_cost_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path}")

# ============================================================
# Plot 7: Dynamic padding batch trace distribution
# ============================================================
def plot_batch_trace(trace):
    s_pads = np.array([int(t['S_pad']) for t in trace])
    wastes = np.array([float(t['padding_waste']) for t in trace])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: S_pad histogram
    ax = axes[0]
    ax.hist(s_pads, bins=40, color='#42A5F5', edgecolor='#333', alpha=0.85)
    ax.axvline(np.mean(s_pads), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(s_pads):.1f}')
    ax.axvline(384, color='#333', linestyle=':', linewidth=1, label='Max=384')
    ax.set_xlabel('S_pad (tokens)', fontsize=11)
    ax.set_ylabel('Number of batches', fontsize=11)
    ax.set_title('Dynamic Padding: S_pad Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: Padding waste histogram
    ax = axes[1]
    ax.hist(wastes * 100, bins=40, color='#FFA726', edgecolor='#333', alpha=0.85)
    ax.axvline(np.mean(wastes)*100, color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(wastes)*100:.1f}%')
    ax.set_xlabel('Padding Waste (%)', fontsize=11)
    ax.set_ylabel('Number of batches', fontsize=11)
    ax.set_title('Dynamic Padding: Waste Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "07_batch_trace_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path}")

# ============================================================
# Plot 8: Op count comparison
# ============================================================
def plot_op_counts(rows):
    fig, ax = plt.subplots(figsize=(12, 6))
    t_tile, k_bwd, S = 256, 2.5, 384

    configs = [
        ('Digital LoRA r=8', 'digital_lora', 1, 8),
        ('LR-TT γ=0 r=8', 'lrtt_6t1c', 0, 8),
        ('LR-TT γ=1 r=8', 'lrtt_6t1c', 1, 8),
        ('TikiTaka γ=0', 'tikitaka_6t1c', 0, 0),
        ('TikiTaka γ=1', 'tikitaka_6t1c', 1, 0),
    ]
    colors_list = [C_DL, C_LR0, C_LR1, C_TT0, C_TT1]

    labels_list = []
    base_mvms = []
    adapter_mvms = []
    update_evts = []

    for label, method, gamma, rank in configs:
        r = get(rows, method, 'all', gamma, rank, t_tile, k_bwd, S)
        if r:
            labels_list.append(label)
            base_mvms.append(int(r['base_mvm_events'])/1e6)
            adapter_mvms.append(int(r['adapter_mvm_events'])/1e6)
            update_evts.append(int(r['adapter_update_events'])/1e6)

    x = np.arange(len(labels_list))
    w = 0.25

    ax.bar(x - w, base_mvms, w, label='Base MVM events (M)', color='#90CAF9', edgecolor='#333')
    ax.bar(x, adapter_mvms, w, label='Adapter MVM events (M)', color='#FFA726', edgecolor='#333')
    ax.bar(x + w, update_evts, w, label='Update events (M)', color='#EF5350', edgecolor='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=9)
    ax.set_ylabel('Events (millions)', fontsize=11)
    ax.set_title('Tile Operation Counts (target=all, S=384)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "08_op_counts.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating plots...")
    rows = load_results()
    trace = load_trace()

    plot_main_comparison(rows)
    plot_seq_scaling(rows)
    plot_tile_sensitivity(rows)
    plot_kbwd_sensitivity(rows)
    plot_rank_sensitivity(rows)
    plot_cost_decomposition(rows)
    plot_batch_trace(trace)
    plot_op_counts(rows)

    print(f"\nAll plots saved to: {OUT_DIR}/")

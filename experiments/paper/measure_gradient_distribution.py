"""Measure actual gradient element distributions for all sublayer types.

Uses paper_experiment.py create_model() with method=ideal, target_layers=all,
IO perfect. Fixed LR (no warmup/scheduler). Captures gradient histograms
(log-scale bins) for Q, K, V, O, FFN1, FFN2 over full 1 epoch.

Histogram approach: 500 log-spaced bins from 10^-7 to 10^-2, accumulated
per step. Memory: ~288KB total regardless of epoch length.
"""
import sys
sys.path.insert(0, "/root/LRTT/experiments/paper")

import os
import re
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, set_seed, default_data_collator
from scipy import stats as scipy_stats

from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext

# Import from paper_experiment directly
from paper_experiment import (
    MODEL_NAME, create_model, load_data, create_optimizer,
    is_target_layer, get_layer_subtype, TARGET_LAYERS,
)

# ── Config ──
SEED = 42
BATCH_SIZES = [16, 8]  # 24 OOM confirmed, start from 16
ANALOG_LR = 0.0357
CLASSIFIER_LR = 0.00076
LN_LR = 0.00076
NUM_STEPS = 0  # 0 = full 1 epoch
OUT_DIR = "/root/results/gradient_distribution"
os.makedirs(OUT_DIR, exist_ok=True)

# Histogram config: 500 log-spaced bins covering 10^-7 to 10^-2 (5 decades, 100 bins/decade)
HIST_LOG_MIN = -7.0
HIST_LOG_MAX = -2.0
HIST_N_BINS = 500
HIST_BIN_EDGES = np.linspace(HIST_LOG_MIN, HIST_LOG_MAX, HIST_N_BINS + 1)
# Extra bins for underflow (<10^-7) and overflow (>10^-2)

# Log summary stats every N steps
LOG_SUMMARY_EVERY = 50
# Save raw gradient samples for first N steps (for detailed per-element plots)
RAW_SAMPLE_STEPS = 5

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Sublayer classification ──
def get_sublayer(name):
    if 'query' in name: return 'Q'
    if 'key' in name: return 'K'
    if 'value' in name: return 'V'
    if 'attention.output.dense' in name: return 'O'
    if 'intermediate.dense' in name: return 'FFN1'
    if '.output.dense' in name and 'attention' not in name: return 'FFN2'
    return 'other'

def get_layer_idx(name):
    m = re.search(r'layer\.(\d+)', name)
    return int(m.group(1)) if m else -1


# ── Create args namespace matching paper_experiment expectations ──
args = argparse.Namespace(
    method="ideal",
    target_layers="all",
    seed=SEED,
    model="bert-base-uncased",
    task="squad_v1.1",
    batch_size=BATCH_SIZES[0],
    grad_accum_steps=1,
    epochs=1,
    max_steps=0,
    analog_lr=ANALOG_LR,
    classifier_lr=CLASSIFIER_LR,
    ln_lr=LN_LR,
    warmup_ratio=0.0,
    min_lr_rate=1.0,
    # Ideal: no bits needed
    n_bits=None,
    dw_min=None,
    desired_bl=31,
    count_pulses=False,
    # No IO quantization (perfect)
    io_bits=0,
    noise_management="abs_max",
    learn_out_scaling=False,
    per_layer_bits=None,
    # Not used for ideal
    gamma=1.0,
    transfer_every=None,
    units_in_mbatch=None,
    fast_lr=None,
    transfer_lr=None,
    scale_transfer_lr=None,
    n_reads_per_transfer=None,
    with_reset_prob=None,
    transfer_bl=None,
    ttv1_mode=None,
    ttv1_fast_pulse_type=None,
    ttv1_transfer_pulse_type=None,
    n_bits_slow=None,
    pulse_type="stochastic",
    device_type="constant_step",
    ls_gamma_up_ratio=1.0,
    ls_gamma_down_ratio=1.0,
    ls_noise_ratio=0.0,
    ls_gamma_up=None,
    ls_gamma_down=None,
    # Diagnostics off
    diag_update_exact=False,
    diag_steps=0,
    diag_carry_path=False,
    diag_vrc_windows="50,200",
    eco_rounding="stochastic",
    output_dir=OUT_DIR,
    mode="fixed",
)

# Temporarily set TARGET_LAYERS to "all"
import paper_experiment
paper_experiment.TARGET_LAYERS = "all"

# ── Model ──
print("\n=== Creating model (ideal, all layers, IO perfect) ===")
model, rpu_config, _ = create_model(args, device_str=str(device))

# ── Data (loaded later after batch size determined) ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Optimizer (fixed LR, no scheduler) ──
print("\n=== Creating optimizer (fixed LR, no scheduler) ===")
optimizer = create_optimizer(model, args)
print(f"  LR: analog={ANALOG_LR}, classifier={CLASSIFIER_LR}, ln={LN_LR}")

# ── Register backward hooks with histogram accumulation ──
print("\n=== Registering gradient hooks ===")

# Per-layer histogram accumulator: name -> counts array (HIST_N_BINS + 2 for under/overflow)
grad_histograms = {}      # name -> np.array of shape (HIST_N_BINS + 2,)
grad_summary = {}          # name -> list of per-step summary dicts (logged every LOG_SUMMARY_EVERY)
grad_raw_samples = {}      # name -> list of subsampled gradient arrays (first RAW_SAMPLE_STEPS steps)
grad_step_counts = {}      # name -> number of steps accumulated
current_step = [0]

def make_bwd_hook(name):
    def hook(module, grad_input, grad_output):
        g = grad_output[0].detach()
        step = current_step[0]
        sub = get_sublayer(name)
        layer = get_layer_idx(name)

        g_flat = g.float().cpu().numpy().flatten()
        abs_g = np.abs(g_flat)
        nz = abs_g[abs_g > 0]

        # ── Accumulate histogram (every step) ──
        if name not in grad_histograms:
            # bins: [underflow] + [HIST_N_BINS regular bins] + [overflow]
            grad_histograms[name] = np.zeros(HIST_N_BINS + 2, dtype=np.int64)
            grad_step_counts[name] = 0

        if len(nz) > 0:
            log_nz = np.log10(nz)
            # Regular bins
            counts, _ = np.histogram(log_nz, bins=HIST_BIN_EDGES)
            grad_histograms[name][1:-1] += counts.astype(np.int64)
            # Underflow (< 10^HIST_LOG_MIN)
            grad_histograms[name][0] += int(np.sum(log_nz < HIST_LOG_MIN))
            # Overflow (> 10^HIST_LOG_MAX)
            grad_histograms[name][-1] += int(np.sum(log_nz > HIST_LOG_MAX))
        grad_step_counts[name] += 1

        # ── Summary stats (every LOG_SUMMARY_EVERY steps) ──
        if step % LOG_SUMMARY_EVERY == 0:
            if name not in grad_summary:
                grad_summary[name] = []
            grad_summary[name].append({
                'step': step, 'sublayer': sub, 'layer': layer,
                'absmax': float(abs_g.max()),
                'mean_abs': float(abs_g.mean()),
                'std': float(g_flat.std()),
                'norm': float(np.linalg.norm(g_flat)),
                'n_elements': len(g_flat),
                'n_nonzero': len(nz),
                'q001': float(np.percentile(abs_g, 0.1)),
                'q01': float(np.percentile(abs_g, 1)),
                'q10': float(np.percentile(abs_g, 10)),
                'q50': float(np.percentile(abs_g, 50)),
                'q90': float(np.percentile(abs_g, 90)),
                'q99': float(np.percentile(abs_g, 99)),
                'q999': float(np.percentile(abs_g, 99.9)),
            })

        # ── Raw samples (first RAW_SAMPLE_STEPS steps only) ──
        if step < RAW_SAMPLE_STEPS:
            if name not in grad_raw_samples:
                grad_raw_samples[name] = []
            if len(g_flat) > 100000:
                idx = np.random.choice(len(g_flat), 100000, replace=False)
                grad_raw_samples[name].append(g_flat[idx])
            else:
                grad_raw_samples[name].append(g_flat.copy())

    return hook

hooks = []
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        sub = get_sublayer(name)
        if sub != 'other':
            hooks.append(module.register_full_backward_hook(make_bwd_hook(name)))
print(f"  Registered {len(hooks)} hooks (all 72 layers: Q/K/V/O/FFN1/FFN2 × 12)")

# ── Training loop with OOM fallback ──
import gc

def run_epoch(batch_size):
    """Attempt 1 epoch with given batch_size. Returns (step, success)."""
    global grad_histograms, grad_summary, grad_raw_samples, grad_step_counts

    print(f"\n=== Loading data (batch_size={batch_size}) ===")
    train_loader, _, _ = load_data(tokenizer, batch_size, seed=SEED)
    total_steps = len(train_loader)
    if NUM_STEPS > 0:
        total_steps = min(total_steps, NUM_STEPS)
    print(f"  Train batches: {len(train_loader)}, steps: {total_steps}")

    print(f"\n=== Running {total_steps} steps (1 epoch, no scheduler, batch={batch_size}) ===")
    print(f"  Histogram: every step ({HIST_N_BINS} log bins, 10^{HIST_LOG_MIN}~10^{HIST_LOG_MAX})")
    print(f"  Summary stats: every {LOG_SUMMARY_EVERY} steps")
    print(f"  Raw samples: first {RAW_SAMPLE_STEPS} steps")
    start_time = time.time()

    # Reset accumulators
    grad_histograms.clear()
    grad_summary.clear()
    grad_raw_samples.clear()
    grad_step_counts.clear()

    model.train()
    step = 0
    for batch in train_loader:
        if NUM_STEPS > 0 and step >= NUM_STEPS:
            break
        current_step[0] = step

        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        except RuntimeError as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "caching" in err_msg or "nvml" in err_msg or "cuda" in err_msg:
                print(f"\n  CUDA/OOM error at step {step} with batch_size={batch_size}: {str(e)[:100]}")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                return step, False
            raise

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step}/{total_steps}, Loss: {loss.item():.4f}, {elapsed:.0f}s")
        step += 1

    elapsed = time.time() - start_time
    print(f"  Completed {step} steps in {elapsed:.0f}s")
    return step, True

# Try batch sizes in order
used_batch_size = None
step = 0
for bs in BATCH_SIZES:
    print(f"\n{'='*60}")
    print(f"  Trying batch_size={bs}")
    print(f"{'='*60}")
    step, success = run_epoch(bs)
    if success:
        used_batch_size = bs
        break
    else:
        print(f"  Falling back to smaller batch size...")
        torch.cuda.empty_cache()
        gc.collect()

if used_batch_size is None:
    print("ERROR: All batch sizes OOM'd. Aborting.")
    sys.exit(1)

print(f"\n  Successfully completed with batch_size={used_batch_size}")

for h in hooks:
    h.remove()

# ── Save data ──
print(f"\n=== Saving results ===")

import csv

# 1. Summary stats CSV
with open(os.path.join(OUT_DIR, "grad_summary.csv"), 'w', newline='') as f:
    fields = ['name', 'step', 'sublayer', 'layer', 'absmax', 'mean_abs', 'std', 'norm',
              'n_elements', 'n_nonzero', 'q001', 'q01', 'q10', 'q50', 'q90', 'q99', 'q999']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for name, stats_list in grad_summary.items():
        for s in stats_list:
            s['name'] = name
            w.writerow(s)
print(f"  Saved grad_summary.csv")

# 2. Per-layer histograms as npz
hist_dict = {'bin_edges': HIST_BIN_EDGES}
for name, counts in grad_histograms.items():
    sub = get_sublayer(name)
    layer = get_layer_idx(name)
    hist_dict[f"{sub}_L{layer}"] = counts
    hist_dict[f"{sub}_L{layer}_nsteps"] = np.array([grad_step_counts[name]])
np.savez_compressed(os.path.join(OUT_DIR, "grad_histograms.npz"), **hist_dict)
print(f"  Saved grad_histograms.npz ({len(grad_histograms)} layers)")

# 3. Raw samples (first few steps)
raw_dict = {}
for name, samples in grad_raw_samples.items():
    sub = get_sublayer(name)
    layer = get_layer_idx(name)
    for i, s in enumerate(samples):
        raw_dict[f"{sub}_L{layer}_step{i}"] = s
np.savez_compressed(os.path.join(OUT_DIR, "grad_raw_samples.npz"), **raw_dict)
print(f"  Saved grad_raw_samples.npz ({len(raw_dict)} arrays)")

# ── Plots ──
print("\n=== Generating plots ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sublayer_types = ['Q', 'K', 'V', 'O', 'FFN1', 'FFN2']
colors = {'Q': '#2196F3', 'K': '#4CAF50', 'V': '#FF9800', 'O': '#F44336',
          'FFN1': '#9C27B0', 'FFN2': '#795548'}

bin_centers = (HIST_BIN_EDGES[:-1] + HIST_BIN_EDGES[1:]) / 2
bin_width = HIST_BIN_EDGES[1] - HIST_BIN_EDGES[0]

# Aggregate histograms by sublayer type (pool all 12 layers)
pooled_hist = {}
pooled_total = {}
for name, counts in grad_histograms.items():
    sub = get_sublayer(name)
    if sub not in pooled_hist:
        pooled_hist[sub] = np.zeros_like(counts)
        pooled_total[sub] = 0
    pooled_hist[sub] += counts
    pooled_total[sub] += counts.sum()

# ── Plot 1: log|g| PDF all sublayers (from accumulated histograms) ──
fig, ax = plt.subplots(figsize=(14, 8))
for sub in sublayer_types:
    if sub not in pooled_hist: continue
    counts = pooled_hist[sub][1:-1]  # exclude under/overflow
    total = pooled_total[sub]
    density = counts / (total * bin_width) if total > 0 else counts
    underflow = pooled_hist[sub][0]
    overflow = pooled_hist[sub][-1]
    ax.plot(bin_centers, density, color=colors[sub], linewidth=1.5,
            label=f'{sub} (N={total:,}, <10⁻⁷:{underflow:,}, >10⁻²:{overflow:,})')

ax.set_xlabel('log₁₀(|gradient element|)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Gradient Element Distribution — All Sublayer Types (1 epoch, {step} steps)\n'
             'IdealDevice, all 72 layers analog, IO perfect, BERT-base SQuAD',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grad_pdf_log_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved grad_pdf_log_all.png")

# ── Plot 2: Per-sublayer with Gaussian/Laplace fit (from histograms) ──
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle(f'Per-Sublayer Gradient Distribution (1 epoch histogram)\n'
             f'log₁₀(|g|) PDF with Gaussian & Laplace fits', fontsize=14, fontweight='bold')

for ax, sub in zip(axes.flat, sublayer_types):
    if sub not in pooled_hist:
        ax.set_visible(False); continue
    counts = pooled_hist[sub][1:-1]
    total = pooled_total[sub]
    density = counts / (total * bin_width) if total > 0 else counts

    ax.plot(bin_centers, density, color=colors[sub], linewidth=2, label='Empirical', zorder=3)

    # Estimate mean/std from histogram for Gaussian fit
    weighted = counts * bin_centers
    mu_est = weighted.sum() / counts.sum() if counts.sum() > 0 else 0
    var_est = (counts * (bin_centers - mu_est)**2).sum() / counts.sum() if counts.sum() > 0 else 1
    sigma_est = np.sqrt(var_est)

    ax.plot(bin_centers, scipy_stats.norm.pdf(bin_centers, mu_est, sigma_est),
            'b--', linewidth=1.5, alpha=0.7, label=f'Gaussian (μ={mu_est:.1f}, σ={sigma_est:.2f})')

    # Laplace fit from histogram
    # median from cumulative
    cum = np.cumsum(counts)
    med_idx = np.searchsorted(cum, cum[-1]/2) if cum[-1] > 0 else len(cum)//2
    mu_l = bin_centers[min(med_idx, len(bin_centers)-1)]
    b_l = (counts * np.abs(bin_centers - mu_l)).sum() / counts.sum() if counts.sum() > 0 else 1
    ax.plot(bin_centers, scipy_stats.laplace.pdf(bin_centers, mu_l, b_l),
            'r:', linewidth=1.5, alpha=0.7, label=f'Laplace (μ={mu_l:.1f}, b={b_l:.2f})')

    underflow = pooled_hist[sub][0]
    overflow = pooled_hist[sub][-1]
    ax.set_title(f'{sub}: total={total:,}, underflow={underflow:,}, overflow={overflow:,}', fontsize=10)
    ax.set_xlabel('log₁₀(|gradient|)'); ax.set_ylabel('Density')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grad_pdf_per_sublayer.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved grad_pdf_per_sublayer.png")

# ── Plot 3: CCDF from histograms ──
fig, ax = plt.subplots(figsize=(14, 8))
for sub in sublayer_types:
    if sub not in pooled_hist: continue
    counts = pooled_hist[sub][1:-1]
    total = counts.sum() + pooled_hist[sub][-1]  # include overflow in total
    # CCDF: P(|g| > x) = sum of counts above x / total
    ccdf = np.cumsum(counts[::-1])[::-1] / total
    # x-axis in linear scale (10^bin_center)
    x_vals = 10**bin_centers
    ax.loglog(x_vals, ccdf, color=colors[sub], linewidth=1.5, label=sub, alpha=0.8)

ax.set_xlabel('|gradient element|', fontsize=12)
ax.set_ylabel('P(|g| > x)  (CCDF)', fontsize=12)
ax.set_title(f'Gradient CCDF (1 epoch) — Straight line = Power-law tail',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grad_ccdf_tail.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved grad_ccdf_tail.png")

# ── Plot 4: Per-layer within each sublayer (from per-layer histograms) ──
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('Per-Layer Gradient Distribution (1 epoch histogram)', fontsize=14, fontweight='bold')
cmap = plt.cm.viridis(np.linspace(0, 1, 12))

for ax, sub in zip(axes.flat, sublayer_types):
    for name, counts in grad_histograms.items():
        s = get_sublayer(name)
        if s != sub: continue
        layer = get_layer_idx(name)
        c = counts[1:-1]
        total = c.sum()
        if total < 100: continue
        density = c / (total * bin_width)
        ax.plot(bin_centers, density, color=cmap[layer], linewidth=0.8, alpha=0.7, label=f'L{layer}')
    ax.set_title(f'{sub}', fontsize=11, fontweight='bold')
    ax.set_xlabel('log₁₀(|gradient|)'); ax.set_ylabel('Density')
    ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grad_pdf_per_layer.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved grad_pdf_per_layer.png")

# ── Plot 5: Heavy-tail metrics ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Gradient Heavy-Tail Metrics by Sublayer (1 epoch)', fontsize=14, fontweight='bold')

# Compute metrics from histograms
sub_stats = {}
for sub in sublayer_types:
    if sub not in pooled_hist: continue
    counts = pooled_hist[sub][1:-1]
    total = counts.sum()
    if total == 0: continue

    # Weighted percentiles from histogram
    cum = np.cumsum(counts)
    def percentile_from_hist(p):
        idx = np.searchsorted(cum, total * p / 100.0)
        return 10**bin_centers[min(idx, len(bin_centers)-1)]

    median_val = percentile_from_hist(50)
    q99_val = percentile_from_hist(99)
    q999_val = percentile_from_hist(99.9)

    # Underflow/overflow
    underflow = int(pooled_hist[sub][0])
    overflow = int(pooled_hist[sub][-1])
    pct_below_1e7 = 100 * underflow / (total + underflow + overflow)

    # Count elements below thresholds
    idx_1e6 = np.searchsorted(bin_centers, -6.0)
    idx_1e5 = np.searchsorted(bin_centers, -5.0)
    pct_below_1e6 = 100 * (counts[:idx_1e6].sum() + underflow) / (total + underflow + overflow)
    pct_below_1e5 = 100 * (counts[:idx_1e5].sum() + underflow) / (total + underflow + overflow)

    sub_stats[sub] = {
        'median': median_val, 'q99': q99_val, 'q999': q999_val,
        'q999_over_median': q999_val / max(median_val, 1e-30),
        'pct_below_1e6': pct_below_1e6,
        'pct_below_1e5': pct_below_1e5,
        'overflow_pct': 100 * overflow / (total + underflow + overflow),
        'underflow_pct': pct_below_1e7,
        'total': total + underflow + overflow,
    }

subs = list(sub_stats.keys())
x = np.arange(len(subs))

# q99.9 / median ratio
ax = axes[0]
vals = [sub_stats[s]['q999_over_median'] for s in subs]
bars = ax.bar(x, vals, color=[colors[s] for s in subs])
ax.set_xticks(x); ax.set_xticklabels(subs)
ax.set_ylabel('q99.9 / median'); ax.set_title('Tail Stretch (q99.9 / median)')
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{v:.0f}', ha='center', va='bottom', fontsize=9)

# Overflow percentage (> 10^-2)
ax = axes[1]
vals = [sub_stats[s]['overflow_pct'] for s in subs]
bars = ax.bar(x, vals, color=[colors[s] for s in subs])
ax.set_xticks(x); ax.set_xticklabels(subs)
ax.set_ylabel('% of elements'); ax.set_title('Fraction > 10⁻² (overflow)')
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{v:.2f}%', ha='center', va='bottom', fontsize=9)

# Near-zero fractions
ax = axes[2]
w = 0.3
vals6 = [sub_stats[s]['pct_below_1e6'] for s in subs]
vals5 = [sub_stats[s]['pct_below_1e5'] for s in subs]
ax.bar(x-w/2, vals6, w, color=[colors[s] for s in subs], alpha=0.6, label='< 10⁻⁶')
ax.bar(x+w/2, vals5, w, color=[colors[s] for s in subs], alpha=0.9, label='< 10⁻⁵')
ax.set_xticks(x); ax.set_xticklabels(subs)
ax.set_ylabel('% of elements'); ax.set_title('Near-Zero Gradients')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grad_heavy_tail_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved grad_heavy_tail_metrics.png")

# ── Summary ──
print(f"\n{'='*90}")
print(f"  Gradient Distribution Summary (1 epoch, {step} steps, all layers)")
print(f"{'='*90}")
print(f"{'Sub':>5} {'Total':>14} {'median':>10} {'q99':>10} {'q99.9':>10} "
      f"{'q999/med':>10} {'%<1e-6':>8} {'%<1e-5':>8} {'%>1e-2':>8}")
for sub in sublayer_types:
    if sub not in sub_stats: continue
    ss = sub_stats[sub]
    print(f"{sub:>5} {ss['total']:>14,} {ss['median']:>10.2e} {ss['q99']:>10.2e} "
          f"{ss['q999']:>10.2e} {ss['q999_over_median']:>10.0f} "
          f"{ss['pct_below_1e6']:>7.1f}% {ss['pct_below_1e5']:>7.1f}% {ss['overflow_pct']:>7.2f}%")

print(f"\nAll results saved to {OUT_DIR}/")
print("Done.")

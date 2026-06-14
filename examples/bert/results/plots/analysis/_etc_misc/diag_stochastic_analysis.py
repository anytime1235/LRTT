#!/usr/bin/env python3
"""
Analysis: Why LRTT + LinearStep + StochasticCompressed underperforms nwd
========================================================================
Uses existing optuna trial data (no GPU needed).

Panels:
  1. F1 distribution: all stochastic trials + nwd/ideal references
  2. Per-epoch convergence: top-5 stochastic + reference lines
  3. n_pulses vs F1 scatter
  4. fast_lr × ab_dw_min 2D landscape with F1 colormap
  5. Noise hypothesis: B/C small-scale comparison
  6. Theoretical: gradient corruption mechanism

Usage:
    python diag_stochastic_analysis.py
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULT_DIR = Path(__file__).parent / "results" / "optuna_bert_squad_lrtt"

# ── Reference values (from external optuna studies) ──────────────────────────
NWD_BEST_F1      = 86.30   # nwd (none_with_device) best: full dataset, 5ep
IDEAL_BEST_F1    = 90.23   # ideal device best (approximate, from logs)
BL               = 31      # burst length for LinearStepDevice
W_BOUND          = 0.6

# ── Parse stochastic trial data ───────────────────────────────────────────────

def parse_stochastic_log(log_path):
    """Parse optuna journal log → {trial_id: {params, epochs, final_f1}}"""
    trials = {}
    params_by_trial = {}

    with open(log_path) as f:
        for line in f:
            d = json.loads(line)
            tid = d.get('trial_id')
            op  = d['op_code']

            if op == 5 and tid is not None:          # param suggestion
                if tid not in params_by_trial:
                    params_by_trial[tid] = {}
                params_by_trial[tid][d['param_name']] = d['param_value_internal']

            elif op == 7 and tid is not None:         # intermediate (per-epoch)
                if tid not in trials:
                    trials[tid] = {'epochs': []}
                trials[tid]['epochs'].append((d['step'], d['intermediate_value']))

            elif op == 6 and tid is not None:         # trial complete
                vals = d.get('values') or []
                if vals and vals[0] is not None:
                    if tid not in trials:
                        trials[tid] = {'epochs': []}
                    trials[tid]['final'] = vals[0]

    # Merge params; infer final from epoch max if missing
    result = {}
    for tid, t in trials.items():
        if not t['epochs']:
            continue
        epochs_sorted = [v for _, v in sorted(t['epochs'])]
        final = t.get('final', max(epochs_sorted))
        result[tid] = {
            'params':  params_by_trial.get(tid, {}),
            'epochs':  epochs_sorted,
            'final_f1': final,
        }
    return result


def n_pulses(fast_lr, ab_dw_min, bl=BL):
    if ab_dw_min <= 0:
        return bl
    return min(round(fast_lr / ab_dw_min), bl)


# ── Load data ─────────────────────────────────────────────────────────────────

stoch_log = RESULT_DIR / "optuna_bert_squad_lrtt_bs48_adam_hybrid_nowd_nomom_nonest_set_linearstep_fwinj_perfect_noos_fica_qkvo_5ep.log"
stoch_trials = parse_stochastic_log(stoch_log)

# Also load from all_trials.json for additional data
with open(RESULT_DIR / "all_trials.json") as f:
    extra_trials = json.load(f)

# Merge: use JSON for final F1, log for epochs
for t in extra_trials:
    tid = t['trial']
    if t.get('value') and tid not in stoch_trials:
        stoch_trials[tid] = {
            'params': t['params'],
            'epochs': [],
            'final_f1': t['value'],
        }

# Load small-scale noise comparison (our GPU experiment)
noise_cmp_path = RESULT_DIR / "diag_noise_comparison_results.json"
noise_cmp = {}
if noise_cmp_path.exists():
    with open(noise_cmp_path) as f:
        noise_cmp = json.load(f)

# ── Prepare arrays ────────────────────────────────────────────────────────────

all_f1    = np.array([t['final_f1'] for t in stoch_trials.values()])
all_np    = np.array([n_pulses(t['params'].get('fast_lr', 0), t['params'].get('ab_dw_min', 1e-9))
                      for t in stoch_trials.values()])
all_flr   = np.array([t['params'].get('fast_lr', 0)   for t in stoch_trials.values()])
all_dw    = np.array([t['params'].get('ab_dw_min', 0)  for t in stoch_trials.values()])
all_te    = np.array([int(t['params'].get('transfer_every', 1)) for t in stoch_trials.values()])

# Top-N trials with per-epoch data
top_with_epochs = sorted(
    [(t['final_f1'], tid, t) for tid, t in stoch_trials.items() if t['epochs']],
    reverse=True
)[:6]

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "Why LRTT + LinearStep + StochasticCompressed underperforms nwd\n"
    "[BERT SQuAD, bs=48, 5 epochs, full dataset]",
    fontsize=12, fontweight='bold'
)

STOCH_COLOR = "crimson"
NWD_COLOR   = "royalblue"
IDEAL_COLOR = "green"

# ─── Panel 1: F1 distribution ────────────────────────────────────────────────
ax = axes[0, 0]
f1_sorted = np.sort(all_f1)[::-1]
ax.bar(range(len(f1_sorted)), f1_sorted, color=STOCH_COLOR, alpha=0.7, label="Stochastic trials")
ax.axhline(NWD_BEST_F1,  color=NWD_COLOR,   linestyle='--', linewidth=2,
           label=f"nwd best = {NWD_BEST_F1:.1f}%")
ax.axhline(max(all_f1),  color=STOCH_COLOR,  linestyle=':',  linewidth=1.5,
           label=f"Stochastic best = {max(all_f1):.1f}%")
ax.fill_between([-0.5, len(f1_sorted)-0.5],
                max(all_f1), NWD_BEST_F1,
                alpha=0.1, color=NWD_COLOR, label=f"Gap = {NWD_BEST_F1-max(all_f1):.1f}%")
ax.set_xlabel("Trial rank")
ax.set_ylabel("Best F1 (%)")
ax.set_title("F1 distribution: all stochastic trials")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 95)

# ─── Panel 2: Per-epoch convergence curves ────────────────────────────────────
ax = axes[0, 1]
colors_top = plt.cm.Reds(np.linspace(0.5, 0.9, len(top_with_epochs)))
for i, (f1, tid, t) in enumerate(top_with_epochs):
    eps = t['epochs']
    np_val = n_pulses(t['params'].get('fast_lr',0), t['params'].get('ab_dw_min',1e-9))
    te     = int(t['params'].get('transfer_every', 1))
    ax.plot(range(1, len(eps)+1), eps, 'o-',
            color=colors_top[i], linewidth=1.8, markersize=5,
            label=f"Trial {tid}: n_pulses={np_val}, te={te} → {f1:.1f}%")

ax.axhline(NWD_BEST_F1, color=NWD_COLOR, linestyle='--', linewidth=2,
           label=f"nwd best = {NWD_BEST_F1:.1f}%")

# Add ceiling annotation
ax.annotate(f"Stochastic ceiling ≈ {max(all_f1):.1f}%",
            xy=(5, max(all_f1)), xytext=(3.5, max(all_f1)+1.5),
            fontsize=8, color=STOCH_COLOR,
            arrowprops=dict(arrowstyle='->', color=STOCH_COLOR, lw=1))
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 (%)")
ax.set_title("Convergence: top stochastic trials vs nwd")
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0, 92)

# ─── Panel 3: n_pulses vs F1 ─────────────────────────────────────────────────
ax = axes[0, 2]
saturated = all_np >= BL
ax.scatter(all_np[~saturated], all_f1[~saturated],
           c='darkorange', s=60, alpha=0.8, label=f"Under-BL (< {BL} pulses)", zorder=3)
ax.scatter(all_np[saturated],  all_f1[saturated],
           c=STOCH_COLOR, s=80, marker='*', alpha=0.9,
           label=f"BL-saturated (= {BL} pulses)", zorder=3)

ax.axvline(BL, color='gray', linestyle='--', linewidth=1.5, label=f"BL={BL} saturation")
ax.axhline(NWD_BEST_F1, color=NWD_COLOR, linestyle='--', linewidth=1.5,
           label=f"nwd best = {NWD_BEST_F1:.1f}%")
# nwd marker
ax.scatter([BL+2], [NWD_BEST_F1], c=NWD_COLOR, s=200, marker='D', zorder=5,
           label="nwd (FP, no quantization)")
ax.annotate("nwd", xy=(BL+2, NWD_BEST_F1), xytext=(BL+4, NWD_BEST_F1+1),
            fontsize=9, color=NWD_COLOR, fontweight='bold')

ax.set_xlabel("Estimated n_pulses = round(fast_lr / ab_dw_min)")
ax.set_ylabel("Best F1 (%)")
ax.set_title(f"n_pulses vs F1\n(BL={BL}: all updates at max strength → no direction info)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 92)

# ─── Panel 4: fast_lr × ab_dw_min landscape ──────────────────────────────────
ax = axes[1, 0]
valid = (all_flr > 0) & (all_dw > 0) & (all_f1 > 10)
sc = ax.scatter(all_flr[valid], all_dw[valid] * 1e3,
                c=all_f1[valid], cmap='RdYlGn', vmin=40, vmax=85,
                s=80, alpha=0.9, zorder=3)
plt.colorbar(sc, ax=ax, label="F1 (%)")

# Iso-n_pulses lines
flr_range = np.logspace(-2, 1.5, 200)
for npv, ls, lbl in [(1, ':', '1 pulse (dead zone)'),
                      (5, '--', '5 pulses'),
                      (20, '-', '20 pulses (optimal)'),
                      (31, '-', '31 pulses (BL saturated)')]:
    dw_iso = flr_range / npv * 1e3  # convert to milli
    mask = (dw_iso > 0.01) & (dw_iso < 200)
    ax.plot(flr_range[mask], dw_iso[mask], ls, color='gray', linewidth=1, alpha=0.6)
    if np.any(mask):
        mid = len(flr_range[mask]) // 2
        ax.annotate(f'n={npv}', xy=(flr_range[mask][mid], dw_iso[mask][mid]),
                    fontsize=7, color='gray', alpha=0.8)

# Mark nwd best
ax.scatter([0.04823], [0.001981 * 1e3], c=NWD_COLOR, s=200, marker='D',
           zorder=5, label='nwd best params')
ax.annotate('nwd best\n(FP)', xy=(0.04823, 0.001981*1e3),
            xytext=(0.1, 0.003*1e3), fontsize=8, color=NWD_COLOR,
            arrowprops=dict(arrowstyle='->', color=NWD_COLOR, lw=1))

ax.set_xlabel("fast_lr")
ax.set_ylabel("ab_dw_min (×10⁻³)")
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title("Parameter landscape: fast_lr × ab_dw_min\n(iso-n_pulses lines shown)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, which='both')

# ─── Panel 5: Noise hypothesis test ──────────────────────────────────────────
ax = axes[1, 1]

exp_labels = ['A\nnwd\n(FP+clip)', 'B\nstoch\n+noise', 'C\nstoch\nno noise\n(std=dtod=0)', 'D\nstoch\nstoch-best\nparams']
exp_ids    = ['A', 'B', 'C', 'D']
exp_colors = ['royalblue', 'crimson', 'purple', 'darkorange']

if noise_cmp:
    f1_vals = [noise_cmp.get(e, {}).get('best_f1', 0) for e in exp_ids]
    bars = ax.bar(range(4), f1_vals, color=exp_colors, alpha=0.8, edgecolor='black', width=0.5)
    for bar, val in zip(bars, f1_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(exp_labels, fontsize=8)
    ylim_min = max(0, min(v for v in f1_vals if v > 0) - 5) if any(v > 0 for v in f1_vals) else 0
    ax.set_ylim(ylim_min, max(f1_vals) + 5 if f1_vals else 60)
    ax.set_ylabel("Best F1 (%)")
else:
    ax.text(0.5, 0.5, "No small-scale\nexperiment data", ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='gray')

ax.set_title(
    "Noise hypothesis test (subset=8000, 5ep)\n"
    "B≈C: LinearStep-specific noise (dw_min_std/dtod)\n"
    "NOT the main culprit — quantization itself is"
)
ax.grid(True, alpha=0.3, axis='y')

# Annotation
if noise_cmp:
    bc_vals = [noise_cmp.get('B', {}).get('best_f1', 0),
               noise_cmp.get('C', {}).get('best_f1', 0)]
    if all(v > 0 for v in bc_vals):
        diff = abs(bc_vals[0] - bc_vals[1])
        ax.annotate(f'B-C diff = {diff:.2f}%\n(noise negligible\nat this scale)',
                    xy=(1.5, (bc_vals[0]+bc_vals[1])/2),
                    xytext=(1.5, (bc_vals[0]+bc_vals[1])/2 - 2),
                    ha='center', fontsize=8, color='gray',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# ─── Panel 6: Theoretical explanation ────────────────────────────────────────
ax = axes[1, 2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Why stochastic < nwd: mechanism", fontsize=11)

text_lines = [
    ("LRTT transfer: C += A@B  (every transfer_every steps)", (0.5, 9.5), 9, 'black', 'left'),
    ("A, B reset to small random after transfer", (0.5, 9.0), 8.5, 'gray', 'left'),
    ("", (0, 8.6), 8.5, 'black', 'left'),
    ("nwd (none_with_device):", (0.5, 8.3), 10, NWD_COLOR, 'left'),
    ("  A[i,k] += -fast_lr × grad[i,k]  (exact FP)", (0.5, 7.8), 9, NWD_COLOR, 'left'),
    ("  A@B at transfer = accurate gradient signal", (0.5, 7.3), 9, NWD_COLOR, 'left'),
    ("  → C learns correct direction each transfer", (0.5, 6.8), 9, NWD_COLOR, 'left'),
    ("  → Best F1 = 86.3%", (0.5, 6.3), 9.5, NWD_COLOR, 'left'),
    ("", (0, 5.9), 8.5, 'black', 'left'),
    ("StochasticCompressed (LinearStepDevice):", (0.5, 5.6), 10, STOCH_COLOR, 'left'),
    ("  n_pulses = round(fast_lr / dw_min)  [capped at BL=31]", (0.5, 5.1), 9, STOCH_COLOR, 'left'),
    ("  A[i,k] += ±dw_min × n_pulses  (quantized, noisy)", (0.5, 4.6), 9, STOCH_COLOR, 'left'),
    ("  Gradient direction corrupted by:", (0.5, 4.1), 9, STOCH_COLOR, 'left'),
    ("    • Stochastic rounding (even if dw_min_std=dtod=0)", (0.7, 3.6), 8.5, STOCH_COLOR, 'left'),
    ("    • BL saturation → sign-only update (no magnitude)", (0.7, 3.1), 8.5, STOCH_COLOR, 'left'),
    ("  A@B at transfer = degraded gradient signal", (0.5, 2.6), 9, STOCH_COLOR, 'left'),
    ("  → C learns with less precision each transfer", (0.5, 2.1), 9, STOCH_COLOR, 'left'),
    ("  → Stochastic ceiling ≈ 80.3%", (0.5, 1.6), 9.5, STOCH_COLOR, 'left'),
    ("  (LinearStep noise (dw_min_std/dtod) = secondary effect)", (0.5, 0.9), 8, 'gray', 'left'),
]

for text, pos, size, color, ha in text_lines:
    if text == "":
        ax.axhline(pos[1], xmin=0.03, xmax=0.97, color='lightgray', linewidth=0.8)
    else:
        ax.text(pos[0], pos[1], text, fontsize=size, color=color, ha=ha, va='top',
                fontfamily='monospace' if '→' not in text else 'sans-serif')

# ─── Final layout ─────────────────────────────────────────────────────────────
plt.tight_layout()
save_path = RESULT_DIR / "diag_stochastic_analysis.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {save_path}")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Stochastic trials: {len(stoch_trials)}")
print(f"F1 range: {all_f1.min():.1f}% ~ {all_f1.max():.1f}%")
print(f"nwd best:       {NWD_BEST_F1:.1f}%")
print(f"Stoch ceiling:  {all_f1.max():.1f}%")
print(f"Gap:            {NWD_BEST_F1 - all_f1.max():.1f}%")
print(f"\nBL-saturated trials: {saturated.sum()} / {len(saturated)}")
print(f"Sub-BL trials:       {(~saturated).sum()} / {len(saturated)}")
print(f"Max F1 (saturated):  {all_f1[saturated].max():.1f}%  (n={saturated.sum()})")
print(f"Max F1 (sub-BL):     {all_f1[~saturated].max():.1f}%  (n={(~saturated).sum()})")

if noise_cmp:
    print("\n=== Small-scale noise comparison (subset=8000) ===")
    for eid in 'ABCD':
        r = noise_cmp.get(eid, {})
        print(f"  {eid} ({r.get('label','')}): F1={r.get('best_f1',0):.2f}%")

if __name__ == "__main__":
    pass

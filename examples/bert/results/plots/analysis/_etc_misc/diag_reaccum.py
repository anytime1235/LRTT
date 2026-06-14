#!/usr/bin/env python3
"""
Analytical visualization: A/B re-accumulation dynamics after reset.

Fixed baseline: nwd-best fast_lr = 0.04823 (and transfer_every=4).
Fixed:  BL = 31  (max pulses per weight update).
Varied: bit depth 5..16  ->  dw_min = 2*W_BOUND / 2^bits = 1.2 / 2^bits.

Comparison:
  nwd:   delta_w = fast_lr  (FP, unaffected by BL or dw_min)
  stoch: n_pulses = min(BL, round(fast_lr / dw_min))
         delta_w  = n_pulses * dw_min

With AnalogAdam, fast_lr is the effective per-step update magnitude (Adam
normalises away gradient magnitude), so n_pulses = round(fast_lr/dw_min).
Saturation (n_pulses = BL=31) -> sign-SGD, magnitude info lost.

The actual study used dw_min=0.001981 -> n_pulses=24, marked as reference.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Params ────────────────────────────────────────────────────────────────────
BL             = 31
W_BOUND        = 0.6
A0             = 0.01 * 0.10      # REINIT_GAIN * typical_init ~ 0.001

FAST_LR        = 0.04823          # nwd-best (baseline for all comparisons)
TRANSFER_EVERY = 4
STUDY_DW_MIN   = 0.001981         # actual dw_min used in the study -> n=24

BITS    = [5, 6, 7, 8, 9, 10, 12, 14, 16]
DW_MINS = [2 * W_BOUND / (2 ** b) for b in BITS]   # 1.2 / 2^bits

N_SHOW  = 60

# ── Core helpers ──────────────────────────────────────────────────────────────

def n_pulses(dw_min):
    return min(BL, round(FAST_LR / max(dw_min, 1e-12)))

def effective_dw(dw_min):
    return n_pulses(dw_min) * dw_min

def simulate(dw_min, n_steps, mode="stoch"):
    dw = FAST_LR if mode == "nwd" else effective_dw(dw_min)
    a = b = A0
    ab = [A0 * A0]
    for _ in range(n_steps):
        a = min(W_BOUND, a + dw)
        b = min(W_BOUND, b + dw)
        ab.append(a * b)
    return np.arange(n_steps + 1), np.array(ab)

# ── Colour map ────────────────────────────────────────────────────────────────
CMAP = plt.cm.turbo
COLS = [CMAP(0.05 + 0.90 * i / (len(BITS) - 1)) for i in range(len(BITS))]

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"fast_lr={FAST_LR}  BL={BL}  transfer_every={TRANSFER_EVERY}\n")
print(f"{'bits':>5}  {'dw_min':>10}  {'n_pulses':>10}  {'eff_dw':>10}  {'efficiency':>12}")
for bits, dw in zip(BITS, DW_MINS):
    n  = n_pulses(dw)
    dw_eff = effective_dw(dw)
    eff    = dw_eff / FAST_LR
    sat    = " <- SAT" if n == BL else ""
    print(f"  {bits:>3}b  {dw:>10.6f}  {n:>10d}  {dw_eff:>10.6f}  {eff:>11.3f}{sat}")
n_study = n_pulses(STUDY_DW_MIN)
print(f"\n  [study] dw_min={STUDY_DW_MIN}  ->  {np.log2(1.2/STUDY_DW_MIN):.1f}-bit equiv  "
      f"n_pulses={n_study}  efficiency={effective_dw(STUDY_DW_MIN)/FAST_LR:.3f}")

# ── Precompute nwd reference ──────────────────────────────────────────────────
steps, ab_nwd = simulate(STUDY_DW_MIN, N_SHOW, "nwd")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)

# ── Panel 0,0: n_pulses vs bit depth ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
n_vals = [n_pulses(dw) for dw in DW_MINS]
ax.plot(BITS, n_vals, lw=2, color="#F44336", marker='o', ms=6, label="stoch n_pulses")
ax.axhline(BL, color="red", lw=1.5, ls=":", label=f"BL={BL} (saturation)")
# study reference
eq_bits_study = np.log2(2 * W_BOUND / STUDY_DW_MIN)
ax.axvline(eq_bits_study, color="black", lw=1.5, ls="--",
           label=f"study dw_min ({eq_bits_study:.1f}b, n={n_study})")
ax.scatter([eq_bits_study], [n_study], color="black", s=80, zorder=5)
ax.set_xlabel("bit depth  (dw_min = 1.2 / 2^bits)")
ax.set_ylabel("n_pulses  (capped at BL=31)")
ax.set_title(f"n_pulses vs bit depth\n(fast_lr={FAST_LR}, BL={BL})")
ax.set_xticks(BITS); ax.set_ylim(0, BL + 5)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Panel 0,1: update efficiency vs bit depth ─────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
effs = [effective_dw(dw) / FAST_LR for dw in DW_MINS]
ax.plot(BITS, effs, lw=2, color="#F44336", marker='o', ms=6, label="stoch efficiency")
ax.axhline(1.0, color="black", lw=2, ls="-", label="nwd (perfect = 1.0)")
ax.axvline(eq_bits_study, color="black", lw=1.5, ls="--",
           label=f"study ({eq_bits_study:.1f}b, eff={effective_dw(STUDY_DW_MIN)/FAST_LR:.3f})")
ax.scatter([eq_bits_study], [effective_dw(STUDY_DW_MIN)/FAST_LR], color="black", s=80, zorder=5)
ax.set_xlabel("bit depth")
ax.set_ylabel("effective_dw / fast_lr")
ax.set_title(f"Update efficiency vs bit depth\n(1.0 = matches nwd, BL={BL})")
ax.set_xticks(BITS); ax.set_ylim(0, 1.15)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# annotate saturation threshold bit
dw_sat_thresh = FAST_LR / BL
bit_sat_thresh = np.log2(2 * W_BOUND / dw_sat_thresh)
ax.axvline(bit_sat_thresh, color="red", lw=1, ls=":",
           label=f"SAT threshold ({bit_sat_thresh:.1f}b)")
ax.legend(fontsize=7)

# ── Panel 0,2: transfer signal at 1st transfer vs bit depth ──────────────────
ax = fig.add_subplot(gs[0, 2])
nwd_ts = ab_nwd[TRANSFER_EVERY]
ts_vals = []
for dw in DW_MINS:
    _, ab_s = simulate(dw, TRANSFER_EVERY, "stoch")
    ts_vals.append(ab_s[TRANSFER_EVERY])
ts_rel = [v / (nwd_ts + 1e-12) for v in ts_vals]

bars = ax.bar(BITS, ts_rel, color=COLS, edgecolor='black', alpha=0.85)
for bar, v, bits, dw in zip(bars, ts_rel, BITS, DW_MINS):
    n = n_pulses(dw)
    label = "S" if n == BL else str(n)
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            label, ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.axhline(1.0, color="black", lw=1.5, ls="--", label="nwd = 1.0")
ax.axvline(eq_bits_study, color="black", lw=1.5, ls="--",
           label=f"study ({eq_bits_study:.1f}b)")
ax.set_xlabel("bit depth"); ax.set_ylabel("stoch transfer signal / nwd")
ax.set_title(f"Transfer signal at step {TRANSFER_EVERY}\n(number = n_pulses, S = saturated)")
ax.set_xticks(BITS); ax.set_ylim(0, 1.15)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

# ── Row 1: Re-accumulation curves ─────────────────────────────────────────────
ax_ab  = fig.add_subplot(gs[1, :2])
ax_rel = fig.add_subplot(gs[1, 2])

ax_ab.plot(steps, ab_nwd, color="black", lw=2.5, label="nwd (FP)", zorder=5)

for bits, dw, col in zip(BITS, DW_MINS, COLS):
    n   = n_pulses(dw)
    sat = " SAT" if n == BL else f" n={n}"
    _, ab_s = simulate(dw, N_SHOW, "stoch")
    ax_ab.plot(steps, ab_s, color=col, lw=1.5, ls="--", label=f"{bits}b{sat}")
    ax_rel.plot(steps, ab_s / (ab_nwd + 1e-12),
                color=col, lw=1.5, ls="--", label=f"{bits}b{sat}")

# study dw_min highlight
_, ab_study = simulate(STUDY_DW_MIN, N_SHOW, "stoch")
ax_ab.plot(steps, ab_study, color="#00C853", lw=2.5, ls="-.",
           label=f"study dw={STUDY_DW_MIN} (n={n_study})")
ax_rel.plot(steps, ab_study / (ab_nwd + 1e-12),
            color="#00C853", lw=2.5, ls="-.", label=f"study n={n_study}")

for ax_ in (ax_ab, ax_rel):
    ax_.axvline(TRANSFER_EVERY, color="k", ls=":", lw=1.5, label="transfer")
    ax_.legend(fontsize=7, ncol=2); ax_.grid(True, alpha=0.3)
    ax_.set_xlabel("steps after reset")

ax_ab.set_ylabel("A×B  (transfer signal)")
ax_ab.set_title(f"A@B accumulation after reset — all bit depths vs nwd\n"
                f"fast_lr={FAST_LR}, transfer_every={TRANSFER_EVERY}, BL={BL}")

ax_rel.set_ylabel("stoch A@B / nwd A@B")
ax_rel.set_ylim(0, 1.15)
ax_rel.axhline(1.0, color="black", lw=1, ls="-", alpha=0.4)
ax_rel.set_title(f"Relative transfer signal\n(1.0 = nwd)")

# ── Row 2: multiple transfer cycles ───────────────────────────────────────────
ax_cyc = fig.add_subplot(gs[2, :2])
ax_cum = fig.add_subplot(gs[2, 2])

N_CYCLES = 8
cum_nwd  = 0.0
cum_stoch = {bits: 0.0 for bits in BITS}
cum_study = 0.0
cum_nwd_hist    = []
cum_study_hist  = []
cum_stoch_hists = {bits: [] for bits in BITS}

for cycle in range(N_CYCLES):
    # nwd: what C accumulates each cycle
    _, ab_c = simulate(STUDY_DW_MIN, TRANSFER_EVERY, "nwd")
    ts_nwd = ab_c[TRANSFER_EVERY]
    cum_nwd += ts_nwd
    cum_nwd_hist.append(cum_nwd)

    # study dw_min stoch
    _, ab_c = simulate(STUDY_DW_MIN, TRANSFER_EVERY, "stoch")
    cum_study += ab_c[TRANSFER_EVERY]
    cum_study_hist.append(cum_study)

    for bits, dw in zip(BITS, DW_MINS):
        _, ab_c = simulate(dw, TRANSFER_EVERY, "stoch")
        cum_stoch[bits] += ab_c[TRANSFER_EVERY]
        cum_stoch_hists[bits].append(cum_stoch[bits])

cyc_x = np.arange(1, N_CYCLES + 1)
ax_cyc.plot(cyc_x, cum_nwd_hist, color="black", lw=2.5, label="nwd (FP)", zorder=5)
ax_cyc.plot(cyc_x, cum_study_hist, color="black", lw=2, ls="-.",
            label=f"study n={n_study}", zorder=4)
for bits, col in zip(BITS, COLS):
    n = n_pulses(DW_MINS[BITS.index(bits)])
    sat = " SAT" if n == BL else f" n={n}"
    ax_cyc.plot(cyc_x, cum_stoch_hists[bits], color=col, lw=1.5, ls="--",
                label=f"{bits}b{sat}")

ax_cyc.set_xlabel("transfer cycle"); ax_cyc.set_ylabel("cumulative transfer signal to C")
ax_cyc.set_title(f"Cumulative C accumulation over {N_CYCLES} transfer cycles\n"
                 f"(proportional to learning progress)")
ax_cyc.legend(fontsize=7, ncol=2); ax_cyc.grid(True, alpha=0.3)

# Final cumulative relative to nwd
final_nwd = cum_nwd_hist[-1]
final_study_rel = cum_study_hist[-1] / (final_nwd + 1e-12)
final_stoch_rel = [cum_stoch_hists[b][-1] / (final_nwd + 1e-12) for b in BITS]
bars2 = ax_cum.bar(BITS, final_stoch_rel, color=COLS, edgecolor='black', alpha=0.85)
for bar, v, bits, dw in zip(bars2, final_stoch_rel, BITS, DW_MINS):
    n = n_pulses(dw)
    lbl = "S" if n == BL else str(n)
    ax_cum.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                lbl, ha='center', va='bottom', fontsize=8, fontweight='bold')
ax_cum.axhline(1.0, color="black", lw=1.5, ls="--", label="nwd = 1.0")
ax_cum.axhline(final_study_rel, color="black", lw=1.5, ls="-.",
               label=f"study n={n_study} ({final_study_rel:.3f})")
ax_cum.set_xlabel("bit depth"); ax_cum.set_ylabel("cumulative signal / nwd")
ax_cum.set_title(f"Final C accumulation relative to nwd\nafter {N_CYCLES} cycles")
ax_cum.set_xticks(BITS); ax_cum.set_ylim(0, 1.15)
ax_cum.legend(fontsize=7); ax_cum.grid(True, alpha=0.3, axis='y')

# ── Save ──────────────────────────────────────────────────────────────────────
fig.suptitle(
    f"A/B Re-accumulation: nwd (FP) vs StochasticCompressed\n"
    f"Baseline: nwd-best fast_lr={FAST_LR}, transfer_every={TRANSFER_EVERY}  |  "
    f"BL={BL} fixed  |  dw_min = 1.2 / 2^bits",
    fontsize=12, fontweight="bold"
)
out_path = Path("results/optuna_bert_squad_lrtt/diag_reaccum.png")
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"\nSaved -> {out_path}")

#!/usr/bin/env python3
"""
LRTT LinearStep Transfer Simulation
=====================================
Numpy simulation matching actual LRTT BERT-SQuAD experiment structure:

  - Weight = C + A@B  (C: slow tile, A/B: fast analog tiles)
  - C is updated ONLY via transfer (no direct gradient to C)
  - A, B receive stochastic gradient updates, then reset after transfer
  - Transfer every TE steps: C += A@B, A/B reset to small random
  - Gradient is RMS-normalised before apply (mimics AdamAnalog behaviour)

Analyses:
  A. Convergence with transfer (ideal vs nwd vs stochastic)
  B. Transfer quality: how accurately does each update mode copy info to C?
  C. Noise floor vs fast_lr  (= n_pulses per step)
  D. Noise floor vs transfer_every

Device params matching LinearStepDevice defaults + best-trial fica value:
  ab_dw_min=0.01368 (best trial), dw_min_std=0.3, dw_min_dtd=0.3,
  w_bound=±0.6, BL=31
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# ── Device params ──────────────────────────────────────────────────────────────
AB_DW_MIN  = 0.01368    # fica best-trial ab_dw_min
DW_MIN_STD = 0.3        # cycle-to-cycle noise
DW_MIN_DTD = 0.3        # device-to-device (permanent per element)
W_BOUND    = 0.6
BL         = 31

# ── Simulation settings ────────────────────────────────────────────────────────
D        = 64
RANK     = 16           # rank_exp=4
N_STEPS  = 2000
N_TRIALS = 20
FAST_LR  = 0.2          # default: ~round(0.2/0.01368)=15 pulses per step
TE       = 2            # transfer_every default

SAVE_PATH = Path(
    "/root/LRTT/examples/bert/results/optuna_bert_squad_lrtt/lrtt_linearstep_analysis.png"
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def new_dtod(shape, dw_min, rng):
    return np.clip(
        np.abs(rng.normal(dw_min, dw_min * DW_MIN_DTD, shape)),
        dw_min * 0.05, dw_min * 10.0
    )

def rms_normalize(g, eps=1e-8):
    """RMS-normalise gradient — mimics Adam variance correction."""
    return g / (np.sqrt(np.mean(g ** 2)) + eps)

def upd_stoch(W, neg_G_norm, fast_lr, dtod, rng):
    n     = np.clip(np.round(neg_G_norm * fast_lr / dtod), -BL, BL)
    cycle = 1.0 + rng.normal(0.0, DW_MIN_STD, W.shape)
    return np.clip(W + n * dtod * cycle, -W_BOUND, W_BOUND)

def upd_nwd(W, neg_G_norm, fast_lr):
    return np.clip(W + neg_G_norm * fast_lr, -W_BOUND, W_BOUND)

def upd_fp(W, neg_G_norm, fast_lr):
    return W + neg_G_norm * fast_lr

def make_W_target(d, rank, rng):
    return (rng.standard_normal((d, rank)) * 0.2) @ (rng.standard_normal((rank, d)) * 0.2)

def ab_reset(d, rank, rng):
    return rng.standard_normal((d, rank)) * 0.025, rng.standard_normal((rank, d)) * 0.025


# ══════════════════════════════════════════════════════════════════════════════
# Core runner: LRTT with transfer
#
# C updated ONLY via transfer (C += A@B every TE steps, A/B reset).
# A, B receive RMS-normalised gradient updates (stochastic / nwd / ideal).
# dtod is fixed throughout — hardware property, survives reset.
# ══════════════════════════════════════════════════════════════════════════════

def run_lrtt(mode, n_steps=N_STEPS, te=TE, fast_lr=FAST_LR,
             ab_dw_min=AB_DW_MIN, seed=0):
    rng  = np.random.default_rng(seed)
    W    = make_W_target(D, RANK, rng)
    Wn   = np.linalg.norm(W, 'fro') + 1e-9

    A, B = ab_reset(D, RANK, rng)
    C    = np.zeros((D, D))

    # dtod sampled once — survives A/B reset (hardware permanent variation)
    dtod_A = new_dtod((D, RANK), ab_dw_min, rng) if mode == 's' else None
    dtod_B = new_dtod((RANK, D), ab_dw_min, rng) if mode == 's' else None

    hist = []
    for t in range(n_steps):
        E  = C + A @ B - W
        hist.append(np.linalg.norm(E, 'fro') / Wn)

        gA = rms_normalize(E @ B.T)   # coupling: uses noisy B
        gB = rms_normalize(A.T @ E)   # coupling: uses noisy A

        if   mode == 'fp':  A = upd_fp(A, -gA, fast_lr);            B = upd_fp(B, -gB, fast_lr)
        elif mode == 'nwd': A = upd_nwd(A, -gA, fast_lr);           B = upd_nwd(B, -gB, fast_lr)
        elif mode == 's':   A = upd_stoch(A, -gA, fast_lr, dtod_A, rng); \
                            B = upd_stoch(B, -gB, fast_lr, dtod_B, rng)

        if (t + 1) % te == 0:
            C += A @ B           # transfer: C accumulates A@B
            A, B = ab_reset(D, RANK, rng)   # reset fast tiles
            # dtod_A, dtod_B unchanged (hardware)

    return np.array(hist)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis B — transfer quality
#
# After TE steps of updates, how accurately does A@B represent
# the intended correction (what ideal A@B would have learned)?
#
# transfer_error = ||A@B_actual - A@B_ideal|| / ||A@B_ideal||
# ══════════════════════════════════════════════════════════════════════════════

def run_transfer_quality(n_windows=100, te=TE, fast_lr=FAST_LR,
                         ab_dw_min=AB_DW_MIN, seed=0):
    rng  = np.random.default_rng(seed)
    W    = make_W_target(D, RANK, rng)

    results = {'nwd': [], 's': []}

    for _ in range(n_windows):
        residual = W - np.zeros((D, D))   # simplified: C=0, learn full W in one window

        for mode in ('nwd', 's'):
            rng2 = np.random.default_rng(rng.integers(1e9))
            A, B = ab_reset(D, RANK, rng2)
            A_fp, B_fp = A.copy(), B.copy()
            dtod_A = new_dtod((D, RANK), ab_dw_min, rng2)
            dtod_B = new_dtod((RANK, D), ab_dw_min, rng2)

            for _ in range(te):
                E    = A @ B - residual
                E_fp = A_fp @ B_fp - residual

                gA    = rms_normalize(E    @ B.T);    gB    = rms_normalize(A.T    @ E)
                gA_fp = rms_normalize(E_fp @ B_fp.T); gB_fp = rms_normalize(A_fp.T @ E_fp)

                if mode == 'nwd':
                    A    = upd_nwd(A, -gA, fast_lr);     B    = upd_nwd(B, -gB, fast_lr)
                else:
                    A    = upd_stoch(A, -gA, fast_lr, dtod_A, rng2)
                    B    = upd_stoch(B, -gB, fast_lr, dtod_B, rng2)
                A_fp = upd_fp(A_fp, -gA_fp, fast_lr); B_fp = upd_fp(B_fp, -gB_fp, fast_lr)

            ideal = A_fp @ B_fp
            actual = A @ B
            nrm = np.linalg.norm(ideal, 'fro') + 1e-9
            results[mode].append(np.linalg.norm(actual - ideal, 'fro') / nrm)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

print(f"LRTT transfer simulation  D={D}, rank={RANK}, N_STEPS={N_STEPS}, N_TRIALS={N_TRIALS}")
print(f"  ab_dw_min={AB_DW_MIN}, dw_min_std={DW_MIN_STD}, dw_min_dtd={DW_MIN_DTD}")
print(f"  fast_lr={FAST_LR}, transfer_every={TE}")
print(f"  n_pulses approx = round(fast_lr/ab_dw_min) = {round(FAST_LR/AB_DW_MIN)} (capped at BL={BL})")

print("\n  [A] Convergence with transfer ...")
conv = {m: np.array([run_lrtt(m, seed=i) for i in range(N_TRIALS)])
        for m in ('fp', 'nwd', 's')}

print("  [B] Transfer quality ...")
N_TQ_TRIALS = N_TRIALS
tq_nwd_all, tq_s_all = [], []
for i in range(N_TQ_TRIALS):
    r = run_transfer_quality(n_windows=20, seed=i)
    tq_nwd_all.extend(r['nwd']); tq_s_all.extend(r['s'])

print("  [C] fast_lr sweep ...")
# Keep n_pulses annotation: n = min(round(flr / AB_DW_MIN), BL)
FAST_LR_SWEEP = [0.01, 0.03, 0.07, 0.14, 0.28, 0.42, 0.7, 1.4]
N_TR_C = 10
flr_err = {'s': [], 'nwd': [], 'fp': []}
flr_npulses = []
for flr in FAST_LR_SWEEP:
    flr_npulses.append(min(round(flr / AB_DW_MIN), BL))
    for mode in ('s', 'nwd', 'fp'):
        errs = [run_lrtt(mode, fast_lr=flr, seed=i)[-1] for i in range(N_TR_C)]
        flr_err[mode].append((np.mean(errs), np.std(errs)))

print("  [D] transfer_every sweep ...")
TE_SWEEP = [1, 2, 5, 10, 20, 50]
N_TR_D = 10
te_err = {'s': [], 'nwd': [], 'fp': []}
for te in TE_SWEEP:
    for mode in ('s', 'nwd', 'fp'):
        errs = [run_lrtt(mode, te=te, seed=i)[-1] for i in range(N_TR_D)]
        te_err[mode].append((np.mean(errs), np.std(errs)))

# ── Plot ───────────────────────────────────────────────────────────────────────
steps = np.arange(N_STEPS)
ALPHA = 0.15

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    'LRTT LinearStep: Transfer-based Simulation (matching actual experiment structure)\n'
    f'[D={D}, rank={RANK}, ab_dw_min={AB_DW_MIN}, dw_min_std={DW_MIN_STD}, '
    f'dw_min_dtd={DW_MIN_DTD}, w_bound=+-{W_BOUND}, BL={BL}]\n'
    f'C updated ONLY via transfer (C+=A@B every TE steps, A/B reset). '
    f'Gradient RMS-normalised (mimics AdamAnalog).',
    fontsize=9
)

COLORS = {'fp': 'green', 'nwd': 'steelblue', 's': 'crimson'}
LABELS = {
    'fp':  'Ideal FP  (no device)',
    'nwd': 'none_with_device  (FP + clip)',
    's':   'Stochastic  (current experiment)',
}

def ribbon(ax, data, label, color, ls='-', lw=1.8):
    m, s = data.mean(0), data.std(0)
    ax.semilogy(steps, m, ls, label=label, color=color, linewidth=lw)
    ax.fill_between(steps, np.maximum(m - s, 1e-6), m + s, color=color, alpha=ALPHA)

def errbar(ax, x, res, color, label, fmt='o-'):
    ms = [m for m, _ in res]; ss = [s for _, s in res]
    ax.errorbar(x, ms, yerr=ss, fmt=fmt, color=color, label=label,
                capsize=4, linewidth=1.8)

# ── A. Convergence ─────────────────────────────────────────────────────────────
ax = axes[0, 0]
for m in ('fp', 'nwd', 's'):
    ribbon(ax, conv[m], LABELS[m], COLORS[m])

f_fp  = conv['fp'].mean(0)[-1]
f_nwd = conv['nwd'].mean(0)[-1]
f_s   = conv['s'].mean(0)[-1]
ax.set_xlabel('Training steps')
ax.set_ylabel('Relative error  ||C+A@B - W_target|| / ||W_target||')
ax.set_title(
    f'A. Convergence with transfer  (fast_lr={FAST_LR}, TE={TE})\n'
    f'C updated only via transfer. A/B reset every {TE} steps.\n'
    f'n_pulses ~ {min(round(FAST_LR/AB_DW_MIN), BL)}'
)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.text(0.97, 0.97,
        f'Final error at step {N_STEPS}:\n'
        f'  ideal  : {f_fp:.4f}\n'
        f'  nwd    : {f_nwd:.4f}\n'
        f'  stoch  : {f_s:.4f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# ── B. Transfer quality ─────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.hist(tq_nwd_all, bins=30, color='steelblue', alpha=0.6, label='nwd', density=True)
ax.hist(tq_s_all,   bins=30, color='crimson',   alpha=0.6, label='stochastic', density=True)
ax.axvline(np.mean(tq_nwd_all), color='steelblue', linewidth=2, linestyle='--')
ax.axvline(np.mean(tq_s_all),   color='crimson',   linewidth=2, linestyle='--')
ax.set_xlabel('Transfer error  ||A@B_actual - A@B_ideal|| / ||A@B_ideal||')
ax.set_ylabel('Density')
ax.set_title(
    f'B. Transfer quality per window  (TE={TE} steps)\n'
    'How accurately does A@B after TE steps match ideal FP trajectory?\n'
    'Error in transfer = permanent noise injected into C.'
)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.text(0.97, 0.97,
        f'Mean transfer error:\n'
        f'  nwd   : {np.mean(tq_nwd_all):.3f}\n'
        f'  stoch : {np.mean(tq_s_all):.3f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# ── C. fast_lr sweep ─────────────────────────────────────────────────────────────
ax = axes[1, 0]
x = np.array(FAST_LR_SWEEP)
for m in ('fp', 'nwd', 's'):
    errbar(ax, x, flr_err[m], COLORS[m], LABELS[m],
           fmt='o-' if m != 'fp' else 's--')
ax.axvline(FAST_LR, color='gray', linestyle='--', linewidth=1.2,
           label=f'Panel A default (fast_lr={FAST_LR})')
for xi, np_ in zip(x, flr_npulses):
    ax.annotate(f'{np_}p', (xi, max(flr_err['s'][FAST_LR_SWEEP.index(xi)][0], 1e-4)),
                textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7)
ax.set_xscale('log')
ax.set_xlabel(f'fast_lr  (annotated: n_pulses = min(round(flr/ab_dw_min), BL={BL}))')
ax.set_ylabel(f'Final relative error at step {N_STEPS}')
ax.set_title(
    f'C. Noise floor vs fast_lr  (TE={TE}, ab_dw_min={AB_DW_MIN})\n'
    'Optimal fast_lr = enough pulses without overshooting.\n'
    'stoch: does an optimal exist that approaches nwd?'
)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

# ── D. transfer_every sweep ─────────────────────────────────────────────────────
ax = axes[1, 1]
for m in ('fp', 'nwd', 's'):
    errbar(ax, TE_SWEEP, te_err[m], COLORS[m], LABELS[m],
           fmt='o-' if m != 'fp' else 's--')
ax.axvline(TE, color='gray', linestyle='--', linewidth=1.2,
           label=f'Panel A default (TE={TE})')
ax.set_xlabel('transfer_every  (steps between C += A@B and A/B reset)')
ax.set_ylabel(f'Final relative error at step {N_STEPS}')
ax.set_title(
    f'D. Noise floor vs transfer_every  (fast_lr={FAST_LR})\n'
    'Longer window: A/B learn more per transfer, but dtod noise accumulates.\n'
    'Shorter window: more frequent corrections, but less signal per transfer.'
)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\nFigure saved -> {SAVE_PATH}")
plt.close()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"Convergence (fast_lr={FAST_LR}, TE={TE}, step {N_STEPS}):")
print(f"  ideal  : {conv['fp'].mean(0)[-1]:.4f}")
print(f"  nwd    : {conv['nwd'].mean(0)[-1]:.4f}")
print(f"  stoch  : {conv['s'].mean(0)[-1]:.4f}")
print(f"\nTransfer quality (mean error per window):")
print(f"  nwd    : {np.mean(tq_nwd_all):.4f}")
print(f"  stoch  : {np.mean(tq_s_all):.4f}")
print(f"\nNoise floor vs fast_lr (stoch / nwd, step {N_STEPS}):")
for flr, np_, (ms, _), (mn, _) in zip(FAST_LR_SWEEP, flr_npulses,
                                        flr_err['s'], flr_err['nwd']):
    marker = " <- Panel A" if flr == FAST_LR else ""
    print(f"  fast_lr={flr:.3f} ({np_:2d} pulses): stoch={ms:.4f}  nwd={mn:.4f}{marker}")
print(f"\nNoise floor vs transfer_every (stoch / nwd, step {N_STEPS}):")
for te, (ms, _), (mn, _) in zip(TE_SWEEP, te_err['s'], te_err['nwd']):
    marker = " <- Panel A" if te == TE else ""
    print(f"  TE={te:2d}: stoch={ms:.4f}  nwd={mn:.4f}{marker}")

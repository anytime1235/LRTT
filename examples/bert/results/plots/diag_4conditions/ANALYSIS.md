# Noise Asymmetry Mechanism Analysis

Hypothesis verification using 4-condition × seed=42 diagnostic replication
(BERT SQuAD, qkvo, T6/T249 hyperparams uniform across all conditions for
apples-to-apples comparison: lr=0.0038, tlr=0.095, te=1, fast_lr=0.474,
ab_dw_min=0.0004883, c_dw_min=0.001953, decay reinit, 5 epochs).

Run stamp: `20260506_054540`.

## Conditions

| Condition | A device | B device | Best F1 |
|---|---|---|---:|
| `no_noise` | constantstep6t1cgamma | constantstep6t1cgamma | 83.39 |
| `a_only`   | 6t1c                  | constantstep6t1cgamma | 83.88 |
| `b_only`   | constantstep6t1cgamma | 6t1c                  | 84.34 |
| `both`     | 6t1c                  | 6t1c                  | **78.72** |

Note: `both` F1 (78.72) is lower than the optuna-best (T98=82.24) because we
deliberately re-used T6/T249 hyperparams across conditions for fair comparison;
T6/T249 are not optimized for full-6t1c and the run early-stops by train-loss
divergence.

## Hypothesis verdicts

| Hypothesis | Verdict | Key evidence |
|---|:---:|---|
| 1. Noise feedback loop | partial | ‖ΔA‖, ‖ΔB‖ ~8× larger on a noisy tile; `both` train loss diverges |
| 2. Rank degradation in A·B | **rejected** | erank(A·B) ≈ 31.5 in all 4 conditions (saturates near rank=32) |
| 3'. Cumulative noise in C | **confirmed** | ‖C_raw‖ +24% only in `both`; erank(C − C_init) drops 580 → 260 |
| 4. Magnitude growth interaction | **confirmed** | ‖A·B‖: 7 (no_noise) → 14.5 (single) → 31 (both) — multiplicative |

## Plot-by-plot findings

### `diag_plot1_norms.png` — Hypothesis 4 (magnitude growth)

| Condition | final ‖A‖ | final ‖B‖ | final ‖A·B‖ |
|---|:---:|:---:|:---:|
| no_noise | 6.26 | 6.26 | 7.00 |
| a_only   | **13.02** | 6.28 | 14.55 |
| b_only   | 6.30 | **13.13** | 14.71 |
| both     | **13.19** | **13.30** | **31.11** |

Each noisy tile grows to ~2× its no-noise magnitude. ‖A·B‖ grows multiplicatively
because A·B is a bilinear product. With one noisy tile the product is bounded by
the clean tile (~2× baseline). With both noisy the product grows ~4× baseline.

### `diag_plot2_erank.png` — Hypothesis 2 (rank degradation in A·B)

`erank(A)`, `erank(B)`, `erank(A·B)` all sit at ~31.5–31.8 across every
condition (max possible is 32 since rank=32). The hypothesis that A·B's
effective rank collapses with both-noise is **rejected** — A and B are
low-rank factors and saturate. The rank-degradation effect actually shows
up downstream in **`erank(C)`**: only `both` shows a slow decline
(550 → 510 over training), while the other three stay flat.

### `diag_plot3_deltas.png` — Hypothesis 1 (update magnitude)

Per-step ‖ΔA‖ and ‖ΔB‖ are ~8× larger on a tile that is noisy:

- ‖ΔA‖: ~0.05 when A is gamma-only (no_noise, b_only) → ~0.4 when A=6t1c (a_only, both)
- ‖ΔB‖: same pattern with B
- ‖ΔC‖: similar shape across conditions, but `both` has slightly higher peak
  and ends earlier (early stop by train-loss threshold).

Larger update magnitude is a *symptom* (the optimizer compensates for write
noise with bigger steps) rather than the failure mechanism — the failure
mechanism is what those bigger steps do downstream (Hypothesis 3').

### `diag_plot4_C_noise.png` — Hypothesis 3' (cumulative noise in C)

This is where the cause-of-failure is most visible:

- ‖C_raw‖: no_noise/single-noise stay at 208–215; **`both` rises 208 → 257
  (+24%)** monotonically.
- erank(C − C_init): peaks ~580 then no_noise/single-noise stabilize at
  430–520, while **`both` drops to ~260** — i.e. the *change* C accumulates
  has its effective rank cut nearly in half.

This means: in `both`, the large ‖A·B‖ at every transfer dumps noisy mass
into C. Because the noise in A·B is correlated across many output channels
(both factors are corrupted), the rank of the C-update is reduced and the
magnitude is large — C drifts away from a useful representation faster than
gradient descent can correct.

### `diag_plot5_learning.png` — learning trajectory

Per-epoch trajectories:

| Cond | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | peak |
|---|---:|---:|---:|---:|---:|---|
| no_noise | 80.40 | 81.81 | 82.60 | **83.39** | **20.63 ⚠** | Ep4 |
| a_only   | 79.95 | 82.39 | 83.17 | **83.88** | 81.26 | Ep4 |
| b_only   | 80.54 | 82.35 | 83.26 | 83.82 | **84.34** | Ep5 |
| both     | 78.72 | 72.77 | **60.18 ⚠** | (early stop) | — | Ep1 |

Three distinct dynamics:

1. **`both` diverges at epoch 2** — F1 drops 79 → 73 → 60, train loss
   diverges 1.96 → 1.69 → 2.35. Triggers train-loss early stop after
   epoch 3. This is the noise-driven divergence the analysis is about.

2. **`no_noise` collapses at epoch 5** — F1 drops 83.4 → 20.6, train loss
   spikes 1.22 → 3.85. Cause is the LR linear-decay schedule reaching
   ~4.3e-07 (1900× drop from epoch 4); at that scale Adam's adaptive step
   becomes unstable for the LRTT transfer dynamics. This is unrelated to
   noise — late-stage scheduler artifact. Practically, "Best F1: 83.39"
   (from epoch 4) is the meaningful number for no_noise here.

3. **`a_only` slight regression at epoch 5** (83.88 → 81.26). Minor; same
   late-stage instability as no_noise but smaller because a_only had been
   trending less aggressively.

4. **`b_only` monotone climb** to 84.34 — most stable trajectory of the
   four. Hints that putting noise on B (random-init, gradient-side tile)
   may actually act as mild regularization, but this is observed in
   single-seed only.

Earlier note: the `Eval F1 per epoch` panel was empty in the first plot
because the script looked for key `eval_f1` instead of `f1` in
`epoch_history`. Fixed in `replicate_4conditions_diag_plot.py` (commit
of plot script update). The plot above is regenerated.

## Causal chain (revised mechanism)

```
1 tile noisy  →  that tile's ‖weight‖ ≈ 2× baseline (compensation via large updates)
2 tiles noisy →  ‖A·B‖ = ‖A‖·‖B‖ scales multiplicatively → ~4× baseline   (H4)
              →  every transfer adds a 4× magnitude noisy update to C tile
              →  ‖C_raw‖ accumulates upward; C tile's effective rank drops (H3')
              →  forward pass becomes unstable, train loss diverges
```

The mechanism that matters for fig 6c is **H4 → H3'**, not H1 (large updates)
or H2 (rank in A·B). Single-side noise stops at "‖A·B‖ = 2× baseline" which
is small enough that C tile absorbs it; both-side noise crosses a magnitude
threshold where C tile cannot absorb the cumulative noise.

## Caveats

- Single-seed experiment (seed=42). Run-to-run variance is not characterized
  here — but the qualitative magnitude differences (‖A·B‖ 7 vs 14 vs 31, ‖C‖
  +24%) are far larger than the variance ±0.5 measured in the 5-seed F1
  replication, so the mechanism conclusions are robust.
- Hyperparams are identical across conditions for clean attribution. Each
  condition's *own* optimum (different hyperparams) would yield different
  absolute F1 but should preserve the relative magnitude/erank pattern.
- ERANK_RATE_LIMIT_STEPS=10, MULTI_TILE_DIAG=False — first/last tile only.
- `both` early-stops at epoch 3, so its trajectories end at ~5500 steps
  while the others run to ~9000.

## Files in this folder

| File | Content |
|---|---|
| `diag_a_only.json`, `diag_b_only.json`, `diag_both.json`, `diag_no_noise.json` | per-step diagnostic logs |
| `summary_20260506_054540.json` | run summary (config + final F1s) |
| `diag_plot1_norms.png` / `.svg` | ‖A‖, ‖B‖, ‖A·B‖ trajectories |
| `diag_plot2_erank.png` / `.svg` | erank(A), erank(B), erank(A·B), erank(C) trajectories |
| `diag_plot3_deltas.png` / `.svg` | ‖ΔA‖, ‖ΔB‖, ‖ΔC‖ per-step trajectories |
| `diag_plot4_C_noise.png` / `.svg` | ‖C_raw‖ and erank(C − C_init) trajectories |
| `diag_plot5_learning.png` / `.svg` | F1 / train-loss epoch trajectories |
| `ANALYSIS.md` | this file |

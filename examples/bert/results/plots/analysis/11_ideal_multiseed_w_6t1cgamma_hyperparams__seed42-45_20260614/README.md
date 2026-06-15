## Experiment
constantstepideal multi-seed test — verify whether collapse is device-independent

## Date: 2026-06-14

## Launch
/tmp/run_ideal_multiseed.py (ad-hoc; not in scripts/ yet — copy if needed)

## Hypothesis
If LRTT bilinear instability is a fundamental property (not 6T1C device specific),
collapse should also occur in fully ideal device (no capacitor decay, no drift).

## Setup
- A_DEVICE = B_DEVICE = C_DEVICE = AB_DEVICE = **constantstepideal**
- REINIT_MODE=decay, TRANSFER_METHOD=onehot, LORA_TARGET=qkvo
- LR=0.0038, transfer_lr=0.095, fast_lr=0.474, transfer_every=1, rank=32
- BS=48, 5 epochs, MIN_LR_RATE=0, AUTO_SCALE=none, FORWARD_INJECT=False
- Minimal diag + G coherence

## Variables
seed ∈ {42, 43, 44, 45}

## CRITICAL CAVEAT: hyperparam mismatch
These hyperparams are **optimized for 6t1cgamma device** via optuna trial 250.
The corresponding **ideal-device optuna search shows different optima**:
- Top abml trial 158: LR=3.28e-3, transfer_lr=0.2499, fast_lr=0.70, te=2, abml=9 → F1=85.29
- Top non-abml trial 131: LR=2.31e-3, transfer_lr=312.96, fast_lr=0.0194, te=800 → F1=81.60
- Our config: transfer_lr=0.095 (2.6× too low for abml-optimum), no abml, te=1

→ Failures observed may be due to hyperparam mismatch rather than device behavior.

## Results

| Seed | Best F1 | Best epoch | L11 cascade | Max ||A·B||_L11 | L0 pattern |
|---|---|---|---|---|---|
| 42 | 7.74 | 1 | YES (step 549) | 2611 | stuck (1.38) |
| 43 | 6.80 | 0 | NO (monotone) | 130 | stuck (1.45) |
| 44 | 7.79 | 4 | YES (step 173) | 3460 | stuck (1.26) |
| 45 | 7.20 | 1 | YES (step 185) | 3470 | stuck (1.28) |

**4/4 collapsed** to F1 ~7%. Pattern:
- L0 ALL stuck (||A·B|| ≈ 1.3 flat throughout)
- L11 cascade in 3/4 (early epoch 2-6%); seed 43 monotone divergence

## Threshold check
cos_G_prev ≈ 0.82 (consistent across seeds), sigma_1/||G||_F ≈ 0.99 (very rank-1 dominant).
Computed threshold ||G||_F > 1/(0.474·0.99) ≈ 2.13.
Observed max ||G||_F = 1.32–1.95 across 4 seeds → **threshold NOT crossed** in coarse-grained measure.
Yet cascade DID happen → threshold formula over-estimates / accumulation effect not captured by single-step formula.

## Verdict (honest)
- Qualitatively consistent with bilinear hypothesis (high coherence, L11 cascade, L0 stuck)
- Quantitatively inconclusive (threshold not crossed at step granularity, hyperparam mismatch)
- Pattern same as 6t1cgamma seed44/45 stuck cases (L0 stuck) — possibly hyperparam-induced

## Files
- diag_ideal_seed{42,43,44,45}.json (gitignored, ~65MB each)
- (cascade analysis plot will be added)
- README.md (this file)

## Recommended next experiment
Re-run with ideal-abml-optimal hyperparams (trial 158 config):
LR=3.28e-3, transfer_lr=0.2499, fast_lr=0.70, te=2, abml=9
→ Should mostly succeed (F1 ~85) with ~14% collapse rate.
→ Then analyze the ~14% collapses for true bilinear verification.

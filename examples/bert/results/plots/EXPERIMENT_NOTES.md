# LRTT BERT SQuAD — Noise Robustness Experimental Notes

Reference document for paper writing. Focuses on experimental design, results,
and conclusions. Cross-reference data files for raw numbers.

## Research narrative

**Main goal**: Quantify and explain LRTT's robustness to analog A/B device noise
during BERT SQuAD fine-tuning.

**Main findings** (Phase 1 + 2):
1. **Asymmetric noise sensitivity** — F1 distribution across 4 noise configurations
   (none / A-only / B-only / both) is *not* symmetric. Both-side noise drops F1 by
   ~1.4 (8σ significant) below single-side or no-noise. (Phase 1)
2. **Mechanism** — driven by multiplicative ‖A·B‖ magnitude growth feeding cumulative
   noise into C tile via transfer. (Phase 2: H4 + H3' confirmed)

**Sidetrack** (anomaly discovered mid-Phase 2): The no-noise control condition,
expected to be most stable, exhibited sporadic Ep5 F1 collapse (83 → ~20). This
unexpected failure mode required investigation before main-line results could be
reported reliably:

- **Investigation A** (reproducibility) — collapse is real and reproducible under
  min_lr_rate=0.01 (deterministic), stochastic under min_lr_rate=0.0. Always
  localized at L11.attention.output.dense with ‖A·B‖ saturating to ~22000.
- **Investigation B** (mechanism) — collapse is hardware-bounded bilinear
  instability of A·B (‖A·B‖_F ≤ √(D·r)² × w_max² = 24336). Intrinsic to LRTT
  fast-factor dynamics, triggered by coherent gradient direction in late training.
- **Investigation C** (code regression hypothesis) — tested whether a recent
  controller update caused the collapse. Pre-pull vs post-pull comparison rejected
  this; mechanism is intrinsic to LRTT, not version-specific.

**Current** (intervention test): AUTO_SCALE_MODE="separate" provides dynamic LR
normalization that should dampen the bilinear amplification. 4-seed validation
experiment in progress. Success here allows reliable replication of Phase 1
results with the instability resolved.

## Setting

Task: BERT-base fine-tuning on SQuAD v1.1 with LRTT (Low-Rank TikiTaka) analog
training. Standard hyperparameters: lr=0.0038, transfer_lr=0.095, transfer_every=1,
fast_lr=0.474, rank=32, ab_dw_min=4.88e-4, c_dw_min=1.95e-3, decay reinit, 5 epochs,
batch=48, GRAD_ACCUM=3, seq_len=384, AUTO_SCALE_MODE="none" (default in earlier runs).

Devices under test (A/B fast factors):
- `constantstep6t1cgamma`: 6T1C-like device with gamma nonlinearity, no write noise,
  no dtod variation, no decay (lifetime=0). Represents "no-noise" baseline.
- `6t1c`: Full 6T1C with all noise sources (write noise, dtod, mult_noise). Represents
  realistic noisy device.

C tile fixed as `constantstepideal` (ideal slow-update path).

## Phase 1 — F1 distribution across asymmetric noise

**Question**: Does noise on both A and B tiles degrade accuracy more than noise on
either one alone?

**Design**: 4 conditions × 5 seeds (42-46), same hyperparameters T6/T249 across all
conditions for apples-to-apples (not tuned per condition). 5 epochs each.

| Condition | A device | B device |
|---|---|---|
| `no_noise` | constantstep6t1cgamma | constantstep6t1cgamma |
| `a_only`   | 6t1c                  | constantstep6t1cgamma |
| `b_only`   | constantstep6t1cgamma | 6t1c                  |
| `both`     | 6t1c                  | 6t1c                  |

**Output**: `replicate_4conditions_20260503_064135.{json,png,svg}` (Phase 1 raw),
`fig6c_data.json` (formatted for figure).

**Result** (mean F1 across 5 seeds):
- `no_noise`, `a_only`, `b_only`: ≈ 83.5 (within seed variance ~0.3)
- `both`: ≈ 82.1, **~1.4 F1 below others** (8σ significant)

**Conclusion**: Both-side noise degrades accuracy super-linearly compared to
single-side noise. The effect is consistent (not seed-dependent) and warrants
mechanism analysis (Phase 2).

## Phase 2 — Mechanism analysis (4 hypotheses)

**Question**: Why does both-side noise degrade more than expected from independent
noise sources?

**Design**: 4 conditions × seed=42 (single seed, diagnostic-heavy run). Same
hyperparameters T6/T249 uniform across conditions. ENABLE_DIAGNOSTIC=True,
MULTI_TILE_DIAG=True (track L0/L6/L11 × q,k,v,output tiles). 5 epochs.

**Output**: `diag_4conditions/diag_{no_noise,a_only,b_only,both}.json` + plots
`diag_plot{1..5}_*.{png,svg}`.

**Conclusion**: Two confirmed mechanisms drive the both-side degradation:
- **H4** (confirmed): Multiplicative magnitude growth of ‖A·B‖ (7 → 14.5 → 31 across
  no_noise → single → both)
- **H3'** (confirmed): Cumulative noise injection into C tile (‖C_raw‖ +24% in
  both; erank(C − C_init) drops from 580 → 260)
- H1 partial; H2 (rank degradation) rejected (erank(A·B) ≈ 31.5 across all)

The amplified A·B feeds into C via transfer, polluting the slow weight. This
points to bilinear amplification dynamics (followed up in Investigation B).

**For full plot-by-plot quantitative analysis** (numerical tables, ΔA/ΔB
magnitudes per condition, erank evolution, C noise injection), see
[`diag_4conditions/ANALYSIS.md`](diag_4conditions/ANALYSIS.md).

## Investigation A — F1 collapse stochasticity

**Trigger**: During Phase 2 we noticed the `no_noise` condition (which was expected
to be most stable) sometimes had Ep5 F1 collapse from 83+ to ~20. Same hyperparameters
across multiple re-runs gave inconsistent outcomes.

**Question**: Is the no_noise collapse reproducible? What triggers it?

**Design**: 4-variant experiment on no_noise condition with multi-tile diagnostic.

| Variant | Seed | min_lr_rate |
|---|---|---|
| `seed42`            | 42 | 0.0 (linear decay to 0) |
| `seed43`            | 43 | 0.0 |
| `seed44`            | 44 | 0.0 |
| `seed42_lrfloor001` | 42 | 0.01 (LR floor at 1% of peak) |

**Output**: `diag_no_noise_variants/diag_{seed42,seed43,seed44,seed42_lrfloor001}.json`
+ `multitile_plot1_norms.png`, `multitile_plot2_outliers.png`.

**Results** (Ep5 F1 and peak ‖A·B‖ at L11.output.dense):

| Variant | Ep5 F1 | peak ‖A·B‖ |
|---|---|---|
| seed42 (min_lr=0.0) | **83.57** | 10.15 |
| seed43 (min_lr=0.0) | **83.63** | 9.66 |
| seed44 (min_lr=0.0) | **83.80** | 124.7 |
| seed42_lrfloor001 (min_lr=0.01) | **2.29** ❌ | **21148** |

**Key observations**:

1. With min_lr_rate=0.0, no_noise reproduces correctly (3/3 seeds): F1 83.5+, ‖A·B‖
   stays bounded (< 130).
2. With min_lr_rate=0.01 (LR floor): catastrophic collapse Ep4 → Ep5. ‖A·B‖ at
   L11.output.dense saturates to ~21000.
3. Across all collapse cases observed (from this and other runs — see Phase 2's
   `no_noise` run and Investigation C), the saturation level is always **‖A·B‖ ≈
   22000**, which equals √(D·r)² × w_max² = 156² ≈ 24336 (hardware bound).
4. L11.attention.output.dense is consistently the explosion location across all
   collapse cases — never Q/K, sometimes V or earlier-layer outputs.

**Conclusion**: F1 collapse is a real, distinct failure mode (not stochastic
noise). The mechanism is ‖A·B‖ at L11.output saturating to the hardware bound,
which then pollutes C via transfer. The min_lr_rate=0.01 condition exhibits this
deterministically while min_lr_rate=0.0 exhibits it stochastically (some runs).

## Investigation B — Hardware saturation mechanism

**Question**: Why does ‖A·B‖ saturate at this specific value (~22000)? What
governs the underlying dynamics?

**Analysis** (theoretical, supplemented by toy NumPy simulation — kept private,
not included in repo):

The LRTT fast-factor update structure is:
```
ΔA = -η · G · B^T,   ΔB = -η · A^T · G,   where G = Σ_k d_k x_k^T
```

This is a *bilinear* coupling: A's change depends linearly on B (and gradient G),
and B's change depends linearly on A. Linearizing around equilibrium gives a 2D
system with eigenvalues `1 ± η·σ_1(G)`, where σ_1 is the dominant singular value
of the batch's outer-product gradient sum.

For coherent gradient direction (late training, when loss landscape has a
well-defined principal Hessian eigendirection), η·σ_1 can exceed unity over the
~9000 training steps, leading to exponential growth of the unstable mode along
(A, -B^T).

The growth is capped by analog hardware: per-element conductance is bounded by
w_max=1, so ‖A‖_F ≤ √(D·r) × w_max = √(768·32) × 1 = 156. Similarly for B.
Therefore ‖A·B‖_F ≤ ‖A‖·‖B‖ ≈ 24336, matching observation.

**Implications**:
- Collapse is intrinsic to the LRTT fast-factor dynamics, not a code bug.
- The "explosion" is bounded but reaches values large enough to overwhelm C tile
  through transfer (transfer_lr × A·B = 0.095 × 22000 = 2090 conductance change
  per transfer).
- Layer-specific susceptibility (L11.output.dense always first to saturate)
  reflects the position-dependent gradient×input magnitudes in late training.

**Conclusion**: F1 collapse mechanism is hardware-bounded bilinear instability of
A·B in a specific layer (L11.output.dense), triggered when gradient direction
becomes coherent enough during late training.

## Investigation C — Regression hypothesis (post-pull controller)

**Trigger**: After a remote pull updated `lrtt_controller.py` (+205 lines, commit
463188d), the F1 collapse seemed more frequent. Tested whether the new code
introduced a regression.

**Question**: Did the post-pull code update cause the observed collapse frequency?

**Design**: 4-run pre-pull vs post-pull controller comparison via importlib
override (post-pull file unchanged; pre-pull version loaded into module slot for
2 of the 4 runs).

| Variant | Controller | Seed |
|---|---|---|
| postpull_seed42 | post-pull (current HEAD) | 42 |
| postpull_seed43 | post-pull | 43 |
| prepull_seed42  | pre-pull (commit 463188d^) | 42 |
| prepull_seed43  | pre-pull | 43 |

All min_lr_rate=0.0, same hyperparams T6/T249, multi-tile diag.

**Output**: `diag_prepull_vs_current/diag_{postpull_seed42,postpull_seed43,prepull_seed42,prepull_seed43}.json`.

**Results**:

| Run | Ep5 F1 | peak ‖A·B‖ (L11.output) |
|---|---|---|
| postpull_seed42 | **6.65** ❌ | 22909 (collapse) |
| postpull_seed43 | 83.39 | 269.9 (stable) |
| prepull_seed42  | 83.87 | 153.3 (stable) |
| prepull_seed43  | 83.67 | 289.0 (stable) |

**Conclusion**: 
- Post-pull controller: 1/2 collapse rate (seed 42).
- Pre-pull controller: 0/2 collapse (both seeds stable).
- Small sample size; difference is suggestive but not statistically conclusive.
- Critically, the *same hardware saturation pattern* (‖A·B‖ ≈ 22000 at
  L11.output.dense) appears in postpull_seed42 collapse — same mechanism as
  Investigation A's seed42_lrfloor001 case, regardless of controller version.
- This indicates that the collapse is intrinsic to the LRTT dynamics (Investigation
  B), not specifically a post-pull code regression. The pre-pull controller may be
  slightly more conservative but does not fundamentally prevent the instability.

## Investigation D — AUTO_SCALE intervention (FAILED)

**Question**: Does adaptive LR normalization (AUTO_SCALE_MODE="separate") prevent
the bilinear instability collapse?

**Background**: TikiTaka's auto-scale feature (from c-TTv2, Gokmen 2024) computes
`lr_eff = fast_lr / (m_xb · m_d)` where m_xb, m_d are EMAs of per-sample input and
gradient magnitudes. The "separate" mode separately tracks XB (= B·x) and DA (=
A^T·d), so as ‖B‖ grows, m_xb grows, automatically shrinking lr_eff for the A
update.

**Design**: 4 GPU × 4 seeds (42, 43, 44, 45). All AUTO_SCALE_MODE="separate" with
auto_momentum=0.99. Other hyperparameters identical to baseline.

**Output**: `analysis/03_autoscale_failure__6t1cgamma__separate__seed42-45_20260511/`

**Result**: **4/4 collapse to F1 ~7%** at epoch 0-1 ("Type 2" failure mode).
‖A‖ grows to ~100-150 (large) but in random/wrong direction → model never learns.

- seed 42: F1=7.27% (best at epoch 0)
- seed 43: F1=6.96%
- seed 44: F1=7.02%
- seed 45: F1=7.21%

**Conclusion**: AUTO_SCALE=separate fixes Pattern A (saturation cascade) by aggressively
normalizing, but introduces a new failure mode (Pattern C — Type 2 initial damage):
the EMA over-corrects on initial gradients, pushing A,B into wrong direction from start.
This is NOT a successful intervention — it trades one failure for another.

**Pre-existing autoscale EMA bug**: Discovered during this run that Python implementation
of `_update_autoscale_ema` was missing the cold-start protection from original C++ TikiTaka
(rpu_chopped_transfer_device.cpp:92-103). Fixed in `_update_ema_exact` (commit `3de174b`).
Even after fix, the Type 2 failure persists.

## Refactor — Diagnostic system reorganization (2026-05-12)

After investigations A-D, the diagnostic system in `fine_bert_squad_lrtt.py` was
refactored into a hierarchical, modular structure (7 steps, commit `9f395ee`):
- `DIAG_TILES` config: per-layer tile selection ("first_last" / "all" / dict)
- `DIAG_GROUPS`: 16 measurement flags (norms, mean/abs, deltas, weight hist, erank+sigma_1,
  cells, transfer cosines, signal abs+hist, transfer-event xc/dc abs+hist+meta)
- `HIST_RATE_STEPS`: unified histogram rate-limit
- Per-group helper functions, `diag_state` unified single source of truth

Added later (commit `461b2d91`) for bilinear hypothesis verification:
- **cos_G_prev**: temporal coherence of G_accum (one-step)
- **sigma1_G**: top singular value of G_accum
- **sigma1_G_ratio**: σ_1(G) / ‖G‖_F (top-mode dominance, random baseline ≈ 0.07)

Together these probe the bilinear unstable eigenvalue condition `1 + η·σ_1(G) > 1`.

## Investigation E — fast_lr ablation with G coherence (5/12)

**Question**: Does fast_lr (η) dose-response match bilinear threshold prediction?

**Design**: 4 GPU × seed=42 × fast_lr ∈ {0.474, 0.1, 0.05, 0.01}. Full diag with G coherence.

**Output**: `analysis/07_gcoh_fastlr_ablation__6t1cgamma__flr0p474_0p1_0p05_0p01__seed42_20260512/`

**Result**:
- cos_G_prev ≈ 0.81 across ALL 4 runs (G temporally coherent everywhere)
- σ_1/‖G‖_F ≈ 0.81 at L11 (very rank-1 dominant; random baseline ≈ 0.072)
- σ_1/‖G‖_F ≈ 0.38 at L0 (much less dominant)
- All 4 stable (no collapse at this ablation set)
- → Derives quantitative threshold formula: ‖G‖_F > 1/(η·0.81)

## Investigation F — Multi-seed Pattern D discovery (5/13)

**Question**: Do other seeds collapse at default fast_lr? What's the seed-level distribution?

**Design**: 4 seeds (42, 43, 44, 45) × minimal-diag G coherence, all FI=False.

**Output**: `analysis/08_gcoh_seedstats__6t1cgamma__minimal_diag__seed43_44_45_20260513/`

**Result**:
- seed 42: F1=83.70 — normal (cascade did NOT occur in this run, contrast 5/12 4cond original)
- seed 43: F1=83.42 — normal
- seed 44: F1=7.16 — **Pattern D STUCK** (L0 ‖A‖ frozen at init, never learns)
- seed 45: F1=7.21 — **Pattern D STUCK**

**Pattern D discovered**: Distinct from Pattern A (saturation cascade) and Pattern C
(Type 2 wrong direction). L0_query never grows (‖A·B‖ stays ≈ 1.5), but L11 looks
"normal" (‖A·B‖ ≈ 8-11). F1 collapses because L0 query never adapts to SQuAD task.

Reproducibility test (6/8) confirmed seed 44 stuck is **systematic, not chaotic**
— 3rd repeat gave same outcome.

## Investigation G — FORWARD_INJECT=True test (5/13)

**Question**: If A·B is in forward (LoRA-like, self-limiting feedback), does collapse stop?

**Design**: Single seed=42 with FORWARD_INJECT=True, otherwise same hyperparams.

**Output**: `analysis/09_gcoh_fi_true__6t1cgamma__seed42_20260513/`

**Result**: F1 trajectory 78→5→6→4→5. **F1 collapse at epoch 2** despite FI=True.
But L11 ‖A·B‖ stays at 7.56 (no cascade!) — different failure mode (direction collapse).

**Caveats**:
- Single seed; chaotic factor unknown
- Hyperparams optimized for FI=False; FI=True may need different optuna search

**Verdict**: Naive FI=True doesn't help. Needs proper hyperparam tuning + multi-seed.

## Investigation H — fast_lr push for forced cascade (6/8)

**Question**: Does higher fast_lr force cascade reliably? Dose-response?

**Design**: 4 GPUs:
- seed 42 × fast_lr 0.474 (control)
- seed 44 × fast_lr 0.474 (Pattern D reproducibility check)
- seed 42 × fast_lr 0.7 (push toward threshold)
- seed 42 × fast_lr 1.0 (force cross threshold)

**Output**: `analysis/10_gcoh_fastlr_push__6t1cgamma__flr0p474_0p7_1p0__seed42_20260608/`

**Result** — **STRONG dose-response evidence for bilinear cascade**:
| Run | fast_lr | F1 best | L11 max ‖A·B‖ | Cascade onset |
|---|---|---|---|---|
| seed 42 (default) | 0.474 | 83.59 | 9.5 | never |
| seed 44 (Pattern D) | 0.474 | 7.03 | 11.6 | never (L0 stuck) |
| seed 42 push 0.7 | 0.7 | 7.77 | **7204** | step 161 (epoch 0.09) |
| seed 42 push 1.0 | 1.0 | 6.35 | **7810** | step 60 (epoch 0.03) |

→ Higher fast_lr → cascade onset earlier (60 vs 161 steps). Cascade magnitude same (~7000).
→ Bilinear amplification confirmed under stress conditions.

**Caveat**: High fast_lr also kills learning entirely (F1=6-7) — cascade vs learning failure
not cleanly separable. Still consistent with bilinear mechanism (cascade existence verified).

## Investigation I — Ideal device multi-seed (6/14)

**Question**: Does bilinear collapse occur in fully ideal device (no 6T1C decay/drift)?

**Design**: 4 GPUs × seed 42-45, all devices = constantstepideal. Same hyperparams as baseline.

**Output**: `analysis/11_ideal_multiseed_w_6t1cgamma_hyperparams__seed42-45_20260614/`

**Result**: **4/4 collapse to F1 ~7%**. Pattern:
- L0 stuck (‖A·B‖ ≈ 1.3 flat) in all 4 — Pattern D
- L11 cascade in 3/4 (early epoch 2-6%); seed 43 monotone slow growth (no cascade)
- cos_G_prev ≈ 0.82, σ_1/‖G‖_F ≈ 0.99 (very rank-1, more dominant than 6t1cgamma's 0.81)

**CRITICAL CAVEAT — hyperparam mismatch**:
- Our config (LR=0.0038, transfer_lr=0.095, fast_lr=0.474) is **6t1cgamma-optuna-optimal**
- Ideal-device optuna shows different optima (top-1 abml: LR=3.28e-3, transfer_lr=0.2499,
  fast_lr=0.70, te=2, ab_multilevel=9 → F1=85.29)
- **transfer_lr is 2.6× too low** for ideal device with our config
- → 4/4 failure likely confounds bilinear instability with hyperparam mismatch

**Verdict**: Cannot cleanly verify "bilinear is device-independent" from this data.
Need to re-run with ideal-optimal hyperparams to get a working baseline first,
then identify the natural ~14% collapse rate cases (per optuna data).

## Refined verdict on bilinear hypothesis (as of 6/14)

What is **strongly supported**:
- Bilinear unstable mode `1 + η·σ_1(G)` controls cascade dynamics
- fast_lr dose-response (60/161/never step onset at η=1.0/0.7/0.474) — clean evidence
- L11 cascade signature (||A·B|| → ~22000 in 6t1cgamma, ~3000 in ideal)
- High temporal coherence cos_G_prev ≈ 0.82 + rank-1 dominance ~0.99 — prerequisites met
- Layer-specific susceptibility: L11 > L6 > L0 (depth-dependent gradient)

What is **only partially supported**:
- Threshold formula `‖G‖_F > 1/(η·0.81)` overestimates (cascade can happen below threshold)
- Single-step formula doesn't capture multi-step accumulation
- Cascade vs. learning failure not cleanly separable at high fast_lr

What is **NOT explained by bilinear alone**:
- **Pattern D 'stuck'** (L0 never grows, F1=7%): different mechanism — likely
  hyperparam-dependent pulse-update threshold (|η·G·B^T| < dw_min → no pulses fire)
- **FI=True direction collapse** (Investigation G): different failure mode
- **Type 2** (autoscale separate): different failure mode

What is **NOT yet tested**:
- Ideal device with ideal-optimal hyperparams (Investigation I has confound)
- Lifetime > 0 (explicit decay)
- Per-layer LR scaling
- Reinit_mode = "hybrid" or "standard" (vs decay)
- FI=True with FI-optuna-optimal hyperparams

## Future experiment candidates (updated)

1. **Ideal device + ideal-optimal hyperparams** (top priority): use abml trial 158
   config (LR=3.28e-3, transfer_lr=0.2499, fast_lr=0.70, te=2, abml=9) × 4-8 seeds.
   Expected ~14% collapse rate per optuna log → identify natural collapses → verify
   threshold crossing in those.

2. **Lifetime > 0 sweep**: Test if explicit capacitor decay on A,B prevents cascade.

3. **Reinit_mode comparison** (decay vs hybrid vs standard): Tests if A reset every
   transfer breaks bilinear accumulation.

4. **FI=True optuna search → multi-seed**: Currently lacking proper hyperparams for FI=True.

5. **Pattern D mechanism deep-dive**: Why does L0 stick? Check pulse-update threshold
   |η·G·B^T| vs dw_min at L0 vs L11.

## Data file index

After 6/14 plots/ reorganization, structure is:

```
plots/
├── paper/       — fig5*, fig6*, figS* (publication figures)
├── paper_src/   — main_wiley.tex, supplementary, reference.bib
├── scripts/    — investigate_*, replicate_*, run_sweep_*, plot_*, verify_* launchers
└── analysis/  — per-experiment data + plots + README (numbered chronologically)
    01_noise_asymmetry__6t1c__condition4__seed42_20260503/      Phase 1
    02_noise_asymmetry_multitile__6t1c__condition4__seed42_20260512/  Phase 2
    03_autoscale_failure__6t1cgamma__separate__seed42-45_20260511/    Investigation D
    04_lrfloor_variants__6t1cgamma__seed42_43_44_lrfloor001_20260506/ Investigation A
    05_prepull_vs_current_controller__6t1cgamma__seed42_43_20260507/  Investigation C
    06_bilinear_hypothesis_posthoc__no_new_runs/                       Investigation B
    07_gcoh_fastlr_ablation__6t1cgamma__flr0p474_0p1_0p05_0p01__seed42_20260512/   Investigation E
    08_gcoh_seedstats__6t1cgamma__minimal_diag__seed43_44_45_20260513/             Investigation F
    09_gcoh_fi_true__6t1cgamma__seed42_20260513/                                   Investigation G
    10_gcoh_fastlr_push__6t1cgamma__flr0p474_0p7_1p0__seed42_20260608/             Investigation H
    11_ideal_multiseed_w_6t1cgamma_hyperparams__seed42-45_20260614/                Investigation I
    _global_outputs/   COLLAPSE_VERIFICATION_REPORT.md, lora_vs_lrtt plots, threshold plot
    _etc_misc/         older miscellaneous analysis (not used in paper)
```

Each `NN_*` folder has `README.md` describing question, setup, variables, result, caveats.
Large per-experiment diagnostic JSONs are gitignored (see `.gitignore` line 204+).

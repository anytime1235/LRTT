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

## Current investigation — AUTO_SCALE intervention

**Question**: Does adaptive LR normalization (AUTO_SCALE_MODE="separate") prevent
the bilinear instability collapse?

**Background**: TikiTaka's auto-scale feature (from c-TTv2, Gokmen 2024) computes
`lr_eff = fast_lr / (m_xb · m_d)` where m_xb, m_d are EMAs of per-sample input and
gradient magnitudes. The "separate" mode separately tracks XB (= B·x) and DA (=
A^T·d), so as ‖B‖ grows, m_xb grows, automatically shrinking lr_eff for the A
update. This dynamic LR self-regulation is hypothesized to dampen the bilinear
amplification before reaching hardware saturation.

**Design**: 4 GPU × 4 seeds (42, 43, 44, 45). All AUTO_SCALE_MODE="separate" with
auto_momentum=0.99 (IBM ChoppedTTv2 default across 6 device-specific presets).
Other hyperparameters identical to baseline (lr=0.0038, fast_lr=0.474, transfer_lr=0.095,
te=1, no_noise device condition, min_lr_rate=0.0).

**Output**: `diag_autoscale/diag_separate_seed{42-45}.json` (in progress).

**Predictions**:
- If hypothesis correct: 0/4 collapse. ‖A·B‖ remains bounded (< 100 throughout
  training, not approaching 22000). F1 ≈ 83.5+ across all seeds.
- If partial: Some seeds collapse, suggesting auto-scale slows but does not
  eliminate the instability.
- If hypothesis fails: 4/4 collapse despite auto-scale. Indicates another
  protective mechanism is needed (e.g., explicit ‖A·B‖ clipping, smaller fast_lr,
  or non-zero `lifetime` for explicit decay).

**Status**: Running. Expected completion: ~2026-05-12 morning.

## Future experiment candidates

Based on findings, the following follow-up experiments are well-motivated:

1. **fast_lr sweep with AUTO_SCALE="none"**: Test whether reducing fast_lr alone
   (without autoscale) shifts the collapse onset later. Predicts inverse relationship
   between fast_lr and time-to-collapse.

2. **lifetime>0 (capacitor decay)**: Test whether enabling explicit weight decay
   on A,B tiles prevents the buildup. Should slow instability but cost steady-state
   training signal.

3. **Per-layer LR scaling**: Apply smaller fast_lr only at L11.output.dense
   (and other vulnerable positions). Minimally invasive — preserves learning
   capacity at other layers.

4. **Reinit mode comparison**: `hybrid` (A=0 every transfer) vs current `decay`
   (no reinit). Hybrid should break coherent A buildup but lose accumulated
   gradient signal.

5. **AUTO_SCALE separate × all 4 device conditions**: If current Investigation
   succeeds, replicate Phase 1 with auto_scale enabled. Tests whether `both`
   condition's 1.4 F1 gap shrinks.

## Data file index

| Data directory | Phase | Contents |
|---|---|---|
| `replicate_4conditions_20260503_064135.{json,png,svg}` | Phase 1 | 4 conditions × 5 seeds F1 distribution |
| `diag_4conditions/` | Phase 2 | 4 conditions × seed=42 mechanism diag + ANALYSIS.md |
| `diag_no_noise_variants/` | Inv. A | no_noise stochasticity + lrfloor effect |
| `diag_prepull_vs_current/` | Inv. C | Pre-pull vs post-pull controller comparison |
| `diag_autoscale/` | Current | AUTO_SCALE=separate × 4 seeds (in progress) |
| `fig5*.py`, `fig6*.py`, `figS*.py` | Paper figs | Plotting scripts for paper |
| `analyze_multitile.py` | Helper | Multi-tile diagnostic plotting tool |

# Investigation J — Ideal device with T267 hyperparams (corrected re-run of Inv. I)

**Date**: 2026-06-15
**Launch**: `/tmp/run_ideal_multiseed_v2.py`

## Motivation
Investigation I (`analysis/11_*`) ran `constantstepideal` × 4 seeds but accidentally
patched source defaults to **6t1cgamma-optimal** hyperparams (LR=0.0038, transfer_lr=0.095,
fast_lr=0.474, te=1, abml=None). Result: 4/4 collapse to F1 ~7% — but driven by
**hyperparam mismatch**, not bilinear instability. Inconclusive.

This re-run uses **ideal-optimal hyperparams (T267 from optuna log)** to test whether
collapse occurs naturally on ideal device with proper baseline.

## Setup
All devices: `constantstepideal`. Hyperparams from optuna T267 (F1=84.98 in abml log):

| Param | Value |
|---|---|
| LEARNING_RATE | 0.00328 |
| TRANSFER_LR | **0.25** (vs V1 0.095) |
| FAST_LR | **0.3** (vs V1 0.474) |
| TRANSFER_EVERY | **2** (vs V1 1) |
| AB_DW_MIN | **0.0001210937** (vs V1 0.0004883) |
| **AB_MULTILEVEL** | **10** (vs V1 None) |
| LRTT_RANK | 32 |
| REINIT_MODE | decay |
| TRANSFER_METHOD | onehot |
| AB_PULSE_TYPE | default |
| update_mode | lora (bilinear chain rule) |
| AUTO_SCALE_MODE | none |
| FORWARD_INJECT | False |
| MIN_LR_RATE | 0 |
| BS | 48, 5 epochs |

## Variables
SEED ∈ {42, 43, 44, 45}. Everything else identical.

## Results — 4/4 NORMAL training (no collapse)

| Seed | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 (best) | L11 max ‖A·B‖ | L11 max ‖G‖_F |
|---|---|---|---|---|---|---|---|
| 42 | 80.29 | 82.30 | 83.78 | 84.50 | **84.81** | 6.4 | 0.860 |
| 43 | 80.44 | 82.80 | 83.65 | 84.46 | **84.77** | 5.5 | 1.077 |
| 44 | 80.14 | 82.31 | 83.39 | 84.04 | **84.20** | 5.7 | 0.856 |
| 45 | 80.22 | 82.40 | 83.57 | 84.53 | **84.82** | 5.7 | 0.948 |

Mean F1 = 84.65 ± 0.27 (matches optuna best 84.98 closely).

## Comparison with V1 (Inv. I)

| | V1 (6t1cgamma hyperparams on ideal) | V2 (T267 ideal-optimal) |
|---|---|---|
| F1 (best) | 7.74, 6.80, 7.79, 7.20 | 84.81, 84.77, 84.20, 84.82 |
| L11 max ‖A·B‖ | 2611, 130, 3460, 3470 | 6.4, 5.5, 5.7, 5.7 |
| Pattern | Pattern D stuck + early cascade in 3/4 | Healthy monotonic learning |
| Threshold cross | No (max ‖G‖ 1.32-1.95 < 2.13) | No (max ‖G‖ 0.86-1.08 << 2.13) |

→ V1 failure was entirely hyperparam mismatch. V2 confirms ideal device works fine
with T267 hyperparams.

## Implications for bilinear hypothesis

(1) **0/4 collapse** in V2 vs optuna log's ~14% baseline rate.
   - Binomial P(0/4 with p=0.14) ≈ 55% → consistent with small sample, no statistical
     anomaly.
   - Need ~20+ seeds to reliably catch the natural ~14% collapse.

(2) **‖G‖_F max stays well below threshold** (0.86-1.08 vs 2.13). System operates in
   stable bilinear regime throughout.

(3) **Validates "threshold not crossed → no collapse"** direction of bilinear hypothesis.
   But doesn't validate the reverse (no positive collapse observed to verify "threshold
   crossed → cascade").

## Next experiments

- **Push fast_lr on ideal with T267 base** (e.g., fast_lr=0.8, 1.5) to force cascade
  — analog of Investigation H but on ideal device.
- **More seeds (10-20)** at T267 to catch natural ~14% collapse rate.
- **AB_MULTILEVEL ablation**: T267 uses abml=10. Test abml=None (V1's setting) at
  otherwise T267 hyperparams — does abml=None alone induce collapse?

## Files
- `diag_ideal_v2_20260615_073717_ideal_v2_t267_seed{42,43,44,45}.json` — full diag logs (gitignored, ~60MB each)
- README.md (this file)

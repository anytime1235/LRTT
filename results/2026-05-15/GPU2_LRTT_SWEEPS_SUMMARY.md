# LRTT — recent tikitaka / LRTT v1 / LRTT v2 sweeps

- **Generated:** 2026-05-15
- **Task:** MNIST classification, LRTT diagnostic harness
- **Search:** Per cell, 30-trial TPE over `(transfer_lr, fast_lr, classifier_lr)` (log-uniform).
- **Grid:** 5×5 over `af_ratio ∈ {0, 1, 2, 5, 10}` × `update_noise_ratio ∈ {0, 1, 3, 5, 10}` → 25 cells × 30 trials = **750 trials per method**.

## Most recent sweep set: `per_cell_tpe_30_wide`

| Method        | Directory                                                          | Finished           | n_cells |
|---------------|--------------------------------------------------------------------|--------------------|---------|
| `tikitaka_v1` | `/root/LRTT/results/per_cell_tpe_30_wide/tikitaka_v1`              | 2026-05-10 19:38 UTC | 25     |
| `lrtt_v1`     | `/root/LRTT/results/per_cell_tpe_30_wide/lrtt_v1`                  | 2026-05-07 10:32 UTC | 25     |
| `lrtt_v2`     | `/root/LRTT/results/per_cell_tpe_30_wide/lrtt_v2`                  | 2026-05-08 06:14 UTC | 25     |

Driver log: `/root/LRTT/logs/per_cell_tpe_30_wide_20260506_143810.log` (last update 2026-05-10 19:38).

### tikitaka_v1 — best acc per cell

af \ update_noise | 0.0 | 1.0 | 3.0 | 5.0 | 10.0
---|---|---|---|---|---
**0.0**  | 97.97 | 97.99 | 97.96 | **98.00** | 97.92
**1.0**  | 97.56 | 97.53 | 97.49 | 97.39 | 97.35
**2.0**  | 97.17 | 97.04 | 96.98 | 97.00 | 96.81
**5.0**  | 96.11 | 96.17 | 96.37 | 95.92 | 96.04
**10.0** | 95.28 | 95.33 | 95.20 | 95.30 | 95.19

- Best cell: `af=0.0, unr=5.0 → 98.00%` (transfer_lr=0.244, fast_lr=0.199, classifier_lr=1.679).
- Update-noise is essentially flat at every af_ratio; af_ratio is the dominant degrader.

### lrtt_v1 — best acc per cell

af \ update_noise | 0.0 | 1.0 | 3.0 | 5.0 | 10.0
---|---|---|---|---|---
**0.0**  | **97.77** | 97.77 | 97.65 | 97.77 | 97.49
**1.0**  | 96.88 | 97.22 | 97.14 | 97.19 | 96.95
**2.0**  | 96.43 | 97.17 | 97.04 | 96.94 | 96.57
**5.0**  | 95.53 | 96.00 | 95.85 | 96.28 | 96.34
**10.0** | 93.74 | 94.22 | 94.76 | 95.40 | 96.03

- Best cell: `af=0.0, unr=0.0/5.0 → 97.77%` (e.g. lr=1.271, tlr=1.57e-4, clr=0.677).
- Stronger af degradation than tikitaka_v1 (97.77 → 93.74 along unr=0). Interestingly, under high af, *more* update_noise helps (10.0,0.0)=93.74 vs (10.0,10.0)=96.03.

### lrtt_v2 — best acc per cell

af \ update_noise | 0.0 | 1.0 | 3.0 | 5.0 | 10.0
---|---|---|---|---|---
**0.0**  | **97.16** | 97.01 | 97.08 | 97.09 | 97.02
**1.0**  | 97.11 | 97.02 | 97.12 | 97.03 | 96.93
**2.0**  | 96.94 | 97.12 | 97.08 | 96.97 | 97.09
**5.0**  | 96.85 | 97.01 | 97.11 | 97.02 | 96.84
**10.0** | 97.11 | 97.03 | 96.94 | 97.07 | 96.80

- Best cell: `af=0.0, unr=0.0 → 97.16%` (lr=0.436, tlr=0.0270, clr=1.306).
- Distinctly flat over the entire grid (range ~96.8–97.2) — robustness over absolute ceiling.

## Method comparison at a glance

| Method        | Best acc | (af, unr) at best | Mean across 25 cells | Range across grid |
|---------------|---------:|--------------------|---------------------:|------------------:|
| tikitaka_v1   | **98.00** | (0.0, 5.0)        | ~96.7                | 95.19 → 98.00 (Δ2.81) |
| lrtt_v1       | 97.77    | (0.0, 0.0)/(0.0,5.0) | ~96.4              | 93.74 → 97.77 (Δ4.03) |
| lrtt_v2       | 97.16    | (0.0, 0.0)        | ~97.0                | 96.80 → 97.16 (Δ0.36) |

## Headline findings
- **tikitaka_v1 wins on peak accuracy** and degrades gently under noise.
- **lrtt_v1 is competitive at low (af, unr) but most fragile** to af scaling — and shows the counter-intuitive "more update_noise helps under high af" pattern.
- **lrtt_v2 is the most robust** across the noise grid (flat at ~97%), trading peak accuracy for variance.
- Best-HP regions differ substantially across methods, so TPE warm-starts are not transferable across `tikitaka_v1 / lrtt_v1 / lrtt_v2`.

## Earlier sweeps (same trio, narrower search) — for reference
- `/root/LRTT/results/per_cell_tpe_30/tikitaka_reset`   — 2026-05-04 14:50 UTC
- `/root/LRTT/results/per_cell_tpe_30/lrtt_v1_reset`    — 2026-05-05 10:58 UTC
- `/root/LRTT/results/per_cell_tpe_30/lrtt_v2_noreset`  — 2026-05-06 05:01 UTC

The newer `per_cell_tpe_30_wide` sweeps superseded these with widened search bounds.

## Raw data layout
Each cell is a single JSON: `{method, af_ratio, update_noise_ratio, search_space, warm_start, trials:[{hp, acc, wall_seconds}, ...]}`. 25 files per method × 30 trials per file.

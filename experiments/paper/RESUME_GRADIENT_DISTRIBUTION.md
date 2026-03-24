# Gradient Distribution Measurement — Resume Guide

Last updated: 2026-03-24

## Overview

Measure actual gradient element distributions for all sublayer types (Q, K, V, O, FFN1, FFN2)
to characterize heavy-tail behavior. Uses IdealDevice with all 72 encoder layers analog,
IO perfect, fixed LR (no warmup/scheduler).

## Why This Experiment

From weight range analysis (TTv1 gamma=1.0, reset=1.0, 4ep):
- Summary stats showed absmax/mean_abs ratio of 1000~2500x → extreme heavy tail
- mean_abs (10⁻⁷~10⁻⁶) is 30~140x smaller than dw_min (1.22e-4 @14b)
- Need actual per-element gradient distribution to confirm and characterize

## Script

```
/root/measure_gradient_distribution.py
```

Key design:
- **Histogram accumulation**: 500 log-spaced bins from 10⁻⁷ to 10⁻² (100 bins/decade)
  - Accumulated every step → memory constant (~288KB) regardless of epoch length
  - Under/overflow bins capture elements outside range
- **Summary stats**: logged every 50 steps (absmax, mean_abs, percentiles, etc.)
- **Raw gradient samples**: first 5 steps only (subsampled to 100k elements per tensor)
- **OOM fallback**: tries batch_size=[16, 8] in order

## Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| method | ideal | IdealDevice, FP32 update |
| target_layers | all | 72 encoder linear layers |
| IO | perfect | No quantization |
| batch_size | 16 (fallback to 8) | 24 confirmed OOM on this GPU |
| grad_accum | 1 | Distribution shape invariant to effective batch size |
| analog_lr | 0.0357 | Same as sensitivity experiments |
| classifier_lr | 0.00076 | |
| ln_lr | 0.00076 | |
| scheduler | none | Fixed LR throughout |
| warmup | none | |
| epochs | 1 | ~2768 steps @bs=16, ~5535 steps @bs=8 |
| seed | 42 | |

## How to Run

```bash
source /root/.venv310/bin/activate
cd /root

# Foreground (see output directly)
python measure_gradient_distribution.py

# Background with nohup
nohup python measure_gradient_distribution.py > results/gradient_distribution/nohup.log 2>&1 &
echo "PID: $!"

# Monitor
tail -f results/gradient_distribution/nohup.log
```

## Expected Runtime

- batch_size=16: ~45 min for 1 epoch
- batch_size=8: ~90 min for 1 epoch

## Output Files

All saved to `/root/results/gradient_distribution/`:

| File | Description |
|------|-------------|
| `grad_histograms.npz` | Per-layer accumulated histograms (500 bins × 72 layers) |
| `grad_summary.csv` | Per-step summary stats (every 50 steps) |
| `grad_raw_samples.npz` | Raw gradient elements (first 5 steps, subsampled) |
| `grad_pdf_log_all.png` | All sublayers PDF on one plot |
| `grad_pdf_per_sublayer.png` | Per-sublayer PDF with Gaussian/Laplace fits |
| `grad_ccdf_tail.png` | CCDF (complementary CDF) — power-law test |
| `grad_pdf_per_layer.png` | Per-layer distributions within each sublayer |
| `grad_heavy_tail_metrics.png` | Kurtosis, dynamic range, near-zero fraction |

## Previous Attempt

- 2026-03-24: batch_size=24 → CUDA OOM → NVML crash
- GPU needs restart before re-running
- Script updated to start from batch_size=16

## What to Verify After Run

1. **Heavy tail confirmed?** Check CCDF plot — straight line on log-log = power-law
2. **FFN vs QKVO difference?** Compare distribution shapes across sublayer types
3. **Gradient vs dw_min**: What fraction of gradient elements are below dw_min?
   - dw_min @14b = 1.22e-4, @8b = 7.87e-3
   - Check overflow bin (>10⁻²) and bins around 10⁻⁴

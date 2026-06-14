# Noise asymmetry — 4-condition single tile diag

**Date**: 2026-05-03 ~ 2026-05-06  
**Launch**: scripts/replicate_4conditions.py, scripts/replicate_4conditions_diag.py

## Question
A device noise vs B device noise: 비대칭이 학습/collapse에 어떤 영향?

## Setup
- LORA_TARGET=qkvo, REINIT_MODE=decay, TRANSFER_METHOD=onehot
- C_DEVICE=constantstepideal (고정)
- LR=0.0038, transfer_lr=0.095, fast_lr=0.474, te=1, rank=32
- BS=48, 5 epochs, seed=42, MIN_LR_RATE=0
- AUTO_SCALE_MODE=none, FORWARD_INJECT=False
- AB_DEVICE=6t1c (실제 noisy device)

## Variables (4 conditions)
| Condition | A device | B device |
|---|---|---|
| no_noise | constantstep6t1cgamma | constantstep6t1cgamma |
| a_only   | 6t1c                  | constantstep6t1cgamma |
| b_only   | constantstep6t1cgamma | 6t1c                  |
| both     | 6t1c                  | 6t1c                  |

## Diag
ENABLE_DIAGNOSTIC=True, MULTI_TILE_DIAG=False (first/last tile만), ERANK_RATE_LIMIT_STEPS=10

## Result Summary
- no_noise, a_only, b_only: F1 83-84 정상
- both: F1=78.72, best_epoch=1 (early degradation)

## Files
- diag_{no_noise,a_only,b_only,both}.json — per-condition diag log
- ANALYSIS.md — detailed analysis (H1/H2/H3/H4 hypothesis testing)
- diag_plot1_norms{,_L11}: ||A||, ||B||, ||A·B|| trajectories
- diag_plot2_erank{,_L11}: effective rank trajectories
- diag_plot3_deltas{,_L11}: per-step weight deltas
- diag_plot4_C_noise{,_L11}: C tile noise accumulation
- diag_plot5_learning: F1 + loss trajectory
- replicate_4conditions_20260503_064135.{json,png,svg}: 사전 4cond replication (no diag)

# no_noise 조건 + MIN_LR_RATE 변형 (collapse trigger 식별)

**Date**: 2026-05-06  
**Launch**: scripts/investigate_no_noise_variants.py

## Question
- seed에 따라 collapse 여부 다른가?
- MIN_LR_RATE>0이면 (LR floor) collapse 더 잘 일어나는가?

## Setup
- no_noise (A=B=constantstep6t1cgamma), C=constantstepideal
- LR=0.0038, transfer_lr=0.095, fast_lr=0.474, rank=32, BS=48, 5ep
- AUTO_SCALE_MODE=none, REINIT_MODE=decay, MULTI_TILE_DIAG=True

## Variables (4 variants)
| Tag | seed | min_lr_rate |
|---|---|---|
| seed42            | 42 | 0    |
| seed43            | 43 | 0    |
| seed44            | 44 | 0    |
| seed42_lrfloor001 | 42 | 0.01 |

## Result Summary
- **seed42 (lr_floor=0)**: F1=83.57 정상 학습
- seed43: F1=83.63 정상
- seed44: F1=83.80 정상
- **seed42_lrfloor001**: F1=82.84 epoch 3까지 OK, 그 후 collapse
- → lr_floor가 epoch 4-5에 LR을 유지 → late cascade trigger
- 같은 setup에서도 seed/lr_floor 변형으로 outcome 갈림 (chaotic threshold-crossing)

## Files
- diag_{seed42,seed43,seed44,seed42_lrfloor001}.json
- multitile_plot1_norms.png, multitile_plot2_outliers.png
- summary_20260506_144818.json

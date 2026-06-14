# Higher fast_lr로 강제 cascade — bilinear hypothesis dose-response

**Date**: 2026-06-08  
**Launch**: ad-hoc (manual 4 GPU)

## Question
fast_lr을 증가시키면 η·σ_1 > 1 threshold 더 쉽게 cross → cascade 더 일찍/확실히 발생?
가설 맞다면 dose-response 곡선 나와야.

## Setup
- no_noise (A=B=constantstep6t1cgamma)
- seed=42, BS=48, 5ep, AUTO_SCALE=none
- minimal diag (G coherence + basic norms)

## Variables (4 runs)
| Run | seed | fast_lr | 기대 |
|---|---|---|---|
| seed42_flr_default | 42 | 0.474 | baseline (sometimes collapse) |
| seed44_rerun | 44 | 0.474 | seed44 stuck 재현성 test |
| seed42_flr0.7 | 42 | 0.7 | threshold push → 더 일찍 cascade |
| seed42_flr1.0 | 42 | 1.0 | threshold 강제 cross → very early cascade |

## Result Summary
- seed42 default: 정상 학습 (F1=83.59) — threshold X
- seed44 rerun: stuck (F1=7) — Pattern D 재현, ‖G‖ > 2.6 but ‖A·B‖=11.6 (no cascade)
- seed42 fast_lr=0.7: **cascade onset step 161** (epoch 0.09), ‖A·B‖_L11 = 7203
- seed42 fast_lr=1.0: **cascade onset step 60** (epoch 0.03), ‖A·B‖_L11 = 7810

→ **fast_lr 증가 → cascade 더 일찍** monotonic dose-response 확인 (bilinear hypothesis 정량 지지)

## Caveats
- High fast_lr은 cascade뿐만 아니라 learning 자체도 망가뜨림 (F1 모두 6-7)
- "bilinear cascade가 F1 망친 건지, high fast_lr이 다른 메커니즘으로 망친 건지" 인과 분리 불가

## Files
- diag_collapse_gcoh_20260608_052553_seed42_flr_default.json
- diag_collapse_gcoh_20260608_052553_seed42_flr{0p7,1p0}.json
- diag_collapse_gcoh_20260608_052553_seed44_rerun_min.json
- fastlr_push_cascade.png — log-scale ||A·B|| trajectories
- phase3_summary.png — F1 + ‖A·B‖ + threshold visualizations

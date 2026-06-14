# G coherence diag 도입 후 fast_lr ablation

**Date**: 2026-05-12  
**Launch**: ad-hoc (fine_bert_squad_lrtt.py 변경 후 manual 4 GPU run)  
**G coherence**: cos_G_prev, sigma1_G, sigma1_G_ratio 측정 추가 (per-step SVD)

## Question
fast_lr (η)에 따른 bilinear unstable mode 발화 정도 측정:
- η·σ_1(G) > 1 임계 검증
- σ_1/‖G‖_F dominance 측정 (random matrix 대비)

## Setup
- no_noise (A=B=constantstep6t1cgamma), C=constantstepideal
- seed=42, BS=48, 5ep
- 다른 hyperparam은 baseline과 동일
- DIAG_TILES="first_last", **FULL DIAG_GROUPS** (G coherence 외에도 erank/cells 등 포함)

## Variables (4 runs)
| fast_lr | Cascade onset (예상) |
|---|---|
| 0.474 (baseline) | rare/late |
| 0.1 | unstable mode 약화 |
| 0.05 | 더 약화 |
| 0.01 | 거의 stable |

## Result Summary
- cos_G_prev ≈ 0.81 (모든 fast_lr, 일관) — G 시간 coherence 매우 높음
- L11에서 σ_1/‖G‖_F ≈ 0.81 (random 0.07 대비) — top-mode dominant
- 4 runs 모두 stable (collapse 안 일어남) — but threshold X
- **σ_1/‖G‖_F formula 도출**: ‖G‖_F threshold = 1/(η·0.81)

## Files
- diag_collapse_gcoh_20260512_161217.json (baseline fast_lr=0.474)
- diag_collapse_gcoh_20260512_161437_fast_lr_0p{01,05,1}.json
- g_coherence_ablation.png — cos_G_prev, sigma1_G_ratio, ‖A·B‖ 4-run 비교

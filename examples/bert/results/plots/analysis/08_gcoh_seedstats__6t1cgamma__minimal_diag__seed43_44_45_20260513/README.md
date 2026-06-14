# G coherence + minimal-diag multi-seed (Pattern D 'stuck' 발견)

**Date**: 2026-05-13  
**Launch**: ad-hoc (manual)

## Question
- G coherence diag가 학습에 영향 주는지 (단순 read-only지만 CUDA non-determinism 매개)
- 같은 setup 다른 seed에서 outcome 분포

## Setup
- baseline과 동일 (no_noise, fast_lr=0.474)
- **DIAG_GROUPS minimal** (g5a/g5b/g3/g3b/g3c 등 disable; G coherence + 기본 norms만)
- ERANK_RATE_LIMIT_STEPS=0 → erank SVD 끔

## Variables (3 seeds)
seed = 43, 44, 45

## Result Summary
- **seed43**: F1=83.42 정상 학습
- **seed44**: F1=7.16 — Pattern D 'stuck' (epoch 1부터, ‖A‖(L0) 안 자람)
- **seed45**: F1=7.21 — Pattern D 'stuck'
- → 같은 setup 안에서 정상 1개 + Pattern D 2개
- 04번에서 seed44 정상 학습한 것과 outcome 다름 → diag code 추가가 outcome 변경시킴 (chaotic)

## Files
- diag_collapse_gcoh_20260513_030033_seed{43,44,45}_min.json

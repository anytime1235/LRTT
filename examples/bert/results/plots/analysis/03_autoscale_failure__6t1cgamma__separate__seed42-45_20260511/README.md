# AUTO_SCALE_MODE="separate" multi-seed test (Type 2 failure 발견)

**Date**: 2026-05-11  
**Launch**: scripts/investigate_autoscale.py

## Question
AUTO_SCALE_MODE="separate"가 saturation collapse (Pattern A)를 막는가?
가설: per-step LR normalization → 더 안정적인 학습 → no collapse

## Setup
- no_noise condition (A=B=constantstep6t1cgamma)
- LR=0.0038, transfer_lr=0.095, fast_lr=0.474, rank=32
- BS=48, 5 epochs, MIN_LR_RATE=0
- **AUTO_SCALE_MODE="separate"** (실험 변수)
- 다른 hyperparam은 1번과 동일

## Variables (4 seeds)
seed = 42, 43, 44, 45

## Result Summary
- **4/4 모두 Type 2 failure** — F1 6.96%, 7.02%, 7.21%, 7.27%
- 학습 자체가 시작 안 됨 (initial damage)
- Pattern A (saturation collapse)는 막혔지만 **다른 failure mode 도입** (Type 2)
- autoscale EMA가 처음부터 wrong direction 잡아서 회복 불가

## Files
- diag_separate_seed{42..45}.json
- summary_20260511_104414.json

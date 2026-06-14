# Planned: 11_ideal_multiseed__constantstepideal__seed42-45_20260610

**Status**: 진행 중 (현재 diag_ideal_multiseed/ 에서 실행 중, 끝나면 이 경로로 이동)

## Question
**모든 device를 constantstepideal로 설정** → bilinear collapse 일어나는가?
- 6T1C decay/drift 없는 진정한 ideal device
- 가설: bilinear math 자체가 원인 → ideal에서도 collapse 발생해야

## Setup
- A_DEVICE = B_DEVICE = C_DEVICE = constantstepideal
- AB_DEVICE = constantstepideal
- 다른 hyperparam은 동일 (LR=0.0038, transfer_lr=0.095, fast_lr=0.474)
- seed = 42, 43, 44, 45 (4 GPU)
- minimal diag (G coherence + basic norms)

## Expected Files (after completion)
- diag_ideal_seed{42..45}.json — per-seed diag log

## Why
이전 모든 collapse 분석은 6t1c-family device에서 수행됨.
사용자 지적: optuna data에서 constantstepideal도 ~15% collapse rate 보임.
→ controlled 실험으로 ideal에서도 collapse 메커니즘 동일한지 직접 검증.

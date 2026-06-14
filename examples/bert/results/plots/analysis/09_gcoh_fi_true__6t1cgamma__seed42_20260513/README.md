# FORWARD_INJECT=True 단일 seed test (self-limiting feedback 가설)

**Date**: 2026-05-13  
**Launch**: ad-hoc (single GPU)

## Question
FI=True에서 A·B가 forward에 들어감 (y = x·C + α·x·A·B) → A·B 폭증 시 loss feedback이 damp.
이게 LoRA가 안정한 이유 가설.
→ LRTT를 FI=True로 돌리면 collapse 사라지는가?

## Setup
- no_noise, seed=42, fast_lr=0.474, default hyperparams
- **FORWARD_INJECT=True**, FI_CONTINUOUS_ALPHA=False
- α는 lora_alpha 기본값 사용
- minimal diag

## Variables
single seed only (단일 run은 chaotic noise로 결론 어려움)

## Result Summary
- Epoch 1: F1=78.36 (좋게 시작)
- Epoch 2: F1=5.52 (collapse)
- ‖A·B‖_L11 = 7.56 (no cascade!) — bilinear cascade 없는데 F1 collapse
- → **FI=True도 collapse**하지만 다른 메커니즘 (direction collapse)
- 단일 seed라 chaotic 운인지 진짜 메커니즘인지 불분명

## Caveats
- FI=True를 위한 hyperparam은 별도 optuna search 안 됨 (FI=False optimal 사용)
- α (lora_alpha) 조정 미실험
- 진정한 검증은 FI=True optuna best params + multi-seed 필요

## Files
- diag_collapse_gcoh_fi_true_20260513_033925.json

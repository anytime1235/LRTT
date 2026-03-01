# Classifier Seed 정보

## 현재 설정: SEED = 42

### 📊 근거:

1. **프로젝트 표준 Seed:**
   - LRTT_transformer 프로젝트 전체에서 `SEED = 42` 사용
   - 모든 sweep 스크립트에서 일관되게 42 사용
   - 예시:
     ```
     sweep_sixt1c_lora_glue_adam.py: SEED = 42
     sweep_tikitaka_v2_bayesian.py: SEED = 42
     run_tikitaka_qkv_experiments.py: SEED = 42
     run_digital_merge_adamw.py: SEED = 42
     ```

2. **Classifier 초기화 방법:**
   ```python
   torch.manual_seed(SEED)  # SEED = 42
   nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
   nn.init.zeros_(model.classifier.bias)
   ```

3. **Frozen Classifier 실험에서의 의미:**
   - Seed=42로 고정 → 모든 trial이 동일한 classifier로 시작
   - 공정한 LoRA alpha 비교 가능
   - 초기 조건이 통제됨

### 🔍 Seed=42가 적절한가?

#### ✅ 적절함:
- 프로젝트 전체의 표준 seed
- 재현성 보장
- 다른 실험들과 일관성 유지

#### 대안 검토:
1. **다른 seed 값 (예: 0, 123, 3407)?**
   - 가능하지만 프로젝트 표준에서 벗어남
   - 특별한 이유 없다면 42 유지가 합리적

2. **여러 seed로 실험?**
   - Classifier frozen → seed 고정 필요
   - 여러 seed 테스트는 별도 실험으로 진행 가능
   - 하지만 현재 실험의 목적은 "lora_alpha 비교"
   - 따라서 classifier seed는 고정하는 것이 맞음

### 📌 결론:

**SEED = 42 사용이 적절합니다:**

1. **프로젝트 표준**: 모든 기존 실험과 동일
2. **재현성**: 동일한 조건으로 재실험 가능
3. **공정성**: 모든 trial이 동일한 frozen classifier
4. **일관성**: 다른 sweep 실험들과 비교 가능

### 🎯 현재 실험 설정:

```python
# sweep_sixt1c_lora_glue_frozen_classifier.py
SEED = 42

# create_glue_model():
if hasattr(model, 'classifier'):
    torch.manual_seed(SEED)  # 42로 고정
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
    nn.init.zeros_(model.classifier.bias)
```

**결과:**
- 모든 50 trials가 정확히 동일한 classifier로 시작
- LoRA alpha (0.01, 0.1, 1.0, 10.0, 100.0) 효과만 비교
- Classifier 변동 요인 제거됨

### ✅ 추천: 현재 설정 유지

특별한 이유가 없다면 **SEED = 42**로 진행하는 것이 가장 합리적입니다.

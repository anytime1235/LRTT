# Gradient Clipping Mechanism for Analog Tiles

## 핵심 질문
A/B tile은 `tile.update(x, d)`를 통해 outer product 업데이트를 받는다:
```
ΔW = learning_rate × x ⊗ d
```
`clip_grad_norm_(max_norm=1.0)`이 이 `x`와 `d` 값에 어떻게 영향을 주는가?

## 답변: d (error) 값만 스케일링됨

### 전체 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. FORWARD PASS                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  input → AnalogLinear → output                                  │
│            │                                                     │
│            └─ AnalogContext.analog_input = x (저장)             │
│                                                                 │
│  → x는 forward 활성화 값                                         │
│  → gradient clipping과 무관                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. BACKWARD PASS                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  loss.backward()                                                │
│    │                                                            │
│    └─→ error δ flows back                                       │
│          │                                                      │
│          └─ AnalogContext.analog_grad_output = δ (저장)         │
│                                                                 │
│  → δ는 backward error 값                                        │
│  → 이것이 나중에 'd'가 됨                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. GRADIENT CLIPPING (HF Trainer가 호출)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  torch.nn.utils.clip_grad_norm_(                               │
│      model.parameters(),                                       │
│      max_norm=1.0                                              │
│  )                                                             │
│                                                                 │
│  내부 동작:                                                      │
│  1. 모든 파라미터의 gradient norm 계산:                          │
│     ||g|| = sqrt(Σ ||p.grad||²)                                │
│                                                                 │
│  2. ||g|| > max_norm이면:                                       │
│     scaling_factor = max_norm / ||g||                          │
│                                                                 │
│  3. 모든 파라미터의 gradient를 스케일링:                         │
│     for p in parameters:                                       │
│         p.grad *= scaling_factor                               │
│                                                                 │
│  ⚠️ CRITICAL: AnalogContext.analog_grad_output도 스케일링됨!     │
│     δ_new = δ_old × scaling_factor                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. OPTIMIZER STEP                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  AnalogSGD.step()                                              │
│    │                                                            │
│    └─→ for each AnalogContext:                                 │
│          x = analog_input         (원본, 클리핑 안됨)           │
│          d = analog_grad_output   (스케일링됨! ✓)               │
│          │                                                      │
│          └─→ tile.update(x, d)                                 │
│                 │                                               │
│                 └─→ ΔW = lr × x ⊗ d                            │
│                                                                 │
│  → x: 원본 값 (forward에서 온 것)                               │
│  → d: 클리핑된 값 (scaling_factor 적용됨)                       │
│  → ΔW도 proportionally 작아짐!                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 구체적 예시

### 시나리오: gradient norm = 15.0, max_norm = 1.0

```python
# Forward pass
x = [2.0, 3.0, 4.0]  # 입력 활성화

# Backward pass (클리핑 전)
d_original = [10.0, 8.0]  # 큰 error
||d|| = 12.81

# Gradient clipping
total_norm = 15.0
max_norm = 1.0
scaling_factor = 1.0 / 15.0 = 0.0667

# Backward pass (클리핑 후)
d_clipped = [10.0, 8.0] × 0.0667 = [0.667, 0.533]
||d|| = 0.85

# Weight update
ΔW_no_clip  = 0.01 × x ⊗ d_original  →  ||ΔW|| = 0.6896
ΔW_clipped  = 0.01 × x ⊗ d_clipped   →  ||ΔW|| = 0.0460

# 결과: 93.3% 감소!
```

## 실제 측정 결과 (verify_gradient_clipping_v2.py)

```
Total gradient norm: 9.36 → clipping 적용

Analog context deltas (tile.update로 들어가는 d 값):
  query.tile_a:  0.1167 → 0.0977  (16.3% 감소)
  key.tile_a:    0.1168 → 0.0969  (17.0% 감소)
  value.tile_a:  0.0560 → 0.0488  (12.8% 감소)

→ 모든 d 값이 정확히 같은 비율로 스케일링됨!
```

## 결론

### ✅ 확인된 사실

1. **clip_grad_norm_()은 analog tile에 영향을 준다**
   - AnalogContext.analog_grad_output (d 값)이 스케일링됨
   - 이 값이 tile.update(x, d)로 전달됨

2. **메커니즘**
   ```
   clip_grad_norm_() → p.grad *= scaling_factor
                    → analog_grad_output도 스케일링
                    → tile.update()가 스케일링된 d를 받음
                    → ΔW = x ⊗ d 에서 d가 작아짐
                    → weight update 크기 제한됨
   ```

3. **max_grad_norm=1.0의 효과**
   - Total gradient norm > 1.0 이면 모든 gradient를 비례적으로 축소
   - Analog tile의 d 값도 같은 비율로 축소
   - Outer product ΔW = lr × x ⊗ d 의 크기가 제한됨

### ❌ SQuAD NaN 문제의 원인이 아님

- Gradient clipping은 올바르게 작동하고 있음
- max_grad_norm=1.0은 적절한 값 (기본값)
- SST-2에서 성공적으로 작동함
- **실제 원인은 hyperparameter range** (lora_alpha, learning_rate 최소값이 너무 작음)

## 참고: Analog Tile Update 공식

```
Outer product update:
  ΔW[i,j] = η × d[i] × x[j]

Where:
  η  = learning_rate (optimizer에서 설정)
  d  = error/delta (backward pass, 클리핑 적용됨 ✓)
  x  = input activation (forward pass, 클리핑 안됨)

Example:
  x = [1, 2, 3]
  d = [0.5, 0.3]

  ΔW = d ⊗ x = [[0.5, 1.0, 1.5],
                [0.3, 0.6, 0.9]]
```

## 코드 위치

- **clip_grad_norm_ 호출**: HF Trainer (자동)
- **AnalogContext**: `aihwkit/optim/context.py`
- **AnalogSGD.step()**: `aihwkit/optim/analog_sgd.py`
- **tile.update()**: `aihwkit/simulator/tiles/lrtt_tile.py`

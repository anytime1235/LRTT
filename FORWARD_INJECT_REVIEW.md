# Forward_inject=True 동작 검증 완료

## 요약

forward_inject=True는 **의도한 설계대로 정확하게 동작**합니다.

## 검증 결과

### 1. Forward Pass 구현 (lrtt_controller.py:1701-1725)

```python
def _forward_inject_analog_unified(self, x, in_trans, out_trans):
    # 1) Input normalization
    x_bf = x.t() if in_trans else x  # [batch, x_size]

    # 2) 세 개의 tile을 각각 forward (analog read 순서: B → A → C)
    g = self.tile_b.forward(x_bf)      # [batch, rank] = B·x
    y_ab = self.tile_a.forward(g)      # [batch, d_size] = A·(B·x)
    y_c = self.tile_c.forward(x_bf)    # [batch, d_size] = C·x

    # 3) C tile과 AB tile 출력 합산
    y = y_c + self.lora_alpha * y_ab   # y = C·x + α·A·(B·x) ✅

    # 4) Output transpose
    return y.t() if out_trans else y
```

**✅ 검증**: C tile 출력과 AB tile 출력이 **올바르게 합산**됩니다.

### 2. Backward Pass 구현 (lrtt_tile.py:455-472)

#### forward_inject=True인 경우:

```python
# Line 453: C gradient 계산
xg_c = self.tile_c.backward(d_bf)       # C^T·d

# Line 457-458: AB gradient 계산
da = self.tile_a.backward(d_bf)          # A^T·d [batch, rank]
xg_ab = self.tile_b.backward(da)         # B^T·(A^T·d) [batch, x_size]

# Line 459: 합산된 gradient를 upstream으로 전달
x_grad = xg_c + self.lora_alpha * xg_ab  # ✅ C + AB 모두 참여
```

**✅ 검증**: Upstream에 **C gradient + AB gradient**가 모두 전달됩니다.

#### forward_inject=False인 경우 (비교):

```python
# Line 453: C gradient만 계산
xg_c = self.tile_c.backward(d_bf)

# Line 463-464: AB gradient는 계산하지만 local용으로만 저장
da = self.tile_a.backward(d_bf)
xg_ab = self.tile_b.backward(da)

# Line 472: Upstream에는 C gradient만 전달
x_grad = xg_c  # ❌ AB gradient는 upstream 불참
```

### 3. LoRA Update 동작 (lrtt_controller.py:653-726)

forward_inject=True일 때 LoRA chain rule이 **자연스럽게 적용**됩니다:

```python
def _ab_weight_update_lora(self, x, d, lr, ...):
    # 1) Projections (analog path)
    XB = self.tile_b.forward(x)     # B·x
    DA = self.tile_a.backward(d)    # A^T·d

    # 2) Effective learning rate
    lr_eff = lr * self.lora_alpha   # α 스케일링

    # 3) ΔA = -lr_eff · d^T · (B·x)
    self.tile_a.update(XB, d)

    # 4) ΔB = -lr_eff · (A^T·d)^T · x
    self.tile_b.update(x, DA)
```

**✅ 검증**: LoRA chain rule이 그대로 사용됩니다.
- x와 d는 forward/backward에서 자연스럽게 전달됨
- 별도의 stored gradient 불필요
- Gradient flow가 전체 네트워크에 연결됨

### 4. 핵심 차이점 비교표

| 항목 | forward_inject=True | forward_inject=False |
|------|---------------------|----------------------|
| **Forward 출력** | `y = C·x + α·A·(B·x)` | `y = C·x` |
| **Loss 계산** | AB가 loss에 영향 ✅ | AB가 loss에 영향 없음 ❌ |
| **Upstream gradient** | `C^T·d + α·B^T·(A^T·d)` | `C^T·d` only |
| **AB gradient flow** | 전체 네트워크에 전파 ✅ | Local만 (stored) ❌ |
| **LoRA update** | 자연스러운 chain rule ✅ | Stored gradient 필요 |

## MDMLP train_analog.py 변경사항

### 변경 전 (불일치 상태):
```python
# Line 455
reinit_mode="hybrid"            # A=0, B decayed (for forward injection)
# Line 462
update_mode="lora"              # LoRA chain rule (expects AB in forward)
# Line 464
forward_inject = False          # ❌ 불일치: AB가 forward 불참
```

### 변경 후 (정렬 완료):
```python
# Line 455
reinit_mode="hybrid"            # A=0, B decayed
# Line 462
update_mode="lora"              # LoRA chain rule
# Line 466-468
# Enable forward injection: y = Cx + α·A·(B·x)
# Required for proper gradient flow with update_mode="lora" and reinit_mode="hybrid"
forward_inject = forward_inject  # ✅ 정렬: AB가 forward 참여
```

### CLI 인터페이스 추가:
```bash
# Forward injection 활성화 (기본값: True)
python train_analog.py --forward-inject ...

# Forward injection 비활성화
python train_analog.py --no-forward-inject ...
```

## 결론

### ✅ forward_inject=True는 다음을 보장합니다:

1. **Forward Pass**: `y = C·x + α·A·(B·x)` 정확히 계산
2. **Backward Pass**: C와 AB 모두의 gradient가 upstream 전달
3. **Loss 영향**: AB tile이 실제로 loss 계산에 참여
4. **LoRA Update**: Chain rule이 자연스럽게 동작
5. **Hybrid Reinit**: A=0에서 시작, gradient로 점진적 구축

### ✅ MDMLP 설정이 이제 올바르게 정렬되었습니다:

- `reinit_mode="hybrid"` ← A=0, B decayed
- `update_mode="lora"` ← LoRA chain rule
- `forward_inject=True` ← AB path 활성화

이제 **gradient가 C와 AB path 모두를 통해 흐르며**, hybrid reinit가 의도한 대로 동작합니다.

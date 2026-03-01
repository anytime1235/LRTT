# Analog Tile Backward Hook 메커니즘

## 질문
> "그렇게 backward되는 설정이 있는건가? 저렇게 W 단위로 update 어떤 설정인건가?"

## 답변: 실제 코드 분석

### 1. 핵심 코드 위치

**파일**: `/aihwkit/simulator/tiles/inference_torch.py`

```python
# Line 163-182
def set_scales(self, scales: Tensor) -> None:
    """Set all scales with a new scale."""

    super().set_scales(scales)

    # - Remove old hook
    if self._backward_hook_handle is not None:
        self._backward_hook_handle.remove()

    # 🔥 핵심: Backward hook 정의
    def hook(grad: Tensor) -> Tensor:
        return grad / scales.to(grad.device).view(-1, 1) ** 2

    # 🔥 Hook 등록
    if self.tile.weight.requires_grad:
        self._backward_hook_handle = self.tile.register_weight_hook(hook)
```

### 2. 언제 활성화되나?

#### Configuration 설정

**파일**: `lora_training_glue/sixt1c_config.py`

```python
# Line 88-96
rpu_config = TorchInferenceRPUConfig()

# 🎯 이 설정이 backward hook을 활성화!
rpu_config.mapping.weight_scaling_omega = 1.0      # omega > 0이면 활성화
rpu_config.mapping.weight_scaling_columnwise = True  # Per-column scaling
```

#### 활성화 조건

**파일**: `aihwkit/simulator/tiles/periphery.py`

```python
# Line 464-480 (set_weights 내부)
def set_weights(..., apply_weight_scaling=True, weight_scaling_omega=None):
    omega = weight_scaling_omega or mapping.weight_scaling_omega

    # 🎯 omega > 0이면 weight scaling 적용
    if omega is not None and omega > 0:
        # Apply the scaling
        if mapping.weight_scaling_columnwise:
            weight_max, _ = torch_max(abs(combined_weights), 1, keepdim=True)
        else:
            weight_max = torch_max(abs(combined_weights)).view(1)

        alpha = weight_max / omega
        alpha[alpha == 0.0] = 1.0  # 🛡️ Zero 보호!

        combined_weights = combined_weights / alpha  # W를 normalize

        self.set_scales(alpha)  # 🔥 여기서 backward hook 등록!
```

### 3. Forward Pass

#### Weight Normalization

```python
# set_weights()에서:
W_normalized = W / alpha

# tile에 저장되는 것은 W_normalized
```

#### Output Scaling

```python
# inference_torch.py Line 227
out = self.apply_out_scaling(out, tensor_view)

# periphery.py Line 785
return values * self.out_scaling_alpha.view(*tensor_view)
```

**전체 Forward:**
```
1. tile.forward(x)로 계산: y_norm = x @ W_normalized
2. apply_out_scaling(): y = y_norm × alpha
3. 결과: y = x @ W (수학적으로 동일!)
```

### 4. Backward Pass

#### Backward Hook 작동

```python
def hook(grad: Tensor) -> Tensor:
    return grad / scales.to(grad.device).view(-1, 1) ** 2
```

**왜 `scales^2`로 나눌까?**

이유:
- Forward: `y = (x @ W_normalized) × alpha`
- PyTorch autograd가 계산하는 gradient:
  ```
  ∂L/∂W_normalized = (∂L/∂y × alpha) @ x^T
  ```
- 우리가 원하는 것: `∂L/∂W` (actual weight)
  ```
  W = W_normalized × alpha
  ∂L/∂W = ∂L/∂W_normalized / alpha
  ```
- 하지만 이미 `∂L/∂y`에 alpha가 곱해져 있음!
- 따라서: `grad_W / alpha^2`

**실제로는:**
```python
# PyTorch가 계산한 gradient
grad_raw = ∂L/∂y @ x^T × alpha  # (output scaling 때문에 alpha 추가)

# Hook이 보정
grad_corrected = grad_raw / alpha^2
                = (∂L/∂y @ x^T × alpha) / alpha^2
                = ∂L/∂y @ x^T / alpha
                ≈ 올바른 gradient scale
```

### 5. 전체 흐름

```
┌─────────────────────────────────────────┐
│ 1. set_weights() 호출                   │
│    - apply_weight_scaling=True          │
│    - omega=1.0 (from rpu_config)        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. Weight normalization                 │
│    alpha = max(|W|) / omega             │
│    alpha[alpha==0] = 1.0                │
│    W_normalized = W / alpha             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. set_scales(alpha) 호출               │
│    - mapping_scales = alpha 저장        │
│    - register_weight_hook(hook) 등록!   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. Forward pass                          │
│    y = (x @ W_normalized) × alpha       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 5. Backward pass                         │
│    grad_raw = PyTorch autograd 계산     │
│    grad_corrected = hook(grad_raw)      │
│                   = grad_raw / alpha^2  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 6. Optimizer step                        │
│    W -= lr × grad_corrected             │
└─────────────────────────────────────────┘
```

### 6. Configuration Parameters

#### MappingParameter 설정

```python
class MappingParameter:
    weight_scaling_omega: float = 0.0  # Default: OFF
    # ↑ 0이면 backward hook 없음!
    # ↑ > 0이면 backward hook 등록!

    weight_scaling_columnwise: bool = False
    # ↑ True: 각 output row별로 normalize
    # ↑ False: 전체 weight matrix에서 max 사용

    weight_scaling_lr_compensation: bool = False
    # ↑ True: lr도 보정 (mapping_lr_scale)
```

#### sixt1c 설정 (default)

```python
# sixt1c_config.py
rpu_config.mapping.weight_scaling_omega = 1.0         # ✅ ON!
rpu_config.mapping.weight_scaling_columnwise = True   # ✅ Per-column
```

### 7. 실제 예시

```python
# 초기 weight
W = [[0.5, 0.8],
     [0.3, 0.6]]

# Step 1: Normalization (omega=1.0, columnwise=True)
alpha = [max(|0.5|, |0.8|) / 1.0,   # = 0.8
         max(|0.3|, |0.6|) / 1.0]   # = 0.6

W_normalized = [[0.5/0.8, 0.8/0.8],   # = [[0.625, 1.0],
                [0.3/0.6, 0.6/0.6]]   #    [0.5,   1.0]]

# Step 2: Forward
y = (x @ W_normalized) × alpha  # alpha는 [0.8, 0.6]

# Step 3: Backward (gradient 10^5 도착)
grad_raw = 10^5 (from PyTorch autograd)

# Hook 작동
grad_corrected = 10^5 / alpha^2
               = 10^5 / [0.64, 0.36]
               ≈ [1.56e5, 2.78e5]  # Row별로 다르게 보정!

# Step 4: Update
W -= lr × grad_corrected
```

### 8. 정리

**질문: "그렇게 backward되는 설정이 있는건가?"**

✅ **있습니다!**

1. **Configuration**:
   ```python
   rpu_config.mapping.weight_scaling_omega = 1.0  # > 0이면 ON
   ```

2. **Activation**:
   ```python
   set_weights(apply_weight_scaling=True)
   → set_scales(alpha)
   → register_weight_hook(hook)
   ```

3. **Hook Function**:
   ```python
   def hook(grad):
       return grad / scales^2
   ```

**질문: "저렇게 W 단위로 update 어떤 설정인건가?"**

✅ **PyTorch standard autograd + custom hook!**

- PyTorch autograd가 W_normalized에 대한 gradient 계산
- Backward hook이 이를 actual W gradient로 변환
- Optimizer가 W를 직접 업데이트
- 다음 forward시 다시 normalize됨

**핵심**:
- Training 중에는 W가 일반 Parameter처럼 업데이트됨
- 하지만 forward마다 normalize되고
- Backward hook이 gradient를 보정해서
- Conductance는 항상 hardware 범위 유지!

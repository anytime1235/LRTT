# LRTT: 현재 Weight를 사용한 Update - Chain Rule 증명

## 질문
> "현재 weight를 기준으로 update하는것 아닌가? 그러면 저 수식이 안맞는것 아닌가?"

## 답변: 정확히 맞습니다! 이것이 Chain Rule입니다!

## 1. LoRA Forward Pass

### 수식

```
output = input @ B @ A + base_output
       = input @ W_lora + base_output

where W_lora = B @ A
```

### 코드

```python
# lora_training_glue/run_glue.py의 실제 forward
x = input                    # [batch, x_size]
h = self.lora_A(x)          # [batch, rank] = x @ A^T (in LoRA, A is transposed)
y = self.lora_B(h)          # [batch, d_size] = h @ B^T
output = y + base_output
```

**LRTT의 경우 (A, B가 analog tile):**
```python
# lrtt_controller.py forward
x = input                        # [batch, x_size]
XA = tile_a.forward(x)          # [batch, rank] = x @ A
Y = tile_b.forward(XA)          # [batch, d_size] = XA @ B
output = Y + base_output
```

## 2. Gradient Computation (Chain Rule)

### ∂L/∂A 계산

**Chain rule:**
```
∂L/∂A = ∂L/∂output × ∂output/∂A

where:
  output = (input @ A) @ B

∂output/∂A = ?
```

**미분 계산:**
```
Let: h = input @ A         [batch, rank]
     output = h @ B        [batch, d_size]

∂output/∂A를 계산하려면:
  output[i,j] = Σ_k h[i,k] × B[k,j]
              = Σ_k (Σ_m input[i,m] × A[m,k]) × B[k,j]

∂output[i,j]/∂A[m,k] = input[i,m] × B[k,j]

따라서:
∂L/∂A[m,k] = Σ_{i,j} (∂L/∂output[i,j]) × input[i,m] × B[k,j]
            = Σ_i (Σ_j ∂L/∂output[i,j] × B[k,j]) × input[i,m]
            = Σ_i (∂L/∂output @ B^T)[i,k] × input[i,m]
            = input^T @ (∂L/∂output @ B^T)
```

**행렬 형태:**
```
∂L/∂A = input^T @ (∂L/∂output @ B^T)
      = input^T @ (d @ B^T)
      = input^T @ (tile_b.backward(d))
      = x^T @ DA

Or equivalently (for batch):
∂L/∂A = (tile_b.backward(d))^T @ input
      = DA^T @ x

Wait, 이게 B update 식이네?
```

**다시 정리 (LRTT convention):**

LRTT에서는:
- tile_a: A matrix [d_size, rank]
- tile_b: B matrix [rank, x_size]
- Forward: y = x @ B @ A (실제로는 이 순서)

Let me recalculate for LRTT:

```
Forward:
  h = x @ B         [batch, rank]
  y = h @ A         [batch, d_size]

∂L/∂A:
  ∂y/∂A = h^T = (x @ B)^T = B^T @ x^T
  ∂L/∂A = (x @ B)^T @ ∂L/∂y
        = (tile_b.forward(x))^T @ d
        = XB^T @ d

In update form (batch average):
  ΔA = -lr × (d^T @ XB) / batch_size
```

### ∂L/∂B 계산

```
Forward:
  h = x @ B         [batch, rank]
  y = h @ A         [batch, d_size]

∂L/∂B:
  ∂y/∂B를 계산:
  y[i,j] = Σ_k h[i,k] × A[k,j]
         = Σ_k (Σ_m x[i,m] × B[m,k]) × A[k,j]

  ∂y[i,j]/∂B[m,k] = x[i,m] × A[k,j]

  ∂L/∂B[m,k] = Σ_{i,j} (∂L/∂y[i,j]) × x[i,m] × A[k,j]
              = Σ_i x[i,m] × (Σ_j ∂L/∂y[i,j] × A[k,j])
              = Σ_i x[i,m] × (∂L/∂y @ A^T)[i,k]
              = x^T @ (d @ A^T)
              = x^T @ (tile_a.backward(d))
              = x^T @ DA

In update form:
  ΔB = -lr × (DA^T @ x) / batch_size
```

## 3. LRTT Update 수식 검증

### A Tile Update

```python
# Code
XB = tile_b.forward(x)      # x @ B (현재 B weight 사용!)
tile_a.update(XB, d)        # ΔA = -lr × (d^T @ XB)

# 수식 확인
ΔA = -lr × (d^T @ XB)
   = -lr × (d^T @ (x @ B))
   = -lr × ((d^T @ x) @ B)? NO!

# 올바른 계산:
XB = x @ B                   [batch, rank]
d^T @ XB = d^T @ (x @ B)

Matrix dimensions:
  d: [batch, d_size]
  d^T: [d_size, batch]
  XB: [batch, rank]
  d^T @ XB: [d_size, rank] ✓

이것이 정확히 ∂L/∂A!
```

### B Tile Update

```python
# Code
DA = tile_a.backward(d)     # d @ A^T (현재 A weight 사용!)
tile_b.update(x, DA)        # ΔB = -lr × (DA^T @ x)

# 수식 확인
ΔB = -lr × (DA^T @ x)
   = -lr × ((d @ A^T)^T @ x)
   = -lr × (A @ d^T @ x)    [rank, x_size]

Wait, 차원이 안 맞는다. 다시 확인:

DA = tile_a.backward(d)
   = d @ A^T                [batch, rank]

DA^T = (d @ A^T)^T
     = A @ d^T              [rank, batch]

DA^T @ x = (A @ d^T) @ x    [rank, x_size] ✓

이것이 정확히 ∂L/∂B!
```

## 4. 핵심 포인트: "현재 weight 사용"이 맞는 이유

### Forward Pass (현재 weight로 계산)

```python
Step 1: x → tile_b (uses current B) → XB
Step 2: XB → tile_a (uses current A) → y
```

### Backward Pass (현재 weight로 gradient 계산)

```python
Step 1: d → tile_a.backward (uses current A) → DA = d @ A^T
Step 2: d, XB → compute ∂L/∂A = d^T @ XB
```

**이것이 chain rule!**

### Why it works

```
Forward에서 계산한 output이:
  y = f(x; A, B) = (x @ B) @ A

이 output으로 loss 계산:
  L = loss(y, target)

Backward에서:
  ∂L/∂A = ∂L/∂y × ∂y/∂A
        = d × (x @ B)^T
        = d^T @ (x @ B)
        = d^T @ XB

여기서 (x @ B)는 forward에서 계산한 값!
즉, **forward에서 사용한 current B를 backward에서도 사용!**

이것이 autograd의 원리!
```

## 5. Analog Tile의 특수성

### tile.forward()와 tile.backward()

```python
# Forward
XB = tile_b.forward(x)
# 내부적으로: return x @ W_b (current weight)

# Backward
DA = tile_a.backward(d)
# 내부적으로: return d @ W_a^T (current weight)
```

**핵심:**
- forward()와 backward()가 **같은 weight**를 사용!
- 이것이 정확히 chain rule이 요구하는 것!
- Forward에서 사용한 weight로 backward 계산해야 함!

## 6. 전체 흐름 검증

```python
# Step 0: 현재 weight 상태
A = [...] # [d_size, rank]
B = [...] # [rank, x_size]

# Step 1: Forward (현재 weight 사용)
h = x @ B          # tile_b.forward(x), uses current B
y = h @ A          # tile_a.forward(h), uses current A
loss = criterion(y, target)

# Step 2: Backward (현재 weight 사용)
d = ∂L/∂y          # from autograd

# A gradient 계산 (chain rule)
∂L/∂A = d^T @ h    # h was computed with current B!
      = d^T @ (x @ B)
      = d^T @ XB

# B gradient 계산 (chain rule)
∂L/∂B = (d @ A^T)^T @ x    # A is current A!
      = DA^T @ x

# Step 3: Update
A -= lr × ∂L/∂A    # tile_a.update(XB, d)
B -= lr × ∂L/∂B    # tile_b.update(x, DA)

# Step 4: 다음 iteration
# 이제 updated A, B가 "current weight"가 됨!
# 다시 Step 1으로...
```

## 7. 수식 정리

### Gradient Descent with Chain Rule

```
Iteration t:

Current weights: A_t, B_t

Forward:
  y_t = f(x; A_t, B_t) = (x @ B_t) @ A_t

Loss:
  L_t = loss(y_t, target)

Backward (chain rule 사용):
  ∂L_t/∂A = ∂L_t/∂y_t × ∂y_t/∂A
          = d_t^T @ (x @ B_t)    ← B_t 사용!

  ∂L_t/∂B = ∂L_t/∂y_t × ∂y_t/∂B
          = (d_t @ A_t^T)^T @ x   ← A_t 사용!

Update:
  A_{t+1} = A_t - lr × ∂L_t/∂A
  B_{t+1} = B_t - lr × ∂L_t/∂B

다음 iteration:
  A_{t+1}, B_{t+1}이 새로운 "current weight"
```

## 8. 핵심 정리

### 질문: "현재 weight 기준으로 update하면 수식이 안 맞지 않나?"

### ✅ 답변: 정확히 맞습니다!

**이것이 바로 chain rule입니다!**

1. **Forward**: 현재 weight (A_t, B_t)로 output 계산
2. **Backward**: 현재 weight (A_t, B_t)로 gradient 계산
3. **Update**: Gradient를 사용해서 weight 변경
4. **Next Iteration**: Updated weight가 새로운 "current"

### 왜 맞는가?

```
∂L/∂A를 계산할 때:

  y = (x @ B) @ A 에서

  ∂y/∂A를 계산하려면 B가 필요!

  어떤 B? → Forward에서 사용한 B! (A를 미분할 때 B는 상수)

따라서:
  ∂L/∂A = d^T @ (x @ B_current)

이것이 정확한 gradient!
```

### 수식 검증

```
LRTT Update:

tile_a.update(XB, d)
  where XB = tile_b.forward(x) = x @ B_current

  ΔA = -lr × (d^T @ XB)
     = -lr × (d^T @ (x @ B_current))
     = -lr × ∂L/∂A  ✓

tile_b.update(x, DA)
  where DA = tile_a.backward(d) = d @ A_current^T

  ΔB = -lr × (DA^T @ x)
     = -lr × ((d @ A_current^T)^T @ x)
     = -lr × (A_current @ d^T @ x)
     = -lr × ∂L/∂B  ✓
```

## 결론

**"현재 weight를 기준으로"가 정확히 맞는 방법입니다!**

- 이것이 표준 gradient descent
- 이것이 chain rule의 정의
- 이것이 PyTorch autograd가 하는 것
- LRTT도 정확히 같은 방식!

**수식이 완벽하게 맞습니다!** ✓

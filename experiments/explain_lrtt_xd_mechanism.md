# LRTT A,B Tile Update: X, D는 어떻게 결정되는가?

## 질문
> "lrtt_controller.py의 a,b weight update방식에서 A tile에 입력, B로 읽은 입력이 X,D로 들어가는데 이 XD가 어떻게 결정되는것이며, gradient기반으로하면 이 두개가 모두 1이되는것아닌가? 그러면 무조건 pulse update되는것아닌가?"

## 핵심 오해 해소

### ❌ 잘못된 이해
"Gradient 기반 → X, D가 모두 1 → 무조건 pulse update"

### ✅ 올바른 이해
"Gradient의 **실제 값**을 사용 → X, D는 실수 벡터 → 값에 비례한 update"

## 1. Analog Tile Update의 실제 동작

### tile.update(x, d) 의미

```python
# Analog tile update는 outer product 기반!
tile.update(x, d)

# 내부적으로 수행:
# W -= lr × (x^T @ d)  # Outer product!
```

**구체적 예시:**

```python
x = torch.tensor([[0.1, 0.5, -0.3]])  # [batch=1, in_features=3]
d = torch.tensor([[0.8, -0.2]])       # [batch=1, out_features=2]

# Outer product
delta_W = x.t() @ d  # [3, 2]
```

계산:
```
delta_W = [[0.1],     [[0.8, -0.2]]
           [0.5],  @
           [-0.3]]

        = [[0.1×0.8,  0.1×(-0.2)],     [[0.08,  -0.02],
           [0.5×0.8,  0.5×(-0.2)],  =   [0.40,  -0.10],
           [-0.3×0.8, -0.3×(-0.2)]]     [-0.24,  0.06]]
```

**W -= lr × delta_W로 업데이트!**

### 핵심 포인트

1. **X, D는 1이 아니라 실제 값들!**
   - x = [0.1, 0.5, -0.3, ...]  (실수)
   - d = [0.8, -0.2, 0.3, ...]  (실수)

2. **"무조건 pulse"가 아니라 "값에 비례"!**
   - x[i] × d[j]가 크면 → W[j,i] 크게 변화
   - x[i] × d[j]가 작으면 → W[j,i] 작게 변화
   - x[i] ≈ 0 or d[j] ≈ 0이면 → W[j,i] 거의 안 변함

3. **Coincidence detection**
   - x[i]와 d[j]가 **동시에** 활성화될 때만 update
   - 이것이 analog hardware의 Hebbian learning!

## 2. LRTT Controller의 실제 X, D

### 코드 분석

**파일**: `lrtt_controller.py` Line 670-720

```python
def ab_weight_update(self, x, d, bias_grad=None, in_trans=False, out_trans=False, lr=1.0):
    """LoRA-style A/B update using chain rule."""

    # 1) 입력 정규화
    if in_trans:
        x = x.t()
    if out_trans:
        d = d.t()
    # 이제 x: [batch, x_size], d: [batch, d_size]

    # 2) Projections 계산
    with torch.no_grad():
        XB = self.tile_b.forward(x)   # [batch, rank] = x @ B
        DA = self.tile_a.backward(d)  # [batch, rank] = d @ A^T

    # 3) A tile update
    # ΔA = -lr × d^T @ XB
    self.tile_a.update(XB, d)

    # 4) B tile update
    # ΔB = -lr × DA^T @ x
    self.tile_b.update(x, DA)
```

### X, D의 실제 값

#### A Tile Update

```python
tile_a.update(XB, d)
```

여기서:
- **XB** = tile_b.forward(x) = x @ B
  - x: [batch, x_size] = input (예: [32, 8])
  - B: [rank, x_size] (예: [8, 128])
  - XB: [batch, rank] (예: [32, 8])
  - **XB의 값**: 실제 숫자들! [0.5, -0.3, 0.8, ...]

- **d** = gradient from output
  - d: [batch, d_size] (예: [32, 128])
  - **d의 값**: ∂L/∂output! [100, -50, 200, ...]

**Update 계산:**
```
ΔA = -lr × (d^T @ XB)  # [d_size, rank]
   = -lr × outer_product(d, XB)
```

#### B Tile Update

```python
tile_b.update(x, DA)
```

여기서:
- **x** = input (raw)
  - x: [batch, x_size] (예: [32, 128])
  - **x의 값**: [0.1, 0.3, -0.2, ...]

- **DA** = tile_a.backward(d) = d @ A^T
  - d: [batch, d_size] (예: [32, 128])
  - A^T: [d_size, rank]
  - DA: [batch, rank] (예: [32, 8])
  - **DA의 값**: [50, -30, 80, ...]

**Update 계산:**
```
ΔB = -lr × (DA^T @ x)  # [rank, x_size]
   = -lr × outer_product(DA, x)
```

## 3. 구체적 예시

### 실제 값으로 계산

```python
# 가정
batch = 2
x_size = 3
d_size = 4
rank = 2
lr = 0.01

# Input
x = torch.tensor([[0.5, 0.3, -0.2],
                  [0.1, 0.8,  0.4]])  # [2, 3]

# Gradient from loss
d = torch.tensor([[100,  -50,  80, -30],
                  [120, -60, 100, -40]])  # [2, 4]

# Current weights
B = torch.tensor([[ 0.1,  0.2, 0.3],
                  [-0.1,  0.4, 0.2]])  # [2, 3]

A = torch.tensor([[ 0.3, -0.2],
                  [ 0.4,  0.1],
                  [-0.2,  0.5],
                  [ 0.1,  0.3]])  # [4, 2]
```

### Step 1: A Tile Update

```python
# Projection
XB = x @ B.t()  # [2, 2]
# = [[0.5, 0.3, -0.2],     [[0.1, -0.1],
#    [0.1, 0.8,  0.4]]  @   [0.2,  0.4],
#                            [0.3,  0.2]]
# = [[0.5×0.1+0.3×0.2-0.2×0.3,  0.5×(-0.1)+0.3×0.4+(-0.2)×0.2],
#    [0.1×0.1+0.8×0.2+0.4×0.3,   0.1×(-0.1)+0.8×0.4+0.4×0.2]]
# = [[0.05,  0.03],
#    [0.29,  0.39]]

# A update: tile_a.update(XB, d)
# ΔA = -lr × (d^T @ XB)  # [4, 2]

d_t_XB = d.t() @ XB  # [4, 2]
# = [[100, 120],      [[0.05, 0.03],
#    [-50, -60],   @   [0.29, 0.39]]
#    [80, 100],
#    [-30, -40]]
# = [[100×0.05+120×0.29,  100×0.03+120×0.39],
#    [...]
# = [[39.8,  49.8],
#    [-20.0, -24.9],
#    [33.0,  41.4],
#    [-13.1, -16.5]]

delta_A = -lr × d_t_XB
        = -0.01 × [[39.8, 49.8], ...]
        = [[-0.398, -0.498],
           [0.200,  0.249],
           [-0.330, -0.414],
           [0.131,  0.165]]

# A는 이만큼 변화!
```

### Step 2: B Tile Update

```python
# Projection
DA = d @ A  # [2, 2]
# = [[100, -50, 80, -30],     [[0.3, -0.2],
#    [120, -60, 100, -40]]  @  [0.4,  0.1],
#                               [-0.2,  0.5],
#                               [0.1,  0.3]]
# = [[7.0,  21.0],
#    [8.0,  26.0]]

# B update: tile_b.update(x, DA)
# ΔB = -lr × (DA^T @ x)  # [2, 3]

DA_t_x = DA.t() @ x  # [2, 3]
# = [[7.0, 8.0],      [[0.5, 0.3, -0.2],
#    [21.0, 26.0]]  @  [0.1, 0.8,  0.4]]
# = [[7.0×0.5+8.0×0.1,  7.0×0.3+8.0×0.8,  7.0×(-0.2)+8.0×0.4],
#    [21.0×0.5+26.0×0.1, 21.0×0.3+26.0×0.8, 21.0×(-0.2)+26.0×0.4]]
# = [[4.3,  8.5,  1.8],
#    [13.1, 27.1, 6.2]]

delta_B = -lr × DA_t_x
        = -0.01 × [[4.3, 8.5, 1.8], ...]
        = [[-0.043, -0.085, -0.018],
           [-0.131, -0.271, -0.062]]

# B는 이만큼 변화!
```

## 4. 핵심 포인트

### ✅ X, D는 실제 값들

- **X (input)**: [0.5, 0.3, -0.2, ...] (실수 벡터)
- **D (gradient)**: [100, -50, 80, ...] (실수 벡터)
- **절대 "모두 1"이 아님!**

### ✅ Pulse는 값에 비례

```python
# Analog tile의 coincidence detection:
W[i,j] -= lr × (x[i] × d[j])

# 예시:
x[2] = 0.8, d[3] = 100  → W[3,2] -= lr × (0.8 × 100) = -0.8lr (큰 변화!)
x[5] = 0.01, d[7] = 50  → W[7,5] -= lr × (0.01 × 50) = -0.005lr (작은 변화)
x[8] = 0.0, d[9] = 200  → W[9,8] -= lr × (0.0 × 200) = 0 (변화 없음!)
```

### ✅ Gradient 크기가 중요

```python
# Gradient가 크면 (loss가 크면):
d = [1000, -500, 800, ...]  → 큰 update!

# Gradient가 작으면 (수렴하면):
d = [0.1, -0.05, 0.08, ...]  → 작은 update!
```

## 5. Hardware 관점

### Analog Crossbar Update

```
     x[0]   x[1]   x[2]
      ↓      ↓      ↓
d[0]→ ●─────●─────●  W[0,:]
      │     │     │
d[1]→ ●─────●─────●  W[1,:]
      │     │     │
d[2]→ ●─────●─────●  W[2,:]
```

**Coincidence detection:**
- x[i]와 d[j]가 동시에 활성화될 때만 W[j,i] update
- Update 크기 ∝ x[i] × d[j]
- 이것이 **Outer Product** 학습!

### Pulse 세기

```
Conductance change = f(x[i] × d[j])

x[i] × d[j] = 0.8 × 100 = 80   → Strong pulse!
x[i] × d[j] = 0.1 × 10 = 1     → Weak pulse
x[i] × d[j] = 0.01 × 5 = 0.05  → Very weak pulse
```

## 6. 정리

### 질문 1: "XD가 어떻게 결정되는가?"

✅ **PyTorch autograd + Forward projection!**

- **X**: Input 또는 Projected input (XB = x @ B)
- **D**: Gradient from loss (∂L/∂output)
- 둘 다 **실제 실수 값들의 벡터/행렬**

### 질문 2: "Gradient 기반이면 모두 1?"

✅ **아니오! Gradient의 실제 값 사용!**

- Gradient ≠ 1
- Gradient = ∂L/∂output의 실제 값 (음수, 양수, 큼, 작음 다 가능)

### 질문 3: "무조건 pulse update?"

✅ **아니오! 값에 비례한 update!**

- x[i] × d[j]가 크면 → 큰 update
- x[i] × d[j]가 작으면 → 작은 update
- x[i] ≈ 0 or d[j] ≈ 0 → 거의 update 안 됨

### 최종 정리

```
LRTT Update = Outer Product Learning

A tile: A -= lr × (d^T @ XB)
        where XB = x @ B (projection)
              d = ∂L/∂output (gradient)

B tile: B -= lr × (DA^T @ x)
        where DA = d @ A^T (projection)
              x = input

모든 값들은 실수!
Update는 값의 크기에 비례!
```

**이것이 LRTT의 핵심 메커니즘입니다!** 🎯

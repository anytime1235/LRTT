# Stochastic Pulse Update: Gradient 10^5를 어떻게 처리하는가?

## 질문
> "Gradient가 10^5 ~ 10^1 범위일 때 stochastic pulse 방식으로 할 때 lrtt_controller.py에서 A,B tile update를 이 gradient 기반으로 하면 무조건 max인데 어떻게 계산된다는 건가?"

## 핵심 오해

### ❌ 잘못된 이해
"Gradient 10^5 → Conductance change = max → 항상 최대값 update"

### ✅ 올바른 이해
"Gradient 10^5 → **Multiple pulses** + **Scaling** → 누적 효과 → 정확한 update"

---

## RPUCuda Stochastic Pulse Update 메커니즘

### 1단계: Input Scaling (Normalization)

```python
# tile.update(x, d) 호출
# x: input [batch, in_size]
# d: gradient [batch, out_size] ← 10^5 같은 큰 값!

# RPUCuda 내부 (C++ 구현):
def update_internal(x, d, learning_rate):
    # Step 1: Find max values
    x_max = max(abs(x))  # 예: 1.0
    d_max = max(abs(d))  # 예: 100000.0 (10^5)

    # Step 2: Normalize to [-1, +1] range
    x_norm = x / x_max  # [-1, +1] 범위
    d_norm = d / d_max  # [-1, +1] 범위

    # Step 3: Scale factor 저장
    scale = learning_rate * x_max * d_max
```

**예시:**
```python
# 입력
d = 100000.0  # Gradient 10^5
x = 0.5       # Input

# Normalization
d_max = 100000.0
d_norm = d / d_max = 1.0  # ← Normalized!

x_max = 0.5
x_norm = x / x_max = 1.0
```

---

### 2단계: Pulse 수 계산

```python
# 각 (i, j) weight에 대해:
def calculate_pulses(x_norm[i], d_norm[j], scale):
    # Coincidence detection
    coincidence = x_norm[i] × d_norm[j]  # [-1, +1]

    # Pulse 수 결정
    # coincidence > 0 → positive pulses (up)
    # coincidence < 0 → negative pulses (down)

    num_pulses = round(abs(coincidence) × max_pulses)
    direction = sign(coincidence)  # +1 or -1

    return num_pulses, direction
```

**예시:**
```python
x_norm[2] = 0.8
d_norm[3] = -0.5

coincidence = 0.8 × (-0.5) = -0.4

# 만약 max_pulses = 1000이면:
num_pulses = round(0.4 × 1000) = 400
direction = -1  # negative pulses

# → W[3,2]에 400개의 down pulses 적용!
```

---

### 3단계: Stochastic Pulse Application

각 pulse마다:

```python
# LinearStepDevice 수식 (devices.py Line 394):
# w_{ij} ← w_{ij} - Δw^d (γ^d · w_{ij} + 1 + σ_c2c · ξ)

for pulse in range(num_pulses):
    # Base step size
    dw_base = dw_min  # 예: 0.001

    # Nonlinearity (weight-dependent)
    nonlin_factor = gamma * w[i,j] + 1.0

    # Stochastic noise (cycle-to-cycle variation)
    noise = randn() * dw_min_std  # ← Stochastic!

    # Actual step (with noise!)
    dw_actual = dw_base * nonlin_factor + noise

    # Update weight
    if direction > 0:
        w[i,j] += dw_actual  # up pulse
    else:
        w[i,j] -= dw_actual  # down pulse

    # Clip to bounds
    w[i,j] = clip(w[i,j], w_min, w_max)
```

**Stochastic 효과:**
```python
# 400 pulses, dw_min = 0.001, dw_min_std = 0.0001

Pulse 1: dw = 0.001 + randn() × 0.0001 = 0.00098  ← random!
Pulse 2: dw = 0.001 + randn() × 0.0001 = 0.00103  ← random!
Pulse 3: dw = 0.001 + randn() × 0.0001 = 0.00095  ← random!
...
Pulse 400: dw = 0.001 + randn() × 0.0001 = 0.00101

# 총 변화 (누적)
ΔW = Σ dw_actual ≈ 400 × 0.001 = 0.4 (평균)
# 하지만 stochastic variation으로 인해:
# ΔW = 0.4 ± sqrt(400) × 0.0001 = 0.4 ± 0.002
```

---

### 4단계: Scale 복원

```python
# 최종 weight change
ΔW_analog = Σ(pulses) × dw_min  # Analog tile에서 계산한 값

# Original scale로 복원
ΔW_actual = ΔW_analog × scale
          = ΔW_analog × (lr × x_max × d_max)
```

**예시:**
```python
# Analog tile update
ΔW_analog = -0.4  # 400 pulses × 0.001

# Scale restoration
scale = lr × x_max × d_max
      = 0.0002 × 0.5 × 100000
      = 10.0

ΔW_actual = -0.4 × 10.0 = -4.0

# 이것이 실제 weight 변화!
W_new = W_old + ΔW_actual
```

---

## 구체적 예시: Gradient 10^5 처리

### 시나리오

```python
# Inputs
batch_size = 32
x = torch.randn(32, 128)  # Input, x_max ≈ 3.0
d = torch.randn(32, 128) × 50000  # Gradient × 50000, d_max ≈ 10^5

lr = 0.0002

# tile.update(x, d) 호출
```

### Step-by-Step

#### Step 1: Normalization
```python
x_max = 3.0
d_max = 100000.0

x_norm = x / 3.0      # [-1, +1]
d_norm = d / 100000.0 # [-1, +1]

scale = 0.0002 × 3.0 × 100000.0 = 60.0
```

#### Step 2: Coincidence & Pulses
```python
# 예시 하나의 weight W[50, 20]
x_norm[5, 20] = 0.6
d_norm[5, 50] = -0.8

coincidence = 0.6 × (-0.8) = -0.48

# max_pulses = 1000 (가정)
num_pulses = round(0.48 × 1000) = 480
direction = -1  # down
```

#### Step 3: Stochastic Update
```python
# Device parameters (sixt1c)
dw_min = 0.001981
dw_min_std = 0.0  # Deterministic (default)
# 만약 stochastic이면: dw_min_std = 0.0001

# 480 pulses 적용
for i in range(480):
    if dw_min_std > 0:
        dw = 0.001981 + randn() × 0.0001  # Stochastic!
    else:
        dw = 0.001981  # Deterministic

    W[50, 20] -= dw  # down pulse

# 총 변화 (deterministic case)
ΔW_analog = -480 × 0.001981 = -0.951
```

#### Step 4: Scale Restoration
```python
ΔW_actual = -0.951 × 60.0 = -57.06

W_new[50, 20] = W_old[50, 20] - 57.06
```

#### 검증: 수식과 비교
```python
# 이론적 수식:
# ΔW = -lr × x^T @ d
#    = -lr × (x[5,20] × d[5,50])
#    = -0.0002 × (0.6 × 3.0) × (-0.8 × 100000)
#    = -0.0002 × 1.8 × (-80000)
#    = -0.0002 × (-144000)
#    = 28.8... (batch 1개만!)

# 실제로는 batch 전체 outer product:
# ΔW = -lr × Σ_batch (x[b,20] × d[b,50])
```

---

## Stochastic vs Deterministic 비교

### Deterministic (dw_min_std = 0)
```python
# 480 pulses, 모두 동일
Pulse 1:   dw = 0.001981
Pulse 2:   dw = 0.001981
Pulse 3:   dw = 0.001981
...
Pulse 480: dw = 0.001981

Total: ΔW = 480 × 0.001981 = 0.951 (exact!)
```

### Stochastic (dw_min_std = 0.0001)
```python
# 480 pulses, 매번 다름
Pulse 1:   dw = 0.001981 + randn() × 0.0001 = 0.001975
Pulse 2:   dw = 0.001981 + randn() × 0.0001 = 0.001989
Pulse 3:   dw = 0.001981 + randn() × 0.0001 = 0.001970
...
Pulse 480: dw = 0.001981 + randn() × 0.0001 = 0.001992

# 평균
E[ΔW] = 480 × 0.001981 = 0.951

# 분산
Var[ΔW] = 480 × (0.0001)^2 = 4.8 × 10^-6
Std[ΔW] = sqrt(480) × 0.0001 = 0.0022

# 실제 값 (stochastic!)
ΔW = 0.951 ± 0.0022  (약 0.2% variation)
```

---

## Gradient 범위별 처리

| **Gradient** | **d_max** | **d_norm** | **Pulses** | **ΔW_analog** | **Scale** | **ΔW_actual** |
|--------------|-----------|------------|------------|---------------|-----------|---------------|
| 10^1 (10)    | 10        | 1.0        | 1000       | 1.98          | 0.006     | 0.012         |
| 10^2 (100)   | 100       | 1.0        | 1000       | 1.98          | 0.06      | 0.119         |
| 10^3 (1000)  | 1000      | 1.0        | 1000       | 1.98          | 0.6       | 1.19          |
| 10^4 (10000) | 10000     | 1.0        | 1000       | 1.98          | 6.0       | 11.9          |
| 10^5 (100000)| 100000    | 1.0        | 1000       | 1.98          | 60.0      | 119           |

(가정: x_max=3.0, lr=0.0002, coincidence=1.0, max_pulses=1000, dw_min=0.001981)

**핵심:**
- **d_norm은 항상 [-1, +1]** (normalize!)
- **Pulse 수는 coincidence에 비례** (max_pulses까지)
- **실제 크기는 scale factor로 복원**
- **Gradient 10^5도 정확히 처리됨!**

---

## 정리

### ❓ "Gradient 10^5가 오면 무조건 max update?"

❌ **아닙니다!**

✅ **올바른 메커니즘:**

1. **Normalization**: d → d_norm ∈ [-1, +1]
2. **Pulse 계산**: num_pulses = coincidence × max_pulses
3. **Stochastic Update**: 각 pulse마다 dw_min + noise
4. **누적**: ΔW_analog = Σ dw_actual
5. **Scale 복원**: ΔW_actual = ΔW_analog × scale

**Gradient 크기는 scale factor에 반영!**

```python
scale = lr × x_max × d_max

Gradient 10^1 → scale = 0.006  → 작은 update
Gradient 10^5 → scale = 60.0   → 큰 update
```

### 🔑 핵심 포인트

1. **Conductance는 항상 [-1, +1]** (hardware 제약)
2. **Pulse 수는 제한됨** (max_pulses, 예: 1000)
3. **실제 weight change는 scale로 조정**
4. **Stochastic = 각 pulse마다 noise**
5. **Multiple pulses로 큰 gradient 정확히 적용**

**이것이 Analog Crossbar의 핵심 메커니즘입니다!** 🎯

---

## Device Parameters (Stochastic 제어)

### Sixt1c Configuration

```python
from aihwkit.simulator.configs.devices import LinearStepDevice

device = LinearStepDevice()

# === Core parameters ===
device.dw_min = 0.001981  # Minimum weight change per pulse
device.dw_min_std = 0.0   # ← Stochastic variation (0 = deterministic)

# === Nonlinearity ===
device.gamma_up = -0.1678    # Weight-dependent scaling (up)
device.gamma_down = 0.1410   # Weight-dependent scaling (down)

# === Bounds ===
device.w_min = -1.0  # Conductance min
device.w_max = 1.0   # Conductance max

# === Pulse type ===
device.up_down = 0.0        # Asymmetry between up/down
device.up_down_dtod = 0.01  # Device-to-device variation
```

### Stochastic 활성화

```python
# Deterministic (current)
device.dw_min_std = 0.0

# Stochastic (약한 noise)
device.dw_min_std = 0.0001  # 0.01% variation

# Stochastic (강한 noise)
device.dw_min_std = 0.001   # 0.1% variation
```

**이것으로 Stochastic pulse update를 제어합니다!**

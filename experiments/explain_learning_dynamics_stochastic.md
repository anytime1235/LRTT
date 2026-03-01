# 학습 중 Gradient 변화와 Stochastic Pulse Update

## 질문
> "학습 초기에 10^5이 잠깐 들어오고 그 이후로 10^1으로 안정화되면 어떻게 업데이트되는 건가?"

## 핵심: 매 Step마다 독립적으로 Normalization!

---

## 학습 시나리오: Step-by-Step

### 가정
```python
# Model
batch_size = 32
in_size = 128
out_size = 128
lr = 0.0002

# Device (sixt1c)
dw_min = 0.001981
dw_min_std = 0.0001  # Stochastic!
max_pulses = 1000
```

---

## Step 1: 학습 초기 (Gradient 10^5)

### Input
```python
x = torch.randn(32, 128)
# x_max = 2.5

d = torch.randn(32, 128) × 50000  # Large gradient!
# d_max = 105000.0  (약 10^5)
```

### Processing

#### 1) Normalization
```python
x_max = 2.5
d_max = 105000.0

x_norm = x / 2.5       # [-1, +1]
d_norm = d / 105000.0  # [-1, +1]

# Scale factor (이 step만 유효!)
scale_step1 = lr × x_max × d_max
            = 0.0002 × 2.5 × 105000
            = 52.5
```

#### 2) Pulse 계산 (예시: 한 weight)
```python
# x_norm[5,20] = 0.7
# d_norm[5,50] = 0.9

coincidence = 0.7 × 0.9 = 0.63

num_pulses = round(0.63 × 1000) = 630 pulses
```

#### 3) Stochastic Update
```python
ΔW_analog = 0
for i in range(630):
    dw = 0.001981 + randn() × 0.0001
    ΔW_analog += dw

ΔW_analog ≈ 630 × 0.001981 = 1.248 ± 0.003
```

#### 4) Scale 복원
```python
ΔW_actual = ΔW_analog × scale_step1
          = 1.248 × 52.5
          = 65.5

W_new = W_old + 65.5  # 큰 변화!
```

**Step 1 결과: Weight가 크게 변함 (65.5)**

---

## Step 2: 여전히 큰 Gradient (Gradient 10^5)

### Input
```python
x = torch.randn(32, 128)
# x_max = 2.3 (달라짐!)

d = torch.randn(32, 128) × 48000  # 여전히 큼
# d_max = 98000.0
```

### Processing

#### 1) Normalization (새로 계산!)
```python
x_max = 2.3
d_max = 98000.0

x_norm = x / 2.3      # 새로운 normalization!
d_norm = d / 98000.0  # 새로운 normalization!

# Scale factor (이 step만!)
scale_step2 = 0.0002 × 2.3 × 98000 = 45.08
```

#### 2) Pulse & Update
```python
# 비슷한 프로세스...
coincidence = 0.65  (예시)
num_pulses = 650

ΔW_analog ≈ 1.288 ± 0.003

ΔW_actual = 1.288 × 45.08 = 58.1

W_new = W_old + 58.1  # 여전히 큰 변화
```

**Step 2 결과: 또 큰 변화 (58.1)**

---

## Step 10: Gradient 감소 중 (Gradient 10^3)

### Input
```python
x = torch.randn(32, 128)
# x_max = 2.1

d = torch.randn(32, 128) × 500  # 줄어듦!
# d_max = 1200.0  (10^3 수준)
```

### Processing

#### 1) Normalization (완전히 새로!)
```python
x_max = 2.1
d_max = 1200.0  # ← 훨씬 작음!

x_norm = x / 2.1    # [-1, +1]
d_norm = d / 1200.0 # [-1, +1]

# Scale factor
scale_step10 = 0.0002 × 2.1 × 1200 = 0.504  # ← 훨씬 작음!
```

#### 2) Pulse & Update
```python
coincidence = 0.68  (예시)
num_pulses = 680  # 비슷한 수

ΔW_analog ≈ 1.347 ± 0.003

ΔW_actual = 1.347 × 0.504 = 0.679  # ← 작은 변화!

W_new = W_old + 0.679
```

**Step 10 결과: 작은 변화 (0.679)**

---

## Step 50: 안정화 (Gradient 10^1)

### Input
```python
x = torch.randn(32, 128)
# x_max = 1.8

d = torch.randn(32, 128) × 5  # 매우 작음!
# d_max = 12.0  (10^1 수준)
```

### Processing

#### 1) Normalization
```python
x_max = 1.8
d_max = 12.0  # ← 매우 작음!

x_norm = x / 1.8   # [-1, +1]
d_norm = d / 12.0  # [-1, +1]

# Scale factor
scale_step50 = 0.0002 × 1.8 × 12.0 = 0.00432  # ← 매우 작음!
```

#### 2) Pulse & Update
```python
coincidence = 0.55  (예시)
num_pulses = 550

ΔW_analog ≈ 1.090 ± 0.002

ΔW_actual = 1.090 × 0.00432 = 0.00471  # ← 매우 작은 변화!

W_new = W_old + 0.00471
```

**Step 50 결과: 매우 작은 변화 (0.00471)**

---

## 전체 학습 곡선

```
Step | Gradient | d_max    | x_max | scale   | ΔW_analog | ΔW_actual | W 누적
-----|----------|----------|-------|---------|-----------|-----------|--------
1    | 10^5     | 105000   | 2.5   | 52.5    | 1.248     | 65.5      | 65.5
2    | 10^5     | 98000    | 2.3   | 45.08   | 1.288     | 58.1      | 123.6
3    | 10^4     | 15000    | 2.4   | 7.2     | 1.150     | 8.28      | 131.9
5    | 10^3     | 2500     | 2.2   | 1.1     | 1.220     | 1.34      | 133.2
10   | 10^3     | 1200     | 2.1   | 0.504   | 1.347     | 0.679     | 133.9
20   | 10^2     | 250      | 1.9   | 0.095   | 1.180     | 0.112     | 134.0
50   | 10^1     | 12       | 1.8   | 0.00432 | 1.090     | 0.00471   | 134.0
100  | 10^1     | 8        | 1.7   | 0.00272 | 1.050     | 0.00286   | 134.0
```

**관찰:**
1. **ΔW_analog은 거의 일정** (~1.0 to 1.3) ← Pulse 수가 비슷하면!
2. **ΔW_actual은 gradient에 비례** ← Scale factor 때문!
3. **학습 초기: 빠른 수렴** (큰 step)
4. **학습 후기: 미세 조정** (작은 step)

---

## 핵심 메커니즘

### 1️⃣ **매 Step마다 독립적 Normalization**

```python
# Step 1
d_max_step1 = max(abs(d_step1))  # 105000
scale_step1 = lr × x_max × d_max_step1  # 52.5

# Step 50
d_max_step50 = max(abs(d_step50))  # 12
scale_step50 = lr × x_max × d_max_step50  # 0.00432

# ← 완전히 독립적!
```

**이전 step의 gradient를 기억하지 않음!**

### 2️⃣ **Conductance는 항상 [-1, +1]**

```python
# 모든 step에서:
d_norm ∈ [-1, +1]
x_norm ∈ [-1, +1]
ΔW_analog ≈ 1.0 ~ 1.5  (pulse 수가 비슷하면)
```

**Hardware 범위 보호!**

### 3️⃣ **실제 변화는 Scale로 결정**

```python
# Step 1: Gradient 10^5
ΔW_actual = 1.25 × 52.5 = 65.5  ← 큰 변화!

# Step 50: Gradient 10^1
ΔW_actual = 1.09 × 0.00432 = 0.0047  ← 작은 변화!
```

**Gradient 크기가 자동으로 반영됨!**

---

## Stochastic 효과

### Step 1 (큰 Gradient)
```python
# 630 pulses
ΔW_analog = Σ(dw_i) where dw_i = 0.001981 ± 0.0001

E[ΔW_analog] = 1.248
Std[ΔW_analog] = sqrt(630) × 0.0001 = 0.0025

# Stochastic variation
ΔW_analog = 1.248 ± 0.0025  (0.2% variation)

# Actual update
ΔW_actual = (1.248 ± 0.0025) × 52.5
          = 65.5 ± 0.13  (0.2% variation)
```

### Step 50 (작은 Gradient)
```python
# 550 pulses
ΔW_analog = 1.090 ± 0.0023

# Actual update
ΔW_actual = (1.090 ± 0.0023) × 0.00432
          = 0.00471 ± 0.00001  (0.2% variation)
```

**Relative variation은 동일! (약 0.2%)**

---

## 비교: Digital vs Analog Stochastic

### Digital (PyTorch autograd)
```python
# Step 1
grad = 100000.0
W -= lr × grad = 0.0002 × 100000 = 20.0  (exact!)

# Step 50
grad = 10.0
W -= lr × grad = 0.0002 × 10 = 0.002  (exact!)
```

**완전히 deterministic, 항상 정확!**

### Analog Stochastic
```python
# Step 1
grad_max = 100000.0
scale = 52.5
ΔW = (1.248 ± 0.0025) × 52.5 = 65.5 ± 0.13

# Step 50
grad_max = 12.0
scale = 0.00432
ΔW = (1.090 ± 0.0023) × 0.00432 = 0.00471 ± 0.00001
```

**약간의 random variation (약 0.2%)**

---

## 학습 안정성

### 초기 (Gradient 10^5)
```python
# 큰 update + 0.2% noise
ΔW = 65.5 ± 0.13

# Noise의 절대값은 크지만 (0.13)
# 상대적으로는 작음 (0.2%)
# → 학습 방향은 정확!
```

### 수렴 (Gradient 10^1)
```python
# 작은 update + 0.2% noise
ΔW = 0.00471 ± 0.00001

# Noise의 절대값은 매우 작음 (0.00001)
# 상대적으로는 여전히 0.2%
# → 미세 조정 가능!
```

**모든 단계에서 안정적!**

---

## 정리

### ❓ "학습 초기 10^5 → 이후 10^1로 안정화되면?"

✅ **Step별 독립 처리:**

**Step 1-5 (Gradient 10^5):**
```
1. d_max = 10^5
2. Normalize: d_norm ∈ [-1, +1]
3. Pulse: ~600-800개
4. Scale: ~50
5. ΔW_actual: ~60 (큰 변화!)
```

**Step 50+ (Gradient 10^1):**
```
1. d_max = 10^1
2. Normalize: d_norm ∈ [-1, +1]  (같은 범위!)
3. Pulse: ~600-800개  (비슷!)
4. Scale: ~0.005  (매우 작음!)
5. ΔW_actual: ~0.005 (작은 변화!)
```

### 🔑 핵심

1. **매 step마다 새로 normalization**
2. **이전 gradient 기억 안 함**
3. **Conductance는 항상 [-1, +1]**
4. **실제 변화는 scale factor로 자동 조정**
5. **Stochastic variation은 항상 ~0.2% (상대)**

**Gradient가 10^5에서 10^1로 줄어들면:**
- Analog tile은 자동으로 작은 update 수행!
- Hardware 범위는 항상 보호됨!
- 학습은 안정적으로 수렴!

**이것이 Analog Crossbar의 자동 적응 메커니즘입니다!** 🎯

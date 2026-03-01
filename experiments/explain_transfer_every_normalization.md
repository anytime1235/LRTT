# transfer_every와 Batch Normalization의 관계

## 질문
> "lrtt 학습에서 transfer_every=100으로 한다고 해서 이게 배치 단위 normalization에 영향을 끼치는가?"

## 답변: ❌ **아니오! 완전히 독립적입니다!**

---

## transfer_every란?

### LRTT의 3-tile 구조

```python
# LRTT Controller
tile_a: [d_size, rank]      # Fast A (LoRA left)
tile_b: [rank, x_size]      # Fast B (LoRA right)
tile_c: [d_size, x_size]    # Visible C (main weight)

# Output
y = C @ x + α × A @ (B @ x)
```

### transfer_every의 역할

```python
transfer_every = 100  # 100 step마다 transfer

# Step 1-99: A, B update
for step in range(1, 100):
    # Forward: y = C @ x + A @ (B @ x)
    # Backward: gradient 계산
    # Update: A, B update (tile.update)
    # C는 고정!

# Step 100: Transfer A⊗B → C
C_new = C + A @ B  # Merge!
A = 0              # Reinit A
B = kaiming_init() # Reinit B

# Step 101-199: 다시 A, B update
for step in range(101, 200):
    # ...
```

**핵심: transfer_every는 weight 관리 전략!**

---

## Batch Normalization (d_max 계산)

### 매 Step마다 독립적

```python
# tile.update(x, d) 내부 (RPUCuda)
def update(x, d, lr):
    # Step 1: 현재 batch에서만 max 찾기
    d_max = max(abs(d))  # ← 현재 batch만!
    x_max = max(abs(x))  # ← 현재 batch만!

    # Step 2: Normalize
    d_norm = d / d_max
    x_norm = x / x_max

    # Step 3: Scale
    scale = lr × x_max × d_max

    # Step 4-5: Update
    # ...
```

**핵심: d_max는 현재 batch만 봄, 이전 step 무관!**

---

## transfer_every와 d_max의 관계

### Step-by-Step 시나리오

```
transfer_every = 100
```

#### Step 1-99 (Transfer 전)

```python
# Step 1
tile_a.update(XB_1, d_1)
  → d_max_1 = max(abs(d_1))  # 현재 batch
  → update A

# Step 50
tile_a.update(XB_50, d_50)
  → d_max_50 = max(abs(d_50))  # 현재 batch
  → update A

# Step 99
tile_a.update(XB_99, d_99)
  → d_max_99 = max(abs(d_99))  # 현재 batch
  → update A
```

#### Step 100 (Transfer!)

```python
# Transfer
C = C + A @ B
A = 0
B = kaiming_init()

# Update는 여전히 동일
tile_a.update(XB_100, d_100)
  → d_max_100 = max(abs(d_100))  # 현재 batch (transfer 무관!)
  → update A (새로운 A)
```

#### Step 101-199 (Transfer 후)

```python
# Step 101
tile_a.update(XB_101, d_101)
  → d_max_101 = max(abs(d_101))  # 현재 batch
  → update A

# 계속...
```

### 관찰

**모든 step에서:**
```python
d_max = max(abs(d_current_batch))
# ← transfer 여부와 무관!
# ← transfer_every 값과 무관!
# ← 오직 현재 batch만!
```

---

## 시각화

```
Step:  1    50   99  100  101  150  199  200  ...
       │    │    │    │    │    │    │    │
A,B:   ●────●────●────┐    ●────●────●────┐
                       │                   │
                       ▼ Transfer          ▼ Transfer
                       C += A@B            C += A@B
                       A = 0               A = 0
                       B = init            B = init

d_max: ●    ●    ●    ●    ●    ●    ●    ●
       ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
       현재 현재 현재 현재 현재 현재 현재 현재
       배치 배치 배치 배치 배치 배치 배치 배치

→ 모든 step에서 d_max는 독립적으로 계산!
→ Transfer는 d_max 계산에 영향 없음!
```

---

## 비교 표

| **항목** | **transfer_every=10** | **transfer_every=100** | **transfer_every=1000** |
|---------|----------------------|------------------------|-------------------------|
| Transfer 주기 | 10 step마다 | 100 step마다 | 1000 step마다 |
| A, B reinit 횟수 | 많음 | 중간 | 적음 |
| **d_max 계산** | **현재 batch** | **현재 batch** | **현재 batch** |
| **Normalization** | **독립적** | **독립적** | **독립적** |

**→ transfer_every 값과 무관하게 d_max는 항상 현재 batch에서만 계산!**

---

## 간접적 영향은 있을 수 있는가?

### 직접적 영향: ❌ 없음

```python
# d_max 계산은 transfer와 완전히 독립
d_max = max(abs(d))  # transfer_every 값 사용 안 함!
```

### 간접적 영향: ⚠️ 있을 수 있음 (학습 dynamics)

#### Scenario 1: transfer_every=10 (자주 transfer)

```python
# Step 1-9: A, B 누적
A, B update...

# Step 10: Transfer
C += A @ B  # 작은 누적
A = 0
B = init

# Step 11: 새로운 A, B로 다시 학습
# A, B가 작음 → gradient가 다르게 나올 수 있음
d_new = compute_gradient(...)
d_max_new = max(abs(d_new))  # ← 이 값이 다를 수 있음!
```

#### Scenario 2: transfer_every=1000 (드물게 transfer)

```python
# Step 1-999: A, B 크게 누적
A, B update...  # 계속 커짐

# Step 1000: Transfer
C += A @ B  # 큰 누적
A = 0
B = init

# Step 1001: 새로운 A, B로 다시 학습
# A, B가 작음 → gradient 급변
d_new = compute_gradient(...)
d_max_new = max(abs(d_new))  # ← 이 값이 크게 다를 수 있음!
```

**하지만 이것은:**
- Normalization **메커니즘**이 바뀌는 것이 아님!
- Gradient **값 자체**가 달라지는 것 (학습 dynamics)
- d_max는 여전히 현재 batch에서만 계산

---

## 실제 예시

### Configuration

```python
transfer_every = 100
lr = 0.0002
```

### Step 99 (Transfer 직전)

```python
# A, B가 99번 누적되어 큼
A_max ≈ 5.0
B_max ≈ 3.0

# Forward
y = C @ x + A @ (B @ x)
# A, B가 크므로 output도 큼

# Gradient
d = compute_grad(...)
d_max = 15000.0  # 예시

# Normalization
d_norm = d / 15000.0
scale = 0.0002 × x_max × 15000 = large

# Update
tile_a.update(XB, d)
# → d_max = 15000 사용
```

### Step 100 (Transfer!)

```python
# Transfer
C += A @ B
A = 0  # Reset!
B = kaiming_init()  # Small!

# Now A, B가 작음
A_max ≈ 0.0
B_max ≈ 0.1

# Forward
y = C @ x + A @ (B @ x)
# A, B가 작으므로 A@(B@x) 기여 작음

# Gradient
d = compute_grad(...)
d_max = 500.0  # 예시 (작아짐!)

# Normalization
d_norm = d / 500.0  # ← d_max가 달라짐!
scale = 0.0002 × x_max × 500 = small

# Update
tile_a.update(XB, d)
# → d_max = 500 사용
```

### 관찰

**Step 99 vs Step 100:**
- Step 99: d_max = 15000 (A, B가 컸음)
- Step 100: d_max = 500 (A, B가 reinit됨)

**이것은:**
- ✅ Normalization이 바뀌어서? NO!
- ✅ Gradient 값이 바뀌어서? YES!

**각 step에서:**
```python
d_max = max(abs(d_current))  # 항상 현재 batch만!
```

---

## 정리

### ❓ "transfer_every=100이 배치 단위 normalization에 영향을 주는가?"

### ✅ **직접적 영향: ❌ 없음**

**Normalization 메커니즘:**
```python
# 모든 step에서 동일
d_max = max(abs(d))  # 현재 batch
d_norm = d / d_max
scale = lr × x_max × d_max
```

**transfer_every는:**
- d_max 계산에 사용되지 않음
- Normalization 로직을 바꾸지 않음
- 완전히 독립적인 메커니즘

### ⚠️ **간접적 영향: 있을 수 있음 (학습 dynamics)**

**Transfer 후:**
- A, B reinit → 작은 weight
- Forward 결과 달라짐
- Gradient 값 달라짐
- → d_max 값이 달라질 수 있음

**하지만 이것은:**
- Normalization이 바뀌는 것이 아님
- Gradient 자체가 바뀌는 것
- d_max는 여전히 "현재 batch의 max"

---

## 핵심 포인트

### 1️⃣ **독립적 메커니즘**

```
tile.update(x, d):
  ↓
  d_max = max(abs(d))  ← 현재 batch만!
  ↓
  Normalization
  ↓
  Update

← transfer_every와 무관!
```

### 2️⃣ **매 Step마다 Fresh**

```python
# Step 1
d_max_1 = max(abs(d_1))

# Step 100 (transfer!)
d_max_100 = max(abs(d_100))  # 새로 계산!

# Step 101
d_max_101 = max(abs(d_101))  # 새로 계산!
```

### 3️⃣ **학습 Dynamics vs Normalization**

```
transfer_every → A, B reinit → Gradient 변화 → d_max 값 변화
                                              ↑
                                              여기서 d_max는
                                              여전히 "현재 batch max"
```

---

## 최종 답변

**transfer_every=100:**
- ❌ Normalization 메커니즘에 영향 없음
- ❌ d_max 계산 방식에 영향 없음
- ✅ 학습 dynamics에 영향 (A, B reinit)
- ✅ Gradient 값이 달라질 수 있음
- ✅ 하지만 d_max는 여전히 현재 batch에서만 계산

**결론:**
```
transfer_every와 batch normalization은 독립적!
단, transfer로 인한 gradient 변화는 있을 수 있음.
```

✅ **완전히 독립적인 메커니즘입니다!** 🎯

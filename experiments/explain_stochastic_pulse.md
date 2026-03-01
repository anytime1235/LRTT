# Stochastic Pulse Update: 어디서 일어나는가?

## 질문
> "그러면 stochastic pulse update는 어떻게 일어나는건가?"

## 답변: 3가지 시나리오 구분

### 1️⃣ **현재 LRTT 학습**: ❌ Stochastic 없음

```python
# lora_training_glue/sixt1c_config.py Line 100
rpu_config.modifier.std_dev = 0.0  # ← NO noise!
rpu_config.noise_model = None      # ← NO stochastic behavior!
```

**현재 상태:**
- 완전히 **Deterministic** (결정론적)
- W_new = W_old - lr × gradient (정확한 계산)
- 같은 input → 항상 같은 output
- **Stochastic pulse update 없음!**

**왜 없나?**
- Training은 digital simulation
- PyTorch Parameter 사용
- GPU/CPU에서 계산
- Reproducibility 필요 (논문, 실험)

---

### 2️⃣ **실제 Analog Hardware Inference**: ✅ Real Stochastic

**실제 hardware에 배포했을 때:**

```
┌────────────────────────────────────┐
│ Analog Crossbar (Memristor Array)  │
├────────────────────────────────────┤
│  Voltage pulse → G(t+1)            │
│  (inherently stochastic!)          │
└────────────────────────────────────┘
```

**Stochastic한 이유:**

1. **Device variability** (소자 간 편차)
   - 같은 전압 → 다른 conductance 변화
   - Manufacturing 차이
   - Material 불균일성

2. **Pulse-to-pulse variation** (펄스 간 변동)
   - 같은 소자, 같은 전압
   - 하지만 매번 다른 ΔG
   - 확률 분포: ΔG ~ N(μ, σ)

3. **Physical noise**
   - Thermal noise (kT)
   - Shot noise (전류)
   - 1/f noise
   - Random telegraph noise (RTN)

4. **Quantum effects**
   - Tunneling의 확률성
   - Filament formation (확률적)

**실제 Hardware 동작:**

```python
# Pseudo-code (actual hardware behavior)
def hardware_pulse_update(conductance, voltage_pulse):
    # Ideal update
    delta_G_ideal = f(voltage_pulse)

    # Stochastic variation
    noise = np.random.normal(0, sigma_device)

    # Actual update
    delta_G_actual = delta_G_ideal + noise

    # Conductance change
    G_new = G_old + delta_G_actual

    return G_new
```

**예시:**

```
Ideal:  Apply +1V → ΔG = +0.1 µS (expected)

Actual (stochastic):
  Trial 1: ΔG = +0.098 µS
  Trial 2: ΔG = +0.103 µS
  Trial 3: ΔG = +0.095 µS
  Trial 4: ΔG = +0.101 µS

  Distribution: ΔG ~ N(0.1, 0.003) µS
```

---

### 3️⃣ **Training시 Stochastic Simulation** (Optional): ⚠️ 가능하지만 비활성화됨

Aihwkit는 hardware non-ideality를 **시뮬레이션**할 수 있습니다.

#### 설정 방법 (현재는 OFF)

```python
# 현재 설정 (sixt1c_config.py Line 100)
rpu_config.modifier.std_dev = 0.0  # ← Noise OFF

# Stochastic 활성화하려면:
rpu_config.modifier.std_dev = 0.01  # ← 1% noise
rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
```

#### Stochastic Simulation 작동 방식

```python
# aihwkit internal simulation
def apply_weight_modifier(weights, rpu_config):
    if rpu_config.modifier.std_dev > 0:
        # Add Gaussian noise to simulate stochastic updates
        noise = torch.randn_like(weights) * rpu_config.modifier.std_dev
        weights_noisy = weights + noise
        return weights_noisy
    else:
        return weights  # No noise
```

**효과:**

```python
# Without noise (current)
W_new = W_old - lr × gradient
# Deterministic: 항상 같은 결과

# With noise (if enabled)
W_new = W_old - lr × gradient + randn() × std_dev
# Stochastic: 매번 다른 결과
```

---

## 전체 비교

| **Scenario** | **Computation** | **Stochastic?** | **Where** | **Purpose** |
|--------------|-----------------|-----------------|-----------|-------------|
| **LRTT Training (현재)** | Digital (PyTorch) | ❌ NO | GPU/CPU | Model 학습 |
| **Training w/ Noise Sim** | Digital + Noise | ⚠️ Optional | GPU/CPU | Robust 학습 |
| **Hardware Inference** | Analog (Real HW) | ✅ YES | Memristor Crossbar | 실제 배포 |

---

## 현재 LRTT 학습 흐름 (Stochastic 없음)

```python
# Step 1: Forward (deterministic)
XB = x @ tile_b.weight  # Digital matmul
y = XB @ tile_a.weight  # Digital matmul

# Step 2: Loss (deterministic)
loss = criterion(y, target)

# Step 3: Backward (deterministic)
loss.backward()  # PyTorch autograd
# → tile_a.weight.grad
# → tile_b.weight.grad

# Step 4: Update (deterministic)
tile_a.weight -= lr × tile_a.weight.grad
tile_b.weight -= lr × tile_b.weight.grad

# NO stochastic noise added!
```

---

## 실제 Hardware에서 Stochastic Pulse

### Hardware 배포 시나리오

1. **Training (Digital, NO stochastic)**
   ```python
   # LRTT training on GPU (deterministic)
   model.train()
   optimizer.step()
   ```

2. **Model Export**
   ```python
   # Export trained weights
   W_a_trained = tile_a.weight.cpu().numpy()
   W_b_trained = tile_b.weight.cpu().numpy()
   ```

3. **Hardware Programming (Stochastic 시작!)**
   ```python
   # Program memristor crossbar
   for i, j in all_devices:
       target_G = W[i, j]

       # Apply voltage pulses (stochastic!)
       while abs(current_G - target_G) > threshold:
           voltage_pulse = calculate_pulse(target_G - current_G)
           current_G += apply_pulse(voltage_pulse)  # ← Stochastic!
   ```

4. **Hardware Inference (Stochastic)**
   ```python
   # Inference on analog hardware
   y_analog = analog_crossbar.forward(x)
   # y_analog ≠ y_digital (due to stochastic G)
   ```

### Stochastic Pulse의 실제 영향

```
Training (Digital):
  W_ideal = 0.5234

Hardware Programming (Stochastic):
  Target: 0.5234

  Pulse 1: G = 0.0000 → 0.1023 (ΔG = 0.1023)
  Pulse 2: G = 0.1023 → 0.2165 (ΔG = 0.1142)  ← stochastic!
  Pulse 3: G = 0.2165 → 0.3287 (ΔG = 0.1122)  ← stochastic!
  Pulse 4: G = 0.3287 → 0.4512 (ΔG = 0.1225)  ← stochastic!
  Pulse 5: G = 0.4512 → 0.5189 (ΔG = 0.0677)  ← stochastic!

  Final: G = 0.5189 ≈ 0.5234 (close but not exact!)
  Error: 0.86%
```

**이 error가 누적되면:**
- Inference accuracy 감소
- Model robustness 필요
- Training시 noise injection으로 대비 가능

---

## Stochastic Simulation 활성화 (연구용)

만약 hardware robustness를 학습하려면:

```python
# sixt1c_config.py 수정
def gen_sixt1c_lora_config(...):
    rpu_config = TorchInferenceRPUConfig()

    # Stochastic weight modifier 활성화
    rpu_config.modifier.std_dev = 0.01  # 1% noise
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

    # 또는 더 정교한 noise model
    from aihwkit.simulator.configs import PCMLikeNoiseModel
    rpu_config.noise_model = PCMLikeNoiseModel(
        prog_noise_scale=0.01,  # Programming noise
        read_noise_scale=0.005, # Read noise
        drift_scale=0.002,      # Drift noise
    )

    return rpu_config
```

**효과:**

```python
# Training step with noise
W_new = W_old - lr × gradient + noise
# noise ~ N(0, std_dev)

# 매 step마다 다른 noise
# → Model이 noise에 robust해짐
# → Hardware deployment시 정확도 향상
```

---

## 정리

### ❓ "Stochastic pulse update는 어떻게 일어나는가?"

✅ **3가지 답변:**

1. **LRTT 학습 (현재)**: **일어나지 않습니다**
   - 완전히 deterministic
   - Digital PyTorch 계산
   - Reproducible

2. **실제 Hardware**: **일어납니다**
   - Voltage pulse의 물리적 성질
   - Device variability
   - Noise, quantum effects
   - **이것이 진짜 stochastic pulse update!**

3. **Simulation (Optional)**: **가능하지만 비활성화됨**
   - `modifier.std_dev = 0.0` (현재)
   - 활성화하면 noise 추가 가능
   - Hardware robustness 학습용

### 🔑 핵심

**Stochastic pulse update는:**
- ❌ Training 중에는 없음 (digital simulation)
- ✅ Hardware deployment시 있음 (real analog)
- ⚠️ Training시 시뮬레이션 가능 (optional)

**현재 LRTT:**
```python
# NO stochastic behavior!
W = W - lr × grad  # Exact, deterministic
```

**실제 Hardware:**
```python
# Real stochastic behavior
G = G + ΔG(V_pulse) + noise(σ_device)  # Stochastic!
```

이것이 stochastic pulse update의 전체 그림입니다!

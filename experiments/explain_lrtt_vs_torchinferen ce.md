# LRTT tile.update() vs TorchInferenceTile: Stochastic Pulse Update

## 핵심 질문
> "lrtt ab weight update에서 tile.update를 사용하는데 이게 stochastic update가 되고 있는 것 아닌가? 즉 LRTT에서는 stochastic pulse update가 되고 있는 것 아닌가?"

## 답변: 2가지 시나리오를 구분해야 합니다!

---

## 시나리오 1: **현재 Sixt1c LoRA Training** ❌ NO Stochastic

### 코드 확인

**related_functions.py Line 172-176:**
```python
# Convert each LoRA layer to analog
analog_layer = AnalogLinear.from_digital(
    original_layer,
    rpu_config,
    tile_module_class=TorchInferenceTile  # ← TorchInferenceTile!
)
```

**run_glue.py Line 753:**
```python
model = convert_lora_layers_only_to_analog(model, sixt1c_rpu_config)
# → 내부적으로 TorchInferenceTile 사용
```

### TorchInferenceTile의 update()

**inference_torch.py:**
```python
class TorchInferenceTile:
    def update(self, x_input, d_input):
        raise NotImplementedError(
            "Torch tile does not support direct update. "
            "Please use the analog optimizer or the post_update step."
        )
```

**❌ update() 호출 불가!**

### 실제 학습 방식

```python
# Training loop
for batch in dataloader:
    # Forward
    output = model(input)

    # Loss
    loss = criterion(output, target)

    # Backward (PyTorch autograd)
    loss.backward()  # ← grad 계산

    # Update (PyTorch optimizer)
    optimizer.step()  # ← W -= lr × grad

# tile.update()는 호출되지 않음!
# 모든 것이 PyTorch autograd + optimizer!
```

### 결론

✅ **TorchInferenceTile (현재 sixt1c LoRA)**:
- ❌ tile.update() 사용 안 함 (NotImplementedError)
- ✅ PyTorch autograd 사용
- ✅ PyTorch optimizer 사용
- ❌ **Stochastic pulse update 없음** (deterministic)

---

## 시나리오 2: **실제 LRTT Controller 사용** ✅ Stochastic 가능!

### LRTT Controller 코드

**lrtt_controller.py Line 43-45:**
```python
def __init__(
    self,
    tile_a: AnalogTileWithoutPeriphery,   # ← AnalogTile!
    tile_b: AnalogTileWithoutPeriphery,   # ← AnalogTile!
    tile_c: AnalogTileWithoutPeriphery,
    ...
)
```

**lrtt_controller.py Line 702-709:**
```python
# A tile update
lr_a_old = self.tile_a.get_learning_rate()
self.tile_a.set_learning_rate(lr_eff)
if hasattr(self.tile_a, '_orig_update'):
    self.tile_a._orig_update(XB, d)
else:
    self.tile_a.update(XB, d)  # ← tile.update() 호출!
self.tile_a.set_learning_rate(lr_a_old)
```

### AnalogTileWithoutPeriphery의 update()

**analog.py Line 310-322:**
```python
class AnalogTileWithoutPeriphery:
    def update(self, x_input: Tensor, d_input: Tensor) -> None:
        """Perform the update pass."""
        return self.tile.update(  # ← RPUCuda backend!
            x_input, d_input, self.analog_bias,
            self.in_trans, self.out_trans, self.non_blocking
        )
```

### RPUCuda Backend

**self.tile**은 C++ 구현 (RPUCuda):
- 실제 analog crossbar 시뮬레이션
- Stochastic pulse update 가능!
- Device configuration에 따라 noise 추가 가능

### Device Configuration

```python
# AnalogTile을 생성할 때
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice

rpu_config = SingleRPUConfig(device=LinearStepDevice())

# Device parameters (stochastic 설정 가능!)
rpu_config.device.dw_min = 0.001       # Minimum weight change
rpu_config.device.dw_min_std = 0.0001  # ← Stochastic variation!
rpu_config.device.up_down = 0.0        # ← Pulse asymmetry (stochastic!)
```

### 결론

✅ **LRTTController + AnalogTile**:
- ✅ tile.update() 사용
- ✅ RPUCuda backend (C++ 구현)
- ✅ Device configuration에서 stochastic 설정 가능
- ✅ **Stochastic pulse update 가능!**

---

## 비교 표

| **항목** | **TorchInferenceTile<br>(현재 sixt1c LoRA)** | **AnalogTile<br>(LRTT Controller)** |
|---------|----------------------------------------|----------------------------------|
| **Tile Type** | TorchInferenceTile | AnalogTileWithoutPeriphery |
| **Backend** | PyTorch (Python) | RPUCuda (C++) |
| **update() 호출** | ❌ NotImplementedError | ✅ 가능 |
| **학습 방식** | PyTorch autograd + optimizer | tile.update() (outer product) |
| **Stochastic Pulse** | ❌ 없음 (deterministic) | ✅ 가능 (device config) |
| **사용처** | Inference simulation, training | Hardware-aware training, LRTT |

---

## 현재 Sixt1c LoRA Training 흐름

```python
# 1. Model conversion
model = convert_lora_layers_only_to_analog(model, sixt1c_rpu_config)
# → lora_A, lora_B are AnalogLinear with TorchInferenceTile

# 2. Training
for batch in dataloader:
    # Forward (TorchInferenceTile.forward)
    output = model(input)
    # → tile.forward(x) → x @ W (PyTorch matmul)

    # Loss
    loss = criterion(output, target)

    # Backward (PyTorch autograd)
    loss.backward()
    # → PyTorch 자동으로 grad 계산
    # → backward hook으로 gradient 보정 (mapping_scales)

    # Update (PyTorch optimizer)
    optimizer.step()
    # → W -= lr × grad (PyTorch)
    # → tile.update() 호출 안 함!

# 3. tile.update()는 사용되지 않음!
```

**❌ tile.update() 호출 없음!**
**❌ Stochastic pulse update 없음!**
**✅ 완전히 deterministic PyTorch 학습!**

---

## LRTT Controller 사용시 흐름 (참고용)

```python
# 1. LRTTController 생성
from aihwkit.simulator.tiles.analog import AnalogTileWithoutPeriphery
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice

# Device with stochastic parameters
device = LinearStepDevice()
device.dw_min = 0.001
device.dw_min_std = 0.0001  # ← Stochastic!
rpu_config = SingleRPUConfig(device=device)

# Create tiles
tile_a = AnalogTileWithoutPeriphery(d_size, rank, rpu_config)
tile_b = AnalogTileWithoutPeriphery(rank, x_size, rpu_config)
tile_c = AnalogTileWithoutPeriphery(d_size, x_size, rpu_config)

lrtt = LRTTController(tile_a, tile_b, tile_c, ...)

# 2. Training with LRTT
for batch in dataloader:
    # Forward
    output = lrtt.forward(input)

    # Backward
    d = compute_gradient(output, target)

    # Update (tile.update() 호출!)
    lrtt.ab_weight_update(input, d, lr)
    # → 내부적으로:
    #   tile_a.update(XB, d)  ← RPUCuda, stochastic 가능!
    #   tile_b.update(x, DA)  ← RPUCuda, stochastic 가능!
```

**✅ tile.update() 호출!**
**✅ Stochastic pulse update 가능!**

---

## 정리

### ❓ "LRTT에서 tile.update()를 사용하는데 stochastic update가 되고 있는 것 아닌가?"

**답변이 2가지입니다:**

### 1️⃣ **현재 Sixt1c LoRA Training (run_glue.py)**:

❌ **LRTT Controller를 사용하지 않습니다!**
- TorchInferenceTile 사용
- PyTorch autograd + optimizer
- tile.update() 호출 안 함
- **Stochastic pulse update 없음** (deterministic)

### 2️⃣ **실제 LRTT Controller를 사용한다면**:

✅ **tile.update()를 사용하고 stochastic 가능!**
- AnalogTileWithoutPeriphery 사용
- RPUCuda backend
- tile.update() 호출
- Device configuration에서 stochastic 설정 가능
- **Stochastic pulse update 가능!**

---

## 결론

**현재 코드 (run_glue.py):**

```python
# related_functions.py Line 175
tile_module_class=TorchInferenceTile  # ← 이것 사용!

# → tile.update() 없음
# → Stochastic 없음
# → 완전히 deterministic PyTorch 학습
```

**만약 LRTT를 실제로 사용한다면:**

```python
# lrtt_controller.py 사용
tile_a: AnalogTileWithoutPeriphery  # ← 이것 사용!

# → tile.update() 사용
# → RPUCuda backend
# → Stochastic pulse update 가능
```

### 🔑 핵심

**"lrtt_controller.py에 tile.update()가 있다"**는 사실이지만,
**"현재 sixt1c LoRA training이 lrtt_controller.py를 사용한다"**는 거짓입니다!

현재는:
- ✅ TorchInferenceTile (inference simulation)
- ✅ PyTorch autograd
- ❌ LRTT Controller 사용 안 함
- ❌ tile.update() 호출 안 함
- ❌ **Stochastic pulse update 없음!**

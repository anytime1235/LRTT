# transfer_lr vs lora_alpha 관계 분석

## 두 파라미터의 역할

### lora_alpha (α)
**사용 위치:**
1. **Forward pass** (line 1722):
   ```python
   y = y_c + self.lora_alpha * y_ab  # y = C·x + α·A·(B·x)
   ```

2. **LoRA update** (line 683):
   ```python
   lr_eff = lr * self.lora_alpha  # 효과적 learning rate
   ```

### transfer_lr
**사용 위치:**
1. **Transfer operation** (line 1080):
   ```python
   delta = self.transfer_lr * (A @ B)
   C_new = C + delta  # C_new = C + transfer_lr·(A@B)
   ```

---

## 수학적 분석

### 시나리오 1: forward_inject=True로 학습

#### 초기 상태 (hybrid reinit)
```
A_0 = 0
B_0 = B_init (decayed from previous)
C_0 = C_init
```

#### 학습 중 (forward_inject=True)
**Forward:**
```
y = C·x + α·A·(B·x)
```

**Gradient 계산:**
```
∂L/∂y = D (loss에서 전파된 gradient)
```

**AB Update (LoRA chain rule):**
```
lr_eff = lr · α

ΔA = -lr_eff · D · (B·x)^T = -lr·α · D · (B·x)^T
ΔB = -lr_eff · (A^T·D) · x^T = -lr·α · (A^T·D) · x^T
```

**k번 업데이트 후:**
```
A_k = A_0 + Σ ΔA_i = 0 + Σ(-lr·α · D_i · (B_i·x)^T)
B_k = B_0 + Σ ΔB_i
```

#### 핵심 관찰
학습 중 forward output:
```
y_train = C·x + α·A_k·(B_k·x)
```

여기서 A_k, B_k는 이미 α 스케일로 업데이트됨 (lr_eff = lr·α)

---

### Transfer 후 출력

**Transfer 연산:**
```
C_new = C + transfer_lr · (A_k @ B_k)
```

**Transfer 후 forward (forward_inject=False, C-only):**
```
y_final = C_new·x = (C + transfer_lr·A_k·B_k)·x
        = C·x + transfer_lr·(A_k·B_k)·x
```

---

## 중요한 질문: transfer_lr = α 여야 하는가?

### Case 1: transfer_lr = α

```
y_final = C·x + α·(A_k·B_k)·x
```

이것은 학습 중 forward output과 **동일한 형태**:
```
y_train = C·x + α·A_k·(B_k·x) = C·x + α·(A_k·B_k)·x  ✅
```

**결론: transfer_lr = α일 때 학습 중 출력 = transfer 후 출력**

### Case 2: transfer_lr ≠ α

```
y_final = C·x + transfer_lr·(A_k·B_k)·x
y_train = C·x + α·(A_k·B_k)·x

y_final ≠ y_train  ❌
```

**출력이 달라집니다!**

---

## 더 깊은 분석: AB가 α에 의존하는가?

### AB의 업데이트가 α를 포함하는 효과

AB는 `lr_eff = lr·α`로 업데이트되므로:

```
A_k ∝ α  (업데이트 크기가 α에 비례)
B_k ∝ α  (업데이트 크기가 α에 비례)
```

**하지만!** 이것은 단순화된 분석입니다. 실제로:

1. **B_0는 이전 transfer에서 decayed**되어 옴 (hybrid reinit)
2. **Gradient D는 α·A·(B·x) 항을 포함**한 loss에서 계산됨
3. **AB의 절대 크기는 학습 dynamics에 의존**

---

## 실험적 관점

### 표준 LoRA에서의 일반적 설정

표준 LoRA (e.g., PEFT 라이브러리):
```python
# 학습 중
output = W·x + (α/r)·A·(B·x)  # r = rank

# Fine-tuning 후 merge
W_merged = W + (α/r)·(A @ B)
```

**Merge 시 동일한 α 사용** ✅

### LRTT의 경우

LRTT도 동일한 원리:
```python
# 학습 중 (forward_inject=True)
y = C·x + α·A·(B·x)

# Transfer
C_new = C + transfer_lr·(A @ B)

# Transfer 후 (forward_inject=False)
y_final = C_new·x
```

**일관성을 위해: transfer_lr = α 권장** ✅

---

## 현재 MDMLP 설정

### train_analog.py (Line 338-339, 697)
```python
parser.add_argument('--lora-alpha', type=float, default=1.0)
# ...
lora_alpha=args.lora_alpha,  # forward에서 사용
```

### sweep_lrtt_cifar10.py (Line 48, 67)
```python
transfer_lr = trial.suggest_float("transfer_lr", 1e-3, 10.0, log=True)
# ...
"--transfer-lr", str(transfer_lr),
```

**문제점:**
- lora_alpha = 1.0 (고정)
- transfer_lr = 1e-3 ~ 10.0 (탐색)

→ **transfer_lr과 lora_alpha가 독립적으로 설정됨!**

---

## 권장 설정

### 옵션 1: transfer_lr = lora_alpha로 고정 (이론적으로 올바름)

```python
# sweep_lrtt_cifar10.py
lora_alpha = trial.suggest_float("lora_alpha", 0.1, 10.0, log=True)
transfer_lr = lora_alpha  # 동일하게 설정

cmd = [
    "--lora-alpha", str(lora_alpha),
    "--transfer-lr", str(transfer_lr),  # = lora_alpha
]
```

### 옵션 2: transfer_lr를 독립 탐색 (실험적)

만약 transfer_lr ≠ lora_alpha가 더 좋은 결과를 준다면:
- 이는 다음을 의미할 수 있음:
  1. **Noise/quantization 보정**: Transfer 과정의 noise를 보정
  2. **Training dynamics**: AB가 예상과 다르게 수렴
  3. **Regularization 효과**: 다른 스케일이 일종의 regularization

현재 sweep은 **옵션 2**를 사용 중입니다.

---

## 실험 제안

### 실험 1: transfer_lr = lora_alpha 고정
```python
lora_alpha = 1.0
transfer_lr = 1.0
```

### 실험 2: transfer_lr = k * lora_alpha (비율 탐색)
```python
lora_alpha = 1.0
k = trial.suggest_float("transfer_lr_ratio", 0.1, 10.0, log=True)
transfer_lr = k * lora_alpha
```

### 실험 3: 독립 탐색 (현재 설정)
```python
lora_alpha = 1.0
transfer_lr = trial.suggest_float("transfer_lr", 1e-3, 10.0, log=True)
```

### 비교 지표
- **Consistency**: validation accuracy (C-only) vs training accuracy (C+AB)의 차이
- **Final accuracy**: transfer 후 최종 성능
- **Training stability**: loss curve의 안정성

---

## 결론

### 이론적 관점
**transfer_lr = lora_alpha가 수학적으로 올바름** ✅
- 학습 중 출력 = transfer 후 출력
- 표준 LoRA와 동일한 원리

### 실용적 관점
**transfer_lr ≠ lora_alpha도 유효할 수 있음**
- Analog noise/quantization 보정
- Empirical tuning의 여지

### 권장 사항
1. **먼저 transfer_lr = lora_alpha로 실험** (베이스라인)
2. **Sweep에서 비율을 탐색** (k = transfer_lr / lora_alpha)
3. **최적 비율을 찾으면 그것을 사용**

---

## 현재 상태 확인 필요

sweep_lrtt_cifar10.py가:
- lora_alpha = 1.0 (고정)
- transfer_lr = 1e-3 ~ 10.0 (탐색)

**만약 최적 transfer_lr이 1.0 근처라면?**
→ 이론이 맞습니다! ✅

**만약 최적 transfer_lr이 1.0에서 멀다면?**
→ Analog hardware 특성이나 다른 요인이 작용 중 🤔

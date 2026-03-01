# Sixt1c-LoRA Two-Mode Search Strategy (With Overlap)

## 핵심 발견: LR × Alpha 상호작용

**안정성 임계값**: `LR × Alpha < 0.004`

### 분석 결과
- **성공 Trial 평균**: LR×Alpha = 0.000511
- **실패 Trial 평균**: LR×Alpha = 0.007489 (14.6배 차이!)
- **최고 성능 (Trial 7)**: lr=0.0145, alpha=0.266, product=0.00386, F1=23.17%

---

## 🔵 Mode 1: Low-Medium LR + Medium-High Alpha (안정적)

### Search Space
```python
Learning Rate: [5e-4, 5e-3] = [0.0005, 0.005]
LoRA Alpha:    [0.3, 1.2]
```

### 특징
- **전략**: 낮은~중간 학습률 + 중간~높은 LoRA 기여도
- **장점**:
  - 안정적인 학습
  - 적절한 LoRA 표현력 (alpha 0.3~1.2)
  - 넓은 LR 커버리지
- **Max Product**: 0.005 × 1.2 = 0.006 (경계선)

### Default 시작점
```python
lr = 0.0015
alpha = 0.8
product = 0.0012  ✅ (안전)
```

### 적합한 경우
- 안정성과 성능의 균형
- 중간 수준의 LoRA 기여도 원할 때
- 넓은 LR 범위 탐색

---

## 🔴 Mode 2: Medium-High LR + Low-Medium Alpha (공격적)

### Search Space
```python
Learning Rate: [3e-3, 2e-2] = [0.003, 0.02]
LoRA Alpha:    [0.1, 0.4]
```

### 특징
- **전략**: 중간~높은 학습률 + 낮은~중간 LoRA 기여도
- **장점**:
  - 빠른 수렴 (Trial 7 전략)
  - Trial 7 영역 포함 (lr=0.0145, alpha=0.266)
  - 높은 LR로 효율적 학습
- **Max Product**: 0.02 × 0.4 = 0.008 (일부 위험)

### Default 시작점
```python
lr = 0.010
alpha = 0.25
product = 0.0025  ✅ (안전)
```

### 적합한 경우
- 최고 성능 추구 (F1 > 20%)
- Trial 7 성공 영역 탐색
- 빠른 학습 필요

---

## 🟣 Overlap Region (중복 영역)

### 범위
```python
Learning Rate: [3e-3, 5e-3] = [0.003, 0.005]
LoRA Alpha:    [0.3, 0.4]
```

### 특징
- **중심점**: lr=0.004, alpha=0.35, product=0.0014
- **의미**: 두 모드가 공통으로 탐색하는 영역
- **장점**:
  - 두 전략의 장점 결합
  - 매우 안전한 영역 (product < 0.002)
  - 일관성 검증 가능

### 예상 결과
- 두 모드에서 유사한 결과 나와야 함
- 재현성 확인 가능
- 중간 성능 (F1 ~12-15%)

---

## 📊 Search Space 커버리지

### 전체 탐색 공간
```
LR: [5e-4, 2e-2] = 40배 범위
Alpha: [0.1, 1.2] = 12배 범위
```

### Mode별 역할
- **Mode 1**: 안정성 중심 (좌측 하단 ~ 중앙)
- **Mode 2**: 성능 중심 (중앙 ~ 우측 상단, Trial 7 포함)
- **Overlap**: 검증 및 일관성 확인

---

## 🚀 실행 방법

### Mode 1 실행
```bash
cd /data/LRTT_transformer/LRTT_glue
nohup /data/venvs/aihwkit_gpu/bin/python sweep_sixt1c_lora_squad_adam.py \
  --mode mode1 --target QKV --n_trials 27 --epochs 3 \
  > mode1_qkv_batch256.log 2>&1 &
```

### Mode 2 실행
```bash
cd /data/LRTT_transformer/LRTT_glue
nohup /data/venvs/aihwkit_gpu/bin/python sweep_sixt1c_lora_squad_adam.py \
  --mode mode2 --target QKV --n_trials 27 --epochs 3 \
  > mode2_qkv_batch256.log 2>&1 &
```

### 동시 실행 (추천)
```bash
# 두 명령을 순차적으로 실행
cd /data/LRTT_transformer/LRTT_glue
nohup /data/venvs/aihwkit_gpu/bin/python sweep_sixt1c_lora_squad_adam.py --mode mode1 --target QKV --n_trials 27 --epochs 3 > mode1.log 2>&1 &
nohup /data/venvs/aihwkit_gpu/bin/python sweep_sixt1c_lora_squad_adam.py --mode mode2 --target QKV --n_trials 27 --epochs 3 > mode2.log 2>&1 &
```

---

## 📈 비교표

| 항목 | Mode 1 | Overlap | Mode 2 |
|------|--------|---------|--------|
| **LR 범위** | 0.0005~0.005 | 0.003~0.005 | 0.003~0.02 |
| **Alpha 범위** | 0.3~1.2 | 0.3~0.4 | 0.1~0.4 |
| **Max Product** | 0.006 | 0.002 | 0.008 |
| **전략** | 안정적 | 검증 | 공격적 |
| **Trial 7 포함** | ❌ | ❌ | ✅ |
| **F1 기대** | 10-15% | 12-15% | 20-25% |
| **실패 확률** | 낮음 | 매우 낮음 | 중간 |

---

## 🎯 제약 조건 분석

### LR × Alpha Product 분포

```
Mode 1:
  Min: 0.0005 × 0.3 = 0.00015  ✅ 매우 안전
  Max: 0.005 × 1.2 = 0.006     ⚠️ 경계선
  Avg: ~0.002                  ✅ 안전

Mode 2:
  Min: 0.003 × 0.1 = 0.0003    ✅ 매우 안전
  Max: 0.02 × 0.4 = 0.008      ⚠️ 위험
  Avg: ~0.004                  ✅ 안전

Overlap:
  Center: 0.004 × 0.35 = 0.0014  ✅ 매우 안전
```

### 안전성 지표
- < 0.002: 매우 안전 (Mode 1 대부분)
- 0.002-0.004: 안전 (양쪽 모두)
- 0.004-0.006: 경계선 (일부 trial)
- > 0.006: 위험 (Mode 2 일부 영역)

---

## 💡 예상 결과

### Mode 1
- 안정적인 학습
- 실패 trial < 10%
- 평균 F1: 12-15%
- 최고 F1: ~18%

### Mode 2
- 빠른 수렴
- 실패 trial ~20%
- 평균 F1: 15-18% (성공 trial만)
- 최고 F1: 20-25% (Trial 7 수준)

### Overlap
- 두 모드 일관성 확인
- 재현성 검증
- 안정적 중간 성능

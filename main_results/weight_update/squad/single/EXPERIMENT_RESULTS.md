# SingleRPU ConstantStep — SQuAD Weight Update 실험 결과

## 실험 개요

BERT-base를 SQuAD v1.1에서 finetuning할 때, analog ConstantStep device가 실제로 학습에 기여하는지 검증한다.

**공통 설정:**
- Model: `bert-base-uncased`
- Device: `ConstantStepDevice` (SingleRPU)
- Analog layers: Attention Q, K, V, O (실제로는 `attention` 카테고리 전체)
- Digital layers: qa_outputs, LayerNorm (`--nontarget-digital`)
- Forward/Backward IO: `is_perfect=True` (FP32, IO 영향 제거)
- dw_min: `2 / 2^bits`

## Phase 1: 10-step Grid Sweep

**파일:** `bit_lr_sweep_summary.csv`

10-step 학습으로 빠르게 analog contribution을 확인.

| 설정 | 값 |
|------|-----|
| Steps | 10 |
| Bits | 4, 6, 8, 10, 12, 14, 16, 32 |
| lr_analog | grid (log-uniform) |
| lr_digital | 0.0003, 0.001, 0.003, 0.01 |

### 핵심 결과

32-bit baseline (digital-only, lr_analog 효과 = 0): **loss = 3.93** (at lr_d=0.01)

| Bit | dw_min | Best Loss | Δ vs 32-bit | Verdict |
|-----|--------|-----------|-------------|---------|
| 4 | 1.25e-1 | 5.41 | +1.48 | HURTS |
| 6 | 3.13e-2 | 4.81 | +0.88 | HURTS |
| 8 | 7.81e-3 | 3.96 | +0.03 | neutral |
| 10 | 1.95e-3 | 3.23 | -0.70 | HELPS |
| 12 | 4.88e-4 | 2.21 | -1.72 | **HELPS** |
| 14 | 1.22e-4 | 1.96 | -1.97 | **HELPS** |
| 16 | 3.05e-5 | 2.16 | -1.77 | **HELPS** |

### 해석

- **12-14 bit에서 analog가 확실히 학습에 기여** (loss가 digital-only 대비 크게 감소)
- 4-8 bit: dw_min이 gradient 대비 너무 커서 destructive update 발생
- 이유: grad_absmean ~ 1.4e-4 ~ 6.4e-4 수준이므로, dw_min ≈ grad일 때 (12-14 bit) pulse update가 유효

## Phase 2: TPE Sweep (SQuAD 2-epoch)

**파일:** `tpe_sweep_squad_summary.csv`

Optuna TPE로 lr_analog, lr_digital 동시 탐색 (full SQuAD 2-epoch training).

| 설정 | 값 |
|------|-----|
| Epochs | 2 |
| Bits | 8, 10, 12, 14 |
| lr_analog | [0.01, 1.0] log-uniform |
| lr_digital | [0.001, 1.0] log-uniform |
| Trials | 20 per bit |
| Optimizer | AnalogSGD |

### Best F1 per Bit

| Bit | Best F1 | Best lr_analog | Best lr_digital |
|-----|---------|----------------|-----------------|
| 8 | 76.73 | 0.1753 | 0.9394 |
| 10 | 78.45 | 0.0128 | 0.7620 |
| 12 | 78.58 | 0.1281 | 0.8637 |
| 14 | 78.78 | 0.0145 | 0.6330 |

### 핵심 발견

1. **Digital LR이 F1을 지배** (상관계수 r ≈ 0.93)
   - lr_digital > 0.3이면 F1 > 77, lr_digital < 0.01이면 F1 < 65
   - lr_analog의 영향은 미미

2. **Bit 간 F1 차이가 작음** (76.7 ~ 78.8)
   - 높은 digital LR이 digital params (qa_outputs, LayerNorm)를 충분히 학습
   - Analog 기여분이 digital 학습에 묻힘

3. **Phase 1과의 차이 원인:**
   - Phase 1 (10-step): digital param이 아직 충분히 학습 안됨 → analog 기여 visible
   - Phase 2 (2-epoch): digital param이 이미 F1~78 달성 → analog 추가 기여 margin 작음

## Phase 3: Analog LR Sweep (Digital LR Fixed)

**파일:** `analog_lr_sweep_partial_summary.csv` (4/10 trials 완료, GPU 장애로 중단)

Digital LR을 per-bit best로 고정하고, analog LR만 TPE sweep.

| 설정 | 값 |
|------|-----|
| Epochs | 2 |
| lr_analog | [0.001, 10.0] log-uniform (기존보다 넓은 범위) |
| lr_digital | per-bit 고정 (Phase 2 best) |
| Trials | 4/10 완료 (GPU disconnection으로 중단) |

### Fixed Digital LR

| Bit | lr_digital (fixed) |
|-----|--------------------|
| 8 | 0.9394 |
| 10 | 0.7620 |
| 12 | 0.8637 |
| 14 | 0.6330 |

### Partial Results (4 trials per bit)

| Bit | lr_analog 범위 (tested) | Best F1 | Best lr_analog | F1 범위 |
|-----|------------------------|---------|----------------|---------|
| 8 | 0.008 ~ 0.095 | 76.07 | 0.0105 | 75.4 ~ 76.1 |
| 10 | 0.001 ~ 1.961 | 78.33 | 0.0070 | 77.9 ~ 78.3 |
| 12 | 0.001 ~ 0.118 | **78.85** | 0.0479 | 77.6 ~ 78.9 |
| 14 | 0.006 ~ 8.645 | 78.47 | 0.0216 | 78.1 ~ 78.5 |

### 해석

- **12-bit에서 F1=78.85**: Phase 2 best(78.58)보다 +0.27 개선 → analog LR 최적화 효과 있음
- 8-bit: F1 범위 좁음 (75.4~76.1) → analog가 거의 기여 못함 (dw_min >> grad)
- 10-14 bit: F1 범위 ~0.5 내외 → analog LR에 대한 약한 의존성
- **나머지 6 trials 실행 필요** (GPU 복구 후)

## Gradient & Update 분석

**파일:** `bit_sweep_constantstep/bit_sweep_constantstep_sweep_summary.csv`

### Per-Layer Gradient 분포 (dw_min=0.001, lr=0.002)

모든 72개 sublayer (12 blocks × Q,K,V,O,FFN1,FFN2)에서:

- `grad_absmean`: 1.4e-4 ~ 6.4e-4
- `grad_deadzone_ratio`: 100% (모든 gradient < dw_min)
- `BL_mean`: 1.0 (모든 layer에서 burst length = 1)
- `dw_zero_ratio`: 91% ~ 99%

### Sublayer별 Gradient 크기

| Sublayer | grad_absmean 범위 | 특징 |
|----------|-------------------|------|
| Q | 1.4e-4 ~ 3.0e-4 | 가장 작음 |
| K | 1.4e-4 ~ 2.7e-4 | Q와 유사 |
| V | 3.5e-4 ~ 6.1e-4 | Q,K의 ~2배 |
| O | 3.9e-4 ~ 6.4e-4 | 가장 큼 |
| FFN1 | 2.7e-4 ~ 4.3e-4 | 중간 |
| FFN2 | 2.1e-4 ~ 3.5e-4 | 중간 |

### Layer 깊이별 (Layer 0~11)

- 평균 gradient: 2.51e-4 (Layer 0) ~ 4.12e-4 (Layer 9)
- **~1.6배 차이 — 모든 layer가 비슷한 order** (LayerNorm 효과)
- 특정 layer만 학습이 안 되는 문제는 아님

### Bit별 grad/dw_min 비율

| Bit | dw_min | grad/dw_min (V,O) | grad/dw_min (Q,K) | 상태 |
|-----|--------|-------------------|-------------------|------|
| 8 | 7.81e-3 | 0.05~0.08 | 0.02~0.04 | dead zone |
| 10 | 1.95e-3 | 0.18~0.33 | 0.07~0.15 | 일부 active |
| 12 | 4.88e-4 | 0.72~1.31 | 0.29~0.61 | **유효** |
| 14 | 1.22e-4 | 2.87~5.25 | 1.15~2.46 | **완전 active** |

## 결론

1. **12-14 bit ConstantStep에서 analog tile은 학습에 기여한다** (Phase 1에서 확인)
2. **Full training에서 analog 기여 관찰이 어려운 이유**: digital param이 높은 LR로 F1~78까지 독자적으로 도달
3. **Analog contribution 분리를 위해**: digital LR 제어 또는 digital-only baseline과의 차이 비교 필요
4. **BL=1 제한**: 모든 layer에서 gradient가 작아 burst length=1, 하지만 12-bit 이상에서는 dw_min ≈ gradient이므로 pulse update 자체는 유효
5. **LayerNorm**: gradient를 작게 만드는 원인이 아님 — layer간 gradient 균일성의 원인. Gradient가 작은 것은 pretrained model finetuning의 본질적 특성

## 스크립트

| 스크립트 | 용도 |
|---------|------|
| `run_bit_lr_sweep.py` | Phase 1 & 2: bit × (lr_a, lr_d) TPE sweep |
| `run_analog_lr_sweep.py` | Phase 3: digital LR 고정, analog LR만 sweep |
| `analyze_phase1.py` | Phase 1 (10-step) 결과 분석 및 시각화 |
| `analyze_analog_contribution.py` | 32-bit baseline 대비 analog 기여 분석 |

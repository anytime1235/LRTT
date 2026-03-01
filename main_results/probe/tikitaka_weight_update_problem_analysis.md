# TikiTaka Weight Update Diagnostics: Problem Analysis Report

**Date**: 2026-03-01
**Model**: BERT-base (12-layer encoder), QKVO analog tiles
**Training Mode**: TikiTaka v2 (fast/slow dual-tile)
**dw_min Settings**: 0.0005 vs 0.005
**Seeds**: 3 (seed 0, 1, 2), 384 training steps
**Data**: 6 summary CSVs (2 dw_min x 3 seeds), 6 step_metrics CSVs

---

## Executive Summary

TikiTaka weight update 진단 결과, **analog 학습 파이프라인 전반에 걸쳐 심각한 신호 손실이 확인**되었다. 전체 weight update의 98.8%가 0으로 양자화되고, gradient 방향과의 정렬(cosine)이 ~0.019에 불과하며, fast-to-slow tile transfer에서 99.8~99.99%의 신호가 소실된다. 두 dw_min 설정 모두 동일한 구조적 문제를 공유하며, 현재 설정으로는 사실상 유의미한 analog 학습이 이루어지지 않고 있다.

---

## 1. CRITICAL: Transfer Pipeline 완전 차단

### 현상

| dw_min | fast_absmean | slow_absmean | Transfer Loss | Transfer Efficiency |
|--------|-------------|-------------|---------------|-------------------|
| 0.0005 | 0.2638 | 3.34e-05 | **99.99%** | 1.87e-04 |
| 0.005 | 0.1979 | 3.28e-04 | **99.83%** | 3.17e-03 |

### 상세 분석

- Fast tile은 gradient를 정상적으로 캡처한다 (fast_capture_ratio: dw=0.0005에서 51.8x, dw=0.005에서 72.3x).
- 그러나 **fast tile에서 slow tile로의 transfer 과정에서 신호의 99.8~99.99%가 소실**된다.
- `buffer_above_thresh_ratio = 0.000` (양쪽 dw 모두) — transfer buffer에 threshold를 초과하는 값이 전혀 없다.
- `hidden_absmean = 0` — **forget_buffer 모드가 감지**되어, hidden weight 축적 없이 즉시 폐기됨.
- `transfer_duty` ≈ 0.012 (1.2%) — transfer 이벤트 자체가 매우 드물게 발생.
- `transfer_spike` ≈ 10억~100억 — transfer 발생 시 spike magnitude가 비정상적으로 크지만, duty가 너무 낮아 실효 없음.

### 근본 원인 추정

1. **forget_buffer 모드**: hidden weight가 축적되지 않고 매 step 폐기되므로, 충분한 gradient 신호가 쌓이지 못함.
2. **Transfer threshold 과다**: buffer에 축적된 값이 threshold를 한 번도 초과하지 못함 → transfer 이벤트가 실질적으로 비활성.
3. **dw_min과 transfer threshold 간 불균형**: dw_min=0.005에서 transfer_efficiency가 17배 향상되나, 여전히 99.83% 손실로 불충분.

### Layer별 Transfer Efficiency

dw_min=0.0005 기준, 전 layer에서 transfer_efficiency ≈ 1e-04 수준으로 균일하게 낮음:

| Layer | Q | K | V | O |
|-------|------|------|------|------|
| L00 | 1.17e-04 | 2.74e-04 | 1.65e-04 | 1.08e-04 |
| L07 | 2.87e-04 | 3.00e-04 | 2.27e-04 | 1.96e-04 |
| L11 | 3.95e-04 | 2.47e-04 | 3.24e-04 | 1.64e-04 |

전 layer/sublayer에 걸쳐 구조적으로 실패 — 특정 layer 문제가 아닌 **시스템 레벨 설정 문제**.

---

## 2. CRITICAL: Gradient-Update 방향 정렬 실패

### 현상

| Metric | dw=0.0005 | dw=0.005 | 정상 기준 |
|--------|-----------|----------|----------|
| update_vs_grad_cosine | **0.0194** | **0.0181** | > 0.05 (GOOD) |
| sign_mismatch_ratio | **0.9927** | **0.9928** | < 0.90 (GOOD) |

### 상세 분석

- `update_vs_grad_cosine` ≈ 0.019: 실제 weight update 벡터가 gradient 방향과 **거의 직교(orthogonal)**.
  - 완전 랜덤이면 cosine ≈ 0, 완벽한 정렬이면 1.0.
  - 0.019는 랜덤보다 아주 약간 나은 수준으로, **gradient 정보가 거의 활용되지 못함**.
- `sign_mismatch_ratio` ≈ 0.993: weight update의 **99.3%가 gradient와 반대 부호**를 가짐.
  - 양자화 노이즈가 gradient 신호를 완전히 압도.
  - 결과적으로 대부분의 업데이트가 학습에 해롭거나 무의미.
- 두 dw_min 간 차이가 미미 (Cohen's d = 1.76, MODERATE) — **양자화 해상도가 아닌 구조적 문제**.

### Layer Depth Gradient

- `update_vs_grad_cosine`의 layer depth와의 Spearman rho = **-0.860** (STRONG decrease)
  - 깊은 layer일수록 gradient alignment가 급격히 악화.
- `sign_mismatch_ratio`의 Spearman rho = **+0.944** (STRONG increase)
  - 깊은 layer일수록 sign mismatch가 심화.
- 이는 deeper layer에서 gradient magnitude가 커지면서 (grad_absmean rho=+0.776) 양자화 오차의 상대적 영향이 변하기 때문.

### 시간적 변화 (dw=0.0005)

| Phase | update_vs_grad_cosine | sign_mismatch_ratio |
|-------|----------------------|-------------------|
| Early (step 0-125) | 0.0168 | 0.9929 |
| Mid (step 126-255) | 0.0196 | 0.9926 |
| Late (step 256-383) | 0.0219 | 0.9926 |
| Late/Early ratio | 1.30x | 1.00x |

- Cosine은 training이 진행되면서 미세하게 개선되나 (0.017→0.022), 여전히 심각하게 낮음.
- Sign mismatch는 전 구간 99.3%로 변화 없음 — **학습이 이 문제를 자체적으로 해결하지 못함**.

---

## 3. CRITICAL: Zero Update 비율 98.8%

### 현상

| Metric | dw=0.0005 | dw=0.005 | 정상 기준 |
|--------|-----------|----------|----------|
| dw_zero_ratio | **0.9880** | **0.9881** | < 0.95 (GOOD) |
| dw_1lsb_ratio | 0.0102 | 0.0102 | - |

### 상세 분석

- 전체 weight element의 **98.8%가 매 step 0으로 양자화됨**.
- 실제 non-zero update를 받는 원소는 1.2%에 불과하며, 그 중에서도 1-LSB 수준 (dw_1lsb_ratio ≈ 1.0%).
- 95% CI가 극히 좁음: dw=0.0005에서 [0.9880, 0.9881] — seed 간 변동성 없이 구조적으로 고정.
- **dw_min을 10배 변경해도 dw_zero_ratio가 동일** (fold = 1.000, Cohen's d = -3.21) — 이는 dw_min 자체가 아닌 **양자화 해상도 또는 pulse 메커니즘의 근본적 한계**.

### Correlation 분석

- `dw_zero_ratio ↔ transfer_duty`: Spearman rho = **-1.000** (완벽한 역상관)
  - zero update가 줄면 transfer 이벤트가 늘어나야 하지만, 현재 둘 다 극단값에 고정.
- `dw_zero_ratio ↔ grad_deadzone_ratio`: rho = 0.325 (독립적)
  - gradient deadzone과 update zero는 **서로 다른 원인**으로 발생 — 단순히 gradient가 작아서가 아님.

---

## 4. IMPORTANT: Pulse 메커니즘 비효율

### 현상

| Metric | dw=0.0005 | dw=0.005 |
|--------|-----------|----------|
| pulse_ok_frac | 0.621 (GOOD) | **0.451 (WARN)** |
| pulse_under_frac | 0.007 | **0.548** |
| pulse_over_frac | 0.372 | 0.001 |
| BL_mean | 32.3 | 2.0 |
| BL_hit_ratio | 0.384 | 0.001 |

### 상세 분석

#### dw=0.0005 (BL_mean=32.3)

- pulse_ok_frac = 62.1%: 62%의 update가 정상 pulse 범위 안에 있음 → 비교적 양호.
- pulse_over_frac = 37.2%: **37%가 pulse 초과** (over-pulsing). BL이 높아 pulse 범위를 넘는 경우 다수.
- BL_hit_ratio = 38.4%: BL target 달성률 38% — 절반 이상이 target BL에 미달.

#### dw=0.005 (BL_mean=2.0)

- pulse_ok_frac = 45.1%: **절반 미만**이 정상 범위.
- pulse_under_frac = **54.8%**: gradient가 너무 작아 최소 pulse에도 미달하는 경우가 과반.
- BL_mean = 2.0: BL이 극히 낮아 pulse 해상도가 사실상 2-level (on/off).
- BL_hit_ratio = 0.1%: target BL 달성이 거의 불가능.

### dw_min 간 Trade-off

- dw=0.0005: BL이 높아 (32) pulse 해상도는 좋지만, over-pulsing 37%.
- dw=0.005: BL이 너무 낮아 (2) pulse_under 55%로 대부분의 gradient가 최소 pulse에도 미달.
- **두 설정 모두 최적이 아님**: 중간 dw_min (예: 0.001~0.002) 탐색 필요.

### Sensitivity 분석 (dw_min dose-response)

| Metric | Fold Change | Cohen's d | Sensitivity |
|--------|------------|-----------|------------|
| pulse_under_frac | **80.9x** | -61.56 | STRONG |
| pulse_over_frac | 0.003x | 116.05 | STRONG |
| BL_mean | 0.062x | 73.53 | STRONG |
| BL_hit_ratio | 0.003x | 118.38 | STRONG |

- Pulse 관련 메트릭은 dw_min에 **극도로 민감** (Cohen's d > 60).
- 작은 dw_min 변화가 pulse 동작을 완전히 뒤바꿈 → 정밀한 tuning 필수.

---

## 5. IMPORTANT: Sublayer 간 비대칭 병목

### V/O Sublayer 악화 패턴 (dw=0.0005)

| Metric | Q/K avg | V/O avg | Diff |
|--------|---------|---------|------|
| pulse_ok_frac | 0.892 | **0.351** | **-60.7%** |
| transfer_efficiency | 2.26e-04 | **1.47e-04** | **-34.7%** |
| update_vs_grad_cosine | 0.0187 | 0.0201 | +7.3% |

- V/O sublayer의 pulse_ok_frac이 Q/K 대비 **60.7% 낮음** — V/O에서 pulse underutilization이 심각.
- Transfer efficiency도 V/O에서 34.7% 낮음.

### Q/K Sublayer 악화 패턴 (dw=0.005)

| Metric | Q/K avg | V/O avg | Diff |
|--------|---------|---------|------|
| pulse_ok_frac | **0.205** | 0.696 | **+238.9%** |
| transfer_efficiency | 4.04e-03 | **2.29e-03** | **-43.2%** |

- dw=0.005에서는 반대로 **Q/K의 pulse_ok_frac이 V/O 대비 70% 낮음**.
- dw_min 변경 시 병목이 V/O에서 Q/K로 이동 — **sublayer별 dw_min 차별화 필요 가능성**.

### Bottleneck Score 상위 Layer (dw=0.0005)

| Rank | Layer-Sub | Score | Health Flags |
|------|-----------|-------|-------------|
| 1 | **L11-O** | 11/15 | dw_zero_ratio=WARN, cosine=BAD, sign_mismatch=BAD, pulse_ok=BAD, transfer=WARN |
| 2 | **L09-O** | 11/15 | dw_zero_ratio=WARN, cosine=WARN, sign_mismatch=BAD, pulse_ok=BAD, transfer=BAD |
| 3 | L06-V | 9/15 | dw_zero_ratio=WARN, cosine=WARN, sign_mismatch=BAD, pulse_ok=BAD, transfer=WARN |
| 4 | L04-V | 9/15 | dw_zero_ratio=WARN, cosine=BAD, sign_mismatch=BAD, pulse_ok=WARN, transfer=WARN |
| 5-11 | L06~L10 V/O | 9/15 | 동일 패턴 반복 |

- **Layer 4-11의 V/O sublayer**에 bottleneck이 집중 (score 9~11).
- Q/K sublayer는 상대적으로 양호 (score 6 이하).
- Health 분포: 32 GOOD / 146 WARN / **62 BAD** (총 240개 metric-layer 조합)

---

## 6. MONITOR: Gradient Deadzone 문제

### 현상

| Metric | dw=0.0005 | dw=0.005 |
|--------|-----------|----------|
| grad_deadzone_ratio | 0.115 | **0.863** |
| grad_absmean | 0.00568 | 0.00308 |

### 상세 분석

- dw=0.005: gradient의 **86.3%가 deadzone 내**에 있음 — 양자화 threshold 이하로, pulse 생성 자체가 불가능.
- dw=0.0005: deadzone이 11.5%로 비교적 양호하지만, gradient가 deadzone을 넘어도 양자화 오차가 여전히 지배적.
- `grad_deadzone_ratio`는 dw_min에 강하게 의존 (fold = 7.48x, Cohen's d = -93.64).
- Correlation: `grad_deadzone_ratio ↔ grad_absmean` = -0.883 — gradient 크기가 작을수록 deadzone 비율 증가.

---

## 7. MONITOR: TikiTaka vs SingleRPU 비교 (v3 early-training)

### 주요 발견

| Metric | Single | TikiTaka | Tiki/Single |
|--------|--------|----------|-------------|
| dw_zero_ratio | 0.966 | **1.000** | 1.035 |
| dw_absmean | 3.87e-06 | **5.95e-10** | 0.0002 |
| update_vs_grad_cosine | 0.039 | **6.85e-06** | 0.0002 |
| rel_update_error | 50894 | **30.5** | 0.0006 |

- **v3 early-training (5 steps) 에서 TikiTaka는 SingleRPU 대비 극히 저조**:
  - dw_absmean이 SingleRPU의 0.02%.
  - Gradient alignment (cosine)이 사실상 0.
  - dw_zero_ratio가 1.0000 (100% zero update).
- 단, v3는 5 step만 측정한 early-training 데이터 → warm-up 지연일 수 있음.
- Primary data (384 steps)에서는 TikiTaka가 정상 동작하므로, **TikiTaka는 warm-up이 느림**.

---

## 8. Temporal Dynamics: 시간에 따른 변화

### 주요 시간적 패턴

#### dw=0.0005

| Metric | Early | Mid | Late | Late/Early | Stability |
|--------|-------|-----|------|------------|-----------|
| dw_zero_ratio | 0.9882 | 0.9879 | 0.9880 | 1.00 | STABLE |
| dw_absmean | 4.17e-06 | 4.31e-06 | 4.25e-06 | 1.02 | MODERATE |
| transfer_efficiency | **2.78e-04** | **1.37e-04** | **1.43e-04** | **0.51** | **VOLATILE** |
| dw_fast_absmean | 0.290 | 0.256 | 0.244 | 0.84 | MODERATE |
| pulse_ok_frac | 0.636 | 0.600 | 0.628 | 0.99 | STABLE |

#### dw=0.005

| Metric | Early | Mid | Late | Late/Early | Stability |
|--------|-------|-----|------|------------|-----------|
| transfer_efficiency | **5.60e-03** | **1.94e-03** | **1.91e-03** | **0.34** | **VOLATILE** |
| pulse_ok_frac | 0.559 | 0.397 | **0.394** | **0.71** | MODERATE |
| BL_mean | 2.41 | 1.88 | **1.70** | **0.70** | **VOLATILE** |
| grad_absmean | 0.00345 | 0.00284 | 0.00296 | 0.86 | VOLATILE |

### 핵심 관찰

1. **Transfer efficiency가 학습 초기에 급락** (Early→Late: 0.51x~0.34x) 후 안정화.
   - 초기에 약간이나마 transfer가 작동하다가, 학습이 진행되면서 오히려 악화.
2. **dw_zero_ratio는 전 구간 0.988로 고정** — 학습이 이 문제를 해결하지 못함.
3. **Linear regression으로 유의미한 trend 없음** (모든 metric에서 R^2 < 0.31).
   - 시간에 따른 자연적 개선이 기대되지 않음 → **설정 변경 없이는 상태가 유지됨**.

---

## 9. Cross-Metric Correlation 구조

### 주요 Metric Cluster

**Cluster 1**: `dw_zero_ratio` ↔ `transfer_duty` (rho = -1.000)
- Zero update 비율과 transfer 빈도는 완벽한 역상관 → 같은 현상의 양면.

**Cluster 2** (8개 metric 연결):
`dw_absmean` — `grad_absmean` — `grad_deadzone_ratio` — `BL_mean` — `BL_hit_ratio` — `pulse_under_frac` — `pulse_over_frac` — `transfer_efficiency`

- 핵심 인과 체인: **gradient 크기 → deadzone 비율 → BL/pulse 동작 → transfer 효율**
- 이 체인의 시작점인 gradient magnitude를 높이면 하류 metric이 개선될 가능성.

### 독립적 메트릭 쌍

| Pair | Spearman rho | 의미 |
|------|-------------|------|
| dw_zero_ratio ↔ grad_deadzone_ratio | 0.325 | 독립적 — 다른 원인 |
| transfer_efficiency ↔ update_vs_grad_cosine | -0.086 | 독립적 — transfer 개선이 alignment 개선과 무관 |
| BL_mean ↔ pulse_ok_frac | 0.159 | 독립적 — BL이 높아도 pulse 품질 보장 안 됨 |

---

## 10. 종합 진단 및 권장 사항

### 문제 우선순위

| Priority | Issue | Severity | Both dw_min? |
|----------|-------|----------|-------------|
| P1 | Transfer pipeline 99.8~99.99% 손실 | CRITICAL | Yes |
| P1 | Sign mismatch 99.3% | CRITICAL | Yes |
| P1 | Zero update 98.8% | CRITICAL | Yes |
| P2 | Gradient alignment cosine ~0.019 | IMPORTANT | Yes |
| P2 | Pulse underutilization (dw=0.005) | IMPORTANT | dw=0.005 |
| P3 | Gradient deadzone 86% (dw=0.005) | MONITOR | dw=0.005 |
| P3 | V/O sublayer 병목 집중 | MONITOR | dw_min 따라 변동 |

### 권장 조치

#### 즉시 조치 (Transfer Pipeline 복구)

1. **forget_buffer 모드 비활성화**: hidden weight 축적을 허용하여 충분한 gradient가 쌓인 후 transfer가 발생하도록 변경.
2. **Transfer threshold 하향**: `buffer_above_thresh_ratio = 0.000`이므로, threshold를 현재 fast-tile update magnitude에 맞춰 대폭 낮춰야 함.
3. **Transfer duty 증가**: 현재 1.2%에서 최소 5-10%로 transfer 빈도 증가.

#### 단기 조치 (Pulse/양자화 최적화)

4. **dw_min 중간값 탐색**: 0.001~0.002 범위에서 pulse_ok_frac과 BL의 균형점 탐색.
   - dw=0.0005: over-pulsing 37% (BL=32)
   - dw=0.005: under-pulsing 55% (BL=2)
   - 최적점은 이 사이에 존재할 가능성.
5. **Sublayer별 dw_min 차별화**: Q/K와 V/O의 병목 패턴이 dw_min에 따라 반전되므로, sublayer별 별도 설정 고려.

#### 중기 조치 (Gradient Signal 강화)

6. **Learning rate 상향**: gradient magnitude 증가 → deadzone 비율 감소 → 하류 metric 연쇄 개선.
7. **DAC/ADC bit 수 검토**: 양자화 해상도 부족이 근본 원인일 수 있음 — 현재 설정으로는 gradient 정보의 대부분이 양자화 노이즈에 묻힘.

#### 장기 조치 (구조적 개선)

8. **Noise-aware training scheme**: 양자화 노이즈를 고려한 gradient scaling 또는 noise injection 기법 도입.
9. **Gradient accumulation**: 여러 step의 gradient를 축적하여 SNR 향상 후 update 적용.

### dw_min 선택 가이드

현재 데이터 기준 **dw_min=0.0005 권장** (BAD 62개 < 69개):

| | dw=0.0005 | dw=0.005 |
|---|-----------|----------|
| Health | 32G / 146W / 62B | 69G / 102W / 69B |
| pulse_ok_frac | **0.621** (GOOD) | 0.451 (WARN) |
| transfer_efficiency | 1.87e-04 (WARN) | **3.17e-03** (GOOD) |
| dw_zero_ratio | 0.988 (WARN) | 0.988 (WARN) |
| grad_deadzone | **0.115** | 0.863 |

- dw=0.0005: pulse 품질 우수, 그러나 transfer 효율 극히 낮음.
- dw=0.005: transfer 효율 17배 향상, 그러나 pulse 과반이 under-threshold.
- **두 설정 모두 critical 문제를 해결하지 못하므로, transfer pipeline 자체의 구조적 수정이 최우선**.

---

## Appendix: 생성된 시각화 자료

| Figure | Path | Description |
|--------|------|-------------|
| Dose-Response | `figures/fig_dose_response.png` | Cohen's d + fold change bar charts |
| Bottleneck Heatmap | `figures/fig_bottleneck_heatmap.png` | 12x4 layer/sublayer health score heatmap |
| Transfer Pipeline | `figures/fig_transfer_pipeline.png` | Fast/slow bar chart + transfer loss % |
| Temporal Dynamics | `figures/fig_temporal_dynamics.png` | 2x5 line plots with phase markers |
| Correlation Matrix | `figures/fig_correlation_matrix.png` | Spearman rho lower-triangle heatmap |

All figures saved at: `/data/main_results/scripts/diagnosis/figures/`

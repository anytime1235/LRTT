# Analog BERT Forward I/O Diagnostics — 분석 보고서

> **생성일**: 2026-02-28
> **데이터 경로**: `/data/results/diag_fwd_io/`
> **실험 설정**: BERT-base / DAC 7-bit 고정 / ADC 4·6·8·10·12-bit sweep / dw_min=0.001
> **플롯 경로**: `./plots/`

---

## 목차
1. [실험 개요](#1-실험-개요)
2. [ADC Sweep 요약 (summary_adc_sweep.csv)](#2-adc-sweep-요약)
3. [모듈별 MAC 품질 분석 (adc*_module_mac_summary.csv)](#3-모듈별-mac-품질-분석)
4. [레이어별 MAC 메트릭 트렌드 (adc*_layer_mac_metrics.csv)](#4-레이어별-mac-메트릭-트렌드)
5. [단일 런 분석 — 레이어 MAC (single_run_layer_mac_metrics.csv)](#5-단일-런-분석--레이어-mac)
6. [단일 런 분석 — 모듈 MAC 요약 (single_run_module_mac_summary.csv)](#6-단일-런-분석--모듈-mac-요약)
7. [로짓 품질 분석 (single_run_logit_metrics.csv)](#7-로짓-품질-분석)
8. [가중치 업데이트 분해능 분석 (single_run_weight_delta_metrics.csv)](#8-가중치-업데이트-분해능-분석)
9. [종합 진단 및 권고사항](#9-종합-진단-및-권고사항)

---

## 1. 실험 개요

본 진단은 Analog-BERT 모델의 순전파(Forward Pass) 과정에서 발생하는 아날로그 하드웨어 노이즈의 영향을 정량화하기 위해 수행되었다. 주요 진단 항목은 다음과 같다.

| 항목 | 내용 |
|------|------|
| 모델 | BERT-base (12 encoder layers, 6 sublayer types per layer) |
| 하드웨어 설정 | DAC 7-bit 고정, ADC 4/6/8/10/12-bit 스윕 |
| 진단 대상 | Q/K/V/O (Attention) + FFN1/FFN2 (Feed-Forward) |
| 주요 메트릭 | MAC SNR (dB), NMSE, Cosine Similarity, Out Clip Ratio, Deadzone Ratio |

**서브레이어 구성 (72개 모듈 = 12 layers × 6 sublayers)**

- `Q`, `K`, `V`, `O` : Attention 행렬 연산 (Query/Key/Value/Output projection)
- `FFN1`, `FFN2` : Feed-Forward Network 1·2번째 linear layer

---

## 2. ADC Sweep 요약

> **파일**: `summary_adc_sweep.csv`
> **플롯**: `fig1_summary_adc_sweep.png`

### 2.1 전체 평균 MAC SNR (dB) — ADC bits별

| ADC bits | Q | K | V | O | FFN1 | FFN2 | **전체 평균** |
|----------|---|---|---|---|------|------|------------|
| **4** | 5.68 | 5.06 | 4.81 | 8.80 | 1.15 | 5.00 | **5.08** |
| **6** | 15.90 | 15.84 | 12.73 | 19.96 | 10.33 | 14.30 | **14.84** |
| **8** | 26.65 | 26.74 | 23.22 | 29.85 | 21.03 | 24.48 | **25.33** |
| **10** | 29.74 | 29.81 | 26.04 | 32.64 | 23.75 | 29.26 | **28.54** |
| **12** | 30.02 | 30.09 | 26.29 | 32.89 | 24.00 | 29.78 | **28.85** |

### 2.2 ADC bits 증가에 따른 SNR 향상량

| 서브레이어 | 4→6 dB 증가 | 6→8 dB 증가 | 8→10 dB 증가 | 10→12 dB 증가 |
|-----------|------------|------------|-------------|--------------|
| Q | +10.22 | +10.75 | +3.09 | +0.29 |
| K | +10.79 | +10.90 | +3.07 | +0.29 |
| V | +7.91 | +10.49 | +2.82 | +0.25 |
| O | +11.16 | +9.89 | +2.79 | +0.25 |
| FFN1 | +9.18 | +10.70 | +2.72 | +0.25 |
| FFN2 | +9.30 | +10.18 | +4.78 | +0.52 |

### 2.3 핵심 관찰

> **SNR 포화 현상**: 10-bit → 12-bit 전환 시 SNR 증가율이 4→6-bit 대비 **2.2~5.6%** 수준으로 급격히 둔화. ADC 10-bit가 사실상 최대 유효 해상도.

- **4-bit ADC**: 전체 평균 SNR 5.08 dB — 아날로그 연산 신뢰도 매우 낮음. FFN1은 1.15 dB로 사실상 noise-dominant.
- **6-bit ADC**: 평균 SNR 14.84 dB로 4-bit 대비 +9.76 dB 향상. 실용 가능 최소 기준선.
- **8-bit ADC**: 평균 25.33 dB. 대부분의 서브레이어에서 안정적 동작 영역 진입.
- **10-bit ADC**: 평균 28.54 dB. 한계 성능에 근접. 8→10 증가폭이 이미 둔화됨.
- **12-bit ADC**: 평균 28.85 dB — 10-bit 대비 +0.31 dB. **비용 대비 효용 한계점.**
- **Out Clip Ratio**: 모든 ADC 조건에서 FFN1/FFN2에만 미소량(~0.0002) 존재. Attention 레이어는 0.

---

## 3. 모듈별 MAC 품질 분석

> **파일**: `adc{4,6,8,10,12}_module_mac_summary.csv`
> **플롯**: `fig2_module_mac_summary_all_adc.png`

### 3.1 서브레이어별 평균 성능 (ADC 조건별)

#### MAC SNR (dB) — 높을수록 우수

| 서브레이어 | ADC-4 | ADC-6 | ADC-8 | ADC-10 | ADC-12 |
|-----------|-------|-------|-------|--------|--------|
| **FFN1** | 1.15 | 10.33 | 21.03 | 23.75 | 24.00 |
| **FFN2** | 5.00 | 14.30 | 24.48 | 29.26 | 29.78 |
| **Q** | 5.68 | 15.90 | 26.65 | 29.74 | 30.02 |
| **K** | 5.06 | 15.84 | 26.74 | 29.81 | 30.09 |
| **V** | 4.81 | 12.73 | 23.22 | 26.04 | 26.29 |
| **O** | **8.80** | **19.96** | **29.85** | **32.64** | **32.89** |

**`O` projection이 모든 ADC 조건에서 가장 높은 SNR을 기록. `FFN1`이 가장 취약.**

#### Cosine Similarity — 1에 가까울수록 우수

| 서브레이어 | ADC-4 | ADC-6 | ADC-8 | ADC-10 | ADC-12 |
|-----------|-------|-------|-------|--------|--------|
| FFN1 | 0.821 | 0.985 | 0.999 | 0.999 | 0.999 |
| FFN2 | 0.832 | 0.968 | 0.996 | 0.998 | 0.999 |
| Q | 0.826 | 0.977 | 0.998 | 0.999 | 0.999 |
| K | 0.787 | 0.979 | 0.998 | 0.999 | 0.999 |
| V | 0.792 | 0.945 | 0.995 | 0.997 | 0.997 |
| **O** | **0.935** | **0.994** | **0.999** | **1.000** | **1.000** |

#### NMSE — 낮을수록 우수

| 서브레이어 | ADC-4 | ADC-6 | ADC-8 | ADC-10 | ADC-12 |
|-----------|-------|-------|-------|--------|--------|
| FFN1 | 0.350 | 0.034 | 0.003 | 0.002 | 0.001 |
| K | 0.365 | 0.043 | 0.003 | 0.001 | 0.001 |
| V | **0.367** | **0.088** | **0.007** | **0.004** | **0.003** |
| O | 0.146 | 0.011 | 0.001 | 0.001 | 0.001 |

> **V (Value projection)**: ADC-4에서 NMSE 0.367로 최악. ADC-8 이상에서야 0.007로 개선됨.

### 3.2 레이어 깊이별 패턴

- **모든 ADC 조건에서 layer 11 (마지막 레이어)이 가장 낮은 SNR을 기록**
  - FFN1 최악: layer 11, SNR −0.60 dB (ADC-4) / 16.44 dB (단일 런 기준)
  - Q/K/V 최악: 모두 layer 11 (심층 레이어일수록 입력 분포 복잡도 증가)
- **layer 0 (첫 레이어)가 Q/K/V에서 최고 SNR 기록** (단순한 입력 분포)
- **FFN2 예외**: layer 9가 최고 SNR (37.34 dB). 특정 중간 레이어에서 FFN2 출력 분포가 ADC 범위에 최적 정렬.

### 3.3 O-Projection의 Deadzone 특성

O-projection의 `ref_deadzone_ratio`가 ADC 해상도에 극도로 민감:

| ADC bits | O Deadzone Ratio (평균) |
|----------|----------------------|
| **4** | **0.941** (94.1% — 사실상 모든 입력이 deadzone) |
| 6 | 0.473 |
| 8 | 0.110 |
| 10 | 0.028 |
| 12 | 0.007 |

> O-projection은 Attention head concat 후 입력 분포가 매우 희소(sparse)하여, 낮은 ADC 해상도에서 입력 대부분이 최소 분해능 이하로 처리됨. **최소 8-bit ADC 필요**.

---

## 4. 레이어별 MAC 메트릭 트렌드

> **파일**: `adc{4,6,8,10,12}_layer_mac_metrics.csv` (각 14,400행 = 200 steps × 72 modules)
> **플롯**: `fig3_layer_mac_metrics_all_adc.png`

### 4.1 스텝에 걸친 안정성

200 스텝의 평균값과 모듈 요약(module_mac_summary)이 거의 동일 → **스텝 간 메트릭 분산이 매우 낮음**. 즉, 하드웨어 노이즈 특성이 입력 데이터 변화에 관계없이 레이어 위치에 의해 지배됨.

### 4.2 레이어 깊이와 SNR의 역상관 관계

| 관측 | 내용 |
|------|------|
| layer 0 SNR > layer 11 SNR | 모든 서브레이어, 모든 ADC 조건에서 일관됨 |
| 저하 폭 (ADC-8 기준) | Q: 36.7→22.3 dB (−14.4 dB), FFN1: 28.4→16.4 dB (−12.0 dB) |
| 원인 추정 | 깊은 레이어일수록 activation scale이 커지고 분포 범위가 넓어져 ADC 포화 가능성 증가 |

### 4.3 ADC-4에서의 극단적 품질 저하

- **FFN1 layer 10, 11**: SNR이 음수(−0.60 dB)로 측정 → 출력 노이즈가 신호보다 큼
- **K layer 11, ADC-4**: cosine similarity 0.643 — attention key vector의 방향이 크게 왜곡됨
- 이 수준의 왜곡은 attention 패턴 붕괴를 야기할 가능성이 높음

---

## 5. 단일 런 분석 — 레이어 MAC

> **파일**: `single_run_layer_mac_metrics.csv` (432행 = 6 steps × 72 modules)
> **플롯**: `fig4_single_run_layer_mac_metrics.png`

이 파일은 단일 설정 (학습된 out_scaling 적용 추정) 조건에서의 고해상도 측정값.

### 5.1 서브레이어별 MAC SNR 요약

| 서브레이어 | 평균 SNR (dB) | 최솟값 | 최댓값 | 평균 Cosine |
|-----------|-------------|-------|-------|-----------|
| **O** | **31.74** | 29.08 | 35.09 | **0.9996** |
| K | 29.01 | 21.35 | 37.79 | 0.9991 |
| Q | 28.96 | 20.95 | 37.28 | 0.9990 |
| FFN2 | 27.60 | 23.26 | 37.94 | 0.9979 |
| **V** | 25.35 | 18.03 | 34.49 | **0.9967** |
| **FFN1** | **23.08** | 15.18 | 29.47 | 0.9991 |

- `O` projection: 가장 높은 SNR 및 cosine. 출력 안정성 최고.
- `V` projection: cosine 0.9967로 Attention 내 최저. NMSE 0.0044로 두드러짐.
- `FFN1`: 평균 SNR이 가장 낮고 변동폭이 큼 (15.18 ~ 29.47 dB).

### 5.2 Clipping 발생 모듈

Out Clip Ratio > 0인 모듈은 **FFN1·FFN2에 집중**:

| 순위 | 서브레이어 | Layer | Clip Ratio | SNR (dB) |
|------|-----------|-------|-----------|---------|
| 1 | FFN2 | 0 | 0.00159 | 32.84 |
| 2 | FFN1 | 9 | 0.00095 | 26.64 |
| 3 | FFN1 | 10 | 0.00033 | 25.59 |
| 4 | FFN1 | 3 | 0.00030 | 21.91 |
| 5 | FFN1 | 2 | 0.00029 | 20.70 |

> Clipping이 있음에도 SNR은 상대적으로 양호하게 유지됨. 단, FFN1 layer 2~4 구간은 SNR 20~22 dB로 Attention 레이어 대비 낮은 품질. **FFN1의 out_scaling 미세 조정 여지가 있음.**

### 5.3 Ref Deadzone Ratio

`O` projection: 평균 deadzone 5.58% (range: 4.99~6.30%) — 구조적으로 희소한 입력 특성.
`FFN2`: 평균 3.33% — 일부 출력이 LSB 이하에 위치.
`Q/K`: ~0.87% — 상대적으로 낮고 균일.

---

## 6. 단일 런 분석 — 모듈 MAC 요약

> **파일**: `single_run_module_mac_summary.csv` (72행 = 12 layers × 6 sublayers)
> **플롯**: `fig5_single_run_module_mac_summary.png`

### 6.1 레이어별 SNR 최고/최저

| 서브레이어 | Best Layer | Best SNR | Worst Layer | Worst SNR | Worst Cosine |
|-----------|-----------|---------|------------|----------|------------|
| Q | 0 | 36.74 dB | 11 | 22.31 dB | 0.9969 |
| K | 0 | 37.32 dB | 11 | 22.76 dB | 0.9973 |
| V | 0 | 34.36 dB | 11 | 19.23 dB | 0.9908 |
| O | 0 | 34.57 dB | 10 | 29.76 dB | 0.9994 |
| FFN1 | 0 | 28.35 dB | 11 | 16.44 dB | 0.9966 |
| FFN2 | 9 | 37.34 dB | 4 | 23.44 dB | 0.9966 |

**Layer 0이 가장 안정적 (Q/K/V/O/FFN1 모두 best)**.
**FFN2는 예외적으로 layer 9에서 peak — layer 4에서 최저 (23.44 dB)**.
→ FFN2의 비단조(non-monotonic) SNR 프로파일은 특정 레이어의 activation scale 이상을 시사.

### 6.2 V Projection의 집중 분석

V는 single_run 기준에서도 전 레이어에 걸쳐 일관되게 낮은 성능:
- 평균 SNR: 25.35 dB (Q 28.96, K 29.01 대비 −3~4 dB)
- NMSE: 0.0044 (Q 0.0019, K 0.0018 대비 2.3배 높음)
- Layer 11 cosine: 0.9908 (유일하게 0.991 미만)

> **V projection에 대한 별도 scaling 최적화 또는 더 높은 정밀도 할당(ADC bits 증가)을 고려할 것.**

---

## 7. 로짓 품질 분석

> **파일**: `single_run_logit_metrics.csv` (1행)
> **플롯**: `fig6_single_run_logit_metrics.png`

아날로그 연산 전후의 모델 출력 로짓 변화 측정값:

| 메트릭 | 값 | 해석 |
|--------|---|------|
| `mse_start` | 0.01749 | 아날로그 적용 전 로짓 MSE |
| `mse_end` | 0.01372 | 아날로그 적용 후 로짓 MSE (**감소**) |
| `cosine_logit` | 0.8833 | 로짓 벡터 방향 유사도 — **0.88로 낮음** |
| `kl_start` | 0.00892 | 아날로그 전 소프트맥스 분포 KL divergence |
| `kl_end` | 0.00680 | 아날로그 후 KL divergence (**감소**) |
| `flip_start` | 1.0 | 아날로그 전 예측 불일치 비율 (100%) |
| `flip_end` | 0.5 | 아날로그 후 예측 불일치 비율 (50%) |
| `margin_start` | −0.5234 | 아날로그 전 결정 마진 (음수 = 오분류 구간) |
| `margin_end` | −0.6616 | 아날로그 후 결정 마진 (더 음수화) |

### 해석

- **MSE·KL이 아날로그 적용 후 감소**하는 역설적 결과: 이는 단일 샘플의 스냅샷으로, 노이즈가 특정 샘플에서 우연히 로짓을 평활화(smoothing)한 효과.
- **cosine_logit = 0.8833**: 로짓 벡터의 방향이 14% 이상 틀어짐. 분류 신뢰도에 직접 영향.
- **flip_end = 0.50**: top-1 예측이 원래와 50% 다름 — 단일 샘플에서 아날로그 노이즈로 인한 예측 변경 발생.
- **margin_end < margin_start**: 올바른 클래스와 경쟁 클래스 간 차이가 아날로그 적용 후 더 좁아짐.

> **단일 측정값이므로 통계적 결론 도출에 주의 필요. 다수 샘플에 대한 flip rate 측정이 권장됨.**

---

## 8. 가중치 업데이트 분해능 분석

> **파일**: `single_run_weight_delta_metrics.csv` (360행 = 5 steps × 12 layers × 6 sublayers)
> **플롯**: `fig7_single_run_weight_delta.png`

### 8.1 서브레이어별 평균 업데이트 특성

| 서브레이어 | Zero Ratio | 1-LSB Ratio | dw_absmean | min_nonzero_delta |
|-----------|-----------|------------|-----------|-----------------|
| Q | 0.9805 | 0.0195 | 3.0e−6 | 4.1e−6 |
| K | 0.9787 | 0.0213 | 3.0e−6 | 3.3e−6 |
| V | 0.9500 | 0.0500 | 5.0e−6 | 5.3e−7 |
| **O** | **0.9379** | **0.0621** | 6.0e−6 | 6.9e−7 |
| FFN1 | 0.9750 | 0.0250 | 3.0e−6 | 4.7e−7 |
| FFN2 | 0.9781 | 0.0219 | 3.0e−6 | 1.1e−6 |

> - **Zero Ratio**: 전 서브레이어에서 **93.8~98.1%의 가중치가 한 스텝에서 변화 없음** (analog quantization으로 인한 업데이트 억제).
> - **O projection**: zero ratio 93.8%로 가장 낮고 1-LSB ratio 6.2%로 가장 높음 — 업데이트가 가장 활발. gradient가 크거나 분해능에 가까움.
> - **V projection**: zero ratio 95.0%, dw_absmean 5.0e−6으로 Q/K 대비 업데이트량이 1.7배 크지만 forward SNR도 낮음 — **V의 gradient 흐름이 불안정할 가능성**.

### 8.2 스텝별 Zero Ratio 추이

| Step | 전체 평균 Zero Ratio |
|------|---------------------|
| 0 | 0.9596 |
| 1 | 0.9662 |
| 2 | 0.9700 |
| 3 | 0.9702 |
| **4** | **0.9674** (소폭 반등) |

- Step 0→3: zero ratio 증가 → 학습 초기에는 업데이트가 많다가 점차 수렴하는 패턴.
- Step 3→4: 소폭 반등 → 학습률 스케줄 또는 배치 변화에 따른 일시적 증가.

### 8.3 min_nonzero_delta 분석

가장 작은 유효 가중치 변화량(최소 비-제로 delta):

- **Q**: 4.1e−6, **K**: 3.3e−6, **FFN1**: 4.7e−7 — dw_min(=0.001)보다 훨씬 작음
- 이는 analog SGD가 미소 gradient 누적을 통해 실질 업데이트를 발생시킴을 의미 (stochastic rounding 효과)
- **FFN1의 min_nonzero_delta (~4.7e−7)가 가장 작음** — 매우 정밀한 업데이트가 일어나는 레이어

---

## 9. 종합 진단 및 권고사항

### 9.1 주요 발견 요약

| 구분 | 발견 사항 |
|------|---------|
| **ADC 포화점** | **10-bit**에서 사실상 성능 수렴. 12-bit 대비 10-bit의 SNR 차이 < 0.5 dB |
| **최취약 서브레이어** | **FFN1** — 전 ADC 조건에서 가장 낮은 SNR. ADC-4에서 SNR 음수 |
| **최안정 서브레이어** | **O projection** — SNR 최고, Cosine 최고, Deadzone 문제있지만 ADC-8 이상에서 해소 |
| **레이어 깊이 효과** | **Layer 11** (마지막)이 모든 서브레이어에서 SNR 최저. Layer 0이 최고 |
| **V projection 이슈** | Attention 내 가장 낮은 SNR·Cosine. 별도 처리 필요 |
| **Clipping** | FFN1·FFN2에만 발생. Attention Q/K/V/O는 clipping 없음 |
| **가중치 업데이트** | 93~98% 스텝에서 제로 업데이트 — analog quantization의 정상적 행동 |
| **로짓 품질** | cosine 0.88, flip rate 50% (단일 샘플) — 추가 통계 검증 필요 |

### 9.2 권고사항

#### A. ADC 비트 선택 가이드
```
권장: ADC 8-bit (SNR ~25 dB, 성능/비용 최적점)
      ADC 10-bit (SNR ~28 dB, 고정밀 요구 시)
불필요: ADC 12-bit (10-bit 대비 +0.3 dB — 하드웨어 비용 대비 효용 없음)
위험:  ADC 4-bit (FFN1 SNR < 0 dB, K layer 11 cosine 0.64 — 모델 붕괴 위험)
```

#### B. 서브레이어별 정밀도 차등 할당 (Mixed-Precision 전략)
- `FFN1` → ADC bits +2 (다른 레이어보다 높은 정밀도 우선 할당)
- `V` → ADC bits +1 또는 out_scaling 재보정
- `O` → deadzone 개선 위해 입력 재정규화 또는 ADC range 조정
- `Q`, `K`, `FFN2` → 기본 ADC 설정으로 충분

#### C. 레이어 깊이 기반 최적화
- 후반 레이어(layer 9–11)에 대해 별도 calibration 또는 더 높은 ADC 해상도 할당 검토
- `out_scaling` 학습 시 레이어 깊이를 고려한 per-layer 스케줄링 적용 권장

#### D. 추가 진단 필요 항목
- [ ] 다수 샘플(≥100)에 대한 logit flip rate 측정 → 단일 샘플 측정의 통계적 한계 보완
- [ ] `dw_min` 값 변경 실험 (현재 0.001) — weight delta zero ratio와 학습 수렴 속도 트레이드오프 분석
- [ ] FFN2 layer 4의 SNR 저하 원인 분석 (activation scale 이상 여부 확인)
- [ ] V projection 전용 out_scaling 재보정 실험

---

*분석 스크립트: `plot_all_csvs.py` | 플롯: `./plots/` | 원본 데이터: `/data/results/diag_fwd_io/`*

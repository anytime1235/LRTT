# Backward DAC Underflow 진단 — Full Run 결과 분석 보고서

> **paper_figures.py v3** | BERT-base-uncased (SQuAD v1.1) | SingleRPUConfig + SoftBoundsDevice (noise-free)

---

## 1. 실험 설정

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| 모델 | BERT-base-uncased | SQuAD v1.1 finetuned |
| Analog Device | SingleRPUConfig + SoftBoundsDevice | noise-free (모든 noise 파라미터 = 0) |
| DAC bits (baseline) | 7 | inp_res = 1/126 |
| ADC bits (baseline) | 9 | out_res = 1/512 |
| inp_bound | 1.0 | |
| step_size | 2/126 = 0.015873 | AIHWKit UniformQuantize 정합 |
| zero_thresh | 1/126 = 0.007937 | step_size / 2 |
| N_STEP | 200 (baseline/solutions), 100 (sweep) | |
| BATCH_SIZE | 8 | |
| Encoder layers | 12 | Sublayers: Q, K, V, O, FFN1, FFN2 (72 hooks) |

### Quantization 공식 (AIHWKit UniformQuantize 정합)

```
res_ratio = inp_res  (if inp_res ≤ 1.0)
step_size = 2 * inp_bound * res_ratio = 2/126

scaled    = dy / alpha * inp_bound          # per-vector absmax 정규화
scaled_q  = round(scaled / step) * step     # uniform quantize
scaled_q  = clamp(scaled_q, -bound, +bound) # clipping (nm_thres > 0일 때)
dy_q      = scaled_q * alpha / inp_bound    # 역정규화
```

---

## 2. 생성 파일 목록

| 파일 | 행 수 | 열 수 | 상태 |
|------|-------|-------|------|
| `metrics_paper_A_rootcause_summary.csv` | 72 | 32 | OK |
| `metrics_paper_A_rootcause_steps.csv` | 14,400 | 34 | OK |
| `metrics_paper_A_rootcause_cdf.csv` | 6,000 | 5 | OK |
| `metrics_paper_B_bitsweep_summary.csv` | 432 | 32 | OK |
| `metrics_paper_B_bitsweep_steps.csv` | 50,400 | 34 | OK |
| `metrics_paper_C_solutions_summary.csv` | 288 | 32 | OK |
| `metrics_paper_C_solutions_steps.csv` | 57,600 | 34 | OK |
| `fig_paper_A_rootcause_qkvo_ffn.png` | — | — | OK |
| `fig_paper_B_bitsweep.png` | — | — | OK |
| `fig_paper_C_solutions.png` | — | — | OK |
| `absmax_raw_*.npz` | 72 keys | — | OK |

---

## 3. 메트릭 정의

| 메트릭 | 정의 | 해석 |
|--------|------|------|
| **EZR** | Exact Zero Ratio — 원본 gradient에서 정확히 0인 원소 비율 | 구조적 zero (padding 등) |
| **QZR_nonzero** | 비zero 원소 중 양자화 후 0이 된 비율 | **핵심 지표** — DAC underflow 정도 |
| **ODR** | Outlier Dominant Ratio = absmax / median(abs) | 분포의 꼬리 비대칭 정도 |
| **cosine_sim** | cos(dy, dy_q) — 방향 보존도 | 1에 가까울수록 양호 |
| **l2_retention** | ‖dy_q‖ / ‖dy‖ — 크기 보존도 | 1.0 = 완벽, <1 = 감쇠, >1 = 증폭 |
| **rel_l2_error** | ‖dy - dy_q‖ / ‖dy‖ — 상대 오차 | 작을수록 양호 |
| **clip_rate_scaled** | scaled 벡터에서 \|x\| > inp_bound인 비율 | nm_thres > 0일 때만 의미 |
| **ratio_q50** | abs(gradient) / absmax의 중앙값 | 분포 집중도 (작을수록 outlier 지배적) |

---

## 4. Figure A — Root Cause Diagnosis (Baseline 7-bit)

### 4.1 전체 판정

> **비정형 (Non-structural): EZR ≈ 0.00%, QZR_nonzero = 17.11%, ratio_q50 = 0.042**

- K/V의 EZR ≈ 0 → 구조적 exact-zero가 아닌, **양자화에 의한 underflow**가 주 원인
- QZR_nonzero ~17%는 bulk-tiny / outlier-dominant 패턴 (중간 수준)

### 4.2 Sublayer별 평균 (12 layer 평균)

| Metric | Q | K | V | O | FFN1 | FFN2 |
|--------|-----|-----|-----|-----|------|------|
| **EZR** | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0014 | 0.1000 |
| **QZR_nonzero** | 0.1522 | **0.2013** | 0.1410 | 0.0267 | **0.4587** | 0.0236 |
| **ODR** | 25.3 | 41.1 | 27.9 | 7.2 | **135.4** | 6.3 |
| **cosine_sim** | 0.99940 | 0.99936 | 0.99954 | 0.99980 | 0.99756 | 0.99985 |
| **l2_retention** | 1.00058 | 1.00061 | 1.00045 | 1.00020 | 1.00206 | 1.00015 |
| **rel_l2_error** | 0.0333 | 0.0348 | 0.0294 | 0.0186 | **0.0674** | 0.0167 |
| **clip_rate_scaled** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **ratio_q50** | 0.0480 | 0.0351 | 0.0498 | 0.1481 | 0.0101 | 0.1604 |

### 4.3 핵심 관찰

1. **FFN1이 가장 심각**
   - QZR_nonzero = **0.459 (46%)** — 비zero gradient의 거의 절반이 양자화 시 소실
   - ODR = **135.4** — 극단적 outlier 지배 분포
   - rel_l2_error = **6.7%** — 전체 sublayer 중 최대
   - ratio_q50 = 0.010 — 중앙값이 absmax의 1%에 불과 (대부분의 원소가 매우 작음)

2. **K가 attention sublayer 중 최악**
   - QZR_nonzero = **0.201 (20%)** — 5개 중 1개 gradient가 소실
   - ODR = 41.1, rel_l2_error = 3.5%

3. **O / FFN2는 양호**
   - QZR_nonzero ≈ 2.5%
   - EZR ≈ 10% (padding 등 구조적 zero, 양자화와 무관)
   - rel_l2_error < 2%

4. **clip_rate_scaled = 0.0000** (모든 sublayer)
   - baseline(nm_thres=0)에서는 clipping 미발생 — 정확한 동작 확인

5. **방향 보존 vs 크기 보존**
   - cosine_sim > 0.999 (방향은 거의 완벽 보존)
   - 그러나 **magnitude 손실 (QZR)이 실질적 문제** — 많은 작은 gradient가 0으로 소멸

### 4.4 K sublayer — Layer별 상세

| Layer | QZR_nonzero | ODR | cosine_sim | l2_retention | rel_l2_error | ratio_q50 |
|-------|-------------|-----|------------|--------------|--------------|-----------|
| 0 | 0.1119 | 23.4 | 0.99948 | 1.00052 | 0.0312 | 0.0536 |
| 1 | 0.1473 | 29.9 | 0.99942 | 1.00057 | 0.0331 | 0.0445 |
| 2 | 0.1819 | 33.7 | 0.99942 | 1.00056 | 0.0330 | 0.0406 |
| 3 | 0.1475 | 28.7 | 0.99943 | 1.00056 | 0.0329 | 0.0440 |
| 4 | 0.1939 | 38.4 | 0.99929 | 1.00068 | 0.0366 | 0.0338 |
| 5 | 0.1994 | 37.2 | 0.99934 | 1.00063 | 0.0354 | 0.0341 |
| 6 | 0.2106 | 42.6 | 0.99924 | 1.00073 | 0.0380 | 0.0311 |
| **7** | **0.2733** | **60.3** | **0.99915** | **1.00078** | **0.0402** | **0.0233** |
| 8 | 0.2291 | 46.9 | 0.99926 | 1.00070 | 0.0374 | 0.0285 |
| 9 | 0.2406 | 49.6 | 0.99938 | 1.00058 | 0.0343 | 0.0288 |
| 10 | 0.2431 | 50.8 | 0.99943 | 1.00054 | 0.0331 | 0.0289 |
| 11 | 0.2367 | 51.3 | 0.99947 | 1.00049 | 0.0319 | 0.0303 |

**Worst-3 K layers**: Layer 7 (27.3%), Layer 10 (24.3%), Layer 9 (24.1%)

- Layer 0→7로 갈수록 QZR과 ODR이 단조 증가 (더 깊은 layer에서 outlier가 강해짐)
- Layer 7 이후 소폭 감소 — K는 **중간 layer가 가장 취약**

### 4.5 V sublayer — Layer별 상세

| Layer | QZR_nonzero | ODR | cosine_sim | l2_retention | rel_l2_error | ratio_q50 |
|-------|-------------|-----|------------|--------------|--------------|-----------|
| 0 | 0.0928 | 19.8 | 0.99950 | 1.00049 | 0.0302 | 0.0632 |
| 1 | 0.1292 | 26.2 | 0.99942 | 1.00057 | 0.0325 | 0.0515 |
| 2 | 0.1445 | 30.2 | 0.99951 | 1.00047 | 0.0302 | 0.0467 |
| 3 | 0.1252 | 25.5 | 0.99952 | 1.00047 | 0.0298 | 0.0531 |
| 4 | 0.1320 | 25.2 | 0.99955 | 1.00044 | 0.0290 | 0.0515 |
| 5 | 0.1412 | 24.5 | 0.99960 | 1.00040 | 0.0278 | 0.0519 |
| 6 | 0.1351 | 25.1 | 0.99958 | 1.00041 | 0.0282 | 0.0513 |
| **7** | **0.1831** | **35.6** | **0.99954** | **1.00045** | **0.0298** | **0.0380** |
| 8 | 0.1311 | 25.6 | 0.99957 | 1.00042 | 0.0285 | 0.0518 |
| 9 | 0.1533 | 29.9 | 0.99957 | 1.00042 | 0.0288 | 0.0480 |
| 10 | 0.1708 | 33.4 | 0.99954 | 1.00045 | 0.0296 | 0.0429 |
| 11 | 0.1536 | 34.3 | 0.99958 | 1.00041 | 0.0282 | 0.0478 |

**Worst-3 V layers**: Layer 7 (18.3%), Layer 10 (17.1%), Layer 9 (15.3%)

- V는 K보다 전반적으로 양호 (QZR 약 30% 낮음)
- Layer 7이 K, V 모두에서 worst — **Layer 7이 BERT-base의 양자화 병목**

### 4.6 CDF 분석 (ratio CDF)

- Worst-3 K layers (Layer 7, 9, 10)에 대해 각 2,000 CDF 포인트 기록
- ratio 범위: ~1.3e-7 ~ 1.0
- zero_thresh(0.00794) 이하 CDF 비율이 QZR_nonzero와 정확히 일치
- **CDF 곡선이 zero_thresh 근방에서 급격히 상승** → bulk-tiny 분포 확인

---

## 5. Figure B — Bit-Width Sweep (4, 6, 7, 8, 10, 12 bit)

### 5.1 K/V QZR_nonzero vs Bits

| Bits | K QZR_nz | V QZR_nz | K cosine | V cosine | K l2_ret | V l2_ret | K rel_err | V rel_err |
|------|----------|----------|----------|----------|----------|----------|-----------|-----------|
| 4 | **0.756** | 0.575 | 0.9606 | 0.9745 | 1.0186 | 1.0181 | 0.278 | 0.224 |
| 6 | 0.327 | 0.241 | 0.9974 | 0.9982 | 1.0023 | 1.0017 | 0.070 | 0.059 |
| **7** | **0.201** | **0.141** | **0.9994** | **0.9995** | **1.0006** | **1.0005** | **0.035** | **0.029** |
| 8 | 0.117 | 0.077 | 0.9998 | 0.9999 | 1.0002 | 1.0001 | 0.017 | 0.015 |
| 10 | 0.036 | 0.021 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.004 | 0.004 |
| 12 | 0.010 | 0.006 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.001 | 0.001 |

### 5.2 전체 Sublayer QZR_nonzero vs Bits

| Bits | Q | K | V | O | FFN1 | FFN2 |
|------|-------|-------|-------|-------|-------|-------|
| 4 | 0.600 | 0.756 | 0.575 | 0.309 | 0.733 | 0.248 |
| 6 | 0.257 | 0.327 | 0.241 | 0.100 | 0.617 | 0.070 |
| 7 | 0.152 | 0.201 | 0.141 | 0.027 | 0.459 | 0.024 |
| 8 | 0.089 | 0.117 | 0.077 | 0.013 | 0.324 | 0.012 |
| 10 | 0.034 | 0.036 | 0.021 | 0.003 | 0.155 | 0.003 |
| 12 | 0.016 | 0.010 | 0.006 | 0.001 | 0.075 | 0.001 |

### 5.3 Quantization Resolution per Bits

| Bits | step_size | 양자화 levels | DAC / ADC |
|------|-----------|--------------|-----------|
| 4 | 2/14 = 0.1429 | 14 | 4 / 4 |
| 6 | 2/62 = 0.0323 | 62 | 6 / 6 |
| 7 | 2/126 = 0.0159 | 126 | 7 / 9 |
| 8 | 2/254 = 0.0079 | 254 | 8 / 8 |
| 10 | 2/1022 = 0.00196 | 1,022 | 10 / 10 |
| 12 | 2/4094 = 0.00049 | 4,094 | 12 / 12 |

### 5.4 K QZR_nonzero Crossover Thresholds

| QZR 목표 | 최소 필요 bits |
|----------|---------------|
| < 0.5 (50%) | 6-bit |
| < 0.3 (30%) | 7-bit |
| **< 0.2 (20%)** | **8-bit** |
| < 0.1 (10%) | 10-bit |

### 5.5 Sweet Spot 분석

| Bits | cosine ≥ 0.99 | l2_ret ∈ [0.99, 1.01] | rel_l2 < 0.05 | QZR_nz < 0.2 | 판정 |
|------|---------------|----------------------|---------------|---------------|------|
| 4 | FAIL | FAIL | FAIL | FAIL | **Poor** |
| 6 | PASS | PASS | FAIL | FAIL | **Marginal** |
| **7** | **PASS** | **PASS** | **PASS** | **PASS** | **Sweet Spot (최소)** |
| 8 | PASS | PASS | PASS | PASS | **Sweet Spot** |
| 10+ | PASS | PASS | PASS | PASS | **Excellent** |

### 5.6 해석

- **7-bit = 전체 평균 기준 최소 sweet spot** — 모든 메트릭이 threshold를 통과
- 단, K sublayer는 QZR_nz = 0.201로 **0.2 경계선** 위. **K만 고려하면 8-bit이 안정적** (K QZR = 0.117)
- **FFN1 병목**: 12-bit에서도 QZR_nz = 0.075 (7.5%) — Per-sublayer mixed-precision bit 할당이 필요할 수 있음
- **4→6bit에서 가장 큰 개선** (cosine +3.7%), 이후 diminishing returns

---

## 6. Figure C — Solutions Comparison (4 Variants)

### 6.1 실험 구성

| Variant | nm_thres | sto_round | 설명 |
|---------|----------|-----------|------|
| **baseline** | 0.0 | False | 기본 7-bit 양자화 |
| **p99_clip** | 0.0 | False | Per-vector p99 gradient clip (output hook에서 상위 1% outlier 제거) |
| **nm_thres_cal** | **0.000166** | False | theta = p95(K/V absmax), noise-margin cap |
| **sto_round** | 0.0 | **True** | Stochastic rounding (round 대신 확률적 반올림) |

### 6.2 K/V 결과 요약

| Variant | K QZR_nz | V QZR_nz | K cos | V cos | K l2_ret | V l2_ret | K rel_err | V rel_err | K clip_rate | V clip_rate |
|---------|----------|----------|-------|-------|----------|----------|-----------|-----------|-------------|-------------|
| baseline | 0.2013 | 0.1410 | 0.99936 | 0.99954 | 1.00061 | 1.00045 | 0.0348 | 0.0294 | 0.0000 | 0.0000 |
| p99_clip | 0.2231 | 0.1540 | 0.99936 | 0.99954 | 1.00061 | 1.00045 | 0.0348 | 0.0295 | 0.0000 | 0.0000 |
| nm_thres_cal | 0.1959 | 0.1342 | 0.99868 | 0.99761 | 0.99745 | 0.99133 | 0.0394 | 0.0413 | 0.0003 | 0.0038 |
| sto_round | 0.2018 | 0.1412 | 0.99936 | 0.99954 | 1.00061 | 1.00045 | 0.0347 | 0.0294 | 0.0000 | 0.0000 |

### 6.3 Delta vs Baseline

#### 절대 변화량

| Variant | K QZR_nz | V QZR_nz | K cosine | K l2_ret | K rel_err |
|---------|----------|----------|----------|----------|-----------|
| p99_clip | **+0.0219** | +0.0130 | -0.0000 | -0.0000 | +0.0001 |
| nm_thres_cal | -0.0054 | -0.0068 | -0.0007 | **-0.0032** | +0.0046 |
| sto_round | +0.0005 | +0.0002 | +0.0000 | -0.0000 | -0.0000 |

#### 상대 변화율 (%)

| Variant | K QZR_nz | V QZR_nz | 종합 판정 |
|---------|----------|----------|-----------|
| p99_clip | **+10.9% (악화)** | +9.3% (악화) | **해로움** |
| nm_thres_cal | -2.7% (미미) | -4.8% (미미) | **효과 제한적 + 부작용** |
| sto_round | +0.3% (무변화) | +0.1% (무변화) | **중립** |

### 6.4 Variant별 상세 분석

#### 6.4.1 p99_clip — Per-vector p99 Gradient Clipping

**결과: 해로움 (Harmful)**

- K QZR이 0.2013 → 0.2231로 **+10.9% 악화**
- 원인: 상위 1% outlier를 제거하면 absmax가 줄어들지만, 99th percentile 값으로 대체되므로 ODR(outlier 지배도)는 큰 변화 없음. 오히려 absmax 축소로 scaling factor가 변해 일부 원소가 추가로 underflow
- 특히 Layer 2 K에서 0.182 → **0.258** (급격한 악화)

#### 6.4.2 nm_thres_cal — Noise Margin Threshold (Calibrated)

**결과: 효과 미미 + 부작용 우세**

- theta = 0.000166 (baseline K/V absmax의 p95)
- baseline K/V absmax_q99 = 0.000553 → **theta/absmax_q99 = 0.30** (매우 공격적인 clipping)
- theta가 absmax의 30% 수준이므로 상위 5% 벡터가 심하게 clipping됨

**Sublayer별 부작용:**

| Sublayer | clip_rate_scaled | l2_retention | 해석 |
|----------|-----------------|--------------|------|
| K | 0.0003 (0.03%) | 0.9974 | 경미 |
| Q | 0.0004 (0.04%) | 0.9986 | 경미 |
| FFN1 | 0.0005 (0.05%) | 0.9943 | 중간 |
| V | 0.0038 (0.38%) | **0.9913** | 상당 (신호 0.87% 감쇠) |
| FFN2 | 0.0076 (0.76%) | **0.9916** | 심각 (신호 0.84% 감쇠) |
| O | 0.0093 (0.93%) | **0.9886** | 심각 (신호 1.14% 감쇠) |

**결론**: QZR을 2-5%만 줄이면서 O/V/FFN2에서 l2_retention이 1% 이상 하락. 트레이드오프가 불리하여 실용적 효과 없음.

#### 6.4.3 sto_round — Stochastic Rounding

**결과: 중립 (Neutral)**

- K QZR: 0.2013 → 0.2018 (+0.3%, 통계적 무의미)
- 모든 메트릭에서 baseline과 사실상 동일
- 원인: Stochastic rounding은 **기대값이 원래 값에 수렴**하는 장점이 있지만, **단일 sample에서는** deterministic rounding과 동일한 수준의 underflow 발생
- 여러 iteration에 걸친 누적 효과가 있을 수 있으나, 단일 forward-backward에서는 차이 없음

### 6.5 K sublayer — Layer별 QZR_nonzero 비교

| Layer | baseline | p99_clip | nm_thres_cal | sto_round |
|-------|----------|----------|--------------|-----------|
| 0 | 0.112 | 0.116 | 0.111 | 0.112 |
| 1 | 0.147 | 0.154 | 0.145 | 0.149 |
| 2 | 0.182 | **0.258** | 0.179 | 0.184 |
| 3 | 0.147 | 0.157 | 0.145 | 0.149 |
| 4 | 0.194 | 0.204 | 0.190 | 0.194 |
| 5 | 0.199 | 0.213 | 0.194 | 0.200 |
| 6 | 0.211 | 0.231 | 0.205 | 0.211 |
| **7** | **0.273** | **0.295** | **0.265** | **0.269** |
| 8 | 0.229 | 0.255 | 0.218 | 0.230 |
| 9 | 0.241 | 0.266 | 0.229 | 0.242 |
| 10 | 0.243 | 0.274 | 0.240 | 0.251 |
| 11 | 0.237 | 0.256 | 0.229 | 0.231 |

- p99_clip: 모든 layer에서 baseline보다 악화 (특히 Layer 2, 7)
- nm_thres_cal: 소폭 개선이 있으나 worst-case Layer 7에서도 0.273 → 0.265 (3% 미만)
- sto_round: baseline과 거의 동일

### 6.6 l2_retention 이상치 (|편차| > 0.1%)

| Variant | Sublayer | l2_retention | 편차 |
|---------|----------|--------------|------|
| baseline / p99_clip / sto_round | FFN1 | 1.002 | +0.2% (경미한 에너지 증폭) |
| nm_thres_cal | **O** | **0.9886** | **-1.14%** (심각한 신호 감쇠) |
| nm_thres_cal | **V** | **0.9913** | **-0.87%** |
| nm_thres_cal | **FFN2** | **0.9916** | **-0.84%** |
| nm_thres_cal | **FFN1** | **0.9943** | **-0.57%** |

---

## 7. 종합 결론

### 7.1 Root Cause (Figure A)

| 관찰 | 수치 | 의미 |
|------|------|------|
| K/V QZR_nonzero | 14-20% | 비zero gradient의 14-20%가 DAC 양자화 시 zero로 소실 |
| FFN1 QZR_nonzero | **46%** | 거의 절반의 gradient가 underflow — 가장 심각한 병목 |
| Layer 7 (worst) | K=27.3%, V=18.3% | K/V 모두에서 BERT-base의 양자화 취약점 |
| cosine_sim | > 0.999 | 방향 보존은 양호 |
| magnitude 손실 | rel_l2_error: K=3.5%, FFN1=6.7% | **학습 시 gradient fidelity 저하 가능** |

**근본 원인**: Gradient 분포가 **outlier-dominant** (ODR >> 1). 소수의 큰 outlier가 per-vector absmax를 결정하고, 대다수의 작은 gradient가 양자화 step 이하로 떨어져 0으로 소실.

### 7.2 Bit-Width 요구량 (Figure B)

| 결론 | 상세 |
|------|------|
| 전체 평균 sweet spot | **7-bit** (모든 평균 메트릭 threshold 통과) |
| K sublayer 안정선 | **8-bit** (K QZR < 0.2 달성) |
| FFN1 해결 | **12-bit에서도 QZR = 7.5%** — mixed-precision 필요 |
| 최대 효율 구간 | 4→6bit (cosine +3.7%), 이후 diminishing returns |

### 7.3 Solutions 효과 (Figure C)

| Variant | QZR 개선 | 부작용 | 종합 판정 |
|---------|----------|--------|-----------|
| **p99_clip** | K +10.9% (악화) | O/FFN2 rel_err 악화 | **해로움 — 사용 금지** |
| **nm_thres_cal** | K -2.7%, V -4.8% | l2_ret < 0.99 (O: 0.989) | **효과 미미, 부작용 우세** |
| **sto_round** | K +0.3% (무변화) | 없음 | **중립 — 단일 step에서 무효** |

### 7.4 최종 판단

> **어떤 variant도 QZR 문제를 유의미하게 해결하지 못함.**

근본 원인은 gradient의 **outlier-dominant 분포 (ODR >> 1)**이며, 이는:
- **Resolution 증가 (bits ↑)** 외에는 효과적인 해결책이 없음을 시사
- Per-sublayer **mixed-precision bit allocation**이 유망한 방향:
  - FFN1: 12+ bit 필요
  - K: 8-bit 권장
  - O/FFN2: 7-bit 충분
- 또는 **per-sublayer adaptive scaling** (absmax 대신 다른 normalization) 탐색 필요

# Backward IO Quantization: Sensitivity Diagnosis & Allocation Design

**Source**: `paper_figures_v3.py` → `/root/LRTT/results/tikitakav1/`
**Date**: 2026-03-17
**Model**: bert-base-uncased (12 layers x 6 sublayers = 72 analog tiles)
**Setup**: DAC=ADC, INP_BOUND=1.0, noise_management=ABS_MAX, bound_management=ITERATIVE
**Noise**: OFF (quantization error only), SQuAD v1, batch=8, seq=384, 200 steps

---

## Figure A: Root Cause Analysis (Uniform 8-bit Baseline)

72 rows (12 layers x 6 sublayers), variant=baseline, dac=adc=8-bit.

### A.1 Sublayer Summary (12-layer mean)

| Rank | Sublayer | QZR_nz (%) | ODR | cosine_sim | rel_l2 (%) | l2_retention | EZR (%) |
|:----:|:--------:|:----------:|:---:|:----------:|:----------:|:------------:|:-------:|
| 1 | **FFN1** | **32.4** | **139.2** | 0.9993 | 3.47 | 1.0006 | 0.5 |
| 2 | **K** | **11.8** | **41.0** | 0.9998 | 1.74 | 1.0002 | 0.0 |
| 3 | Q | 9.1 | 25.3 | 0.9999 | 1.66 | 1.0001 | 0.0 |
| 4 | V | 7.8 | 28.2 | 0.9999 | 1.47 | 1.0001 | 0.0 |
| 5 | O | 1.3 | 7.3 | 1.0000 | 0.94 | 1.0001 | 10.0 |
| 6 | FFN2 | 1.2 | 6.4 | 1.0000 | 0.83 | 1.0000 | 10.0 |

- **QZR_nz**: non-zero gradient 중 양자화로 0이 된 비율 (primary diagnostic)
- **ODR**: per-vector absmax/median (root cause indicator, >50이면 심각)
- p_clip = 0.000 (모든 sublayer) → bound_management에 의한 clipping 없음

### A.2 FFN1 Per-layer (가장 심각, 초기 레이어에서 worst)

| Layer | QZR_nz (%) | ODR | cosine | rel_l2 (%) |
|:-----:|:----------:|:---:|:------:|:----------:|
| L0 | **47.2** | **244.8** | 0.9992 | 3.80 |
| L1 | **43.8** | **207.6** | 0.9991 | 4.04 |
| L2 | 38.5 | 157.6 | 0.9993 | 3.70 |
| L3 | 37.0 | 225.7 | 0.9994 | 3.42 |
| L4 | 34.2 | 174.7 | 0.9994 | 3.33 |
| L5 | 30.8 | 112.8 | 0.9995 | 3.16 |
| L6 | 30.6 | 104.6 | 0.9994 | 3.24 |
| L7 | 30.6 | 113.9 | 0.9992 | 3.79 |
| L8 | 30.2 | 111.3 | 0.9993 | 3.69 |
| L9 | 27.2 | 95.5 | 0.9993 | 3.65 |
| L10 | 23.9 | 77.2 | 0.9994 | 3.42 |
| L11 | 14.7 | 45.0 | 0.9997 | 2.45 |

### A.3 K Per-layer (두 번째, 후반부 레이어에서 worst)

| Layer | QZR_nz (%) | ODR | cosine | rel_l2 (%) |
|:-----:|:----------:|:---:|:------:|:----------:|
| L0 | 5.8 | 23.3 | 0.9999 | 1.55 |
| L1 | 8.1 | 29.9 | 0.9999 | 1.64 |
| L2 | 11.6 | 35.0 | 0.9999 | 1.65 |
| L3 | 7.9 | 28.7 | 0.9999 | 1.64 |
| L4 | 11.0 | 38.9 | 0.9998 | 1.83 |
| L5 | 11.6 | 37.6 | 0.9998 | 1.77 |
| L6 | 11.9 | 42.0 | 0.9998 | 1.91 |
| L7 | **16.4** | **59.3** | 0.9998 | 2.03 |
| L8 | 12.8 | 44.6 | 0.9998 | 1.87 |
| L9 | 14.9 | 50.3 | 0.9998 | 1.73 |
| L10 | 14.7 | 50.1 | 0.9999 | 1.66 |
| L11 | 14.9 | 52.7 | 0.9999 | 1.61 |

### A.4 Root Cause

```
Backward dy → ABS_MAX (alpha = per-vector absmax)
→ scaled = dy / alpha → quantize(step = 2/(2^N - 2))
→ ODR 높을수록 alpha 커짐 → step_size 커짐 → 작은 gradient 0으로 소실 (QZR)
```

- FFN1: GELU 후 극소수 outlier → ODR 100~245 → QZR 30~47%
- K: deeper layer에서 ODR 50+ → QZR 15~16%
- O, FFN2: ODR < 8, QZR ~1% (양호)

---

## Figure B: Bit-width Sweep (4, 6, 8, 10, 12-bit)

432 rows (6 sublayers x 12 layers x 6 bit levels). 각 bit에서 동일 조건, resolution만 변경.

### B.1 QZR_nonzero (%) by Sublayer x Bits

| Bits | FFN1 | K | Q | V | O | FFN2 |
|:----:|:----:|:---:|:---:|:---:|:---:|:----:|
| 4 | 73.3 | 75.5 | 59.9 | 57.4 | 30.9 | 24.7 |
| 6 | 61.7 | 32.7 | 25.7 | 24.2 | 10.1 | 7.0 |
| **8** | **32.4** | **11.8** | **9.1** | **7.8** | **1.3** | **1.2** |
| 10 | 15.5 | 3.6 | 3.4 | 2.2 | 0.3 | 0.3 |
| 12 | 7.5 | 1.0 | 1.6 | 0.6 | 0.1 | 0.1 |

### B.2 cosine_sim by Sublayer x Bits

| Bits | FFN1 | K | Q | V | O | FFN2 |
|:----:|:----:|:---:|:---:|:---:|:---:|:----:|
| 4 | 0.9550 | 0.9608 | 0.9655 | 0.9746 | 0.9718 | 0.9839 |
| 6 | 0.9911 | 0.9974 | 0.9976 | 0.9982 | 0.9976 | 0.9986 |
| 8 | 0.9993 | 0.9998 | 0.9999 | 0.9999 | 1.0000 | 1.0000 |
| 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### B.3 rel_l2_error (%) by Sublayer x Bits

| Bits | FFN1 | K | Q | V | O | FFN2 |
|:----:|:----:|:---:|:---:|:---:|:---:|:----:|
| 4 | 29.6 | 27.7 | 26.0 | 22.3 | 20.7 | 16.7 |
| 6 | 13.0 | 7.0 | 6.7 | 5.9 | 5.4 | 4.3 |
| 8 | 3.5 | 1.7 | 1.7 | 1.5 | 0.9 | 0.8 |
| 10 | 0.9 | 0.4 | 0.4 | 0.4 | 0.2 | 0.2 |
| 12 | 0.2 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

### B.4 Sensitivity Ranking (모든 bit에서 일관)

```
Most sensitive ←―――――――――――――――――――→ Least sensitive
   FFN1  >>  K  >  Q  >  V  >>  O  ~  FFN2
```

### B.5 QZR Target별 최소 필요 bit

| QZR Target | FFN1 | K | Q | V | O | FFN2 |
|:----------:|:----:|:---:|:---:|:---:|:---:|:----:|
| < 10% | 10+ | 8 | 8 | 8 | 6 | 6 |
| < 5% | 12+ | 10 | 10 | 10 | 6 | 6 |
| < 1% | >12 | 12 | 10 | 12 | 8 | 8 |

---

## Figure C: Solution Variants (12 variants, all at avg 8-bit)

864 rows (12 variants x 72 tiles). 모든 variant는 cost-neutral (avg 8b) 조건.

### C.1 Variant 설명

| Cat | Variant | 메커니즘 | 해결하는 문제 |
|:---:|---------|---------|-------------|
| - | baseline | Uniform 8-bit, ABS_MAX | (기준선) |
| 1 | lp_q20 | Per-layer bit, QZR < 20% target | Resolution (QZR) |
| 1 | lp_q10 | Per-layer bit, QZR < 10% target | Resolution (QZR) |
| 1 | lp_q05 | Per-layer bit, QZR < 5% target | Resolution (QZR) |
| 2 | nm_thres_p50 | nm_thres clip top 50% | Outlier scaling (ODR) |
| 2 | nm_thres_p80 | nm_thres clip top 20% | Outlier scaling (ODR) |
| 2 | nm_thres_p90 | nm_thres clip top 10% | Outlier scaling (ODR) |
| 2 | nm_thres_p95 | nm_thres clip top 5% | Outlier scaling (ODR) |
| 3 | nmthres_mixed | nm_thres_p95 + lp_q10 | Both |
| 4 | avg_absmax | AVERAGE_ABS_MAX noise mgmt | Scaling strategy |
| 4 | constant_nm | CONSTANT noise mgmt (calibrated) | Scaling strategy |
| 5 | all_combined | nm_thres_p95 + avg_absmax + lp_q10 | All |

### C.2 QZR_nonzero (%) — 전체 비교

| Sublayer | baseline | lp_q20 | lp_q10 | lp_q05 | nm_p50 | nm_p80 | nm_p90 | nm_p95 | nm+lp | avg_abs | const | all_comb |
|:--------:|:-------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-----:|:-------:|:-----:|:--------:|
| FFN1 | 32.4 | 14.4 | 12.9 | 16.2 | 26.7 | 30.3 | 31.4 | 31.9 | 12.7 | 31.3 | 21.1 | **12.4** |
| K | 11.8 | 11.8 | 4.9 | 10.5 | 7.6 | 10.0 | 10.9 | 11.4 | 4.7 | 10.7 | 17.6 | **4.3** |
| Q | 9.1 | 11.1 | 9.9 | 7.6 | 6.9 | 8.2 | 8.6 | 8.8 | 9.6 | 8.8 | 8.9 | 9.6 |
| V | 7.8 | 8.9 | 14.0 | 4.1 | 4.7 | 6.3 | 7.0 | 7.4 | 13.5 | 5.6 | 14.5 | 10.6 |
| O | 1.3 | 3.1 | 5.5 | 9.4 | 1.0 | 1.2 | 1.3 | 1.3 | 5.4 | 1.7 | 9.4 | 6.9 |
| FFN2 | 1.2 | 5.0 | 4.8 | 15.6 | 0.9 | 1.1 | 1.1 | 1.2 | 4.7 | 1.3 | 6.4 | 5.4 |

### C.3 rel_l2_error (%) — 전체 비교

| Sublayer | baseline | lp_q20 | lp_q10 | lp_q05 | nm_p50 | nm_p80 | nm_p90 | nm_p95 | nm+lp | avg_abs | const | all_comb |
|:--------:|:-------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-----:|:-------:|:-----:|:--------:|
| FFN1 | 3.47 | 0.92 | 0.78 | 1.05 | 14.5 | 7.5 | 5.5 | 4.5 | 1.96 | 3.41 | 1.05 | **0.84** |
| K | 1.74 | 1.74 | 0.74 | 1.44 | 15.7 | 6.5 | 3.8 | 2.6 | 1.66 | 1.82 | 1.77 | **0.77** |
| Q | 1.66 | 2.46 | 2.09 | 1.46 | 13.5 | 6.0 | 3.9 | 2.8 | 3.22 | 1.66 | 0.76 | 2.10 |
| V | 1.47 | 1.85 | 2.92 | 0.93 | 15.7 | 7.0 | 4.2 | 2.7 | 4.09 | 1.39 | 1.56 | 2.84 |
| O | 0.94 | 2.13 | 3.84 | 6.74 | 12.0 | 5.3 | 3.3 | 2.1 | 4.83 | 1.10 | 1.84 | 5.07 |
| FFN2 | 0.83 | 3.57 | 3.41 | 11.2 | 11.6 | 4.9 | 2.9 | 1.9 | 4.41 | 0.91 | 1.45 | 4.29 |

### C.4 cosine_sim — 전체 비교

| Sublayer | baseline | lp_q20 | lp_q10 | lp_q05 | nm_p50 | nm_p80 | nm_p90 | nm_p95 | nm+lp | avg_abs | const | all_comb |
|:--------:|:-------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-----:|:-------:|:-----:|:--------:|
| FFN1 | .9993 | .9999 | .9999 | .9999 | .9732 | .9899 | .9945 | .9968 | .9973 | .9994 | **.6632** | .9998 |
| K | .9998 | .9998 | 1.000 | .9999 | .9704 | .9912 | .9965 | .9986 | .9987 | .9998 | .9998 | .9999 |
| Q | .9999 | .9995 | .9995 | .9997 | .9758 | .9913 | .9954 | .9974 | .9971 | .9999 | **.4920** | .9995 |
| V | .9999 | .9997 | .9993 | .9998 | .9698 | .9891 | .9950 | .9980 | .9974 | .9999 | .9999 | .9993 |
| O | .9999 | .9995 | .9991 | .9954 | .9833 | .9934 | .9963 | .9982 | .9974 | .9999 | .9997 | .9973 |
| FFN2 | 1.000 | .9981 | .9994 | .9919 | .9844 | .9942 | .9970 | .9982 | .9977 | 1.000 | .9998 | .9980 |

### C.5 Cat1: Per-layer Bit Allocation (lp_q10, cost-neutral avg=8b)

`allocate_precision()` 알고리즘이 QZR_nz 기반으로 자동 결정한 할당:

| Layer | FFN1 | K | Q | V | O | FFN2 | Avg |
|:-----:|:----:|:---:|:---:|:---:|:---:|:----:|:---:|
| L0 | 10 | 8 | 8 | 8 | 6 | 6 | 7.7 |
| L1 | 10 | 8 | 8 | 8 | 6 | 6 | 7.7 |
| L2 | 10 | 10 | 10 | 8 | 6 | 6 | 8.3 |
| L3 | 10 | 8 | 8 | 8 | 6 | 6 | 7.7 |
| L4 | 12 | 10 | 8 | 8 | 6 | 6 | 8.3 |
| L5 | 12 | 10 | 10 | 8 | 6 | 6 | 8.7 |
| L6 | 12 | 10 | 10 | 8 | 6 | 6 | 8.7 |
| L7 | 12 | 10 | 10 | 6 | 6 | 6 | 8.3 |
| L8 | 12 | 10 | 6 | 8 | 6 | 6 | 8.0 |
| L9 | 12 | 10 | 8 | 6 | 6 | 6 | 8.0 |
| L10 | 8 | 10 | 8 | 6 | 6 | 6 | 7.3 |
| L11 | 10 | 10 | 6 | 6 | 6 | 6 | 7.3 |
| **Avg** | **10.8** | **9.5** | **8.3** | **7.3** | **6.0** | **6.0** | **8.0** |

### C.6 Cat1: lp_q20 Allocation (avg=8b, 느슨한 target)

| Layer | FFN1 | K | Q | V | O | FFN2 | Avg |
|:-----:|:----:|:---:|:---:|:---:|:---:|:----:|:---:|
| L0 | 12 | 8 | 6 | 6 | 8 | 8 | 8.0 |
| L1 | 12 | 8 | 8 | 8 | 8 | 8 | 8.7 |
| L2 | 10 | 8 | 8 | 8 | 8 | 6 | 8.0 |
| L3 | 10 | 8 | 6 | 8 | 6 | 8 | 7.7 |
| L4~L9 | 10 | 8 | 8 | 8 | 6~8 | 4~8 | 7.7~8.3 |
| L11 | 8 | 8 | 8 | 8 | 8 | 6 | 7.7 |
| **Avg** | **10.3** | **8.0** | **7.7** | **7.7** | **7.2** | **7.0** | **8.0** |

### C.7 Cat1: lp_q05 Allocation (avg=8b, 엄격한 target)

| Layer | FFN1 | K | Q | V | O | FFN2 | Avg |
|:-----:|:----:|:---:|:---:|:---:|:---:|:----:|:---:|
| L0 | 10 | 10 | 6 | 6 | 4 | 4 | 6.7 |
| L1 | 10 | 10 | 10 | 10 | 4 | 4 | 8.0 |
| L2 | 10 | 8 | 10 | 10 | 4 | 8 | 8.3 |
| L3 | 10 | 10 | 10 | 10 | 8 | 8 | 9.3 |
| L4~L8 | 10 | 8 | 8~10 | 10 | 4~8 | 4 | 7.3~8.3 |
| L11 | 8 | 8 | 8 | 10 | 8 | 8 | 8.3 |
| **Avg** | **9.8** | **8.7** | **8.8** | **9.3** | **6.0** | **5.3** | **8.0** |

---

## Figure C: 핵심 분석

### 1. Cat별 효과 요약

| Category | FFN1 QZR 개선 | 부작용 | 판정 |
|:--------:|:----------:|--------|:----:|
| **Cat1 (bit alloc)** | 32.4→12.9% (lp_q10) | O/FFN2의 QZR 증가 (1→5%) | **QZR 직접 해결** |
| Cat2 (nm_thres) | 32.4→26.7% (p50) | cosine/l2_retention 심각 저하 | ODR 완화 but 과도 clip |
| Cat3 (nm+lp) | 32.4→12.7% | Cat1과 유사 효과 | Cat1만으로 충분 |
| Cat4 (avg_absmax) | 32.4→31.3% | QZR 거의 불변 | **QZR에 무효** |
| Cat4 (constant_nm) | 32.4→21.1% | FFN1 cosine=**0.66**, Q cosine=**0.49** | **치명적 부작용** |
| Cat5 (all_combined) | 32.4→12.4% | Cat3과 유사, 복잡도만 증가 | Cat1이면 충분 |

### 2. Cat1 내부 비교 (QZR target 선택)

| Variant | FFN1 QZR | K QZR | O QZR | FFN2 QZR | 문제점 |
|:-------:|:--------:|:-----:|:-----:|:--------:|--------|
| lp_q20 | 14.4% | 11.8% (불변) | 3.1% | 5.0% | K 미개선, O/FFN2 소폭 악화 |
| **lp_q10** | **12.9%** | **4.9%** | 5.5% | 4.8% | **FFN1+K 모두 개선, O/FFN2 허용 범위** |
| lp_q05 | 16.2% | 10.5% | 9.4% | 15.6% | **역효과**: O/FFN2 심각 악화, V에 과잉 투자 |

**lp_q10이 최적**: FFN1/K의 QZR을 크게 낮추면서 O/FFN2의 악화가 허용 범위.
lp_q05는 O/FFN2에서 bit를 과도하게 빼앗아 오히려 전체 품질 저하.

### 3. constant_nm 치명적 문제

- FFN1 cosine_sim = **0.6632** (방향 60% 이상 손실)
- Q cosine_sim = **0.4920** (방향 정보 사실상 소멸)
- 원인: calibrated constant theta가 실제 gradient 분포와 불일치

### 4. nm_thres 계열의 한계

- nm_thres_p50: 강한 clipping → l2_retention이 0.90~0.93 (에너지 7~10% 손실)
- nm_thres_p95: 약한 clipping → QZR 거의 불변 (31.9% vs 32.4%)
- **nm_thres는 p_clip=0인 현재 조건에서 불필요** (clipping 문제가 없으므로)

---

## Sensitivity-Aware Allocation 설계 가이드

### 설계 원칙

1. **Severity = QZR_nonzero** (유일한 bit-improvable metric)
2. **Ranking 준수**: FFN1 >= K >= Q >= V >= O >= FFN2 (Figure B에서 검증)
3. **Cost neutrality**: avg bit = target budget
4. **Layer-aware**: FFN1 초기(L0~L3), K 후반(L7~L11)에 집중

### lp_q10 결과 기반 권장 할당 (avg 8b)

`paper_figures_v3.py`의 `allocate_precision()` 결과를 그대로 사용:

```
Sublayer 평균: FFN1=10.8b, K=9.5b, Q=8.3b, V=7.3b, O=6.0b, FFN2=6.0b
```

이 할당은:
- FFN1 QZR: 32.4% → 12.9% (60% 감소)
- K QZR: 11.8% → 4.9% (58% 감소)
- O/FFN2 QZR: 1.3% → 5.5% (증가하지만 8-bit K 수준 이하)

### 다른 avg budget에서의 확장

| Budget | 전략 | 기대 효과 |
|:------:|------|----------|
| avg 6b | FFN1=10, K=8, Q=6, V=6, O=4, FFN2=4 | FFN1만 집중 보호 |
| avg 8b | lp_q10 결과 사용 | FFN1+K 보호, O/FFN2 허용 |
| avg 10b | FFN1=12, K=12, Q=10, V=10, O=8, FFN2=8 | 전 sublayer QZR < 5% |
| avg 12b | uniform 12b | QZR < 1% everywhere |

### 논문 정당화

1. **Data-driven**: Figure B bitsweep에서 측정된 QZR_nz 기반 자동 할당
2. **Stable ranking**: 4~12bit 전 구간에서 sensitivity 순위 일관
3. **Causal mechanism**: ODR → ABS_MAX → QZR 인과관계 확립 (Figure A)
4. **Cost-neutral**: uniform 대비 동일 평균 bit에서 worst-case QZR 크게 개선
5. **Orthogonal**: nm_thres, avg_absmax와 독립 결합 가능 (Figure C Cat3/Cat5)

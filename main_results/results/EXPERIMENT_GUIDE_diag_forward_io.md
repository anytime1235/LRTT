# diag_forward_io 실험 가이드

> **생성일**: 2026-03-11
> **목적**: 모든 diag_forward_io 실험의 셋업 조합을 한눈에 비교하고, 각 실험의 목적·결과·파일 위치를 빠르게 파악

---

## 1. 공통 설정 (모든 실험 동일)

| 항목 | 값 |
|------|---|
| 모델 | BERT-base (12 encoder layers × 6 sublayers = 72 모듈) |
| 서브레이어 | Q, K, V, O (Attention) + FFN1, FFN2 (Feed-Forward) |
| DAC | 7-bit (고정) |
| dw_min | 0.001 |
| inp_bound | 1.0 |
| out_bound | 12.0 (기본값, calib 시 모듈별 재설정) |
| 학습 스텝 | 200 steps |
| 배치 크기 | 8 |
| Optimizer | AnalogSGD (lr=0.002) |
| Seed | 42 |
| Framework | PyTorch 2.3.1+cu121, Transformers 4.47.1, AIHWKit 1.0.0 |

---

## 2. 실험 전체 매트릭스

### 2.1 차이 파라미터 요약

| # | 실험 태그 | 데이터셋 | ADC Sweep | OB-Calib | Mixed Prec. | Depth Boost | Sto.Round | 디렉토리 |
|---|----------|---------|-----------|----------|------------|-------------|-----------|---------|
| 1 | `baseline_sst2` | SST-2 (GLUE) | 4,6,8,10,12 | - | - | - | - | `diag_fwd_io_glue/` |
| 2 | `obcal_sst2` | SST-2 (GLUE) | 4 (일부) | **per_module** | - | - | - | `diag_fwd_io_glue/` |
| 3 | `baseline` | SQuAD | 4,6,8,10,12 | - | - | - | - | `diag_fwd_io_mitigations/` |
| 4 | `obcal_per_module` | SQuAD | 4,6,8,10,12 | **per_module** | - | - | - | `diag_fwd_io_mitigations/` |
| 5 | `mp_base6` | SQuAD | 6 | - | **base=6, FFN1+2, V+1** | - | - | `diag_fwd_io_mitigations/` |
| 6 | `mp_base6_depth` | SQuAD | 6 | - | **base=6, FFN1+2, V+1** | **layers 9-11: +1** | - | `diag_fwd_io_mitigations/` |
| 7 | `mp_base6_obcal` | SQuAD | 6 | **per_module** | **base=6, FFN1+2, V+1** | - | - | `diag_fwd_io_mitigations/` |
| 8 | `baseline_sr` | SQuAD | 4,6 | - | - | - | **ON** | `diag_fwd_io_mitigations/` |
| 9 | `obcal_per_module_sr` | SQuAD | 4,6 | **per_module** | - | - | **ON** | `diag_fwd_io_mitigations/` |
| 10 | `analysis_adc468` | SQuAD | (분석) | - | - | - | - | `diag_fwd_io_mitigations/` |

### 2.2 파라미터 상세 설명

| 파라미터 | 설명 |
|---------|------|
| **OB-Calib (Output-Bound Calibration)** | `calib_out_bound=true`, 32 배치로 모듈별 출력 범위를 측정하여 ADC의 `out_bound`를 자동 조정. quantile=0.999, margin=1.05 |
| **Mixed Precision** | `mixed_precision=true`, 기본 ADC=6-bit 위에 취약 서브레이어에 추가 비트 할당: FFN1 +2bit(=8), V +1bit(=7). 나머지 Q/K/O/FFN2는 6-bit 유지 |
| **Depth Boost** | `depth_boost="9-11:+1"`, 심층 레이어(9,10,11)의 모든 서브레이어에 ADC +1bit 추가 할당 |
| **Stochastic Rounding** | `sto_round=true`, ADC 양자화 시 확률적 반올림 적용. 양자화 바이어스 감소 효과 |

---

## 3. 실험별 목적 및 설계 의도

### 그룹 A: 베이스라인 (ADC Sweep)

#### #1 `baseline_sst2` — GLUE 분류 태스크 베이스라인
- **목적**: SST-2 감정분류에서 ADC 비트별 MAC 품질 + 로짓 품질 동시 측정
- **특이점**: `logit_eval_batches=10`으로 logit_kl, flip_rate 등 분류 성능 영향도 직접 측정
- **ADC sweep**: 4, 6, 8, 10, 12-bit 전구간

#### #2 `obcal_sst2` — GLUE + Output-Bound Calibration
- **목적**: GLUE 태스크에서 OB-Calib의 효과 검증
- **비교 대상**: #1 `baseline_sst2` (동일 조건에서 calib 유무 비교)
- **주의**: ADC 4-bit만 실행됨 (sweep이 아닌 단일 조건)

#### #3 `baseline` — SQuAD 베이스라인
- **목적**: SQuAD QA 태스크에서 ADC 비트별 순수 MAC 노이즈 특성 측정
- **비교 대상**: #1과의 태스크 간 차이 분석 (SST-2 vs SQuAD)
- **ADC sweep**: 4, 6, 8, 10, 12-bit 전구간

### 그룹 B: Calibration 계열

#### #4 `obcal_per_module` — Per-Module Output-Bound Calibration
- **목적**: 모듈별 ADC 범위 자동 보정의 SNR 개선 효과를 전 ADC 범위에서 검증
- **비교 대상**: #3 `baseline` (동일 ADC sweep에서 calib 유무 비교)
- **추가 산출물**: `calib_table.csv` (72개 모듈별 보정된 out_bound 값)

### 그룹 C: Mixed Precision 계열

#### #5 `mp_base6` — Mixed Precision 기본
- **목적**: 취약 서브레이어(FFN1, V)에 추가 비트를 차등 할당하는 전략 검증
- **설정**: base=6-bit, FFN1→8-bit(+2), V→7-bit(+1)
- **비교 대상**: #3 `baseline` ADC-6 조건과 비교
- **평균 ADC bits**: 전체 약 6.5-bit (추가 비트의 하드웨어 비용 대비 효용 평가)

#### #6 `mp_base6_depth` — Mixed Precision + Depth Boosting
- **목적**: #5에 레이어 깊이 보정 추가. 후반 레이어(9-11)에 일괄 +1-bit
- **설정**: #5 + layers 9,10,11 전체 서브레이어 ADC +1-bit
- **비교 대상**: #5 `mp_base6` (depth boost 유무 차이만)
- **평균 ADC bits**: 약 7.3-bit

#### #7 `mp_base6_obcal` — Mixed Precision + Output-Bound Calibration
- **목적**: 두 가지 독립적 완화 전략(정밀도 차등 + 범위 보정)의 결합 효과 검증
- **설정**: #5 mixed precision + #4 OB-Calib 동시 적용
- **비교 대상**: #5 (mixed precision만), #4 (calib만), #3 (baseline)

### 그룹 D: Stochastic Rounding 계열

#### #8 `baseline_sr` — Stochastic Rounding
- **목적**: 양자화 반올림을 확률적으로 처리할 때의 SNR 변화 측정
- **ADC sweep**: 4, 6-bit (저해상도 구간에서의 효과에 집중)
- **비교 대상**: #3 `baseline` ADC-4,6 조건과 비교

#### #9 `obcal_per_module_sr` — OB-Calib + Stochastic Rounding
- **목적**: 두 가지 독립적 완화 전략(범위 보정 + 확률적 반올림)의 결합 효과
- **ADC sweep**: 4, 6-bit
- **비교 대상**: #4 (calib만), #8 (SR만), #3 (baseline)

### 그룹 E: 교차 분석

#### #10 `analysis_adc468` — 2차 집계 분석
- **목적**: #3 baseline의 ADC 4,6,8 결과를 정리하여 SNR gain 테이블 및 비교 플롯 생성
- **산출물**: `analysis_summary_table.csv`, `analysis_snr_gain.csv`, 비교 플롯
- **참고**: 새 실험이 아니라 기존 데이터의 재분석

---

## 4. 주요 결과 비교표

### 4.1 ADC 6-bit 기준 전체 평균 MAC SNR (dB) 비교

| 실험 | 전체 평균 SNR | Q | K | V | O | FFN1 | FFN2 |
|------|-------------|---|---|---|---|------|------|
| **baseline** (#3) | 14.84 | 15.90 | 15.84 | 12.73 | 19.96 | 10.33 | 14.30 |
| **obcal_per_module** (#4) | 16.30 | 19.19 | 18.93 | 14.78 | 6.74 | 14.21 | 23.95 |
| **mp_base6** (#5) | 17.92 | 16.66 | 16.65 | 18.90 | 20.19 | 20.69 | 14.42 |
| **mp_base6_depth** (#6) | 19.11 | 18.06 | 18.05 | 19.92 | 21.51 | 21.18 | 15.91 |
| **mp_base6_obcal** (#7) | 17.51 | 19.34 | 19.02 | 15.43 | 6.75 | 20.54 | 23.97 |
| **baseline_sr** (#8) | 12.37 | 13.28 | 13.26 | 10.27 | 17.60 | 7.68 | 12.13 |
| **obcal_per_module_sr** (#9) | 14.54 | 17.04 | 17.26 | 13.72 | 6.66 | 11.36 | 21.17 |

### 4.2 ADC 4-bit 기준 전체 평균 MAC SNR (dB) 비교

| 실험 | 전체 평균 SNR | FFN1 | 비고 |
|------|-------------|------|------|
| **baseline** (#3) | 5.08 | 1.15 | FFN1 사실상 noise-dominant |
| **obcal_per_module** (#4) | 8.48 | 2.45 | OB-Calib으로 +3.4 dB 개선 |
| **baseline_sr** (#8) | 2.07 | -2.60 | SR이 저비트에서 오히려 악화 |
| **obcal_per_module_sr** (#9) | 6.41 | -1.92 | Calib이 SR 악화 일부 보상 |
| **baseline_sst2** (#1) | 4.89 | 0.93 | SST-2에서도 유사 패턴 |
| **obcal_sst2** (#2) | 8.06 | 1.00 | GLUE에서도 Calib 효과 확인 |

### 4.3 핵심 관찰 요약

| 전략 | 효과 | 주의사항 |
|------|------|---------|
| **OB-Calib** | 전반적 SNR +2~3 dB 개선 | **O projection SNR 급락** (32→6 dB): 희소 입력에 out_bound 축소가 역효과 |
| **Mixed Precision** | 취약 모듈(FFN1, V) 집중 개선. 전체 +3 dB | 평균 ADC 비용 증가 (~0.5 bit) |
| **Depth Boost** | 심층 레이어 +1~2 dB 추가 개선 | 추가 하드웨어 비용 (평균 +0.8 bit) |
| **MP + OB-Calib** | FFN2에서 최대 SNR (23.97 dB). 부분적 시너지 | O projection 문제 여전히 존재 |
| **Stochastic Rounding** | 저비트(4-bit)에서 **악화**. 6-bit에서 미미한 차이 | 현재 설정에서는 비권장 |
| **SR + OB-Calib** | Calib이 SR 악화를 일부 보상 | baseline 대비 여전히 낮은 성능 |

---

## 5. 파일 구조

### 5.1 디렉토리 레이아웃

```
results/
├── diag_fwd_io_glue/                     # GLUE 태스크 (SST-2)
│   ├── baseline_sst2/                    # #1 베이스라인
│   │   ├── baseline_sst2_meta.json
│   │   ├── baseline_sst2_sweep_summary.csv
│   │   ├── baseline_sst2_adc{N}_summary_row.csv
│   │   ├── baseline_sst2_adc{N}_layer_mac_metrics.csv
│   │   ├── baseline_sst2_adc{N}_module_mac_summary.csv
│   │   ├── baseline_sst2_adc{N}_logit_eval.csv     ← GLUE 전용
│   │   └── baseline_sst2_adc{N}_heatmap_*.png
│   └── obcal_sst2/                       # #2 OB-Calib
│       ├── obcal_sst2_meta.json
│       ├── obcal_sst2_calib_table.csv               ← 보정 테이블
│       ├── obcal_sst2_adc4_*.csv
│       └── obcal_sst2_adc4_records.npz
│
├── diag_fwd_io_mitigations/              # SQuAD 태스크 + Mitigation 실험
│   ├── baseline/                         # #3 베이스라인
│   ├── obcal_per_module/                 # #4 OB-Calib (calib_table.csv 포함)
│   ├── mp_base6/                         # #5 Mixed Precision
│   ├── mp_base6_depth/                   # #6 MP + Depth Boost
│   ├── mp_base6_obcal/                   # #7 MP + OB-Calib
│   ├── baseline_sr/                      # #8 Stochastic Rounding
│   ├── obcal_per_module_sr/              # #9 OB-Calib + SR
│   └── analysis_adc468/                  # #10 교차 분석 집계
│
├── csv/diag_fwd_io/                      # 초기 단일 런 분석 (15개 CSV)
│   ├── summary_adc_sweep.csv
│   ├── single_run_*.csv
│   └── adc{N}_*.csv
│
└── reports/diag_fwd_io/
    └── ANALYSIS_REPORT.md                # csv/diag_fwd_io/ 분석 보고서
```

### 5.2 실험별 주요 파일

| 파일 패턴 | 설명 | 행 수 |
|----------|------|------|
| `{tag}_meta.json` | 전체 실험 설정 (재현 가능) | - |
| `{tag}_sweep_summary.csv` | ADC sweep 전체 요약 (서브레이어별 SNR, clip ratio) | ADC 조건 수 |
| `{tag}_adc{N}_summary_row.csv` | 특정 ADC 조건 1행 요약 | 1행 |
| `{tag}_adc{N}_module_mac_summary.csv` | 72개 모듈별 MAC 메트릭 평균 | 72행 |
| `{tag}_adc{N}_layer_mac_metrics.csv` | 스텝×모듈 세부 메트릭 | 200×72 = 14,400행 |
| `{tag}_adc{N}_logit_eval.csv` | 로짓 품질 (GLUE만) | 스텝 수 |
| `{tag}_calib_table.csv` | OB-Calib 실험만: 모듈별 보정 out_bound | 72행 |
| `{tag}_adc{N}_records.npz` | 전체 MAC 데이터 바이너리 아카이브 | - |

---

## 6. 실험 간 비교 관계도

```
                    baseline (#3)
                   /      |       \
                  /       |        \
           obcal (#4)  baseline_sr (#8)   mp_base6 (#5)
              |            |               /        \
              |      obcal_sr (#9)   mp_depth (#6)  mp_obcal (#7)
              |            |
              +------------+
              (calib vs SR 비교)


           baseline_sst2 (#1)
                  |
           obcal_sst2 (#2)
           (GLUE 태스크 교차 검증)
```

- **세로**: 같은 기반에서 전략 추가
- **가로**: 독립적 전략 비교
- **#10 analysis_adc468**: #3의 ADC 4,6,8 부분을 정리한 2차 분석

---

## 7. 메트릭 해석 가이드

| 메트릭 | 우수 | 양호 | 주의 | 위험 |
|--------|------|------|------|------|
| MAC SNR (dB) | > 28 | 20~28 | 10~20 | < 10 |
| Cosine Similarity | > 0.998 | 0.99~0.998 | 0.95~0.99 | < 0.95 |
| NMSE | < 0.002 | 0.002~0.01 | 0.01~0.05 | > 0.05 |
| Out Clip Ratio | 0 | < 0.001 | 0.001~0.01 | > 0.01 |
| Ref Deadzone Ratio | < 0.03 | 0.03~0.1 | 0.1~0.5 | > 0.5 |
| Logit Flip Rate | < 0.05 | 0.05~0.2 | 0.2~0.5 | > 0.5 |

---

## 8. 관련 스크립트

| 스크립트 | 용도 |
|---------|------|
| `scripts/forward/diag_forward_io_glue.py` | GLUE 실험 실행 (#1, #2) |
| `scripts/forward/diag_forward_io_single_rpu.py` | SQuAD 실험 실행 (#3~#9) |
| `scripts/forward/diag_fwdio_utils.py` | 공통 유틸리티 (MAC 측정, 저장) |
| `scripts/shell/run_glue_forward_io.sh` | GLUE 파이프라인 일괄 실행 |
| `scripts/shell/run_mitigation_plan.sh` | Mitigation 파이프라인 일괄 실행 |

---

*이 문서는 기존 `reports/diag_fwd_io/ANALYSIS_REPORT.md` (단일 베이스라인 심층 분석)을 보완하며, 전체 실험 셋업 비교에 초점을 맞춥니다.*

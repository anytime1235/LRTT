# Forward Pass 분석

## 개요

Forward pass에서 **analog MAC (Multiply-Accumulate) 연산의 ADC 양자화**, 출력 클리핑,
weight read noise가 BERT inference에 미치는 영향을 진단한다.
Digital reference 대비 analog output의 SNR, 클리핑 비율, logit 발산을
layer별/sublayer별로 측정한다.

## 핵심 지표 (Metrics)

| 지표 | 정의 | 의미 |
|------|------|------|
| mac_snr_db | 10·log₁₀(var(y_ref) / var(y_ref − y_ana)) | Analog MAC 신호대잡음비 |
| mac_nmse | mean((y_ref − y_ana)²) / (mean(y_ref²) + ε) | 정규화 MSE |
| cosine | cosine_similarity(y_ref, y_ana) row-wise mean | 출력 방향 보존 |
| out_clip_ratio | mean(\|y_ana\| > out_bound·(1−out_res)) | ADC 출력 클리핑 비율 |
| ref_deadzone_ratio | mean(\|y_ref\| < out_res·out_bound) | ADC deadzone에 빠지는 비율 |
| mean_abs_err | \|y_ref − y_ana\| 평균 | 절대 오차 평균 |
| p95_abs_err | \|y_ref − y_ana\| 95th percentile | 절대 오차 꼬리 |
| logit KL | KL(softmax(analog) ‖ softmax(ideal)) | 분류 확률 왜곡 |
| flip_rate | argmax 불일치 비율 | 예측 변경 빈도 |
| margin | top1 − top2 logit gap | 결정 여유도 |
| dw_zero_ratio | weight update = 0인 비율 | Weight update 양자화 영향 |
| dw_1lsb_ratio | update = ±1 LSB인 비율 | dw_min 해상도 한계 |

## 스크립트 목록

### Production 스크립트 (3개)

#### 1. `diag_forward_io_single_rpu.py` (~1,800 lines)
SQuAD forward 진단 메인. MAC fidelity + logit divergence + weight delta 3축 진단.
Training loop 포함. ADC sweep / dw_min sweep 지원.

- **입력**: SQuAD v1.1 train, `bert-base-uncased` QA
- **출력**: `{out_dir}/{tag}/` 하위 `*_layer_mac_metrics.csv`, `*_module_mac_summary.csv`, `*_logit_metrics.csv`, `*_weight_delta_metrics.csv`, heatmap/sweep PNG
- **CLI**: `--adc-bits-sweep 4,6,8,10,12 --dw-min-sweep ... --calib-out-bound --mixed-precision --tag ...`

#### 2. `diag_forward_io_glue.py` (~900 lines)
GLUE forward 진단. Inference-only (no backward). ADC sweep + seed sweep.
Classification(KL/flip) & regression(Pearson/Spearman drift) 별도 처리.

- **입력**: GLUE 9 tasks, `bert-base-uncased` classification
- **출력**: `{out_dir}/{tag}/` 하위 CSV + `{tag}_sweep_summary.csv`, `{tag}_seed_sweep_summary.csv`, PNG
- **CLI**: `--glue-task sst2 --adc-bits-sweep 4,6,8,10,12 --seed-sweep 42,43,44 --calib-out-bound`

#### 3. `diag_fwdio_utils.py` (~800 lines)
공용 유틸리티 라이브러리. 직접 실행하지 않음.

- 주요 API: `ForwardMACStats`, `register_forward_hooks`, `create_rpu_config`, `calibrate_out_bounds`, `compute_mixed_precision_assignment`
- 상수: `OUT_BOUND=12.0`, `INP_BOUND=1.0`, `N_LAYERS=12`, `SUBLAYER_ORDER=[Q,K,V,O,FFN1,FFN2]`

### Diagnostic / Test 스크립트 (3개)

#### 4. `analyze_adc_results.py`
기존 ADC sweep CSV를 읽어 SNR gain table, sublayer bottleneck, monotonicity check 등 2차 분석.

- **입력**: `results/diag_fwd_io_mitigations/baseline/` 하위 CSV
- **출력**: `sweep_summary.csv`, `analysis_*.csv`, 비교 plot PNG

#### 5. `test_forward_compare.py`
ALBERT analog 변환 전후 forward 일관성 검증.
Start/end logit correlation > 0.99이면 PASS.

- **출력**: stdout only

#### 6. `diag_forward_compare.py`
MobileBERT MRPC 3-way 비교 (TikiTaka+Digital / TikiTaka+SingleRPU / All-Digital).
Analog 열화 원인 분리 디버깅용.

- **출력**: stdout only

## 결과 디렉토리

| 경로 | 내용 | 생성 스크립트 |
|------|------|--------------|
| `results/diag_fwd_io_mitigations/baseline/` | SQuAD ADC sweep 기본 결과 | `diag_forward_io_single_rpu.py` |
| `results/diag_fwd_io_mitigations/analysis_adc468/` | ADC 2차 분석 결과 | `analyze_adc_results.py` |
| `results/diag_fwd_io_glue/baseline_sst2/` | GLUE SST-2 baseline | `diag_forward_io_glue.py` |
| `results/diag_fwd_io_glue/obcal_sst2/` | GLUE SST-2 output-bound calibrated | `diag_forward_io_glue.py` |

> **참고**: `results/` 경로는 `/data/main_results/results/` 기준 상대 경로이다.

## 실행 예시

```bash
# SQuAD ADC sweep (4~12 bit)
python diag_forward_io_single_rpu.py --adc-bits-sweep 4,6,8,10,12 --calib-out-bound --tag baseline

# GLUE SST-2 ADC sweep + seed sweep
python diag_forward_io_glue.py --glue-task sst2 --adc-bits-sweep 4,6,8,10,12 --seed-sweep 42,43,44 --calib-out-bound --tag obcal_sst2

# 기존 결과 2차 분석
python analyze_adc_results.py
```

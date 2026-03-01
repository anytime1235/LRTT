# Weight Update 분석

## 개요

Analog weight update (SGD/TikiTaka) 과정에서 **dw_min 양자화**, burst-length 포화,
A→B tile transfer 효율을 진단한다. BERT-base의 72개 analog layer에서 실제 weight delta를
gradient proxy와 비교하여 update fidelity를 측정한다.

## 핵심 지표 (Metrics)

### 공통 지표

| 지표 | 정의 | 의미 |
|------|------|------|
| dw_zero_ratio | weight update = 0인 비율 | 양자화 사각지대 |
| dw_1lsb_ratio | update = ±1 LSB(dw_min)인 비율 | 해상도 한계 |
| dw_absmean / dw_absmax | update 크기 통계 | update 스케일 |
| grad_absmean | gradient proxy g_ij = d·x 크기 | 기대 update 스케일 |
| grad_deadzone_ratio | \|lr × grad\| < dw_min인 비율 | 이론적 사각지대 |
| update_vs_grad_cosine | actual Δw vs FP gradient cosine | update 방향 충실도 |
| eff_lr_slope | Δw_eff ≈ slope × Δw_fp 회귀 기울기 | 유효 학습률 |
| sign_mismatch_ratio | Δw_eff와 Δw_fp 부호 불일치 비율 | 역방향 update 빈도 |
| rel_update_error | ‖Δw_eff − slope·Δw_fp‖ / ‖slope·Δw_fp‖ | 상대 update 오차 |
| BL_mean / BL_hit_ratio | 평균 burst length / 목표 BL 도달 비율 | pulse 정밀도 |
| pulse_ok_frac / pulse_under_frac / pulse_over_frac | 3-zone pulse 분류 | pulse 품질 |
| pulse_sat_ratio | pulse ≥ 0.9·BL·dw_min 비율 | 포화 비율 |
| bound_sat_ratio | \|w\| ≥ 0.98·w_max 비율 | weight 포화 |

### TikiTaka 전용 지표

| 지표 | 정의 | 의미 |
|------|------|------|
| dw_fast_\* / dw_slow_\* | A-tile(fast) / B-tile(slow) delta 통계 | tile별 update |
| transfer_efficiency | slow_absmean / fast_absmean | A→B 전달 효율 |
| transfer_duty | slow tile update ≠ 0 비율 | transfer 활성도 |
| hidden_trunc_nonzero_ratio | buffer 절단 후 nonzero 비율 | buffer 활용률 |
| buffer_cleared_ratio | step 후 buffer 0이 된 비율 | buffer 소진률 (v2) |
| cols/rows_updated_ratio | transfer에서 update된 열/행 비율 | coverage (v2) |

## 스크립트 목록

### Production 스크립트 (4개)

#### 1. `diag_weight_update_bert.py` (~1,800 lines)
BERT/SQuAD analog weight update 진단 메인. SingleRPU & TikiTaka 모드.
Per-step per-layer 전체 지표 수집. Forward/backward hook으로 gradient proxy 계산.

- **입력**: SQuAD v1.1 train, `bert-base-uncased`
- **출력**: `{output_dir}/{mode}/run_{sha1}/{tag}/{tag}_step_metrics.csv`, `{tag}_summary.csv`, `config.json`
- **CLI**: `--mode single|tiki --dw-min 0.001 --dw-min-sweep "0.0005,0.001,0.005" --seeds 0,1,2 --sample-k 512 --steps 100`

#### 2. `diag_weight_update_bert_v2.py` (~2,000 lines)
v1 확장. Transfer 관찰 기반 감지, column/row coverage, buffer pre/post 스냅샷,
per_column 샘플링 추가.

- **추가 CLI**: `--sample-mode per_column --buffer-granularity 0.001 --sweep-transfer-diagnosis`

#### 3. `deep_analysis_weight_update.py` (~1,400 lines)
7-stage 사후 분석. 기존 CSV를 읽어 dose-response(Cohen's d), bottleneck heatmap,
transfer pipeline loss, temporal dynamics, correlation matrix, mode comparison,
recommendations 생성.

- **입력**: `results/weight_update/squad/tiki/run_{hash}/` 하위 summary/step CSV × 6 + `results/diagnosis/diag_weight_update/` 하위 mode comparison CSV
- **출력**: `results/diagnosis/figures/fig_{dose_response,bottleneck_heatmap,transfer_pipeline,temporal_dynamics,correlation_matrix}.png`

#### 4. `analyze_seed_variance.py` (~450 lines)
3-seed 충분성 분석. CV, var_ratio(between-dw_min / within-seed),
seed-pair Spearman rho 계산. 최종 판정: `SINGLE SEED SUFFICIENT` / `THREE SEEDS RECOMMENDED`.

- **입력**: `results/weight_update/squad/tiki/run_{hash}/` 하위 6개 summary CSV
- **출력**: stdout only

### Early-stage Monitor (2개)

#### 5. `gradient_monitor_qkv.py` (~600 lines)
ALBERT/MRPC TikiTaka v1. Tile weight delta L2/max/mean 추적.
`UPDATING` vs `STATIC` 판정.

- **출력**: `/data/results/tikitakav1/gradient_monitor_qkv_results.json`

#### 6. `gradient_monitor_attn.py` (~650 lines)
ALBERT/MRPC LRTT LoRA. A/B/C tile별 epoch-wise delta 추적.
`ACTIVE` / `MODERATE` / `WEAK` / `DEAD` 판정.

- **출력**: `/data/results/gradient_monitor_attn/gradient_analysis.json`

## 결과 디렉토리

| 경로 | 내용 | 생성 스크립트 |
|------|------|--------------|
| `results/weight_update/squad/tiki/run_{hash}/` | TikiTaka per-seed per-dw_min CSV | `diag_weight_update_bert.py` |
| `results/weight_update/squad/single/run_{hash}/` | SingleRPU CSV | `diag_weight_update_bert.py` |
| `results/diagnosis/diag_weight_update/` | v3 모드 비교 CSV (single vs tiki) | `diag_weight_update_bert.py` |
| `results/diagnosis/figures/` | deep_analysis 7-stage figures | `deep_analysis_weight_update.py` |
| `results/diagnosis/weight_update_traces/` | TikiTaka trace run raw data | `diag_weight_update_bert.py` |

> **참고**: `results/` 경로는 `/data/main_results/results/` 기준 상대 경로이다.

## 실행 예시

```bash
# TikiTaka 모드, dw_min sweep, 3 seeds
python diag_weight_update_bert.py --mode tiki --dw-min-sweep "0.0005,0.001,0.005" --seeds 0,1,2 --steps 100

# SingleRPU 모드
python diag_weight_update_bert.py --mode single --dw-min 0.001 --steps 100

# v2: transfer 진단 포함
python diag_weight_update_bert_v2.py --mode tiki --sweep-transfer-diagnosis --sample-mode per_column

# 7-stage 사후 분석
python deep_analysis_weight_update.py

# Seed 충분성 분석
python analyze_seed_variance.py
```

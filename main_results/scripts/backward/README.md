# Backward Pass 분석

## 개요

Backward pass에서 **AbsMax DAC 양자화**로 인한 gradient underflow 문제를 진단한다.
BERT-base encoder의 **12 layer × 6 sublayer** (Q/K/V/O/FFN1/FFN2)에 대해
gradient가 DAC에 의해 0으로 양자화되는 비율(QZR), outlier 지배도(ODR),
양자화 후 방향 보존(cosine_sim) 등을 측정한다.

## 핵심 지표 (Metrics)

| 지표 | 정의 | 의미 |
|------|------|------|
| EZR (Exact Zero Ratio) | FP32 gradient에서 정확히 0인 비율 | 구조적 영점 (padding 등) |
| QZR_all | DAC 양자화 후 0이 되는 전체 비율 | 양자화 영향 (EZR 포함) |
| QZR_nonzero | 원래 nonzero였으나 DAC 후 0이 된 비율 | **핵심 underflow 지표** |
| ODR (Outlier Dominance Ratio) | per-vector absmax / median | 분포 꼬리 집중도 |
| cosine_sim | FP32 vs DAC-quantized gradient 코사인 유사도 | 방향 보존 |
| l2_retention | ‖dy_q‖ / ‖dy‖ | 에너지 보존 |
| rel_l2_error | ‖dy − dy_q‖ / ‖dy‖ | 상대 L2 오차 |
| clip_rate_scaled | \|dy\| > INP_BOUND인 비율 | 클리핑 손실 |
| CCR (Cap Clipping Rate) | P(absmax > nm_thres) | nm_thres 캡 영향 |
| ΔQZR | QZR_run1 − QZR_run2 | nm_thres 효과 (≈0이면 무효) |

## 스크립트 목록

### Production 스크립트 (7개)

#### 1. `paper_figures.py` (~2,000 lines)
SQuAD backward 진단 메인. Fig A(root cause) / B(bit sweep) / C(solutions) / D(layerwise mixed-precision). Multi-seed (42/43/44).

- **입력**: HuggingFace `squad` train, `bert-base-uncased` QA
- **출력**: `results/squad/seed_{42,43,44}/metrics_{A,B,C,D}_*.csv`, `fig_*.png`, `absmax_raw_*.npz`
- **CLI**: `--seeds 42,43,44 --figures ABCD --n-step 200 --run-tag v3`

#### 2. `paper_figures_glue.py` (~2,000 lines)
GLUE 8-task backward 진단. Fig A/B/D + cross-task Fig E(required bits heatmap) / F(seed variance).

- **입력**: HuggingFace `nyu-mll/glue/{task}` train, `bert-base-uncased` classification
- **출력**: `results/glue/{task}/seed_{seed}/metrics_*.csv`, `results/glue/aggregate/fig_E_*.png`, `fig_F_*.png`
- **CLI**: `--tasks cola,rte,mrpc,... --seeds 42,43,44 --figures ABDEF`

#### 3. `paper_figures_v3.py` (~1,600 lines)
SQuAD backward 진단 초기 버전. Fig A/B/C. Single seed.

- **출력**: `/data/results/tikitakav1/metrics_paper_{A,B,C}_*.csv`

#### 4. `diag_kv_rootcause.py` (~1,200 lines)
K/V underflow 근본원인 3-part 분석.
- Part A: 구조적 zero vs bulk-tiny 구분
- Part B: nm_thres Pareto sweep
- Part C: 대안 비교 (sto_round, dac8bit, p99_scale)

- **출력**: `/data/results/tikitakav1/metrics_rootcause.csv`, `fig_rootcause_diagnosis.pdf`

#### 5. `diag_backward_outlier.py` (~600 lines)
최초 PoC 진단. ODR/QZR 기본 측정. IdealDevice, Q/K/V/O만.

- **출력**: `/data/results/tikitakav1/metrics_backward_outlier.csv`, `fig_backward_outlier_diagnosis.pdf`

#### 6. `calib_nm_thres.py` (~600 lines)
1-run 분석적 nm_thres 캘리브레이션. 30-point theta grid에서 QZR_before/after 오프라인 계산.

- **출력**: `/data/results/tikitakav1/calib_*.csv`, `theta_{global,layerwise}.json`

#### 7. `calib_nm_thres_tworun.py` (~600 lines)
2-run nm_thres 실험. Run1(nm_thres=0) → theta 계산 → Run2(nm_thres=theta).
ΔQZR ≈ 0 확인 (K/V 문제는 nm_thres로 해결 불가).

- **출력**: `/data/results/tikitakav1/tworun_*.csv`, `fig_nm_thres_tworun.pdf`

### Diagnostic / Test 스크립트 (4개)

#### 8. `test_backward_outlier_diag.py`
ALBERT LRTT LoRA on STS-B. backward d_input 통계 캡처.
default vs `is_perfect=True` 비교 (30 steps).

#### 9. `test_backward_perfect.py`
MobileBERT LRTT on MRPC. `backward.is_perfect=True` smoke test (3 epochs).

#### 10. `test_backward_perfect_effectiveness.py`
ALBERT LRTT on STS-B. 5-diagnostic 비교: delta_A/B, LoRA contribution, Spearman r, grad_nonzero (5 epochs).

#### 11. `test_backward_perfect_sweep.py`
ALBERT LRTT on STS-B. lora_alpha × target_ab_lr × backward_perfect 24-experiment grid search.

- **출력**: `/data/probe/lora/sweep_*.json`, `sweep_*.png`

## 결과 디렉토리

| 경로 | 내용 | 생성 스크립트 |
|------|------|--------------|
| `results/squad/seed_{42,43,44}/` | SQuAD per-seed CSV/NPZ/PNG | `paper_figures.py` |
| `results/squad/aggregate/` | SQuAD cross-seed summary | `paper_figures.py` |
| `results/glue/{task}/seed_{seed}/` | GLUE per-task/seed CSV/NPZ/PNG | `paper_figures_glue.py` |
| `results/glue/aggregate/` | GLUE cross-task Fig E/F | `paper_figures_glue.py` |
| `/data/results/tikitakav1/` | 초기 진단 결과 (diag_\*, calib_\*) | `diag_backward_outlier.py`, `diag_kv_rootcause.py`, `calib_nm_thres*.py` |
| `/data/probe/lora/` | Perfect backward sweep | `test_backward_perfect_sweep.py` |

> **참고**: `results/` 경로는 `/data/main_results/results/` 기준 상대 경로이다.

## 실행 예시

```bash
# SQuAD 전체 Figure (A/B/C/D), 3 seeds
python paper_figures.py --seeds 42,43,44 --figures ABCD --n-step 200 --run-tag v3

# GLUE 전체 task, Fig A/B/D/E/F
python paper_figures_glue.py --tasks cola,sst2,mrpc,stsb,qqp,mnli,qnli,rte --seeds 42,43,44 --figures ABDEF

# nm_thres 캘리브레이션
python calib_nm_thres.py
python calib_nm_thres_tworun.py
```

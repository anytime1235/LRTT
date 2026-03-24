# MNIST 6T1C LRTT 재실험 계획

**Date**: 2026-03-24
**Branch**: MLP
**Goal**: Reset vs Decay 모드의 TE 운용구간 경향성을 rank별로 일관되게 입증

---

## 기존 실험의 한계

1. 각 (Rank, TE) cell이 독립적으로 HP 최적화 → TE 효과 vs HP 차이 분리 불가
2. Seed 반복 없음 → 통계적 유의성 주장 불가
3. Lifetime 혼재 (1000~100000)

---

## 실험 공통 설정

```yaml
network: MLP (784 → 256 → 10, Sigmoid, MNIST)
device: 6T1C (dw_min=0.001981, gamma_up=-0.1678, gamma_down=+0.1410)
lifetime: 0          # 사용하지 않음
epochs: 30
batch_size: 64
optimizer: AnalogSGD
scheduler: StepLR(step_size=10, gamma=0.5)
output_layer: IdealizedPreset
bias: True
activation: Sigmoid
loss: NLLLoss
lora_alpha: 1.0
forward_inject: False
reinit_gain: 0.1
```

### Grid

```
Ranks: [1, 4, 8, 16, 32, 64]
TEs:   [1, 10, 50, 100, 500, 1000]
Modes: [Reset, Decay]
Seeds: [42, 43, 44]
```

- Reset: `reinit_mode="standard"` (A=0, B=Kaiming)
- Decay: `reinit_mode="decay"` (A, B 유지)

---

## 3단계 실험 계획

### Step 1 — 고정 HP로 전체 Grid 실행

**목적**: HP를 완전히 동일하게 고정하여 순수 (Mode, Rank, TE) 효과만 측정

```python
lr  = 0.3
tlr = 0.005
# 모든 mode, rank, te 조합에서 동일
```

| 항목 | 값 |
|------|-----|
| 총 cell | 6 × 6 × 2 = 72 |
| Seed 반복 | 72 × 3 = **216 runs** |
| 예상 시간 | **~1 GPU-hour** |
| 출력 | `reexp_step1_results.json`, `.csv` |

```bash
python sweep_mnist_6t1c_reexp.py --step 1
```

### Step 2 — 경향성 분석

Step 1 결과를 분석하여 다음을 확인:

**확인 항목**:
1. Reset vs Decay의 TE에 따른 accuracy 경향이 rank별로 일관되는가?
2. Rank가 accuracy에 유의한 영향을 주는가? (ANOVA)
3. Crossover TE (Reset ≈ Decay) 구간은 어디인가?
4. tlr=0.005가 특정 (Mode, TE) 구간에서 과소/과대한가?

**판단 기준**:
- 6개 rank 중 5개 이상에서 동일한 TE 경향 → 주장 가능
- 특정 구간에서 accuracy가 급락 → tlr 부적합 → Step 3 필요
- rank별 편차 < 2%p → "Rank-invariant" 주장 가능

```bash
python analyze_step1.py --input reexp_step1_results.json
```

### Step 3 — transfer_lr Sweep (Step 2 결과에 따라 조건부 실행)

Step 2에서 tlr=0.005가 특정 구간에서 부적합한 경우에만 실행.

**경우 A**: Step 1 결과가 충분히 일관됨 → Step 3 불필요, 논문 작성 진행

**경우 B**: 특정 TE 구간에서 성능 급락 → 해당 구간만 tlr sweep

```
tlr_values: [0.001, 0.002, 0.005, 0.01, 0.02]
대상: Step 2에서 식별된 underperforming cells만
Seeds: [42, 43, 44]
```

| 항목 | 값 (최대) |
|------|----------|
| 대상 cell | ~24 (2 modes × 2 TEs × 6 ranks 가정) |
| tlr sweep | 24 × 5 tlr values × 3 seeds = **360 runs** |
| 예상 시간 | **~2 GPU-hours** |
| 출력 | `reexp_step3_tlr_sweep.json`, `.csv` |

```bash
python sweep_mnist_6t1c_reexp.py --step 3 \
    --tlr_values 0.001,0.002,0.005,0.01,0.02 \
    --target_cells step2_underperforming.json
```

Step 3 결과로 tlr의 TE/Rank 종속성을 정량화:
- `tlr ∝ 1/√TE` 인지, `tlr ∝ 1/√Rank` 인지, 또는 둘 다인지 확인
- 최적 tlr 공식을 도출하고 해당 공식으로 최종 재실험

---

## 실행 스크립트

**파일**: `experiments/sweep_mnist_6t1c_reexp.py`

### 모델 구조

```python
AnalogSequential(
    AnalogLinear(784, 256, rpu_config=LRTT_6T1C, bias=True),
    Sigmoid(),
    AnalogLinear(256, 10, rpu_config=IdealizedPreset, bias=True),
    LogSoftmax(dim=1),
)
```

### 실행 명령

```bash
cd /path/to/LRTT/experiments

# Step 1: 고정 HP 전체 grid (216 runs, ~1h)
python sweep_mnist_6t1c_reexp.py --step 1

# Step 2: 분석 (실행 없음, 결과 분석만)
python analyze_step1.py --input reexp_step1_results.json

# Step 3: tlr sweep (필요시만, ~2h)
python sweep_mnist_6t1c_reexp.py --step 3 \
    --tlr_values 0.001,0.002,0.005,0.01,0.02 \
    --target_cells step2_underperforming.json
```

### 출력 파일

| Step | 파일 | 내용 |
|------|------|------|
| 1 | `reexp_step1_results.json` | 72 cells × 3 seeds, mean ± std |
| 1 | `reexp_step1_results.csv` | 요약 테이블 |
| 2 | `step1_analysis_report.md` | 경향성 분석, 주장 가능 여부 판단 |
| 2 | `step2_underperforming.json` | Step 3 대상 cell 목록 (해당시) |
| 3 | `reexp_step3_tlr_sweep.json` | tlr sweep 결과 (해당시) |

---

## 논문 Figure 계획

### Main Figure: Rank × TE Heatmap (1×3)
- (a) Reset mode accuracy heatmap (mean)
- (b) Decay mode accuracy heatmap (mean)
- (c) Decay − Reset difference (with significance markers)

### Supplementary
- Per-Rank TE sweep curve — rank별 일관된 패턴 입증
- ANOVA table (Rank 효과 유의성)
- (Step 3 실행 시) tlr sensitivity 분석

---

## 참고: Lifetime 미사용 근거

6T1C capacitor의 물리적 retention (τ=775min)은 30-epoch MNIST 학습 시간 (~10min) 대비 충분히 길어 학습 중 실질적 decay가 무시 가능하다. lifetime=0으로 통일하여 변수를 제거한다.

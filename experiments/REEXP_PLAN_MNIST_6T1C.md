# MNIST 6T1C LRTT 재실험 계획

**Date**: 2026-03-24
**Branch**: MLP
**Goal**: Reset vs Decay 모드의 Rank × TE 운용구간을 논문에서 통계적으로 강하게 주장하기 위한 재실험

---

## 기존 실험 요약

- **Grid**: Rank ∈ {1, 4, 8, 16, 32, 64}, TE ∈ {1, 10, 50, 100, 500, 1000}
- **Mode**: Reset (A=0 hard reset, B unchanged) / Decay (A, B both decay)
- **Network**: MLP 784→256→10 (MNIST)
- **결과**: Reset best 96.45% (R4, TE100), Decay best 97.54% (R64, TE1)

### 핵심 발견

1. **TE가 accuracy의 지배적 변수** (Rank는 거의 무관)
2. **두 모드의 최적 TE가 정반대**: Reset → TE=50~100, Decay → TE=1~10
3. Crossover 지점 TE≈50~100 부근

### 기존 실험의 한계

1. 각 cell이 single best (seed 반복 없음) → 통계적 유의성 주장 불가
2. TE 해상도 부족 (10→50, 100→500 구간 gap)
3. Decay Rank=64, TE=100 붕괴 현상 미해명
4. Rank=2, 128 부재 → Rank 무관성 주장 약함

---

## 실험 설정 (공통)

```yaml
# 공통 설정
network: MLP (784 → 256 → 10, Sigmoid, MNIST)
device: 6T1C (dw_min=0.001981, gamma_up=-0.1678, gamma_down=+0.1410)
lifetime: 0  # 사용하지 않음 — 학습에 영향 없음
epochs: 30
batch_size: 64
optimizer: AnalogSGD
scheduler: StepLR(step_size=10, gamma=0.5)
seeds: [42, 43, 44]  # 3-seed 반복
output_layer: IdealizedPreset (FloatingPoint equivalent)
bias: True
activation: Sigmoid
loss: NLLLoss
```

### Hyperparameter Search (per cell)

각 (Rank, TE, Mode) 조합에 대해 Optuna로 best hyperparameter를 먼저 찾은 뒤, 해당 HP로 3-seed 반복.

```yaml
search_space:
  lr: [0.01, 1.0]         # log-uniform
  transfer_lr: [0.0001, 0.1]  # log-uniform
  lora_alpha: 1.0          # 고정
  forward_inject: False     # 고정
  reinit_gain: 0.1          # 고정
n_trials: 15               # Optuna trials per cell
```

---

## 재실험 우선순위

### Priority 1 — 핵심 Grid 3-seed 반복 (필수)

**목적**: 기존 6×6 grid의 모든 cell에 대해 mean ± std 확보

```
Ranks: [1, 4, 8, 16, 32, 64]
TEs:   [1, 10, 50, 100, 500, 1000]
Modes: [Reset, Decay]
Seeds: [42, 43, 44]
```

| 항목 | 값 |
|------|-----|
| 총 cell | 6 × 6 × 2 = 72 |
| Optuna search | 72 × 15 trials = 1,080 runs |
| Seed 반복 | 72 × 3 seeds = 216 runs |
| **총 runs** | **~1,296 runs** |

**이 실험 완료 시 주장 가능한 것**:
- "Decay TE≤10이 Reset 대비 통계적으로 유의하게 우수" (t-test p < 0.05)
- "Reset TE=50~100이 Reset 모드 내 최적" (mean ± std bar plot)
- "Rank는 1~64에서 accuracy에 유의한 영향 없음" (ANOVA)

---

### Priority 2 — TE 해상도 보강 (필수)

**목적**: Crossover point 정밀 특정 + 추세 곡선 완성

```
Ranks: [1, 4, 8, 16, 32, 64]
TEs:   [5, 20, 30, 200, 300]    # 기존 gap 보완
Modes: [Reset, Decay]
Seeds: [42, 43, 44]
```

| 추가 TE | 보완 구간 | 목적 |
|---------|----------|------|
| **5** | 1↔10 | Decay TE=1이 진짜 최적인지, TE=5도 동등한지 확인 |
| **20** | 10↔50 | Reset 최적 구간 진입점 특정 |
| **30** | 10↔50 | Reset 최적 구간 세밀화 |
| **200** | 100↔500 | **Crossover point** 정밀 특정 (가장 중요) |
| **300** | 100↔500 | Crossover 확증 |

| 항목 | 값 |
|------|-----|
| 총 cell | 6 × 5 × 2 = 60 |
| Optuna search | 60 × 15 = 900 runs |
| Seed 반복 | 60 × 3 = 180 runs |
| **총 runs** | **~1,080 runs** |

**이 실험 완료 시 추가 주장 가능한 것**:
- "Crossover point는 TE=X ± Y" (정밀 수치)
- TE에 대한 accuracy curve를 smooth하게 그릴 수 있음

---

### Priority 3 — Rank 확장 (권장)

**목적**: "Rank 무관" 주장 강화 + 극단값 확인

```
Ranks: [2, 128]
TEs:   [1, 10, 50, 100, 500, 1000]
Modes: [Reset, Decay]
Seeds: [42, 43, 44]
```

| 항목 | 값 |
|------|-----|
| 총 cell | 2 × 6 × 2 = 24 |
| Optuna search | 24 × 15 = 360 runs |
| Seed 반복 | 24 × 3 = 72 runs |
| **총 runs** | **~432 runs** |

**이 실험 완료 시 추가 주장 가능한 것**:
- "Rank=2에서도 97%+ 달성 가능" → 하드웨어 비용 최소화 주장
- "Rank=128은 Rank=64 대비 이점 없음" → upper bound 확인
- Rank=1~128 전 구간에서 ANOVA 무유의 → Rank 무관성 확정

---

### Priority 4 — Decay Rank=64 붕괴 분석 (권장)

**목적**: Decay 모드의 instability region 특정 및 논문 내 명시적 설명

```
Ranks: [32, 64]
TEs:   [60, 70, 80, 90, 100, 120, 150]  # 붕괴 구간 세밀 탐색
Mode:  Decay only
Seeds: [42, 43, 44, 45, 46]  # 5-seed (붕괴 확률 추정 위해)
```

기존 anomaly 데이터:
- Rank=64, TE=100: **전 lifetime에서 11.35% 붕괴** (5/5 runs)
- Rank=4, TE=500: 일부 run에서 47% 붕괴 (3/5 runs)

| 항목 | 값 |
|------|-----|
| 총 cell | 2 × 7 × 1 = 14 |
| Optuna search | 14 × 15 = 210 runs |
| Seed 반복 | 14 × 5 = 70 runs |
| **총 runs** | **~280 runs** |

**이 실험 완료 시 추가 주장 가능한 것**:
- "Decay 모드에서 high-rank + mid-TE 조합은 training collapse 위험이 있으며, TE < X 또는 TE > Y 에서 안정적"
- 붕괴 확률을 수치화하여 practical guideline 제시

---

## 전체 실험 규모 요약

| Priority | 실험 | Runs | 누적 |
|----------|------|------|------|
| **P1** | 핵심 Grid 3-seed | ~1,296 | 1,296 |
| **P2** | TE 해상도 보강 | ~1,080 | 2,376 |
| **P3** | Rank 확장 | ~432 | 2,808 |
| **P4** | 붕괴 분석 | ~280 | 3,088 |

- MNIST MLP 30 epoch → ~15~20 sec/run (GPU)
- **P1+P2 (필수)**: ~2,376 runs ≈ **10~13 GPU-hours**
- **전체**: ~3,088 runs ≈ **13~17 GPU-hours**

---

## 실험 완료 후 논문 Figure 계획

### Main Figure: Rank × TE Heatmap (1×3)
- (a) Reset mode accuracy heatmap (mean)
- (b) Decay mode accuracy heatmap (mean)
- (c) Decay − Reset difference (with significance markers)

### Main Figure: TE-Accuracy Curve
- x축: TE (log scale), y축: Accuracy (%)
- Reset / Decay 각각 mean ± std band
- Rank별 curve (or Rank 평균 + shading)
- Crossover point annotation

### Supplementary Figure
- Per-Rank TE sweep (6 panels)
- Collapse probability heatmap (P4 데이터)
- Rank별 ANOVA 결과 table

---

## 실행 순서

```bash
# Step 1: P1 — 핵심 Grid
python sweep_mnist_6t1c.py --ranks 1,4,8,16,32,64 --tes 1,10,50,100,500,1000 \
    --modes reset,decay --seeds 42,43,44 --lifetime 0 --n_trials 15

# Step 2: P2 — TE 보강
python sweep_mnist_6t1c.py --ranks 1,4,8,16,32,64 --tes 5,20,30,200,300 \
    --modes reset,decay --seeds 42,43,44 --lifetime 0 --n_trials 15

# Step 3: P3 — Rank 확장
python sweep_mnist_6t1c.py --ranks 2,128 --tes 1,10,50,100,500,1000 \
    --modes reset,decay --seeds 42,43,44 --lifetime 0 --n_trials 15

# Step 4: P4 — 붕괴 분석
python sweep_mnist_6t1c.py --ranks 32,64 --tes 60,70,80,90,100,120,150 \
    --modes decay --seeds 42,43,44,45,46 --lifetime 0 --n_trials 15
```

---

## 참고: Lifetime 미사용 근거

기존 실험에서 lifetime 값 (1000, 10000, 46505, 100000)이 혼재되어 있었으나, 6T1C capacitor의 물리적 retention (τ=775min)은 **30-epoch MNIST 학습 시간 (~10min) 대비 충분히 길어** 학습 중 실질적 decay가 무시 가능하다. 재실험에서는 lifetime=0 (retention 비활성화)으로 통일하여 변수를 제거한다.

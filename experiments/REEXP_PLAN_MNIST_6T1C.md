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
2. **HP confound**: 각 (Rank, TE) cell이 독립적으로 HP 최적화 → TE 효과인지 HP 차이인지 분리 불가
3. Decay Rank=64, TE=100 붕괴 현상 미해명

#### HP Confound 구체 사례

```
Decay Rank=16~64, TE=1  best HP: lr=0.089, tlr=0.001  → 96.88~97.54%
Decay Rank=4~8,   TE=100 best HP: lr=0.494, tlr=0.011  → 96.92~97.34%

→ lr 5.5배, tlr 8.8배 차이
→ "TE=1이 좋다"가 아니라 "이 HP가 좋다"일 수 있음
```

**TE에 따른 경향성을 주장하려면, 같은 Rank 내에서 HP를 통제한 상태로 TE만 변화시켜야 한다.**

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

---

## Hyperparameter Search 전략: 2-Stage Per-Rank Search

### 문제

기존 방식 (cell별 독립 HP search)은 TE 간 HP가 달라져서 TE 효과를 오염시킴.

### 해결: Rank별 HP 통제

각 (Rank, Mode) 조합에 대해 **하나의 공통 HP를 찾은 뒤**, 해당 HP로 모든 TE를 sweep.

```
Stage 1: Per-Rank HP Search
  - 각 (Rank, Mode) 조합에 대해 reference TE에서 Optuna search
  - Reference TE = 중간값 (TE=50 또는 100) 사용
  - 이 HP를 해당 Rank의 "base HP"로 확정

Stage 2: TE Sweep with Fixed HP
  - 확정된 base HP로 모든 TE에 대해 3-seed 반복
  - HP가 통제되었으므로 accuracy 차이 = 순수 TE 효과
```

#### Stage 1: Per-Rank HP Search

```yaml
search_space:
  lr: [0.01, 1.0]           # log-uniform
  transfer_lr: [0.0001, 0.1]  # log-uniform
  lora_alpha: 1.0            # 고정
  forward_inject: False       # 고정
  reinit_gain: 0.1            # 고정

reference_TE: 100             # 중간값에서 HP 탐색
n_trials: 20                  # Optuna trials per (Rank, Mode)
```

| 항목 | 값 |
|------|-----|
| 조합 수 | 6 Rank × 2 Mode = 12 |
| Optuna search | 12 × 20 = **240 runs** |

#### Stage 2: TE Sweep with Fixed HP + 3-Seed

```
per (Rank, Mode):
  HP = Stage 1에서 찾은 best HP
  for TE in [1, 10, 50, 100, 500, 1000]:
    for seed in [42, 43, 44]:
      run(Rank, TE, Mode, HP, seed)
```

| 항목 | 값 |
|------|-----|
| 조합 수 | 6 Rank × 6 TE × 2 Mode = 72 |
| Seed 반복 | 72 × 3 = **216 runs** |

#### Stage 1b (선택): Per-Rank Narrow Re-Search

Stage 2 결과에서 특정 TE가 base HP에서 성능이 크게 떨어지면,
해당 TE 근처에서 **좁은 범위** re-search를 수행하여 HP sensitivity 확인.

```yaml
# base HP 기준 ±3x 범위 내에서만 search
narrow_search_space:
  lr: [base_lr / 3, base_lr * 3]
  transfer_lr: [base_tlr / 3, base_tlr * 3]
n_trials: 10
```

이 결과는 논문에서 "HP를 조정해도 TE 경향성이 유지됨"을 보이는 supplementary로 활용.

| 항목 | 값 |
|------|-----|
| 대상 | 성능 하락 cell만 (예상 ~12개) |
| Runs | ~12 × 10 trials × 3 seeds = **~360 runs** |

---

## 재실험 우선순위

### Priority 1 — 2-Stage HP 통제 TE Sweep (필수)

**목적**: HP confound 제거 후 순수 TE 효과 측정, rank별 일관된 TE 경향성 확보

```
Ranks: [1, 4, 8, 16, 32, 64]
TEs:   [1, 10, 50, 100, 500, 1000]
Modes: [Reset, Decay]
Seeds: [42, 43, 44]
```

| Stage | 내용 | Runs |
|-------|------|------|
| Stage 1 | Per-Rank HP Search (ref TE=100) | 240 |
| Stage 2 | Fixed HP, all TE, 3-seed | 216 |
| Stage 1b | Narrow re-search (optional) | ~360 |
| **총 runs** | | **~816 runs** |

**이 실험 완료 시 주장 가능한 것**:
- "HP를 통제했을 때, Decay는 TE↓에서, Reset은 TE↑에서 일관되게 우수" (rank별 curve로 입증)
- "이 경향성은 Rank=1~64 전 구간에서 일관됨" (6개 rank 모두 같은 패턴)
- mean ± std로 통계적 유의성 확보 (t-test, ANOVA)

---

### Priority 2 — Decay Rank=64 붕괴 분석 (권장)

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

P1의 Stage 1에서 찾은 base HP를 사용하여 HP 통제 상태에서 붕괴 발생 여부 확인.

| 항목 | 값 |
|------|-----|
| 총 cell | 2 × 7 × 1 = 14 |
| Seed 반복 | 14 × 5 = 70 runs |
| **총 runs** | **~70 runs** (HP는 P1에서 확정) |

**이 실험 완료 시 추가 주장 가능한 것**:
- "Decay 모드에서 high-rank + mid-TE 조합은 training collapse 위험이 있으며, TE < X 또는 TE > Y 에서 안정적"
- 붕괴 확률을 수치화하여 practical guideline 제시

---

## 전체 실험 규모 요약

| Priority | 실험 | Runs | 누적 |
|----------|------|------|------|
| **P1** | 2-Stage HP 통제 TE Sweep | ~816 | 816 |
| **P2** | 붕괴 분석 | ~70 | 886 |

- MNIST MLP 30 epoch → ~15~20 sec/run (GPU)
- **P1 (필수)**: ~816 runs ≈ **3~5 GPU-hours**
- **전체**: ~886 runs ≈ **4~5 GPU-hours**

---

## 실험 완료 후 논문 Figure 계획

### Main Figure: Rank × TE Heatmap (1×3)
- (a) Reset mode accuracy heatmap (mean, HP 통제)
- (b) Decay mode accuracy heatmap (mean, HP 통제)
- (c) Decay − Reset difference (with significance markers)

### Main Figure: Per-Rank TE Curve
- x축: TE (log scale), y축: Accuracy (%)
- 각 Rank별 curve (Reset/Decay 각각)
- **HP 통제 상태이므로 TE 효과만 반영 → 논문 주장 강화**
- 6개 Rank가 일관된 패턴 → "Rank-invariant TE effect"

### Supplementary Figure
- HP sensitivity 분석 (Stage 1b narrow re-search 결과)
- Collapse probability heatmap (P2 데이터)
- Rank별 ANOVA 결과 table

---

## 실행 순서

```bash
# Step 1: Stage 1 — Per-Rank HP Search at reference TE=100
python sweep_mnist_6t1c.py --ranks 1,4,8,16,32,64 --tes 100 \
    --modes reset,decay --seeds 42 --lifetime 0 --n_trials 20 \
    --stage hp_search

# Step 2: Stage 2 — Fixed HP, TE Sweep, 3-seed
python sweep_mnist_6t1c.py --ranks 1,4,8,16,32,64 --tes 1,10,50,100,500,1000 \
    --modes reset,decay --seeds 42,43,44 --lifetime 0 \
    --stage te_sweep --hp_from stage1_results.json

# Step 3 (optional): Stage 1b — Narrow Re-Search for underperforming cells
python sweep_mnist_6t1c.py --ranks 1,4,8,16,32,64 --tes <underperforming_TEs> \
    --modes reset,decay --seeds 42,43,44 --lifetime 0 --n_trials 10 \
    --stage narrow_search --hp_from stage1_results.json --hp_range 3x

# Step 4: P2 — 붕괴 분석 (Decay only, 5-seed)
python sweep_mnist_6t1c.py --ranks 32,64 --tes 60,70,80,90,100,120,150 \
    --modes decay --seeds 42,43,44,45,46 --lifetime 0 \
    --stage te_sweep --hp_from stage1_results.json
```

---

## 참고: Lifetime 미사용 근거

기존 실험에서 lifetime 값 (1000, 10000, 46505, 100000)이 혼재되어 있었으나, 6T1C capacitor의 물리적 retention (τ=775min)은 **30-epoch MNIST 학습 시간 (~10min) 대비 충분히 길어** 학습 중 실질적 decay가 무시 가능하다. 재실험에서는 lifetime=0 (retention 비활성화)으로 통일하여 변수를 제거한다.

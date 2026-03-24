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
2. **HP confound**: 각 (Rank, TE) cell이 독립적으로 HP 최적화 → TE 효과 vs HP 차이 분리 불가

---

## HP 경향성 분석 (curve fitting)

기존 실험의 best HP 데이터에 대해 curve fitting을 수행하여 Rank/TE에 따른 체계적 규칙을 도출.

### Reset 모드

| 파라미터 | Rank 경향 | fitting | R² |
|---------|----------|---------|-----|
| **lr** | 경향 없음 | `lr = 0.3` (constant) | — |
| **tlr** | Rank↑ → tlr↓ | `tlr = 0.009 / sqrt(Rank)` | 0.81 |

```
실측 vs 수식:
Rank |  actual  |  0.009/√R  |  비율
   1 |  0.0648  |  0.0090    |  (outlier: best TE=500, 다른 rank은 TE=50~100)
   4 |  0.0039  |  0.0045    |  0.87x
   8 |  0.0042  |  0.0032    |  1.31x
  16 |  0.0022  |  0.0023    |  0.96x
  32 |  0.0012  |  0.0016    |  0.76x
  64 |  0.0013  |  0.0011    |  1.18x
```

Rank=1 (tlr=0.065)은 outlier — best TE=500이어서 다른 rank(best TE=50~100)과 운용 구간이 다름. 수식 `0.009/√1 = 0.009`로 통일.

### Decay 모드

| 파라미터 | TE 경향 | fitting | R² |
|---------|---------|---------|-----|
| **lr** | TE↑ → lr↑ | `lr = 0.25 · log₁₀(TE) + 0.11` | 0.81 |
| **tlr** | TE↑ → tlr↑ | `tlr = 0.005 · log₁₀(TE) + 0.0015` | 0.97 |

Rank 무관 확인: 같은 TE=1에서 R16, R32, R64 모두 동일 HP (lr=0.089, tlr=0.001).
같은 TE=100에서 R4, R8 모두 동일 HP (lr=0.494, tlr=0.011).

```
실측 vs 수식:
  TE |  lr실측  |  lr수식   |   tlr실측   |  tlr수식
   1 |  0.089  |  0.110   |  0.0013    |  0.0015
  50 |  0.674  |  0.535   |  0.0118    |  0.0100
 100 |  0.494  |  0.611   |  0.0112    |  0.0115
```

TE=500, 1000은 기존 데이터 없음 → 수식 외삽:
```
  TE |  lr수식   |  tlr수식     |  비고
  10 |  0.360   |  0.0065     |
 500 |  0.786   |  0.0150     |  lr clamp max 0.8
1000 |  0.861   |  0.0165     |  lr clamp max 0.8
```

---

## 최종 HP 공식

```python
def get_hp(mode, rank, te):
    """수학적으로 일관된 HP 할당.

    Reset: lr 고정, tlr ∝ 1/√Rank
    Decay: lr, tlr ∝ log₁₀(TE) — Rank 무관
    """
    if mode == "reset":
        lr  = 0.3
        tlr = 0.009 / math.sqrt(rank)
    elif mode == "decay":
        log_te = math.log10(max(te, 1))
        lr  = min(0.25 * log_te + 0.11, 0.8)
        tlr = 0.005 * log_te + 0.0015
    return lr, tlr
```

### Reset HP Table

| Rank | lr | tlr |
|------|-----|------|
| 1 | 0.300 | 0.00900 |
| 4 | 0.300 | 0.00450 |
| 8 | 0.300 | 0.00318 |
| 16 | 0.300 | 0.00225 |
| 32 | 0.300 | 0.00159 |
| 64 | 0.300 | 0.00113 |

### Decay HP Table

| TE | lr | tlr |
|----|-----|------|
| 1 | 0.110 | 0.0015 |
| 10 | 0.360 | 0.0065 |
| 50 | 0.535 | 0.0100 |
| 100 | 0.611 | 0.0115 |
| 500 | 0.786 | 0.0150 |
| 1000 | 0.800 | 0.0165 |

---

## 실험 설정 (공통)

```yaml
network: MLP (784 → 256 → 10, Sigmoid, MNIST)
device: 6T1C (dw_min=0.001981, gamma_up=-0.1678, gamma_down=+0.1410)
lifetime: 0  # 사용하지 않음
epochs: 30
batch_size: 64
optimizer: AnalogSGD
scheduler: StepLR(step_size=10, gamma=0.5)
seeds: [42, 43, 44]
output_layer: IdealizedPreset
bias: True
activation: Sigmoid
loss: NLLLoss
lora_alpha: 1.0
forward_inject: False
reinit_gain: 0.1
```

---

## 재실험 내용

```
Ranks: [1, 4, 8, 16, 32, 64]
TEs:   [1, 10, 50, 100, 500, 1000]
Modes: [Reset, Decay]
Seeds: [42, 43, 44]
HP:    get_hp(mode, rank, te) — search 없음, 수식 기반 고정
```

| 항목 | 값 |
|------|-----|
| 총 cell | 6 Rank × 6 TE × 2 Mode = 72 |
| Seed 반복 | 72 × 3 = **216 runs** |
| 예상 시간 | **~1 GPU-hour** (30 epoch, ~15 sec/run) |

### 주장 가능한 것

- HP를 **수식으로 통제**했으므로 accuracy 차이 = 순수 TE 효과
- 6개 Rank 전부 일관된 패턴 → **"Rank-invariant TE effect"**
- mean ± std (3 seeds) → t-test, ANOVA 가능

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

- LRTT: `PythonLRTTPreset.sixt1c_ab(rank, te, reinit_mode, ...)`
- Reset: `reinit_mode="standard"` (A=0, B=Kaiming)
- Decay: `reinit_mode="decay"` (A,B 유지)
- `include_retention=False`, `dt_batch_sec=0.0`
- `forward_inject=False`, `reinit_gain=0.1`, `lora_alpha=1.0`

### 실행 명령

```bash
cd /path/to/LRTT/experiments

# 전체 실험 (216 runs, ~1 GPU-hour)
python sweep_mnist_6t1c_reexp.py --priority 1

# 단일 cell 테스트
python sweep_mnist_6t1c_reexp.py --modes decay --ranks 8 --tes 100 --seeds 42
```

### 출력 파일

- `reexp_P1_results.json` — 전체 결과 (mode, rank, te, lr, tlr, seed_accs, mean, std)
- `reexp_P1_results.csv` — 요약 테이블

---

## 논문 Figure 계획

### Main Figure: Rank × TE Heatmap (1×3)
- (a) Reset mode accuracy heatmap (mean)
- (b) Decay mode accuracy heatmap (mean)
- (c) Decay − Reset difference (with significance markers)

### Supplementary
- Per-Rank TE sweep curve (6 panels) — rank별 일관된 패턴 입증
- ANOVA table

---

## 참고: Lifetime 미사용 근거

6T1C capacitor의 물리적 retention (τ=775min)은 30-epoch MNIST 학습 시간 (~10min) 대비 충분히 길어 학습 중 실질적 decay가 무시 가능하다. lifetime=0으로 통일하여 변수를 제거한다.

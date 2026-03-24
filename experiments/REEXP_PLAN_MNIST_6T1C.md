# MNIST 6T1C LRTT 재실험 계획

**Date**: 2026-03-24
**Branch**: MLP
**Goal**: Reset vs Decay 모드의 TE 운용구간 경향성을 rank별로 일관되게 입증

---

## 실험 코드

기존 sweep 코드 (`sweep_rank8_nn_decay.py`, `sweep_rank8_nn_hybrid.py`)와 **동일한 모델/학습/평가 코드** 사용. 변경점: rank를 config에서 읽도록, lifetime=0, weight_scaling_omega 제거.

### 핵심 설정 (기존과 동일)

```
A/B tiles: 6T1C LinearStepDevice (dw_min=0.001981, gamma_up=-0.1678, gamma_down=0.1410)
C tile:    SoftBoundsDevice (no noise)
Output:    FloatingPointRPUConfig
Network:   784 → 256 (ReLU) → 10 (LogSoftmax)
reinit_gain: 1.0
update_mode: "lora"
transfer_mode: "off"
out_noise: 0.0
Epochs: 30, Early stop patience: 5
```

### 변경점

```
lifetime: 0 (기존 46505 → 비활성화)
weight_scaling_omega: 제거
rank: config에서 읽음 (기존 8 고정)
```

---

## Step 1 — 고정 HP로 전체 Grid

**목적**: HP 완전 통제 상태에서 순수 (Mode, Rank, TE) 효과 측정

```
lr  = 0.3    (고정)
tlr = 0.005  (고정)
Ranks: [1, 4, 8, 16, 32, 64]
TEs:   [1, 10, 50, 100, 500, 1000]
Modes: [Decay, Hybrid(Reset)]
Trials: 3 per cell
```

| 항목 | 값 |
|------|-----|
| 총 cell | 6 × 6 × 2 = 72 |
| 총 runs | 72 × 3 = **216 runs** |
| 예상 시간 | **~1 GPU-hour** |

```bash
cd experiments
python generate_reexp_config.py
python sweep_reexp.py --config reexp_sweep_configs.json --mode decay
python sweep_reexp.py --config reexp_sweep_configs.json --mode hybrid
```

---

## Step 2 — 경향성 분석

Step 1 결과에서 확인:
1. Reset vs Decay의 TE 경향이 rank별로 일관되는가?
2. Rank가 accuracy에 유의한 영향을 주는가?
3. tlr=0.005가 특정 구간에서 부적합한가?

---

## Step 3 — tlr = 0.009/√Rank 로 재실험 (Rank=1 제외)

**목적**: transfer_lr을 rank에 반비례하게 조정하여 Step 1과 비교

```
lr  = 0.3            (고정)
tlr = 0.009/√Rank    (rank 종속)
Ranks: [4, 8, 16, 32, 64]   (Rank=1 제외)
TEs:   [1, 10, 50, 100, 500, 1000]
Modes: [Decay, Hybrid(Reset)]
Trials: 3 per cell
```

| Rank | tlr |
|------|------|
| 4 | 0.004500 |
| 8 | 0.003182 |
| 16 | 0.002250 |
| 32 | 0.001591 |
| 64 | 0.001125 |

| 항목 | 값 |
|------|-----|
| 총 cell | 5 × 6 × 2 = 60 |
| 총 runs | 60 × 3 = **180 runs** |
| 예상 시간 | **~50 min** |

```bash
cd experiments
python generate_reexp_config.py --tlr_rule sqrt_rank
python sweep_reexp.py --config reexp_sweep_configs_sqrt_rank.json --mode decay
python sweep_reexp.py --config reexp_sweep_configs_sqrt_rank.json --mode hybrid
```

---

## Step 4 — 추가 tlr sweep (Step 2/3 결과에 따라 조건부)

Step 2에서 특정 구간 성능 급락 시, 해당 cell만 tlr sweep:

```
tlr_values: [0.001, 0.002, 0.005, 0.01, 0.02]
대상: underperforming cells만
```

---

## 전체 실험 요약

| Step | 내용 | Runs | 누적 |
|------|------|------|------|
| 1 | 고정 HP (lr=0.3, tlr=0.005) | 216 | 216 |
| 3 | tlr=0.009/√Rank | 180 | 396 |
| 4 | tlr sweep (필요시) | ~360 | ~756 |

---

## 출력 파일

```
results/reexp_decay/results_final.json       # Step 1 Decay
results/reexp_hybrid/results_final.json      # Step 1 Hybrid
results/reexp_sqrt_decay/results_final.json  # Step 3 Decay
results/reexp_sqrt_hybrid/results_final.json # Step 3 Hybrid
```

---

## 참고: Lifetime 미사용 근거

6T1C capacitor의 물리적 retention (τ=775min)은 30-epoch MNIST 학습 시간 (~10min) 대비 충분히 길어 학습 중 실질적 decay가 무시 가능하다. lifetime=0으로 통일하여 변수를 제거한다.

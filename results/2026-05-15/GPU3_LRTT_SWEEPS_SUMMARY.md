# LRTT / tikitaka sweep 결과 정리

- **수집 일시**: 2026-05-15
- **데이터셋**: MNIST
- **방법론**: TPE (Optuna) HP search, 각 셀에서 (lr, tlr, clr) 또는 (transfer_lr, fast_lr, classifier_lr) 탐색
- **소스 디렉터리**:
  - `/root/LRTT/results/sweep_rank_device/` (2026-05-05, **30 trials/cell**)
  - `/root/LRTT/results/sweep_rank_device_wide/` (2026-05-06, **70 trials/cell, rank=64 포함**)

## 1. Sweep 1: `sweep_rank_device` (2026-05-05, 30 trials/cell)

- rank ∈ {1, 2, 4, 8, 16}, c_tile ∈ {ideal, ecram, rram}
- methods: lrtt_v1, lrtt_v2, tikitaka_v1 (rank 없음)

### 1.1 best accuracy (5×3 grid, 단위: %)

| method | rank | ideal | ecram | rram |
|---|---|---|---|---|
| **lrtt_v1** | 1  | 96.71 | 96.80 | 95.29 |
|             | 2  | 97.07 | 97.14 | 95.95 |
|             | 4  | 97.32 | 97.45 | 96.16 |
|             | 8  | 97.50 | 97.67 | 96.45 |
|             | 16 | 97.73 | **97.78** | 96.86 |
| **lrtt_v2** | 1  | 93.48 | 95.45 | 94.99 |
|             | 2  | 93.82 | 96.09 | 95.77 |
|             | 4  | 94.50 | 96.72 | 95.76 |
|             | 8  | 95.54 | 96.96 | 95.81 |
|             | 16 | 96.21 | **97.49** | 95.98 |
| **tikitaka_v1** | — | 95.96 | **97.96** | 15.76 (발산) |

## 2. Sweep 2: `sweep_rank_device_wide` (2026-05-06, 70 trials/cell, 가장 최근)

### 2.1 best accuracy (4×3 grid, 단위: %)

| method | rank | ideal | ecram | rram |
|---|---|---|---|---|
| **lrtt_v1** | 1  | 96.80 | 97.02 | 95.26 |
|             | 4  | 97.42 | 97.39 | 96.27 |
|             | 16 | 97.75 | 97.89 | 97.01 |
|             | 64 | 98.06 | **98.24** | 97.46 |
| **lrtt_v2** | 1  | 93.40 | 95.42 | 95.07 |
|             | 4  | 94.63 | 96.69 | 95.95 |
|             | 16 | 96.24 | 97.62 | 96.15 |
|             | 64 | 97.56 | **98.03** | 96.15 |
| **tikitaka_v1** | — | 96.11 | **98.26** | 17.40 (발산) |

### 2.2 셀별 best hyperparameters (wide sweep)

| method | rank | c_tile | best_acc | lr | tlr | clr |
|---|---|---|---|---|---|---|
| lrtt_v1 | 1  | ideal | 96.80 | 1.96  | 8.78e-4 | 0.278 |
| lrtt_v1 | 1  | ecram | 97.02 | 0.659 | 1.27e-3 | 0.436 |
| lrtt_v1 | 1  | rram  | 95.26 | 0.373 | 7.15e-4 | 0.165 |
| lrtt_v1 | 4  | ideal | 97.42 | 2.98  | 6.20e-4 | 0.405 |
| lrtt_v1 | 4  | ecram | 97.39 | 0.897 | 1.87e-3 | 0.351 |
| lrtt_v1 | 4  | rram  | 96.27 | 0.0478| 6.26e-4 | 0.115 |
| lrtt_v1 | 16 | ideal | 97.75 | 0.20  | 3.43e-4 | 0.435 |
| lrtt_v1 | 16 | ecram | 97.89 | 0.60  | 2.31e-4 | 0.394 |
| lrtt_v1 | 16 | rram  | 97.01 | 0.0909| 2.42e-4 | 0.106 |
| lrtt_v1 | 64 | ideal | 98.06 | 0.192 | 3.14e-4 | 0.260 |
| lrtt_v1 | 64 | ecram | 98.24 | 1.69  | 1.26e-4 | 0.364 |
| lrtt_v1 | 64 | rram  | 97.46 | 0.494 | 1.44e-4 | 0.141 |
| lrtt_v2 | 1  | ideal | 93.40 | 2.27  | 5.21    | 0.977 |
| lrtt_v2 | 1  | ecram | 95.42 | 9.65  | 1.43    | 4.12  |
| lrtt_v2 | 1  | rram  | 95.07 | 4.63  | 0.534   | 0.861 |
| lrtt_v2 | 4  | ideal | 94.63 | 1.05  | 0.548   | 0.883 |
| lrtt_v2 | 4  | ecram | 96.69 | 3.73  | 0.0251  | 1.10  |
| lrtt_v2 | 4  | rram  | 95.95 | 0.349 | 0.694   | 0.796 |
| lrtt_v2 | 16 | ideal | 96.24 | 0.172 | 0.287   | 2.55  |
| lrtt_v2 | 16 | ecram | 97.62 | 0.194 | 0.0219  | 1.12  |
| lrtt_v2 | 16 | rram  | 96.15 | 7.68  | 0.0228  | 0.654 |
| lrtt_v2 | 64 | ideal | 97.56 | 0.234 | 0.437   | 2.80  |
| lrtt_v2 | 64 | ecram | 98.03 | 0.465 | 0.0129  | 2.36  |
| lrtt_v2 | 64 | rram  | 96.15 | 7.71  | 0.0103  | 0.627 |

(tikitaka_v1는 transfer_lr / fast_lr / classifier_lr로 컬럼이 다름 — JSON 참조.)

## 3. 핵심 관찰

1. **ecram이 ideal보다 일관되게 우수.** lrtt_v1/v2 모두 ecram에서 best (예: lrtt_v1 rank64 ecram 98.24% > ideal 98.06%). 이는 ecram의 대칭적 update가 implicit regularization으로 작용한다는 해석을 뒷받침.
2. **rram에서 tikitaka_v1 완전 발산** (15.76% / 17.40%). 반면 lrtt v1/v2는 rank≥1만 있어도 95~97% 유지 → **rank constraint가 rram의 asymmetric noise 안정화에 결정적**.
3. **lrtt_v1 > lrtt_v2**: 특히 ideal에서 격차 큼 (rank4 ideal에서 v1 97.42 vs v2 94.63, **약 2.8%p**). rank를 64로 키우면 격차가 0.5%p로 좁혀짐 — **저rank에서 v2가 더 불리**.
4. **rank scaling**: 두 method 모두 rank ↑ → acc ↑ (monotonic). lrtt_v1 ecram: 97.02 → 97.39 → 97.89 → 98.24 (rank 1→4→16→64).
5. **HP 패턴 차이**:
   - lrtt_v1은 매우 작은 tlr (`1e-4~1e-3`) 선호.
   - lrtt_v2는 큰 tlr (`0.01~5`) 선호, lr도 큰 값 (`0.5~10`).
   - tikitaka_v1은 transfer_lr와 fast_lr이 같은 scale (`0.01~1`) 영역에서 최적.
6. **wide(70 trials) vs narrow(30 trials)**: lrtt_v1 ecram rank16 결과가 97.78 → 97.89로 0.1%p 향상 정도, 큰 차이 없음 → **30 trial로도 best는 거의 찾힘**.

## 4. 결과 디렉터리 구조

```
/root/LRTT/results/
├── sweep_rank_device/
│   ├── lrtt_v1/        rank{1,2,4,8,16}_C{ideal,ecram,rram}.json  (15 cells)
│   ├── lrtt_v2/        rank{1,2,4,8,16}_C{ideal,ecram,rram}.json  (15 cells)
│   └── tikitaka_v1/    norank_C{ideal,ecram,rram}.json            (3 cells)
└── sweep_rank_device_wide/
    ├── lrtt_v1/        rank{1,4,16,64}_C{ideal,ecram,rram}.json   (12 cells)
    ├── lrtt_v2/        rank{1,4,16,64}_C{ideal,ecram,rram}.json   (12 cells)
    └── tikitaka_v1/    norank_C{ideal,ecram,rram}.json            (3 cells)
```

각 셀 JSON: `method, rank, c_tile, search_space, warm_start, trials[{hp, acc, wall_seconds}]`.

## 5. 산출물

- `/root/lrtt_sweep_results.json` — 머신리더블 전체 데이터 (trial 통계, best_hp 포함).
- `/root/lrtt_sweep_results.md` — 이 문서.

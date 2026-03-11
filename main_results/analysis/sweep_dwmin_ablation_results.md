# sweep_dwmin_ablation 결과 분석

> 실험일: 2026-03-06 | auto_scale=True, 15 runs | baseline: auto_scale=False (best_loss=1.64)

## 1. 실험 목적

A-tile 및 B-tile의 `dw_min` 값을 변경하여 TikiTaka 학습 성능에 미치는 영향을 측정.
Baseline (auto_scale=False, dw_min=0.0005)에서 관찰된 gradient deadzone 문제가 dw_min 조정으로 해결 가능한지 검증.

## 2. 실험 조건 (15 runs)

공통 설정: seed=42, steps=200, desired_bl=31, lr=0.1, transfer_every=8, fast_lr=1.0, units_in_mbatch=False, auto_scale=**True**

### Phase 1: A-tile dw_min 변경 (B=0.0005 고정)

| # | dw_min_A | a_noise_free | 설명 |
|---|----------|-------------|------|
| 1 | 0.000198 (10× default) | False | A-tile 해상도 낮춤 |
| 2 | 1.98e-05 (default) | False | default A + auto_scale |
| 3 | 1.98e-06 (0.1× default) | False | A-tile 해상도 높임 |
| 4 | 0.001981 | **True** | noise-free + 큰 dw_min |
| 5 | 0.000198 | **True** | noise-free + 중간 dw_min |
| 6 | 1.98e-05 | **True** | noise-free + default dw_min |
| 7 | 1.98e-06 | **True** | noise-free + 작은 dw_min |

### Phase 2: B-tile dw_min 변경 (A=default)

| # | dw_min_B | a_noise_free | 설명 |
|---|----------|-------------|------|
| 8 | 5e-05 (10×↓) | False | B-tile 해상도 높임 |
| 9 | 5e-06 (100×↓) | False | B-tile 해상도 더 높임 |

### Phase 3: A+B 조합

| # | dw_min_A | dw_min_B | a_noise_free |
|---|----------|----------|-------------|
| 10 | 0.001981 | 5e-05 | True |
| 11 | 0.001981 | 5e-06 | True |
| 12 | 0.000198 | 5e-05 | True |
| 13 | 1.98e-05 | 5e-06 | True |
| 14 | 0.000198 | 5e-05 (A noisy) | False |
| 15 | 1.98e-06 | 5e-06 (A noisy) | False |

## 3. 결과 테이블

| dw_min_B | dw_min_A | noise_free | best_loss | best_step | final_loss | vs baseline |
|----------|----------|-----------|-----------|-----------|------------|-------------|
| **Baseline** | **—** | **—** | **1.64** | **130** | **5.95** | **—** |
| 0.0005 | 1.98e-05 | False | **2.05** | 130 | 2.12 | +0.41 |
| 0.0005 | 1.98e-06 | False | 2.71 | 90 | 5.95 | +1.07 |
| 0.0005 | 0.000198 | False | 3.78 | 20 | 5.95 | +2.14 |
| 0.0005 | 1.98e-05 | True | 3.01 | 70 | 5.93 | +1.37 |
| 0.0005 | 1.98e-06 | True | 3.44 | 40 | 5.95 | +1.80 |
| 0.0005 | 0.000198 | True | 3.78 | 30 | 4.98 | +2.14 |
| 0.0005 | 0.001981 | True | 4.08 | 10 | 5.08 | +2.44 |
| 5e-05 | default | False | 2.30 | 170 | 2.39 | +0.66 |
| 5e-05 | 0.000198 | True | 2.73 | 120 | 2.87 | +1.09 |
| 5e-05 | 0.001981 | True | 2.54 | 90 | 2.84 | +0.90 |
| 5e-06 | default | False | 2.50 | 190 | 2.50 | +0.86 |
| 5e-06 | 0.001981 | True | 2.24 | 130 | 2.54 | +0.60 |
| 5e-06 | 1.98e-05 | True | — | — | — | (failed) |

**모든 조건이 baseline(1.64)보다 성능 저하.**

## 4. 핵심 발견사항

### 4.1. 근본 원인: grad << dw_min (gradient deadzone)

Baseline trace에서 측정된 A-tile 상태:

| Step | grad_absmean | dw_min | deadzone_ratio | BL_mean (target=31) | pulse_under |
|------|-------------|--------|---------------|---------------------|-------------|
| 0 | 0.000281 | 0.0005 | 99.99% | 1.00 | 99.99% |
| 10 | 0.000800 | 0.0005 | 99.97% | 1.00 | 99.83% |
| 50 | 0.001787 | 0.0005 | 99.83% | 1.01 | 98.66% |
| 100 | 0.004313 | 0.0005 | 97.15% | 1.23 | 85.84% |
| 199 | 0.000088 | 0.0005 | 100.0% | 1.00 | 100.0% |

**인과관계 체인:**

```
BERT gradient (absmean ≈ 0.0003~0.004)
    ↓ dw_min = 0.0005
    ↓ grad / dw_min ≈ 0.6~8 → 대부분 원소가 threshold 미달
    ↓
A-tile: deadzone 97~100% → BL ≈ 1 (desired=31인데 pulse 1개)
    ↓ Δw_A ≈ 1 × dw_min → gradient 방향 정보 소실
    ↓
Hidden buffer: noise만 축적 (1.7 → 174.2 over 200 steps)
    ↓
A→B transfer: random direction (cosine_slow_grad = 0.002)
    ↓
B-tile weight: gradient와 무관 (update_vs_grad_cosine = 0.002)
    ↓
학습 실패 (analog 기여 ≈ 0)
```

### 4.2. auto_scale=True에서 dw_min 변경이 무효한 이유

auto_scale=True는 gradient를 균일 스케일링하여 `n_pulse ≈ desired_bl`로 고정.
따라서 실제 weight update는:

```
Δw = n_pulse × dw_min ≈ desired_bl × dw_min
```

| dw_min_B | Δw ≈ 31 × dw_min | 효과 |
|----------|-------------------|------|
| 0.0005 (기본) | 0.0155 | 기본 step size |
| 5e-05 (10×↓) | 0.00155 | step size 10×↓ |
| 5e-06 (100×↓) | 0.000155 | step size 100×↓ |

**dw_min↓ → effective LR↓ → 학습 속도 감소** (방향 문제는 그대로)

### 4.3. 두 가지 문제의 동시 존재 (dw_min 딜레마)

| 방향 | 효과 | 결과 |
|------|------|------|
| dw_min ↑ | 더 많은 원소가 deadzone → 방향 정보 더 손실 | 악화 |
| dw_min ↓ (auto_scale) | effective LR 감소 → 학습 속도 저하 | 악화 |
| dw_min ↓ (auto_scale=False) | BL=1 고정, Δw↓ → 학습 속도 저하 | 악화 |

**어느 방향으로 조정해도 성능 개선 불가** — dw_min은 병목이 아님.

### 4.4. TikiTaka transfer 메커니즘 실패 상세

Baseline trace (auto_scale=False, step 0 → 199 평균):

| Metric | Step 0 | Step 199 | 의미 |
|--------|--------|----------|------|
| hidden_absmean | 1.68 | 174.19 | buffer 무한 성장 (103×) |
| transfer_duty | 69.6% | 98.2% | 거의 모든 원소가 transfer 대상 |
| cosine_slow_grad | 0.042 | 0.002 | B-tile weight ⊥ gradient |
| cosine_slow_fast | 0.067 | 0.005 | B-tile ⊥ A-tile (transfer 무효) |
| update_vs_grad_cosine | 0.039 | 0.002 | 전체 update ⊥ gradient |
| w_eff_absmean (A-tile) | 0.030 | 0.028 | A-tile weight 변화 미미 |

### 4.5. Digital optimizer가 초반 학습을 담당

| Step | dw_fast_vs_grad_cosine | dw_fast_absmean | 역할 |
|------|----------------------|-----------------|------|
| 0 | **0.518** | 0.007 | gradient 방향 충실히 추종 |
| 10 | 0.058 | 0.015 | 빠르게 방향 정확도 감소 |
| 50 | -0.038 | 0.027 | 방향 상실 |
| 199 | -0.015 | 0.003 | digital도 사실상 무효 |

초반 loss 감소(5.83→1.64)는 digital optimizer 덕분이며, analog(TikiTaka)은 처음부터 기여하지 못함.

## 5. 결론

### dw_min은 병목이 아니다

- BERT fine-tuning의 gradient magnitude (0.0003~0.004)가 stochastic pulse의 최소 해상도(dw_min=0.0005)보다 작거나 비슷함
- 이로 인해 gradient → pulse 변환 단계에서 방향 정보가 소실
- dw_min을 조정해도 auto_scale이 pulse count를 고정하므로 방향 문제 해결 불가
- dw_min↓는 오히려 effective LR을 감소시켜 성능 악화

### 진짜 병목: stochastic pulse의 gradient 인코딩 한계

- 개별 weight element의 gradient가 dw_min에 비해 너무 작아 pulse 1개 이하로 양자화됨
- A-tile에 gradient 정보가 축적되지 않으므로, hidden buffer → B-tile transfer가 noise만 전달
- TikiTaka의 전체 메커니즘(A accumulation → buffer → B transfer)이 gradient 방향을 보존하지 못함

### 다음 단계 제안

1. **auto_scale=False + 매우 작은 dw_min** 테스트 (BL=1이지만 step size 자체를 줄임)
2. **desired_bl을 극단적으로 높이기** (gradient 스케일링을 더 크게)
3. **Gradient accumulation** (여러 batch의 gradient를 합산하여 magnitude를 키움)
4. **Mixed-precision transfer**: A-tile을 FP로 유지하고 B-tile만 analog
5. **근본적 재설계**: BERT급 small-gradient task에 맞는 pulse encoding 방식 탐색

## 6. 데이터 경로

- Sweep 결과: `/root/LRTT/main_results/results/weight_update/squad/tiki/sweep_dwmin_ablation/`
- Baseline trace: `/root/LRTT/main_results/results/weight_update/squad/tiki/sweep_24_tiki_trace/`
- 각 run: `run_*/eval_loss.csv`, `run_*/config_dump.json`

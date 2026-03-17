# Phase 0 / Phase 1 실험 분석 보고서

> 작성일: 2026-03-17
> 모델: `bert-base-uncased` / Task: `SQuAD v1.1`
> Analog 변환 대상: Encoder Attention (QKVO) — 48개 Linear, FFN 및 qa_outputs/pooler는 digital 유지
> 공통 설정: `seed=42`, `batch_size=48`, `dw_min=1.22e-4` (14-bit), `desired_bl=31`, `io_perfect=True`

---

## 1. Phase 0A — Smoke Test (5 methods × 50 steps)

### 1.1 목적

- 5가지 analog 학습 방법의 코드 경로 검증
- 초기 50 step에서의 loss 수렴 비교
- Update diagnostics를 통한 gradient 충실도 측정

### 1.2 실험 설정

| Method | Config Type | 핵심 특성 |
|--------|------------|-----------|
| `single_rpu` | SingleRPUConfig | ConstantStep, stochastic pulse update |
| `mixed_precision` | MixedPrecisionCompound | FP32 chi matrix 누적 + pulse transfer |
| `ttv1` | TransferCompound | 2-tile (Fast+Slow), transfer 기반 학습 |
| `cttv2` | ChoppedTransferCompound | TTv1 + auto_scale + input chopping |
| `ideal` | IdealDevice | FP32 update (성능 상한선) |

- `max_steps=50`, `ln_lr=0.016` (ablation 이전 기본값)
- `--diag-update-exact --diag-steps 50`으로 weight update 진단 활성화

### 1.3 Loss 수렴 비교

```
Step    single_rpu   mixed_prec         ttv1        ideal
────────────────────────────────────────────────────────────
   1        5.9602       5.9602       5.9602       5.9602
   5        4.3975       4.4056       4.3860       4.3750
  10        4.0803       4.1024       4.0781       4.0768
  20        3.9232       3.9279       3.9652       3.9078
  30        3.8124       3.8083       3.7371       3.7789
  40        3.4873       3.5126       3.5070       3.3997
  50        3.6712       3.8116       3.7335       3.6809
```

50 step 시점에서 4가지 방법 간 loss 차이가 미미 (3.67~3.81). 모든 코드 경로 정상 동작 확인.
`cttv2`는 로그가 기록되지 않아 결과 없음 (config 문제 추정).

### 1.4 속도 비교

| Method | Wall Time (50 steps) | 상대 속도 |
|--------|---------------------|-----------|
| ideal | 267s | 1.0× (baseline) |
| single_rpu | 2,609s | ~9.8× |
| mixed_precision | 2,621s | ~9.8× |
| ttv1 | 2,619s | ~9.8× |

Analog 방법들은 ideal 대비 약 10배 느림. 이는 stochastic pulse 연산의 오버헤드에 기인.

### 1.5 Update Diagnostics — Gradient 충실도

`bert.encoder.layer.0.attention.self.query` 기준, 50 step 평균:

| Method | mean cosine_sim | mean zero_frac | 해석 |
|--------|----------------|----------------|------|
| **ideal** | **0.980** | 0.9998 | FP32 → 거의 완벽한 gradient 방향 복원 |
| mixed_precision | 0.359 | 0.9997 | FP32 chi 누적 → 중간 수준 충실도 |
| single_rpu | 0.103 | 0.9057 | Stochastic pulse → 방향 노이즈 큼 |
| **ttv1** | **0.006** | 0.9999 | Slow tile에 gradient 미반영 (transfer 대기) |

- `cosine_sim`: 실제 weight update 방향과 이상적 gradient 방향 간 코사인 유사도
- `zero_frac`: weight matrix에서 값이 0인 비율

**핵심 발견**:
- TTv1의 cosine_sim이 0.006으로 거의 0. 이는 gradient가 Fast tile에만 축적되고, Slow tile(forward에 사용)에는 아직 전달되지 않았기 때문.
- single_rpu는 cosine_sim=0.103으로 방향성은 있으나 stochastic pulse 노이즈가 큼.
- mixed_precision은 FP32로 gradient를 누적하므로 cosine_sim=0.359로 single_rpu보다 약 3.5× 높은 충실도.

---

## 2. Phase 0B — LayerNorm LR Ablation

### 2.1 목적

LayerNorm의 학습률을 analog_lr(0.016)과 동일하게 할지, classifier_lr(0.003)과 동일하게 할지 결정.

### 2.2 실험 설정

| 설정 | method | ln_lr | analog_lr | classifier_lr | epochs | n_bits |
|------|--------|-------|-----------|---------------|--------|--------|
| ln=analog | single_rpu | 0.016 | 0.016 | 0.003 | 2 | 14 |
| ln=classifier | single_rpu | 0.003 | 0.016 | 0.003 | 2 | 14 |

### 2.3 학습 곡선

```
                  ln=analog (0.016)          ln=classifier (0.003)
Step     Epoch    Loss                       Loss
─────────────────────────────────────────────────────────────────
  20       1      3.964                      4.111
 100       1      2.511                      2.240
 500       1      2.043                      1.529
1000       1      1.781                      1.346
1500       1      1.964                      1.645
1845       1      1.981  (F1=74.52)          1.634  (F1=82.19)
3690       2      1.509  (F1=79.17)          1.194  (F1=84.21)
```

### 2.4 결과

| 설정 | Ep1 F1 | Ep1 EM | Ep2 F1 | Ep2 EM | Ep1→Ep2 변화 |
|------|--------|--------|--------|--------|-------------|
| ln=analog (0.016) | 74.52 | 64.45 | 79.17 | 69.40 | +4.65 |
| **ln=classifier (0.003)** | **82.19** | **72.68** | **84.21** | **75.09** | **+2.02** |

### 2.5 분석

- `ln_lr=0.003`이 `ln_lr=0.016` 대비 **Ep1에서 +7.67, Ep2에서 +5.04 F1** 우위.
- ln_lr=0.016은 step 20에서 loss가 더 낮지만(3.964 < 4.111), step 100 이후부터 역전됨. 초반에 LN이 빠르게 변하면서 일시적으로 loss를 낮추지만, 이후 normalization 불안정으로 수렴이 느려짐.
- ln_lr=0.003은 loss가 step 500에서 이미 1.529까지 하락하여 꾸준히 수렴.
- **결론**: LayerNorm은 classifier와 동일한 낮은 lr(0.003)로 학습시키는 것이 최적. 이후 모든 실험에 `ln_lr=0.003` 적용.

---

## 3. Phase 1 — TTv1 Regime Discovery

### 3.1 목적

TTv1(TransferCompound)의 핵심 하이퍼파라미터 `transfer_every`, `units_in_mbatch`, `gamma` 조합에 따른 학습 regime 탐색.

### 3.2 실험 설계

3가지 transfer 설정 × 2가지 gamma = 6개 실험:

```
m_batch = batch_size × seq_len = 48 × 384 = 18,432
BERT attention in_features = 768 columns
```

| Config | uim | te | Transfer/step | 전체 column sweep 주기 |
|--------|-----|-----|---------------|---------------------|
| A | false | 24 | 18432/24 = **768** | 매 step 전체 sweep |
| B | false | 2400 | 18432/2400 ≈ **8** | ~96 steps |
| C | true | 1 | **1** | 768 steps |

공통: `gamma ∈ {0.0, 0.1}`, `n_bits=14`, `epochs=2`, `ln_lr=0.003`, `fast_lr=1.0(default)`, `transfer_lr=1.0(default)`, `with_reset_prob=0.0(default)`

### 3.3 결과 요약

| Config | uim | te | γ | loss@100 | loss@1845 | Ep1 F1 | Ep2 F1 | Ep1→Ep2 |
|--------|-----|-----|---|---------|----------|--------|--------|---------|
| A | F | 24 | 0.0 | 6.011 | 5.909 | 6.42 | (중단) | — |
| B | F | 2400 | 0.0 | 2.486 | 2.180 | 66.99 | 47.70 | **-19.29** |
| B | F | 2400 | 0.1 | 3.952 | 3.248 | 36.73 | 16.00 | **-20.73** |
| **C** | **T** | **1** | **0.0** | **2.629** | **1.907** | **77.46** | **77.67** | **+0.21** |
| C | T | 1 | 0.1 | 3.668 | 2.829 | 59.74 | 48.55 | **-11.19** |
| A | F | 24 | 0.1 | — | — | — | — | (로그 없음) |

### 3.4 상세 분석

#### 3.4.1 Transfer 빈도의 영향 (γ=0.0 고정)

**Config A (768 transfers/step) — 학습 실패**

- Loss가 step 20(4.144) 이후 오히려 상승하여 step 100부터 ~6.0에서 고착.
- Epoch 1 F1 = 6.42 (사실상 랜덤 수준).
- 원인: 매 step마다 Fast tile의 768개 column 전체가 Slow tile로 transfer됨. Fast tile에 gradient가 1 step밖에 누적되지 않은 상태에서 Slow tile을 덮어쓰므로, 각 transfer의 signal-to-noise ratio가 극히 낮음. Slow tile이 사실상 noise로 오염.

**Config B (8 transfers/step) — 학습되지만 Ep2 붕괴**

- Epoch 1에서 F1=66.99까지 학습 성공.
- 그러나 Epoch 2에서 F1이 47.70으로 **19.3점 급락**.
- 원인 추정: `with_reset_prob=0.0`이므로 Fast tile에 gradient가 무한히 누적. Epoch 2 시점에서 Fast tile이 w_max/w_min bound에 saturation되면서 transfer 시 corrupt된 값이 Slow tile로 전달. 또한 step당 8개 column만 transfer하므로 column 간 정보 비대칭 발생.

**Config C (1 transfer/step) — 안정적 수렴**

- Epoch 1 F1=77.46, Epoch 2 F1=77.67로 **안정적 유지/소폭 개선**.
- 매 step에 1 column만 점진적으로 transfer하므로 Slow tile이 점진적으로 업데이트됨.
- Loss도 step 500 이후 ~1.8대로 안정적.

**결론**: Transfer 빈도가 낮을수록 안정적. 과도한 transfer는 Slow tile을 noise로 오염시키고, 적절한 빈도라도 `with_reset_prob=0.0`에서 Fast tile saturation으로 인해 epoch이 진행될수록 성능 하락.

```
Transfer 빈도와 성능:
  768/step (A) ──→ 완전 실패   (F1=6.42)
    8/step (B) ──→ 학습 후 붕괴 (F1=66.99→47.70)
    1/step (C) ──→ 안정적 수렴  (F1=77.46→77.67)
```

#### 3.4.2 Gamma의 영향

| | γ=0.0 | γ=0.1 | 차이 |
|--|-------|-------|------|
| Config B Ep1 | 66.99 | 36.73 | **-30.26** |
| Config B Ep2 | 47.70 | 16.00 | **-31.70** |
| Config C Ep1 | 77.46 | 59.74 | **-17.72** |
| Config C Ep2 | 77.67 | 48.55 | **-29.12** |

- γ=0.1로 Fast tile을 forward에 10%만 포함시켜도 성능이 **18~32 F1 하락**.
- Fast tile은 noisy gradient가 직접 누적되는 tile이므로, forward에 포함시키면 inference가 noise에 의해 크게 방해받음.
- γ=0.1에서는 Epoch 2 하락이 더 극심함 (Config C: 59.74→48.55, -11.19).
- **결론**: γ=0.0이 필수. Fast tile은 gradient accumulator로만 사용하고, forward에는 Slow tile만 사용해야 함.

#### 3.4.3 Loss 수렴 패턴 비교

```
                Config A (te=24)    Config B (te=2400)    Config C (te=1)
Step             uim=F, γ=0.0       uim=F, γ=0.0         uim=T, γ=0.0
────────────────────────────────────────────────────────────────────────
  20              4.144               4.119                 4.120
 100              6.011               2.486                 2.629
 500              5.975               1.865                 1.837
1000              5.957               2.024                 1.829
1845 (Ep1)        5.909               2.180                 1.907
```

- Config A: step 100부터 loss가 6.0으로 발산 — transfer 과다로 학습 불가
- Config B: step 500에서 1.865까지 수렴했으나 이후 다시 상승 (2.180) — 불안정
- Config C: step 500 이후 ~1.8~1.9 범위에서 안정적 유지

### 3.5 TTv1 vs 다른 방법 비교 (2 epochs, 14-bit 기준)

| Method | Best F1 | Final F1 | Final EM | 실험 |
|--------|---------|----------|----------|------|
| single_rpu (ln=0.003) | 84.21 | 84.21 | 75.09 | Phase 0B |
| **TTv1 configC (best)** | **77.67** | **77.67** | **67.28** | **Phase 1** |
| TTv1 configB | 66.99 | 47.70 | 35.34 | Phase 1 |
| single_rpu (ln=0.016) | 79.17 | 79.17 | 69.40 | Phase 0B |

TTv1 최선(configC)은 single_rpu(ln=0.003) 대비 **-6.54 F1**.

이 gap의 원인:
1. **Gradient 전달의 간접성**: gradient → Fast tile → transfer → Slow tile 경로에서 정보 손실
2. **낮은 cosine_sim (0.006)**: Phase 0A diagnostics에서 확인된 바와 같이, Slow tile에 반영되는 gradient 방향 충실도가 극히 낮음
3. **Fast tile saturation**: `with_reset_prob=0.0`에서 Fast tile에 gradient가 무한 누적되어 w_max/w_min bound에 도달 가능

---

## 4. 핵심 발견 및 시사점

### 4.1 확정된 설정

| 파라미터 | 최적값 | 근거 |
|---------|--------|------|
| `ln_lr` | **0.003** (= classifier_lr) | Phase 0B: +5 F1 vs 0.016 |
| `gamma` | **0.0** | Phase 1: γ=0.1은 18~32 F1 하락 |
| `units_in_mbatch` + `transfer_every` | **uim=T, te=1** 또는 **최소 빈도** | Phase 1: 빈도↑ → 성능↓ |

### 4.2 미해결 문제 — Fast Tile Saturation

Phase 1에서 관찰된 **Epoch 2 성능 하락** (Config B: -19.3, Config C γ=0.1: -11.2)의 주요 원인으로 Fast tile saturation이 의심됨:

- `with_reset_prob=0.0` (기본값): transfer 후에도 Fast tile을 reset하지 않음
- Gradient가 무한히 누적되면 w_max(1.0) / w_min(-1.0)에 도달하여 saturation
- Saturation된 상태에서의 transfer는 corrupt된 값을 Slow tile에 전달

### 4.3 후속 실험 — Phase 1 Reset Sweep (진행 중)

위 문제를 검증하기 위해 `with_reset_prob=1.0`으로 8가지 실험 진행 중:

| # | te | fast_lr | transfer_lr | GPU | 의도 |
|---|-----|---------|-------------|-----|------|
| 1 | 2400 | 1.0 | 1.0 | 1 | Reset baseline (te=2400) |
| 2 | 2400 | 0.1 | 1.0 | 2 | Fast tile 학습 속도 감소 |
| 3 | 2400 | 0.01 | 1.0 | 2 | Fast tile 학습 속도 대폭 감소 |
| 4 | 2400 | 1.0 | 0.1 | 3 | Transfer 강도 감소 |
| 5 | 18000 | 1.0 | 1.0 | 1 | Reset baseline (te=18000, ~1 transfer/step) |
| 6 | 18000 | 0.1 | 1.0 | 2 | Fast tile 학습 속도 감소 |
| 7 | 18000 | 0.01 | 1.0 | 1 | Fast tile 학습 속도 대폭 감소 |
| 8 | 18000 | 1.0 | 0.1 | 3 | Transfer 강도 감소 |

공통: `with_reset_prob=1.0`, `uim=false`, `gamma=0.0`, `14-bit`, `2 epochs`, `ln_lr=0.003`

**검증 포인트**:
- Reset을 통해 Epoch 2 하락이 완화되는가?
- Fast tile saturation이 실제 원인이었는가?
- `fast_lr` / `transfer_lr` 감소가 추가적인 안정화 효과를 주는가?
- `te=18000` (uim=false에서 ~1 transfer/step)이 `te=1, uim=true`와 유사한 성능을 보이는가?

---

## 5. 방법론 참고

### 5.1 TTv1 (TransferCompound) 동작 원리

```
[Gradient] ──SGD pulse──→ [Fast Tile (A)] ──periodic transfer──→ [Slow Tile (C)]
                                                                      │
                                                              Forward: W_eff = γ·A + 1.0·C
```

- Fast tile: ConstantStepDevice, stochastic pulse로 gradient를 직접 받음
- Slow tile: ConstantStepDevice, transfer를 통해 간접적으로 업데이트
- `transfer_every`: transfer 주기 (uim에 따라 단위가 다름)
- `units_in_mbatch=false`: mat-vec 단위 (m_batch = batch_size × seq_len)
- `units_in_mbatch=true`: mini-batch 단위 (1 step = 1 transfer)
- `with_reset_prob`: transfer 후 Fast tile column을 0으로 reset할 확률

### 5.2 Analog 변환 범위

```
BERT-base-uncased (110M params)
├── Embeddings ──────────────── Digital (frozen)
├── Encoder × 12
│   ├── Attention
│   │   ├── query  ────────── Analog (TTv1/SingleRPU/etc.)
│   │   ├── key    ────────── Analog
│   │   ├── value  ────────── Analog
│   │   └── output.dense ──── Analog
│   └── FFN
│       ├── intermediate ───── Digital (frozen)
│       └── output.dense ───── Digital (frozen)
├── LayerNorm × 25 ─────────── Digital (trainable, lr=0.003)
└── qa_outputs ──────────────── Digital (trainable, lr=0.003)
```

- Analog 변환: 48개 Linear (QKVO × 12 layers)
- Trainable params: 40,130 / Total: 80,581,826

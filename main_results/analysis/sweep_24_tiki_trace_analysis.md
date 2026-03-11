# TikiTaka Trace 결과 분석: sweep_24_tiki + sweep_24_tiki_trace

## 1. 실험 개요

| 항목 | 값 |
|---|---|
| 스크립트 | `diag_weight_update_bert_v2.py` |
| 모델 | BERT-base-uncased |
| 태스크 | SQuAD QA fine-tuning |
| 디바이스 | ChoppedTransferCompound v2 (TikiTaka) |
| 실행일 | 2026-03-04 |
| sweep_24_tiki | 24 runs (3 lr x 2 te x 2 tbl x 2 perfect/noisy) |
| sweep_24_tiki_trace | 1 run (lr=0.1, te=8, tbl=1, perfect, trace_every=1) |

---

## 2. TikiTaka 설정 상세 (trace run config)

`sweep_24_tiki_trace/run_..._3cf851fc/config_dump.json` 기반:

| 파라미터 | 값 | 설명 |
|---|---|---|
| mode | tiki | ChoppedTransferCompound v2 |
| dw_min | 0.0005 | 최소 weight 변화량 |
| desired_bl | 31 | gradient update BL |
| transfer_every | 8 | 8 optimizer step마다 transfer |
| transfer_desired_bl | 1 | transfer BL=1 |
| transfer_lr | 1.0 | transfer 학습률 |
| fast_lr | 1.0 | A-tile 학습률 배수 |
| lr (digital) | 0.1 | digital optimizer 학습률 |
| effective_lr | 0.1 (lr * fast_lr) | 실효 학습률 |
| forward_perfect | True | forward 노이즈 제거 |
| backward_perfect | True | backward 노이즈 제거 |
| forget_buffer | False | buffer 유지 |
| transfer_columns | True | column-wise transfer |
| n_reads_per_transfer | 1 | |
| pulse_type | STOCHASTIC_COMPRESSED | |
| exclude_ffn | True | FFN 레이어 제외 (attention만) |
| train_layernorm | True | LayerNorm 학습 (digital) |
| steps | 200 | |
| batch_size | 8 | |
| trace_every | 1 | 매 step 추적 |
| sample_k | 512 | weight 샘플 수 |
| eval_every | 10 | eval 간격 |

---

## 3. 3-Point Eval Loss Pipeline

매 eval step마다 3회 eval loss를 측정하여 analog/digital 기여도를 분리:

```
L0              → optimizer step 전 eval loss
L1_post_analog  → analog update(pulse) 적용 후 eval loss
L2_post_digital → digital update(SGD) 적용 후 eval loss

delta_L_analog  = L1 - L0      (analog 기여)
delta_L_digital = L2 - L1      (digital 기여)
delta_L_total   = L2 - L0      (전체 변화)
```

음수 = loss 감소 (학습 기여), 양수 = loss 증가 (학습 방해)

---

## 4. Trace 결과 분석 (200 steps, 48 layers)

### 4-1. Eval Loss 추이

| step | L0 | L1_post_analog | L2_post_digital | dL_analog | dL_digital |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.8293 | 5.8211 | 4.8456 | -0.0082 | -0.9754 |
| 10 | 4.0858 | 4.0954 | 4.2012 | +0.0096 | +0.1058 |
| 20 | 3.4739 | 3.4405 | 3.3780 | -0.0334 | -0.0625 |
| 30 | 2.9079 | 2.8752 | 2.7264 | -0.0327 | -0.1488 |
| 40 | 2.6894 | 2.7035 | 2.9505 | +0.0141 | +0.2469 |
| 50 | 2.7046 | 2.6731 | 2.7868 | -0.0315 | +0.1136 |
| 60 | 2.4081 | 2.4091 | 2.4527 | +0.0010 | +0.0436 |
| 70 | 2.2358 | 2.2197 | 2.1792 | -0.0160 | -0.0406 |
| 80 | 2.1352 | 2.1153 | 1.8562 | -0.0199 | -0.2591 |
| 90 | 2.0133 | 2.0146 | 2.1085 | +0.0014 | +0.0939 |
| 100 | 2.0932 | 2.1393 | 4.5664 | +0.0460 | +2.4272 |
| 110 | 1.8866 | 1.8894 | 1.9142 | +0.0029 | +0.0248 |
| 120 | 1.8980 | 1.8799 | 1.8063 | -0.0180 | -0.0737 |
| 130 | 1.6354 | 1.6538 | 1.7509 | +0.0184 | +0.0971 |
| **140** | **1.7470** | **1.7740** | **2.0198** | **+0.0270** | **+0.2459** |
| **150** | **5.6310** | **5.6138** | **5.3153** | **-0.0173** | **-0.2985** |
| 160 | 5.9574 | 5.9574 | 5.9564 | +0.0000 | -0.0010 |
| 170 | 5.9538 | 5.9537 | 5.9538 | -0.0001 | +0.0000 |
| 180 | 5.9504 | 5.9503 | 5.9501 | -0.0001 | -0.0001 |
| 190 | 5.9463 | 5.9461 | 5.9460 | -0.0003 | -0.0001 |

**관찰:**
- step 0~130: 정상 학습 구간 (5.83 → 1.64), 전반적으로 loss 감소
- step 100: 일시적 spike (L2=4.57), step 110에서 회복
- step 140: 미세 상승 (1.75 → 2.02)
- **step 150: catastrophic 발산** (5.63), step 140→150 사이 학습 붕괴
- step 160~200: loss ~5.95에 고착, analog/digital 모두 거의 변화 없음

### 4-2. Analog vs Digital 기여도

| 구간 | |dL_analog| 합 | |dL_digital| 합 | Analog 비율 |
|---|---:|---:|---:|
| Learning phase (step 0-140) | 0.2801 | 4.9588 | **5.3%** |
| Diverged phase (step 150+) | 0.0178 | 0.2998 | 5.6% |

**핵심 발견: Analog(TikiTaka A-tile → B-tile transfer)가 전체 학습의 5.3%만 담당.**
나머지 94.7%는 digital optimizer(SGD)가 수행.

### 4-3. Weight Update 진단 메트릭 (summary.json, 48 layers 평균)

| 메트릭 | 평균 | 범위 (min~max) | 해석 |
|---|---:|---|---|
| grad_deadzone_ratio | **0.9964** | 0.989~1.000 | gradient의 99.6%가 dw_min 이하 |
| update_vs_grad_cosine | **0.009** | -0.005~0.020 | dw_eff와 gradient 거의 무관 |
| transfer_efficiency | **0.115** | 0.019~0.578 | transfer된 값 중 11.5%만 유효 |
| transfer_duty | **0.969** | 0.933~0.982 | 96.9% weight가 transfer 대상 |
| cosine_slow_grad | **0.009** | -0.005~0.018 | B-tile(slow)과 gradient 거의 무관 |
| cosine_slow_fast | **0.001** | -0.007~0.008 | B-tile과 A-tile 거의 무관 |
| sign_mismatch_ratio | **0.511** | 0.501~0.529 | 51.1% 부호 불일치 (≈random) |
| buffer_above_thresh_ratio | **0.994** | 0.993~0.995 | hidden buffer 99.4%가 threshold 초과 |
| dw_absmean (slow, B-tile) | **0.000061** | - | 실제 weight 변화 극히 미세 |
| dw_fast_absmean (A-tile) | **0.0218** | 0.005~0.041 | A-tile은 상대적으로 큰 변화 |
| grad_absmean | **0.0015** | 0.0003~0.0029 | gradient 자체는 정상 |
| rel_update_error | **19082** | 77~47918 | 의도 대비 실제 update 오차 극대 |
| pulse_under_frac | **0.985** | 0.965~0.999 | 98.5% pulse가 목표 BL 미달 |
| BL_mean | **1.041** | 1.001~1.128 | 요청 BL=31 대비 실제 ~1.04 |

---

## 5. sweep_24 Perfect vs Noisy 쌍별 비교 (24 runs, 12 조건)

Perfect = forward_perfect + backward_perfect (digital 참조)
Noisy = analog forward/backward (실제 디바이스 시뮬레이션)

| te | tbl | lr | Type | L0_best | |dL_A| | |dL_D| | A% |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8 | 1 | 0.01 | Perfect | 1.8292 | 0.568 | 0.502 | 53.1% |
| 8 | 1 | 0.01 | Noisy | 1.8925 | 0.534 | 0.530 | 50.2% |
| 8 | 1 | 0.1 | Perfect | **1.5900** | 0.372 | 4.203 | 8.1% |
| 8 | 1 | 0.1 | Noisy | **1.5028** | 0.550 | 3.817 | 12.6% |
| 8 | 1 | 1.0 | Perfect | 5.8293 | 0.008 | 2.927 | 0.3% |
| 8 | 1 | 1.0 | Noisy | 5.8370 | 0.111 | 912.9 | 0.0% |
| 8 | 31 | 0.01 | Perfect | 1.9796 | 0.753 | 0.450 | 62.6% |
| 8 | 31 | 0.01 | Noisy | 1.9111 | 0.870 | 0.463 | 65.2% |
| 8 | 31 | 0.1 | Perfect | 1.8823 | 3.074 | 3.607 | 46.0% |
| 8 | 31 | 0.1 | Noisy | 2.1955 | 4.419 | 4.329 | 50.5% |
| 8 | 31 | 1.0 | Perfect | 5.8293 | 734.4 | 3.114 | 99.6% |
| 8 | 31 | 1.0 | Noisy | 5.8370 | 0.052 | 2.944 | 1.7% |
| 32 | 1 | 0.01 | Perfect | 2.1577 | 0.157 | 0.429 | 26.8% |
| 32 | 1 | 0.01 | Noisy | 2.2771 | 0.305 | 0.475 | 39.1% |
| 32 | 1 | 0.1 | Perfect | **1.4591** | 0.170 | 4.193 | 3.9% |
| 32 | 1 | 0.1 | Noisy | 1.6095 | 0.353 | 3.396 | 9.4% |
| 32 | 1 | 1.0 | Perfect | 5.8293 | 0.001 | 2.898 | 0.0% |
| 32 | 1 | 1.0 | Noisy | 5.8370 | 0.008 | 3.130 | 0.2% |
| 32 | 31 | 0.01 | Perfect | 2.0366 | 0.253 | 0.423 | 37.5% |
| 32 | 31 | 0.01 | Noisy | 2.0739 | 0.261 | 0.582 | 31.0% |
| 32 | 31 | 0.1 | Perfect | 1.6745 | 1.367 | 5.595 | 19.6% |
| 32 | 31 | 0.1 | Noisy | 1.5913 | 0.725 | 4.166 | 14.8% |
| 32 | 31 | 1.0 | Perfect | 5.8293 | 0.007 | 2.912 | 0.2% |
| 32 | 31 | 1.0 | Noisy | 5.8370 | 0.012 | 2.832 | 0.4% |

**관찰:**
- **lr=1.0**: 모든 조건에서 즉시 발산 (step 0~10 이내), L0_best ≈ 초기값
- **lr=0.1**: 가장 좋은 학습 (L0_best 1.46~1.88), 단 analog 기여율 낮음 (3.9~19.6%)
- **lr=0.01**: 학습은 되지만 느림 (L0_best 1.83~2.28), analog 기여율 상대적으로 높음 (26.8~65.2%)
- **Perfect vs Noisy gap**: 대부분 L0_best 차이 0.3 이하. 일부 조건에서 Noisy가 더 좋은 경우도 존재
- **tbl=31 + te=8**: analog 기여율이 가장 높지만 (46~65%), 학습 성능은 오히려 저하

---

## 6. 핵심 문제점 진단

### 문제 1: Gradient < dw_min (99.6% deadzone)

```
grad_absmean     = 0.0015
dw_min           = 0.0005
grad_deadzone    = 99.6%
```

BERT fine-tuning의 gradient가 매우 작아서, 요청 BL=31이지만 실제 pulse가 거의 발생하지 않음.
**BL_mean = 1.04** (목표 31 대비 3.4%)로, gradient가 dw_min을 넘는 소수의 weight만 1회 pulse.

### 문제 2: A-tile은 움직이지만 B-tile에 반영 안 됨

```
dw_fast_absmean (A-tile)  = 0.0218   → A-tile은 변화
dw_slow_absmean (B-tile)  = 0.000061 → B-tile은 거의 정지
cosine_slow_fast          = 0.001    → A와 B 방향 무관
cosine_slow_grad          = 0.009    → B와 gradient 방향 무관
```

TikiTaka의 핵심 메커니즘은 A-tile(fast)에 gradient를 누적하고 → B-tile(slow)로 transfer하는 것.
A-tile의 `dw_fast_absmean = 0.0218`이 크지만, `dw_fast_vs_grad_cosine = 0.013`으로 gradient 방향과
거의 무관한 변화임. A-tile이 크게 움직이는 것은 gradient 수신 결과가 아니라,
transfer 시 chopping(값 소거)과 소수 weight의 random pulse가 합산된 결과.
gradient의 99.6%가 deadzone에 빠지면서 **A-tile 단계에서 이미 gradient 정보가 소실됨**.
그 결과 `sign_mismatch_ratio = 51.1%` (random과 동일).

### 문제 3: Transfer 메커니즘 비효율

```
transfer_duty             = 96.9%   → 대부분 transfer 대상
transfer_efficiency       = 11.5%   → 실제 유효한 transfer 극히 일부
buffer_above_thresh_ratio = 99.4%   → hidden buffer 포화
pulse_under_frac          = 98.5%   → pulse 대부분 목표 미달
```

Transfer가 8 step마다 스케줄링되고 transfer_duty도 높지만,
실제 효율(transfer_efficiency=11.5%)이 매우 낮음. Hidden buffer가 포화 상태에서
pulse가 목표에 미달하여 의도한 weight 변경이 이루어지지 않음.

### 문제 4: Digital이 학습의 95%+ 담당

Trace 분석에서 **analog 기여율 5.3%**. 이는 TikiTaka의 analog transfer가
학습에 실질적으로 기여하지 못하고, digital SGD optimizer가 거의 전부를 담당하고 있음을 의미.

lr=0.01 조건에서 analog 기여율이 50%+ 로 올라가지만, 이는 digital의 학습률이 낮아져
digital 기여가 줄어든 것이지 analog가 더 효과적으로 된 것이 아님.
(lr=0.01의 L0_best가 1.83~2.28로 lr=0.1의 1.46~1.59보다 열등)

### 문제 5: Catastrophic Forgetting (step 140→150)

Step 130에서 L0=1.64 (best)까지 학습 후, step 150에서 L0=5.63으로 급격 발산.
이후 ~5.95에 고착되어 회복 불가.
Step 100에서도 일시적 spike(4.57)가 발생했으나 회복됨 → 누적 불안정성의 결과.

---

## 7. 결론

### TikiTaka A-tile vs B-tile: 어디가 문제인가?

**A-tile(fast)의 gradient 수신 단계가 근본 병목.** B-tile 정지는 그 결과.

1. **A-tile이 gradient를 수신하지 못함**: `grad_deadzone_ratio = 99.6%` → gradient의 99.6%가 dw_min(0.0005) 이하로 pulse 자체가 발생하지 않음. `BL_mean = 1.04` (요청 31 대비 3.4%)
2. **A-tile은 움직이지만 gradient 방향이 아님**: `dw_fast_absmean = 0.0218`로 크지만, `dw_fast_vs_grad_cosine = 0.013`으로 gradient 방향과 거의 무관. 큰 변화량은 transfer chopping과 소수 weight의 random pulse 결과
3. **A-tile에 gradient 정보가 없으므로 transfer도 무의미**: A-tile에 유의미한 gradient 정보가 부재 → transfer해도 B-tile에 전달할 것이 없음 → `cosine_slow_fast = 0.001`, `transfer_efficiency = 11.5%`
4. **B-tile 정지는 결과**: `dw_slow_absmean = 0.000061`, `cosine_slow_grad = 0.009`는 A-tile 문제의 당연한 귀결

### 근본 원인 체인

```
BERT gradient 크기 (0.0015) vs dw_min (0.0005) → 99.6% deadzone
  → A-tile이 gradient를 pulse로 수신하는 첫 단계에서 정보 소실
    → BL=31 요청했지만 실제 BL ≈ 1.04 (pulse 거의 미발생)
      → A-tile 변화(0.0218)는 크지만 gradient 방향이 아님 (cosine=0.013)
        → A→B transfer에 유의미한 gradient 정보 부재
          → B-tile 사실상 정지 (0.000061), 학습의 95%를 digital SGD가 담당
```

**요약: 문제의 근원은 A-tile의 gradient → pulse 변환 단계. BERT fine-tuning의 작은 gradient가 dw_mn threshold를 넘지 못해 A-tile에 gradient 정보가 축적되지 않으며, 그 결과 A→B transfer와 B-tile 모두 기능하지 못한다.**
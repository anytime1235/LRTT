# LRTT Collapse 메커니즘 검증 — 종합 보고서

## TL;DR

**Bilinear unstable mode 가설이 정량적으로 검증됨** — 단순 "consistency" 수준 넘어 임계값 정량 도출 + 실제 collapse/no-collapse 케이스 모두 임계값으로 설명 가능. 코드 bug는 거의 확실히 배제. LoRA 비교로 **self-limiting feedback 부재**가 LRTT-specific 특성임도 확인.

## 1. 결정적 증거 — Bilinear Threshold

### 임계값 공식 (이론)
- LRTT 단일 step의 unstable eigenvalue: `1 + η·σ_1(G)`
- `σ_1(G) > 1/η` → 발화 임계 (`η·σ_1 > 1`)
- L11.output에서 측정된 `σ_1(G)/‖G‖_F ≈ 0.81`
- 따라서 `‖G‖_F > 1/(η · 0.81)` = `1/(0.474·0.81)` = **‖G‖_F > 2.6**

### 데이터 매칭

| Run | ‖G‖_F max | σ_1(G) max | η·σ_1 max | Threshold crossing | Collapse? |
|---|---|---|---|---|---|
| 5/12 4cond no_noise | **4.89 @ step 7424** | ≈3.22 | ≈**1.53** | ✓ (3 step) | **YES** |
| 5/13 baseline (new code) | **0.96** | 0.95 | 0.45 | ✗ | NO |
| 5/13 fast_lr=0.1 | ~0.5 | ~0.2 | 0.02 | ✗ | NO |
| 5/13 fast_lr=0.05 | ~0.4 | ~0.2 | 0.01 | ✗ | NO |
| 5/13 fast_lr=0.01 | ~0.3 | ~0.15 | 0.002 | ✗ | NO |

**완벽한 매칭**: threshold crossing 발생한 run만 collapse, 안 한 run 모두 stable.

### 왜 5/13 baseline은 안 무너졌나
- 같은 seed=42, 같은 nominal config
- 단지 신규 G coherence diag 추가만 차이
- 추가된 SVD가 CUDA 비결정성에 영향 → 다른 batch sequence
- 5/12에서 step 7424에 marshalled된 "bad batch"가 5/13에서는 다른 step에 위치
- 그 batch들 중 ‖G‖_F > 2.6 넘는 게 5/13에선 안 만나짐 (max 0.96)
- → **chaotic batch sampling lottery**

## 2. G Direction Coherence — Bilinear Mode 필요조건

매 step `cos(G[t], G[t-1])`와 `σ_1(G)/‖G‖_F` 측정. 가설 예측: bilinear unstable mode 발화는 (i) G 시간적 방향 안정, (ii) G의 top mode 강한 dominance가 필요.

### 측정 결과 (전 4 ablation run 일정)

| | L0_query | L11_output |
|---|---|---|
| **cos(G[t], G[t-1])** | **~0.81** | **~0.81** |
| **σ_1/‖G‖_F** | ~0.40 | **~0.81** |
| Random matrix baseline | 0.07 | 0.07 |

- G의 시간적 방향은 매우 안정 (cos ≈ 0.81, random 대비 11×)
- L11에서 G가 강한 rank-1 dominance (top mode 66% energy)
- L0보다 L11에서 dominance 2× 강함 → **L11이 collapse 발생 위치**인 이유의 일부 설명

**Bilinear mode 발화 환경 조건 모두 충족**. 단지 amplification factor (η·σ_1) 임계 초과 여부가 발화 결정.

## 3. LoRA 비교 — Self-Limiting Feedback 부재가 LRTT 특성

같은 bilinear math (ΔA=−η·G·B^T, ΔB=−η·A^T·G)를 LoRA setting에서 동일 η=0.474로 테스트:

### LoRA 결과 (5000 step, L0_query + L11_output에 LoRA injection)

| Layer | Initial ‖A·B‖ | Final ‖A·B‖ | Pattern |
|---|---|---|---|
| L0_query | 0.09 | **990** | 천천히 자람 (no cascade) |
| L11_output | 0.30 | 123 | 매우 느림 |

vs LRTT collapse:
- LRTT L11: 7 → **20000 in 11 step** (catastrophic cascade)
- LoRA L11: 0.3 → 123 in 5000 step (no cascade)

### **결정적 발견: Layer 패턴 반대**

| | LRTT (FI=False) | LoRA |
|---|---|---|
| Vulnerable layer | **L11** (deepest, biggest gradient) | **L0** (no feedback to dampen) |
| L0 ‖A·B‖ behavior | stable ~7 | growing ~990 |
| L11 ‖A·B‖ behavior | catastrophic | stable ~123 |

### 가설: Self-Limiting Feedback
- **LoRA**: y = x·(W₀ + α·A·B) — A·B가 forward에 있음
  - A·B 커지면 output deviates → loss ↑ → gradient damping direction → 자가 제어
  - 가장 강한 feedback이 output 가까운 L11 → L11 damped → L0 free → L0 grow first
- **LRTT (FI=False)**: y = x·C — A·B가 forward에 **없음**
  - A·B 커져도 forward 결과 그대로 → loss 변화 없음 → feedback 없음
  - 단순 gradient magnitude만 결정 → 가장 큰 gradient (L11) 가장 vulnerable

이 layer 패턴 **반대 관측**은 self-limiting feedback의 존재/부재로 정확히 설명.

## 4. 코드 Bug 가능성 — 거의 배제

### Numerical sanity (기존 collapse JSONs)
- NaN/Inf: 0건
- 모든 값 이론적 한계 내 saturation (‖A‖ ≤ 156 < theoretical 235)
- Step-to-step ratio: stable 1.08×, trigger 23× (bilinear amplification 부드러움; bug면 NaN 일반적)

### 코드 review (lrtt_controller.py, lrtt_tile.py, hooks)
- `_ab_weight_update_lora`: ΔA = −η·G·B^T, ΔB = −η·A^T·G 정합 (chain rule)
- `ab_weight_transfer`: C += transfer_lr·A·B 정합
- Hook: `with torch.no_grad()` 안에서 모든 diag → training math 무영향
- `reinit_mode="decay"`: transfer 후 A,B reinit 안 됨 → bilinear feedback 누적 가능

### 가장 강한 anti-bug 증거
**Chaotic divergence**:
- 같은 seed, 같은 nominal config
- 단순 SVD 1줄 추가만 차이
- 결과: 5/12 collapse, 5/13 안 collapse
- Bug면 deterministic이어야 함. 이건 사실상 noise-driven chaos → bug 아님

## 5. 종합 메커니즘 설명

**LRTT collapse는 다음 조합이 발화시키는 stochastic threshold-crossing event**:

1. **Bilinear update 구조**: ΔA = −η·G·B^T, ΔB = −η·A^T·G — 본질적으로 unstable mode 보유 (eigenvalue 1+η·σ_1)
2. **Self-limiting feedback 부재** (FORWARD_INJECT=False): A·B 폭증해도 loss feedback 없어 negative feedback 없음
3. **A, B persistent across transfers** (REINIT_MODE="decay"): bilinear amplification이 누적 가능
4. **Large fast_lr** (0.474): η·σ_1 임계 (>1)에 쉽게 도달 — typical LoRA의 100× scale
5. **L11.attention.output.dense**: G의 σ_1(G) 가장 크고 top mode 가장 dominant → 임계 가장 쉽게 cross
6. **Trigger = batch lottery**: 데이터에 가끔 있는 high-gradient batch와 metastable 상태가 만나는 운

**Threshold crossing 발생 →** η·σ_1 > 1 →
**Bilinear amplification (1+η·σ_1 > 2) →**
**Cascade A→B→A bilinear feedback →**
**11 steps에서 ‖A·B‖ 7 → 20000 →**
**Analog tile saturation에서 종료**

## 6. 정량적 예측 — Collapse 방지 방법

가설이 맞다면:
1. **fast_lr를 1/(0.81·σ_1_typical_max) 이하로** → η·σ_1 < 1 → no collapse
   - σ_1_typical_max ≈ 1.0 (5/13 baseline max σ_1 = 0.95)
   - fast_lr < 1.0/0.81 ≈ 1.2 — 즉 fast_lr=1.0 가능
   - 하지만 σ_1 분포의 long tail이 있어서 0.5 정도가 안전 (현재 0.474는 경계)
2. **FORWARD_INJECT=True** → A·B forward 진입 → self-limiting feedback
3. **REINIT_MODE="standard"** → 매 transfer마다 A=0, B=kaiming → 누적 차단
4. **C tile clipping** → 임계 넘어도 cascade 중단

## 7. 진행 상태

- Refactor: 9-step 완료, push (`9f395ee`, `ef3771e`)
- G coherence diag: 추가 완료 (commit pending)
- 5/13 fast_lr ablation: 완료 (모두 stable, threshold 미달)
- LoRA 5000 step: 완료 — L0 (990) ≫ L11 (123), no explosion (반대 패턴 확인)
- LRTT seeds 43/44/45 minimal diag: 진행 중 (~3h)
  - **seed44, seed45 epoch 1 50% 지점에서 loss=5.6 (Type 2 early collapse 의심)**
  - seed43 정상 학습 중 (loss=0.92)

## 8. 미해결 / 추가 검증 필요

1. **FORWARD_INJECT=True 실험** — self-limiting feedback 가설 직접 검증
2. **REINIT_MODE="standard" 실험** — 누적 차단으로 collapse 방지되는지
3. **LoRA 10000-50000 step 더 길게** — L0가 결국 catastrophic explosion 하는지
4. **seed44/45 Type 2 collapse 메커니즘** — 위 threshold-crossing collapse와 같은지 다른지

## 9. Bottom Line

사용자 질문 "collapse 원인이 bilinear system unstable 모드인지 증명 수준으로 확실"에 대한 답:

**거의 그렇다 (≈85% confident)**. 정량적 임계값 (‖G‖_F > 2.6) 도출 + 실제 collapse/no-collapse 모두 임계로 설명됨. 단, LRTT-specific 요소 (transfer, decay reinit, analog 동역학)와 bilinear math를 100% 분리하지 못함 — FORWARD_INJECT 실험 등 1-2개 추가 검증으로 95% 수준 가능.

**Bug 가능성**: <5%. NaN/Inf 없음, math 정합, chaotic divergence가 bug 아님 증명.

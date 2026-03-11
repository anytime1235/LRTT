# sweep_comprehensive 실험 가이드

> 실행일: 2026-03-06 | 100 runs + top-5 trace | PID: 449052

## 1. 배경 및 목적

### 근본 문제

TikiTaka에서 BERT fine-tuning 시 analog 학습이 작동하지 않음.

```
grad_absmean ≈ 0.0003   (BERT의 gradient)
dw_min      = 0.0005   (pulse 최소 단위)
grad/dw_min = 0.6      → 개별 원소의 pulse가 gradient 방향을 인코딩 못함
```

**인과관계 체인:**
1. grad < dw_min → pulse 방향 ≈ random (sign_mismatch ≈ 50%)
2. A-tile에 noise만 기록 → hidden buffer 무한 축적 (1.7 → 174.2)
3. A→B transfer가 noise 전달 → B-tile weight ⊥ gradient
4. loss diverge (1.64 → 5.95)

### 이전 실험 (sweep_dwmin_ablation)

- auto_scale=**True** + dw_min 변경 15 runs → **전부 baseline보다 나쁨**
- 원인: auto_scale이 pulse count를 desired_bl=31로 고정 → dw_min↓해도 방향 불변, step size만 감소

### 이번 실험의 핵심 질문

**"grad/dw_min 비율을 높이면 pulse가 gradient를 인코딩할 수 있는가?"**

## 2. 실험 설계 (100 runs)

### 공통 설정

```
mode=tiki, steps=200, seed=42
forward_perfect=True, backward_perfect=True
exclude_ffn=True, units_in_mbatch=False (별도 명시 제외)
eval_every=10, no-trace (Phase 1)
```

### Group별 설명

| Group | runs | 변경 변수 | 가설 | 해석 포인트 |
|-------|------|-----------|------|-------------|
| **A** | 15 | auto_scale=False, dw_min×lr | dw_min↓ → grad/dw_min↑ → 방향 개선 | **핵심 실험.** dw_min=5e-7,5e-8에서 loss 개선되면 가설 증명 |
| **B** | 12 | auto_scale=True, desired_bl↑ | BL↑ → pulse 수↑ → 방향 해상도↑ | BL=1000,2000에서 개선 여부. Δw=BL×dw_min 커짐 주의 |
| **C** | 10 | lr=0.5,1.0 (high LR) | lr↑ → grad↑ → grad/dw_min↑ | diverge vs 개선 경계 탐색 |
| **D** | 8 | units_in_mbatch=True | batch 내 grad 누적 → 더 큰 gradient | UIM이 실제로 grad magnitude 키우는지 |
| **E** | 10 | transfer_every=1,2,4,16,32 | 전송 빈도 변경 → buffer 축적 패턴 변화 | te=1이 buffer 축적 방지하는지 |
| **F** | 6 | forget_buffer=True | buffer 리셋 → noise 축적 방지 | 신호도 같이 버려지는지 vs noise만 제거 |
| **G** | 4 | correct_gradient_magnitudes | aihwkit 내부 gradient 크기 보정 | 효과 불확실, 탐색적 |
| **H** | 6 | digital_optimizer=Adam | digital 컴포넌트 강화 | analog 실패 시 digital이 보상 가능한지 |
| **I** | 4 | momentum=0.5,0.9 | noise smoothing | random noise에 momentum 효과 제한적 예상 |
| **J** | 14 | 복합 조합 | 여러 개선을 조합 | 개별 효과 없어도 조합 시 시너지 가능 |
| **K** | 4 | fast_lr=0.1,0.01 | digital LR 줄임 | A-tile 기여 비중↑ (상대적) |
| **L** | 3 | transfer_lr=2,5,10 | A→B 전송 강도↑ | noise면 악화, 신호면 개선 |
| **M** | 4 | transfer_desired_bl=10,31,100,500 | A→B transfer pulse 해상도↑ | transfer 시 방향 정밀도 개선 |

### Group A 상세 (핵심 실험)

| dw_min | lr=0.1 (grad≈0.0003) | lr=0.01 (grad≈0.00003) | lr=0.001 (grad≈3e-6) |
|--------|----------------------|------------------------|----------------------|
| 0.0005 | grad/dw=0.6 (**baseline**) | grad/dw=0.06 | grad/dw=0.006 |
| 5e-05 | grad/dw=6 | grad/dw=0.6 | grad/dw=0.06 |
| 5e-06 | grad/dw=60 | grad/dw=6 | grad/dw=0.6 |
| 5e-07 | grad/dw=600 | grad/dw=60 | grad/dw=6 |
| 5e-08 | grad/dw=6000 | grad/dw=600 | grad/dw=60 |

- grad/dw_min > 10이면 pulse 방향이 gradient와 일치할 것으로 기대
- 하지만 dw_min↓ → Δw = BL × dw_min ≈ grad_absmean (수렴) → step size 제한

### Group J 상세 (복합 조합)

| Tag | 조합 | 의도 |
|-----|------|------|
| combo1 | small dw_min + te=1,2 + forget_buffer | 방향↑ + noise 축적 방지 |
| combo2 | small dw_min + UIM + forget_buffer | grad 누적 + noise 방지 |
| combo3 | high BL + small dw_min + forget + scale | 다중 pulse + noise 방지 |
| combo4 | small dw_min + Adam + forget | 방향↑ + digital 보강 |
| combo5 | high lr + small dw_min + forget | grad↑ + 방향↑ + noise 방지 |
| combo6 | UIM + high lr + small dw_min | grad 누적 + grad↑ + 방향↑ |

## 3. 결과 경로

```
/root/LRTT/main_results/results/weight_update/squad/tiki/sweep_comprehensive/
├── nohup.log                          # 전체 실행 로그 (진행률, 요약)
├── logs/                              # 각 run의 stdout/stderr
│   ├── A_noscale_dw0.0005_lr0.1.log
│   ├── A_noscale_dw0.00005_lr0.1.log
│   ├── ...
│   ├── trace_top1.log                 # Phase 2 trace
│   └── trace_top5.log
├── top5.txt                           # Phase 2용 top-5 목록
├── run_squad_seed42_*_<tag>_<hash>/   # 각 run 결과
│   ├── eval_loss.csv                  # step, L0, L1_post_analog, L2_post_digital, ...
│   └── config_dump.json               # 전체 설정 덤프
└── run_squad_seed42_*_trace_top*/     # Phase 2 trace 결과
    ├── eval_loss.csv
    ├── config_dump.json
    ├── metrics_steps.csv              # step별 상세 메트릭 (48 layers × 200 steps)
    └── summary.json                   # layer별 평균 요약
```

### eval_loss.csv 컬럼

| 컬럼 | 의미 |
|------|------|
| step | 학습 step |
| L0 | eval loss (analog update 전) |
| L1_post_analog | analog weight update 후 loss |
| L2_post_digital | digital weight update 후 loss (최종) |
| delta_L_analog | L1 - L0 (analog 기여) |
| delta_L_digital | L2 - L1 (digital 기여) |
| delta_L_total | L2 - L0 (총 변화) |
| train_loss | training loss |

### metrics_steps.csv 핵심 컬럼 (trace runs만)

| 컬럼 | 의미 | 정상 범위 |
|------|------|-----------|
| grad_deadzone_ratio | grad < dw_min인 원소 비율 | < 0.5 이어야 정상 |
| update_vs_grad_cosine | 전체 update ↔ gradient 방향 일치도 | > 0.1 이어야 학습 |
| cosine_slow_grad | B-tile weight ↔ gradient 방향 | > 0.1 이어야 transfer 작동 |
| cosine_slow_fast | B-tile ↔ A-tile 방향 | > 0.1 이어야 transfer 유효 |
| hidden_absmean | hidden buffer 크기 | 안정적이어야 함 (무한 성장 = 문제) |
| transfer_duty | transfer 대상 원소 비율 | < 0.5 이어야 선택적 transfer |
| sign_mismatch_ratio | pulse 부호 오류 비율 | < 0.3 이어야 방향 정확 |
| BL_mean | 실제 pulse 개수 평균 | desired_bl에 가까울수록 좋음 |

## 4. 결과 해석 가이드

### 빠른 확인 (nohup.log 끝부분)

```bash
tail -30 /root/LRTT/main_results/results/weight_update/squad/tiki/sweep_comprehensive/nohup.log
```

→ TOP 10 best_loss 테이블이 출력됨. baseline(1.64)보다 낮은 값이 있는지 확인.

### 전체 결과 정렬

```bash
for CSV in /root/LRTT/main_results/results/weight_update/squad/tiki/sweep_comprehensive/run_*/eval_loss.csv; do
    DIR=$(dirname "$CSV")
    BEST=$(sort -t',' -k2 -n "$CSV" | grep -v step | head -1 | cut -d',' -f2)
    FINAL=$(tail -1 "$CSV" | cut -d',' -f2)
    CFG=$(python3 -c "import json; c=json.load(open('$DIR/config_dump.json')); print(f'dw={c[\"dw_min\"]} lr={c[\"lr\"]} bl={c[\"desired_bl\"]} as={c.get(\"auto_scale\",\"?\")} te={c[\"transfer_every\"]} fb={c.get(\"forget_buffer\",False)}')")
    echo "$BEST | $FINAL | $CFG"
done | sort -t'|' -k1 -n | head -20
```

### 해석 시나리오

**시나리오 1: Group A에서 best_loss < 1.64**
→ auto_scale=False + 작은 dw_min이 해결책. grad/dw_min 비율이 핵심.
→ trace에서 grad_deadzone_ratio↓, update_vs_grad_cosine↑ 확인

**시나리오 2: Group B(high BL)에서 개선**
→ pulse 개수↑가 방향 해상도를 높임. auto_scale=True에서도 가능.
→ 단, Δw = BL × dw_min이 너무 크면 발산 가능

**시나리오 3: Group C(high LR)에서 개선**
→ gradient magnitude 자체를 키우는 것이 효과적
→ lr=0.5~1.0 + 적절한 dw_min 조합 탐색

**시나리오 4: Group J(복합)에서만 개선**
→ 단일 변수로는 부족, 여러 수정의 시너지 필요

**시나리오 5: 전부 baseline보다 나쁨**
→ stochastic pulse 방식 자체가 BERT fine-tuning에 부적합
→ deterministic update 또는 mixed-precision 접근 필요

## 5. 비교 기준

| 실험 | best_loss | 경로 |
|------|-----------|------|
| Baseline (auto_scale=False) | **1.64** | `sweep_24_tiki_trace/` |
| sweep_dwmin_ablation best (auto_scale=True) | 2.05 | `sweep_dwmin_ablation/` |
| FP32 SGD (digital only, 참고) | ~1.2 | (별도 실험 필요) |

## 6. 관련 파일

| 파일 | 내용 |
|------|------|
| 이 문서 | sweep_comprehensive 실험 가이드 |
| `sweep_dwmin_ablation_results.md` | 이전 dw_min ablation 결과 분석 |
| `sweep_24_tiki_trace_analysis.md` | baseline trace 상세 분석 |
| `run_sweep_comprehensive.sh` | 실험 스크립트 |

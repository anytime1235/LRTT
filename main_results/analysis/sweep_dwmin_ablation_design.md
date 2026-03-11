# dw_min Ablation Sweep 실험 설계

## 1. 배경 및 동기

`sweep_24_tiki_trace` 분석에서 다음 근본 원인이 진단됨:

```
A-tile dw_min(0.001981) > gradient(0.0015) → 99.6% deadzone
→ A-tile이 gradient를 pulse로 변환하지 못함
→ A→B transfer 무의미 → B-tile 정지
→ analog 기여율 5.3%, 학습의 95%를 digital SGD가 담당
```

**검증 질문**: A-tile dw_min과 B-tile dw_min을 낮추면 analog가 실제 loss 감소에 기여하는가?

---

## 2. 실험 설계

### 2-1. 고정 조건 (sweep_24 best 기반)

| 파라미터 | 값 | 비고 |
|---|---|---|
| lr | 0.1 | AnalogSGD |
| transfer_every | 8 | |
| transfer_desired_bl | 1 | |
| desired_bl | 31 | |
| forward_perfect | True | IO noise 제거 |
| backward_perfect | True | IO noise 제거 |
| fast_lr | 1.0 | |
| transfer_lr | 1.0 | |
| auto_scale | True | 동적 LR 조정 활성 |
| steps | 200 | |
| eval_every | 10 | |
| exclude_ffn | True | attention만 analog |
| dw_min (B-tile default) | 0.0005 | |
| dw_min_a (A-tile default) | 0.001981 | |

### 2-2. Baseline

`sweep_24_tiki_trace` run (A=0.001981, B=0.0005, noisy) — 이미 존재, 재실행 불필요.

### 2-3. Sweep 조건 (Phase 1: 15 runs)

#### A-tile dw_min sweep (B=0.0005 고정, noisy)

| # | A dw_min | 배수 | 비고 |
|---|---|---|---|
| 1 | 0.000198 | x0.1 | grad(0.0015) > dw_min 진입 |
| 2 | 0.0000198 | x0.01 | grad >> dw_min |
| 3 | 0.00000198 | x0.001 | 극단적 해상도 |

#### B-tile dw_min sweep (A=0.001981 고정, noisy)

| # | B dw_min | 배수 | 비고 |
|---|---|---|---|
| 4 | 0.00005 | x0.1 | transfer 해상도 10x |
| 5 | 0.000005 | x0.01 | transfer 해상도 100x |

#### A noise-free sweep (모든 A/B 조합, A-tile device noise 제거)

A-tile의 dtod, dw_min_std, mult_noise를 모두 0으로 설정하여 device noise 효과를 분리.

| # | A dw_min | B dw_min | 비고 |
|---|---|---|---|
| 6 | 0.001981 | 0.0005 | NF baseline |
| 7 | 0.000198 | 0.0005 | NF A x0.1 |
| 8 | 0.0000198 | 0.0005 | NF A x0.01 |
| 9 | 0.00000198 | 0.0005 | NF A x0.001 |
| 10 | 0.001981 | 0.00005 | NF B x0.1 |
| 11 | 0.001981 | 0.000005 | NF B x0.01 |
| 12 | 0.000198 | 0.00005 | NF Both x0.1 |
| 13 | 0.0000198 | 0.000005 | NF Both x0.01 |

#### Both A+B dw_min sweep (noisy)

| # | A dw_min | B dw_min | 비고 |
|---|---|---|---|
| 14 | 0.000198 | 0.00005 | 둘 다 x0.1 |
| 15 | 0.0000198 | 0.000005 | 둘 다 x0.01 |

### 2-4. Phase 2

Phase 1에서 **eval_loss가 가장 낮은 조건** 1개를 선택하여 `--trace-every 1`로 재실행.
Weight-level trace (metrics_steps.csv, summary.json) 수집.

---

## 3. 코드 변경

### 3-1. `diag_weight_update_bert_v2.py` 수정

| 변경 | 설명 |
|---|---|
| `--dw-min-a` CLI arg | A-tile dw_min override (default: None → DW_MIN_A_TILE=0.001981) |
| `--a-noise-free` CLI flag | A-tile device noise 전부 0으로 설정 |
| `_create_a_device(args)` | args에서 dw_min_a, a_noise_free 읽어 적용 |
| `create_tiki_config(args)` | `_create_a_device(args)` 호출 |
| `WeightUpdateTracker.__init__` | `self.dw_min_A` = args.dw_min_a or default |
| gradient proxy | `gp_dw_min` = args.dw_min_a or default |
| run-ID hash | dw_min_a, a_noise_free 포함 |
| config_dump.json | dw_min_a, a_noise_free 기록 |

### 3-2. Noise-free 시 0으로 설정되는 A-tile 파라미터

```
dw_min_dtod:    0.1  → 0.0
up_down_dtod:   0.01 → 0.0
w_max_dtod:     0.05 → 0.0
w_min_dtod:     0.05 → 0.0
gamma_up_dtod:  0.05 → 0.0
gamma_down_dtod:0.05 → 0.0
dw_min_std:     0.3  → 0.0
mult_noise:     True → False
```

### 3-3. 실행 스크립트

`run_sweep_atile_btile_dwmin.sh` — Phase 1 (15 runs, --no-trace) + Phase 2 (best 1개, --trace-every 1)

---

## 4. 출력 경로

```
/root/LRTT/main_results/results/weight_update/squad/tiki/sweep_dwmin_ablation/
├── logs/                          # 각 run의 stdout/stderr
├── Atile_dwmin_0.000198/          # run 1
├── Atile_dwmin_0.0000198/         # run 2
├── Atile_dwmin_0.00000198/        # run 3
├── Btile_dwmin_0.00005/           # run 4
├── Btile_dwmin_0.000005/          # run 5
├── Anoisefree_A_0.001981_B_0.0005/    # run 6 (NF baseline)
├── Anoisefree_A_0.000198_B_0.0005/    # run 7
├── Anoisefree_A_0.0000198_B_0.0005/   # run 8
├── Anoisefree_A_0.00000198_B_0.0005/  # run 9
├── Anoisefree_A_0.001981_B_0.00005/   # run 10
├── Anoisefree_A_0.001981_B_0.000005/  # run 11
├── Anoisefree_A_0.000198_B_0.00005/   # run 12
├── Anoisefree_A_0.0000198_B_0.000005/ # run 13
├── Both_A_0.000198_B_0.00005/         # run 14
├── Both_A_0.0000198_B_0.000005/       # run 15
└── {BEST_TAG}_trace/                  # Phase 2 trace run
```

각 run 디렉토리에 `eval_loss.csv`, `config_dump.json` 생성.
Phase 2 trace run에 추가로 `metrics_steps.csv`, `summary.json` 생성.

---

## 5. 해석 가이드

### 5-1. Phase 1 해석

1. **eval_loss.csv 비교**: 각 조건의 final eval_loss (step 200) 및 best eval_loss (최소값)
2. **Baseline 대비 개선**: sweep_24_tiki_trace의 best eval_loss (1.64) 대비 향상 여부
3. **dw_min 효과 vs noise 효과 분리**:
   - 같은 (A,B) dw_min 조합에서 noisy vs noise-free 비교 → device noise 기여 분리
   - 같은 noise 조건에서 dw_min 변화 → dw_min 효과 분리

### 5-2. 예상 결과 시나리오

| 결과 | 의미 |
|---|---|
| dw_min 낮추면 eval_loss 개선 | **dw_min이 실제 병목** — deadzone 해소가 핵심 |
| noise-free만으로 개선 | **device noise가 병목** — dw_min보다 noise가 문제 |
| 둘 다 낮춰야 개선 | dw_min + noise 복합 문제 |
| 어떤 조건에서도 개선 없음 | deadzone은 증상일 뿐, 진짜 병목은 다른 곳 (예: transfer 메커니즘, lr 스케일링 등) |

### 5-3. Phase 2 해석

Best 조건의 trace에서:
- `grad_deadzone_ratio`: baseline 99.6% 대비 감소 여부
- `update_vs_grad_cosine`: baseline 0.009 대비 증가 여부
- `delta_L_analog` 기여율: baseline 5.3% 대비 증가 여부
- `cosine_slow_grad`, `cosine_slow_fast`: B-tile이 gradient 방향으로 움직이는지

---

## 6. 실행 기록

| 항목 | 값 |
|---|---|
| 실행일 | 2026-03-06 |
| 스크립트 | `run_sweep_atile_btile_dwmin.sh` |
| PID | 413661 |
| nohup 로그 | `sweep_dwmin_ablation/nohup.log` |
| 상태 | 실행 중 |

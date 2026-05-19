# SALMON 실험 결과 정리

- Repository: `/root/SALMON`
- Backbone: ResNet18, CIFAR-10
- Trainer: 300 epochs, batch 128, AnalogSGD lr+momentum 0.9, cosine annealing (eta_min = lr/100), seed 12345
- Venv: `/root/.venv310/bin/python`
- **GPU: 0** (`CUDA_VISIBLE_DEVICES=0`, A100-SXM4-40GB의 MIG 3g.20gb instance가 device 0으로 enumerate). `scripts/run_core8_gpu0.sh`, `scripts/run_tt1_gpu0.sh`, `logs/_gpu0_*.log`의 `LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]`로 검증. 이 문서의 **모든 실험(core-8 idealized/ecram_int4 + TT1)은 GPU 0에서 실행**됨. MIG fallback UUID `MIG-5f657128-1500-56b1-bd6b-dce2c3cbeaef`.
- 본 문서 작성 시점: 2026-05-15 · **최종 갱신: 2026-05-19 (TT1 sweep 12/12 완료 반영)**

---

## 1. Core-8 DPU Precision Sweep

DPU precision (`off / fp32 / fp16 / int8_qat / int4_qat`) × RPU preset (`Idealized / EcRam`) 셀에 대한 ablation.

### 완료된 6개 셀 (test/acc 최종 step 기준)

| task_name | preset | dpu | sa_enabled | lr | eta_min | test/acc | test/loss |
|---|---|---|---|---|---|---|---|
| core8_idealized_off | Idealized | off | false | 0.10 | 0.001 | 0.8173 | 0.5496 |
| core8_idealized_off_lr01 | Idealized | off | false | 0.01 | 0.0001 | 0.7957 | 0.5859 |
| core8_idealized_int8_qat | Idealized | int8_qat | true | 0.10 | 0.001 | **0.8716** | 0.5354 |
| core8_idealized_int8_qat_lr01 | Idealized | int8_qat | true | 0.01 | 0.0001 | 0.8454 | 0.4726 |
| core8_ecram_int4_qat_lr10 | EcRam | int4_qat | true | 0.10 | 0.001 | 0.5358 | 1.2850 |
| core8_ecram_int4_qat_lr01 | EcRam | int4_qat | true | 0.01 | 0.0001 | 0.5796 | 1.1618 |

### 핵심 관찰
- **Idealized 셀**: lr=0.1 > lr=0.01 (off: +2.16pp, int8_qat: +2.62pp)
- **Idealized × INT8-QAT > Idealized × off**: lr=0.1에서 +5.43pp, lr=0.01에서 +4.97pp → INT8-QAT가 idealized device에서 SA와 결합하여 일관된 이득
- **EcRam × INT4-QAT 붕괴**: 53–58% 수준. lr=0.01이 lr=0.1보다 +4.38pp 우수 (idealized와 반대 방향)
- 공격적 양자화(INT4)는 비이상 device dynamics와 결합 시 손실 증폭

### 미실행 6개 셀
`idealized × {fp32, fp16}`, `ecram × {off, fp32, fp16, int8_qat}`

### 인프라
- Dispatcher: `scripts/run_core8_gpu{0,1,2,3}.sh` (GPU별 셀 분담)
- 로그: `logs/<task_name>/runs/<timestamp>/csv/version_0/metrics.csv`
- 요약 CSV: `results/core8_dpu_precision_summary.csv` (현재 미채워짐, `collect_core8_results.py`로 수집)
- INT4-QAT 지원: `src/models/components/dpu_quant.py::prepare_dpu_int4_qat`, `src/models/exp7_1_module.py::_QAT_PRECISIONS`

---

## 2. TikiTaka v1 (TT1) Fast × Transfer LR Sweep

단일-타일 RPU preset을 `TransferCompound`로 교체하고 각 셀의 best baseline SA/lr을 유지한 채 `fast_lr × transfer_lr ∈ {0.1, 1.0}²` 격자 탐색.

- 3 cells × 4 LR combos = **12 runs 계획**
- 2026-05-12 09:09 UTC GPU0 sequential dispatcher 시작
- `scale_transfer_lr=False` (절대 transfer_lr)

### 현재 상태: **12 / 12 완료** (sweep ALL DONE, GPU 0, 2026-05-19 05:05 UTC)

`logs/_gpu0_tt1_dispatch.log` 마지막 줄: `[GPU0] 2026-05-19 05:05:35 TT1 SWEEP ALL DONE`. 5/15 스냅샷의 1 크래시(`tt1_idealized_int8_qat_f01_t10`)는 dispatcher 재개(2026-05-15 06:13) 후 정상 완료됨.

#### idealized_off 셀 (baseline 0.8173)

| task_name | fast_lr | transfer_lr | test/acc | best val/acc | Δ vs baseline |
|---|---|---|---|---|---|
| tt1_idealized_off_f01_t01 | 0.1 | 0.1 | 0.5950 | 0.6118 | −22.23 pp |
| tt1_idealized_off_f01_t10 | 0.1 | 1.0 | 0.6135 | 0.6220 | −20.38 pp |
| tt1_idealized_off_f10_t01 | 1.0 | 0.1 | 0.6540 | 0.6668 | −16.33 pp |
| tt1_idealized_off_f10_t10 | 1.0 | 1.0 | **0.6747** | 0.6834 | −14.26 pp |

#### idealized_int8_qat 셀 (baseline 0.8716)

| task_name | fast_lr | transfer_lr | test/acc | best val/acc | Δ vs baseline |
|---|---|---|---|---|---|
| tt1_idealized_int8_qat_f01_t01 | 0.1 | 0.1 | 0.7384 | 0.7608 | −13.32 pp |
| tt1_idealized_int8_qat_f01_t10 | 0.1 | 1.0 | 0.7477 | 0.7574 | −12.39 pp |
| tt1_idealized_int8_qat_f10_t01 | 1.0 | 0.1 | 0.8046 | 0.8178 | −6.70 pp |
| tt1_idealized_int8_qat_f10_t10 | 1.0 | 1.0 | **0.8063** | 0.8128 | −6.53 pp |

> `f01_t10`은 2026-05-15 02:35 1차 크래시(GPU backend, MIG-only 전이) 후 06:13 재개에서 정상 완료 (`run_timestamp 2026-05-15_06-13-03`).

#### ecram_int4_qat 셀 (baseline 0.5796, lr=0.01) — **원 가설의 핵심**

| task_name | fast_lr | transfer_lr | test/acc | best val/acc | Δ vs baseline |
|---|---|---|---|---|---|
| tt1_ecram_int4_qat_f01_t01 | 0.1 | 0.1 | 0.7223 | 0.7316 | **+14.27 pp** |
| tt1_ecram_int4_qat_f01_t10 | 0.1 | 1.0 | 0.6028 | 0.6106 | +2.32 pp |
| tt1_ecram_int4_qat_f10_t01 | 1.0 | 0.1 | **0.7470** | 0.7514 | **+16.74 pp** ★ sweep headline |
| tt1_ecram_int4_qat_f10_t10 | 1.0 | 1.0 | 0.6758 | 0.6836 | +9.62 pp |

### 핵심 관찰
- **★ 원 가설 확정: TikiTaka v1이 EcRam int4_qat 붕괴를 회복.** ecram_int4_qat 4 run 모두 단일-타일 baseline(0.5796) 초과. 최고 `tt1_ecram_int4_qat_f10_t01 = 0.7470 (+16.74pp)`, val 0.7514.
- **TikiTaka는 약한 device는 살리고 좋은 device는 손해.** idealized_off −14~−22pp, idealized_int8_qat −6.5~−13.3pp(여전히 0.8716 회복 못함), ecram_int4_qat +2.3~+16.7pp. 즉 TransferCompound는 비이상 device dynamics를 복구하지만 단일-타일이 이미 near-ideal일 때는 이득 없음(오히려 손실).
- **operating point는 3셀 공통**: `fast_lr=1.0 ≫ fast_lr=0.1` 항상; fast_lr=1.0일 때 ecram은 transfer_lr=0.1이 명백히 우수(0.7470 > 0.6758), idealized_int8_qat은 사실상 무차별(0.8046 vs 0.8063). 독립적인 GPU2/GPU3 core8_tikitaka_ecram sweep(최적 f=1.0/t=0.1)과 일치.
- 최악 조합은 대체로 `fast_lr=0.1, transfer_lr=1.0` (ecram 0.6028로 셀 최저).
- **격자 상향 확장 불필요**(5/15의 권고 폐기): ecram_int4_qat 가설은 f=1.0/t=0.1에서 이미 잘 회복되고 transfer_lr>0.1은 ecram을 단조 악화시킴.

### 후속 권장 (sweep 완료, 재개 불필요)
1. `tt1_ecram_int4_qat_f10_t01`에 `sa_enabled=false` ablation → +16.74pp 회복 중 TransferCompound vs SA/DPU 경로 기여 분리
2. `scripts/collect_core8_results.py`로 `results/core8_dpu_precision_summary.{csv,md}` 채워 단일-타일 + TT1 통합 테이블화

### 인프라
- Factory: `src/utils/rpu_factory.py::build_tikitaka_v1_rpu_config(preset, fast_lr, transfer_lr, scale_transfer_lr)`
- Dispatcher: `scripts/run_tt1_gpu0.sh`. `~model.integrated_resnet.rpu_config` 제거 후 `+...`로 factory 주입
- 로그: `logs/_gpu0_tt1_dispatch.log`, 개별 run은 `logs/<task_name>/runs/<timestamp>/`
- Per-run wall time ≈ 12–15 h (단일-타일 core-8 ≈ 8 h보다 transfer 오버헤드)

### 재개 시 필요 조치 — 해결 완료 (2026-05-19)
> ~~호스트 GPU 복구 후 6번째 셀부터 재시작~~ — dispatcher가 2026-05-15 06:13 자동 재개되어 나머지 7 run을 순차 완료, 2026-05-19 05:05 UTC sweep 전체 종료. 추가 조치 불필요. 후속 권장은 §2 참조.

---

## 3. 보조 산출물 (Smoke / Timing) — 무시 가능

`logs/`에 sanity-check 용도의 단기 run이 함께 존재. production 결과로 취급하지 말 것.

| dir | rows in metrics.csv | timestamp | 목적 |
|---|---:|---|---|
| `smoke_off` | 4 | 2026-05-08 12:44 | dispatcher dry-run |
| `smoke_fp32` | 4 | 2026-05-08 12:45 | dispatcher dry-run |
| `smoke_fp16` | 4 | 2026-05-08 12:45 | dispatcher dry-run |
| `smoke_int8_qat` | 4 | 2026-05-08 12:46 | dispatcher dry-run |
| `smoke_ecram_int8_qat` | 4 | 2026-05-08 12:47 | dispatcher dry-run |
| `timing_test` | 3 | 2026-05-08 13:01 | per-run wall time 측정 |

`results/core8_dpu_precision_summary.{csv,md}`는 비어있는 템플릿 (`collect_core8_results.py`로 채우는 흐름이나 현재 미채워짐).

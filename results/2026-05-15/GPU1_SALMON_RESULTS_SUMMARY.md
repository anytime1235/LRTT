# SALMON 실험 결과 정리

- Repository: `/root/SALMON`
- Backbone: ResNet18, CIFAR-10
- Trainer: 300 epochs, batch 128, AnalogSGD lr+momentum 0.9, cosine annealing (eta_min = lr/100), seed 12345
- Venv: `/root/.venv310/bin/python`
- 본 문서 작성 시점: 2026-05-15

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

### 현재 상태: 5 완료 / 1 크래시 / 6 미실행

| task_name | cell baseline | fast_lr | transfer_lr | test/acc | best val/acc | Δ vs baseline |
|---|---|---|---|---|---|---|
| tt1_idealized_off_f01_t01 | 0.8173 | 0.1 | 0.1 | 0.5950 | 0.6118 | −22.23 pp |
| tt1_idealized_off_f01_t10 | 0.8173 | 0.1 | 1.0 | 0.6135 | 0.6220 | −20.38 pp |
| tt1_idealized_off_f10_t01 | 0.8173 | 1.0 | 0.1 | 0.6540 | 0.6668 | −16.33 pp |
| tt1_idealized_off_f10_t10 | 0.8173 | 1.0 | 1.0 | **0.6747** | 0.6834 | −14.26 pp |
| tt1_idealized_int8_qat_f01_t01 | 0.8716 | 0.1 | 0.1 | **0.7384** | 0.7608 | −13.32 pp |
| tt1_idealized_int8_qat_f01_t10 | 0.8716 | 0.1 | 1.0 | — (crashed) | — | — |

### 크래시 / 미실행
- `tt1_idealized_int8_qat_f01_t10`: 2026-05-15 02:35:04 UTC 시작 → 5초 후 `MisconfigurationException: No supported gpu backend found!`로 실패. `metrics.csv` 미생성. `nvidia-smi`가 `Failed to initialize NVML: Unknown Error` 반환 → 호스트 GPU 손실.
- Dispatcher PID 2535996은 이미 종료됨.
- 미실행 6개: `idealized_int8_qat × {f10_t01, f10_t10}`, `ecram_int4_qat × {f01_t01, f01_t10, f10_t01, f10_t10}` (ecram_int4_qat 셀 전체 미시작)

### 핵심 관찰
- **idealized_off 4셀 모두 baseline 대비 14–22pp 손실**. 현 격자에서는 TikiTaka v1으로 단일-타일 idealized_off 회복 불가
- **(f01, t01)에서 SA+int8_qat vs off Δ = +14.34pp** (core-8 동일 비교 +5.43pp의 약 2.6배). TikiTaka의 보수적 transfer 영역에서 off 셀이 더 크게 망가지고, INT8-QAT가 손실을 일부 흡수
- **fast_lr = transfer_lr = 1.0이 idealized_off 셀에서 최우수**. 다른 셀로 일반화될 가능성. 격자를 위로 더 확장할 필요
- ecram_int4_qat 셀 회복 여부(원 가설의 핵심)는 **아직 검증 불가** — 전체 4 run 미실행

### 인프라
- Factory: `src/utils/rpu_factory.py::build_tikitaka_v1_rpu_config(preset, fast_lr, transfer_lr, scale_transfer_lr)`
- Dispatcher: `scripts/run_tt1_gpu0.sh`. `~model.integrated_resnet.rpu_config` 제거 후 `+...`로 factory 주입
- 로그: `logs/_gpu0_tt1_dispatch.log`, 개별 run은 `logs/<task_name>/runs/<timestamp>/`
- Per-run wall time ≈ 12–15 h (단일-타일 core-8 ≈ 8 h보다 transfer 오버헤드)

### 재개 시 필요 조치
1. 호스트 GPU/드라이버 복구 (`nvidia-smi` 정상 출력 확인)
2. Dispatcher 수정 또는 분기: 완료된 5 run의 task_name과 중복되지 않도록 6번째 셀부터 재시작
3. Baseline ecram_int4_qat (0.5796) 대비 ecram TT1 셀의 변화가 sweep의 핵심 가설

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

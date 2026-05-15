# SALMON core-8 실험 결과 정리

- **수집 일시**: 2026-05-15
- **공통 설정**: `core8` (8-task split), ResNet 백본, 300 epoch, seed=12345
- **소스**: `/root/SALMON/logs/<run_id>/runs/<timestamp>/csv/version_0/metrics.csv`
- **관련 스크립트**:
  - `scripts/run_core8_dpu_precision.sh` (baseline)
  - `scripts/queue_tikitaka_ecram.sh` (tikitaka sweep)

## 1. 결과 요약 테이블

| run_id | rpu_preset | DPU precision | SA | tikitaka (f/t) | epochs | best_val_backbone | test_backbone_acc | test_loss |
|---|---|---|---|---|---|---|---|---|
| core8_ecram_fp32          | EcRamPreset    | fp32 | ON | —          | 169/300 (중단) | 0.389 | — | — |
| core8_ecram_fp32_lr01     | EcRamPreset    | fp32 | ON | —          | 300 | 0.595 | **0.584** | 1.155 |
| core8_ecram_fp16_lr01     | EcRamPreset    | fp16 | ON | —          | 300 | 0.590 | **0.576** | 1.166 |
| core8_tikitaka_ecram_f01_t01 | tikitaka_ecram | fp32 | ON | 0.1 / 0.1 | 300 | 0.717 | **0.705** | 0.829 |
| core8_tikitaka_ecram_f01_t10 | tikitaka_ecram | fp32 | ON | 0.1 / 1.0 | 300 | 0.601 | **0.590** | 1.139 |
| core8_tikitaka_ecram_f10_t01 | tikitaka_ecram | fp32 | ON | 1.0 / 0.1 | 300 | **0.751** | **0.748** | **0.742** |
| core8_tikitaka_ecram_f10_t10 | tikitaka_ecram | fp32 | ON | 1.0 / 1.0 | 300 | 0.669 | **0.658** | 0.962 |

> `f` = fast_lr, `t` = transfer_lr (tikitaka 이중 LR). 모든 tikitaka run은 `sa_enabled=true, dpu_precision=fp32, dpu_schedule=full`.

## 2. 핵심 관찰

1. **lr 스케일이 결정적**: 첫 baseline (`core8_ecram_fp32`, lr=기본값)는 169 epoch에서 학습이 사실상 정체 (val 0.28). lr=0.01로 낮춘 `_lr01` 버전부터 정상 수렴 (0.58 수준).
2. **fp32 vs fp16 DPU**: lr=0.01 baseline 비교에서 fp32 0.584 vs fp16 0.576 → **0.8%p 차이로 사실상 동등**, DPU를 fp16으로 낮춰도 성능 손실 거의 없음.
3. **tikitaka 효과**: best tikitaka (f10_t01) 0.748 vs ecram baseline 0.584 → **+16.4%p**. tikitaka_ecram 도입이 SALMON core-8 최대 게임체인저.
4. **f/t 그리드**:
   - f10_t01 = **0.748** (최고)
   - f01_t01 = 0.705
   - f10_t10 = 0.658
   - f01_t10 = 0.590 (최저, baseline 수준으로 회귀)
   - → **fast_lr는 높게(1.0), transfer_lr는 낮게(0.1)** 유지하는 것이 최적. transfer_lr를 1.0으로 키우면 두 조건 모두에서 성능 하락.
5. **SA ablation 부재**: 모든 tikitaka run이 `sa_enabled=true`로만 실행됨 → tikitaka 효과 중 SA 기여분은 현재 데이터로 분리 불가.

## 3. 누락/미수행 condition

`scripts/collect_core8_results.py:24-31` 의 원래 grid 8셀 중 실제 결과가 있는 건 다음과 같음:

| condition | 상태 |
|---|---|
| core8_idealized_off | ❌ 미실행 |
| core8_idealized_fp32 | ❌ 미실행 |
| core8_idealized_fp16 | ❌ 미실행 |
| core8_idealized_int8_qat | ❌ 미실행 |
| core8_ecram_off (SA off) | ❌ 미실행 — **SA ablation 누락** |
| core8_ecram_fp32 | ⚠️ 169 ep에서 중단 (lr 문제), `_lr01`이 대체본 |
| core8_ecram_fp16 | ✅ `_lr01` 버전 완료 |
| core8_ecram_int8_qat | ❌ 미실행 |

결과 요약 CSV (`results/core8_dpu_precision_summary.csv`)는 헤더만 있고 값이 비어 있음 — `collect_core8_results.py` 미실행.

## 4. 다음 단계 권장

- `core8_ecram_off` (SA disabled, no tikitaka) — 진짜 baseline.
- `core8_tikitaka_ecram_f10_t01` + `sa_enabled=false` — tikitaka 환경에서의 SA marginal effect 측정.
- `int8_qat` 정밀도 실험 — fp16과 동등성 확인 시 하드웨어 의의 큼.
- Idealized preset 그리드 4셀 — 디바이스 노이즈 영향 격리.

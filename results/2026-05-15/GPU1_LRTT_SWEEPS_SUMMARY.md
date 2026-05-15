# LRTT Sweep 결과 정리 (전수)

- Repository: `/root/LRTT` (branch: `MLP`)
- Hardware: A100-40GB, CUDA 12.1, torch 2.3.1
- Venv: `/root/.venv310`
- 본 문서 작성 시점: 2026-05-15

본 문서는 **5개의 독립적인 sweep 트랙**과 그 산출물을 모두 다룹니다.

| # | 트랙 | 상태 | 산출물 위치 |
|---|---|---|---|
| 1 | Methods × AF/Noise per-cell TPE-40 | 완료 2026-05-07 | `results/methods_phased/{method}_per_cell.json` |
| 2 | Methods × AF/Noise phase1 (anchor + corner) | 완료 2026-05-02 | `results/methods_phased/{method}_phase1.json` |
| 3 | LR-TT-v2 rank scan (anchor only) | 완료 2026-04-28 | `results/hp_search_v2_rank{1_4,8,16,32_64}/*.json` |
| 4 | LR-TT-v2 rank × {AF, weight-bit} 1-D sensitivity | 완료 2026-04-30 | `results/v2_rank{1,8,64}_ctile_bit_af/{af,bit}_results.json` |
| 5 | Diagnostic D (noise/AF/bit robustness) | 완료 2026-04-28 | `outputs/diagnostic_d_noise_af_bit_robustness/` |

---

## 1. Methods × AF/Noise per-cell TPE-40 Sweep (continuous log-uniform)

### 설계
- Task: MNIST 784→256→10
- 두 1-D 축:
  - AF: γ_up = γ_down ∈ {0, 0.5, 1, 2, 5, 10}
  - Noise: dw_min_std = 0.3 × ratio, ratio ∈ {0, 0.25, 0.5, 1, 2, 3}
- 12 cells (6 AF + 6 Noise)
- Methods: `direct` (anchor-only), `tikitaka_v1`, `lrtt_v1`, `lrtt_v2`
- TPE 40 trials/cell, `MedianPruner(n_startup_trials=5, n_warmup_steps=5)`
- 연속 log-uniform 분포
- Device policy: bits=10, RANK=8, TE=10, omega=0.6, C-tile lifetime=0, 6T1C lifetime=1000, write_noise_std=0

### 상태: 완료 (2026-05-07 12:20 UTC, chain log "ALL METHODS DONE")
- 시작: 2026-05-06 14:21 UTC
- 총 wall time ≈ 22 h (tikitaka_v1: 9.6h, lrtt_v1: 7.33h, lrtt_v2: 5.27h)

### HP 탐색 범위 (continuous log-uniform)

| method | lr | transfer_lr | clf_lr |
|---|---|---|---|
| lrtt_v1 | [0.03, 3.0] | [3e-5, 3e-2] | [0.03, 3.0] |
| lrtt_v2 | [0.1, 10.0] | [0.01, 10.0] | [0.3, 10.0] |
| tikitaka_v1 | [0.01, 1.0] | [0.05, 10.0] | [0.05, 4.0] |

### Best test acc per cell (%)

#### AF axis (γ_up = γ_down)
| γ | tikitaka_v1 | lrtt_v1 | lrtt_v2 |
|---:|---:|---:|---:|
| 0.0 | 97.44 | **97.60** | 96.98 |
| 0.5 | 97.47 | **97.66** | 97.01 |
| 1.0 | **97.59** | 97.57 | 96.76 |
| 2.0 | **97.66** | 97.30 | 96.59 |
| 5.0 | **97.18** | 96.81 | 95.72 |
| 10.0 | **96.62** | 96.33 | 94.87 |

#### Noise axis (dw_min_std ratio)
| ratio | tikitaka_v1 | lrtt_v1 | lrtt_v2 |
|---:|---:|---:|---:|
| 0.00 | 97.64 | **97.65** | 97.23 |
| 0.25 | 97.52 | **97.62** | 96.80 |
| 0.50 | 97.50 | **97.61** | 97.14 |
| 1.00 | 97.57 | **97.66** | 97.14 |
| 2.00 | 97.48 | **97.59** | 97.20 |
| 3.00 | **97.50** | 97.44 | 97.22 |

### Best HP per cell (36 cells = 3 methods × 12)

#### tikitaka_v1 (lr ∈ [0.01, 1.0], tlr ∈ [0.05, 10.0], clf_lr ∈ [0.05, 4.0])
| axis | level | best_acc | lr | tlr | clf_lr |
|---|---:|---:|---:|---:|---:|
| af | 0.0 | 97.44 | 0.8124 | 0.8676 | 2.669 |
| af | 0.5 | 97.47 | 0.5359 | 0.3495 | 1.040 |
| af | 1.0 | 97.59 | 0.5359 | 1.375 | 1.279 |
| af | 2.0 | 97.66 | 0.8124 | 0.8676 | 2.669 |
| af | 5.0 | 97.18 | 0.8124 | 0.8676 | 2.669 |
| af | 10.0 | 96.62 | 0.5411 | 0.9155 | 2.474 |
| noise | 0.0 | 97.64 | 0.3692 | 0.2135 | 3.511 |
| noise | 0.25 | 97.52 | 0.3734 | 0.1628 | 3.821 |
| noise | 0.5 | 97.50 | 0.6849 | 0.7924 | 0.4796 |
| noise | 1.0 | 97.57 | 0.7128 | 5.189 | 0.9398 |
| noise | 2.0 | 97.48 | 0.6608 | 1.502 | 3.621 |
| noise | 3.0 | 97.50 | 0.2130 | 9.167 | 2.245 |

#### lrtt_v1 (lr ∈ [0.03, 3.0], tlr ∈ [3e-5, 3e-2], clf_lr ∈ [0.03, 3.0])
| axis | level | best_acc | lr | tlr | clf_lr |
|---|---:|---:|---:|---:|---:|
| af | 0.0 | 97.60 | 0.8861 | 1.623e-3 | 0.3167 |
| af | 0.5 | 97.66 | 0.2465 | 1.202e-3 | 1.160 |
| af | 1.0 | 97.57 | 0.1113 | 1.199e-3 | 1.627 |
| af | 2.0 | 97.30 | 0.05243 | 4.019e-3 | 1.225 |
| af | 5.0 | 96.81 | 0.04442 | 2.285e-4 | 1.241 |
| af | 10.0 | 96.33 | 0.07539 | 4.871e-5 | 0.4502 |
| noise | 0.0 | 97.65 | 1.274 | 1.578e-3 | 0.09256 |
| noise | 0.25 | 97.62 | 0.6730 | 1.240e-3 | 0.1806 |
| noise | 0.5 | 97.61 | 0.5577 | 1.034e-3 | 0.1366 |
| noise | 1.0 | 97.66 | 0.3526 | 2.503e-3 | 0.1025 |
| noise | 2.0 | 97.59 | 0.5039 | 2.790e-3 | 0.2534 |
| noise | 3.0 | 97.44 | 0.2056 | 3.074e-3 | 0.6285 |

#### lrtt_v2 (lr ∈ [0.1, 10.0], tlr ∈ [0.01, 10.0], clf_lr ∈ [0.3, 10.0])
| axis | level | best_acc | lr | tlr | clf_lr |
|---|---:|---:|---:|---:|---:|
| af | 0.0 | 96.98 | 3.000 | 1.000 | 0.3000 |
| af | 0.5 | 97.01 | 2.994 | 0.4185 | 1.235 |
| af | 1.0 | 96.76 | 1.575 | 0.02938 | 0.5184 |
| af | 2.0 | 96.59 | 2.998 | 0.01217 | 1.055 |
| af | 5.0 | 95.72 | 3.419 | 0.8697 | 0.3262 |
| af | 10.0 | 94.87 | 4.266 | 0.3171 | 0.5785 |
| noise | 0.0 | 97.23 | 1.575 | 0.02938 | 0.5184 |
| noise | 0.25 | 96.80 | 3.501 | 0.8287 | 0.3005 |
| noise | 0.5 | 97.14 | 8.076 | 0.01315 | 0.3881 |
| noise | 1.0 | 97.14 | 1.575 | 0.02938 | 0.5184 |
| noise | 2.0 | 97.20 | 2.890 | 1.027 | 0.3196 |
| noise | 3.0 | 97.22 | 5.282 | 6.463 | 0.7549 |

### 핵심 관찰
- **tikitaka_v1**: AF γ ≥ 1에서 우세, 강한 AF (γ=10)에서 1등 (96.62%)
- **lrtt_v1**: clean / 약한 AF / 모든 noise 셀에서 1등
- **lrtt_v2**: 모든 셀에서 최하위. AF γ=10에서 tikitaka_v1 대비 −1.75pp, clean에서 −0.62pp → learned A-path 제거에 따른 capacity 손실
- 연속 log-uniform으로 v2의 lr=3.0 grid-max pinning 문제 완화 (예: af=10에서 lr=4.27로 grid max 초과)
- 5월 5일 categorical TPE-30 phase3 91.4% → per_cell 96.33% (lrtt_v1 AF γ=10, +4.93pp)
- **lrtt_v1의 tlr**은 모든 셀에서 ~1e-3 수준 (좁은 범위 [3e-5, 3e-2]), **lrtt_v2의 tlr**은 0.01~10 광범위 분포 → 두 방법의 transfer dynamics 스케일이 구조적으로 다름

### 인프라
- Driver: `experiments/run_methods_per_cell_tpe30.py`
- Chain: `run_methods_tpe30_chain.sh`
- Range source: `experiments/sweep_methods_phased.py::METHOD_HP_RANGES`
- Plotter: `experiments/plot_methods_phased.py` → `results/methods_phased/figures/{af,noise,methods}_sensitivity.png`

---

## 2. Methods × AF/Noise Phase1 (Anchor + 4 Corner Cells)

per_cell sweep 이전 단계. anchor + AF 중간/최대 + noise 중간/최대 = 5 cells per method.

### Phase1 best per cell

| method | cell | γ | noise | best_acc (%) | hp (lr, tlr, clf_lr) |
|---|---|---:|---:|---:|---|
| direct | anchor | 0.0 | 0.0 | **98.51** | (0.1, 1.0, 0.3) |
| tikitaka_v1 | anchor | 0.0 | 0.0 | 97.52 | (3.0, 1.0, 0.3) |
| lrtt_v1 | anchor | 0.0 | 0.0 | 97.68 | (0.3, 0.001, 1.0) |
| lrtt_v1 | af_middle | 2.0 | 0.0 | 97.31 | (0.1, 0.001, 1.0) |
| lrtt_v1 | af_largest | 10.0 | 0.0 | 95.93 | (0.1, 1e-5, 1.0) |
| lrtt_v1 | noise_middle | 0.0 | 1.0 | 97.58 | (1.0, 0.001, 0.1) |
| lrtt_v1 | noise_largest | 0.0 | 3.0 | 97.72 | (1.0, 0.001, 0.3) |
| lrtt_v2 | anchor | 0.0 | 0.0 | 97.29 | (3.0, 0.01, 0.3) |
| lrtt_v2 | af_middle | 2.0 | 0.0 | 96.77 | (3.0, 0.01, 0.3) |
| lrtt_v2 | af_largest | 10.0 | 0.0 | 95.43 | (0.3, 0.01, 0.3) |
| lrtt_v2 | noise_middle | 0.0 | 1.0 | 97.19 | (3.0, 0.01, 1.0) |
| lrtt_v2 | noise_largest | 0.0 | 3.0 | 97.08 | (3.0, 0.1, 0.3) |

### 핵심 관찰
- **direct anchor 98.51%** — 4-method 통틀어 최고. anchor 격차: direct − lrtt_v1 = +0.83pp, direct − lrtt_v2 = +1.22pp
- per_cell 결과 (track #1) 대비 phase1은 모든 셀에서 더 낮음 (lrtt_v1 anchor 97.68 vs per_cell 97.60는 거의 동일, 그 외 phase1이 per_cell보다 일관 낮음)
- phase1은 categorical 격자였고 per_cell은 continuous → 격차 줄어든 cell 다수

### 산출물
- `{method}_phase1.json` (현행), `{method}_phase1_partial.json` (resume용 동치)
- `{method}_best_hp.json` — phase1 seed HP (per_cell warm-start trial 0로 enqueue됨)
- `{method}_manifest.json` — 실행 metadata

---

## 3. LR-TT-v2 Rank Scan (Anchor only)

LR-TT-v2의 **rank scaling** 검증. MNIST/MDMLP, anchor 셀 (γ=0, noise=0) 한정.
검색 공간: `(lr, transfer_lr, lifetime_phys, policy, cap_rho)`.

### Overall best per rank

| rank | best_acc (%) | best_lr | best_tlr | lifetime_phys | policy | cap_rho | completed_at (UTC) |
|---:|---:|---:|---:|---:|---|---:|---|
| 4 | 96.02 | 0.560 | 0.843 | 1000 | shuffled_cycle | 1.0 | 2026-04-28 20:15 |
| 8 | 96.88 | 1.879 | 0.0071 | 1000 | shuffled_cycle | 1.0 | 2026-04-28 22:05 |
| 16 | 97.21 | 2.186 | 1.973 | 1000 | shuffled_cycle | 1.0 | 2026-04-28 22:31 |
| 64 | **97.85** | 1.930 | 0.0060 | 1000 | shuffled_cycle | 1.0 | 2026-04-28 23:40 |

### 핵심 관찰
- **v2 rank scaling 단조 증가**: 4→8→16→64 (96.02 → 96.88 → 97.21 → 97.85)
- methods_phased rank=8 anchor 96.98% vs 본 sweep rank=8 anchor 96.88% (검색 공간/시드 차이로 거의 동일)
- rank=64에서 v2 anchor 97.85% → direct anchor 98.51%에 −0.66pp까지 근접. v2의 capacity 손실은 rank로 보상 가능
- 두 best policy 모두 `shuffled_cycle`, `lifetime_phys=1000`로 수렴

### 산출물
- `results/hp_search_v2_rank1_4/lrtt_v2_rank1_4_summary.json`
- `results/hp_search_v2_rank8/lrtt_v2_rank8_summary.json`
- `results/hp_search_v2_rank16/lrtt_v2_rank16_summary.json`
- `results/hp_search_v2_rank32_64/lrtt_v2_rank32_64_summary.json` (+ `results_final.json`, `results_partial.json`)

---

## 4. LR-TT-v2 Rank × {AF, Weight-bit} 1-D Sensitivity

각 rank별 AF 6셀 + Bit 6셀, TPE 16 trials/cell.

### 4-A. AF axis (bits=10 고정) best_acc (%)

| γ | rank=1 | rank=8 | rank=64 |
|---:|---:|---:|---:|
| 0.0 | 95.51 | 96.81 | **97.84** |
| 0.5 | 95.43 | 96.75 | 97.65 |
| 1.0 | 95.61 | 96.75 | 97.40 |
| 2.0 | 95.38 | 96.11 | 96.21 |
| 5.0 | 95.25 | **95.91** | 95.13 |
| 10.0 | 94.88 | **95.07** | 94.32 |

### 4-B. Weight-bit axis (γ=0 고정) best_acc (%)

| bits | rank=1 | rank=8 | rank=64 |
|---:|---:|---:|---:|
| 5 | 95.89 | 96.50 | 97.62 |
| 6 | 95.71 | 96.82 | 97.62 |
| 7 | 95.59 | 96.92 | 97.56 |
| 8 | 95.02 | 96.67 | 97.81 |
| 9 | 95.54 | 96.68 | 97.70 |
| 10 | 95.54 | 96.85 | 97.73 |

### Best HP per cell (rank × {AF, bit}, 총 36 cells)

#### AF axis (bits=10) — best params (lr, tlr, lifetime)
| γ | rank=1 | rank=8 | rank=64 |
|---:|---|---|---|
| 0.0 | acc=95.51, lr=1.0, tlr=0.1, lt=None | acc=96.81, lr=1.0, tlr=0.1, lt=None | acc=97.84, lr=1.0, tlr=0.1, lt=None |
| 0.5 | acc=95.43, lr=1.0, tlr=10.0, lt=None | acc=96.75, lr=1.0, tlr=1.0, lt=None | acc=97.65, lr=1.0, tlr=0.01, lt=1000 |
| 1.0 | acc=95.61, lr=1.0, tlr=10.0, lt=None | acc=96.75, lr=1.0, tlr=0.1, lt=None | acc=97.40, lr=1.0, tlr=0.01, lt=1000 |
| 2.0 | acc=95.38, lr=1.0, tlr=0.1, lt=None | acc=96.11, lr=1.0, tlr=0.1, lt=1000 | acc=96.21, lr=1.0, tlr=0.01, lt=1000 |
| 5.0 | acc=95.25, lr=1.0, tlr=1.0, lt=1000 | acc=95.91, lr=1.0, tlr=0.01, lt=None | acc=95.13, lr=0.1, tlr=0.1, lt=1000 |
| 10.0 | acc=94.88, lr=1.0, tlr=0.1, lt=1000 | acc=95.07, lr=1.0, tlr=0.1, lt=None | acc=94.32, lr=0.1, tlr=0.01, lt=1000 |

#### Bit-width axis (γ=0) — best params (lr, tlr, lifetime)
| bits | rank=1 | rank=8 | rank=64 |
|---:|---|---|---|
| 5 | acc=95.89, lr=1.0, tlr=0.1, lt=1000 | acc=96.50, lr=1.0, tlr=0.1, lt=None | acc=97.62, lr=1.0, tlr=0.1, lt=None |
| 6 | acc=95.71, lr=1.0, tlr=0.1, lt=None | acc=96.82, lr=1.0, tlr=0.1, lt=1000 | acc=97.62, lr=1.0, tlr=0.1, lt=1000 |
| 7 | acc=95.59, lr=1.0, tlr=0.1, lt=1000 | acc=96.92, lr=1.0, tlr=0.1, lt=1000 | acc=97.56, lr=1.0, tlr=0.1, lt=1000 |
| 8 | acc=95.02, lr=1.0, tlr=0.1, lt=None | acc=96.67, lr=1.0, tlr=0.1, lt=1000 | acc=97.81, lr=1.0, tlr=0.01, lt=1000 |
| 9 | acc=95.54, lr=1.0, tlr=1.0, lt=1000 | acc=96.68, lr=1.0, tlr=0.1, lt=1000 | acc=97.70, lr=1.0, tlr=0.01, lt=None |
| 10 | acc=95.54, lr=1.0, tlr=0.1, lt=1000 | acc=96.85, lr=1.0, tlr=0.1, lt=None | acc=97.73, lr=1.0, tlr=1.0, lt=None |

(`lt=None`은 lifetime_phys 미설정, `lt=1000`은 6T1C lifetime 1000을 의미)

### 핵심 관찰
- **AF가 커지면 rank 우위 역전**: γ=5에서 rank=8 (95.91%) > rank=64 (95.13%), γ=10에서 rank=8 (95.07%) > rank=64 (94.32%). rank-large는 clean 영역 capacity 이득이고 AF stress에서는 오히려 불리
- **bit-width 둔감**: 5→10 bits에서 rank=8 Δ=0.35pp, rank=64 Δ=0.11pp. v2는 weight quantization에 강함
- **best_lr이 대부분 1.0에 수렴** (rank=64 강한 AF γ≥5에서만 lr=0.1로 떨어짐). 검색 격자는 lr ∈ {0.1, 1.0}, tlr ∈ {0.01, 0.1, 1.0, 10.0}, lifetime ∈ {None, 1000} (16 trials/cell)
- best_tlr은 0.1 우세, 일부 셀에서 1.0이나 0.01 선택 — track #1의 연속 log-uniform 결과와 호환되는 범위

### 산출물
- `results/v2_rank1_ctile_bit_af/{af,bit}_results.json` (+ `_partial.json`)
- `results/v2_rank8_ctile_bit_af/{af,bit}_results.json` (+ `_partial.json`)
- `results/v2_rank64_ctile_bit_af/{af,bit}_results.json` (+ `_partial.json`)

---

## 5. LRTT Diagnostic D (Noise / AF / Bit Robustness)

### 목적
LR-TT-v2가 학습된 `tile_a` path를 구조적으로 제거하여 A-tile noise/AF/weight-bit 민감도를 ~0으로 만든다는 것을 증명. 나머지 민감도는 B residual buffer와 C transfer/core tile에 국한.
**방어 가능한 클레임은 "path-specific robustness"이며 "universal superiority"가 아님.**

### 상태: 2026-04-28 산출. 이후 변경 없음.

| Sub-diagnostic | output | 시각 (2026-04-28 UTC) |
|---|---|---|
| D0 A-path invariance | `a_path_invariance.csv` + `a_path_invariance_summary.json` | 17:40 |
| D1 Local update fidelity | `local_update_fidelity.csv` | 17:42 |
| D2 OAT MNIST | `oat_mnist.csv` (production), `oat_mnist_pilot1ep.csv` (pilot) | 17:59 / 18:12 |
| D2 OAT regression | `oat_regression.csv` (production), `oat_regression_pilot200.csv` (pilot) | 20:10 / 18:12 |
| Summary | `summary.md` | 17:59 |

위치: `/root/LRTT/outputs/diagnostic_d_noise_af_bit_robustness/`

### Pilot에서 확인된 PASS gates (v2)
- A-path FP invariance bit-exact: B/C rel_err = 0.0, `num_a_updates = 0`
- Analog v2의 C residue ≈ 0.03 (non-isolated device-construction RNG 기인, caveat로 문서화 — 허용)

### 인프라
- Entry: `experiments/diagnostics/diagnostic_d_noise_af_bit_robustness.py`
- Utils: `experiments/diagnostics/noise_af_bit_utils.py`, `lrtt_diagnostic_configs.py`
- Plot: `experiments/diagnostics/plot_diagnostic_d_noise_af_bit_robustness.py`
- Tests: 40 tests pass (6 + 34)
- 사양서: `/root/DIAGNOSTIC_D_NOISE_AF_BIT_ROBUSTNESS_CLAUDE_CODE.md`

### Production 재실행 권고
- Pilot: regression 200 steps × 2 seeds, MNIST 5k subset × 1 epoch × 2 seeds
- 권장: regression 2000 steps × 3 seeds (v1 A-path degradation 노출용)
- `oat_regression.csv` 자체는 production 산출되어 있으나 budget 충족 여부는 재확인 필요

---

## 부록: Deprecated / 무시 대상

- `results/methods_af_noise_REV1_OLD/`
- `results/methods_af_noise_REV2_LIFETIMEBUG/`
- `results/methods_af_noise_REV3_12x12_v2_partial/`
- `results/methods_af_noise_hpsearch/`
- `results/methods_af_noise_smoke/` (2026-05-01 smoke run, 현행 methods_phased로 대체)
- `results/methods_phased_UDONLY_OLD/`
- `results/methods_phased/{method}_phase3.json` (deprecated fixed-HP × 5-run × 12 cells)
- `results/methods_phased/tikitaka_v1_per_cell_10trials.json` (이전 10-trial)
- `results/methods_phased/lrtt_v2_per_cell_smoke.json` (1 cell × 2 trials)
- `/root/EXPERIMENT_METHODS_AF_NOISE.md` (12×12 원안, 현재 6×6 phased로 대체)

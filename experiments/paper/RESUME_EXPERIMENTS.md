# ECO Diagnostic Experiments — Resume Guide

> 최종 업데이트: 2026-03-24
> Branch: transformer

## 환경 정보

- **GPU**: NVIDIA A100-SXM4-40GB, MIG 1개 (~20GB)
- **Python**: `/root/.venv310/bin/python`
- **Working dir**: `/root/LRTT/experiments/paper`
- **결과 경로**: `results/paper/`

---

## 완료된 실험 (3/19 ~ 3/24)

### 1. Training (사전 완료, results/paper/ 내 존재)

| 실험 | 위치 | 상태 |
|------|------|------|
| Mixed Precision 10b (qkvo/ffn/all) | `mixed_prec_10b/` | 완료 |
| TTv1 gamma/reset sweep | `phase1c_4ep/`, `phase1c_4ep_b16/` | 완료 |
| TTv1 fast-bit sweep (8~16b) | `phase1c_4ep/`, `phase1c_4ep_b16/` | 완료 |
| TTv1 slow-tile sweep (6~9b) | `slow_tile_sweep_fast14b/` | 완료 |
| Single RPU stoch bit sweep (6~10b) | `single_rpu_stoch_bit_sweep/` | 완료 |
| IO sweep | `io_sweep/` | 완료 |

### 2. D0: Smoke Test

| 위치 | 상태 |
|------|------|
| `diag_D0_smoke/` (4 methods) | 완료 |

### 3. D1: Sub-pulse Mapping (128 steps)

| 실험 | 위치 | 상태 |
|------|------|------|
| single_rpu stoch 8/10/12/14b | `diag_D1_subpulse/single_rpu_stoch_*` | **완료** |
| single_rpu det 6/8/10/12/14b | `diag_D1_subpulse/single_rpu_det_*` | **완료** |
| eco_ref 8b, 10b | — | **FAIL** (grad_accum 호환 문제) |

### 4. D2: Carry-path (1024 steps)

| 실험 | 위치 | 상태 |
|------|------|------|
| single_rpu stoch 8/10/12b | `diag_D2_carrypath/single_rpu_stoch_*` | **완료** (carry_path_summary 존재) |
| single_rpu stoch 6b, 14b | `diag_D2_carrypath/` | **미완료** (config만 존재) |
| ttv1_rl_g1r1 s8/s10/s12 | `diag_D2_carrypath/ttv1_rl_g1r1_*` | **완료** |
| ttv1_rl_g1r1 s6, s14 | `diag_D2_carrypath/` | **미완료** |
| ttv1_rl_g0r0 s8/s10/s12/s14 | `diag_D2_carrypath/ttv1_rl_g0r0_*` | **미완료** (negative control) |

### 5. W_max Sweep

| 실험 | 위치 | 상태 |
|------|------|------|
| single_rpu 14b wmax=0.1 | `diag_wmax_sweep/D1_single_rpu_14b_omega0_wmax0.1` | **완료** |
| single_rpu 14b wmax=0.01 | `diag_wmax_sweep/D1_single_rpu_14b_omega0_wmax0.01` | **완료** (training_log만, diagnostics summary 없음) |
| TTv1 fast weight baseline | `diag_wmax_sweep/D1_ttv1_fast_weight_baseline` | **완료** (fast tile |w| max=0.147) |

### 6. TTv1 Fast Tile Sub-pulse (w_max_fast=0.25) — 중단됨

| 실험 | 위치 | 상태 |
|------|------|------|
| ttv1 fast 8b wmax=0.25 | `diag_D1_ttv1_fast_wmax025/ttv1_fast8b_wmax025` | **중단** (~84/128 step) |
| ttv1 fast 10b wmax=0.25 | `diag_D1_ttv1_fast_wmax025/ttv1_fast10b_wmax025` | 미실행 |
| ttv1 fast 12b wmax=0.25 | `diag_D1_ttv1_fast_wmax025/ttv1_fast12b_wmax025` | 미실행 |
| ttv1 fast 14b wmax=0.25 | `diag_D1_ttv1_fast_wmax025/ttv1_fast14b_wmax025` | 미실행 |
| ttv1 fast 16b wmax=0.25 | — | 미실행 (BL saturation 분석용 추가 예정) |
| ttv1 fast 18b wmax=0.25 | — | 미실행 (BL saturation 분석용 추가 예정) |

---

## 이어서 실행할 실험

### Step 1: TTv1 fast tile sub-pulse (8/10/12/14b) 재실행

```bash
cd /root/LRTT/experiments/paper
bash launchers/run_D1_ttv1_fast_wmax025.sh 2>&1 | tee /root/D1_ttv1_fast_wmax025.log
```

**설정:**
- method: ttv1, ttv1-mode: residual_lane
- fast tile: 8/10/12/14b sweep, **w_max_fast=0.25**
- slow tile: 10b (w_max=1.0)
- gamma=1.0, reset_prob=1.0, fast_lr=0.1, transfer_lr=1.0, transfer_every=4
- batch_size=12, grad_accum=4 (effective batch=48), analog_lr=0.016
- 128 steps, diag layers: 0,5,11
- **예상 시간**: run당 ~40분, 총 ~2.5시간

### Step 2: 16b, 18b 추가 (BL saturation 분석용)

desired_bl=31 saturation 영역 커버를 위해 추가 실행.

```bash
cd /root/LRTT/experiments/paper
PYTHON="/root/.venv310/bin/python"

COMMON="--mode fixed --seed 42 --epochs 1 --warmup-ratio 0 --min-lr-rate 1.0 --analog-lr 0.016"
DIAG="--diag-update-exact --diag-layer-set 0,5,11"
D1="$COMMON --max-steps 128 --batch-size 12 --grad-accum-steps 4 $DIAG --log-every 1"
TTv1_BASE="--method ttv1 --ttv1-mode residual_lane --n-bits-slow 10 --gamma 1.0 --with-reset-prob 1.0 --fast-lr 0.1 --transfer-lr 1.0 --units-in-mbatch true --transfer-every 4 --w-max-fast 0.25"
RESULTS="results/paper/diag_D1_ttv1_fast_wmax025"

# 16b (dw_min=7.63e-6, μ_p99≈92 > BL=31)
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py $D1 $TTv1_BASE \
    --n-bits 16 --output-dir "$RESULTS/ttv1_fast16b_wmax025"

# 18b (dw_min=1.91e-6, μ_p99≈367 > BL=31)
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py $D1 $TTv1_BASE \
    --n-bits 18 --output-dir "$RESULTS/ttv1_fast18b_wmax025"
```

### Step 3: 플롯 생성

```bash
cd /root/LRTT/experiments/paper

# TTv1 fast tile μ ECDF (wmax=0.25, 8~18b)
/root/.venv310/bin/python plot_D1_ttv1_fast_wmax025.py
# → /root/D1_ttv1_fast_wmax025_figure.png

# 기존 single_rpu μ ECDF (비교용)
/root/.venv310/bin/python plot_D1_paper_figures.py
# → /root/D1_main_figure.png, /root/D1_supplementary_figures.png
```

### Step 4 (선택): 미완료 D2 carry-path run

```bash
bash launchers/run_D2_fixed.sh 2>&1 | tee /root/D2_fixed.log
```

---

## 3-regime 분석 프레임워크

Sub-pulse 분석의 핵심 변수: `μ = |grad_element| / dw_min`

```
dw_min = 2 × w_max / 2^n_bits

μ < 1:       sub-pulse (update 소실, noise 지배)
1 ≤ μ ≤ BL:  유효 범위 (정상 학습)
μ > BL:      BL saturation (update clipping → 학습 느려짐)

BL = desired_bl = 31 (default)
```

### dw_min 예측 테이블 (w_max_fast=0.25)

| fast bits | dw_min | μ_median | μ_p99 | 예상 regime |
|-----------|--------|----------|-------|-------------|
| 8b | 1.95e-3 | 0.002 | 0.4 | 완전 sub-pulse |
| 10b | 4.88e-4 | 0.008 | 1.4 | 대부분 sub-pulse |
| 12b | 1.22e-4 | 0.03 | 5.7 | sub-pulse + 일부 유효 |
| 14b | 3.05e-5 | 0.13 | 23 | sub-pulse + 유효 |
| 16b | 7.63e-6 | 0.52 | 92 | sub-pulse + BL saturation 공존 |
| 18b | 1.91e-6 | 2.1 | 367 | 유효 + 심한 BL saturation |

---

## 핵심 코드 수정사항

### paper_experiment.py
- `--diag-at-steps`: 특정 step에서만 진단
- `--diag-layer-set`: 진단 대상 layer 제한 (0,5,11)
- `--w-max-fast`: TTv1 fast tile w_max 설정
- `--log-every`: training log 기록 빈도

### update_diagnostics.py
- `accumulate_microbatch()`: grad_accum>1에서 microbatch별 G_l 누적
- `snapshot_weights_before()`: 첫 microbatch 전 w_before 캡처
- Per-microbatch mu (`mb_mu`) + effective mu (`eff_mu`) 이중 추적
- mu histogram: `(|delta_target|/n_mb) / dw_min` — 실제 tile.update() 호출 기준

### carry_path_diagnostics.py
- `snapshot_weights_before()`: grad_accum 대응
- layer_set 필터
- TTv1 slow/fast tile 분리 진단

---

## 파일 구조

```
experiments/paper/
├── paper_experiment.py          # 메인 실험 스크립트
├── update_diagnostics.py        # D1 sub-pulse 진단
├── carry_path_diagnostics.py    # D2 carry-path 진단
├── rpu_configs.py               # RPU config 빌더
├── plot_D1_paper_figures.py     # D1 single_rpu μ ECDF 플롯
├── plot_D1_ttv1_fast_wmax025.py # D1 TTv1 fast tile μ ECDF 플롯
├── launchers/
│   ├── run_D1_ttv1_fast_wmax025.sh  # ★ TTv1 fast tile 8/10/12/14b sweep
│   ├── run_D1_wmax_sweep.sh         # w_max sweep (0.1, 0.01)
│   ├── run_D1_ttv1_fast_weight.sh   # TTv1 fast weight 분포 측정
│   ├── run_D1_D2_final.sh           # D1+D2 순차 실행
│   ├── run_D1_D2_recovery.sh        # D1+D2 재실행
│   ├── run_D1_D2_wmax.sh            # D1+D2 w_max 변형
│   ├── run_D2_fixed.sh              # D2 미완료 run 보충
│   ├── diag_D0_smoke.sh             # D0 smoke test
│   ├── diag_D1_subpulse.sh          # D1 원본
│   ├── diag_D2_carrypath.sh         # D2 원본
│   ├── diag_D2b_pulse_ablation.sh   # D2b pulse ablation (미실행)
│   └── diag_D3_gamma.sh             # D3 gamma sweep (미실행)
└── results/paper/
    ├── diag_D0_smoke/
    ├── diag_D1_subpulse/            # single_rpu stoch/det 6~14b ✓
    ├── diag_D1_ttv1_fast_wmax025/   # TTv1 fast tile (중단됨, 재실행 필요)
    ├── diag_D2_carrypath/           # 1024-step carry-path (일부 완료)
    └── diag_wmax_sweep/             # w_max=0.1/0.01 + TTv1 baseline ✓
```

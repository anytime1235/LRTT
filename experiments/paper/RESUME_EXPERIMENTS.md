# 실험 재개 가이드

> GPU 연결 끊김으로 중단됨 (2026-03-17 ~11:00 UTC)
> Branch: transformer, Commit: d948c48

## 환경 설정

```bash
cd /root/LRTT/experiments/paper
export PYTHON=/root/.venv310/bin/python
```

---

## 1. GPU 0 — Mixed Precision 10-bit (재개 필요)

### 완료된 실험
| Tag | target | analog_lr | Best F1 |
|-----|--------|-----------|---------|
| mp_10b_qkvo | attention | 0.0357 | **87.61** |
| mp_10b_ffn | ffn | 0.0357 | 완료 (summary 확인 필요) |

### 남은 실험 (순차 실행)

```bash
# 1) mp_10b_all (alr=0.0357) — 중단됨, 재실행
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --seed 42 \
    --epochs 4 --n-bits 10 \
    --target-layers all \
    --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
    --output-dir results/paper/mixed_prec_10b/mp_10b_all \
    --log-every 20

# 2) mp_10b_qkvo (alr=0.00357, 1/10)
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --seed 42 \
    --epochs 4 --n-bits 10 \
    --target-layers attention \
    --analog-lr 0.00357 --classifier-lr 0.00076 --ln-lr 0.00076 \
    --output-dir results/paper/mixed_prec_10b/mp_10b_qkvo_alr0.00357 \
    --log-every 20

# 3) mp_10b_ffn (alr=0.00357)
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --seed 42 \
    --epochs 4 --n-bits 10 \
    --target-layers ffn \
    --analog-lr 0.00357 --classifier-lr 0.00076 --ln-lr 0.00076 \
    --output-dir results/paper/mixed_prec_10b/mp_10b_ffn_alr0.00357 \
    --log-every 20

# 4) mp_10b_all (alr=0.00357)
CUDA_VISIBLE_DEVICES=0 $PYTHON paper_experiment.py \
    --mode fixed --method mixed_precision --seed 42 \
    --epochs 4 --n-bits 10 \
    --target-layers all \
    --analog-lr 0.00357 --classifier-lr 0.00076 --ln-lr 0.00076 \
    --output-dir results/paper/mixed_prec_10b/mp_10b_all_alr0.00357 \
    --log-every 20
```

또는 스크립트로:
```bash
nohup bash launchers/gpu0_mixed_prec_10b.sh > /root/nohup_mp_10b.log 2>&1 &
# 주의: 이미 완료된 mp_10b_qkvo, mp_10b_ffn은 output-dir이 존재하면 덮어쓸 수 있음
# 완료된 결과 백업 후 실행 권장
```

---

## 2. GPU 1,2,3 — Phase 1C (4ep) TTv1 Gamma×Reset Sweep (재개 필요)

### 완료된 실험
| gamma | reset=0 | reset=1.0 |
|-------|---------|-----------|
| 0.01 | 82.21 | 79.87 |
| 0.05 | **84.78** | 81.24 |
| 0.5 | 83.41 (ep4 붕괴→73.9) | **85.08** |

### 남은 실험 (중단된 것 포함)

```bash
# 공통 설정
# uim=true, te=1, fast_lr=0.1, scale_transfer_lr=false, transfer_lr=gamma
# Fast=14bit (--n-bits 14), Slow=10bit (--n-bits-slow 10), 4 epochs

# GPU 1: g0.1_r0 (중단됨, 재실행)
CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 0.1 --units-in-mbatch true --transfer-every 1 \
    --with-reset-prob 0 --fast-lr 0.1 \
    --transfer-lr 0.1 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --output-dir results/paper/phase1c_4ep/g0.1_r0 \
    --log-every 20

# GPU 1: g0.3_r0
CUDA_VISIBLE_DEVICES=1 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 0.3 --units-in-mbatch true --transfer-every 1 \
    --with-reset-prob 0 --fast-lr 0.1 \
    --transfer-lr 0.3 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --output-dir results/paper/phase1c_4ep/g0.3_r0 \
    --log-every 20

# GPU 2: g0.1_r1.0 (중단됨, 재실행)
CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 0.1 --units-in-mbatch true --transfer-every 1 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 0.1 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --output-dir results/paper/phase1c_4ep/g0.1_r1.0 \
    --log-every 20

# GPU 2: g0.3_r1.0
CUDA_VISIBLE_DEVICES=2 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 0.3 --units-in-mbatch true --transfer-every 1 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 0.3 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --output-dir results/paper/phase1c_4ep/g0.3_r1.0 \
    --log-every 20

# GPU 3: g1.0_r0 (중단됨, 재실행)
CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 1 \
    --with-reset-prob 0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --output-dir results/paper/phase1c_4ep/g1.0_r0 \
    --log-every 20

# GPU 3: g1.0_r1.0
CUDA_VISIBLE_DEVICES=3 $PYTHON paper_experiment.py \
    --mode fixed --method ttv1 --seed 42 \
    --epochs 4 --n-bits 14 --n-bits-slow 10 \
    --gamma 1.0 --units-in-mbatch true --transfer-every 1 \
    --with-reset-prob 1.0 --fast-lr 0.1 \
    --transfer-lr 1.0 --scale-transfer-lr false \
    --ln-lr 0.003 \
    --output-dir results/paper/phase1c_4ep/g1.0_r1.0 \
    --log-every 20
```

또는 런처 스크립트:
```bash
nohup bash launchers/phase1c_4ep.sh > /root/nohup_phase1c_4ep.log 2>&1 &
# 주의: 이미 완료된 실험의 output-dir 존재 시 덮어쓸 수 있음
# 완료된 결과 백업 후 실행 권장
```

---

## 3. 핵심 발견 요약 (중단 시점)

### TTv1 연속성 보장 transfer_lr=gamma

W_eff = gamma * W_fast + 1.0 * W_slow에서 transfer+reset 시 연속성 조건:
- `transfer_amount = gamma * W_fast` → `transfer_lr = gamma, scale_transfer_lr=False`
- 이전 실험(transfer_lr=1.0 고정)에서 gamma>0 성능 하락은 transfer 시 W_eff 불연속이 원인

### 4ep 결과 핵심
- **gamma=0.5 + reset=1.0 = F1 85.08** (TTv1 최고)
- **gamma=0.5 + reset=0 → ep4에서 73.9로 붕괴** (saturation)
- 작은 gamma(0.01, 0.05)는 reset이 오히려 해로움
- 큰 gamma(0.5+)는 reset 필수

### Mixed Precision 10-bit
- **QKVO: F1=87.61** (14-bit optuna best 87.53 초과)
- FFN: ~85+ (완료 확인 필요)

---

## 4. 다음 실험 계획

### Phase 3: Bit-Width Sweep
스크립트 준비 완료: `launchers/phase3_bit_sweep.sh`
- Single RPU: 8,10,12,14,16 bit
- TTv1: Fast={8,10,12,14,16}bit, Slow=10bit (best gamma/reset 사용)
- Mixed Precision: 10bit (완료/진행 중)

### 파일 위치
- 실험 드라이버: `experiments/paper/paper_experiment.py`
- RPU 설정: `experiments/paper/rpu_configs.py`
- 진단: `experiments/paper/update_diagnostics.py`
- 런처: `experiments/paper/launchers/`
- 결과: `experiments/paper/results/paper/`
- Optuna DB: `experiments/paper/results/optuna_dbs/`

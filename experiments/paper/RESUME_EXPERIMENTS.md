# 실험 재개 가이드

> 최종 업데이트: 2026-03-19
> Branch: transformer

## 환경 설정

```bash
cd /root/LRTT/experiments/paper
export PYTHON=/root/.venv310/bin/python
```

---

## 1. Mixed Precision 10-bit 결과

### 전체 결과

| Tag | Target | analog_lr | min_lr_rate | Best F1 | EM | 상태 |
|-----|--------|-----------|-------------|---------|------|------|
| mp_10b_qkvo | attention | 0.0357 | 0.5 | **87.61** | 79.87 | 완료 |
| mp_10b_all_minlr0.05 | all | 0.0357 | 0.05 | **87.35** | 80.17 | 완료 |
| mp_10b_all | all | 0.0357 | 0.5 | 86.84 | 79.15 | 완료 |
| mp_10b_ffn | ffn | 0.0357 | 0.5 | 86.09 | 78.14 | 완료 |
| mp_10b_ffn_alr0.00357 | ffn | 0.00357 | 0.5 | 83.21 | 73.91 | 완료 |
| mp_10b_all_alr0.00357 | all | 0.00357 | 0.5 | — | — | 중단 (1ep) |

### 관찰

- **QKVO (attention) layers가 최고 성능** (F1=87.61), 14-bit optuna best 87.53 초과
- all layers에서는 min_lr_rate=0.05가 0.5보다 우수 (87.35 vs 86.84)
- analog_lr=0.00357 (1/10)은 성능 하락이 큼
- 공통: classifier_lr=0.00076, ln_lr=0.00076, epochs=4, seed=42, n_bits=10

---

## 2. TTv1 (TikiTaka v1) 결과

### Phase 1C: Gamma × Reset Sweep (4ep, attention layers)

공통 설정: uim=true, te=1, fast_lr=0.1, scale_transfer_lr=false, transfer_lr=gamma
Fast=14bit, Slow=10bit, 4 epochs

| gamma | reset=0 (F1) | reset=1.0 (F1) |
|-------|-------------|----------------|
| 0.01 | 82.21 | 79.87 |
| 0.05 | **84.78** | 81.24 |
| 0.1 | 미완료 | 미완료 |
| 0.3 | 미완료 | 미완료 |
| 0.5 | 83.41 (ep4 붕괴→73.9) | **85.08** |
| 1.0 | 미완료 | 미완료 |

### TTv1 4ep Fixed (target layer 비교)

| Tag | Target | gamma | reset | Best F1 | EM | 상태 |
|-----|--------|-------|-------|---------|------|------|
| g1.0_r1.0_ffn | ffn (24 layers) | 1.0 | 1.0 | 84.93 | 76.95 | 완료 |
| g1.0_r1.0_all | all (72 layers) | 1.0 | 1.0 | — | — | OOM/CUBLAS 실패 |

### TTv1 All Layers OOM 문제

MIG 파티션 20GB 한도에서 72 layers TransferCompound `analog_tile.update()` 시 메모리 초과:
- batch_size=4, grad_accum=12 → OOM
- batch_size=2, grad_accum=24 → CUBLAS initialization failed
- batch_size=1, grad_accum=48 → 시도 중 (GPU 연결 끊김)

---

## 3. 핵심 발견 요약

### TTv1 연속성 보장 transfer_lr=gamma

W_eff = gamma * W_fast + 1.0 * W_slow에서 transfer+reset 시 연속성 조건:
- `transfer_amount = gamma * W_fast` → `transfer_lr = gamma, scale_transfer_lr=False`
- 이전 실험(transfer_lr=1.0 고정)에서 gamma>0 성능 하락은 transfer 시 W_eff 불연속이 원인

### 4ep 결과 핵심
- **gamma=0.5 + reset=1.0 = F1 85.08** (TTv1 attention 최고)
- **gamma=0.5 + reset=0 → ep4에서 73.9로 붕괴** (saturation)
- 작은 gamma(0.01, 0.05)는 reset이 오히려 해로움
- 큰 gamma(0.5+)는 reset 필수

### Mixed Precision 10-bit
- **QKVO: F1=87.61** (14-bit optuna best 87.53 초과)
- **All layers (minlr0.05): F1=87.35, EM=80.17** (EM 최고)
- FFN: F1=86.09

---

## 4. 남은 실험

### Phase 1C 미완료 (TTv1 Gamma×Reset)
- g0.1_r0, g0.1_r1.0, g0.3_r0, g0.3_r1.0, g1.0_r0 — 재실행 필요
- 런처: `launchers/phase1c_4ep.sh`

### TTv1 All Layers
- OOM 해결 필요 (더 큰 GPU 또는 gradient checkpointing)

### Phase 3: Bit-Width Sweep
스크립트 준비 완료: `launchers/phase3_bit_sweep.sh`
- Single RPU: 8,10,12,14,16 bit
- TTv1: Fast={8,10,12,14,16}bit, Slow=10bit (best gamma/reset 사용)
- Mixed Precision: 10bit (완료)

---

## 5. 파일 위치
- 실험 드라이버: `experiments/paper/paper_experiment.py`
- RPU 설정: `experiments/paper/rpu_configs.py`
- 런처: `experiments/paper/launchers/`
- 결과: `experiments/paper/results/paper/`
- Optuna DB: `experiments/paper/results/optuna_dbs/`

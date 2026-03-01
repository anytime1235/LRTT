# LRTT-LoRA Optuna Hyperparameter Sweep Guide

## Overview

`sweep_lrtt_lora_optuna.py`는 LRTT-LoRA의 `lora_alpha`와 `learning_rate`를 최적화하는 Optuna 기반 hyperparameter sweep 스크립트입니다.

## Features

### ✅ 자동 적용되는 LRTT-LoRA 기본 설정

다음 설정들은 `lrtt_lora_config.py`에 내장되어 있어 **자동으로 적용**됩니다:

**핵심 LoRA 설정:**
- ✅ `forward_inject=True` - LoRA composition 활성화
- ✅ C tile (base_layer) **freeze** - Pretrained weights 고정
- ✅ A/B tiles **trainable** - Low-rank adapters 학습
- ✅ `transfer disabled` - A/B 분리 유지 (collapse 방지)

**Device & IO 설정 (LRTT_setup.txt 기준):**
- ✅ Device: 6T1C LinearStepDevice (또는 FP mode에서 FloatingPointDevice)
- ✅ IO quantization: 8-bit (inp_res=1/(2^8-2), out_res=1/(2^8-2))
- ✅ Noise management: ABS_MAX
- ✅ Bound management: ITERATIVE

**Mapping & Scaling:**
- ✅ `weight_scaling_omega=1.0` - Backward hook 활성화
- ✅ `learn_out_scaling=True` - Output scaling 학습
- ✅ `out_scaling_columnwise=True` - Column-wise scaling

**Training 설정:**
- ✅ Optimizer: **AnalogSGD** (not Adam, as per LRTT_setup)
- ✅ Bias: **Frozen** (일반 layer bias)
- ✅ Classifier/qa_outputs: **Trainable** (weights + bias 모두)
- ✅ Early stopping: patience=3
- ✅ Warmup: Linear warmup (10% of total steps)

**Task 설정:**
- ✅ GLUE: max_seq_length=128
- ✅ SQuAD: max_seq_length=384, doc_stride=128

### 🔍 Sweep되는 Hyperparameters

| Parameter | Search Space | Type |
|-----------|--------------|------|
| `lora_alpha` | [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] | Categorical |
| `learning_rate` | [1e-5, 1e-3] | Log-uniform |

**Note:** `rank`는 기본값 8로 고정 (필요시 sweep 가능)

### 🎯 지원 모드

| Mode | Device | Alpha Scaling | 용도 |
|------|--------|---------------|------|
| `fp_lora` | FloatingPointDevice | 1.0x | Baseline, exact arithmetic |
| `sixt1c_lora` | 6T1C LinearStepDevice | 0.917x | Analog device simulation |

**Alpha Scaling:** 6T1C mode에서는 IO quantization으로 인한 스케일 차이를 보정하기 위해 alpha에 0.917배를 자동 적용합니다.

---

## Usage Examples

### 1. GLUE SST-2 with FP-LoRA

```bash
cd /data/LRTT_transformer/experiments

/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue \
  --task_name sst2 \
  --mode fp_lora \
  --n_trials 50
```

### 2. GLUE SST-2 with 6T1C-LoRA

```bash
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue \
  --task_name sst2 \
  --mode sixt1c_lora \
  --n_trials 50
```

### 3. GLUE MRPC (작은 데이터셋, 빠른 테스트)

```bash
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue \
  --task_name mrpc \
  --mode fp_lora \
  --n_trials 30
```

### 4. SQuAD with FP-LoRA

```bash
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task squad \
  --mode fp_lora \
  --n_trials 30
```

### 5. Parallel Execution (4 GPUs)

```bash
# 각 GPU에서 독립적으로 trial 실행
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue \
  --task_name sst2 \
  --mode fp_lora \
  --n_trials 100 \
  --n_jobs 4
```

### 6. Resume from Existing Study (SQLite DB)

```bash
# 첫 실행 (DB 생성)
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue \
  --task_name sst2 \
  --mode fp_lora \
  --n_trials 50 \
  --study_name sst2_fp_sweep \
  --storage sqlite:///lrtt_lora_optuna.db

# 중단 후 재개
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue \
  --task_name sst2 \
  --mode fp_lora \
  --n_trials 100 \
  --study_name sst2_fp_sweep \
  --storage sqlite:///lrtt_lora_optuna.db
```

### 7. Custom LoRA Rank and Target Modules

```bash
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue \
  --task_name sst2 \
  --mode fp_lora \
  --rank 16 \
  --target_modules query key value dense \
  --n_trials 50
```

---

## Command-Line Arguments

### Required Arguments

- `--task`: Task type (`glue` or `squad`)
- `--mode`: LoRA mode (`fp_lora` or `sixt1c_lora`)

### Optional Arguments

**Task Settings:**
- `--task_name`: GLUE task name (default: `sst2`)
  - Options: `sst2`, `mrpc`, `qqp`, `qnli`, `rte`, `wnli`, `cola`, `mnli`, `stsb`

**Model Settings:**
- `--rank`: LoRA rank (default: `8`)
- `--target_modules`: Target modules for LoRA (default: `query key value`)
  - Options: `query`, `key`, `value`, `dense`, `intermediate`, `output`

**Optuna Settings:**
- `--n_trials`: Number of trials (default: `50`)
- `--n_jobs`: Number of parallel jobs (default: `1`)
- `--study_name`: Optuna study name (default: auto-generated)
- `--storage`: Optuna storage URL (e.g., `sqlite:///optuna.db`)
- `--timeout`: Timeout in seconds (default: `None`)

---

## Output Files

### 1. Best Parameters (JSON)

```
/tmp/lrtt_lora_optuna_results/best_params_{study_name}.json
```

Example:
```json
{
  "best_trial": 23,
  "best_value": 0.9234,
  "best_params": {
    "lora_alpha": 1.0,
    "learning_rate": 0.0002134
  },
  "task": "glue",
  "task_name": "sst2",
  "mode": "fp_lora",
  "rank": 8
}
```

### 2. Checkpoints

```
/tmp/lrtt_lora_optuna_results/trial_{trial_number}/
```

### 3. Optuna Database (if using --storage)

```
lrtt_lora_optuna.db
```

---

## Monitoring with Weights & Biases (W&B)

스크립트는 자동으로 W&B 통합을 시도합니다:

```bash
# W&B 설치 및 로그인 (선택 사항)
pip install wandb
wandb login

# W&B 없이 실행하려면 그냥 실행 (자동으로 비활성화)
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py ...
```

---

## Expected Performance

### GLUE SST-2 (based on prior experiments)

| Mode | Best Alpha | Best LR | Expected Accuracy |
|------|------------|---------|-------------------|
| FP-LoRA | 0.5-1.0 | ~2e-4 | 91-92% |
| 6T1C-LoRA | 0.5-1.0 (scaled) | ~2e-4 | 90-91% |

### GLUE MRPC (small dataset, faster)

| Mode | Best Alpha | Best LR | Expected F1 |
|------|------------|---------|-------------|
| FP-LoRA | 1.0-2.0 | ~5e-4 | 88-90% |
| 6T1C-LoRA | 1.0-2.0 (scaled) | ~5e-4 | 87-89% |

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution:**
- Reduce batch size (edit `BATCH_SIZE` in script)
- Use gradient accumulation
- Reduce `--n_jobs` (parallel workers)

### Issue 2: Very Slow Training

**Cause:** Full GLUE SST-2 has 67K samples

**Solution:** Use smaller tasks first for quick testing:
```bash
# MRPC has only 3.7K samples
--task_name mrpc
```

### Issue 3: All Trials Fail

**Check:**
1. Correct Python environment: `/data/venvs/aihwkit_gpu/bin/python`
2. LRTT-LoRA modules accessible (paths added in script)
3. GPU available: `torch.cuda.is_available()`

### Issue 4: NaN Loss

**Cause:** `lora_alpha=0.0` disables LoRA (no gradients)

**Solution:** Alpha search space starts at 0.1 (safe)

---

## Advanced: Customizing Search Space

Edit the script to modify search spaces:

```python
# Line ~96
LORA_ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Modify this

# In objective() function
lora_alpha = trial.suggest_categorical("lora_alpha", LORA_ALPHAS)
lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)  # Modify range

# Add rank search (optional)
rank = trial.suggest_categorical("rank", [4, 8, 16, 32])
```

---

## Verification of LRTT-LoRA Settings

스크립트 실행 시 다음이 자동 적용됨을 확인:

```bash
# 실행 로그에서 확인:
[2/4] Converting 24 layers to LRTT-LoRA...
  - Forward: y = C·x + α·A·(B·x)         # forward_inject=True
  - Trainable: A, B tiles                # A/B trainable
  - Frozen: C tile (pretrained weights)  # C frozen
  - LoRA rank: 8
  - LoRA alpha: 1.0
```

---

## Comparison: FP-LoRA vs 6T1C-LoRA

권장 사용 시나리오:

### FP-LoRA (FloatingPointDevice)
- ✅ Baseline 성능 확인
- ✅ Hyperparameter 범위 탐색
- ✅ 빠른 프로토타이핑
- ✅ 최대 정확도 달성

### 6T1C-LoRA (6T1C LinearStepDevice)
- ✅ Analog hardware 시뮬레이션
- ✅ Hardware constraints 고려
- ✅ Device noise/variation 효과 분석
- ✅ Real deployment 준비

**권장 워크플로우:**
1. FP-LoRA로 최적 hyperparameters 찾기
2. 6T1C-LoRA로 재검증 및 fine-tuning
3. Alpha scaling (0.917x) 적용하여 비교

---

## Quick Start Checklist

- [ ] Python 환경 확인: `/data/venvs/aihwkit_gpu/bin/python`
- [ ] GPU 사용 가능 확인
- [ ] Task 선택 (glue/squad)
- [ ] Mode 선택 (fp_lora/sixt1c_lora)
- [ ] Trial 수 결정 (빠른 테스트: 20-30, 본격 탐색: 50-100)
- [ ] (Optional) W&B 설정
- [ ] (Optional) DB storage 설정
- [ ] 스크립트 실행!

---

## Example Complete Run

```bash
# 1. Quick test with MRPC (small dataset)
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue --task_name mrpc \
  --mode fp_lora --n_trials 20

# 2. Full sweep with SST-2
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue --task_name sst2 \
  --mode fp_lora --n_trials 50 \
  --study_name sst2_fp_full \
  --storage sqlite:///optuna_sst2.db

# 3. Verify with 6T1C mode
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
  --task glue --task_name sst2 \
  --mode sixt1c_lora --n_trials 30 \
  --study_name sst2_6t1c \
  --storage sqlite:///optuna_sst2.db

# 4. Check results
cat /tmp/lrtt_lora_optuna_results/best_params_sst2_fp_full.json
cat /tmp/lrtt_lora_optuna_results/best_params_sst2_6t1c.json
```

**Done!** 🎉

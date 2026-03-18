# SA Mixed-Precision E2E Experiments

## Overview

QZR-guided sensitivity-aware mixed-precision IO resolution allocation 결과를 사용한 full 4-epoch fine-tuning 실험.

**현재 실행 분배:**
- **Server A** (현재): avg 9b, 8b, 7b (순차 실행 중)
- **Server B** (로컬): avg 6b, 5b (이 가이드 참고)

## Prerequisites

```bash
# 1. Clone & checkout
git clone https://github.com/nmdlkg/LRTT.git
cd LRTT
git checkout transformer

# 2. Python environment (3.10)
python -m venv .venv310
source .venv310/bin/activate

# 3. Install aihwkit (GPU wheel)
pip install aihwkit-1.0.0+cuda121-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 4. Install dependencies
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm matplotlib pandas
```

## Run 6b and 5b experiments

```bash
cd experiments/paper
source ../../.venv310/bin/activate

COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 4 --seed 42 --mode fixed \
  --min-lr-rate 0.05"

# === avg 6b ===
nohup python paper_experiment.py $COMMON \
  --io-bits 8 \
  --per-layer-bits sa_v4/sensitivity_allocation/precision_map_budget6.json \
  --output-dir sa_v4/e2e_budget6 \
  > sa_v4/e2e_budget6.log 2>&1 &
echo "6b PID: $!"

# === avg 5b (after 6b finishes, or on another GPU) ===
nohup python paper_experiment.py $COMMON \
  --io-bits 8 \
  --per-layer-bits sa_v4/sensitivity_allocation/precision_map_budget5.json \
  --output-dir sa_v4/e2e_budget5 \
  > sa_v4/e2e_budget5.log 2>&1 &
echo "5b PID: $!"
```

## Precision Map Summary

| Budget | FFN1 | K | Q | V | O | FFN2 |
|:------:|:----:|:---:|:---:|:---:|:---:|:----:|
| 5b | 4.5 | 6.0 | 5.7 | 5.8 | 4.1 | 4.0 |
| 6b | 6.9 | 6.9 | 6.4 | 6.2 | 4.9 | 4.6 |
| 7b | 9.4 | 7.8 | 7.4 | 7.2 | 5.2 | 5.0 |
| 8b | 10.8 | 8.7 | 8.2 | 8.2 | 6.1 | 6.0 |
| 9b | 11.8 | 9.8 | 9.3 | 9.2 | 6.9 | 7.0 |

## Training config

| Parameter | Value |
|-----------|-------|
| Model | bert-base-uncased |
| Task | SQuAD v1.1 |
| Epochs | 4 |
| Batch size | 24 (effective 48 with grad_accum=2) |
| Analog LR | 0.0357 |
| Classifier/LN LR | 0.00076 |
| Min LR rate | 0.05 |
| Warmup ratio | 0.05 |
| Seed | 42 |
| IO resolution | per-layer mixed (see precision maps) |
| Forward/Backward | tied (same resolution) |
| Noise management | ABS_MAX |

## Output structure

```
sa_v4/
  e2e_budget5/
    config.json          # experiment config
    training_log.csv     # step, epoch, loss, f1, em, ...
    summary.json         # best_f1, final_f1, final_em
  e2e_budget6/
    ...
```

## Monitoring

```bash
# Check progress
tail -f sa_v4/e2e_budget6.log

# Check latest metrics
tail -5 sa_v4/e2e_budget6/training_log.csv
```

## After completion

결과를 push하면 Server A에서 통합 분석:
```bash
git add sa_v4/e2e_budget5 sa_v4/e2e_budget6
git commit -m "SA e2e results: 5b, 6b"
git push origin transformer
```

## Offline allocator (추가 budget 생성)

모델 재실행 없이 임의 budget의 precision map 즉시 생성:
```bash
python paper_figures_v4.py --offline-allocate \
  --target-avg-bits 5.5 6.5 7.5 \
  --out-dir sa_v4
```

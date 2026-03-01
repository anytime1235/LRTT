# TikiTaka v2 GLUE Results Summary

## Model: google/mobilebert-uncased
## Method: TikiTaka v2 (ChoppedTransferCompound)
## Target: QKV only (query, key, value)

---

## SST-2 Best Results

| Experiment | Optimizer | Classifier | Pruning | Best Acc | Note |
|-----------|-----------|------------|---------|----------|------|
| 50-trial sweep | AnalogAdam | trainable | ON | **90.14%** | 비교용 baseline (LRTT_setup 기준 아님) |
| 50-trial sweep (killed) | AnalogSGD | frozen(seed=42) | ON | 86.58% | 19 trial에서 중단 |
| 50-trial sweep (running) | AnalogSGD | frozen(seed=42) | OFF | 85.55%* | *실행중, 업데이트 필요 |

## Ablation Study (AnalogAdam, SST-2 best HP)

| Condition | Accuracy | Delta |
|-----------|----------|-------|
| Full (classifier trainable + te=491) | 90.14% | baseline |
| Classifier frozen + te=491 | 85.32% | -4.82%p |
| Classifier frozen + te=10^7 | 76.38% | -13.76%p |
| AnalogSGD (classifier trainable + te=491) | 82.45% | -7.69%p |

### Component Contribution
- **out_scaling_alpha only**: 76.38%
- **C tile transfer**: +8.94%p
- **Classifier training**: +4.82%p

## Files
- `sst2_sweep_analogadam_batch64_bias_frozen.json` - AnalogAdam 50-trial sweep
- `sst2_sweep_analogsgd_batch64_nopruning.json` - AnalogSGD 50-trial sweep (실행중)
- `sst2_sweep_analogsgd_batch64_pruning.json` - AnalogSGD sweep (pruning, 중단됨)
- `sst2_ablation_studies.json` - Ablation experiments
- `glue_eval_analogadam_sst2_best_hp.json` - GLUE 5-task eval

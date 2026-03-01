# TikiTaka v2 Sweep Analysis Report

**Date:** 2026-02-05
**Model:** google/mobilebert-uncased
**Hardware:** NVIDIA H200 GPU

## 1. Sweep Execution Summary

| Parameter | Value |
|-----------|-------|
| Script | `sweep_tikitaka_all_tasks.py` |
| Trials per Task | 10 |
| Epochs per Trial | 1 |
| Parallel Jobs | 2 |
| Status | **Hung after COLA** (killed PID 90985) |

## 2. Completed Tasks Results

### Performance Summary

| Task | Best Trial | Best Value | Metric | Start Point | Improvement | Digital Baseline | Gap |
|------|------------|------------|--------|-------------|-------------|------------------|-----|
| RTE | 1 | 0.5199 | accuracy | 0.473 | +4.7%p | 0.5632 | -4.3%p |
| MRPC | 0 | 0.6544 | F1 | 0.316 | +33.8%p | 0.9004 | -24.6%p |
| STSB | 0 | -0.0532 | spearman | -0.042 | -1.1%p | 0.8771 | **FAILED** |
| COLA | 8 | 0.6932 | matthews | 0.309 | +38.4%p | 0.0* | N/A |

*COLA digital baseline 0.0 appears to be a measurement error.

### Best Hyperparameters by Task

#### RTE (Trial 1, Accuracy: 0.5199)
```
learning_rate: 0.000136
transfer_lr: 2.378
transfer_every: 20
fast_lr: 0.487
auto_granularity: 169.64
in_chop_prob: 0.035
```

#### MRPC (Trial 0, F1: 0.6544)
```
learning_rate: 0.000131
transfer_lr: 7.36
transfer_every: 160
fast_lr: 0.86
auto_granularity: 306.0
in_chop_prob: 0.02
```

#### COLA (Trial 8, Matthews: 0.6932)
```
learning_rate: 0.000194
transfer_lr: 2.637
transfer_every: 74
fast_lr: 0.457
auto_granularity: 340.91
in_chop_prob: 0.029
```

## 3. STSB Failure Analysis

### Training Loss Comparison
| Task | Final Loss | Task Type |
|------|------------|-----------|
| STSB | 75,118,642,462,720 | Regression |
| RTE | 2,805,440 | Classification |
| MRPC | 89,293 | Classification |
| COLA | 37,272 | Classification |

### Root Cause
1. **STSB is a Regression Task** (predicts continuous values 0-5 using MSELoss)
2. **Analog Output Scale Mismatch**: Target range (0-5) mismatched with model output
3. **MSE Loss Explosion**: Squared errors cause exponential growth
4. **Classification vs Regression Stability**:
   - Classification: Softmax normalizes output → stable
   - Regression: Direct value prediction → sensitive to analog noise

### Recommended Fixes
1. Apply output scaling (sigmoid × 5)
2. Normalize labels to 0-1 range
3. Task-specific hyperparameter search for STSB
4. Exclude STSB and use classification tasks only

## 4. Process Hang Analysis

### Timeline
- COLA Trial 9 completed at 16:01:04
- Process hung during SST2 transition
- Process hung for ~10 hours before kill

### Error Logs
```
wandb.errors.UsageError: Run (xxx) is finished.
The call to `log` will be ignored.
```

### Root Causes
1. **Parallel Job (n_jobs=2) Deadlock**: Worker thread cleanup hung
2. **WandB Session Management**: Parallel execution causes run termination conflicts
3. **Optuna Study Transition**: COLA → SST2 transition hung

## 5. Pending Tasks

| Task | Dataset Size | Expected Duration |
|------|-------------|-------------------|
| SST2 | 67K | ~10 trials |
| QNLI | 105K | ~10 trials |
| QQP | 364K | ~10 trials |
| MNLI | 393K | ~10 trials |
| SQUAD | 87K (10K subset) | ~10 trials |

## 6. Recommendations

### Immediate Actions
1. ✅ Kill hung process (PID 90985) - **DONE**
2. ✅ Save completed task results - **DONE**
3. Re-run remaining tasks with `--n_jobs 1`

### Re-run Command
```bash
cd /data/LRTT_transformer/experiments
/data/venvs/aihwkit_gpu/bin/python sweep_tikitaka_all_tasks.py \
    --tasks sst2 qnli qqp mnli squad \
    --n_trials 10 \
    --n_jobs 1
```

### Alternative: Skip SST2 if time-constrained
```bash
/data/venvs/aihwkit_gpu/bin/python sweep_tikitaka_all_tasks.py \
    --tasks qnli qqp mnli squad \
    --n_trials 10 \
    --n_jobs 1
```

## 7. Parameter Patterns Observed

| Parameter | Effective Range | Observation |
|-----------|-----------------|-------------|
| learning_rate | 0.0001 - 0.0002 | Lower end of search space works better |
| transfer_lr | 2.4 - 7.4 | Mid-range values effective |
| transfer_every | 20 - 160 | Task-dependent; smaller for simpler tasks |
| fast_lr | 0.46 - 0.86 | Mid-range values work well |
| auto_granularity | 170 - 341 | Lower to mid values effective |
| in_chop_prob | 0.02 - 0.035 | Around 0.02-0.035 seems optimal |

## 8. Files Generated

- `/data/tikitaka_sweep/sweep_results_20260204.json` - Complete sweep results
- `/data/tikitaka_sweep/best_params_summary.json` - Best parameters per task
- `/data/tikitaka_sweep/analysis_report.md` - This analysis report

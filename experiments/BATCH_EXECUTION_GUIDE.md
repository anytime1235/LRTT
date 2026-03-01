# LRTT-LoRA Batch Execution Guide

**Updated**: 2026-02-11
**Status**: ✅ Ready to execute

---

## 📋 Overview

자동화된 hyperparameter sweep 실행:

### Experiments
1. **SQuAD**: 15 epochs, 30 trials
2. **GLUE** (9 tasks): 3 epochs, 10 trials each
   - Order: SST-2, QQP, MNLI, QNLI, MRPC, RTE, CoLA, STS-B, WNLI

### Settings
- **Mode**: `sixt1c_lora` (6T1C-LoRA)
- **Search Space**:
  - `lora_alpha`: [0.01, 100] log-uniform
  - `learning_rate`: [1e-4, 1e-1] log-uniform
- **Fixed**:
  - `rank`: 8
  - `target_modules`: query, key, value
  - `batch_size`: 32 (from LRTT_setup.txt)

### Results
- **Location**: `/data/results/lora_baseline/`
- **Format**: JSON files with hyperparameters and results
- **Databases**: SQLite (Optuna) for each task
- **Logs**: Individual log files per task

---

## 🚀 Quick Start

### 1. Start Batch Execution

```bash
cd /data/LRTT_transformer/experiments

# Start all experiments (runs with nohup)
bash run_lrtt_lora_sweep_batch.sh
```

**서버 연결이 끊겨도 계속 실행됩니다!**

### 2. Monitor Progress

```bash
# Check overall progress
bash monitor_sweep_progress.sh

# Watch specific log
tail -f /data/results/lora_baseline/logs/squad_sixt1c_lora_*.log

# Watch batch log
tail -f /data/results/lora_baseline/logs/batch_run_*.log
```

### 3. Check Running Processes

```bash
# List all sweep processes
ps aux | grep sweep_lrtt_lora_optuna.py

# Check resource usage
htop  # or top
```

---

## 📊 Results Structure

```
/data/results/lora_baseline/
├── logs/
│   ├── batch_run_YYYYMMDD_HHMMSS.log
│   ├── squad_sixt1c_lora_YYYYMMDD_HHMMSS.log
│   ├── glue_sst2_sixt1c_lora_YYYYMMDD_HHMMSS.log
│   ├── glue_qqp_sixt1c_lora_YYYYMMDD_HHMMSS.log
│   └── ...
├── optuna_squad_sixt1c_lora.db
├── optuna_glue_sst2_sixt1c_lora.db
├── optuna_glue_qqp_sixt1c_lora.db
├── ...
├── best_params_squad_sixt1c_lora_*.json
├── best_params_glue_sst2_sixt1c_lora_*.json
└── batch_summary_YYYYMMDD_HHMMSS.txt
```

---

## 📈 Expected Timeline

| Task | Epochs | Trials | Est. Time per Trial | Total Time |
|------|--------|--------|---------------------|------------|
| SQuAD | 15 | 30 | ~30-40 min | ~15-20 hours |
| SST-2 | 3 | 10 | ~15-20 min | ~2.5-3 hours |
| QQP | 3 | 10 | ~20-25 min | ~3-4 hours |
| MNLI | 3 | 10 | ~20-25 min | ~3-4 hours |
| QNLI | 3 | 10 | ~15-20 min | ~2.5-3 hours |
| MRPC | 3 | 10 | ~5-8 min | ~1 hour |
| RTE | 3 | 10 | ~5-8 min | ~1 hour |
| CoLA | 3 | 10 | ~5-8 min | ~1 hour |
| STS-B | 3 | 10 | ~5-8 min | ~1 hour |
| WNLI | 3 | 10 | ~3-5 min | ~30 min |

**Total Estimated Time**: ~30-40 hours (1.5-2 days)

---

## 🔍 Monitoring Commands

### Real-time Monitoring

```bash
# Overall progress
watch -n 60 'bash monitor_sweep_progress.sh'

# Current task log (updates every 10 sec)
watch -n 10 'tail -20 /data/results/lora_baseline/logs/squad_*.log'

# Trial count in database (requires sqlite3)
sqlite3 /data/results/lora_baseline/optuna_squad_sixt1c_lora.db \
    "SELECT COUNT(*) FROM trials;"

# Best trial so far
sqlite3 /data/results/lora_baseline/optuna_squad_sixt1c_lora.db \
    "SELECT trial_id, value FROM trial_values ORDER BY value DESC LIMIT 5;"
```

### Check Completion

```bash
# Count completed tasks (check for JSON files)
ls /data/results/lora_baseline/best_params_*.json | wc -l

# Expected: 10 files (1 SQuAD + 9 GLUE tasks)
```

---

## ⚙️ Configuration Verification

### Batch Size
✅ **32** (matches LRTT_setup.txt)

### Optimizer
✅ **AnalogSGD** (as per LRTT_setup.txt)

### Device & IO
✅ All settings from LRTT_setup.txt applied via `lrtt_lora_config.py`:
- forward_inject=True
- out_noise=0.0
- noise_management=ABS_MAX
- bound_management=ITERATIVE
- weight_scaling_omega=1.0
- learn_out_scaling=True

### Training
✅ Warmup: 10% linear
✅ Early stopping: patience=3
✅ Scheduler: Linear with warmup

---

## 🛑 Stop/Pause Execution

### Stop All Sweeps

```bash
# Find and kill all sweep processes
pkill -f sweep_lrtt_lora_optuna.py

# Or manually kill specific PID
kill <PID>
```

### Resume Later

The experiments use SQLite storage, so you can resume:

```bash
# Resume from existing database
/data/venvs/aihwkit_gpu/bin/python sweep_lrtt_lora_optuna.py \
    --task squad \
    --mode sixt1c_lora \
    --n_trials 30 \
    --study_name squad_sixt1c_lora_20260211_120000 \
    --storage sqlite:////data/results/lora_baseline/optuna_squad_sixt1c_lora.db
```

---

## 📝 Results Analysis

### View Best Parameters (Python)

```python
import json

# Load best parameters
with open('/data/results/lora_baseline/best_params_squad_sixt1c_lora_*.json') as f:
    best = json.load(f)

print(f"Best trial: {best['best_trial']}")
print(f"Best value: {best['best_value']}")
print(f"Best params: {best['best_params']}")
```

### Query Optuna Database (Python)

```python
import optuna

study = optuna.load_study(
    study_name='squad_sixt1c_lora_20260211_120000',
    storage='sqlite:////data/results/lora_baseline/optuna_squad_sixt1c_lora.db'
)

print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Get all trials
df = study.trials_dataframe()
print(df.head())
```

---

## 🐛 Troubleshooting

### Issue 1: Out of Memory

**Symptoms**: CUDA OOM error in logs

**Solutions**:
1. Reduce batch size (edit script: `BATCH_SIZE = 16`)
2. Reduce max_seq_length
3. Run tasks sequentially instead of parallel

### Issue 2: NaN Loss

**Symptoms**: Loss becomes NaN during training

**Causes**:
- lora_alpha too high/low
- Learning rate too high

**Action**: Optuna will prune these trials automatically

### Issue 3: Process Died

**Check**:
```bash
# Check if still running
ps aux | grep sweep_lrtt_lora_optuna.py

# Check system logs
dmesg | tail -50

# Check disk space
df -h /data/results/
```

### Issue 4: Slow Progress

**Normal**: Some tasks (SQuAD, large GLUE) take hours per trial

**Check**: Monitor via logs to confirm it's progressing

---

## ✅ Completion Checklist

- [ ] Batch script executed
- [ ] All 10 tasks started (check logs/)
- [ ] Monitor script running
- [ ] SQuAD completed (30 trials)
- [ ] SST-2 completed (10 trials)
- [ ] QQP completed (10 trials)
- [ ] MNLI completed (10 trials)
- [ ] QNLI completed (10 trials)
- [ ] MRPC completed (10 trials)
- [ ] RTE completed (10 trials)
- [ ] CoLA completed (10 trials)
- [ ] STS-B completed (10 trials)
- [ ] WNLI completed (10 trials)
- [ ] All JSON files created (10 files)
- [ ] Summary file generated

---

## 🎯 Next Steps After Completion

1. **Analyze Results**:
   ```bash
   cd /data/results/lora_baseline
   cat batch_summary_*.txt
   ```

2. **Compare Best Parameters**:
   ```bash
   for f in best_params_*.json; do
       echo "=== $f ==="
       cat $f
   done
   ```

3. **Run Single Training** with best parameters:
   ```bash
   # Use best params from JSON files
   /data/venvs/aihwkit_gpu/bin/python run_single_training.py \
       --task sst2 \
       --mode sixt1c_lora \
       --lora_alpha <best_alpha> \
       --lr <best_lr> \
       --rank 8
   ```

---

**Execution Command**:
```bash
cd /data/LRTT_transformer/experiments
bash run_lrtt_lora_sweep_batch.sh
```

**Monitor Command**:
```bash
bash monitor_sweep_progress.sh
```

**Ready to launch!** 🚀

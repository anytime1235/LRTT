# FP-LoRA Batch Execution Guide

**Created**: 2026-02-11
**Status**: ✅ Ready to execute

---

## 📋 Overview

FP-LoRA (FloatingPointDevice) 모드로 동일한 hyperparameter sweep 실행:

### Experiments
1. **SQuAD**: 15 epochs, 30 trials
2. **GLUE** (9 tasks): 3 epochs, 10 trials each
   - Order: SST-2, QQP, MNLI, QNLI, MRPC, RTE, CoLA, STS-B, WNLI

### Settings (Identical to 6T1C-LoRA)
- **Mode**: `fp_lora` (FloatingPointDevice)
- **Search Space**:
  - `lora_alpha`: [0.01, 100] log-uniform
  - `learning_rate`: [1e-4, 1e-1] log-uniform
- **Fixed**:
  - `rank`: 8
  - `target_modules`: query, key, value
  - `batch_size`: 32

### Results
- **Location**: `/data/results/lora_fp/`
- **Format**: JSON files with hyperparameters and results
- **Databases**: SQLite (Optuna) for each task
- **Logs**: Individual log files per task

---

## 🚀 Execution

### 1. Start Batch Execution

```bash
cd /data/LRTT_transformer/experiments

# Start FP-LoRA sweeps (runs with nohup)
bash run_lrtt_lora_sweep_batch_fp.sh
```

**서버 연결이 끊겨도 계속 실행됩니다!**

### 2. Monitor Progress

```bash
# Check FP-LoRA progress
bash monitor_sweep_progress_fp.sh

# Watch specific log
tail -f /data/results/lora_fp/logs/squad_fp_lora_*.log

# Watch batch log
tail -f /data/results/lora_fp/logs/batch_run_*.log
```

### 3. Check Running Processes

```bash
# List FP-LoRA sweep processes
ps aux | grep sweep_lrtt_lora_optuna.py | grep fp_lora

# Check resource usage
htop
```

---

## 📊 Results Structure

```
/data/results/lora_fp/
├── logs/
│   ├── batch_run_YYYYMMDD_HHMMSS.log
│   ├── squad_fp_lora_YYYYMMDD_HHMMSS.log
│   ├── glue_sst2_fp_lora_YYYYMMDD_HHMMSS.log
│   └── ...
├── optuna_squad_fp_lora.db
├── optuna_glue_sst2_fp_lora.db
├── ...
├── best_params_squad_fp_lora_*.json
├── best_params_glue_sst2_fp_lora_*.json
└── batch_summary_YYYYMMDD_HHMMSS.txt
```

---

## ⚖️ Comparison with 6T1C-LoRA

### Differences
| Setting | FP-LoRA | 6T1C-LoRA |
|---------|---------|-----------|
| **Device** | FloatingPointDevice | LinearStepDevice (6T1C) |
| **Alpha Scaling** | None (1.0x) | 0.917x |
| **Arithmetic** | Exact (no quantization) | 8-bit quantization |
| **Results Directory** | `/data/results/lora_fp/` | `/data/results/lora_baseline/` |

### Identical Settings
- Search space: alpha [0.01, 100] log, lr [1e-4, 1e-1] log
- Batch size: 32
- Epochs: SQuAD=15, GLUE=3
- Trials: SQuAD=30, GLUE=10
- Target modules: query, key, value
- Optimizer: AnalogSGD
- Warmup, early stopping, scheduler settings

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

## 🛑 Stop/Pause Execution

### Stop FP-LoRA Sweeps

```bash
# Find and kill FP-LoRA processes
pkill -f "sweep_lrtt_lora_optuna.py.*fp_lora"

# Or manually kill specific PID
kill <PID>
```

---

## 📝 Key Commands Summary

```bash
# Execute
cd /data/LRTT_transformer/experiments
bash run_lrtt_lora_sweep_batch_fp.sh

# Monitor
bash monitor_sweep_progress_fp.sh

# View logs
tail -f /data/results/lora_fp/logs/squad_fp_lora_*.log
```

---

**Ready to launch!** 🚀

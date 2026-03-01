# LRTT Setup Comparison (sweep_lrtt_lora_optuna.py vs LRTT_setup.txt)

**Date**: 2026-02-11
**Purpose**: Final verification that all settings match LRTT_setup.txt

---

## ✅ Matching Settings

| Setting | LRTT_setup.txt | sweep_lrtt_lora_optuna.py | Status |
|---------|----------------|----------------------------|--------|
| **Batch Size** | 32 | 32 (`BATCH_SIZE = 32`) | ✅ MATCH |
| **Max Seq Length (GLUE)** | 128 | 128 (`TASK_CONFIGS["glue"]`) | ✅ MATCH |
| **Max Seq Length (SQuAD)** | 384 | 384 (`TASK_CONFIGS["squad"]`) | ✅ MATCH |
| **Doc Stride (SQuAD)** | 128 | 128 (`TASK_CONFIGS["squad"]`) | ✅ MATCH |
| **Optimizer** | AnalogSGD | AnalogSGD | ✅ MATCH |
| **Bias** | Frozen | Frozen (via lrtt_lora_conversion) | ✅ MATCH |
| **Classifier** | Trainable | Trainable (via lrtt_lora_conversion) | ✅ MATCH |
| **Classifier Bias** | Trainable | Trainable (via lrtt_lora_conversion) | ✅ MATCH |

---

## Device & IO Settings (Applied via lrtt_lora_config.py)

| Setting | LRTT_setup.txt | lrtt_lora_config.py | Status |
|---------|----------------|---------------------|--------|
| **forward_inject** | True | True | ✅ MATCH |
| **C-tile** | Frozen | Frozen (SoftBoundsDevice) | ✅ MATCH |
| **A/B tiles** | Trainable | Trainable (6T1C/FloatingPoint) | ✅ MATCH |
| **IO quantization** | 8-bit | 8-bit (inp_res=out_res=1/254) | ✅ MATCH |
| **out_noise** | 0.0 | 0.0 (forward & backward) | ✅ MATCH |
| **noise_management** | ABS_MAX | ABS_MAX | ✅ MATCH |
| **bound_management** | ITERATIVE | ITERATIVE | ✅ MATCH |
| **weight_scaling_omega** | 1.0 | 1.0 | ✅ MATCH |
| **learn_out_scaling** | True | True | ✅ MATCH |
| **out_scaling_columnwise** | True | True | ✅ MATCH |

---

## Training Settings

| Setting | LRTT_setup.txt | sweep_lrtt_lora_optuna.py | Status |
|---------|----------------|----------------------------|--------|
| **Warmup** | Linear warmup | Linear warmup (10% of steps) | ✅ MATCH |
| **Early stopping** | Patience=3 | Patience=3 (`EarlyStoppingCallback`) | ✅ MATCH |
| **Scheduler** | Linear with warmup | `get_linear_schedule_with_warmup` | ✅ MATCH |

---

## ⚠️ Differences (Expected/Intentional)

| Setting | LRTT_setup.txt | sweep_lrtt_lora_optuna.py | Reason |
|---------|----------------|----------------------------|--------|
| **Epochs (GLUE)** | Not specified | 15 (configurable) | Hyperparameter sweep default |
| **Epochs (SQuAD)** | Not specified | 3 (configurable) | Standard SQuAD training |
| **Logging steps** | Not specified | 100 | For monitoring |
| **Target modules** | Not specified | ["query", "key", "value"] (default) | LoRA target layers |
| **Rank** | Not specified | 8 (default, configurable) | LoRA rank |
| **Alpha search** | Not specified | [0.01, 100] log-uniform | Hyperparameter to sweep |
| **LR search** | Not specified | [1e-4, 1e-1] log-uniform | Hyperparameter to sweep |

---

## Additional Features in sweep_lrtt_lora_optuna.py

**Not in LRTT_setup.txt but added for functionality:**

1. **Optuna Integration**:
   - TPE Sampler
   - Median Pruner
   - Early trial pruning

2. **Multiple Modes**:
   - FP-LoRA (FloatingPoint)
   - 6T1C-LoRA (with 0.917x alpha scaling)

3. **Task Support**:
   - GLUE (all tasks)
   - SQuAD

4. **Monitoring**:
   - W&B integration (optional)
   - SQLite storage for resuming

5. **Flexibility**:
   - Configurable target modules
   - Configurable rank
   - Parallel execution support

---

## ✅ Final Verdict

**ALL CRITICAL SETTINGS FROM LRTT_setup.txt ARE CORRECTLY IMPLEMENTED**

The differences are intentional additions for:
- Hyperparameter search functionality
- Multi-task support
- Experiment tracking
- Flexibility

**Batch Size**: ✅ 32 (matches LRTT_setup.txt)

**Core Settings**: ✅ All match (device, IO, mapping, optimizer, trainability)

**Status**: ✅ **READY FOR USE**

---

**Checked by**: Manual comparison
**Date**: 2026-02-11
**Result**: ✅ All settings verified

# LRTT Transformer Environment Setup Guide

## Overview
This document describes how to reproduce the LRTT transformer training environment.

## System Requirements
- Python 3.10 (venv uses Python 3.10)
- CUDA 12.1
- Linux (manylinux2014_x86_64)

---

## Step 1: Create Virtual Environment

```bash
# Create Python 3.10 virtual environment
python3.10 -m venv ~/.venv310
source ~/.venv310/bin/activate
```

---

## Step 2: Install PyTorch with CUDA 12.1

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 3: Install aihwkit-gpu (CUDA 12.1)

Option A: From wheel file (if available)
```bash
pip install /path/to/aihwkit-1.0.0+cuda121-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Option B: Build from source
```bash
git clone https://github.com/IBM/aihwkit.git
cd aihwkit
pip install cmake
pip install . --install-option="--with-cuda"
```

---

## Step 4: Clone LRTT Repository

```bash
# Clone the main LRTT repo (contains LRTT configs)
git clone https://github.com/nmdlkg/LRTT.git
cd LRTT
git checkout main  # or specific branch

# Clone LRTT_transformer repo (transformer branch)
git clone https://github.com/nmdlkg/LRTT.git LRTT_transformer
cd LRTT_transformer
git checkout transformer
```

---

## Step 5: Install Python Dependencies

```bash
pip install transformers==4.47.1
pip install datasets==4.5.0
pip install evaluate==0.4.6
pip install optuna==4.7.0
pip install wandb==0.24.1
pip install accelerate==1.12.0
pip install scipy==1.15.3
pip install scikit-learn==1.7.2
pip install matplotlib==3.10.8
pip install pandas==2.3.3
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

---

## Step 6: Setup LRTT Python Path

The LRTT configs are imported via sys.path in each script:
```python
import sys
sys.path.insert(0, '/path/to/LRTT/src')

from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
```

**Important files in LRTT/src/aihwkit/simulator/:**
- `configs/lrtt_config.py` - LRTT configuration classes
- `configs/lrtt_python.py` - PythonLRTTDevice implementation
- `configs/lrtt_rpu_config.py` - RPU config wrapper
- `tiles/lrtt_controller.py` - LRTT controller (3-tile orchestrator)

---

## Directory Structure

```
/data/
├── LRTT/                          # Main LRTT source (for configs)
│   └── src/
│       └── aihwkit/
│           └── simulator/
│               ├── configs/
│               │   ├── lrtt_config.py
│               │   ├── lrtt_python.py
│               │   └── lrtt_rpu_config.py
│               └── tiles/
│                   └── lrtt_controller.py
│
├── LRTT_transformer/              # Transformer experiments (transformer branch)
│   ├── LRTT_glue/                 # SQuAD/GLUE sweep scripts
│   │   ├── sweep_lrtt_squad_rank8.py
│   │   ├── sweep_lrtt_squad_rank8_sgd.py
│   │   ├── sweep_sixt1c_lora_squad_adam.py
│   │   └── ...
│   ├── experiments/
│   ├── lora_training/
│   ├── lora_training_glue/
│   └── tikitaka/
│
└── results/                       # Experiment results
```

---

## Key Package Versions

| Package | Version |
|---------|---------|
| Python | 3.10 |
| torch | 2.3.1+cu121 |
| torchvision | 0.18.1+cu121 |
| aihwkit | 1.0.0+cuda121 |
| transformers | 4.47.1 |
| datasets | 4.5.0 |
| optuna | 4.7.0 |
| wandb | 0.24.1 |
| evaluate | 0.4.6 |
| accelerate | 1.12.0 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |

---

## Quick Start Commands

```bash
# 1. Activate environment
source ~/.venv310/bin/activate

# 2. Run SQuAD sweep (example)
cd /data/LRTT_transformer/LRTT_glue
python sweep_sixt1c_lora_squad_adam.py --target V --n_trials 30 --epochs 3

# 3. Run with specific targets
python sweep_sixt1c_lora_squad_adam.py --target Q --n_trials 30 --epochs 3
python sweep_sixt1c_lora_squad_adam.py --target K --n_trials 30 --epochs 3
python sweep_sixt1c_lora_squad_adam.py --target QKV --n_trials 30 --epochs 3
```

---

## Full Requirements (pip freeze)

```
accelerate==1.12.0
aihwkit==1.0.0+cuda121
datasets==4.5.0
evaluate==0.4.6
huggingface_hub==0.36.2
matplotlib==3.10.8
numpy==2.2.6
optuna==4.7.0
pandas==2.3.3
safetensors==0.7.0
scikit-learn==1.7.2
scipy==1.15.3
tokenizers==0.21.4
torch==2.3.1+cu121
torchvision==0.18.1+cu121
tqdm==4.67.2
transformers==4.47.1
wandb==0.24.1
```

---

## Notes

1. **aihwkit wheel**: The pre-built wheel file `aihwkit-1.0.0+cuda121-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` is required for GPU support.

2. **LRTT path**: Scripts use `sys.path.insert(0, '/path/to/LRTT/src')` to import LRTT modules. Update this path according to your installation.

3. **WandB**: Configure wandb login or set `WANDB_MODE=offline` for offline logging.

4. **GPU Memory**: MobileBERT + LRTT requires ~2-4GB GPU memory per process.

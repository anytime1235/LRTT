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

**중요: aihwkit은 CPU 버전과 GPU 버전이 별도로 존재합니다.**
- PyPI (`pip install aihwkit`)로 설치하면 **CPU 전용** 버전이 설치됩니다.
- GPU 지원을 위해서는 반드시 IBM S3에서 제공하는 **GPU wheel**을 직접 다운로드하여 설치해야 합니다.
- 소스 빌드는 불필요합니다.

### 3-1. GPU wheel 다운로드

IBM 공식 S3 버킷에서 Python 버전에 맞는 GPU wheel을 다운로드합니다.

**aihwkit 1.0.0 + CUDA 12.1 (torch 2.3.1 호환):**
```bash
# Python 3.10
wget https://aihwkit-gpu-demo.s3.us-east.cloud-object-storage.appdomain.cloud/aihwkit-1.0.0+cuda121-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

> 참고: 최신 버전 wheel은 아래 URL에서 확인할 수 있습니다.
> https://aihwkit.readthedocs.io/en/latest/advanced_install.html
>
> 단, aihwkit 1.1.0은 torch==2.10.0+cu126을 요구하므로 본 환경(torch 2.3.1+cu121)과 호환되지 않습니다.
> 반드시 **1.0.0+cuda121** 버전을 사용하세요.

### 3-2. GPU wheel 설치

```bash
pip install aihwkit-1.0.0+cuda121-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### 3-3. GPU 동작 검증

설치 후 반드시 아래 스크립트로 GPU 지원 여부를 확인하세요.

```bash
python -c "
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.nn import AnalogLinear
import torch

model = AnalogLinear(10, 5, rpu_config=InferenceRPUConfig())
model = model.cuda()
x = torch.randn(2, 10).cuda()
y = model(x)
print('aihwkit GPU: OK')
print(f'Output: {y.shape}, Device: {y.device}')
"
```

정상 출력 예시:
```
aihwkit GPU: OK
Output: torch.Size([2, 5]), Device: cuda:0
```

만약 `aihwkit has not been compiled with CUDA support` 에러가 발생하면 CPU 전용 버전이 설치된 것입니다. 아래 순서로 재설치하세요:
```bash
pip uninstall aihwkit -y
pip install aihwkit-1.0.0+cuda121-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
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

1. **aihwkit GPU vs CPU**: `pip install aihwkit`은 CPU 전용입니다. GPU 지원이 필요하면 반드시 IBM S3 버킷에서 GPU wheel을 다운로드하여 설치해야 합니다. (Step 3 참조)

2. **aihwkit 버전 호환성**: aihwkit 버전마다 요구하는 torch/CUDA 버전이 다릅니다.
   - `aihwkit 1.0.0+cuda121` → `torch 2.3.1+cu121` (본 환경)
   - `aihwkit 1.1.0` → `torch 2.10.0+cu126` (호환 안됨)

3. **LRTT path**: Scripts use `sys.path.insert(0, '/path/to/LRTT/src')` to import LRTT modules. Update this path according to your installation.

4. **WandB**: Configure wandb login or set `WANDB_MODE=offline` for offline logging.

5. **GPU Memory**: MobileBERT + LRTT requires ~2-4GB GPU memory per process.

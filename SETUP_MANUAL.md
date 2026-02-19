# LRTT 환경 설정 매뉴얼

LRTT (Low-Rank Tiled Training) 실험 환경 설정 가이드입니다.

**테스트 완료 환경:** Python 3.10, Ubuntu Linux, CUDA 12.1

---

## Part 1: CUDA 버전 설치 (GPU 사용)

### 1.1 conda 환경 생성 + CUDA 빌드 도구 설치

`conda-forge` 채널에서 `cuda-version=12.1`을 지정하면 모든 CUDA 패키지가 12.1로 통일됩니다.

```bash
conda create --prefix /path/to/venv -y -c conda-forge \
  python=3.10 \
  cuda-version=12.1 \
  cuda-nvcc \
  cuda-cudart-dev \
  cuda-cccl \
  libcublas-dev \
  libcurand-dev \
  openblas \
  gxx_linux-64=12.4.0 gcc_linux-64=12.4.0
conda activate /path/to/venv
```

> **주의**: nvidia 채널이 아닌 **conda-forge** 채널을 사용해야 합니다. nvidia 채널은 메타패키지만 12.1이고 플랫폼 패키지가 13.x로 설치되는 문제가 있습니다.
>
> apt 환경인 경우: `sudo apt install -y build-essential cmake python3-venv python3-dev libopenblas-dev nvidia-cuda-toolkit`

### 1.2 PyTorch 및 빌드 도구 설치

```bash
pip install --upgrade pip wheel setuptools
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install scikit-build pybind11 cmake mypy
```

> 다른 GPU는 하단 [GPU 아키텍처 참고](#gpu-아키텍처-참고) 표에서 PyTorch 버전 확인.

### 1.3 aihwkit + LRTT 빌드 (Editable)

```bash
cd /path/to/LRTT_vit
rm -rf _skbuild build

export GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.')
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
export CMAKE_ARGS="-DRPU_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH"
USE_CUDA=1 pip install -e .
USE_CUDA=1 python setup.py build_ext --inplace
```

### 1.4 Python 의존성 설치

```bash
pip install transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 \
    optuna==4.7.0 optuna-integration==4.7.0 botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 \
    scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.8 pandas==2.3.3 \
    numpy==2.2.6 safetensors==0.7.0 tokenizers==0.21.4 \
    huggingface_hub==0.36.2 tqdm==4.67.2
```

### 1.5 설치 확인

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
from aihwkit.nn import AnalogConv2d
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.tiles.lrtt_controller import LRTTController
print('aihwkit + LRTT OK!')
"
```

---

## Part 2: CPU 버전 설치 (GPU 없이)

```bash
# 사전 요구사항
sudo apt install -y build-essential cmake python3-venv python3-dev libopenblas-dev

# 가상환경 + PyTorch
conda create --prefix /path/to/venv_cpu python=3.10 openblas -y -c conda-forge
conda activate /path/to/venv_cpu
pip install --upgrade pip wheel setuptools
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install scikit-build pybind11 cmake mypy

# 빌드
cd /path/to/LRTT_vit
rm -rf _skbuild build
USE_CUDA=0 pip install -e .
USE_CUDA=0 python setup.py build_ext --inplace

# 의존성
pip install transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 \
    optuna==4.7.0 optuna-integration==4.7.0 botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 \
    scipy==1.15.3 scikit-learn==1.7.2
```

**주의**: CPU 버전은 훈련 속도가 매우 느립니다.

---

## 빠른 설치 (한 줄 명령어)

### CUDA 12.1 (권장)

```bash
conda create --prefix /path/to/venv -y -c conda-forge \
  python=3.10 cuda-version=12.1 cuda-nvcc cuda-cudart-dev cuda-cccl \
  libcublas-dev libcurand-dev openblas gxx_linux-64=12.4.0 gcc_linux-64=12.4.0 && \
conda activate /path/to/venv && \
pip install --upgrade pip wheel setuptools scikit-build pybind11 cmake mypy && \
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
cd /path/to/LRTT_vit && rm -rf _skbuild build && \
export GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.') && \
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH && \
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH && \
export CMAKE_ARGS="-DRPU_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH" && \
USE_CUDA=1 pip install -e . && \
USE_CUDA=1 python setup.py build_ext --inplace && \
pip install transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 optuna==4.7.0 optuna-integration==4.7.0 botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.8 pandas==2.3.3 numpy==2.2.6 safetensors==0.7.0 tokenizers==0.21.4 huggingface_hub==0.36.2 tqdm==4.67.2
```

### CPU 버전

```bash
conda create --prefix /path/to/venv_cpu python=3.10 openblas -y -c conda-forge && \
conda activate /path/to/venv_cpu && \
pip install --upgrade pip wheel setuptools scikit-build pybind11 cmake mypy && \
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu && \
cd /path/to/LRTT_vit && rm -rf _skbuild build && \
USE_CUDA=0 pip install -e . && USE_CUDA=0 python setup.py build_ext --inplace && \
pip install transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 optuna==4.7.0 optuna-integration==4.7.0 botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 scipy==1.15.3 scikit-learn==1.7.2
```

---

## GPU 아키텍처 참고

`GPU_ARCH` 확인: `nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.'`

| GPU 시리즈 | 아키텍처 | GPU_ARCH | PyTorch 설치 명령어 |
|-----------|---------|----------|-------------------|
| H100, H200 | Hopper | 90 | `pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| RTX 40xx, L40 | Ada Lovelace | 89 | `pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| A100, A30 | Ampere | 80 | `pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| RTX 30xx | Ampere | 86 | `pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| V100 | Volta | 70 | `pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118` |
| RTX 20xx | Turing | 75 | `pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118` |

---

## LRTT 소스 구조

Editable 설치 시 `sys.path.insert`는 불필요합니다.

```
LRTT_vit/
├── src/aihwkit/simulator/
│   ├── configs/
│   │   ├── lrtt_config.py          # LRTT 설정 클래스
│   │   ├── lrtt_python.py          # PythonLRTTDevice
│   │   └── lrtt_rpu_config.py      # RPU config wrapper
│   └── tiles/
│       ├── lrtt_controller.py      # LRTT 3-tile 컨트롤러
│       └── lrtt_tile.py            # LRTT 타일 생성
├── examples/
│   ├── cifar10_resnet_rlrtt_scratch.py
│   └── Mobilebert/
├── setup.py
├── pyproject.toml
└── CMakeLists.txt
```

---

## 전체 의존성 목록

| 패키지 | 버전 | 비고 |
|---------|------|------|
| Python | 3.10 | |
| torch | 2.3.1+cu121 | |
| torchvision | 0.18.1+cu121 | |
| aihwkit | 1.0.0 | editable |
| transformers | 4.47.1 | |
| datasets | 4.5.0 | |
| evaluate | 0.4.6 | |
| optuna | 4.7.0 | |
| optuna-integration | 4.7.0 | BoTorchSampler 등 |
| botorch | 0.16.1 | BoTorchSampler 백엔드 |
| wandb | 0.24.1 | |
| accelerate | 1.12.0 | |
| scipy | 1.15.3 | |
| scikit-learn | 1.7.2 | |
| matplotlib | 3.10.8 | |
| pandas | 2.3.3 | |
| numpy | 2.2.6 | |
| safetensors | 0.7.0 | |
| tokenizers | 0.21.4 | |
| huggingface_hub | 0.36.2 | |
| tqdm | 4.67.2 | |
| scikit-build | - | 빌드용 |
| pybind11 | - | 빌드용 |
| cmake | - | 빌드용 |
| mypy | - | 빌드용 |
| cuda-version | 12.1 | conda-forge, 버전 핀 |
| cuda-nvcc | 12.1 | conda-forge, 빌드용 |
| cuda-cudart-dev | 12.1 | conda-forge, 빌드용 |
| cuda-cccl | 12.1 | conda-forge, CUDA C++ 헤더 |
| libcublas-dev | 12.1 | conda-forge, 빌드용 |
| libcurand-dev | 12.1 | conda-forge, 빌드용 |
| openblas | - | conda-forge, BLAS 라이브러리 |
| gcc/g++ | 12.4.0 | conda-forge, 빌드용 |

# LRTT 환경 설정 매뉴얼 (Conda + aihwkit-gpu)

Conda 기반 LRTT 환경 구축 가이드입니다. CUDA toolkit과 C++ 컴파일러를 conda로 관리하여 시스템 의존성을 최소화합니다.

**테스트 완료 환경:** Python 3.10, CUDA 12.1, PyTorch 2.3.1, NVIDIA H200 (Hopper, SM 90)

---

## Part 1: Conda 환경 생성

### 1.1 Conda 환경 + CUDA/C++ 빌드 도구 설치

시스템에 NVIDIA 드라이버가 설치되어 있어야 합니다 (`nvidia-smi`로 확인).

```bash
conda create --prefix /path/to/venv -y -c conda-forge \
    python=3.10 \
    cuda-version=12.1 \
    cuda-nvcc \
    cuda-cudart-dev \
    cuda-cccl \
    libcublas-dev \
    libcurand-dev \
    gxx_linux-64=12.4.0 \
    gcc_linux-64=12.4.0
```

> **참고**: `--prefix /path/to/venv` 대신 `--name myenv`를 사용해도 됩니다.

### 1.2 환경 활성화

```bash
# prefix 방식
source activate /path/to/venv
# 또는 name 방식
conda activate myenv
```

> **확인**: `which python`이 conda 환경 내 python을 가리키는지 확인합니다.

### 1.3 PyTorch 및 빌드 도구 설치

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install scikit-build pybind11 cmake mypy
```

### 1.4 aihwkit + LRTT 빌드 (Editable)

```bash
cd /path/to/LRTT
rm -rf _skbuild build

export GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.')
export CMAKE_ARGS="-DRPU_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH"
USE_CUDA=1 pip install -e .
USE_CUDA=1 python setup.py build_ext --inplace
```

> **참고**: conda에서 설치한 `cuda-nvcc`, `gcc`, `g++`가 자동으로 PATH에 잡히므로 별도 CUDA toolkit apt 설치가 불필요합니다.

### 1.5 Python 의존성 설치

```bash
pip install transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 \
    optuna==4.7.0 optuna-integration==4.7.0 botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 \
    scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.8 pandas==2.3.3 \
    numpy==2.2.6 safetensors==0.7.0 tokenizers==0.21.4 \
    huggingface_hub==0.36.2 tqdm==4.67.2
```

### 1.6 설치 확인

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

## Part 2: 빠른 설치 (한 줄 명령어)

CUDA 12.1 + Conda 기준:

```bash
conda create --prefix /path/to/venv -y -c conda-forge \
    python=3.10 cuda-version=12.1 cuda-nvcc cuda-cudart-dev cuda-cccl \
    libcublas-dev libcurand-dev gxx_linux-64=12.4.0 gcc_linux-64=12.4.0 && \
source activate /path/to/venv && \
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
pip install scikit-build pybind11 cmake mypy && \
cd /path/to/LRTT && rm -rf _skbuild build && \
export GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.') && \
export CMAKE_ARGS="-DRPU_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH" && \
USE_CUDA=1 pip install -e . && \
USE_CUDA=1 python setup.py build_ext --inplace && \
pip install transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 optuna==4.7.0 optuna-integration==4.7.0 \
    botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 scipy==1.15.3 scikit-learn==1.7.2 \
    matplotlib==3.10.8 pandas==2.3.3 numpy==2.2.6 safetensors==0.7.0 tokenizers==0.21.4 \
    huggingface_hub==0.36.2 tqdm==4.67.2
```

---

## apt 방식 vs Conda 방식 비교

| 항목 | apt 방식 (`SETUP_MANUAL.md`) | Conda 방식 (이 문서) |
|------|------|------|
| CUDA toolkit | `apt install cuda-toolkit-12-1` | `conda: cuda-nvcc, cuda-cudart-dev, cuda-cccl` |
| C++ 컴파일러 | `apt install build-essential` | `conda: gcc_linux-64, gxx_linux-64` |
| BLAS | `apt install libopenblas-dev` | conda에서 자동 의존성 해결 |
| 격리성 | 시스템 전역 설치 | 환경별 완전 격리 |
| 장점 | 단순, 빠름 | 여러 CUDA 버전 공존 가능 |
| 단점 | 시스템 충돌 가능 | conda 저장소 크기가 큼 |

> **권장**: 여러 프로젝트에서 다른 CUDA 버전이 필요하거나, root 권한 없이 설치해야 하는 경우 Conda 방식을 추천합니다.

---

## Conda 환경에서 설치된 주요 패키지

| 채널 | 패키지 | 용도 |
|------|--------|------|
| conda-forge | `python=3.10` | Python 인터프리터 |
| conda-forge | `cuda-version=12.1` | CUDA 버전 고정 |
| conda-forge | `cuda-nvcc` | NVIDIA CUDA 컴파일러 (nvcc) |
| conda-forge | `cuda-cudart-dev` | CUDA Runtime 헤더/라이브러리 |
| conda-forge | `cuda-cccl` | CUDA C++ Core Libraries |
| conda-forge | `libcublas-dev` | cuBLAS 헤더/라이브러리 |
| conda-forge | `libcurand-dev` | cuRAND 헤더/라이브러리 |
| conda-forge | `gcc_linux-64=12.4.0` | GCC C 컴파일러 |
| conda-forge | `gxx_linux-64=12.4.0` | G++ C++ 컴파일러 |

---

## GPU 아키텍처 참고

`GPU_ARCH` 확인: `nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.'`

| GPU 시리즈 | 아키텍처 | GPU_ARCH | PyTorch 설치 명령어 |
|-----------|---------|----------|-------------------|
| H100, H200 | Hopper | 90 | `torch==2.3.1+cu121` |
| RTX 40xx, L40 | Ada Lovelace | 89 | `torch==2.3.1+cu121` |
| A100, A30 | Ampere | 80 | `torch==2.3.1+cu121` |
| RTX 30xx | Ampere | 86 | `torch==2.3.1+cu121` |
| V100 | Volta | 70 | `torch==2.3.1+cu118` |
| RTX 20xx | Turing | 75 | `torch==2.3.1+cu118` |

---

## 전체 pip 의존성 목록

| 패키지 | 버전 | 비고 |
|---------|------|------|
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

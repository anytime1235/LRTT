# LRTT 환경 설정 매뉴얼

LRTT (Low-Rank Tiled Training) 실험 환경 설정 가이드입니다.

**테스트 완료 환경:** Python 3.10, Ubuntu 22.04, CUDA 12.1

---

## Part 1: CUDA 버전 설치 (GPU 사용)

### 1.1 시스템 빌드 의존성 + CUDA Toolkit 설치

NVIDIA 드라이버가 설치되어 있어야 합니다 (`nvidia-smi`로 확인).

```bash
# 빌드 도구 + OpenBLAS
apt update
apt install -y build-essential cmake ninja-build python3-dev libopenblas-dev wget tmux

# NVIDIA CUDA 12.1 Toolkit (이미 nvcc가 있으면 건너뛰기: nvcc --version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-toolkit-12-1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

> **확인**: `nvcc --version`으로 CUDA 12.1이 출력되어야 합니다.
>
> 이미 CUDA toolkit이 설치된 환경(Docker GPU 이미지, 클라우드 인스턴스 등)에서는 apt install cuda-toolkit 단계를 건너뛰면 됩니다.

### 1.2 uv, PyTorch 및 빌드 도구 설치

```bash
pip install uv
uv pip install --system torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --system scikit-build pybind11 mypy
```

> 다른 GPU는 하단 [GPU 아키텍처 참고](#gpu-아키텍처-참고) 표에서 PyTorch 버전 확인.

### 1.3 aihwkit + LRTT 빌드 (Editable)

```bash
cd /root/LRTT
rm -rf _skbuild build

export GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.')
export CMAKE_ARGS="-GNinja -DRPU_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH"
export CMAKE_BUILD_PARALLEL_LEVEL=2   # CUDA 컴파일 메모리 부족 방지 (WSL2 등)
USE_CUDA=1 pip install -e .
USE_CUDA=1 python setup.py build_ext --inplace
```

> **참고**: editable 빌드(`pip install -e .`)는 C++ 확장 빌드를 포함하므로 uv 대신 pip을 사용합니다. `build_ext --inplace`는 컴파일된 C++ 확장(.so)을 소스 트리에 배치하여 editable 모드에서 import할 수 있게 합니다.
>
> **WSL2 메모리 주의**: CUDA 파일 컴파일은 파일당 2-4GB RAM을 소비합니다. Ninja 기본 설정은 모든 CPU 코어를 병렬로 사용하므로, WSL2 등 메모리가 제한된 환경에서는 OOM으로 인해 WSL이 크래시할 수 있습니다. `CMAKE_BUILD_PARALLEL_LEVEL=2`로 병렬 수를 제한하세요.
>
> **Blackwell GPU (RTX 50xx, compute_120)**: CUDA toolkit 12.0에서는 `compute_120`을 지원하지 않아 빌드가 실패합니다. 이 경우 `GPU_ARCH=89`로 수동 지정하면 PTX JIT 호환으로 동작합니다:
> ```bash
> export CMAKE_ARGS="-GNinja -DRPU_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_ARCHITECTURES=89"
> ```

### 1.4 Python 의존성 설치

```bash
uv pip install --system transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 \
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
apt update
apt install -y build-essential cmake ninja-build python3-dev libopenblas-dev

# uv + PyTorch + 빌드 도구
pip install uv
uv pip install --system torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --system scikit-build pybind11 mypy

# 빌드
cd /root/LRTT
rm -rf _skbuild build
USE_CUDA=0 pip install -e .
USE_CUDA=0 python setup.py build_ext --inplace

# 의존성
uv pip install --system transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 \
    optuna==4.7.0 optuna-integration==4.7.0 botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 \
    scipy==1.15.3 scikit-learn==1.7.2
```

**주의**: CPU 버전은 훈련 속도가 매우 느립니다.

---

## 빠른 설치 (한 줄 명령어)

### CUDA 12.1 (권장)

NVIDIA 드라이버(`nvidia-smi`)만 있으면 CUDA toolkit 유무와 관계없이 동작합니다.

```bash
apt update && apt install -y build-essential cmake ninja-build python3-dev libopenblas-dev wget tmux && \
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
dpkg -i cuda-keyring_1.1-1_all.deb && apt update && apt install -y cuda-toolkit-12-1 && \
export PATH=/usr/local/cuda-12.1/bin:$PATH && \
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH && \
pip install uv && \
uv pip install --system scikit-build pybind11 mypy && \
uv pip install --system torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
cd /root/LRTT && rm -rf _skbuild build && \
export GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.') && \
export CMAKE_ARGS="-GNinja -DRPU_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH" && \
USE_CUDA=1 pip install -e . && \
USE_CUDA=1 python setup.py build_ext --inplace && \
uv pip install --system transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 optuna==4.7.0 optuna-integration==4.7.0 botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.8 pandas==2.3.3 numpy==2.2.6 safetensors==0.7.0 tokenizers==0.21.4 huggingface_hub==0.36.2 tqdm==4.67.2
```

### CPU 버전

```bash
apt update && apt install -y build-essential cmake ninja-build python3-dev libopenblas-dev tmux && \
pip install uv && \
uv pip install --system scikit-build pybind11 mypy && \
uv pip install --system torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu && \
cd /root/LRTT && rm -rf _skbuild build && \
export CMAKE_ARGS="-GNinja" && \
USE_CUDA=0 pip install -e . && \
USE_CUDA=0 python setup.py build_ext --inplace && \
uv pip install --system transformers==4.47.1 datasets==4.5.0 evaluate==0.4.6 optuna==4.7.0 optuna-integration==4.7.0 botorch==0.16.1 wandb==0.24.1 accelerate==1.12.0 scipy==1.15.3 scikit-learn==1.7.2
```

---

## GPU 아키텍처 참고

`GPU_ARCH` 확인: `nvidia-smi --query-gpu=compute_cap -i 0 --format=csv,noheader | tr -d '.'`

| GPU 시리즈 | 아키텍처 | GPU_ARCH | PyTorch 설치 명령어 |
|-----------|---------|----------|-------------------|
| RTX 50xx, B200 | Blackwell | 120 (→ **89로 지정**) | CUDA 12.8+ 필요. toolkit이 12.0~12.6이면 `GPU_ARCH=89`로 수동 지정 (PTX JIT 호환) |
| H100, H200 | Hopper | 90 | `uv pip install --system torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| RTX 40xx, L40 | Ada Lovelace | 89 | `uv pip install --system torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| A100, A30 | Ampere | 80 | `uv pip install --system torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| RTX 30xx | Ampere | 86 | `uv pip install --system torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| V100 | Volta | 70 | `uv pip install --system torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118` |
| RTX 20xx | Turing | 75 | `uv pip install --system torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118` |

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
| uv | - | 빠른 패키지 설치 |
| scikit-build | - | 빌드용 |
| pybind11 | - | 빌드용 |
| mypy | - | 빌드용 (stubgen) |
| libopenblas-dev | - | apt, BLAS 라이브러리 |
| build-essential | - | apt, GCC/G++ 컴파일러 |
| ninja-build | - | apt, 빠른 빌드 시스템 |
| tmux | - | apt, SSH 끊김 시 실험 보호 |
| cuda-toolkit-12-1 | 12.1 | apt (NVIDIA repo), CUDA 없을 때만 |

curl -fsSL https://claude.ai/install.sh | bash

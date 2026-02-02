# CIFAR10 ResNet RLRTT 환경 설정 매뉴얼

`cifar10_resnet_rlrtt_scratch.py` 실행을 위한 환경 설정 가이드입니다.

**테스트 완료 환경:**
- Python 3.12.3
- Ubuntu (WSL2)
- CUDA 12.0 / 12.8

---

## 시스템 요구사항

- **Python**: 3.8 ~ 3.12 (권장: 3.12)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows
- **RAM**: 최소 8GB (권장: 16GB+)
- **저장공간**: 최소 10GB
- **빌드 도구**: cmake, g++

---

## Part 1: CUDA 버전 설치 (GPU 사용)

### 1.1 사전 요구사항

#### CUDA Toolkit 설치 확인
```bash
nvcc --version
```

지원 CUDA 버전: 11.8, 12.0 ~ 12.8 (권장)

CUDA가 없다면 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)에서 설치하세요.

#### 빌드 도구 설치
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev libopenblas-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake python3-devel openblas-devel
```

### 1.2 가상환경 생성

```bash
cd /path/to/LoRA
python3 -m venv .venv_cuda
source .venv_cuda/bin/activate
pip install --upgrade pip wheel setuptools
```

### 1.3 PyTorch 설치 (GPU에 맞게 선택)

| GPU 시리즈 | 명령어 |
|-----------|--------|
| **RTX 50xx** (Blackwell) | `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128` |
| **RTX 40xx** (Ada) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| **RTX 30xx** (Ampere) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **RTX 20xx / GTX 16xx** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |

### 1.4 의존성 설치 (requirements 파일 사용)

```bash
cd examples
pip install -r requirements_cuda.txt
```

### 1.5 aihwkit 빌드 및 설치

```bash
cd /path/to/LoRA
rm -rf _skbuild build   # 기존 캐시 삭제 (중요!)

# 개발 모드 설치 (권장: 소스 수정 시 재설치 불필요)
pip install -e .

# 또는 일반 설치
pip install .
```

**빌드 시간**: 약 5-15분 소요 (시스템 사양에 따라 다름)

### 1.6 설치 확인

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
from aihwkit.nn import AnalogConv2d
print('All OK!')
"
```

### 1.7 예제 실행

```bash
cd examples
python cifar10_resnet_rlrtt_scratch.py
```

---

## Part 2: CPU 버전 설치 (GPU 없이)

### 2.1 사전 요구사항

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev libopenblas-dev

# macOS
xcode-select --install
brew install cmake openblas
```

### 2.2 가상환경 생성

```bash
cd /path/to/LoRA
python3 -m venv .venv_cpu
source .venv_cpu/bin/activate   # Windows: .venv_cpu\Scripts\activate
pip install --upgrade pip wheel setuptools
```

### 2.3 PyTorch CPU 버전 설치

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2.4 의존성 설치 (requirements 파일 사용)

```bash
cd examples
pip install -r requirements_cpu.txt
```

### 2.5 aihwkit 빌드 및 설치

```bash
cd /path/to/LoRA
rm -rf _skbuild build

# 개발 모드 설치 (권장: 소스 수정 시 재설치 불필요)
USE_CUDA=0 pip install -e .

# 또는 일반 설치
USE_CUDA=0 pip install .
```

### 2.6 설치 확인 및 실행

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
from aihwkit.nn import AnalogConv2d
print('All OK!')
"

cd examples
python cifar10_resnet_rlrtt_scratch.py
```

**주의**: CPU 버전은 훈련 속도가 매우 느립니다.

---

## 빠른 설치 (한 줄 명령어)

> `-e` 옵션은 개발 모드 (소스 수정 시 재설치 불필요). 일반 사용자는 `-e` 제거 가능.

### CUDA 버전 (RTX 50 시리즈)
```bash
cd /path/to/LoRA && python3 -m venv .venv && source .venv/bin/activate && \
pip install --upgrade pip wheel setuptools && \
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 && \
pip install -r examples/requirements_cuda.txt && \
rm -rf _skbuild build && pip install -e .
```

### CUDA 버전 (RTX 40/30 시리즈)
```bash
cd /path/to/LoRA && python3 -m venv .venv && source .venv/bin/activate && \
pip install --upgrade pip wheel setuptools && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
pip install -r examples/requirements_cuda.txt && \
rm -rf _skbuild build && pip install -e .
```

### CPU 버전
```bash
cd /path/to/LoRA && python3 -m venv .venv && source .venv/bin/activate && \
pip install --upgrade pip wheel setuptools && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
pip install -r examples/requirements_cpu.txt && \
rm -rf _skbuild build && USE_CUDA=0 pip install -e .
```

---

## GPU 아키텍처 참고

| GPU 시리즈 | 아키텍처 | Compute Capability | PyTorch 버전 |
|-----------|---------|-------------------|-------------|
| RTX 50xx | Blackwell | sm_120 | nightly (cu128) |
| RTX 40xx | Ada Lovelace | sm_89 | 2.0+ (cu124) |
| RTX 30xx | Ampere | sm_86 | 1.9+ (cu121) |
| RTX 20xx | Turing | sm_75 | 1.9+ (cu118) |
| GTX 16xx | Turing | sm_75 | 1.9+ (cu118) |
| GTX 10xx | Pascal | sm_61 | 1.9+ (cu118) |

---

## 문제 해결

### 1. `No module named 'aihwkit'`
```bash
rm -rf _skbuild build && pip install -e .
```

### 2. `rpu_base` import 오류 (undefined symbol)
PyTorch 버전 변경 후 재빌드 필요:
```bash
rm -rf _skbuild build && pip uninstall aihwkit -y && pip install -e .
```

### 3. GPU 인식 안됨 (sm_XX not compatible)
GPU에 맞는 PyTorch 버전 재설치:
```bash
pip uninstall torch torchvision -y
# GPU에 맞는 버전 설치 (1.3 참고)
```

### 4. OpenBLAS not found
```bash
sudo apt install libopenblas-dev
```

### 5. CMake 버전 오류
```bash
pip install --upgrade cmake
```

### 6. CUDA 버전 불일치
```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 시스템 CUDA 버전 확인
nvcc --version

# 버전이 다르면 PyTorch 재설치 (1.3 표 참고)
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 7. 빌드 오류: `pybind11 not found`
```bash
pip install pybind11>=2.6.2
```

### 8. 메모리 부족 (OOM)
`cifar10_resnet_rlrtt_scratch.py`에서 `BATCH_SIZE = 64`로 줄임

### 9. wandb 로그인
```bash
wandb login
```

### 10. `python3-venv` 없음
```bash
sudo apt install python3.12-venv
```

---

## 전체 의존성 목록

`requirements_cuda.txt` / `requirements_cpu.txt` 파일이 없는 경우 수동 설치:

```bash
# 빌드 의존성
pip install cmake>=3.18 scikit-build>=0.11.1 pybind11>=2.6.2 ninja

# 런타임 의존성
pip install scipy numpy>=1.22,<2 protobuf>=4.21.6 tqdm requests>=2.25,<3
pip install wandb scikit-learn  # 선택사항
```

---

## 참고 링크

- [PyTorch 설치](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [wandb](https://docs.wandb.ai/)

# CIFAR10 ResNet RLRTT 환경 설정 매뉴얼

`cifar10_resnet_rlrtt_scratch.py` 실행을 위한 환경 설정 가이드입니다.

## 시스템 요구사항

- **Python**: 3.8 ~ 3.12 (권장: 3.12)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows
- **RAM**: 최소 8GB (권장: 16GB+)
- **저장공간**: 최소 10GB

---

## Part 1: CUDA 버전 설치 (GPU 사용)

### 1.1 사전 요구사항

#### CUDA Toolkit 설치 확인
```bash
nvcc --version
```

지원 CUDA 버전:
- CUDA 11.8
- CUDA 12.0 ~ 12.8 (권장)

CUDA가 설치되어 있지 않다면 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)에서 설치하세요.

#### 빌드 도구 설치
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake python3-devel
```

### 1.2 가상환경 생성

```bash
# 프로젝트 디렉토리로 이동
cd /path/to/LoRA

# 가상환경 생성
python3 -m venv .venv_cuda

# 가상환경 활성화
source .venv_cuda/bin/activate

# pip 업그레이드
pip install --upgrade pip wheel setuptools
```

### 1.3 PyTorch (CUDA 버전) 설치

#### CUDA 12.8 (최신)
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### CUDA 12.4
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 1.4 빌드 의존성 설치

```bash
pip install cmake>=3.18 scikit-build>=0.11.1 pybind11>=2.6.2 ninja
```

### 1.5 기타 의존성 설치

```bash
pip install scipy numpy>=1.22,<2 protobuf>=4.21.6 tqdm requests>=2.25,<3
pip install wandb  # 로깅용
pip install scikit-learn  # 선택사항
```

### 1.6 aihwkit 설치 (소스에서 빌드)

```bash
cd /path/to/LoRA

# 개발 모드로 설치 (권장: 소스 수정 가능)
pip install -e .

# 또는 일반 설치
pip install .
```

**빌드 시간**: 약 5-15분 소요 (시스템 사양에 따라 다름)

### 1.7 설치 확인

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import aihwkit
print(f'aihwkit: OK')
"
```

### 1.8 예제 실행

```bash
cd examples
python cifar10_resnet_rlrtt_scratch.py
```

---

## Part 2: CPU 버전 설치 (GPU 없이)

### 2.1 사전 요구사항

#### 빌드 도구 설치
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev

# macOS
xcode-select --install
brew install cmake

# Windows
# Visual Studio Build Tools 설치 필요
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### 2.2 가상환경 생성

```bash
# 프로젝트 디렉토리로 이동
cd /path/to/LoRA

# 가상환경 생성
python3 -m venv .venv_cpu

# 가상환경 활성화
# Linux/macOS:
source .venv_cpu/bin/activate
# Windows:
.venv_cpu\Scripts\activate

# pip 업그레이드
pip install --upgrade pip wheel setuptools
```

### 2.3 PyTorch (CPU 버전) 설치

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2.4 빌드 의존성 설치

```bash
pip install cmake>=3.18 scikit-build>=0.11.1 pybind11>=2.6.2 ninja
```

### 2.5 기타 의존성 설치

```bash
pip install scipy numpy>=1.22,<2 protobuf>=4.21.6 tqdm requests>=2.25,<3
pip install wandb  # 로깅용
pip install scikit-learn  # 선택사항
```

### 2.6 aihwkit 설치 (소스에서 빌드)

```bash
cd /path/to/LoRA

# CPU 전용 빌드를 위한 환경 변수 설정
export USE_CUDA=0

# 개발 모드로 설치
pip install -e .

# 또는 일반 설치
pip install .
```

### 2.7 설치 확인

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')  # False 예상

import aihwkit
print(f'aihwkit: OK')
"
```

### 2.8 예제 실행

```bash
cd examples
python cifar10_resnet_rlrtt_scratch.py
```

**주의**: CPU 버전은 GPU에 비해 훈련 속도가 매우 느립니다 (약 10-50배).

---

## 문제 해결

### 1. `No module named 'aihwkit'`
```bash
# aihwkit 재설치
cd /path/to/LoRA
pip install -e .
```

### 2. CMake 버전 오류
```bash
# CMake 업그레이드
pip install --upgrade cmake
```

### 3. CUDA 버전 불일치
```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 시스템 CUDA 버전 확인
nvcc --version

# 버전이 다르면 PyTorch 재설치
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # 맞는 버전으로
```

### 4. 빌드 오류: `pybind11 not found`
```bash
pip install pybind11>=2.6.2
```

### 5. 메모리 부족 (OOM)
`cifar10_resnet_rlrtt_scratch.py`에서 배치 크기 조정:
```python
BATCH_SIZE = 64  # 128에서 줄임
```

### 6. wandb 로그인
```bash
wandb login
# API 키 입력 (https://wandb.ai/authorize 에서 확인)
```

---

## 전체 의존성 목록

```
# requirements_cuda.txt
torch>=2.0
torchvision>=0.15
scipy>=1.9
numpy>=1.22,<2
protobuf>=4.21.6
tqdm>=4.60
requests>=2.25,<3
wandb>=0.15
cmake>=3.18
scikit-build>=0.11.1
pybind11>=2.6.2
ninja
scikit-learn>=1.0
```

## 빠른 설치 (한 줄 명령어)

### CUDA 12.4 버전
```bash
cd /path/to/LoRA && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install --upgrade pip wheel setuptools && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
pip install cmake scikit-build pybind11 ninja scipy numpy protobuf tqdm requests wandb && \
pip install -e .
```

### CPU 버전
```bash
cd /path/to/LoRA && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install --upgrade pip wheel setuptools && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
pip install cmake scikit-build pybind11 ninja scipy numpy protobuf tqdm requests wandb && \
USE_CUDA=0 pip install -e .
```

---

## 참고 링크

- [PyTorch 공식 설치 가이드](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [wandb 문서](https://docs.wandb.ai/)

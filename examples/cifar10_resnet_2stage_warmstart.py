# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""CIFAR-10 ResNet18 LRTT training with 2-stage warm-start.

Two-stage training approach:
- Stage 1 (FullAnalog): Train C matrix only (A/B disabled) for initial convergence
- Stage 2 (LRTT): Load C weights, initialize A=0/B=Kaiming, train with LRTT

Key features:
- In-memory weight transfer (no file I/O between stages)
- Prevents potential bugs from save/load process
- Consistent setup between stages
- Single script execution

Based on 03_mnist_training_lrtt_warmup.py pattern.
"""
# pylint: disable=invalid-name

import os
from time import time

import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

from aihwkit.nn import AnalogConv2d
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset, PythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType, UpdateParameters
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice, LinearStepDevice

# Logging
import wandb

# ==============================================================================
# Configuration
# ==============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_2STAGE_WARMSTART")
os.makedirs(RESULTS, exist_ok=True)

# Training - Stage 1 (FullAnalog)
N_EPOCHS_STAGE1 = 0  # FullAnalog warm-start epochs
LEARNING_RATE_STAGE1 = 0.1
WARMUP_RATIO_STAGE1 = 0.0
LR_SCHEDULE_STAGE1 = "cosine"     # LR schedule: "constant", "cosine", "multistep"
LR_MILESTONES_STAGE1 = [150, 225]   # multistep용: LR 감소 epoch 리스트
LR_GAMMA_STAGE1 = 0.1               # multistep용: LR 감소 비율 (lr *= gamma)

# Training - Stage 2 (LRTT)
N_EPOCHS_STAGE2 = 300  # LRTT fine-tuning epochs
LEARNING_RATE_STAGE2 = 0.1  # Lower LR for fine-tuning (BN/conv1/fc already converged)
WARMUP_RATIO_STAGE2 = 0.0  # No warmup for stage 2
LR_SCHEDULE_STAGE2 = "cosine"       # LR schedule: "constant", "cosine", "multistep"
LR_MILESTONES_STAGE2 = [30, 60]     # multistep용: LR 감소 epoch 리스트
LR_GAMMA_STAGE2 = 0.1               # multistep용: LR 감소 비율 (lr *= gamma)

# Common
SEED = 1
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NESTEROV = True
N_CLASSES = 10
NUM_WORKERS = 4

# ==============================================================================
# LRTT Configuration
# ==============================================================================

# --- 기본 LRTT 파라미터 ---
LRTT_RANK_CONV = 16             # Conv 레이어 LoRA rank (낮을수록 파라미터 감소, 높을수록 표현력 증가)
LRTT_RANK_FC = 16               # FC 레이어 LoRA rank
TRANSFER_EVERY = 1000            # A⊗B → C 전송 주기 (mini-batch 단위)
LORA_ALPHA = 2.0                # LoRA 스케일 팩터 α (y = Cx + α*A(Bx))
TRANSFER_LR = LORA_ALPHA        # 전송 learning rate (기본적으로 α와 동일)

# --- Transfer LR 스케일링 ---
TRANSFER_LR_SCALE = "none"      # transfer_lr 자동 스케일링 모드
                                #   "none": 스케일링 없음, transfer_lr 그대로 사용
                                #   "sqrt_rank": transfer_lr / sqrt(rank) - rank가 커질수록 줄임
                                #   "rank": transfer_lr / rank - 더 강한 정규화

# --- Transfer 방식 선택 ---
USE_ONEHOT_TRANSFER = False      # True: one-hot 전송 (아날로그 현실적)
                                # False: direct 전송 (get_weights() 직접 접근)

# --- Transfer 캘리브레이션 모드 ---
TRANSFER_MODE = "off"           # 전송 캘리브레이션 모드
                                #   "pilot": 파일럿 기반 γ 보정 (실측 후 스케일 조정)
                                #   "sigma_delta": ΣΔ 양자화 (잔여 누적, 정수 펄스 전송)
                                #   "off": 캘리브레이션 없음, 직접 전송
TRANSFER_MICRO_STEPS = 1        # micro-transfer 반복 횟수 (>=2 권장, 분산 감소)
TRANSFER_PILOT_FRAC = 1.0/16.0  # 파일럿 전송 lr 비율 (transfer_mode='pilot' 시)
SD_QUANTUM = None               # ΣΔ 단위 양자 g (None이면 |transfer_lr|/micro_steps로 자동 계산)

# --- Transfer 전처리 ---
TRANSFER_CENTERING = False      # 행/열 평균 제거 (DC offset 보정, 기본 off)
TRANSFER_NORMALIZE = False      # 랭크별 ℓ2 정규화 (gradient 왜곡 가능성으로 기본 off)

# --- Reinit 모드 ---
REINIT_MODE = "orthogonal"        # 전송 후 A/B 재초기화 전략
                                #   "standard": A=0, B=Kaiming (원래 LRTT)
                                #   "decay": A*=decay_factor, B*=decay_factor (점진적 감쇠)
                                #   "hybrid": A=0, B*=decay_factor (하이브리드)
                                #   "orthogonal": A=0, B=직교행렬(고정) - B@B^T=I, 투영 유지
REINIT_DECAY_FACTOR = 0.9       # decay/hybrid 모드 감쇠 계수 (0 < factor < 1)

# ==============================================================================
# Read Noise Reduction (읽기 노이즈 감소)
# ==============================================================================

# --- 오버샘플링 ---
READ_N_AVG = 1                  # 오버샘플링 횟수 (노이즈 1/√N 감소)
                                #   1: 기본 단일 읽기
                                #   4-8: 권장 (노이즈 2~2.8x 감소)

# --- Legacy 멀티-리드 (호환성용) ---
NUM_READS = 1                   # (구버전 호환) per-rank 읽기 횟수
MULTI_READ_MODE = "average"     # "average": 평균 후 전송, "per_read": 읽기마다 전송

# ==============================================================================
# AGC (Automatic Gain Control) - 자동 이득 제어
# ==============================================================================

AGC_ENABLED = False             # AGC 활성화 (읽기 amplitude 최적화)
                                # True: Binary search로 ADC 클리핑 없이 SNR 최대화
AGC_MARGIN = 0.85               # 출력 경계 비율 (0.85 = 85%, 클리핑 방지 마진)
AGC_MAX_ITERS = 6               # AGC binary search 최대 반복 횟수

# ==============================================================================
# Two-Amplitude Differential Read (홀수차 왜곡 제거)
# ==============================================================================

TWO_AMP_ENABLED = False         # Two-amplitude 읽기 활성화
                                # True: 두 amplitude로 홀수차 왜곡 상쇄
                                #   d(α1) = α1·w + b_odd
                                #   d(α2) = α2·w + b_odd
                                #   → w = (d(α2) - d(α1)) / (α2 - α1)
TWO_AMP_RATIO = 0.5             # α1/α2 비율 (기본 0.5 = 저amplitude가 고amplitude의 절반)

# ==============================================================================
# Update Mode (A/B 업데이트 방식)
# ==============================================================================

UPDATE_MODE = "lora"            # A/B 업데이트 모드
                                #   "lora": LoRA chain rule (원래 LRTT, forward_inject=True 필요)
                                #     ΔA = -lr * D^T @ (B@X)
                                #     ΔB = -lr * (A^T@D)^T @ X
                                #
                                #   "reconstruction": TikiTaka 스타일 gradient reconstruction
                                #     forward_inject=False에서 사용
                                #     A@B ≈ -G (C의 이상적 gradient) 근사
                                #     L_rec = 0.5*||AB + G||_F² + (λ_A/2)*||A||_F² + (λ_B/2)*||B||_F²

# ==============================================================================
# Reconstruction Update Parameters (UPDATE_MODE="reconstruction" 시)
# ==============================================================================

RECON_LAMBDA_A = 1e-3           # A에 대한 L2 정규화 계수
RECON_LAMBDA_B = 1e-3           # B에 대한 L2 정규화 계수

RECON_USE_SCALAR_STABILIZER = False  # 스칼라 근사 안정화 사용
                                     # True: BB^T ≈ sB*I, A^TA ≈ sA*I로 근사
                                     # 계산 효율적이나 덜 정확

RECON_USE_EXACT_GRAM = False    # 정확한 Gram 행렬 사용 (디버깅용)
                                # True: BB^T, A^TA 정확히 계산 (O(rank^2) 비용)

RECON_EXACT_GRAM_EVERY = 0      # N 스텝마다 exact Gram 사용 (0 = 비활성)
                                # 주기적 정확 안정화용

RECON_EMA_BETA = 0.9            # sA, sB norm 추적용 EMA 감쇠 (0.9~0.99 권장)

RECON_LR_SCALE = 1.0            # reconstruction 업데이트 추가 lr 스케일 (0.1~1.0)

RECON_USE_CLIP_NORM = False     # A,B norm 클리핑 활성화 (안전 fallback)
RECON_CLIP_NORM = 10.0          # 최대 norm (RECON_USE_CLIP_NORM=True 시)

# ==============================================================================
# Device Configuration
# ==============================================================================

# A/B tile device type
USE_6T1C_AB = False              # A/B 타일에 6T1C 디바이스 사용
                                # True: 6T1C (커패시터 기반, retention decay 있음)
                                # False: IdealizedPresetDevice (이상적, 노이즈만)

# FloatingPoint (완전 digital) 모드
USE_FLOATING_POINT = False       # True: 모든 타일을 FloatingPointDevice로 사용 (완전 digital, 노이즈 없음)
                                # False: 위 설정에 따라 analog 디바이스 사용



# ==============================================================================
# FullAnalog tile for Stage 1
# ==============================================================================

from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from aihwkit.exceptions import TileError
from dataclasses import dataclass

class LRTTSimulatorTileFullAnalog(LRTTSimulatorTile):
    """Regular LRTT tile for fullanalog training (C-only update)."""

    def _hook_tile_updates(self) -> None:
        """Override parent's hook to update only C matrix for fullanalog training."""
        # Store original update methods
        if hasattr(self, 'tile_a'):
            self.tile_a._orig_update = self.tile_a.update
        if hasattr(self, 'tile_b'):
            self.tile_b._orig_update = self.tile_b.update
        self.tile_c._orig_update = self.tile_c.update

        # Track if we've already handled this batch
        self._update_handled = False

        parent_tile = self

        # Hook tile_c.update() to update C only
        def tile_c_update_wrapper(x_input, d_input, bias=False, in_trans=False,
                                 out_trans=False, non_blocking=False):
            if bias:
                raise TileError("LRTT does not support bias")

            # Prevent double updates
            if parent_tile._update_handled:
                return None
            parent_tile._update_handled = True

            # Update C tile directly
            parent_tile.tile_c._orig_update(x_input, d_input)
            return None

        # Hook tile_a and tile_b to do nothing (no A, B updates for fullanalog)
        def noop_update(*args, **kwargs):
            return None

        # Replace update methods
        if hasattr(self, 'tile_a'):
            self.tile_a.update = noop_update
        if hasattr(self, 'tile_b'):
            self.tile_b.update = noop_update
        self.tile_c.update = tile_c_update_wrapper


@dataclass
class PythonLRTTDeviceFullAnalog(PythonLRTTDevice):
    """Regular LRTT device for fullanalog training."""

    def get_default_tile_module_class(self):
        """Return the fullanalog regular LRTT tile class."""
        return LRTTSimulatorTileFullAnalog


def create_fullanalog_config(rank=1):
    """Create FullAnalog LRTT configuration (C-only training)."""

    # C tile configuration (A/B not used in fullanalog mode)
    if USE_FLOATING_POINT:
        c_device = FloatingPointDevice()
    else:
        c_device = IdealizedPresetDevice(
            dw_min=0.0002,
            dw_min_dtod=0.3,
            dw_min_std=0.3,
            up_down=0.0,
            up_down_dtod=0.0,
            w_max=1.0,
            w_min=-1.0,
            w_max_dtod=0.3,
            w_min_dtod=0.3,
        )

    device_config = PythonLRTTDeviceFullAnalog(
        rank=rank,  # Minimal rank (not used in forward)
        transfer_every=10000000000,  # Very large to avoid transfers
        lora_alpha=1.0,
        forward_inject=True,  # Use only C matrix in forward pass
        correct_gradient_magnitudes=False,
        unit_cell_devices=[
            IdealizedPresetDevice(),  # A tile (not used)
            IdealizedPresetDevice(),  # B tile (not used)
            c_device                  # C tile (main weight matrix)
        ]
    )

    # Add mapping configuration (same as rlrtt_scratch)
    mapping = MappingParameter(
        weight_scaling_omega=0.0,  #0.6
        learn_out_scaling=False,
        weight_scaling_lr_compensation=False,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=False,
        max_input_size=512,
        max_output_size=512
    )

    # Add forward/backward IO configuration (same as rlrtt_scratch)
    forward_io = IOParameters(
        inp_res=0.00,   #0.007937
        inp_bound=1.0,
        inp_noise=0.0,
        inp_sto_round=False,
        out_res=0.00,   #0.001961
        out_bound=12.0,
        out_noise=0.0,     #0.06
        w_noise=0.0,
        w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=False,   #False
        max_bm_factor=1000,
    )

    # Update parameters - disable BL management for debugging
    update_params = UpdateParameters(
        desired_bl=31,
        update_bl_management=True,
        update_management=True,
    )

    return PythonLRTTRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io, update=update_params)


# ==============================================================================
# LRTT configuration for Stage 2
# ==============================================================================

def create_6t1c_device(dt_batch_sec=1.0, include_retention=True):
    """Create 6T1C device for A/B tiles.

    6T1C Device Characteristics:
        - ~1000 conductance states per direction
        - Capacitor-based weight storage with exponential decay
        - Time constant τ ≈ 775 min (12.9 hours)

    Args:
        dt_batch_sec: Assumed time per mini-batch in seconds (for retention calculation)
        include_retention: Whether to include retention effects
    """
    import math

    # Calculate lifetime from physical τ for 6T1C
    TAU_SEC = 46505.0  # Physical time constant: 775.1 min = 46505 sec
    if include_retention and dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        lifetime = 1.0 / delta
    else:
        lifetime = 0.0  # No retention

    return LinearStepDevice(
        # Core update parameters (fitted from 6T1C data)
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,

        # Device-to-device variation
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,

        # Cycle-to-cycle variation
        dw_min_std=0.3,
        write_noise_std=0.0182,

        # LinearStepDevice specific
        mean_bound_reference=True,

        # Retention (capacitor leakage)
        lifetime=lifetime,
        lifetime_dtod=0.1 if include_retention else 0.0,
        reset=0.0,  # Decay toward 0V
        reset_dtod=0.0,
    )


def create_lrtt_config(rank, is_conv=True):
    """Create LRTT configuration for Stage 2.

    Uses global configuration variables defined at the top of the file.
    All LRTT features can be configured via those variables.
    """

    # FloatingPoint mode: 완전 digital (노이즈 없음)
    if USE_FLOATING_POINT:
        ab_device = FloatingPointDevice()
        c_device = FloatingPointDevice()
    else:
        # Select devices for A/B tiles
        if USE_6T1C_AB:
            ab_device = create_6t1c_device()
        else:
            ab_device = IdealizedPresetDevice(
                dw_min=0.0002,
                dw_min_dtod=0.3,
                dw_min_std=0.3,
                up_down=0.0,
                up_down_dtod=0.0,
                w_max=1.0,
                w_min=-1.0,
                w_max_dtod=0.3,
                w_min_dtod=0.3,
            )

        # C tile uses IdealizedPresetDevice
        c_device = IdealizedPresetDevice(
            dw_min=0.0002,
            dw_min_dtod=0.3,
            dw_min_std=0.3,
            up_down=0.0,
            up_down_dtod=0.0,
            w_max=1.0,
            w_min=-1.0,
            w_max_dtod=0.3,
            w_min_dtod=0.3,
        )

    device_config = PythonLRTTDevice(
        rank=rank,
        # --- 기본 LRTT 파라미터 ---
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        transfer_lr=TRANSFER_LR,
        transfer_lr_scale=TRANSFER_LR_SCALE,

        # --- Forward/Update 모드 ---
        forward_inject=False,  # Use C matrix only in forward (A⊗B accumulated via transfers)
        update_mode=UPDATE_MODE,
        correct_gradient_magnitudes=True,

        # --- Transfer 방식 ---
        use_onehot_transfer=USE_ONEHOT_TRANSFER,

        # --- Reinit 모드 ---
        reinit_gain=0.1,
        reinit_mode=REINIT_MODE,
        decay_factor=REINIT_DECAY_FACTOR,

        # --- Transfer 캘리브레이션 ---
        transfer_mode=TRANSFER_MODE,
        transfer_micro_steps=TRANSFER_MICRO_STEPS,
        transfer_pilot_frac=TRANSFER_PILOT_FRAC,
        sd_quantum=SD_QUANTUM,

        # --- Read noise reduction ---
        read_n_avg=READ_N_AVG,
        num_reads=NUM_READS,  # Legacy compatibility
        multi_read_mode=MULTI_READ_MODE,

        # --- AGC ---
        agc_enabled=AGC_ENABLED,
        agc_margin=AGC_MARGIN,
        agc_max_iters=AGC_MAX_ITERS,

        # --- Two-Amplitude ---
        two_amp_enabled=TWO_AMP_ENABLED,
        two_amp_ratio=TWO_AMP_RATIO,

        # --- Reconstruction parameters ---
        recon_lambda_a=RECON_LAMBDA_A,
        recon_lambda_b=RECON_LAMBDA_B,
        recon_use_scalar_stabilizer=RECON_USE_SCALAR_STABILIZER,
        recon_use_exact_gram=RECON_USE_EXACT_GRAM,
        recon_exact_gram_every=RECON_EXACT_GRAM_EVERY,
        recon_ema_beta=RECON_EMA_BETA,
        recon_lr_scale=RECON_LR_SCALE,
        recon_use_clip_norm=RECON_USE_CLIP_NORM,
        recon_clip_norm=RECON_CLIP_NORM,

        # --- Device configuration ---
        unit_cell_devices=[ab_device, ab_device, c_device]
    )

    # Add mapping configuration (same as rlrtt_scratch)
    mapping = MappingParameter(
        weight_scaling_omega=0.0,    #0.6
        learn_out_scaling=False,
        weight_scaling_lr_compensation=False,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=False,
        max_input_size=512,
        max_output_size=512
    )

    # Add forward/backward IO configuration (same as rlrtt_scratch)
    forward_io = IOParameters(
        inp_res=0.00, #0.007937
        inp_bound=1.0,
        inp_noise=0.0,
        inp_sto_round=False,
        out_res=0.00,  #0.001961
        out_bound=12.0,
        out_noise=0.0,  #0.06
        w_noise=0.0,
        w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=False,   #False
        max_bm_factor=1000,
    )

    # Update parameters - disable BL management for debugging
    update_params = UpdateParameters(
        desired_bl=31,
        update_bl_management=True,
        update_management=True,
    )

    return PythonLRTTRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io, update=update_params)


# ==============================================================================
# ResNet18 model builders
# ==============================================================================

class BasicBlock(nn.Module):
    """ResNet BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, rpu_config=None, use_analog=True):
        super().__init__()

        if use_analog:
            self.conv1 = AnalogConv2d(in_planes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False, rpu_config=rpu_config)
            self.conv2 = AnalogConv2d(planes, planes, kernel_size=3, stride=1,
                                     padding=1, bias=False, rpu_config=rpu_config)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                  padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if use_analog:
                self.shortcut = nn.Sequential(
                    AnalogConv2d(in_planes, self.expansion * planes, kernel_size=1,
                               stride=stride, bias=False, rpu_config=rpu_config),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                            stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet18 for CIFAR-10."""

    def __init__(self, block, num_blocks, num_classes=10, rpu_config=None,
                 conv1_use_analog=False, fc_use_analog=False):
        super().__init__()
        self.in_planes = 64

        # First conv layer (digital for both stages)
        if conv1_use_analog:
            self.conv1 = AnalogConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                     bias=False, rpu_config=rpu_config)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        # ResNet layers (analog)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, rpu_config=rpu_config)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, rpu_config=rpu_config)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, rpu_config=rpu_config)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, rpu_config=rpu_config)

        # Final FC layer (digital for both stages)
        if fc_use_analog:
            # Note: AnalogLinear not imported, using Conv2d hack
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, rpu_config):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rpu_config, use_analog=True))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def build_fullanalog_model():
    """Build FullAnalog model for Stage 1."""
    fullanalog_config = create_fullanalog_config(rank=1)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=N_CLASSES,
                   rpu_config=fullanalog_config,
                   conv1_use_analog=False, fc_use_analog=False)

    if USE_CUDA:
        model = model.to(DEVICE)

    print("\n" + "="*70)
    print("Stage 1: FullAnalog Model")
    print("="*70)
    print("  - All ResNet blocks: FullAnalog (C-only training)")
    print("  - conv1: Digital (FloatingPoint)")
    print("  - fc: Digital (FloatingPoint)")
    print(f"  - Total epochs: {N_EPOCHS_STAGE1}")
    print(f"  - Learning rate: {LEARNING_RATE_STAGE1} ({LR_SCHEDULE_STAGE1})")
    if LR_SCHEDULE_STAGE1 == "cosine":
        print(f"  - Warmup ratio: {WARMUP_RATIO_STAGE1}")
    elif LR_SCHEDULE_STAGE1 == "multistep":
        print(f"  - Milestones: {LR_MILESTONES_STAGE1}, gamma: {LR_GAMMA_STAGE1}")
    print("="*70 + "\n")

    return model


def configure_lrtt_controllers(model):
    """Configure LRTT controller settings from global variables.

    Sets all LRTT controller attributes including:
    - Transfer settings: micro_steps, centering, normalize, mode, pilot_frac, sd_quantum
    - Read noise reduction: read_n_avg
    - AGC: agc_enabled, agc_margin, agc_max_iters
    - Two-amplitude: two_amp_enabled, two_amp_ratio
    - Reconstruction: all recon_* parameters
    """
    configured = 0
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            ctrl = module.analog_module.controller

            # --- Transfer 캘리브레이션 ---
            ctrl.transfer_micro_steps = TRANSFER_MICRO_STEPS
            ctrl.transfer_mode = TRANSFER_MODE
            ctrl.transfer_pilot_frac = TRANSFER_PILOT_FRAC
            ctrl.sd_quantum = SD_QUANTUM

            # --- Transfer 전처리 ---
            ctrl.transfer_centering = TRANSFER_CENTERING
            ctrl.transfer_normalize = TRANSFER_NORMALIZE

            # --- Read noise reduction ---
            ctrl.read_n_avg = READ_N_AVG

            # --- AGC ---
            ctrl.agc_enabled = AGC_ENABLED
            ctrl.agc_margin = AGC_MARGIN
            ctrl.agc_max_iters = AGC_MAX_ITERS

            # --- Two-Amplitude ---
            ctrl.two_amp_enabled = TWO_AMP_ENABLED
            ctrl.two_amp_ratio = TWO_AMP_RATIO

            # --- Reconstruction parameters ---
            ctrl.recon_lambda_a = RECON_LAMBDA_A
            ctrl.recon_lambda_b = RECON_LAMBDA_B
            ctrl.recon_use_scalar_stabilizer = RECON_USE_SCALAR_STABILIZER
            ctrl.recon_use_exact_gram = RECON_USE_EXACT_GRAM
            ctrl.recon_exact_gram_every = RECON_EXACT_GRAM_EVERY
            ctrl.recon_ema_beta = RECON_EMA_BETA
            ctrl.recon_lr_scale = RECON_LR_SCALE
            ctrl.recon_use_clip_norm = RECON_USE_CLIP_NORM
            ctrl.recon_clip_norm = RECON_CLIP_NORM

            configured += 1
    return configured


def build_lrtt_model():
    """Build LRTT model for Stage 2 (without weight loading)."""
    lrtt_config_conv = create_lrtt_config(LRTT_RANK_CONV, is_conv=True)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=N_CLASSES,
                   rpu_config=lrtt_config_conv,
                   conv1_use_analog=False, fc_use_analog=False)

    if USE_CUDA:
        model = model.to(DEVICE)

    # Configure LRTT controller transfer robustness settings
    num_configured = configure_lrtt_controllers(model)

    print("\n" + "="*70)
    print("Stage 2: LRTT Model")
    print("="*70)
    print(f"  - All ResNet blocks: LRTT (rank={LRTT_RANK_CONV})")
    print("  - conv1: Digital (FloatingPoint)")
    print("  - fc: Digital (FloatingPoint)")
    print(f"  - Transfer every: {TRANSFER_EVERY} steps")
    print(f"  - LoRA alpha: {LORA_ALPHA}")
    print(f"  - Transfer robustness:")
    print(f"      micro_steps={TRANSFER_MICRO_STEPS}, mode='{TRANSFER_MODE}'")
    print(f"      pilot_frac={TRANSFER_PILOT_FRAC:.4f}, sd_quantum={SD_QUANTUM}")
    print(f"      centering={TRANSFER_CENTERING}, normalize={TRANSFER_NORMALIZE}")
    print(f"  - Read noise reduction:")
    print(f"      num_reads={NUM_READS}, mode='{MULTI_READ_MODE}', read_n_avg={READ_N_AVG}")
    print(f"  - AGC: enabled={AGC_ENABLED}, margin={AGC_MARGIN}")
    print(f"  - Two-Amp: enabled={TWO_AMP_ENABLED}, ratio={TWO_AMP_RATIO}")
    print(f"  - Update/Reinit: mode={UPDATE_MODE}, reinit={REINIT_MODE}")
    print(f"  - Device: 6T1C_AB={USE_6T1C_AB}, onehot={USE_ONEHOT_TRANSFER}")
    print(f"  - Total epochs: {N_EPOCHS_STAGE2}")
    print(f"  - Learning rate: {LEARNING_RATE_STAGE2} ({LR_SCHEDULE_STAGE2})")
    if LR_SCHEDULE_STAGE2 == "cosine":
        print(f"  - Warmup ratio: {WARMUP_RATIO_STAGE2}")
    elif LR_SCHEDULE_STAGE2 == "multistep":
        print(f"  - Milestones: {LR_MILESTONES_STAGE2}, gamma: {LR_GAMMA_STAGE2}")
    print(f"  - Configured {num_configured} LRTT controllers")
    print("="*70 + "\n")

    return model


# ==============================================================================
# Weight transfer from FullAnalog to LRTT
# ==============================================================================

@torch.no_grad()
def transfer_weights_fullanalog_to_lrtt(fullanalog_model, lrtt_model):
    """Transfer weights from FullAnalog model to LRTT model.

    Transfers:
    1. Analog C matrices (FullAnalog → LRTT C tiles)
    2. BatchNorm parameters (weight, bias, running_mean, running_var)
    3. Digital layers (conv1, fc)

    LRTT A, B matrices are reinitialized (A=0, B=Kaiming).
    """
    print("\n" + "="*70)
    print("Transferring weights: FullAnalog → LRTT")
    print("="*70)

    transferred = 0

    # Get state dicts
    fullanalog_dict = fullanalog_model.state_dict()
    lrtt_dict = lrtt_model.state_dict()

    # Transfer BatchNorm and digital layer parameters (skip analog_module entirely)
    for name in lrtt_dict.keys():
        if name in fullanalog_dict:
            # Only transfer if it's NOT analog_module related
            if 'analog_module' not in name:
                # Check if it's a tensor (not dict or other types)
                if isinstance(fullanalog_dict[name], torch.Tensor):
                    lrtt_dict[name].copy_(fullanalog_dict[name])
                    transferred += 1

    # Load state dict
    lrtt_model.load_state_dict(lrtt_dict, strict=False)

    print(f"✓ Transferred {transferred} non-analog parameters (BatchNorm, conv1, fc)")

    # Transfer analog C matrices
    analog_transferred = 0
    for (fa_name, fa_module), (lrtt_name, lrtt_module) in zip(
        fullanalog_model.named_modules(), lrtt_model.named_modules()
    ):
        if isinstance(fa_module, AnalogConv2d) and isinstance(lrtt_module, AnalogConv2d):
            if hasattr(fa_module, 'analog_module') and hasattr(lrtt_module, 'analog_module'):
                # Get FullAnalog weights
                if hasattr(fa_module.analog_module, 'get_lrtt_component_weights'):
                    C_fa, _, _ = fa_module.analog_module.get_lrtt_component_weights()
                else:
                    C_fa = fa_module.analog_module.get_weights()[0]

                # Set LRTT C matrix
                if hasattr(lrtt_module.analog_module, 'set_lrtt_component_weights'):
                    C_lrtt, A_lrtt, B_lrtt = lrtt_module.analog_module.get_lrtt_component_weights()
                    lrtt_module.analog_module.set_lrtt_component_weights(
                        C_fa.to(C_lrtt.device), A_lrtt, B_lrtt
                    )
                    analog_transferred += 1
                    print(f"  ✓ {fa_name}: C matrix transferred")

    print(f"✓ Transferred {analog_transferred} analog C matrices")

    # Reinitialize LRTT A, B matrices
    reinit_count = 0
    for name, module in lrtt_model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            module.analog_module.controller.reinit()
            reinit_count += 1

    print(f"✓ Reinitialized {reinit_count} LRTT layers (A=0, B=Kaiming)")
    print("="*70 + "\n")

    return lrtt_model


# ==============================================================================
# Data loading
# ==============================================================================

def load_images():
    """Load CIFAR-10 dataset."""
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Validation transforms (no augmentation)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(PATH_DATASET, train=True, download=True,
                                     transform=transform_train)
    val_dataset = datasets.CIFAR10(PATH_DATASET, train=False, download=True,
                                   transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader


# ==============================================================================
# Training and evaluation
# ==============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()

    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output.data, 1)
        epoch_total += labels.size(0)
        epoch_correct += (predicted == labels).sum().item()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}',
                         'Acc': f'{100 * epoch_correct / epoch_total:.2f}%'})

    train_loss = epoch_loss / len(train_loader.dataset)
    train_acc = 100 * epoch_correct / epoch_total

    return train_loss, train_acc


def evaluate(model, val_loader, criterion):
    """Evaluate model."""
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            output = model(images)
            loss = criterion(output, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100 * val_correct / val_total

    return val_loss, val_acc


def apply_constant_lr(optimizer, base_lr):
    """Apply constant learning rate (no decay)."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr
    return base_lr


def apply_warmup_cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_ratio=0.0, min_lr=1e-5):
    """Apply learning rate warmup + cosine annealing (epoch-based)."""
    import math

    warmup_epochs = int(total_epochs * warmup_ratio)

    if epoch <= warmup_epochs:
        # Linear warmup
        lr = base_lr * (epoch / warmup_epochs) if warmup_epochs > 0 else base_lr
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def apply_multistep_lr(optimizer, epoch, base_lr, milestones, gamma=0.1):
    """Apply multi-step learning rate decay.

    Args:
        optimizer: Optimizer to update
        epoch: Current epoch (1-indexed)
        base_lr: Initial learning rate
        milestones: List of epoch indices where LR is decayed
        gamma: Multiplicative factor for LR decay

    Returns:
        Current learning rate
    """
    lr = base_lr
    for milestone in milestones:
        if epoch > milestone:
            lr *= gamma

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def apply_lr_schedule(optimizer, epoch, total_epochs, base_lr, schedule="cosine",
                      warmup_ratio=0.0, milestones=None, gamma=0.1, min_lr=1e-5):
    """Apply learning rate schedule based on schedule type.

    Args:
        optimizer: Optimizer to update
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        base_lr: Initial learning rate
        schedule: Schedule type - "constant", "cosine", or "multistep"
        warmup_ratio: Warmup ratio for cosine schedule
        milestones: Milestone epochs for multistep schedule
        gamma: Decay factor for multistep schedule
        min_lr: Minimum LR for cosine schedule

    Returns:
        Current learning rate
    """
    if schedule == "constant":
        return apply_constant_lr(optimizer, base_lr)
    elif schedule == "cosine":
        return apply_warmup_cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_ratio, min_lr)
    elif schedule == "multistep":
        return apply_multistep_lr(optimizer, epoch, base_lr, milestones or [], gamma)
    else:
        raise ValueError(f"Unknown LR schedule: {schedule}. Must be 'constant', 'cosine', or 'multistep'")


# ==============================================================================
# Main training loop
# ==============================================================================

def train_2stage():
    """Two-stage training: FullAnalog → LRTT."""
    torch.manual_seed(SEED)

    # Initialize wandb
    # Build detailed run name with all important parameters

    # Get config for IO/mapping parameters (each stage has its own)
    s1_config = create_fullanalog_config(rank=1)
    s1_mapping = s1_config.mapping
    s1_fwd = s1_config.forward
    s1_inp_res = 1.0/(2**7-2) if s1_fwd.inp_res == -1 else s1_fwd.inp_res
    s1_out_res = 1.0/(2**9-2) if s1_fwd.out_res == -1 else s1_fwd.out_res
    # Stage 1 C tile device params
    s1_c_device = s1_config.device.unit_cell_devices[2]  # C tile is index 2

    s2_config = create_lrtt_config(LRTT_RANK_CONV, is_conv=True)
    s2_mapping = s2_config.mapping
    s2_fwd = s2_config.forward
    s2_inp_res = 1.0/(2**7-2) if s2_fwd.inp_res == -1 else s2_fwd.inp_res
    s2_out_res = 1.0/(2**9-2) if s2_fwd.out_res == -1 else s2_fwd.out_res
    # Stage 2 device params: A/B (gradient update) and C (transfer update)
    s2_ab_device = s2_config.device.unit_cell_devices[0]  # A/B tile
    s2_c_device = s2_config.device.unit_cell_devices[2]   # C tile

    # ===== Common parameters (truly shared) =====
    common_name = f"bs{BATCH_SIZE}_wd{WEIGHT_DECAY}"

    # ===== Stage 1 specific parameters =====
    s1_name = f"S1_e{N_EPOCHS_STAGE1}_lr{LEARNING_RATE_STAGE1}_{LR_SCHEDULE_STAGE1}"
    if LR_SCHEDULE_STAGE1 == "cosine" and WARMUP_RATIO_STAGE1 > 0:
        s1_name += f"_wr{WARMUP_RATIO_STAGE1}"
    elif LR_SCHEDULE_STAGE1 == "multistep":
        s1_name += f"_ms{LR_MILESTONES_STAGE1}_g{LR_GAMMA_STAGE1}"
    # Stage 1 IO/mapping/device
    s1_name += f"_fwdIR{s1_inp_res:.6f}_fwdOR{s1_out_res:.6f}_fwdIN{s1_fwd.inp_noise}_fwdON{s1_fwd.out_noise}_mapW{s1_mapping.weight_scaling_omega}"
    s1_name += f"_dwmin{s1_c_device.dw_min}_dwdtod{s1_c_device.dw_min_dtod}_dwstd{s1_c_device.dw_min_std}"

    # ===== Stage 2 specific parameters =====
    s2_name = f"S2_e{N_EPOCHS_STAGE2}_lr{LEARNING_RATE_STAGE2}_{LR_SCHEDULE_STAGE2}"
    if LR_SCHEDULE_STAGE2 == "cosine" and WARMUP_RATIO_STAGE2 > 0:
        s2_name += f"_wr{WARMUP_RATIO_STAGE2}"
    elif LR_SCHEDULE_STAGE2 == "multistep":
        s2_name += f"_ms{LR_MILESTONES_STAGE2}_g{LR_GAMMA_STAGE2}"

    # LRTT config (Stage 2 only)
    s2_name += f"_r{LRTT_RANK_CONV}_a{LORA_ALPHA}_te{TRANSFER_EVERY}_fi{int(s2_config.device.forward_inject)}"

    # Transfer settings (Stage 2 only)
    s2_name += f"_tm{TRANSFER_MODE}_tms{TRANSFER_MICRO_STEPS}"
    if USE_ONEHOT_TRANSFER:
        s2_name += "_onehot"

    # Update/Reinit mode (Stage 2 only)
    s2_name += f"_up{UPDATE_MODE}_ri{REINIT_MODE}"
    if REINIT_MODE in ["decay", "hybrid"]:
        s2_name += f"_df{REINIT_DECAY_FACTOR}"

    # Device config (Stage 2 only)
    s2_name += f"_{'6T1C' if USE_6T1C_AB else 'ideal'}"

    # Read noise reduction (Stage 2 only)
    s2_name += f"_navg{READ_N_AVG}"
    if NUM_READS > 1:
        s2_name += f"_nr{NUM_READS}_{MULTI_READ_MODE}"

    # AGC/Two-Amp (Stage 2 only)
    s2_name += f"_agc{int(AGC_ENABLED)}_2amp{int(TWO_AMP_ENABLED)}"

    # Stage 2 IO/mapping
    s2_name += f"_fwdIR{s2_inp_res:.6f}_fwdOR{s2_out_res:.6f}_fwdIN{s2_fwd.inp_noise}_fwdON{s2_fwd.out_noise}_mapW{s2_mapping.weight_scaling_omega}"
    # Stage 2 device params: AB (gradient) and C (transfer)
    s2_name += f"_ABdwmin{s2_ab_device.dw_min}_ABdwdtod{s2_ab_device.dw_min_dtod}_ABdwstd{s2_ab_device.dw_min_std}"
    s2_name += f"_Cdwmin{s2_c_device.dw_min}_Cdwdtod{s2_c_device.dw_min_dtod}_Cdwstd{s2_c_device.dw_min_std}"

    # Combine: Common | Stage1 | Stage2
    run_name = f"{common_name}__{s1_name}__{s2_name}"

    wandb.init(
        project="cifar10-resnet18-2stage-warmstart",
        name=run_name,
        config={
            "stage1_epochs": N_EPOCHS_STAGE1,
            "stage1_lr": LEARNING_RATE_STAGE1,
            "stage1_warmup": WARMUP_RATIO_STAGE1,
            "stage1_lr_schedule": LR_SCHEDULE_STAGE1,
            "stage1_lr_milestones": LR_MILESTONES_STAGE1,
            "stage1_lr_gamma": LR_GAMMA_STAGE1,
            "stage2_epochs": N_EPOCHS_STAGE2,
            "stage2_lr": LEARNING_RATE_STAGE2,
            "stage2_warmup": WARMUP_RATIO_STAGE2,
            "stage2_lr_schedule": LR_SCHEDULE_STAGE2,
            "stage2_lr_milestones": LR_MILESTONES_STAGE2,
            "stage2_lr_gamma": LR_GAMMA_STAGE2,
            "batch_size": BATCH_SIZE,
            "lrtt_rank": LRTT_RANK_CONV,
            "lora_alpha": LORA_ALPHA,
            "transfer_every": TRANSFER_EVERY,
            # Transfer robustness settings
            "transfer_micro_steps": TRANSFER_MICRO_STEPS,
            "transfer_mode": TRANSFER_MODE,
            "transfer_pilot_frac": TRANSFER_PILOT_FRAC,
            "sd_quantum": SD_QUANTUM,
            # Transfer preprocessing
            "transfer_centering": TRANSFER_CENTERING,
            "transfer_normalize": TRANSFER_NORMALIZE,
            # Multi-read / oversampling settings
            "num_reads": NUM_READS,
            "multi_read_mode": MULTI_READ_MODE,
            "read_n_avg": READ_N_AVG,
            # AGC settings
            "agc_enabled": AGC_ENABLED,
            "agc_margin": AGC_MARGIN,
            "agc_max_iters": AGC_MAX_ITERS,
            # Two-amplitude settings
            "two_amp_enabled": TWO_AMP_ENABLED,
            "two_amp_ratio": TWO_AMP_RATIO,
            # Update mode
            "update_mode": UPDATE_MODE,
            # Reinit settings
            "reinit_mode": REINIT_MODE,
            "reinit_decay_factor": REINIT_DECAY_FACTOR,
            # Reconstruction parameters
            "recon_lambda_a": RECON_LAMBDA_A,
            "recon_lambda_b": RECON_LAMBDA_B,
            "recon_use_scalar_stabilizer": RECON_USE_SCALAR_STABILIZER,
            "recon_use_exact_gram": RECON_USE_EXACT_GRAM,
            "recon_exact_gram_every": RECON_EXACT_GRAM_EVERY,
            "recon_ema_beta": RECON_EMA_BETA,
            "recon_lr_scale": RECON_LR_SCALE,
            "recon_use_clip_norm": RECON_USE_CLIP_NORM,
            "recon_clip_norm": RECON_CLIP_NORM,
            # Device settings
            "use_6t1c_ab": USE_6T1C_AB,
            # A/B device params (read from actual config)
            "ab_dw_min": s2_ab_device.dw_min,
            "ab_dw_min_dtod": s2_ab_device.dw_min_dtod,
            "ab_dw_min_std": s2_ab_device.dw_min_std,
            # C device params (read from actual config)
            "c_dw_min": s2_c_device.dw_min,
            "c_dw_min_dtod": s2_c_device.dw_min_dtod,
            "c_dw_min_std": s2_c_device.dw_min_std,
            # Transfer method
            "use_onehot_transfer": USE_ONEHOT_TRANSFER,
            "transfer_lr_scale": TRANSFER_LR_SCALE,
            # Forward inject
            "forward_inject": s2_config.device.forward_inject,
        }
    )

    # Load data
    train_loader, val_loader = load_images()
    criterion = nn.CrossEntropyLoss()

    # ========================================================================
    # Stage 1: FullAnalog training
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: FullAnalog Training")
    print("="*70 + "\n")

    model_stage1 = build_fullanalog_model()
    optimizer_stage1 = AnalogSGD(model_stage1.parameters(), lr=LEARNING_RATE_STAGE1,
                                 momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
    optimizer_stage1.regroup_param_groups(model_stage1)

    # Handle N_EPOCHS_STAGE1 = 0 case (skip Stage 1, use random init)
    if N_EPOCHS_STAGE1 == 0:
        print("N_EPOCHS_STAGE1 = 0: Skipping Stage 1 training (random initialization)")
        val_loss, val_acc = evaluate(model_stage1, val_loader, criterion)
        print(f"Initial Val Accuracy (random): {val_acc:.2f}%\n")
    else:
        for epoch in range(N_EPOCHS_STAGE1):
            # Apply LR schedule for Stage 1
            lr = apply_lr_schedule(
                optimizer_stage1, epoch + 1, N_EPOCHS_STAGE1, LEARNING_RATE_STAGE1,
                schedule=LR_SCHEDULE_STAGE1, warmup_ratio=WARMUP_RATIO_STAGE1,
                milestones=LR_MILESTONES_STAGE1, gamma=LR_GAMMA_STAGE1
            )

            # Train
            train_loss, train_acc = train_one_epoch(model_stage1, train_loader, optimizer_stage1, criterion)

            # Validate
            val_loss, val_acc = evaluate(model_stage1, val_loader, criterion)

            print(f"[Stage1 {epoch+1:02d}/{N_EPOCHS_STAGE1}] "
                  f"Loss={train_loss:.4f} TrainAcc={train_acc:.2f}% "
                  f"ValAcc={val_acc:.2f}% LR={lr:.6f} ({LR_SCHEDULE_STAGE1})")

            wandb.log({
                "stage": 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": lr,
            })

        print(f"\n✓ Stage 1 Complete: Val Accuracy = {val_acc:.2f}%\n")

    # ========================================================================
    # Weight Transfer: FullAnalog → LRTT
    # ========================================================================
    model_stage2 = build_lrtt_model()

    # ========================================================================
    # DEBUG: Compare forward pass BEFORE transfer
    # ========================================================================
    print("\n" + "="*70)
    print("DEBUG: Comparing Forward Pass Before/After Transfer")
    print("="*70)

    # Get a sample batch for comparison
    sample_images, sample_labels = next(iter(val_loader))
    sample_images = sample_images.to(DEVICE)

    # Forward pass with FullAnalog model
    model_stage1.eval()
    with torch.no_grad():
        output_fullanalog = model_stage1(sample_images)

    # Forward pass with LRTT model BEFORE transfer (random init)
    model_stage2.eval()
    with torch.no_grad():
        output_lrtt_before = model_stage2(sample_images)

    print(f"FullAnalog output mean: {output_fullanalog.mean().item():.6f}, std: {output_fullanalog.std().item():.6f}")
    print(f"LRTT (before transfer) output mean: {output_lrtt_before.mean().item():.6f}, std: {output_lrtt_before.std().item():.6f}")

    # Transfer weights
    model_stage2 = transfer_weights_fullanalog_to_lrtt(model_stage1, model_stage2)

    # Forward pass with LRTT model AFTER transfer
    model_stage2.eval()
    with torch.no_grad():
        output_lrtt_after = model_stage2(sample_images)

    print(f"\nFullAnalog output mean: {output_fullanalog.mean().item():.6f}, std: {output_fullanalog.std().item():.6f}")
    print(f"LRTT (after transfer) output mean: {output_lrtt_after.mean().item():.6f}, std: {output_lrtt_after.std().item():.6f}")

    # Compare outputs
    diff = (output_fullanalog - output_lrtt_after).abs()
    print(f"\nOutput difference (FullAnalog vs LRTT after transfer):")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Relative diff: {(diff.mean() / output_fullanalog.abs().mean()).item():.6f}")

    # Check if predictions match
    pred_fullanalog = output_fullanalog.argmax(dim=1)
    pred_lrtt = output_lrtt_after.argmax(dim=1)
    pred_match = (pred_fullanalog == pred_lrtt).float().mean().item() * 100
    print(f"  Prediction match: {pred_match:.2f}%")
    print("="*70 + "\n")

    # Verify transfer
    print("Verifying weight transfer...")
    val_loss_after_transfer, val_acc_after_transfer = evaluate(model_stage2, val_loader, criterion)
    print(f"✓ LRTT model after transfer: Val Accuracy = {val_acc_after_transfer:.2f}%")
    print(f"  (Should match Stage 1 final accuracy: {val_acc:.2f}%)\n")

    wandb.log({
        "stage1_final_accuracy": val_acc,
        "stage2_initial_accuracy": val_acc_after_transfer,
        "transfer_accuracy_match": abs(val_acc - val_acc_after_transfer) < 0.5,
    })

    # Clean up Stage 1 model
    del model_stage1, optimizer_stage1
    torch.cuda.empty_cache() if USE_CUDA else None

    # ========================================================================
    # Stage 2: LRTT training
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: LRTT Training")
    print("="*70 + "\n")

    # Handle N_EPOCHS_STAGE2 = 0 case (skip Stage 2, use transferred weights only)
    if N_EPOCHS_STAGE2 == 0:
        print("N_EPOCHS_STAGE2 = 0: Skipping Stage 2 training (transfer only)")
        val_loss, val_acc = evaluate(model_stage2, val_loader, criterion)
        print(f"Final Val Accuracy (transfer only): {val_acc:.2f}%\n")
    else:
        optimizer_stage2 = AnalogSGD(model_stage2.parameters(), lr=LEARNING_RATE_STAGE2,
                                     momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
        optimizer_stage2.regroup_param_groups(model_stage2)

        for epoch in range(N_EPOCHS_STAGE2):
            # Apply LR schedule for Stage 2
            lr = apply_lr_schedule(
                optimizer_stage2, epoch + 1, N_EPOCHS_STAGE2, LEARNING_RATE_STAGE2,
                schedule=LR_SCHEDULE_STAGE2, warmup_ratio=WARMUP_RATIO_STAGE2,
                milestones=LR_MILESTONES_STAGE2, gamma=LR_GAMMA_STAGE2
            )

            # Train
            train_loss, train_acc = train_one_epoch(model_stage2, train_loader, optimizer_stage2, criterion)

            # Validate
            val_loss, val_acc = evaluate(model_stage2, val_loader, criterion)

            print(f"[Stage2 {epoch+1:02d}/{N_EPOCHS_STAGE2}] "
                  f"Loss={train_loss:.4f} TrainAcc={train_acc:.2f}% "
                  f"ValAcc={val_acc:.2f}% LR={lr:.6f} ({LR_SCHEDULE_STAGE2})")

            # Log LRTT statistics
            if epoch % 5 == 0 or epoch == N_EPOCHS_STAGE2 - 1:
                for name, module in model_stage2.named_modules():
                    if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
                        ctrl = module.analog_module.controller
                        print(f"  {name}: Transfers={ctrl.num_transfers}, "
                              f"A_updates={ctrl.num_a_updates}, B_updates={ctrl.num_b_updates}")

            wandb.log({
                "stage": 2,
                "epoch": N_EPOCHS_STAGE1 + epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": lr,
            })

        print(f"\n✓ Stage 2 Complete: Val Accuracy = {val_acc:.2f}%\n")

    # Final statistics
    print("\n" + "="*70)
    print("Final LRTT Statistics")
    print("="*70)
    for name, module in model_stage2.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            ctrl = module.analog_module.controller
            print(f"  {name}:")
            print(f"    Transfers: {ctrl.num_transfers}")
            print(f"    A updates: {ctrl.num_a_updates}")
            print(f"    B updates: {ctrl.num_b_updates}")
    print("="*70 + "\n")

    wandb.log({
        "final_val_accuracy": val_acc,
    })

    wandb.finish()

    return model_stage2, val_acc


# ==============================================================================
# Main entry point
# ==============================================================================

def main():
    """Main function."""
    print("\n" + "="*70)
    print("CIFAR-10 ResNet18: 2-Stage LRTT Training with Warm-Start")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Stage 1: {N_EPOCHS_STAGE1} epochs (FullAnalog), LR={LEARNING_RATE_STAGE1} ({LR_SCHEDULE_STAGE1})")
    print(f"Stage 2: {N_EPOCHS_STAGE2} epochs (LRTT), LR={LEARNING_RATE_STAGE2} ({LR_SCHEDULE_STAGE2})")
    print(f"LRTT Rank: {LRTT_RANK_CONV}")
    print(f"LoRA Alpha: {LORA_ALPHA}")
    print(f"Transfer Every: {TRANSFER_EVERY} steps")
    print(f"Transfer Robustness: micro_steps={TRANSFER_MICRO_STEPS}, "
          f"mode='{TRANSFER_MODE}', pilot_frac={TRANSFER_PILOT_FRAC:.4f}")
    print(f"Transfer Preprocessing: centering={TRANSFER_CENTERING}, normalize={TRANSFER_NORMALIZE}")
    print(f"Multi-read: num_reads={NUM_READS}, mode='{MULTI_READ_MODE}', read_n_avg={READ_N_AVG}")
    print(f"AGC: enabled={AGC_ENABLED}, margin={AGC_MARGIN}, max_iters={AGC_MAX_ITERS}")
    print(f"Two-Amp: enabled={TWO_AMP_ENABLED}, ratio={TWO_AMP_RATIO}")
    print(f"Update Mode: {UPDATE_MODE}, Reinit Mode: {REINIT_MODE}")
    print(f"Device: 6T1C A/B={USE_6T1C_AB}, One-hot Transfer={USE_ONEHOT_TRANSFER}")
    print("="*70 + "\n")

    t0 = time()
    model, final_acc = train_2stage()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Final Validation Accuracy: {final_acc:.2f}%")
    print(f"Total Time: {(time() - t0) / 60:.2f} min")
    print(f"{'='*70}\n")

    # Save final model
    save_path = os.path.join(RESULTS, "final_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}\n")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example: ViT with SPT+LSA for CIFAR10 using LRTT layers.

Based on the paper settings:
- SPT (Shifted Patch Tokenization) + LSA (Locality Self-Attention)
- 4 Transformer blocks
- ~4.3M trainable parameters, 18 linear layers
- All linear and conv layers are analog, normalization layers are FP
- 40 epochs, batch size 8, no image augmentation
- ReduceLROnPlateau scheduler

Reference: https://github.com/kentaroy47/vision-transformers-cifar10
"""
# pylint: disable=invalid-name

import os
import math
import json

import torch
from torch import nn, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets, transforms

from tqdm import tqdm
import wandb

from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn import AnalogLinear, AnalogConv2d
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice, PCMPresetDevice, ReRamESPresetDevice
from aihwkit.simulator.configs.devices import ConstantStepDevice, SoftBoundsDevice, LinearStepDevice, FloatingPointDevice


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "VITSPTLSA_LRTT_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "cifar10_vitsptlsa_lrtt_scratch_model_weight.pth")

# Training parameters (from paper)
SEED = 1
N_EPOCHS = 40  # Paper: 40 epochs
BATCH_SIZE = 8  # Paper: batch size 8
LEARNING_RATE = 0.0009856  # Initial LR (will be reduced on plateau)
LR_REDUCTION_FACTOR = 0.1  # Paper: reduce LR by 0.1 on plateau
LR_PATIENCE = 5  # Patience for ReduceLROnPlateau
EARLY_STOP_PATIENCE = 7  # Stop if no improvement for N epochs
WEIGHT_DECAY = 0.001003
OPTIMIZER = "AnalogSGD"  # "AnalogSGD", "AnalogAdam"
N_CLASSES = 10
NUM_WORKERS = 4  # WSL에서는 0이 가장 빠름
IMAGE_SIZE = 32  # CIFAR-10 native size (no resize for this model)

# ViT model configuration (SPT+LSA from paper)
# Target: ~4,337,642 trainable parameters, 18 linear layers
PATCH_SIZE = 4  # Smaller patches for CIFAR-10 (32/4 = 8 patches per side)
EMBED_DIM = 288  # Tuned to match ~4.1M parameters (288 is divisible by 8)
DEPTH = 4  # Paper: 4 transformer blocks
NUM_HEADS = 8  # Number of attention heads (288 / 8 = 36 head_dim)
MLP_RATIO = 4.0  # MLP expansion ratio (standard ViT uses 4.0)
DROPOUT = 0.0  # Dropout rate

# LRTT configuration parameters
LRTT_RANK = 8
TRANSFER_EVERY = 826
LORA_ALPHA = 1.708
TRANSFER_LR = LORA_ALPHA
TRANSFER_LR_SCALE = 1.0  # Scaling factor for transfer_lr (effective = transfer_lr * scale)
REINIT_MODE = "decay"  # "standard", "decay", "hybrid", "orthogonal_zero", "orthogonal_decay"
DECAY_FACTOR = 1.0  # Decay factor for reinit (0 < decay_factor <= 1, used with "decay" and "hybrid" modes)

# Paper-aligned analog device parameters
# nstates = 200 -> dw_min = 2.0 / 200 = 0.01 (assuming w_max=1, w_min=-1)
# λA = 0.075 -> This affects the effective learning rate for A/B matrices
N_STATES = 200  # Paper: number of device conductance states
DW_MIN = 2.0 / N_STATES  # Step size for ConstantStepDevice (= 0.01)
USE_REALISTIC_DEVICE = False  # Set True to use ConstantStepDevice (200 states)

# Device configuration for LRTT tiles
# AB_DEVICE: Device for A, B tiles - "6t1c", "idealized", "constantstep", "floating_point"
# C_DEVICE: Device for C tile - "softbounds", "idealized", "pcm", "rram", "floating_point"
AB_DEVICE = "6t1c"  # "6t1c", "idealized", "constantstep"
C_DEVICE = "softbounds"   # "softbounds", "idealized", "pcm", "rram"

# 6T1C Retention (capacitor leakage) parameters
SIXT1C_TAU_SEC = 46505.0       # Physical time constant: 775.1 min = 46505 sec
SIXT1C_DT_BATCH_SEC = 1.0      # Assumed time per mini-batch in seconds
SIXT1C_INCLUDE_RETENTION = True  # Whether to include retention effects

# Layer configuration
# Paper: All linear and conv layers are analog, normalization layers are FP
USE_ANALOG_FOR_ALL_LINEAR = True
USE_ANALOG_FOR_ALL_CONV = True


def _create_6t1c_device():
    """Create 6T1C LinearStepDevice.

    6T1C (6 Transistors, 1 Capacitor) is a volatile analog memory device.
    Parameters are fitted from experimental 6T1C device data.
    """
    import math

    # Calculate lifetime from physical τ for 6T1C retention
    if SIXT1C_INCLUDE_RETENTION and SIXT1C_DT_BATCH_SEC > 0:
        delta = 1 - math.exp(-SIXT1C_DT_BATCH_SEC / SIXT1C_TAU_SEC)
        lifetime = 1.0 / delta
    else:
        lifetime = 0.0

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
        write_noise_std=0, #0.0182
        # LinearStepDevice specific
        mean_bound_reference=True,
        # Retention (capacitor leakage)
        lifetime=lifetime,
        lifetime_dtod=0.1 if SIXT1C_INCLUDE_RETENTION else 0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def _create_ab_device():
    """Create device for A/B tiles based on AB_DEVICE setting."""
    if AB_DEVICE == "6t1c":
        return _create_6t1c_device()
    elif AB_DEVICE == "constantstep":
        return ConstantStepDevice(
            w_max=1.0,
            w_min=-1.0,
            dw_min=DW_MIN,
            dw_min_std=0.0,
            dw_min_dtod=0.0,
            up_down=0.0,
        )
    elif AB_DEVICE == "floating_point":
        return FloatingPointDevice()
    else:  # idealized
        return IdealizedPresetDevice(
            w_max=1.0,
            w_min=-1.0,
            dw_min=0.0002,
            dw_min_dtod=0.3,
            dw_min_std=0.3,
            up_down=0.0,
            up_down_dtod=0.0,
        )


def _create_c_device():
    """Create device for C tile based on C_DEVICE setting."""
    if C_DEVICE == "softbounds":
        # SoftBoundsDevice with NO NOISE (matches sweep_softbounds_lifetime.py)
        # Comments show aihwkit default values
        return SoftBoundsDevice(
            # Weight bounds
            w_max=1.0,               # default: 0.6
            w_min=-1.0,              # default: -0.6
            w_max_dtod=0.0,          # default: 0.3
            w_min_dtod=0.0,          # default: 0.3
            # Update step size
            dw_min=0.001,            # default: 0.001
            dw_min_dtod=0.0,         # default: 0.3
            dw_min_std=0.0,          # default: 0.3
            # Up/down asymmetry
            up_down=0.0,             # default: 0.0
            up_down_dtod=0.0,        # default: 0.01
            # Noise
            mult_noise=True,         # default: True
            write_noise_std=0.0,     # default: 0.0
        )
    elif C_DEVICE == "pcm":
        return PCMPresetDevice()
    elif C_DEVICE == "rram":
        return ReRamESPresetDevice()
    elif C_DEVICE == "floating_point":
        return FloatingPointDevice()
    else:  # idealized
        return IdealizedPresetDevice(
            w_max=1.0,
            w_min=-1.0,
            dw_min=0.0002,
            dw_min_dtod=0.3,
            dw_min_std=0.3,
            up_down=0.0,
            up_down_dtod=0.0,
        )


def create_lrtt_config():
    """Create LRTT configuration for linear/conv layers."""

    print(f"Device config: AB={AB_DEVICE}, C={C_DEVICE}")
    print(f"  LRTT: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, lora_alpha={LORA_ALPHA}")
    print(f"  Reinit: mode={REINIT_MODE}, decay_factor={DECAY_FACTOR}")
    if AB_DEVICE == "6t1c":
        print(f"  6T1C: retention={SIXT1C_INCLUDE_RETENTION}, tau={SIXT1C_TAU_SEC}s")

    # Create A/B and C devices
    ab_device = _create_ab_device()
    c_device = _create_c_device()
    unit_devices = [ab_device, ab_device, c_device]

    # Configure PythonLRTTDevice with explicit parameters
    device_config = PythonLRTTDevice(
        rank=LRTT_RANK,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        transfer_lr_scale=TRANSFER_LR_SCALE,
        forward_inject=False,
        reinit_mode=REINIT_MODE,
        reinit_gain=0.1,  # Default reinit gain
        decay_factor=DECAY_FACTOR,  # Decay factor for "decay" and "hybrid" reinit modes
        unit_cell_devices=unit_devices
    )

    device_config.transfer_lr = TRANSFER_LR

    mapping = MappingParameter(
        weight_scaling_omega=1.0,
        learn_out_scaling=False,
        weight_scaling_lr_compensation=True,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=True,
        max_input_size=1024,
        max_output_size=1024
    )

    forward_io = IOParameters(
        inp_res=0.007937,
        inp_bound=1.0,
        inp_noise=0.0,
        inp_sto_round=False,
        out_res=0.001961,
        out_bound=12.0,
        out_noise=0.06,
        w_noise=0.0,
        w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=False,
        max_bm_factor=1000,
    )

    return PythonLRTTRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io)


class ShiftedPatchTokenization(nn.Module):
    """Shifted Patch Tokenization (SPT) module.

    Applies diagonal shifts to the image before patch embedding to capture
    more local context at patch boundaries.
    """

    def __init__(self, in_channels=3, embed_dim=256, patch_size=4, use_analog=True):
        super().__init__()
        self.patch_size = patch_size

        # 5 shifted versions: original + 4 diagonal shifts
        # Each shift creates additional context
        self.in_channels = in_channels * 5  # 3 * 5 = 15 channels after shifting

        # Patch embedding projection - always digital (first layer, needs full expressivity)
        self.proj = nn.Conv2d(
            self.in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def shift_features(self, x):
        """Apply shifted patch tokenization."""
        B, C, H, W = x.shape

        # Shift amounts (half patch size for diagonal shifts)
        shift = self.patch_size // 2

        # Original image
        x_orig = x

        # Diagonal shifts (pad and crop)
        # Top-left shift
        x_tl = F.pad(x, (shift, 0, shift, 0))[:, :, :H, :W]
        # Top-right shift
        x_tr = F.pad(x, (0, shift, shift, 0))[:, :, :H, shift:]
        # Bottom-left shift
        x_bl = F.pad(x, (shift, 0, 0, shift))[:, :, shift:, :W]
        # Bottom-right shift
        x_br = F.pad(x, (0, shift, 0, shift))[:, :, shift:, shift:]

        # Concatenate all shifted versions
        x = torch.cat([x_orig, x_tl, x_tr, x_bl, x_br], dim=1)

        return x

    def forward(self, x):
        # Apply shifts
        x = self.shift_features(x)
        # Project to embedding dimension
        x = self.proj(x)
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class LocalitySelfAttention(nn.Module):
    """Locality Self-Attention (LSA) module.

    Enhances self-attention with learnable temperature and diagonal masking
    to improve locality bias for small datasets.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, use_analog=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learnable temperature parameter (initialized to sqrt(head_dim))
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * math.sqrt(self.head_dim))

        # QKV projection
        if use_analog:
            self.qkv = AnalogLinear(
                embed_dim, embed_dim * 3,
                bias=True,
                rpu_config=create_lrtt_config()
            )
        else:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # Output projection
        if use_analog:
            self.proj = AnalogLinear(
                embed_dim, embed_dim,
                bias=True,
                rpu_config=create_lrtt_config()
            )
        else:
            self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Diagonal mask for locality (excluding self-attention diagonal)
        self.register_buffer('mask', None)

    def forward(self, x):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention with learnable temperature
        attn = (q @ k.transpose(-2, -1)) / self.temperature

        # Apply diagonal mask (mask out self-attention for better locality)
        if self.mask is None or self.mask.shape[-1] != N:
            # Create mask that excludes diagonal
            mask = torch.eye(N, device=x.device, dtype=torch.bool)
            self.mask = mask

        # Apply mask: set diagonal to large negative value before softmax
        attn = attn.masked_fill(self.mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Handle NaN from all-masked rows (shouldn't happen with our mask)
        attn = torch.nan_to_num(attn)

        # Combine values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class MLP(nn.Module):
    """MLP block with analog linear layers."""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0, use_analog=True):
        super().__init__()

        if use_analog:
            self.fc1 = AnalogLinear(
                in_features, hidden_features,
                bias=True,
                rpu_config=create_lrtt_config()
            )
            self.fc2 = AnalogLinear(
                hidden_features, out_features,
                bias=True,
                rpu_config=create_lrtt_config()
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with LSA and MLP."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.0, use_analog=True):
        super().__init__()

        # Layer norms are always FP (as per paper)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        # LSA attention
        self.attn = LocalitySelfAttention(
            embed_dim, num_heads, dropout,
            use_analog=use_analog
        )

        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            embed_dim, mlp_hidden_dim, embed_dim, dropout,
            use_analog=use_analog
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT_SPT_LSA(nn.Module):
    """Vision Transformer with SPT and LSA for CIFAR-10.

    Architecture:
    - SPT: Shifted Patch Tokenization (1 linear layer in conv form)
    - 4 Transformer blocks with LSA (each has: qkv, proj, fc1, fc2 = 4 linear layers)
    - MLP head (1 linear layer)

    Total: 1 + 4*4 + 1 = 18 linear layers
    """

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=256, depth=4, num_heads=4,
                 mlp_ratio=2.0, dropout=0.0, use_analog=True):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2  # 64 patches for 32x32 with patch_size=4
        self.embed_dim = embed_dim

        # SPT patch embedding (1 conv layer)
        self.patch_embed = ShiftedPatchTokenization(
            in_channels, embed_dim, patch_size,
            use_analog=use_analog
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks (4 blocks, each with 4 linear layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout,
                use_analog=use_analog
            )
            for _ in range(depth)
        ])

        # Final layer norm (FP)
        self.norm = nn.LayerNorm(embed_dim)

        # MLP head - always digital (small layer, needs full expressivity)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding with SPT
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Take class token
        x = self.head(x)

        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_linear_layers(model):
    """Count linear layers (including AnalogLinear and conv layers used as linear)."""
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, AnalogLinear, nn.Conv2d, AnalogConv2d)):
            count += 1
    return count


def create_model():
    """Create ViT-SPT-LSA model with LRTT layers."""

    model = ViT_SPT_LSA(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        num_classes=N_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        use_analog=USE_ANALOG_FOR_ALL_LINEAR
    )

    num_params = count_parameters(model)
    num_linear = count_linear_layers(model)

    print(f"\nCreated ViT-SPT-LSA model:")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Num patches: {(IMAGE_SIZE // PATCH_SIZE) ** 2}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Depth: {DEPTH} transformer blocks")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  MLP ratio: {MLP_RATIO}")
    print(f"  Trainable parameters: {num_params:,} (target: ~4,337,642)")
    print(f"  Linear/Conv layers: {num_linear} (18 total: 16 LRTT + 2 digital [SPT, head])")
    print(f"  Normalization: Digital (FP)")
    print(f"  Device A/B: {AB_DEVICE}")
    print(f"  Device C: {C_DEVICE}")
    print(f"  LRTT rank: {LRTT_RANK}")
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  Transfer LR scale: {TRANSFER_LR_SCALE}")
    print(f"  Reinit mode: {REINIT_MODE}")
    print(f"  Decay factor: {DECAY_FACTOR}\n")

    return model


def load_images():
    """Load CIFAR-10 images without augmentation (as per paper)."""

    # No augmentation as per paper
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)

    train_data = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False
    )
    validation_data = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False
    )

    return train_data, validation_data


def create_optimizer(model, learning_rate, weight_decay):
    """Create analog-aware optimizer."""
    if OPTIMIZER == "AnalogSGD":
        optimizer = AnalogSGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    elif OPTIMIZER == "AnalogAdam":
        optimizer = AnalogAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}. Choose from: AnalogSGD, AnalogAdam")
    optimizer.regroup_param_groups(model)
    return optimizer


def toggle_forward_inject(model, enabled=True):
    """Toggle forward_inject for all LRTT layers."""
    for module in model.modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            module.analog_module.controller.forward_inject_enabled = enabled


def test_evaluation(validation_data, model, criterion):
    """Evaluate model on validation set."""
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    # Store original forward_inject state
    original_forward_inject_state = {}
    for module in model.modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            original_forward_inject_state[module] = module.analog_module.controller.forward_inject_enabled

    # Disable forward_inject during evaluation
    toggle_forward_inject(model, enabled=False)

    pbar = tqdm(validation_data, desc="Validating", leave=False)

    with no_grad():
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch_max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()

            current_acc = 100 * predicted_ok / total_images
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        epoch_loss = total_loss / len(validation_data.dataset)
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

    # Restore original forward_inject state
    for module, original_state in original_forward_inject_state.items():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            module.analog_module.controller.forward_inject_enabled = original_state

    return model, epoch_loss, error, accuracy


def main():
    """Train ViT-SPT-LSA with LRTT on CIFAR-10."""
    manual_seed(SEED)

    # Initialize wandb
    wandb.init(
        project="cifar10_vitsptlsa_lrtt_scratch",
        name=f"vitsptlsa_bs{BATCH_SIZE}_e{N_EPOCHS}_lr{LEARNING_RATE}_r{LRTT_RANK}_t{TRANSFER_EVERY}_alpha{LORA_ALPHA}",
        config={
            "model": "ViT-SPT-LSA",
            "dataset": "CIFAR-10",
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
            "mlp_ratio": MLP_RATIO,
            "dropout": DROPOUT,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lr_reduction_factor": LR_REDUCTION_FACTOR,
            "lr_patience": LR_PATIENCE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER,
            "seed": SEED,
            "lrtt_rank": LRTT_RANK,
            "transfer_every": TRANSFER_EVERY,
            "lora_alpha": LORA_ALPHA,
            "transfer_lr": TRANSFER_LR,
            "transfer_lr_scale": TRANSFER_LR_SCALE,
            "reinit_mode": REINIT_MODE,
            "decay_factor": DECAY_FACTOR,
            # Paper-aligned device parameters
            "n_states": N_STATES,
            "dw_min": DW_MIN,
            "use_realistic_device": USE_REALISTIC_DEVICE,
            "ab_device": AB_DEVICE,
            "c_device": C_DEVICE,
            # 6T1C retention parameters (when AB_DEVICE="6t1c")
            "sixt1c_include_retention": SIXT1C_INCLUDE_RETENTION,
            "sixt1c_tau_sec": SIXT1C_TAU_SEC,
            "sixt1c_dt_batch_sec": SIXT1C_DT_BATCH_SEC,
            "use_analog_linear": USE_ANALOG_FOR_ALL_LINEAR,
            "use_analog_conv": USE_ANALOG_FOR_ALL_CONV,
            "augmentation": False,  # No augmentation as per paper
            "device": str(DEVICE),
            "use_cuda": USE_CUDA,
        }
    )

    # Load data
    train_data, validation_data = load_images()
    print(f"Training samples: {len(train_data.dataset)}")
    print(f"Validation samples: {len(validation_data.dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {len(train_data)}")

    # Create model
    model = create_model()

    if USE_CUDA:
        model = model.to(DEVICE)
    print(f"Model moved to {DEVICE}")

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)

    # ReduceLROnPlateau scheduler (as per paper)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_REDUCTION_FACTOR,
        patience=LR_PATIENCE
    )

    best_accuracy = 0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_history = []  # Track epoch-wise results for plotting

    print(f"\n{'='*60}")
    print(f"Starting training: {N_EPOCHS} epochs (max), batch_size={BATCH_SIZE}")
    print(f"LR schedule: ReduceLROnPlateau (factor={LR_REDUCTION_FACTOR}, patience={LR_PATIENCE})")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}")
    print(f"No image augmentation (as per paper)")
    print(f"{'='*60}\n")

    epoch_pbar = tqdm(range(N_EPOCHS), desc="Overall Progress", position=0)

    global_step = 0

    for epoch in epoch_pbar:
        model.train()

        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        batch_pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, (images, labels) in enumerate(batch_pbar):
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

            current_acc = 100 * epoch_correct / epoch_total
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

            global_step += 1

        # Calculate epoch statistics
        train_loss = epoch_loss / len(train_data.dataset)
        train_acc = 100 * epoch_correct / epoch_total

        # Validation
        model.eval()
        _, val_loss, val_error, val_accuracy = test_evaluation(validation_data, model, criterion)
        model.train()

        # Update LR scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc / 100,
            "eval/loss": val_loss,
            "eval/accuracy": val_accuracy / 100,
            "eval/error": val_error,
            "learning_rate": current_lr,
        }, step=global_step)

        # Save epoch history for plotting
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_error": val_error,
            "learning_rate": current_lr,
        })

        # Track best accuracy and early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        epoch_pbar.set_postfix({
            'Train': f'{train_acc:.2f}%',
            'Val': f'{val_accuracy:.2f}%',
            'Best': f'{best_accuracy:.2f}%',
            'LR': f'{current_lr:.2e}',
            'NoImp': f'{epochs_without_improvement}/{EARLY_STOP_PATIENCE}'
        })

        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch + 1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            tqdm.write(f"Epoch {epoch + 1:3d}: "
                      f"Train Loss {train_loss:.4f} (Acc {train_acc:.2f}%) | "
                      f"Val Loss {val_loss:.4f} (Acc {val_accuracy:.2f}%) | "
                      f"LR {current_lr:.2e}")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")
    print(f"Best validation error: {100 - best_accuracy:.2f}%")
    print(f"Model weights saved to: {WEIGHT_PATH}")
    print(f"{'='*60}")

    # Save epoch history to JSON for plotting
    history_path = os.path.join(RESULTS, "epoch_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            "method": "LRTT",
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch + 1,
            "history": epoch_history
        }, f, indent=2)
    print(f"Epoch history saved to: {history_path}")

    # Final summary comparison with paper
    print(f"\n{'='*60}")
    print("Comparison with paper results:")
    print(f"  Paper FP baseline: 29.3% error")
    print(f"  Paper TTv2 (no noise): 36.1% error")
    print(f"  Paper c-TTv2 (no noise): 35.9% error")
    print(f"  Our result: {100 - best_accuracy:.1f}% error")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    main()

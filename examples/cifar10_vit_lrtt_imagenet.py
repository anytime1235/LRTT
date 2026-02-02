# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example: ViT (Vision Transformer) with CIFAR10 using LRTT layers + ImageNet pretrained weights.

CIFAR10 dataset on a ViT-B/16 network using LRTT (Low-Rank Tensor-Train)
analog layers with ImageNet pretrained weights as initialization.
"""
# pylint: disable=invalid-name

# Imports
import os

# Imports from PyTorch.
import torch
from torch import nn, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms, models

# Progress bar
from tqdm import tqdm

# Logging
import wandb

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset, PythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "VIT_REGULAR_LRTT_IMAGENET")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "cifar10_vit_regular_lrtt_imagenet_model_weight.pth")

# ImageNet pretrained weights loading
USE_IMAGENET_PRETRAINED = True  # Set to True to load ImageNet pretrained weights into C matrices

# Training parameters
SEED = 1
N_EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 0.01  # Lower LR for ViT (Transformer is more sensitive)
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001  # Lower weight decay for ViT
NESTEROV = True
WARMUP_RATIO = 0.1  # Higher warmup ratio for Transformer
N_CLASSES = 10
NUM_WORKERS = 4
IMAGE_SIZE = 224  # ViT requires 224x224 images

# LRTT configuration parameters
LRTT_RANK = 8  # Rank for linear layers
TRANSFER_EVERY = 1000
LORA_ALPHA = 2.0
TRANSFER_LR = LORA_ALPHA

# ReLoRA-style configuration
ENABLE_RELORA = False
RELORA_RESET_EVERY = 500
RELORA_WARMUP_STEPS = 50

# ViT model configuration
VIT_MODEL = 'vit_b_16'  # Options: 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'

# Layer-wise digital/analog configuration for ViT
# ViT-B/16 structure:
# - conv_proj: Patch embedding conv layer (768 dim)
# - encoder.layers[0-11]: 12 Transformer blocks
#   - Each block has:
#     - self_attention: ln_1, qkv (in_proj), out_proj
#     - mlp: ln_2, fc1 (mlp.0), fc2 (mlp.3)
# - heads: Final classification head
LAYER_CONFIG = {
    'conv_proj': 'digital',  # Patch embedding (Conv2d, keep digital for stability)

    # Transformer encoder blocks (12 blocks for ViT-B)
    # Each block: self_attention (qkv + out_proj) + mlp (fc1 + fc2)
    'encoder_block_0': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_1': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_2': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_3': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_4': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_5': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_6': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_7': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_8': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_9': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_10': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},
    'encoder_block_11': {'qkv': 'analog', 'out_proj': 'analog', 'mlp_fc1': 'analog', 'mlp_fc2': 'analog'},

    'heads': 'digital',  # Final classification head
}


def create_lrtt_config():
    """Create LRTT configuration for linear layers."""
    print(f"Using Standard LRTT with rank={LRTT_RANK}")
    device_config = PythonLRTTDevice(
        rank=LRTT_RANK,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        forward_inject=False,
        unit_cell_devices=[
            IdealizedPresetDevice(),
            IdealizedPresetDevice(),
            IdealizedPresetDevice(),
        ]
    )

    device_config.transfer_lr = TRANSFER_LR

    # Add mapping for larger layers
    mapping = MappingParameter(
        weight_scaling_omega=1.0,
        learn_out_scaling=False,
        weight_scaling_lr_compensation=True,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=True,
        max_input_size=1024,  # Larger for ViT hidden dimensions
        max_output_size=4096  # MLP expansion (768 * 4 = 3072)
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


class AnalogMultiheadAttention(nn.Module):
    """Multi-head attention with analog linear layers."""

    def __init__(self, embed_dim, num_heads, dropout=0.0,
                 use_analog_qkv=True, use_analog_out=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection (combined for efficiency)
        qkv_rpu_config = create_lrtt_config() if use_analog_qkv else FloatingPointRPUConfig()
        self.qkv = AnalogLinear(
            embed_dim, embed_dim * 3,
            bias=True,
            rpu_config=qkv_rpu_config
        )

        # Output projection
        out_rpu_config = create_lrtt_config() if use_analog_out else FloatingPointRPUConfig()
        self.out_proj = AnalogLinear(
            embed_dim, embed_dim,
            bias=True,
            rpu_config=out_rpu_config
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Combine values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.proj_dropout(x)

        return x


class AnalogMLP(nn.Module):
    """MLP block with analog linear layers."""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0,
                 use_analog_fc1=True, use_analog_fc2=True):
        super().__init__()

        fc1_rpu_config = create_lrtt_config() if use_analog_fc1 else FloatingPointRPUConfig()
        fc2_rpu_config = create_lrtt_config() if use_analog_fc2 else FloatingPointRPUConfig()

        self.fc1 = AnalogLinear(
            in_features, hidden_features,
            bias=True,
            rpu_config=fc1_rpu_config
        )
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = AnalogLinear(
            hidden_features, out_features,
            bias=True,
            rpu_config=fc2_rpu_config
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class AnalogEncoderBlock(nn.Module):
    """Transformer encoder block with analog layers."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0,
                 use_analog_qkv=True, use_analog_out=True,
                 use_analog_fc1=True, use_analog_fc2=True):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.self_attention = AnalogMultiheadAttention(
            embed_dim, num_heads, dropout,
            use_analog_qkv=use_analog_qkv,
            use_analog_out=use_analog_out
        )

        self.ln_2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = AnalogMLP(
            embed_dim, mlp_hidden_dim, embed_dim, dropout,
            use_analog_fc1=use_analog_fc1,
            use_analog_fc2=use_analog_fc2
        )

    def forward(self, x):
        x = x + self.self_attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AnalogViT(nn.Module):
    """Vision Transformer with analog LRTT layers."""

    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 num_classes=10, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, dropout=0.0, layer_config=None):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding (Conv2d) - use digital for stability
        conv_use_digital = (layer_config['conv_proj'] == 'digital') if layer_config else True
        if conv_use_digital:
            self.conv_proj = nn.Conv2d(
                in_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size
            )
        else:
            from aihwkit.nn import AnalogConv2d
            self.conv_proj = AnalogConv2d(
                in_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size,
                bias=True,
                rpu_config=create_lrtt_config()
            )

        # Class token and positional embedding
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.encoder = nn.ModuleList()
        for i in range(depth):
            block_key = f'encoder_block_{i}'
            if layer_config and block_key in layer_config:
                block_config = layer_config[block_key]
                use_analog_qkv = (block_config['qkv'] == 'analog')
                use_analog_out = (block_config['out_proj'] == 'analog')
                use_analog_fc1 = (block_config['mlp_fc1'] == 'analog')
                use_analog_fc2 = (block_config['mlp_fc2'] == 'analog')
            else:
                # Default: all analog
                use_analog_qkv = True
                use_analog_out = True
                use_analog_fc1 = True
                use_analog_fc2 = True

            self.encoder.append(AnalogEncoderBlock(
                embed_dim, num_heads, mlp_ratio, dropout,
                use_analog_qkv=use_analog_qkv,
                use_analog_out=use_analog_out,
                use_analog_fc1=use_analog_fc1,
                use_analog_fc2=use_analog_fc2
            ))

        # Final layer norm
        self.ln = nn.LayerNorm(embed_dim)

        # Classification head
        heads_use_digital = (layer_config['heads'] == 'digital') if layer_config else True
        if heads_use_digital:
            self.heads = nn.Linear(embed_dim, num_classes)
        else:
            self.heads = AnalogLinear(
                embed_dim, num_classes,
                bias=True,
                rpu_config=create_lrtt_config()
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for class token and positional embedding."""
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.conv_proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add class token
        class_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer encoder
        for block in self.encoder:
            x = block(x)

        # Classification head (use class token)
        x = self.ln(x)
        x = x[:, 0]  # Take class token
        x = self.heads(x)

        return x


def create_model():
    """Create ViT model with LRTT layers.

    Returns:
        nn.Module: ViT model with LRTT
    """
    # ViT-B/16 configuration
    if VIT_MODEL == 'vit_b_16':
        embed_dim = 768
        depth = 12
        num_heads = 12
        patch_size = 16
    elif VIT_MODEL == 'vit_b_32':
        embed_dim = 768
        depth = 12
        num_heads = 12
        patch_size = 32
    elif VIT_MODEL == 'vit_l_16':
        embed_dim = 1024
        depth = 24
        num_heads = 16
        patch_size = 16
    elif VIT_MODEL == 'vit_l_32':
        embed_dim = 1024
        depth = 24
        num_heads = 16
        patch_size = 32
    else:
        raise ValueError(f"Unknown ViT model: {VIT_MODEL}")

    model = AnalogViT(
        image_size=IMAGE_SIZE,
        patch_size=patch_size,
        in_channels=3,
        num_classes=N_CLASSES,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        layer_config=LAYER_CONFIG
    )

    # Print configuration summary
    print(f"\nCreated {VIT_MODEL} with per-block analog/digital layer configuration:")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Depth: {depth} blocks")
    print(f"  Num heads: {num_heads}")
    print(f"  conv_proj: {'Digital' if LAYER_CONFIG['conv_proj'] == 'digital' else 'Analog (LRTT)'}")
    print(f"  Encoder blocks: 12 blocks with per-layer analog/digital config")
    print(f"  heads: {'Digital' if LAYER_CONFIG['heads'] == 'digital' else 'Analog (LRTT)'}")
    print(f"  LRTT rank: {LRTT_RANK}")
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  Transfer LR: {TRANSFER_LR}")
    print(f"  Note: Weights will be initialized from ImageNet pretrained ViT\n")

    return model


def load_pretrained_weights(analog_model):
    """Load ImageNet pretrained weights into analog ViT model.

    Args:
        analog_model: Analog ViT model with LRTT layers

    Returns:
        int: Number of layers with weights transferred
    """
    print(f"\n{'='*70}")
    print(f"Loading ImageNet Pretrained Weights for ViT")
    print(f"{'='*70}")

    # Load standard PyTorch ViT pretrained weights
    if VIT_MODEL == 'vit_b_16':
        pretrained_model = models.vit_b_16(weights='IMAGENET1K_V1')
    elif VIT_MODEL == 'vit_b_32':
        pretrained_model = models.vit_b_32(weights='IMAGENET1K_V1')
    elif VIT_MODEL == 'vit_l_16':
        pretrained_model = models.vit_l_16(weights='IMAGENET1K_V1')
    elif VIT_MODEL == 'vit_l_32':
        pretrained_model = models.vit_l_32(weights='IMAGENET1K_V1')
    else:
        print(f"Unknown ViT model: {VIT_MODEL}")
        return 0

    transferred_count = 0

    # 1. Transfer conv_proj (patch embedding)
    print("  Transferring conv_proj (patch embedding)...")
    try:
        if hasattr(analog_model.conv_proj, 'weight'):
            # Regular Conv2d
            analog_model.conv_proj.weight.data.copy_(pretrained_model.conv_proj.weight.data)
            if analog_model.conv_proj.bias is not None:
                analog_model.conv_proj.bias.data.copy_(pretrained_model.conv_proj.bias.data)
            transferred_count += 1
            print(f"    Transferred conv_proj (digital)")
        elif hasattr(analog_model.conv_proj, 'analog_module'):
            # AnalogConv2d
            weight = pretrained_model.conv_proj.weight.data
            bias = pretrained_model.conv_proj.bias.data if pretrained_model.conv_proj.bias is not None else None
            analog_model.conv_proj.analog_module.set_weights(weight, bias)
            transferred_count += 1
            print(f"    Transferred conv_proj (analog)")
    except Exception as e:
        print(f"    Failed to transfer conv_proj: {e}")

    # 2. Transfer class token
    print("  Transferring class_token...")
    try:
        analog_model.class_token.data.copy_(pretrained_model.class_token.data)
        transferred_count += 1
        print(f"    Transferred class_token")
    except Exception as e:
        print(f"    Failed to transfer class_token: {e}")

    # 3. Transfer positional embedding (may need interpolation for different sizes)
    print("  Transferring pos_embedding...")
    try:
        pretrained_pos = pretrained_model.encoder.pos_embedding.data  # (1, 197, 768)
        analog_pos_shape = analog_model.pos_embedding.shape  # (1, num_patches+1, embed_dim)

        if pretrained_pos.shape == analog_pos_shape:
            analog_model.pos_embedding.data.copy_(pretrained_pos)
            transferred_count += 1
            print(f"    Transferred pos_embedding directly")
        else:
            # Need interpolation (e.g., if image size differs)
            print(f"    Pos embedding shape mismatch: {pretrained_pos.shape} vs {analog_pos_shape}")
            # For now, use pretrained as-is if shapes match
            analog_model.pos_embedding.data.copy_(pretrained_pos)
            transferred_count += 1
            print(f"    Transferred pos_embedding (matched)")
    except Exception as e:
        print(f"    Failed to transfer pos_embedding: {e}")

    # 4. Transfer encoder blocks
    print("  Transferring encoder blocks...")
    for i, (analog_block, pretrained_block) in enumerate(zip(analog_model.encoder, pretrained_model.encoder.layers)):
        block_transferred = 0

        # Transfer LayerNorm parameters
        try:
            analog_block.ln_1.weight.data.copy_(pretrained_block.ln_1.weight.data)
            analog_block.ln_1.bias.data.copy_(pretrained_block.ln_1.bias.data)
            analog_block.ln_2.weight.data.copy_(pretrained_block.ln_2.weight.data)
            analog_block.ln_2.bias.data.copy_(pretrained_block.ln_2.bias.data)
            block_transferred += 2
        except Exception as e:
            print(f"    Block {i}: Failed to transfer LayerNorm: {e}")

        # Transfer attention weights (qkv and out_proj)
        try:
            # In torchvision ViT: self_attention.in_proj_weight/bias for qkv
            # self_attention.out_proj.weight/bias for output
            pretrained_qkv_weight = pretrained_block.self_attention.in_proj_weight.data
            pretrained_qkv_bias = pretrained_block.self_attention.in_proj_bias.data
            pretrained_out_weight = pretrained_block.self_attention.out_proj.weight.data
            pretrained_out_bias = pretrained_block.self_attention.out_proj.bias.data

            # Transfer qkv
            if hasattr(analog_block.self_attention.qkv, 'analog_module'):
                analog_block.self_attention.qkv.analog_module.set_weights(
                    pretrained_qkv_weight, pretrained_qkv_bias
                )
            else:
                analog_block.self_attention.qkv.weight.data.copy_(pretrained_qkv_weight)
                if analog_block.self_attention.qkv.bias is not None:
                    analog_block.self_attention.qkv.bias.data.copy_(pretrained_qkv_bias)
            block_transferred += 1

            # Transfer out_proj
            if hasattr(analog_block.self_attention.out_proj, 'analog_module'):
                analog_block.self_attention.out_proj.analog_module.set_weights(
                    pretrained_out_weight, pretrained_out_bias
                )
            else:
                analog_block.self_attention.out_proj.weight.data.copy_(pretrained_out_weight)
                if analog_block.self_attention.out_proj.bias is not None:
                    analog_block.self_attention.out_proj.bias.data.copy_(pretrained_out_bias)
            block_transferred += 1

        except Exception as e:
            print(f"    Block {i}: Failed to transfer attention: {e}")

        # Transfer MLP weights
        try:
            # In torchvision ViT: mlp.0 = fc1, mlp.3 = fc2
            pretrained_fc1_weight = pretrained_block.mlp[0].weight.data
            pretrained_fc1_bias = pretrained_block.mlp[0].bias.data
            pretrained_fc2_weight = pretrained_block.mlp[3].weight.data
            pretrained_fc2_bias = pretrained_block.mlp[3].bias.data

            # Transfer fc1
            if hasattr(analog_block.mlp.fc1, 'analog_module'):
                analog_block.mlp.fc1.analog_module.set_weights(
                    pretrained_fc1_weight, pretrained_fc1_bias
                )
            else:
                analog_block.mlp.fc1.weight.data.copy_(pretrained_fc1_weight)
                if analog_block.mlp.fc1.bias is not None:
                    analog_block.mlp.fc1.bias.data.copy_(pretrained_fc1_bias)
            block_transferred += 1

            # Transfer fc2
            if hasattr(analog_block.mlp.fc2, 'analog_module'):
                analog_block.mlp.fc2.analog_module.set_weights(
                    pretrained_fc2_weight, pretrained_fc2_bias
                )
            else:
                analog_block.mlp.fc2.weight.data.copy_(pretrained_fc2_weight)
                if analog_block.mlp.fc2.bias is not None:
                    analog_block.mlp.fc2.bias.data.copy_(pretrained_fc2_bias)
            block_transferred += 1

        except Exception as e:
            print(f"    Block {i}: Failed to transfer MLP: {e}")

        transferred_count += block_transferred
        print(f"    Block {i}: Transferred {block_transferred} components")

    # 5. Transfer final LayerNorm
    print("  Transferring final LayerNorm...")
    try:
        analog_model.ln.weight.data.copy_(pretrained_model.encoder.ln.weight.data)
        analog_model.ln.bias.data.copy_(pretrained_model.encoder.ln.bias.data)
        transferred_count += 1
        print(f"    Transferred final LayerNorm")
    except Exception as e:
        print(f"    Failed to transfer final LayerNorm: {e}")

    # 6. Classification head - reinitialize for CIFAR-10 (10 classes vs 1000)
    print("  Reinitializing classification head for CIFAR-10...")
    try:
        import math
        if hasattr(analog_model.heads, 'weight'):
            # Regular Linear
            nn.init.kaiming_uniform_(analog_model.heads.weight, a=math.sqrt(5))
            if analog_model.heads.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(analog_model.heads.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(analog_model.heads.bias, -bound, bound)
        elif hasattr(analog_model.heads, 'analog_module'):
            # AnalogLinear
            in_features = analog_model.heads.in_features
            out_features = analog_model.heads.out_features
            weight = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            analog_model.heads.analog_module.set_weights(weight, None)
        transferred_count += 1
        print(f"    Reinitialized heads for {N_CLASSES} classes")
    except Exception as e:
        print(f"    Failed to reinitialize heads: {e}")

    # 7. Reinitialize A, B matrices for LRTT layers
    print(f"\n  Reinitializing A, B matrices (A=0, B=Kaiming)...")
    ab_reinit_count = 0
    for name, module in analog_model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            controller = module.analog_module.controller
            controller.reinit()
            ab_reinit_count += 1

    print(f"    Reinitialized {ab_reinit_count} LRTT layers (A=0, B=Kaiming)")
    print(f"\n Transfer Summary:")
    print(f"  - Total components transferred: {transferred_count}")
    print(f"  - LRTT reinit: {ab_reinit_count}")
    print(f"{'='*70}\n")

    return transferred_count


def load_images():
    """Load images for train from torchvision datasets with data augmentation.

    Returns:
        Dataset, Dataset: train data and validation data"""
    # Training transforms with augmentation (resize to 224x224 for ViT)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize for ViT
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(IMAGE_SIZE, padding=28),  # Proportional padding
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize for ViT
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=train_transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=val_transform)
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)

    return train_data, validation_data


def create_sgd_optimizer(model, learning_rate, momentum=0.9, weight_decay=5e-4):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate
        momentum (float): momentum factor for SGD
        weight_decay (float): weight decay factor

    Returns:
        Optimizer: created analog optimizer
    """
    optimizer = AnalogSGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=NESTEROV
    )
    optimizer.regroup_param_groups(model)

    return optimizer


def train_step(train_data, model, criterion, optimizer, epoch_num):
    """Train network for one epoch."""
    total_loss = 0
    correct = 0
    total = 0

    model.train()

    desc = f"Epoch {epoch_num}"
    pbar = tqdm(train_data, desc=desc, leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_acc = 100 * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })

    epoch_loss = total_loss / len(train_data.dataset)
    epoch_acc = 100 * correct / total

    return model, optimizer, epoch_loss, epoch_acc


def test_evaluation(validation_data, model, criterion):
    """Test trained network."""
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    # Store original forward_inject state before evaluation
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

    # Restore original forward_inject state after evaluation
    for module, original_state in original_forward_inject_state.items():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            module.analog_module.controller.forward_inject_enabled = original_state

    return model, epoch_loss, error, accuracy


def toggle_forward_inject(model, enabled=True):
    """Toggle forward_inject for all LRTT layers in the model."""
    for module in model.modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            module.analog_module.controller.forward_inject_enabled = enabled


def get_base_cosine_lr(global_step, total_steps, base_lr, warmup_steps, min_lr=1e-5):
    """Get base cosine schedule LR at given step."""
    import math

    if global_step < warmup_steps:
        return base_lr * (global_step / max(1, warmup_steps))

    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_decay


def apply_relora_jagged_lr(optimizer, model, epoch, global_step, total_steps, base_lr,
                            reset_every_steps, restart_warmup_steps, initial_warmup_steps,
                            min_lr=1e-5):
    """Apply ReLoRA-style jagged cosine LR schedule."""
    import math

    base_schedule_lr = get_base_cosine_lr(
        global_step, total_steps, base_lr, initial_warmup_steps, min_lr
    )

    steps_since_reset = global_step % reset_every_steps

    if steps_since_reset < restart_warmup_steps and global_step >= reset_every_steps:
        warmup_progress = steps_since_reset / max(1, restart_warmup_steps)
        jagged_lr = base_schedule_lr * warmup_progress
    else:
        jagged_lr = base_schedule_lr

    from aihwkit.optim.context import AnalogContext

    for param_group in optimizer.param_groups:
        is_lrtt_group = False
        for param in param_group['params']:
            if isinstance(param, AnalogContext):
                if hasattr(param.analog_tile, 'controller'):
                    is_lrtt_group = True
                    break

        if is_lrtt_group:
            param_group['lr'] = jagged_lr
        else:
            param_group['lr'] = base_schedule_lr

    return jagged_lr, base_schedule_lr


def main():
    """Train a PyTorch ViT analog model with LRTT to classify CIFAR10."""
    manual_seed(SEED)

    # Get configuration parameters for run name
    lrtt_config = create_lrtt_config()
    mapping = lrtt_config.mapping
    forward_io = lrtt_config.forward

    inp_res = 1.0/(2**7-2) if forward_io.inp_res == -1 else forward_io.inp_res
    out_res = 1.0/(2**9-2) if forward_io.out_res == -1 else forward_io.out_res

    device_config = lrtt_config.device
    device_types = []

    try:
        if hasattr(device_config, 'unit_cell_devices') and device_config.unit_cell_devices:
            for d in device_config.unit_cell_devices:
                device_name = d.__class__.__name__
                if 'Idealized' in device_name:
                    device_types.append('idealized')
                elif 'FloatingPoint' in device_name:
                    device_types.append('fp')
                elif 'ConstantStep' in device_name:
                    device_types.append('cs')
                else:
                    device_types.append('unknown')
        else:
            device_types = ['idealized', 'idealized', 'idealized']
    except Exception:
        device_types = ['idealized', 'idealized', 'idealized']

    device_type_str = '_'.join(device_types[:3]) if device_types else 'idealized_idealized_idealized'

    cgm = device_config.correct_gradient_magnitudes if hasattr(device_config, 'correct_gradient_magnitudes') else False
    fwd_inject = device_config.forward_inject if hasattr(device_config, 'forward_inject') else True

    # Initialize wandb
    wandb.init(
        project="cifar10_vit_regularlrtt_imagenet",
        name=f"{VIT_MODEL}_cifar10_imagenet_bs{BATCH_SIZE}_e{N_EPOCHS}_wr{WARMUP_RATIO}_mm{device_type_str}_aLR{LEARNING_RATE}_wd{WEIGHT_DECAY}_r{LRTT_RANK}_t{TRANSFER_EVERY}_alpha{LORA_ALPHA}_tlr{TRANSFER_LR}_relora{str(ENABLE_RELORA).lower()}",
        config={
            "model": VIT_MODEL,
            "dataset": "CIFAR-10",
            "pretrained": "imagenet" if USE_IMAGENET_PRETRAINED else "none",
            "image_size": IMAGE_SIZE,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "nesterov": NESTEROV,
            "warmup_ratio": WARMUP_RATIO,
            "seed": SEED,
            "lrtt_rank": LRTT_RANK,
            "transfer_every": TRANSFER_EVERY,
            "lora_alpha": LORA_ALPHA,
            "transfer_lr": TRANSFER_LR,
            "enable_relora": ENABLE_RELORA,
            "relora_reset_every": RELORA_RESET_EVERY,
            "relora_warmup_steps": RELORA_WARMUP_STEPS,
            "device_type_str": device_type_str,
            "correct_gradient_magnitudes": cgm,
            "forward_inject": fwd_inject,
            "forward_inp_res": inp_res,
            "forward_out_res": out_res,
            "forward_inp_noise": forward_io.inp_noise,
            "forward_out_noise": forward_io.out_noise,
            "device": str(DEVICE),
            "use_cuda": USE_CUDA,
            "num_workers": NUM_WORKERS
        }
    )

    # Load the images
    train_data, validation_data = load_images()

    # Make the model
    model = create_model()

    if USE_CUDA:
        model = model.to(DEVICE)

    print(f"Model moved to {DEVICE}")

    # Load ImageNet pretrained weights if requested
    if USE_IMAGENET_PRETRAINED:
        print(f"\n{'='*70}")
        print("Loading ImageNet Pretrained Weights")
        print(f"{'='*70}")
        transferred_count = load_pretrained_weights(model)

        if transferred_count > 0:
            print(f"Successfully loaded ImageNet pretrained weights")
            print(f"  - Transferred weights to {transferred_count} components")
            print("  Training will start from ImageNet pretrained initialization")

            # Evaluate pretrained model immediately after loading
            print(f"\n{'='*70}")
            print("Evaluating loaded pretrained model (before any CIFAR-10 training)...")
            print(f"{'='*70}")

            model.eval()
            with torch.no_grad():
                _, pretrained_loss, pretrained_error, pretrained_acc = test_evaluation(validation_data, model, nn.CrossEntropyLoss())
            print(f"Pretrained model validation accuracy on CIFAR-10: {pretrained_acc:.2f}%")
            print(f"  (ImageNet weights -> CIFAR-10, before any fine-tuning)")
            print(f"{'='*70}\n")
            model.train()
        else:
            print("No pretrained weights loaded - training from random initialization")
        print(f"{'='*70}\n")

    # Count parameters
    pytorch_params = sum(p.numel() for p in model.parameters())
    print(f"PyTorch registered parameters: {pytorch_params:,}")
    print("Note: Analog tile weights are stored internally in C++ and not counted here")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_sgd_optimizer(model, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)

    best_accuracy = 0
    best_epoch = 0

    # Calculate training steps for ReLoRA
    steps_per_epoch = len(train_data)
    total_training_steps = N_EPOCHS * steps_per_epoch
    initial_warmup_steps = int(WARMUP_RATIO * total_training_steps)

    print("\nStarting LRTT training on CIFAR10 with ViT...")
    if ENABLE_RELORA:
        print(f"  ReLoRA enabled: reset every {RELORA_RESET_EVERY} steps, warmup {RELORA_WARMUP_STEPS} steps")
    print("=" * 60)

    # Special case: Save initial model when N_EPOCHS = 0
    if N_EPOCHS == 0:
        save(model.state_dict(), WEIGHT_PATH)
        print(f"\nN_EPOCHS = 0: Initial model saved to {WEIGHT_PATH}")
        print("No training performed.")
        wandb.finish()
        return

    epoch_pbar = tqdm(range(N_EPOCHS), desc="Overall Progress", position=0)

    global_step = 0

    for epoch in epoch_pbar:
        model.train()

        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        batch_pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, (images, labels) in enumerate(batch_pbar):
            # Apply learning rate schedule
            if ENABLE_RELORA:
                jagged_lr, base_lr = apply_relora_jagged_lr(
                    optimizer, model, epoch + 1, global_step, total_training_steps,
                    LEARNING_RATE, RELORA_RESET_EVERY, RELORA_WARMUP_STEPS,
                    initial_warmup_steps
                )
                analog_lrtt_lr = jagged_lr
                digital_lr = base_lr
            else:
                current_lr = get_base_cosine_lr(
                    global_step, total_training_steps, LEARNING_RATE, initial_warmup_steps
                )

                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                analog_lrtt_lr = current_lr
                digital_lr = current_lr

            # Forward and backward pass
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            # Update statistics
            epoch_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

            current_acc = 100 * epoch_correct / epoch_total
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

            # Log learning rates to wandb
            if ENABLE_RELORA:
                wandb.log({
                    "learning_rate/analog_lrtt": analog_lrtt_lr,
                    "learning_rate/digital": digital_lr,
                }, step=global_step, commit=False)
            else:
                wandb.log({
                    "learning_rate": digital_lr,
                }, step=global_step, commit=False)

            global_step += 1

        # Calculate epoch-level statistics
        train_loss = epoch_loss / len(train_data.dataset)
        train_acc = 100 * epoch_correct / epoch_total

        # Run validation after each epoch
        model.eval()
        _, val_loss, val_error, val_accuracy = test_evaluation(validation_data, model, criterion)
        model.train()

        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc / 100,
            "eval/loss": val_loss,
            "eval/accuracy": val_accuracy / 100,
            "eval/error": val_error,
        }

        if ENABLE_RELORA:
            log_dict["learning_rate/analog_lrtt"] = analog_lrtt_lr
            log_dict["learning_rate/digital"] = digital_lr
        else:
            log_dict["learning_rate"] = digital_lr

        wandb.log(log_dict, step=global_step)

        latest_val_acc = val_accuracy
        if latest_val_acc > best_accuracy:
            best_accuracy = latest_val_acc
            best_epoch = epoch
            save(model.state_dict(), WEIGHT_PATH)
        epoch_pbar.set_postfix({
            'Train_Acc': f'{train_acc:.2f}%',
            'Val_Acc': f'{latest_val_acc:.2f}%',
            'Best': f'{best_accuracy:.2f}%'
        })

        if (epoch + 1) % 10 == 0:
            val_info = f", Val Acc {latest_val_acc:.2f}%"
            tqdm.write(f"Epoch {epoch + 1:3d}: "
                      f"Train Loss {train_loss:.4f} (Acc {train_acc:.2f}%)"
                      f"{val_info}")

    print("=" * 60)
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")
    print(f"Model weights saved to: {WEIGHT_PATH}")


if __name__ == "__main__":
    main()

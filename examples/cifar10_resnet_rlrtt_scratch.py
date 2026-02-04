# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 18 with LRTT: ResNet32 CNN with CIFAR10 using LRTT layers.

CIFAR10 dataset on a ResNet inspired network using LRTT (Low-Rank Tensor-Train)
analog layers based on the paper: https://arxiv.org/abs/1512.03385
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

from torchvision import datasets, transforms

# Progress bar
from tqdm import tqdm

# Logging
import wandb

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset, PythonLRTTDevice
from aihwkit.simulator.configs.spatial_lrtt_python import SpatialPythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_REGULAR_LRTT_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "regular_lrtt_scratch_model_weight.pth")

# Baseline model loading (2-stage training)
# Set to None to skip baseline loading (train from scratch)
# Set to path to load pretrained baseline weights into LRTT C matrices
BASELINE_CHECKPOINT_PATH = "results/RESNET_FULLANALOG_SCRATCH_REGULAR/fullanalog_scratch_model_weight_100epoch.pth"  # Example: "results/RESNET_FULLANALOG_SCRATCH/fullanalog_scratch_model_weight_0epoch.pth"
LOAD_BASELINE = True  # Set to True to load baseline weights

# Training parameters
SEED = 1
N_EPOCHS = 200  # Reduced for LRTT demonstration
BATCH_SIZE = 128  # Reduced to prevent CUDA memory issues
LEARNING_RATE = 0.1
MOMENTUM = 0.9  # SGD momentum
WEIGHT_DECAY = 0.0005  # L2 regularization
NESTEROV = True  # Nesterov momentum
WARMUP_RATIO = 0.04  # Warmup ratio (10% of total epochs)
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# LRTT configuration parameters
LRTT_RANK_CONV = 32  # Rank for convolutional layers
LRTT_RANK_FC = 32  # Rank for fully connected layers
TRANSFER_EVERY = 1000  # Transfer A⊗B to C more frequently for better convergence
LORA_ALPHA = 2.0  # LoRA scaling factor
TRANSFER_LR = LORA_ALPHA  # Transfer learning rate (defaults to LORA_ALPHA, can be set independently)

# Spatial LRTT for parameter reduction
USE_SPATIAL_LRTT = False  # Use Regular LRTT (standard parameter count)

# ReLoRA-style configuration
ENABLE_RELORA = False  # Enable ReLoRA-style jagged cosine LR schedule
RELORA_RESET_EVERY = 500  # LR cycle period for jagged cosine (≈10 epochs with batch_size=128)
                            # CIFAR-10: 50000/128 ≈ 390 steps/epoch, so 10 epochs = 3900 steps
                            # Note: A,B weight reinit happens via TRANSFER_EVERY (independent)
RELORA_WARMUP_STEPS = 50  # LR warmup steps after each cycle (must be < RELORA_RESET_EVERY)


# Layer-wise digital/analog configuration
# Set which layers use analog (LRTT) vs digital (FloatingPoint)
# Options: 'analog' (LRTT), 'digital' (FloatingPoint)
#
# ResNet18 structure:
# - conv1: First 7x7/3x3 conv layer
# - layer1: 2 blocks, NO downsample (64 -> 64 channels)
# - layer2: 2 blocks, downsample in block0 (64 -> 128 channels)
# - layer3: 2 blocks, downsample in block0 (128 -> 256 channels)
# - layer4: 2 blocks, downsample in block0 (256 -> 512 channels)
# - fc: Final fully connected layer
LAYER_CONFIG = {
    'conv1': 'digital',           # First convolutional layer

    # Layer1 (2 blocks, no downsample)
    'layer1_block0': {
        'conv1': 'analog',
        'conv2': 'analog',
    },
    'layer1_block1': {
        'conv1': 'analog',
        'conv2': 'analog',
    },

    # Layer2 (2 blocks, downsample in block0)
    'layer2_block0': {
        'conv1': 'analog',
        'conv2': 'analog',
        'downsample': 'analog',
    },
    'layer2_block1': {
        'conv1': 'analog',
        'conv2': 'analog',
    },

    # Layer3 (2 blocks, downsample in block0)
    'layer3_block0': {
        'conv1': 'analog',
        'conv2': 'analog',
        'downsample': 'analog',
    },
    'layer3_block1': {
        'conv1': 'analog',
        'conv2': 'analog',
    },

    # Layer4 (2 blocks, downsample in block0)
    'layer4_block0': {
        'conv1': 'analog',
        'conv2': 'analog',
        'downsample': 'analog',
    },
    'layer4_block1': {
        'conv1': 'analog',
        'conv2': 'analog',
    },

    'fc': 'digital',              # Final fully connected layer
}


def create_lrtt_config_conv():
    """Create LRTT configuration for convolutional layers.
    
    Returns:
        PythonLRTTRPUConfig: LRTT configuration for conv layers
    """
    # Choose device preset:
    # - idealized: IdealizedPresetDevice (no noise, perfect)
    # - constant_step: ConstantStepDevice (realistic analog)
    #device_config = PythonLRTTPreset.idealized(
    #    rank=LRTT_RANK_CONV,
    #    transfer_every=TRANSFER_EVERY,
    #    lora_alpha=LORA_ALPHA
    #)
    
    # Alternative: ConstantStepDevice
    # device_config = PythonLRTTPreset.constant_step(
    #     rank=LRTT_RANK_CONV,
    #     transfer_every=TRANSFER_EVERY,
    #     dw_min=0.01
    # )
    
    # Alternative: Custom devices (FloatingPoint for all tiles)
    # device_config = PythonLRTTDevice(
    #     rank=LRTT_RANK_CONV,
    #     transfer_every=TRANSFER_EVERY,
    #     lora_alpha=LORA_ALPHA,
    #     unit_cell_devices=[
    #         FloatingPointDevice(),  # A 행렬: floating point
    #         FloatingPointDevice(),  # B 행렬: floating point  
    #         FloatingPointDevice(),  # C 행렬: floating point
    #     ]
    # )
    
    # Alternative: A,B=FloatingPoint, C=Idealized with custom parameters
    if USE_SPATIAL_LRTT:
        # Use spatial LRTT for parameter reduction
        print(f"Using Spatial LRTT with rank={LRTT_RANK_CONV} for parameter reduction")
        device_config = SpatialPythonLRTTDevice(
            rank=LRTT_RANK_CONV,
            transfer_every=TRANSFER_EVERY,
            lora_alpha=LORA_ALPHA,
            forward_inject=False,  # Disable forward_inject for conv layers
            correct_gradient_magnitudes=False,
            unit_cell_devices=[
                IdealizedPresetDevice(),  # A 행렬: idealized device
                IdealizedPresetDevice(),  # B 행렬: idealized device
                IdealizedPresetDevice(),  # C 행렬: use all defaults
            ]
        )
    else:
        # Use standard LRTT
        print(f"Using Standard LRTT with rank={LRTT_RANK_CONV}")
        device_config = PythonLRTTDevice(
            rank=LRTT_RANK_CONV,
            transfer_every=TRANSFER_EVERY,
            lora_alpha=LORA_ALPHA,
            forward_inject=False,  # Disable forward_inject for conv layers
            correct_gradient_magnitudes=False,
            unit_cell_devices=[
                IdealizedPresetDevice(),  # A 행렬: idealized device
                IdealizedPresetDevice(),  # B 행렬: idealized device
                IdealizedPresetDevice(),  # C 행렬: use all defaults
            ]
        )
    
    device_config.transfer_lr = TRANSFER_LR

    # Add mapping for larger layers
    mapping = MappingParameter(
        weight_scaling_omega=0.6,
        learn_out_scaling=False,
        weight_scaling_lr_compensation=False,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=False,
        max_input_size=512,
        max_output_size=512
    )
    
    # Optional: Add I/O configuration for forward/backward passes
    forward_io = IOParameters(
        # DAC (input) configuration (PresetIOParameters defaults)
        inp_res=0.00,     # default: 7-bit DAC 1.0/(2**7-2) (≈0.007937)
        inp_bound=1.0,            # default: 1.0 
        inp_noise=0.0,            # default: 0.0 (no input noise)
        inp_sto_round=False,      # default: False (no stochastic rounding)
        
        # ADC (output) configuration (PresetIOParameters defaults) 
        out_res=0.00,     # default: 9-bit ADC 1.0/(2**9-2) (≈0.001961)
        out_bound=12.0,           # default: 12.0 (dynamic range ratio)
        out_noise=0.0,            # default: 0.06 (~1 LSB of ADC)
        
        # Weight noise configuration (PresetIOParameters defaults)
        w_noise=0.0,              # default: 0.0 (no read noise)
        w_noise_type=WeightNoiseType.NONE,  # default: NONE
        
        # Management configuration (PresetIOParameters defaults)
        bound_management=BoundManagementType.ITERATIVE,  # default: ITERATIVE
        noise_management=NoiseManagementType.ABS_MAX,    # default: ABS_MAX
        is_perfect=False,         # default: False
        max_bm_factor=1000,       # default: 1000
    )
    
    return PythonLRTTRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io)
    # return PythonLRTTRPUConfig(device=device_config, mapping=mapping)


def create_lrtt_config_fc():
    """Create LRTT configuration for fully connected layers.
    
    Returns:
        PythonLRTTRPUConfig: LRTT configuration for FC layers
    """
    # Choose device preset:
    # - idealized: IdealizedPresetDevice (no noise, perfect)
    # - constant_step: ConstantStepDevice (realistic analog)
    device_config = PythonLRTTPreset.idealized(
        rank=LRTT_RANK_FC,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        forward_inject=False,
        correct_gradient_magnitudes=False
    )
    
    # Alternative: ConstantStepDevice  
    # device_config = PythonLRTTPreset.constant_step(
    #     rank=LRTT_RANK_FC,
    #     transfer_every=TRANSFER_EVERY,
    #     dw_min=0.01
    # )
    # device_config.forward_inject = True
    # device_config.correct_gradient_magnitudes = True
    
    # Alternative: Custom devices (mixed devices)
    # device_config = PythonLRTTDevice(
    #     rank=LRTT_RANK_FC,
    #     transfer_every=TRANSFER_EVERY,
    #     lora_alpha=LORA_ALPHA,
    #     forward_inject=True,
    #     correct_gradient_magnitudes=True,
    #     unit_cell_devices=[
    #         ConstantStepDevice(dw_min=0.01),  # A: 빠른 업데이트
    #         ConstantStepDevice(dw_min=0.01),  # B: 빠른 업데이트
    #         FloatingPointDevice(),            # C: 정확한 저장
    #     ]
    # )
    
    # Alternative: A,B=FloatingPoint, C=Idealized
    # device_config = PythonLRTTDevice(
    #     rank=LRTT_RANK_FC,
    #     transfer_every=TRANSFER_EVERY,
    #     lora_alpha=LORA_ALPHA,
    #     forward_inject=True,
    #     correct_gradient_magnitudes=True,
    #     unit_cell_devices=[
    #         FloatingPointDevice(),    # A 행렬: 정확한 floating point
    #         FloatingPointDevice(),    # B 행렬: 정확한 floating point
    #         IdealizedPresetDevice(),  # C 행렬: idealized analog
    #     ]
    # )
    
    device_config.transfer_lr = TRANSFER_LR

    # Optional: Add I/O configuration for FC layers (PresetIOParameters defaults)
    # forward_io = IOParameters(
    #     # DAC (input) configuration (PresetIOParameters defaults)
    #     inp_res=1.0/(2**7-2),     # default: 7-bit DAC (≈0.007937)
    #     inp_bound=1.0,            # default: 1.0
    #     inp_noise=0.0,            # default: 0.0 (no input noise)
    #     inp_sto_round=False,      # default: False (no stochastic rounding)
    #     
    #     # ADC (output) configuration (PresetIOParameters defaults)
    #     out_res=1.0/(2**9-2),     # default: 9-bit ADC (≈0.001961)
    #     out_bound=20.0,           # default: 20.0 (dynamic range ratio)
    #     out_noise=0.1,            # default: 0.1 (~1 LSB of ADC)
    #     
    #     # Weight noise configuration (PresetIOParameters defaults)
    #     w_noise=0.0,              # default: 0.0 (no read noise)
    #     w_noise_type=WeightNoiseType.NONE,  # default: NONE
    #     
    #     # Management configuration (PresetIOParameters defaults)  
    #     bound_management=BoundManagementType.ITERATIVE,  # default: ITERATIVE
    #     noise_management=NoiseManagementType.ABS_MAX,    # default: ABS_MAX
    #     is_perfect=False,         # default: False
    # )
    
    # return PythonLRTTRPUConfig(device=device_config, forward=forward_io, backward=forward_io)
    return PythonLRTTRPUConfig(device=device_config)


class ResidualBlockLRTT(nn.Module):
    """Residual block with LRTT analog convolutional layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1,
                 use_analog_conv1=True, use_analog_conv2=True, use_analog_convskip=True):
        super().__init__()

        # Conv1 configuration
        if use_analog_conv1:
            rpu_config_conv1 = create_lrtt_config_conv()
            bias_conv1 = False  # LRTT doesn't support bias
        else:
            rpu_config_conv1 = FloatingPointRPUConfig()
            bias_conv1 = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)

        # Conv2 configuration
        if use_analog_conv2:
            rpu_config_conv2 = create_lrtt_config_conv()
            bias_conv2 = False
        else:
            rpu_config_conv2 = FloatingPointRPUConfig()
            bias_conv2 = False

        # Convskip configuration
        if use_analog_convskip:
            rpu_config_convskip = create_lrtt_config_conv()
            bias_convskip = False
        else:
            rpu_config_convskip = FloatingPointRPUConfig()
            bias_convskip = False

        # Build layers with individual configurations
        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch,
            kernel_size=3, padding=1, stride=stride,
            bias=bias_conv1,
            rpu_config=rpu_config_conv1
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)

        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=bias_conv2,
            rpu_config=rpu_config_conv2
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=bias_convskip,
                rpu_config=rpu_config_convskip
            )
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass"""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks_lrtt(in_ch, hidden_ch, num_layer, first_layer=False,
                                  block_configs=None):
    """Concatenate multiple LRTT residual blocks to form a layer.

    Args:
        in_ch: Input channels
        hidden_ch: Hidden channels
        num_layer: Number of residual blocks
        first_layer: Whether this is the first layer
        block_configs: List of config dicts for each block, each containing:
                      {'conv1': 'analog'/'digital', 'conv2': 'analog'/'digital',
                       'downsample': 'analog'/'digital' (optional)}

    Returns:
       List: list of layer blocks
    """
    if block_configs is None:
        # Default: all analog
        block_configs = [{'conv1': 'analog', 'conv2': 'analog', 'downsample': 'analog'}] * num_layer

    layers = []
    for i in range(num_layer):
        config = block_configs[i]
        use_analog_conv1 = (config['conv1'] == 'analog')
        use_analog_conv2 = (config['conv2'] == 'analog')
        use_analog_downsample = (config.get('downsample', 'analog') == 'analog')

        if i == 0 and not first_layer:
            # First block with downsampling
            layers.append(ResidualBlockLRTT(
                in_ch, hidden_ch, use_conv=True, stride=2,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_downsample
            ))
        else:
            # Other blocks without downsampling
            layers.append(ResidualBlockLRTT(
                hidden_ch, hidden_ch,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_conv1  # Not used, but kept for consistency
            ))
    return layers


def create_model():
    """ResNet18 inspired analog model with LRTT layers.

    Returns:
       nn.Module: created model with LRTT
    """

    block_per_layers = (2, 2, 2, 2)  # ResNet18 structure
    base_channel = 64  # Standard ResNet18 channel size
    channel = (base_channel, 2 * base_channel, 4 * base_channel, 8 * base_channel)  # (64, 128, 256, 512)

    # Input layer - use configuration from LAYER_CONFIG
    input_use_digital = (LAYER_CONFIG['conv1'] == 'digital')
    if input_use_digital:
        input_rpu_config = FloatingPointRPUConfig()
        input_bias = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)
    else:
        input_rpu_config = create_lrtt_config_conv()
        input_bias = False  # LRTT doesn't support bias

    l0 = nn.Sequential(
        AnalogConv2d(
            3, channel[0],
            kernel_size=3, stride=1, padding=1,
            bias=input_bias,
            rpu_config=input_rpu_config
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    # Residual blocks - use per-block configuration from LAYER_CONFIG
    # Layer1 (2 blocks, no downsample)
    l1 = nn.Sequential(
        *concatenate_layer_blocks_lrtt(
            channel[0], channel[0], block_per_layers[0],
            first_layer=True,
            block_configs=[
                LAYER_CONFIG['layer1_block0'],
                LAYER_CONFIG['layer1_block1'],
            ]
        )
    )

    # Layer2 (2 blocks, downsample in block0)
    l2 = nn.Sequential(
        *concatenate_layer_blocks_lrtt(
            channel[0], channel[1], block_per_layers[1],
            block_configs=[
                LAYER_CONFIG['layer2_block0'],
                LAYER_CONFIG['layer2_block1'],
            ]
        )
    )

    # Layer3 (2 blocks, downsample in block0)
    l3 = nn.Sequential(
        *concatenate_layer_blocks_lrtt(
            channel[1], channel[2], block_per_layers[2],
            block_configs=[
                LAYER_CONFIG['layer3_block0'],
                LAYER_CONFIG['layer3_block1'],
            ]
        )
    )

    # Layer4 (2 blocks, downsample in block0)
    l4_conv = nn.Sequential(
        *concatenate_layer_blocks_lrtt(
            channel[2], channel[3], block_per_layers[3],
            block_configs=[
                LAYER_CONFIG['layer4_block0'],
                LAYER_CONFIG['layer4_block1'],
            ]
        )
    )

    # Final classification layer - use configuration from LAYER_CONFIG
    from aihwkit.nn import AnalogLinear
    fc_use_digital = (LAYER_CONFIG['fc'] == 'digital')
    if fc_use_digital:
        fc_rpu_config = FloatingPointRPUConfig()
        fc_bias = True
    else:
        fc_rpu_config = create_lrtt_config_fc()
        fc_bias = False

    l5_fc = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(
            channel[3], N_CLASSES,  # 512 -> 10 for CIFAR-10
            bias=fc_bias,
            rpu_config=fc_rpu_config
        )
    )

    model = nn.Sequential(l0, l1, l2, l3, l4_conv, l5_fc)

    # Print configuration summary
    def format_block_config(block_name):
        """Format block configuration for printing"""
        config = LAYER_CONFIG[block_name]
        parts = []
        for conv_type in ['conv1', 'conv2', 'downsample']:
            if conv_type in config:
                parts.append(f"{conv_type}={'A' if config[conv_type] == 'analog' else 'D'}")
        return f"{block_name}: {', '.join(parts)}"

    print(f"\nCreated ResNet18 with per-block analog/digital layer configuration:")
    print(f"  conv1: {'Digital (FloatingPoint)' if input_use_digital else f'Analog (LRTT, rank={LRTT_RANK_CONV})'}")
    print(f"  Layer1:")
    print(f"    {format_block_config('layer1_block0')}")
    print(f"    {format_block_config('layer1_block1')}")
    print(f"  Layer2:")
    print(f"    {format_block_config('layer2_block0')}")
    print(f"    {format_block_config('layer2_block1')}")
    print(f"  Layer3:")
    print(f"    {format_block_config('layer3_block0')}")
    print(f"    {format_block_config('layer3_block1')}")
    print(f"  Layer4:")
    print(f"    {format_block_config('layer4_block0')}")
    print(f"    {format_block_config('layer4_block1')}")
    print(f"  fc: {'Digital (FloatingPoint)' if fc_use_digital else f'Analog (LRTT, rank={LRTT_RANK_FC})'}")
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  Transfer LR: {TRANSFER_LR}")
    print(f"  Using random initialization (no pretrained weights)\n")

    # Apply Kaiming initialization to ensure consistent initialization
    initialize_resnet_weights(model)

    return model


def initialize_resnet_weights(model):
    """Apply PyTorch ResNet-style kaiming_normal initialization to all layers.

    This ensures the initialization matches standard PyTorch ResNet18 behavior
    for consistent results across different implementations.

    Args:
        model (nn.Module): Model to initialize
    """
    import math

    print("\nApplying ResNet-style Kaiming initialization...")

    for name, module in model.named_modules():
        if isinstance(module, (AnalogConv2d,)):
            # For AnalogConv2d layers, initialize the analog tile weights
            if hasattr(module, 'analog_module'):
                # Get the weight dimensions
                if hasattr(module, 'out_channels') and hasattr(module, 'in_channels'):
                    out_channels = module.out_channels
                    in_channels = module.in_channels
                    kernel_size = module.kernel_size

                    # Create temporary weight tensor with correct shape for initialization
                    if isinstance(kernel_size, tuple):
                        k_h, k_w = kernel_size
                    else:
                        k_h = k_w = kernel_size

                    temp_weight = torch.empty(out_channels, in_channels, k_h, k_w)

                    # Apply kaiming_normal initialization (ResNet default)
                    # mode='fan_out', nonlinearity='relu' for Conv2d in ResNet
                    nn.init.kaiming_normal_(temp_weight, mode='fan_out', nonlinearity='relu')

                    # Set the initialized weights to the analog tile
                    try:
                        # Check if we need to reshape for analog tile format
                        if hasattr(module.analog_module, 'get_weights'):
                            weights, bias = module.analog_module.get_weights()
                            weight_shape = weights.shape

                            # Reshape temp_weight to match analog tile format
                            if len(weight_shape) == 2:
                                # Flattened format [out_ch, in_ch*k*k]
                                reshaped_weight = temp_weight.view(out_channels, in_channels * k_h * k_w)
                                module.analog_module.set_weights(reshaped_weight, bias)
                            else:
                                # Conv format [out_ch, in_ch, k, k]
                                module.analog_module.set_weights(temp_weight, bias)

                            print(f"  Initialized {name}: Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size})")
                    except Exception as e:
                        print(f"  Warning: Could not initialize {name}: {e}")

        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm: constant initialization (weight=1, bias=0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif hasattr(module, '__class__') and 'AnalogLinear' in module.__class__.__name__:
            # For AnalogLinear (FC layer)
            if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'get_weights'):
                try:
                    weights, bias = module.analog_module.get_weights()
                    in_features = module.in_features if hasattr(module, 'in_features') else weights.shape[1]
                    out_features = module.out_features if hasattr(module, 'out_features') else weights.shape[0]

                    # Create temporary weight for initialization
                    temp_weight = torch.empty(out_features, in_features)

                    # Apply kaiming_uniform initialization for Linear layers (PyTorch default)
                    nn.init.kaiming_uniform_(temp_weight, a=math.sqrt(5))

                    # Set to analog tile
                    module.analog_module.set_weights(temp_weight, bias)
                    print(f"  Initialized {name}: Linear({in_features}, {out_features})")
                except Exception as e:
                    print(f"  Warning: Could not initialize {name}: {e}")

    print("Weight initialization completed\n")


def load_baseline_weights_to_lrtt(lrtt_model, baseline_checkpoint_path):
    """Load baseline model weights into LRTT model (ALL non-LoRA parameters).

    This function loads weights from a trained baseline model and initializes:
    1. LRTT C matrices (base weights) - from analog tile weights
    2. BatchNorm parameters (weight, bias, running_mean, running_var)
    3. Digital layer parameters (FloatingPoint conv1, fc)

    LoRA matrices (A, B) are NOT loaded and remain randomly initialized.

    Args:
        lrtt_model: LRTT model with spatial/regular LRTT layers
        baseline_checkpoint_path: Path to baseline model checkpoint (.pth file)

    Returns:
        dict: Statistics of loaded parameters
    """
    print(f"\n{'='*70}")
    print(f"Loading Baseline Weights (C + BatchNorm + Digital)")
    print(f"{'='*70}")
    print(f"Baseline checkpoint: {baseline_checkpoint_path}")

    # Load baseline checkpoint
    if not os.path.exists(baseline_checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {baseline_checkpoint_path}")
        return {'analog_c': 0, 'batchnorm': 0, 'digital': 0}

    baseline_state_dict = torch.load(baseline_checkpoint_path, map_location=DEVICE, weights_only=False)
    print(f"✓ Loaded baseline checkpoint with {len(baseline_state_dict)} keys")

    # Categorize baseline parameters
    analog_weights = {}  # Analog tile weights (for C matrix)
    batchnorm_params = {}  # BatchNorm parameters
    digital_params = {}  # Other digital parameters

    for key, value in baseline_state_dict.items():
        if '.analog_module.tile_c.analog_tile_state' in key:
            # LRTT layer C matrix (from fullanalog with spatial LRTT format)
            layer_name = key.replace('.analog_module.tile_c.analog_tile_state', '')
            if isinstance(value, dict) and 'analog_tile_weights' in value:
                analog_weights[layer_name] = value['analog_tile_weights']
        elif '.analog_module.analog_tile_state' in key:
            # Regular analog tile weights (digital layers: conv1, fc)
            layer_name = key.replace('.analog_module.analog_tile_state', '')
            if isinstance(value, dict) and 'analog_tile_weights' in value:
                analog_weights[layer_name] = value['analog_tile_weights']
        elif '.bn' in key or 'BatchNorm' in key:
            # BatchNorm parameters
            batchnorm_params[key] = value
        else:
            # Other parameters (digital layers, etc.)
            digital_params[key] = value

    print(f"✓ Baseline breakdown:")
    print(f"  - Analog layers: {len(analog_weights)}")
    print(f"  - BatchNorm params: {len(batchnorm_params)}")
    print(f"  - Digital params: {len(digital_params)}")

    # Statistics counters
    analog_c_loaded = 0
    analog_c_skipped = 0
    batchnorm_loaded = 0
    batchnorm_skipped = 0
    digital_loaded = 0
    digital_skipped = 0

    # ========================================================================
    # Step 1: Load Analog C matrices (LRTT only)
    # ========================================================================
    print(f"\n1. Loading Analog C Matrices:")
    print(f"   (Loading ALL layers including first/last - conv1, fc)")
    print(f"-" * 70)

    # No layers to skip - load all layers including first/last
    def should_skip_analog_layer(name):
        # Load all layers including '0.0' (conv1) and '5.2' (fc)
        return False

    for name, module in lrtt_model.named_modules():
        if isinstance(module, AnalogConv2d) or (hasattr(module, '__class__') and 'AnalogLinear' in module.__class__.__name__):
            if hasattr(module, 'analog_module'):
                # Skip first/last layers
                if should_skip_analog_layer(name):
                    print(f"  ⏭️  {name}: Skipped (first/last layer - keep initialized)")
                    continue

                if name not in analog_weights:
                    continue  # Skip if no baseline weight

                baseline_weight = analog_weights[name]

                try:
                    # LRTT layer - load into C matrix only
                    if hasattr(module.analog_module, 'get_lrtt_component_weights'):
                        C, A, B = module.analog_module.get_lrtt_component_weights()
                        if baseline_weight.shape == C.shape:
                            module.analog_module.set_lrtt_component_weights(
                                baseline_weight.to(C.device),  # New C from baseline
                                A,  # Keep A unchanged (LoRA)
                                B   # Keep B unchanged (LoRA)
                            )
                            analog_c_loaded += 1
                            print(f"  ✓  {name}: C matrix loaded (shape={C.shape})")
                        else:
                            print(f"  ⚠️  {name}: Shape mismatch (C={C.shape}, baseline={baseline_weight.shape})")
                            analog_c_skipped += 1

                    # Regular analog layer - direct weight setting
                    elif hasattr(module.analog_module, 'set_weights'):
                        current_weights, current_bias = module.analog_module.get_weights()
                        if baseline_weight.shape == current_weights.shape:
                            module.analog_module.set_weights(baseline_weight.to(current_weights.device), current_bias)
                            analog_c_loaded += 1
                            print(f"  ✓  {name}: Weight loaded (Regular Analog)")
                        else:
                            print(f"  ⚠️  {name}: Shape mismatch")
                            analog_c_skipped += 1

                except Exception as e:
                    print(f"  ❌  {name}: Error - {e}")
                    analog_c_skipped += 1

    # ========================================================================
    # Step 1.5: Reinitialize A, B matrices (must be done AFTER loading C)
    # ========================================================================
    print(f"\n1.5. Reinitializing A, B matrices (A=0, B=Kaiming):")
    print(f"-" * 70)

    ab_reinit_count = 0
    for name, module in lrtt_model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            controller = module.analog_module.controller
            # Force reinit to ensure A=0, B=Kaiming (standard mode)
            controller.reinit()
            ab_reinit_count += 1
            if ab_reinit_count <= 3:  # Print first 3 for verification
                # Verify A is zero
                A_weights = controller.tile_a.get_weights()[0]
                B_weights = controller.tile_b.get_weights()[0]
                print(f"  ✓  {name}: A norm={A_weights.norm().item():.6f}, B norm={B_weights.norm().item():.6f}")

    print(f"  ✓  Reinitialized {ab_reinit_count} LRTT layers (A=0, B=Kaiming)")

    # ========================================================================
    # Step 2: Load BatchNorm parameters
    # ========================================================================
    print(f"\n2. Loading BatchNorm Parameters:")
    print(f"   (Loading ALL BatchNorm layers including first layer - 0.1)")
    print(f"-" * 70)

    # No BatchNorm layers to skip - load all including first layer
    def should_skip_bn_layer(name):
        # Load all BatchNorm layers including '0.1' (first BatchNorm after conv1)
        return False

    for name, module in lrtt_model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            # Skip first layer's BatchNorm
            if should_skip_bn_layer(name):
                print(f"  ⏭️  {name}: Skipped (first layer BatchNorm - keep initialized)")
                continue

            # Try to load all BatchNorm parameters
            for param_name in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                key = f"{name}.{param_name}"
                if key in batchnorm_params:
                    try:
                        target_param = getattr(module, param_name, None)
                        if target_param is not None:
                            if isinstance(target_param, torch.nn.Parameter):
                                # For weight and bias (learnable parameters)
                                target_param.data.copy_(batchnorm_params[key].to(DEVICE))
                            else:
                                # For buffers (running_mean, running_var, num_batches_tracked)
                                target_param.copy_(batchnorm_params[key].to(DEVICE))
                            batchnorm_loaded += 1
                        else:
                            batchnorm_skipped += 1
                    except Exception as e:
                        print(f"  ⚠️  {key}: Error - {e}")
                        batchnorm_skipped += 1

    print(f"  ✓  Loaded {batchnorm_loaded} BatchNorm parameters")
    if batchnorm_skipped > 0:
        print(f"  ⚠️  Skipped {batchnorm_skipped} BatchNorm parameters")

    # ========================================================================
    # Step 3: Load Digital layer parameters (if any)
    # ========================================================================
    print(f"\n3. Loading Digital Layer Parameters:")
    print(f"   (Loading ALL digital layers including first/last)")
    print(f"-" * 70)

    # No digital layers to skip - load all including first/last
    def should_skip_digital_layer(name):
        # Load all digital layers including '0.0' (conv1) and '5.2' (fc)
        return False

    # Digital layers don't have analog_module, so we use standard PyTorch loading
    # Create a filtered state dict with only digital parameters
    lrtt_state_dict = lrtt_model.state_dict()
    digital_state_dict = {}

    for key in lrtt_state_dict.keys():
        # Skip analog_module and BatchNorm (already loaded)
        if 'analog_module' not in key and '.bn' not in key and 'BatchNorm' not in key:
            # Skip first/last layer digital parameters
            if should_skip_digital_layer(key):
                print(f"  ⏭️  {key}: Skipped (first/last layer - keep initialized)")
                continue
            if key in digital_params:
                digital_state_dict[key] = digital_params[key]

    if len(digital_state_dict) > 0:
        try:
            # Use load_state_dict with strict=False to allow partial loading
            missing_keys, unexpected_keys = lrtt_model.load_state_dict(digital_state_dict, strict=False)
            digital_loaded = len(digital_state_dict)
            print(f"  ✓  Loaded {digital_loaded} digital parameters")
            if len(missing_keys) > 0:
                print(f"  ⚠️  Missing keys: {len(missing_keys)}")
        except Exception as e:
            print(f"  ❌  Error loading digital params: {e}")
            digital_skipped = len(digital_state_dict)
    else:
        print(f"  ℹ️  No digital parameters to load")

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"-" * 70)
    print(f"\n✓ Loading Complete:")
    print(f"  Analog C matrices: {analog_c_loaded} loaded, {analog_c_skipped} skipped")
    print(f"  BatchNorm params: {batchnorm_loaded} loaded, {batchnorm_skipped} skipped")
    print(f"  Digital params: {digital_loaded} loaded, {digital_skipped} skipped")
    print(f"\n✓ LoRA matrices (A, B) remain randomly initialized")
    print(f"{'='*70}\n")

    return {
        'analog_c': analog_c_loaded,
        'batchnorm': batchnorm_loaded,
        'digital': digital_loaded
    }


def load_images():
    """Load images for train from torchvision datasets with data augmentation.

    Returns:
        Dataset, Dataset: train data and validation data"""
    # Training transforms with basic augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
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
    """Train network for one epoch.

    Args:
        train_data (DataLoader): Training data loader
        model (nn.Module): Model to be trained
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        epoch_num (int): Current epoch number

    Returns:
        nn.Module, Optimizer, float, float: model, optimizer, epoch loss, epoch accuracy
    """
    total_loss = 0
    correct = 0
    total = 0

    model.train()

    # Create progress bar
    desc = f"Epoch {epoch_num}"
    pbar = tqdm(train_data, desc=desc, leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        
        # Update progress bar
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}', 
            'Acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = total_loss / len(train_data.dataset)
    epoch_acc = 100 * correct / total

    return model, optimizer, epoch_loss, epoch_acc



def test_evaluation(validation_data, model, criterion):
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float: model, test epoch loss, test error, and test accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()
    
    # Store original forward_inject state before evaluation
    original_forward_inject_state = {}
    for module in model.modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            original_forward_inject_state[module] = module.analog_module.controller.forward_inject_enabled
    
    # Disable forward_inject during evaluation (use only C matrix - pretrained weights)
    toggle_forward_inject(model, enabled=False)

    # Create progress bar for validation
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
            
            # Update progress bar
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
    """Toggle forward_inject for all LRTT layers in the model.
    
    Args:
        model (nn.Module): Model with LRTT layers
        enabled (bool): Whether to enable or disable forward_inject
    """
    for module in model.modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            module.analog_module.controller.forward_inject_enabled = enabled


def get_base_cosine_lr(global_step, total_steps, base_lr, warmup_steps, min_lr=1e-5):
    """Get base cosine schedule LR at given step (ReLoRA-style).

    Args:
        global_step: Current training step (0-indexed)
        total_steps: Total training steps
        base_lr: Base learning rate
        warmup_steps: Initial warmup steps
        min_lr: Minimum LR (absolute value, default 1e-5 to match epoch-based schedule)

    Returns:
        float: Learning rate at this step
    """
    import math

    # Initial warmup
    if global_step < warmup_steps:
        return base_lr * (global_step / max(1, warmup_steps))

    # Cosine decay after warmup
    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_decay


def apply_relora_jagged_lr(optimizer, model, epoch, global_step, total_steps, base_lr,
                            reset_every_steps, restart_warmup_steps, initial_warmup_steps,
                            min_lr=1e-5):
    """Apply ReLoRA-style jagged cosine LR schedule to LoRA (A, B).

    Digital layers (BatchNorm, bias, FloatingPoint) use normal cosine schedule.
    Analog LRTT layers (A, B, C) use jagged schedule, but only A, B actually use LR
    since C doesn't receive direct gradients (forward_inject=False).

    At each reset boundary (every reset_every_steps):
    1. LR drops to 0
    2. Warms up from 0 to base cosine schedule value over restart_warmup_steps
    3. Creates "jagged" sawtooth pattern on top of base cosine decay

    Args:
        optimizer: AnalogSGD optimizer
        model: Model with LRTT layers
        epoch: Current epoch (1-indexed)
        global_step: Global training step (0-indexed)
        total_steps: Total training steps
        base_lr: Base learning rate
        reset_every_steps: Reset LoRA every N steps
        restart_warmup_steps: Warmup steps after each reset
        initial_warmup_steps: Initial warmup steps
        min_lr: Minimum LR (absolute value, default 1e-5 to match epoch-based)

    Returns:
        tuple: (jagged_lr for analog, base_lr for digital)
    """
    import math

    # Get base cosine schedule value at this step
    base_schedule_lr = get_base_cosine_lr(
        global_step, total_steps, base_lr, initial_warmup_steps, min_lr
    )

    # Calculate steps since last reset
    steps_since_reset = global_step % reset_every_steps

    # During restart warmup: ramp from 0 to base schedule value
    if steps_since_reset < restart_warmup_steps and global_step >= reset_every_steps:
        warmup_progress = steps_since_reset / max(1, restart_warmup_steps)
        jagged_lr = base_schedule_lr * warmup_progress
    else:
        # Use base schedule value
        jagged_lr = base_schedule_lr

    # CRITICAL FIX: Set param_group['lr'] correctly BEFORE optimizer.step()
    # The optimizer internally calls analog_tile.set_learning_rate(param_group['lr'])
    # during step(), so we must set the correct LR for each param_group.
    #
    # - LRTT analog layers (A, B tiles) → jagged_lr (with warmup restarts)
    # - Digital layers (BatchNorm, bias, first/last FloatingPoint) → base_schedule_lr

    from aihwkit.optim.context import AnalogContext

    for param_group in optimizer.param_groups:
        # Check if this param_group contains LRTT analog parameters
        is_lrtt_group = False
        for param in param_group['params']:
            if isinstance(param, AnalogContext):
                # Check if it's LRTT layer (has controller attribute)
                if hasattr(param.analog_tile, 'controller'):
                    is_lrtt_group = True
                    break

        # Apply appropriate LR
        if is_lrtt_group:
            # LRTT analog layers get jagged LR with warmup restarts
            param_group['lr'] = jagged_lr
        else:
            # Digital layers get base cosine schedule LR (no warmup restarts)
            param_group['lr'] = base_schedule_lr

    # NOTE: No need to manually call analog_module.set_learning_rate()
    # The optimizer.step() will automatically apply param_group['lr'] to analog tiles

    return jagged_lr, base_schedule_lr


def trigger_relora_reset(model):
    """Trigger ReLoRA reset: merge A⊗B → C and reinit A, B.

    This manually triggers LRTT transfer for all layers, which:
    1. Merges A⊗B into C (C ← C + α·(A⊗B))
    2. Automatically reinitializes A and B (via controller.reinit())

    Args:
        model: Model with LRTT layers

    Returns:
        int: Number of layers reset
    """
    print(f"\n{'='*70}")
    print("ReLoRA: Triggering manual transfer (merge + reinit)")
    print(f"{'='*70}")

    reset_count = 0

    for name, module in model.named_modules():
        if hasattr(module, 'analog_module'):
            if hasattr(module.analog_module, 'controller'):
                controller = module.analog_module.controller

                # Manually trigger transfer (merge A⊗B → C)
                # This also calls controller.reinit() automatically
                controller.ab_weight_transfer()

                reset_count += 1
                print(f"  ✓ {name}: Transferred and reinitialized (mode={controller.reinit_mode})")

    print(f"\n✓ Reset {reset_count} LRTT layers")
    print(f"{'='*70}\n")

    return reset_count



def main():
    """Train a PyTorch ResNet analog model with LRTT to classify CIFAR10."""
    # Seed
    manual_seed(SEED)
    
    # Get configuration parameters for run name
    lrtt_config = create_lrtt_config_conv()
    mapping = lrtt_config.mapping
    forward_io = lrtt_config.forward
    
    # Calculate actual resolution values
    inp_res = 1.0/(2**7-2) if forward_io.inp_res == -1 else forward_io.inp_res
    out_res = 1.0/(2**9-2) if forward_io.out_res == -1 else forward_io.out_res
    
    # Get device types from unit_cell_devices
    device_config = lrtt_config.device
    device_types = []
    
    try:
        if hasattr(device_config, 'unit_cell_devices') and device_config.unit_cell_devices:
            for device in device_config.unit_cell_devices:
                device_name = device.__class__.__name__
                if 'Idealized' in device_name:
                    device_types.append('idealized')
                elif 'FloatingPoint' in device_name:
                    device_types.append('fp')
                elif 'ConstantStep' in device_name:
                    device_types.append('cs')
                else:
                    device_types.append('unknown')
        else:
            # Fallback if unit_cell_devices is not accessible
            device_types = ['idealized', 'idealized', 'idealized']  # Default based on code
    except Exception:
        device_types = ['idealized', 'idealized', 'idealized']  # Default fallback
    
    # Create device type string: A_B_C format
    device_type_str = '_'.join(device_types[:3]) if device_types else 'idealized_idealized_idealized'
    
    # Get correct_gradient_magnitudes from device config
    cgm = device_config.correct_gradient_magnitudes if hasattr(device_config, 'correct_gradient_magnitudes') else False
    
    # Get forward_inject from device config
    fwd_inject = device_config.forward_inject if hasattr(device_config, 'forward_inject') else True
    
    # Initialize wandb
    wandb.init(
        project="new_cifar10_resnet18_regularlrtt_warmstart10epoch",
        name=f"resnet18_cifar10_scratch_bs{BATCH_SIZE}_e{N_EPOCHS}_wr{WARMUP_RATIO}_mm{device_type_str}_aLR{LEARNING_RATE}_wd{WEIGHT_DECAY}_fwdIR{inp_res:.6f}_fwdOR{out_res:.6f}_fwdIN{forward_io.inp_noise}_fwdON{forward_io.out_noise}_mapW{mapping.weight_scaling_omega}_mapLOS{str(mapping.learn_out_scaling).lower()}_mapWSLC{str(mapping.weight_scaling_lr_compensation).lower()}_cgm{str(cgm).lower()}_fwdInj{str(fwd_inject).lower()}_r{LRTT_RANK_CONV}_t{TRANSFER_EVERY}_alpha{LORA_ALPHA}_tlr{TRANSFER_LR}_relora{str(ENABLE_RELORA).lower()}_rre{RELORA_RESET_EVERY}_rws{RELORA_WARMUP_STEPS}",
        config={
            # Model and dataset
            "model": "ResNet32-LRTT",
            "dataset": "CIFAR-10",
            "pretrained": "none",

            # Basic training parameters
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "nesterov": NESTEROV,
            "warmup_ratio": WARMUP_RATIO,
            "seed": SEED,

            # LRTT configuration
            "lrtt_rank_conv": LRTT_RANK_CONV,
            "lrtt_rank_fc": LRTT_RANK_FC,
            "transfer_every": TRANSFER_EVERY,
            "lora_alpha": LORA_ALPHA,
            "transfer_lr": TRANSFER_LR,
            "use_spatial_lrtt": USE_SPATIAL_LRTT,

            # ReLoRA configuration
            "enable_relora": ENABLE_RELORA,
            "relora_reset_every": RELORA_RESET_EVERY,
            "relora_warmup_steps": RELORA_WARMUP_STEPS,

            # Layer configuration - per-block analog/digital selection
            "layer_config_conv1": LAYER_CONFIG['conv1'],

            # Layer1 (2 blocks, no downsample)
            "layer_config_layer1_block0_conv1": LAYER_CONFIG['layer1_block0']['conv1'],
            "layer_config_layer1_block0_conv2": LAYER_CONFIG['layer1_block0']['conv2'],
            "layer_config_layer1_block1_conv1": LAYER_CONFIG['layer1_block1']['conv1'],
            "layer_config_layer1_block1_conv2": LAYER_CONFIG['layer1_block1']['conv2'],

            # Layer2 (2 blocks, downsample in block0)
            "layer_config_layer2_block0_conv1": LAYER_CONFIG['layer2_block0']['conv1'],
            "layer_config_layer2_block0_conv2": LAYER_CONFIG['layer2_block0']['conv2'],
            "layer_config_layer2_block0_downsample": LAYER_CONFIG['layer2_block0']['downsample'],
            "layer_config_layer2_block1_conv1": LAYER_CONFIG['layer2_block1']['conv1'],
            "layer_config_layer2_block1_conv2": LAYER_CONFIG['layer2_block1']['conv2'],

            # Layer3 (2 blocks, downsample in block0)
            "layer_config_layer3_block0_conv1": LAYER_CONFIG['layer3_block0']['conv1'],
            "layer_config_layer3_block0_conv2": LAYER_CONFIG['layer3_block0']['conv2'],
            "layer_config_layer3_block0_downsample": LAYER_CONFIG['layer3_block0']['downsample'],
            "layer_config_layer3_block1_conv1": LAYER_CONFIG['layer3_block1']['conv1'],
            "layer_config_layer3_block1_conv2": LAYER_CONFIG['layer3_block1']['conv2'],

            # Layer4 (2 blocks, downsample in block0)
            "layer_config_layer4_block0_conv1": LAYER_CONFIG['layer4_block0']['conv1'],
            "layer_config_layer4_block0_conv2": LAYER_CONFIG['layer4_block0']['conv2'],
            "layer_config_layer4_block0_downsample": LAYER_CONFIG['layer4_block0']['downsample'],
            "layer_config_layer4_block1_conv1": LAYER_CONFIG['layer4_block1']['conv1'],
            "layer_config_layer4_block1_conv2": LAYER_CONFIG['layer4_block1']['conv2'],

            "layer_config_fc": LAYER_CONFIG['fc'],

            # Device configuration from run name
            "device_type_str": device_type_str,
            "correct_gradient_magnitudes": cgm,
            "forward_inject": fwd_inject,

            # Forward I/O parameters
            "forward_inp_res": inp_res,
            "forward_out_res": out_res,
            "forward_inp_noise": forward_io.inp_noise,
            "forward_out_noise": forward_io.out_noise,
            "forward_inp_bound": forward_io.inp_bound,
            "forward_out_bound": forward_io.out_bound,
            "forward_w_noise": forward_io.w_noise,
            "forward_w_noise_type": str(forward_io.w_noise_type),
            "forward_bound_management": str(forward_io.bound_management),
            "forward_noise_management": str(forward_io.noise_management),
            "forward_is_perfect": forward_io.is_perfect,
            "forward_max_bm_factor": forward_io.max_bm_factor,

            # Mapping parameters
            "mapping_weight_scaling_omega": mapping.weight_scaling_omega,
            "mapping_learn_out_scaling": mapping.learn_out_scaling,
            "mapping_weight_scaling_lr_compensation": mapping.weight_scaling_lr_compensation,
            "mapping_digital_bias": mapping.digital_bias,
            "mapping_weight_scaling_columnwise": mapping.weight_scaling_columnwise,
            "mapping_out_scaling_columnwise": mapping.out_scaling_columnwise,
            "mapping_max_input_size": mapping.max_input_size,
            "mapping_max_output_size": mapping.max_output_size,

            # System info
            "device": str(DEVICE),
            "use_cuda": USE_CUDA,
            "num_workers": NUM_WORKERS
        }
    )

    # Load the images.
    train_data, validation_data = load_images()

    # Make the model
    model = create_model()

    if USE_CUDA:
        model = model.to(DEVICE)

    print(f"Model moved to {DEVICE}")

    # Load baseline weights if requested (2-stage training)
    if LOAD_BASELINE and BASELINE_CHECKPOINT_PATH is not None:
        print(f"\n{'='*70}")
        print("2-Stage Training: Loading baseline weights")
        print(f"{'='*70}")
        load_stats = load_baseline_weights_to_lrtt(model, BASELINE_CHECKPOINT_PATH)
        total_loaded = load_stats['analog_c'] + load_stats['batchnorm'] + load_stats['digital']
        if total_loaded > 0:
            print(f"✓ Successfully loaded baseline weights:")
            print(f"  - Analog C matrices: {load_stats['analog_c']}")
            print(f"  - BatchNorm params: {load_stats['batchnorm']}")
            print(f"  - Digital params: {load_stats['digital']}")
            print("  Training will start from pretrained baseline weights")

            # Evaluate baseline immediately after loading to verify
            print(f"\n{'='*70}")
            print("Evaluating loaded baseline (before any training)...")
            print(f"{'='*70}")

            # Check forward_inject status and C matrix
            forward_inject_status = []
            for name, module in model.named_modules():
                if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
                    controller = module.analog_module.controller
                    forward_inject_status.append(controller.forward_inject_enabled)
                    if len(forward_inject_status) == 1:  # Print first one as example
                        print(f"Forward inject status (first LRTT layer): {controller.forward_inject_enabled}")
                        print(f"Reinit mode: {controller.reinit_mode}")

                        # Check C matrix values
                        C_weights = controller.tile_c.get_weights()[0]
                        print(f"C matrix shape: {C_weights.shape}")
                        print(f"C matrix mean: {C_weights.mean().item():.6f}")
                        print(f"C matrix std: {C_weights.std().item():.6f}")
                        print(f"C matrix norm: {C_weights.norm().item():.6f}")

            model.eval()
            with torch.no_grad():
                _, baseline_loss, baseline_error, baseline_acc = test_evaluation(validation_data, model, nn.CrossEntropyLoss())
            print(f"✓ Baseline validation accuracy: {baseline_acc:.2f}%")
            print(f"  (This should match the fullanalog training final accuracy)")
            print(f"{'='*70}\n")
            model.train()
        else:
            print("⚠️  No baseline weights loaded - training from scratch")
        print(f"{'='*70}\n")

    # Count parameters - analog tiles don't register weights as PyTorch parameters
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

    print("\nStarting LRTT training on CIFAR10...")
    if ENABLE_RELORA:
        print(f"  ReLoRA enabled: reset every {RELORA_RESET_EVERY} steps (≈{RELORA_RESET_EVERY // steps_per_epoch} epochs), warmup {RELORA_WARMUP_STEPS} steps")
    print("=" * 60)

    # Special case: Save initial model when N_EPOCHS = 0
    if N_EPOCHS == 0:
        save(model.state_dict(), WEIGHT_PATH)
        print(f"\nN_EPOCHS = 0: Initial model saved to {WEIGHT_PATH}")
        print("No training performed.")
        wandb.finish()
        return

    # Create overall progress bar for epochs
    epoch_pbar = tqdm(range(N_EPOCHS), desc="Overall Progress", position=0)

    global_step = 0

    for epoch in epoch_pbar:
        # Training epoch with batch-level processing
        model.train()

        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        # Create batch progress bar
        batch_pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, (images, labels) in enumerate(batch_pbar):
            # Apply learning rate schedule (at each training step)
            if ENABLE_RELORA:
                # Use ReLoRA jagged cosine LR
                jagged_lr, base_lr = apply_relora_jagged_lr(
                    optimizer, model, epoch + 1, global_step, total_training_steps,
                    LEARNING_RATE, RELORA_RESET_EVERY, RELORA_WARMUP_STEPS,
                    initial_warmup_steps
                )
                # Store both LRs for logging
                analog_lrtt_lr = jagged_lr
                digital_lr = base_lr
            else:
                # Use standard step-based cosine schedule (no jagged warmup restarts)
                current_lr = get_base_cosine_lr(
                    global_step, total_training_steps, LEARNING_RATE, initial_warmup_steps
                )

                # Apply to all param groups
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                # Both use same LR when ReLoRA is disabled
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

            # Update progress bar
            current_acc = 100 * epoch_correct / epoch_total
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

            # Log learning rates to wandb at each step (for jagged LR visualization)
            if ENABLE_RELORA:
                wandb.log({
                    "learning_rate/analog_lrtt": analog_lrtt_lr,
                    "learning_rate/digital": digital_lr,
                }, step=global_step, commit=False)
            else:
                wandb.log({
                    "learning_rate": digital_lr,
                }, step=global_step, commit=False)

            # Increment global_step after processing this batch
            global_step += 1

        # Calculate epoch-level statistics
        train_loss = epoch_loss / len(train_data.dataset)
        train_acc = 100 * epoch_correct / epoch_total

        # Run validation after each epoch
        model.eval()
        _, val_loss, val_error, val_accuracy = test_evaluation(validation_data, model, criterion)
        model.train()
        
        # Log both train and eval metrics together in one call to maintain proper step counting
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc / 100,  # Convert from percentage to ratio (0-1)
            "eval/loss": val_loss,
            "eval/accuracy": val_accuracy / 100,  # Convert from percentage to ratio (0-1)
            "eval/error": val_error,
        }

        # Log learning rates separately for analog and digital layers when ReLoRA is enabled
        if ENABLE_RELORA:
            log_dict["learning_rate/analog_lrtt"] = analog_lrtt_lr  # Jagged LR for A, B
            log_dict["learning_rate/digital"] = digital_lr  # Base LR for BatchNorm, bias, FloatingPoint
        else:
            log_dict["learning_rate"] = digital_lr  # Same LR for all when ReLoRA disabled

        # Log with explicit step (global_step) for correct x-axis in wandb
        wandb.log(log_dict, step=global_step)
        
        # Track best accuracy
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
        
        # Print detailed progress (every 10 epochs to reduce overhead)
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
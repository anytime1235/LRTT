# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example: ResNet18 CNN with CIFAR100 using FullAnalog layers + ImageNet pretrained weights.

CIFAR100 dataset on a ResNet18 network using FullAnalog (C-only training, no LRTT A/B)
analog layers with ImageNet pretrained weights as initialization.
Uses Regular LRTT tile structure but only trains C matrix.
Based on the paper: https://arxiv.org/abs/1512.03385
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
from aihwkit.nn import AnalogConv2d
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset, PythonLRTTDevice
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
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_REGULAR_FULLANALOG_IMAGENET_CIFAR100")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "cifar100_resnet_regular_fullanalog_imagenet_model_weight.pth")

# ImageNet pretrained weights loading
USE_IMAGENET_PRETRAINED = True  # Set to True to load ImageNet pretrained weights into C matrices

# Training parameters
SEED = 1
N_EPOCHS = 300  # Reduced for demonstration
BATCH_SIZE = 128  # Reduced to prevent CUDA memory issues
LEARNING_RATE = 0.03
MOMENTUM = 0.9  # SGD momentum
WEIGHT_DECAY = 0.0005  # L2 regularization
NESTEROV = True  # Nesterov momentum
WARMUP_RATIO = 0.04  # Warmup ratio (4% of total epochs)
N_CLASSES = 100
NUM_WORKERS = 4  # For faster data loading

# Layer-wise digital/analog configuration
# Set which layers use analog (FullAnalog) vs digital (FloatingPoint)
# Options: 'analog' (FullAnalog C-only), 'digital' (FloatingPoint)
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


def create_fullanalog_config_conv():
    """Create FullAnalog configuration for convolutional layers.

    Uses Regular LRTT tile structure but only trains C matrix (no A/B updates).
    """
    from dataclasses import dataclass
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
    from aihwkit.exceptions import TileError

    # Define custom tile class that updates only C matrix
    class LRTTSimulatorTileFullAnalog(LRTTSimulatorTile):
        """Regular LRTT tile for fullanalog training (C-only update)."""

        def _hook_tile_updates(self) -> None:
            """Override parent's hook to update only C matrix for fullanalog training.

            CRITICAL: Optimizer calls tile_c.update() directly (not self.update()!),
            so we must hook tile_c.update() to implement C-only training.
            """
            # Store original update methods
            if hasattr(self, 'tile_a'):
                self.tile_a._orig_update = self.tile_a.update
            if hasattr(self, 'tile_b'):
                self.tile_b._orig_update = self.tile_b.update
            self.tile_c._orig_update = self.tile_c.update

            # Track if we've already handled this batch
            self._update_handled = False

            parent_tile = self  # Capture reference to parent tile

            # Hook tile_c.update() to process spatial blocks and update C only
            def tile_c_update_wrapper(x_input, d_input, bias=False, in_trans=False,
                                     out_trans=False, non_blocking=False):
                if bias:
                    raise TileError("LRTT does not support bias")

                # Prevent double updates
                if parent_tile._update_handled:
                    return None
                parent_tile._update_handled = True

                # Update C tile directly (inputs already in correct format)
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

    # Define custom device config that uses the fullanalog tile
    @dataclass
    class PythonLRTTDeviceFullAnalog(PythonLRTTDevice):
        """Regular LRTT device for fullanalog training."""

        def get_default_tile_module_class(self):
            """Return the fullanalog regular LRTT tile class."""
            return LRTTSimulatorTileFullAnalog

    # CRITICAL: Custom RPU Config to override tile_class
    @dataclass
    class PythonLRTTRPUConfigFullAnalog(PythonLRTTRPUConfig):
        """Custom RPU Config that uses LRTTSimulatorTileFullAnalog."""

        tile_class: type = LRTTSimulatorTileFullAnalog

        def get_default_tile_module_class(self, out_size: int = 0, in_size: int = 0) -> type:
            """Return the fullanalog tile class."""
            return LRTTSimulatorTileFullAnalog

    # Use Regular LRTT FullAnalog device which updates only C matrix
    device_config = PythonLRTTDeviceFullAnalog(
        rank=1,  # Minimum rank (not used in forward)
        transfer_every=10000000000,  # Very large to avoid transfers during training
        lora_alpha=1.0,
        forward_inject=False,  # Only use C matrix in forward pass
        unit_cell_devices=[
            IdealizedPresetDevice(),
            IdealizedPresetDevice(),
            IdealizedPresetDevice(),
        ]
    )

    # Add mapping for larger layers
    mapping = MappingParameter(
        weight_scaling_omega=1.0,
        learn_out_scaling=False,
        weight_scaling_lr_compensation=True,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=True,
        max_input_size=512,
        max_output_size=512
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

    return PythonLRTTRPUConfigFullAnalog(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io)


def create_fullanalog_config_fc():
    """Create FullAnalog configuration for fully connected layers."""
    device_config = PythonLRTTPreset.idealized(
        rank=1,
        transfer_every=10000000000,
        lora_alpha=1.0,
        forward_inject=False,
        correct_gradient_magnitudes=False
    )
    return PythonLRTTRPUConfig(device=device_config)


class ResidualBlockFullAnalog(nn.Module):
    """Residual block with FullAnalog convolutional layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1,
                 use_analog_conv1=True, use_analog_conv2=True, use_analog_convskip=True):
        super().__init__()

        rpu_config_conv1 = create_fullanalog_config_conv() if use_analog_conv1 else FloatingPointRPUConfig()
        rpu_config_conv2 = create_fullanalog_config_conv() if use_analog_conv2 else FloatingPointRPUConfig()
        rpu_config_convskip = create_fullanalog_config_conv() if use_analog_convskip else FloatingPointRPUConfig()

        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch,
            kernel_size=3, padding=1, stride=stride,
            bias=False,
            rpu_config=rpu_config_conv1
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)

        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=False,
            rpu_config=rpu_config_conv2
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=False,
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


def concatenate_layer_blocks_fullanalog(in_ch, hidden_ch, num_layer, first_layer=False,
                                  block_configs=None):
    """Concatenate multiple FullAnalog residual blocks to form a layer.

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
            layers.append(ResidualBlockFullAnalog(
                in_ch, hidden_ch, use_conv=True, stride=2,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_downsample
            ))
        else:
            # Other blocks without downsampling
            layers.append(ResidualBlockFullAnalog(
                hidden_ch, hidden_ch,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_conv1  # Not used, but kept for consistency
            ))
    return layers


def create_model():
    """ResNet18 inspired analog model with FullAnalog layers.

    Returns:
       nn.Module: created model with FullAnalog
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
        input_rpu_config = create_fullanalog_config_conv()
        input_bias = False

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
        *concatenate_layer_blocks_fullanalog(
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
        *concatenate_layer_blocks_fullanalog(
            channel[0], channel[1], block_per_layers[1],
            block_configs=[
                LAYER_CONFIG['layer2_block0'],
                LAYER_CONFIG['layer2_block1'],
            ]
        )
    )

    # Layer3 (2 blocks, downsample in block0)
    l3 = nn.Sequential(
        *concatenate_layer_blocks_fullanalog(
            channel[1], channel[2], block_per_layers[2],
            block_configs=[
                LAYER_CONFIG['layer3_block0'],
                LAYER_CONFIG['layer3_block1'],
            ]
        )
    )

    # Layer4 (2 blocks, downsample in block0)
    l4_conv = nn.Sequential(
        *concatenate_layer_blocks_fullanalog(
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
        fc_rpu_config = create_fullanalog_config_fc()
        fc_bias = False

    l5_fc = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(
            channel[3], N_CLASSES,  # 512 -> 100 for CIFAR-100
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
    print(f"  conv1: {'Digital (FloatingPoint)' if input_use_digital else 'Analog (FullAnalog C-only)'}")
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
    print(f"  fc: {'Digital (FloatingPoint)' if fc_use_digital else 'Analog (FullAnalog C-only)'}")
    print(f"  Note: Weights will be initialized from ImageNet pretrained ResNet18\n")

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


def load_pretrained_weights(analog_model):
    """Load ImageNet pretrained weights into analog model.

    Args:
        analog_model: Analog model with FullAnalog layers

    Returns:
        int: Number of layers with weights transferred
    """
    print(f"\n{'='*70}")
    print(f"Loading ImageNet Pretrained Weights")
    print(f"{'='*70}")

    # Load standard PyTorch ResNet18 pretrained weights
    pretrained_model = models.resnet18(weights='IMAGENET1K_V1')

    def transfer_weights(analog_layer, pretrained_layer):
        """Transfer weights to analog layer (both conv and linear)"""
        try:
            if hasattr(analog_layer, 'set_weights'):
                # For analog layers, set the visible weights (C matrix)
                weight = pretrained_layer.weight.data
                bias = pretrained_layer.bias.data if pretrained_layer.bias is not None else None
                analog_layer.set_weights(weight, bias)
                return True
            elif hasattr(analog_layer, 'weight') and analog_layer.weight is not None:
                # For regular layers with PyTorch parameters, direct copy
                analog_layer.weight.data.copy_(pretrained_layer.weight.data)
                if hasattr(analog_layer, 'bias') and analog_layer.bias is not None:
                    if pretrained_layer.bias is not None:
                        analog_layer.bias.data.copy_(pretrained_layer.bias.data)
                return True
            elif hasattr(analog_layer, 'analog_module'):
                # For FloatingPointTile or other analog tiles without PyTorch parameters
                weight = pretrained_layer.weight.data
                bias = pretrained_layer.bias.data if pretrained_layer.bias is not None else None
                analog_layer.analog_module.set_weights(weight, bias)
                return True
            else:
                return False
        except Exception:
            return False

    def find_analog_conv_layers(module):
        """Recursively find AnalogConv2d layers"""
        analog_layers = []
        for child in module.children():
            if isinstance(child, (AnalogConv2d,)):
                analog_layers.append(child)
            else:
                analog_layers.extend(find_analog_conv_layers(child))
        return analog_layers

    def find_analog_linear_layers(module):
        """Recursively find AnalogLinear layers"""
        from aihwkit.nn import AnalogLinear
        analog_layers = []
        for child in module.children():
            if isinstance(child, AnalogLinear):
                analog_layers.append(child)
            else:
                analog_layers.extend(find_analog_linear_layers(child))
        return analog_layers

    # Get pretrained layers
    pretrained_conv_layers = []
    pretrained_linear_layers = []

    # Extract conv layers from ResNet18
    def extract_conv_layers(module):
        layers = []
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                layers.append(child)
            else:
                layers.extend(extract_conv_layers(child))
        return layers

    def extract_linear_layers(module):
        layers = []
        for child in module.children():
            if isinstance(child, nn.Linear):
                layers.append(child)
            else:
                layers.extend(extract_linear_layers(child))
        return layers

    pretrained_conv_layers = extract_conv_layers(pretrained_model)
    pretrained_linear_layers = extract_linear_layers(pretrained_model)

    # Get analog layers
    analog_conv_layers = find_analog_conv_layers(analog_model)
    analog_linear_layers = find_analog_linear_layers(analog_model)

    transferred_count = 0

    # Transfer conv layers
    min_conv_layers = min(len(analog_conv_layers), len(pretrained_conv_layers))
    print(f"  Transferring {min_conv_layers} conv layers...")
    for i in range(min_conv_layers):
        analog_layer = analog_conv_layers[i]
        pretrained_layer = pretrained_conv_layers[i]

        # Check and reshape weights for compatibility
        try:
            pretrained_weight = pretrained_layer.weight.data  # [out_ch, in_ch, k, k]
            pretrained_shape = pretrained_weight.shape

            # Handle both FullAnalog (LRTT-based) and regular analog layers
            if hasattr(analog_layer, 'analog_module'):
                # For Regular LRTT: d_size = out_ch, x_size = in_ch*k*k
                if hasattr(analog_layer.analog_module, 'd_size'):  # Regular LRTT
                    analog_out_size = analog_layer.analog_module.d_size
                    analog_in_size = analog_layer.analog_module.x_size
                else:
                    # Not LRTT, treat as regular analog layer (FloatingPointTile)
                    try:
                        # For FloatingPointTile, get weights from the tile
                        weights_result = analog_layer.analog_module.get_weights()
                        if isinstance(weights_result, tuple):
                            weights = weights_result[0]  # Get weight tensor
                        else:
                            weights = weights_result  # Might return tensor directly
                        analog_weight_shape = weights.shape
                        analog_out_size, analog_in_size = analog_weight_shape[0], analog_weight_shape[1] * analog_weight_shape[2] * analog_weight_shape[3]
                    except Exception:
                        # Try alternative approach - use the expected shape from layer definition
                        try:
                            # Get expected dimensions from conv layer definition
                            out_features = getattr(analog_layer, 'out_channels', None)
                            in_features = getattr(analog_layer, 'in_channels', None)
                            kernel_size = getattr(analog_layer, 'kernel_size', (3, 3))
                            if isinstance(kernel_size, int):
                                kernel_size = (kernel_size, kernel_size)

                            if out_features and in_features:
                                analog_out_size = out_features
                                analog_in_size = in_features * kernel_size[0] * kernel_size[1]
                            else:
                                continue
                        except Exception:
                            continue
            elif hasattr(analog_layer, 'weight') and analog_layer.weight is not None:
                # Regular analog layer (FloatingPoint)
                analog_weight_shape = analog_layer.weight.shape
                analog_out_size, analog_in_size = analog_weight_shape[0], analog_weight_shape[1] * analog_weight_shape[2] * analog_weight_shape[3]
            else:
                continue  # Skip if not analog layer

            # Reshape conv weight [out_ch, in_ch, k, k] -> [out_ch, in_ch*k*k] for regular LRTT
            pretrained_shape = pretrained_weight.shape
            out_ch, in_ch, k_h, k_w = pretrained_shape

            # Check if this is regular LRTT or regular analog layer
            is_regular_lrtt = (analog_out_size == out_ch and analog_in_size == in_ch * k_h * k_w)
            # Regular analog layer: has analog_module but not LRTT (e.g., FloatingPointTile)
            is_regular_analog = (hasattr(analog_layer, 'analog_module') and
                                not hasattr(analog_layer.analog_module, 'd_size'))

            reshaped_weight = None

            if is_regular_lrtt:
                # Regular LRTT: [out_ch, in_ch, k, k] -> [out_ch, in_ch*k*k]
                reshaped_weight = pretrained_weight.view(out_ch, in_ch * k_h * k_w)
                transfer_type = "Regular LRTT"

            elif is_regular_analog:
                # Regular analog layer (FloatingPoint) - keep original conv weight format
                try:
                    # For regular analog layer (FloatingPoint) - handle different weight formats
                    if hasattr(analog_layer, 'weight') and analog_layer.weight is not None:
                        # Has PyTorch parameter - keep original conv weight format
                        target_shape = analog_layer.weight.shape  # Should be [out_ch, in_ch, k, k]
                        if pretrained_shape == target_shape:
                            reshaped_weight = pretrained_weight
                            transfer_type = "Direct copy"
                        elif k_h > target_shape[2]:  # Need to crop kernel (e.g., 7x7 -> 3x3)
                            start = (k_h - target_shape[2]) // 2
                            end = start + target_shape[2]
                            reshaped_weight = pretrained_weight[:, :, start:end, start:end]
                            transfer_type = f"Cropped {k_h}x{k_w} -> {target_shape[2]}x{target_shape[3]}"
                        else:
                            continue
                    else:
                        # FloatingPointTile stores weight as flattened [out_ch, in_ch*k*k]
                        # Reshape pretrained weight to match: [out_ch, in_ch, k, k] -> [out_ch, in_ch*k*k]
                        if k_h == 7 and analog_out_size == out_ch and analog_in_size == in_ch * 3 * 3:
                            # ImageNet ResNet first layer: 7x7 -> 3x3 (CIFAR-100 typical)
                            start = (k_h - 3) // 2  # Center crop 7x7 -> 3x3
                            end = start + 3
                            cropped_weight = pretrained_weight[:, :, start:end, start:end]  # [64, 3, 3, 3]
                            reshaped_weight = cropped_weight.reshape(out_ch, in_ch * 3 * 3)  # [64, 27]
                            transfer_type = f"Cropped and flattened {k_h}x{k_w} -> 3x3 -> [{out_ch}, {in_ch * 3 * 3}]"
                        elif analog_out_size == out_ch and analog_in_size == in_ch * k_h * k_w:
                            # Standard case: flatten to match analog tile format
                            reshaped_weight = pretrained_weight.view(out_ch, in_ch * k_h * k_w)
                            transfer_type = f"Flattened [{out_ch}, {in_ch}, {k_h}, {k_w}] -> [{out_ch}, {in_ch * k_h * k_w}]"
                        else:
                            continue
                except Exception:
                    continue

            if reshaped_weight is not None:
                # Create modified pretrained layer with reshaped weight
                class MockLayer:
                    def __init__(self, weight):
                        self.weight = torch.nn.Parameter(weight)
                        self.bias = None

                mock_layer = MockLayer(reshaped_weight)

                if transfer_weights(analog_layer, mock_layer):
                    transferred_count += 1

        except Exception:
            continue

    # Transfer linear layers - adjust for CIFAR-100 (100 classes vs ImageNet 1000)
    if len(analog_linear_layers) > 0 and len(pretrained_linear_layers) > 0:
        print(f"  Transferring linear layers...")
        analog_fc = analog_linear_layers[0]  # Final FC layer
        pretrained_fc = pretrained_linear_layers[0]  # ImageNet FC layer

        # For CIFAR-100, we only transfer the input projection weights (512 dim)
        # but not the output weights (1000 -> 10 classes)
        analog_in_features = getattr(analog_fc, 'in_features', None)
        if hasattr(analog_fc, 'analog_module'):
            analog_in_features = analog_fc.analog_module.in_size

        if analog_in_features == pretrained_fc.in_features:
            # Create a new weight tensor with proper output size
            analog_out_features = getattr(analog_fc, 'out_features', None)
            if hasattr(analog_fc, 'analog_module'):
                analog_out_features = analog_fc.analog_module.out_size

            analog_weight_shape = (analog_out_features, analog_in_features)  # [100, 512] for CIFAR-100
            pretrained_weight = pretrained_fc.weight.data  # [1000, 512] for ImageNet

            # Use Kaiming initialization for CIFAR-100 classifier (PyTorch nn.Linear default)
            # Don't copy ImageNet classes since they're unrelated to CIFAR-100
            import torch.nn.init as init
            import math
            new_weight = torch.empty(analog_weight_shape)
            init.kaiming_uniform_(new_weight, a=math.sqrt(5))

            try:
                if hasattr(analog_fc, 'set_weights'):
                    analog_fc.set_weights(new_weight, None)  # No bias for simplicity
                else:
                    analog_fc.weight.data.copy_(new_weight)
                transferred_count += 1
            except Exception:
                pass

    # Transfer BatchNorm parameters and statistics
    print(f"\n  Transferring BatchNorm parameters and statistics...")

    # Extract all BatchNorm layers from both models (in order)
    analog_bn_layers = []
    pretrained_bn_layers = []

    for module in analog_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            analog_bn_layers.append(module)

    for module in pretrained_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            pretrained_bn_layers.append(module)

    bn_transferred = 0
    min_bn_layers = min(len(analog_bn_layers), len(pretrained_bn_layers))

    for i in range(min_bn_layers):
        analog_bn = analog_bn_layers[i]
        pretrained_bn = pretrained_bn_layers[i]

        try:
            # Transfer learnable parameters (γ, β)
            analog_bn.weight.data.copy_(pretrained_bn.weight.data)
            analog_bn.bias.data.copy_(pretrained_bn.bias.data)

            # Transfer running statistics (for inference and initial training)
            analog_bn.running_mean.copy_(pretrained_bn.running_mean)
            analog_bn.running_var.copy_(pretrained_bn.running_var)
            analog_bn.num_batches_tracked.copy_(pretrained_bn.num_batches_tracked)

            bn_transferred += 1
        except Exception as e:
            print(f"  ⚠️  Failed to transfer BatchNorm layer {i}: {e}")
            continue

    print(f"  ✓  Transferred {bn_transferred} BatchNorm layers (weights + running statistics)")

    print(f"\n✓ Transfer Summary:")
    print(f"  - Conv/Linear layers: {transferred_count}")
    print(f"  - BatchNorm layers: {bn_transferred}")
    print(f"{'='*70}\n")

    return transferred_count


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
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    train_set = datasets.CIFAR100(PATH_DATASET, download=True, train=True, transform=train_transform)
    val_set = datasets.CIFAR100(PATH_DATASET, download=True, train=False, transform=val_transform)
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

    return model, epoch_loss, error, accuracy


def get_base_cosine_lr(global_step, total_steps, base_lr, warmup_steps, min_lr=1e-5):
    """Get base cosine schedule LR at given step.

    Args:
        global_step: Current training step (0-indexed)
        total_steps: Total training steps
        base_lr: Base learning rate
        warmup_steps: Initial warmup steps
        min_lr: Minimum LR (absolute value, default 1e-5)

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


def main():
    """Train a PyTorch ResNet analog model with FullAnalog to classify CIFAR100."""
    # Seed
    manual_seed(SEED)

    # Get configuration parameters for run name
    fullanalog_config = create_fullanalog_config_conv()
    mapping = fullanalog_config.mapping
    forward_io = fullanalog_config.forward

    # Calculate actual resolution values
    inp_res = 1.0/(2**7-2) if forward_io.inp_res == -1 else forward_io.inp_res
    out_res = 1.0/(2**9-2) if forward_io.out_res == -1 else forward_io.out_res

    # Initialize wandb
    wandb.init(
        project="new_cifar100_resnet18_regularfullanalog_imagenet",
        name=f"resnet18_cifar100_imagenet_fullanalog_bs{BATCH_SIZE}_e{N_EPOCHS}_wr{WARMUP_RATIO}_mm_idealized_aLR{LEARNING_RATE}_wd{WEIGHT_DECAY}_fwdIR{inp_res:.6f}_fwdOR{out_res:.6f}_fwdIN{forward_io.inp_noise}_fwdON{forward_io.out_noise}_mapW{mapping.weight_scaling_omega}_mapLOS{str(mapping.learn_out_scaling).lower()}_mapWSLC{str(mapping.weight_scaling_lr_compensation).lower()}",
        config={
            # Model and dataset
            "model": "ResNet18-FullAnalog",
            "dataset": "CIFAR-100",
            "pretrained": "imagenet" if USE_IMAGENET_PRETRAINED else "none",

            # Basic training parameters
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "nesterov": NESTEROV,
            "warmup_ratio": WARMUP_RATIO,
            "seed": SEED,

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

    # Load ImageNet pretrained weights if requested
    if USE_IMAGENET_PRETRAINED:
        print(f"\n{'='*70}")
        print("Loading ImageNet Pretrained Weights")
        print(f"{'='*70}")
        transferred_count = load_pretrained_weights(model)

        if transferred_count > 0:
            print(f"✓ Successfully loaded ImageNet pretrained weights")
            print(f"  - Transferred weights to {transferred_count} layers")
            print("  Training will start from ImageNet pretrained initialization")

            # Evaluate pretrained model immediately after loading to verify
            print(f"\n{'='*70}")
            print("Evaluating loaded pretrained model (before any CIFAR-100 training)...")
            print(f"{'='*70}")

            model.eval()
            with torch.no_grad():
                _, pretrained_loss, pretrained_error, pretrained_acc = test_evaluation(validation_data, model, nn.CrossEntropyLoss())
            print(f"✓ Pretrained model validation accuracy on CIFAR-100: {pretrained_acc:.2f}%")
            print(f"  (ImageNet weights → CIFAR-100, before any fine-tuning)")
            print(f"{'='*70}\n")
            model.train()
        else:
            print("⚠️  No pretrained weights loaded - training from random initialization")
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

    # Calculate training steps
    steps_per_epoch = len(train_data)
    total_training_steps = N_EPOCHS * steps_per_epoch
    initial_warmup_steps = int(WARMUP_RATIO * total_training_steps)

    print("\nStarting FullAnalog training on CIFAR100 with ImageNet pretrained weights...")
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
            current_lr = get_base_cosine_lr(
                global_step, total_training_steps, LEARNING_RATE, initial_warmup_steps
            )

            # Apply to all param groups
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

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

            # Log learning rates to wandb at each step
            wandb.log({
                "learning_rate": current_lr,
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
            "learning_rate": current_lr,
        }

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

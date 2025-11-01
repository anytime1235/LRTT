# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit baseline: ResNet18 CNN with CIFAR10 using mixed digital/analog layers.

CIFAR10 dataset on a ResNet18 network with configurable digital (FloatingPoint)
and analog (base trainable) layers. No LRTT - just standard analog tiles.
Based on cifar10_resnet_lrtt_scratch.py but with LRTT removed.
"""
# pylint: disable=invalid-name

# Imports
import os
from datetime import datetime

# Imports from PyTorch.
import torch
from torch import nn, Tensor, device, no_grad, manual_seed, save
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
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.presets.devices import IdealizedPresetDevice, EcRamPresetDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice, ConstantStepDevice


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_FULLANALOG_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "fullanalog_scratch_model_weight_50epoch.pth")

# Training parameters
SEED = 1
N_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9  # SGD momentum
WEIGHT_DECAY = 0.0005  # L2 regularization
NESTEROV = True  # Nesterov momentum
WARMUP_RATIO = 0.04  # No warmup
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# Analog device type: 'idealized', 'constant_step', 'floating_point', or 'ecram'
ANALOG_DEVICE_TYPE = 'idealized'

# Layer-wise digital/analog configuration
# Set which layers use analog vs digital (FloatingPoint)
# Options: 'analog' (trainable base), 'digital' (FloatingPoint)
#
# ResNet18 structure:
# - conv1: First 3x3 conv layer
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


def create_analog_config():
    """Create analog configuration for trainable base layers (no LRTT).

    Returns:
        SingleRPUConfig: Analog configuration with trainable base weights
    """
    # Choose device type
    if ANALOG_DEVICE_TYPE == 'idealized':
        device_config = IdealizedPresetDevice()
    elif ANALOG_DEVICE_TYPE == 'constant_step':
        device_config = ConstantStepDevice(dw_min=0.01)
    elif ANALOG_DEVICE_TYPE == 'floating_point':
        device_config = FloatingPointDevice()
    elif ANALOG_DEVICE_TYPE == 'ecram':
        device_config = EcRamPresetDevice()
    else:
        # Default to idealized
        device_config = IdealizedPresetDevice()

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
        # DAC (input) configuration
        inp_res=0.007937,     # 7-bit DAC
        inp_bound=1.0,
        inp_noise=0.0,
        inp_sto_round=False,

        # ADC (output) configuration
        out_res=0.001961,     # 9-bit ADC
        out_bound=12.0,
        out_noise=0.06,

        # Weight noise configuration
        w_noise=0.0,
        w_noise_type=WeightNoiseType.NONE,

        # Management configuration
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=False,
        max_bm_factor=1000,
    )

    return SingleRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io)


class ResidualBlockBaseline(nn.Module):
    """Residual block with configurable digital/analog convolutional layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1,
                 use_analog_conv1=False, use_analog_conv2=False, use_analog_convskip=False):
        super().__init__()

        # Conv1 configuration
        if use_analog_conv1:
            rpu_config_conv1 = create_analog_config()
            bias_conv1 = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)
        else:
            rpu_config_conv1 = FloatingPointRPUConfig()
            bias_conv1 = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)

        # Conv2 configuration
        if use_analog_conv2:
            rpu_config_conv2 = create_analog_config()
            bias_conv2 = False
        else:
            rpu_config_conv2 = FloatingPointRPUConfig()
            bias_conv2 = False

        # Convskip configuration
        if use_analog_convskip:
            rpu_config_convskip = create_analog_config()
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


def concatenate_layer_blocks_baseline(in_ch, hidden_ch, num_layer, first_layer=False,
                                  block_configs=None):
    """Concatenate multiple residual blocks to form a layer.

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
        # Default: all digital
        block_configs = [{'conv1': 'digital', 'conv2': 'digital', 'downsample': 'digital'}] * num_layer

    layers = []
    for i in range(num_layer):
        config = block_configs[i]
        use_analog_conv1 = (config['conv1'] == 'analog')
        use_analog_conv2 = (config['conv2'] == 'analog')
        use_analog_downsample = (config.get('downsample', 'digital') == 'analog')

        if i == 0 and not first_layer:
            # First block with downsampling
            layers.append(ResidualBlockBaseline(
                in_ch, hidden_ch, use_conv=True, stride=2,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_downsample
            ))
        else:
            # Other blocks without downsampling
            layers.append(ResidualBlockBaseline(
                hidden_ch, hidden_ch,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_conv1  # Not used, but kept for consistency
            ))
    return layers


def create_model():
    """ResNet18 model with configurable digital/analog layers.

    Returns:
       nn.Module: created model
    """

    block_per_layers = (2, 2, 2, 2)  # ResNet18 structure
    base_channel = 64  # Standard ResNet18 channel size
    channel = (base_channel, 2 * base_channel, 4 * base_channel, 8 * base_channel)  # (64, 128, 256, 512)

    # Input layer - use configuration from LAYER_CONFIG
    input_use_analog = (LAYER_CONFIG['conv1'] == 'analog')
    if input_use_analog:
        input_rpu_config = create_analog_config()
        input_bias = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)
    else:
        input_rpu_config = FloatingPointRPUConfig()
        input_bias = False  # Standard ResNet: no bias in Conv (BatchNorm handles it)

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
        *concatenate_layer_blocks_baseline(
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
        *concatenate_layer_blocks_baseline(
            channel[0], channel[1], block_per_layers[1],
            block_configs=[
                LAYER_CONFIG['layer2_block0'],
                LAYER_CONFIG['layer2_block1'],
            ]
        )
    )

    # Layer3 (2 blocks, downsample in block0)
    l3 = nn.Sequential(
        *concatenate_layer_blocks_baseline(
            channel[1], channel[2], block_per_layers[2],
            block_configs=[
                LAYER_CONFIG['layer3_block0'],
                LAYER_CONFIG['layer3_block1'],
            ]
        )
    )

    # Layer4 (2 blocks, downsample in block0)
    l4_conv = nn.Sequential(
        *concatenate_layer_blocks_baseline(
            channel[2], channel[3], block_per_layers[3],
            block_configs=[
                LAYER_CONFIG['layer4_block0'],
                LAYER_CONFIG['layer4_block1'],
            ]
        )
    )

    # Final classification layer - use configuration from LAYER_CONFIG
    fc_use_analog = (LAYER_CONFIG['fc'] == 'analog')
    if fc_use_analog:
        fc_rpu_config = create_analog_config()
        fc_bias = True
    else:
        fc_rpu_config = FloatingPointRPUConfig()
        fc_bias = True

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
    print(f"  Analog device type: {ANALOG_DEVICE_TYPE}")
    print(f"  conv1: {'Analog (trainable base)' if input_use_analog else 'Digital (FloatingPoint)'}")
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
    print(f"  fc: {'Analog (trainable base)' if fc_use_analog else 'Digital (FloatingPoint)'}")
    print(f"  Using random initialization (no pretrained weights)\n")

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
        if isinstance(module, AnalogConv2d):
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

        elif isinstance(module, AnalogLinear):
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


def apply_warmup_cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_ratio=0.0, min_lr=1e-5):
    """Apply learning rate warmup + cosine annealing.

    Args:
        optimizer: SGD optimizer
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        base_lr: Base learning rate
        warmup_ratio: Fraction of epochs for warmup (0.0 = no warmup)
        min_lr: Minimum learning rate
    """
    import math

    warmup_epochs = int(total_epochs * warmup_ratio)

    if epoch <= warmup_epochs:
        # Linear warmup: lr = base_lr * (epoch / warmup_epochs)
        current_lr = base_lr * (epoch / warmup_epochs)
    else:
        # Cosine annealing after warmup
        # Progress through the cosine schedule (0 to π)
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        current_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


def main():
    """Train a PyTorch ResNet model with mixed analog/digital to classify CIFAR10."""
    # Seed
    manual_seed(SEED)

    # Get configuration parameters for run name
    analog_config = create_analog_config()
    mapping = analog_config.mapping
    forward_io = analog_config.forward

    # Calculate actual resolution values
    inp_res = 1.0/(2**7-2) if forward_io.inp_res == -1 else forward_io.inp_res
    out_res = 1.0/(2**9-2) if forward_io.out_res == -1 else forward_io.out_res

    # Initialize wandb
    wandb.init(
        project="aihwkit-resnet18-cifar10-fullanalog-scratch",
        name=f"resnet18_cifar10_fullanalog_scratch_bs{BATCH_SIZE}_e{N_EPOCHS}_wr{WARMUP_RATIO}_dev{ANALOG_DEVICE_TYPE}_aLR{LEARNING_RATE}_wd{WEIGHT_DECAY}_fwdIR{inp_res:.6f}_fwdOR{out_res:.6f}_fwdIN{forward_io.inp_noise}_fwdON{forward_io.out_noise}_mapW{mapping.weight_scaling_omega}",
        config={
            # Model and dataset
            "model": "ResNet18-Fullanalog-Scratch",
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

            # Analog device configuration
            "analog_device_type": ANALOG_DEVICE_TYPE,

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

            # Mapping parameters
            "mapping_weight_scaling_omega": mapping.weight_scaling_omega,
            "mapping_learn_out_scaling": mapping.learn_out_scaling,
            "mapping_digital_bias": mapping.digital_bias,
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

    # Initialize weights with Kaiming normal (ResNet default)
    initialize_resnet_weights(model)

    if USE_CUDA:
        model = model.to(DEVICE)

    print(f"Model moved to {DEVICE}")

    # Count parameters - analog tiles don't register weights as PyTorch parameters
    pytorch_params = sum(p.numel() for p in model.parameters())
    print(f"PyTorch registered parameters: {pytorch_params:,}")
    print("Note: Analog tile weights are stored internally in C++ and not counted here")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_sgd_optimizer(model, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)

    best_accuracy = 0
    best_epoch = 0

    print("\nStarting scratch fullanalog training on CIFAR10...")
    print("=" * 60)

    # Create overall progress bar for epochs
    epoch_pbar = tqdm(range(N_EPOCHS), desc="Overall Progress", position=0)

    for epoch in epoch_pbar:
        # Apply warmup + cosine annealing learning rate schedule
        apply_warmup_cosine_lr(optimizer, epoch + 1, N_EPOCHS, LEARNING_RATE, WARMUP_RATIO)

        # Train one epoch
        model, optimizer, train_loss, train_acc = train_step(
            train_data, model, criterion, optimizer, epoch + 1
        )

        # Run validation after each epoch
        model.eval()
        _, val_loss, val_error, val_accuracy = test_evaluation(validation_data, model, criterion)
        model.train()

        # Log both train and eval metrics together in one call to maintain proper step counting
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc / 100,  # Convert from percentage to ratio (0-1)
            "eval/loss": val_loss,
            "eval/accuracy": val_accuracy / 100,  # Convert from percentage to ratio (0-1)
            "eval/error": val_error,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

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

        # Print detailed progress
        if (epoch + 1) % 1 == 0:
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

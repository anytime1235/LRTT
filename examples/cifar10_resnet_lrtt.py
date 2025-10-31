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
from datetime import datetime

# Imports from PyTorch.
import torch
from torch import nn, Tensor, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from torchvision import datasets, transforms, models

# Progress bar
from tqdm import tqdm

# Logging
import wandb

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset, PythonLRTTDevice
from aihwkit.simulator.configs.spatial_lrtt_python import SpatialPythonLRTTPreset, SpatialPythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_LRTT")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "example_18_lrtt_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 300  # Reduced for LRTT demonstration
BATCH_SIZE = 128  # Reduced to prevent CUDA memory issues
LEARNING_RATE = 0.03
MOMENTUM = 0.9  # SGD momentum
WEIGHT_DECAY = 0.0002  # L2 regularization
NESTEROV = True  # Nesterov momentum
WARMUP_RATIO = 0.03  # Warmup ratio (10% of total epochs)
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# LRTT configuration parameters
LRTT_RANK_CONV = 8  # Rank for convolutional layers
LRTT_RANK_FC = 8  # Rank for fully connected layers
TRANSFER_EVERY = 1000  # Transfer A⊗B to C more frequently for better convergence
LORA_ALPHA = 2.0  # LoRA scaling factor

# Spatial LRTT for parameter reduction
USE_SPATIAL_LRTT = True  # Use spatial LRTT to reduce parameter count


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
            forward_inject=False,  # Enable forward_inject for conv layers
            correct_gradient_magnitudes=True,
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
            forward_inject=False,  # Enable forward_inject for conv layers
            unit_cell_devices=[
                IdealizedPresetDevice(),  # A 행렬: idealized device
                IdealizedPresetDevice(),  # B 행렬: idealized device
                IdealizedPresetDevice(),  # C 행렬: use all defaults
            ]
        )
    
    device_config.transfer_lr = device_config.lora_alpha
    
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
    
    # Optional: Add I/O configuration for forward/backward passes
    forward_io = IOParameters(
        # DAC (input) configuration (PresetIOParameters defaults)
        inp_res=0.007937,     # default: 7-bit DAC 1.0/(2**7-2) (≈0.007937)
        inp_bound=1.0,            # default: 1.0 
        inp_noise=0.0,            # default: 0.0 (no input noise)
        inp_sto_round=False,      # default: False (no stochastic rounding)
        
        # ADC (output) configuration (PresetIOParameters defaults) 
        out_res=0.001961,     # default: 9-bit ADC 1.0/(2**9-2) (≈0.001961)
        out_bound=12.0,           # default: 12.0 (dynamic range ratio)
        out_noise=0.06,            # default: 0.06 (~1 LSB of ADC)
        
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
    
    device_config.transfer_lr = device_config.lora_alpha
    
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

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        # Use LRTT analog convolutional layers
        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch, 
            kernel_size=3, padding=1, stride=stride,
            bias=False,  # LRTT doesn't support bias
            rpu_config=create_lrtt_config_conv()
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        
        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=False,
            rpu_config=create_lrtt_config_conv()
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=False,
                rpu_config=create_lrtt_config_conv()
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


def concatenate_layer_blocks_lrtt(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple LRTT residual blocks to form a layer.

    Returns:
       List: list of layer blocks
    """
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlockLRTT(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlockLRTT(hidden_ch, hidden_ch))
    return layers


def load_pretrained_weights(analog_model):
    """Load ImageNet pretrained weights into analog model.
    
    Args:
        analog_model: Analog model with LRTT layers
    """
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
            
            # Handle both LRTT and regular analog layers
            if hasattr(analog_layer, 'analog_module'):
                # For Spatial LRTT: d_size = c_out×k, x_size = c_in×k
                if hasattr(analog_layer.analog_module, 'c_out'):  # Spatial LRTT
                    analog_out_size = analog_layer.analog_module.c_out * analog_layer.analog_module.k
                    analog_in_size = analog_layer.analog_module.c_in * analog_layer.analog_module.k
                elif hasattr(analog_layer.analog_module, 'd_size'):  # Regular LRTT
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
            # or -> [out_ch*k, in_ch*k] for Spatial LRTT
            pretrained_shape = pretrained_weight.shape
            out_ch, in_ch, k_h, k_w = pretrained_shape
            
            # Check if this is spatial LRTT (parameter reduction) or regular analog layer
            is_spatial_lrtt = (analog_out_size == out_ch * k_h and analog_in_size == in_ch * k_w)
            is_regular_lrtt = (analog_out_size == out_ch and analog_in_size == in_ch * k_h * k_w)
            # Regular analog layer: has analog_module but not LRTT (e.g., FloatingPointTile)
            is_regular_analog = (hasattr(analog_layer, 'analog_module') and 
                                not hasattr(analog_layer.analog_module, 'c_out') and 
                                not hasattr(analog_layer.analog_module, 'd_size'))
            
            
            reshaped_weight = None
            
            if is_spatial_lrtt:
                # Handle kernel size mismatch (e.g., 7x7 -> 3x3)
                target_k = int((analog_out_size / out_ch)**0.5) if out_ch > 0 else k_h
                
                if k_h != target_k:
                    # Center crop if source kernel is larger
                    if k_h > target_k:
                        start = (k_h - target_k) // 2
                        end = start + target_k
                        pretrained_weight = pretrained_weight[:, :, start:end, start:end]
                        k_h = k_w = target_k
                        transfer_type = f"Spatial LRTT (cropped {pretrained_shape[2]}x{pretrained_shape[3]} -> {k_h}x{k_w})"
                    else:
                        continue
                else:
                    transfer_type = "Spatial LRTT"
                
                # Spatial LRTT: [out_ch, in_ch, k, k] -> [out_ch*k, in_ch*k]
                # Rearrange spatial dimensions
                weight_reshaped = pretrained_weight.permute(0, 2, 1, 3)  # [out_ch, k, in_ch, k]
                reshaped_weight = weight_reshaped.reshape(out_ch * k_h, in_ch * k_w)
                
            elif is_regular_lrtt:
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
                            # ImageNet ResNet first layer: 7x7 -> 3x3 (CIFAR-10 typical)
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
    
    # Transfer linear layers - adjust for CIFAR-10 (10 classes vs ImageNet 1000)
    if len(analog_linear_layers) > 0 and len(pretrained_linear_layers) > 0:
        print(f"  Transferring linear layers...")
        analog_fc = analog_linear_layers[0]  # Final FC layer
        pretrained_fc = pretrained_linear_layers[0]  # ImageNet FC layer
        
        # For CIFAR-10, we only transfer the input projection weights (512 dim)
        # but not the output weights (1000 -> 10 classes)
        analog_in_features = getattr(analog_fc, 'in_features', None)
        if hasattr(analog_fc, 'analog_module'):
            analog_in_features = analog_fc.analog_module.in_size
            
        if analog_in_features == pretrained_fc.in_features:
            # Create a new weight tensor with proper output size
            analog_out_features = getattr(analog_fc, 'out_features', None)
            if hasattr(analog_fc, 'analog_module'):
                analog_out_features = analog_fc.analog_module.out_size
            
            analog_weight_shape = (analog_out_features, analog_in_features)  # [10, 512] for CIFAR-10
            pretrained_weight = pretrained_fc.weight.data  # [1000, 512] for ImageNet
            
            # Use Xavier initialization for CIFAR-10 classifier
            # Don't copy ImageNet classes since they're unrelated to CIFAR-10
            import torch.nn.init as init
            new_weight = torch.empty(analog_weight_shape)
            init.xavier_uniform_(new_weight)
            
            try:
                if hasattr(analog_fc, 'set_weights'):
                    analog_fc.set_weights(new_weight, None)  # No bias for simplicity
                else:
                    analog_fc.weight.data.copy_(new_weight)
                transferred_count += 1
            except Exception:
                pass


def create_model(use_pretrained=True):
    """ResNet18 inspired analog model with LRTT layers.

    Args:
        use_pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
       nn.Module: created model with LRTT
    """

    block_per_layers = (2, 2, 2, 2)  # ResNet18 structure
    base_channel = 64  # Standard ResNet18 channel size
    channel = (base_channel, 2 * base_channel, 4 * base_channel, 8 * base_channel)  # (64, 128, 256, 512)

    # Input layer uses FloatingPointDevice for better stability
    l0 = nn.Sequential(
        AnalogConv2d(
            3, channel[0],
            kernel_size=3, stride=1, padding=1,
            bias=True,  # Can use bias with FloatingPoint
            rpu_config=FloatingPointRPUConfig()
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    l1 = nn.Sequential(
        *concatenate_layer_blocks_lrtt(channel[0], channel[0], block_per_layers[0], first_layer=True)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks_lrtt(channel[0], channel[1], block_per_layers[1]))
    l3 = nn.Sequential(*concatenate_layer_blocks_lrtt(channel[1], channel[2], block_per_layers[2]))
    l4_conv = nn.Sequential(*concatenate_layer_blocks_lrtt(channel[2], channel[3], block_per_layers[3]))
    
    # Final classification layer uses FloatingPointDevice for better stability
    from aihwkit.nn import AnalogLinear
    l5_fc = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(
            channel[3], N_CLASSES,  # 512 -> 10 for CIFAR-10
            bias=True,  # Can use bias with FloatingPoint
            rpu_config=FloatingPointRPUConfig()
        )
    )

    model = nn.Sequential(l0, l1, l2, l3, l4_conv, l5_fc)
    
    # Load pretrained weights if requested
    if use_pretrained:
        print("Loading ImageNet pretrained weights...")
        load_pretrained_weights(model)
    
    print(f"\nCreated ResNet with LRTT layers:")
    print(f"  Input layer: FloatingPointDevice")
    print(f"  Conv layers rank: {LRTT_RANK_CONV} (LRTT)")
    print(f"  Final FC layer: FloatingPointDevice") 
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  Pretrained: {use_pretrained}\n")
    
    return model


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


def calculate_eval_steps(transfer_every, steps_per_epoch):
    """Calculate evaluation steps closest to epoch boundaries.
    
    Args:
        transfer_every: LRTT transfer frequency
        steps_per_epoch: Number of training steps per epoch
        
    Returns:
        int: Evaluation frequency that's a multiple of transfer_every
    """
    # Find multiple of transfer_every closest to steps_per_epoch
    multiples = []
    for i in range(1, 20):  # Check first 20 multiples
        multiple = transfer_every * i
        diff = abs(multiple - steps_per_epoch)
        multiples.append((diff, multiple))
        
        # If we're getting too far from steps_per_epoch, stop
        if multiple > steps_per_epoch * 2:
            break
    
    # Return the closest multiple
    closest_multiple = min(multiples)[1]
    print(f"Using evaluation frequency: every {closest_multiple} steps "
          f"(transfer_every={transfer_every}, steps_per_epoch={steps_per_epoch})")
    return closest_multiple


def apply_warmup_cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_ratio=0.1, min_lr=1e-5):
    """Apply learning rate warmup + cosine annealing.
    
    Args:
        optimizer: SGD optimizer
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        base_lr: Base learning rate
        warmup_ratio: Fraction of epochs for warmup
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


def print_lrtt_statistics(model, epoch):
    """Print LRTT statistics for monitoring.
    
    Args:
        model (nn.Module): Model with LRTT layers
        epoch (int): Current epoch number
    """
    if epoch % 1 == 0:  # Print every 10 epochs
        print(f"\nLRTT Statistics at epoch {epoch}:")
        
        # Count LRTT layers and get statistics
        lrtt_count = 0
        total_transfers = 0
        total_original_params = 0
        total_spatial_params = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
                controller = module.analog_module.controller
                lrtt_count += 1
                total_transfers += controller.num_transfers
                
                # Check if it's spatial LRTT for parameter info
                if hasattr(module.analog_module, 'get_parameter_info'):
                    param_info = module.analog_module.get_parameter_info()
                    total_original_params += param_info['original_params']
                    total_spatial_params += param_info['spatial_params']
                    
                    if lrtt_count <= 3:  # Print first 3 layers
                        print(f"  {name}: A/B updates={controller.num_a_updates}, "
                              f"Transfers={controller.num_transfers}, "
                              f"Param reduction={param_info['reduction_percentage']}")
                else:
                    if lrtt_count <= 3:  # Print first 3 layers
                        print(f"  {name}: A/B updates={controller.num_a_updates}, "
                              f"Transfers={controller.num_transfers}")
        
        print(f"  Total LRTT layers: {lrtt_count}")
        print(f"  Total transfers: {total_transfers}")
        
        if total_original_params > 0:
            total_reduction = 1.0 - (total_spatial_params / total_original_params)
            print(f"  Total parameter reduction: {total_reduction:.1%} "
                  f"({total_original_params:,} → {total_spatial_params:,})")
        print()


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
        project="aihwkit-lrtt-resnet18-cifar10",
        name=f"resnet18_cifar10_imagenet_bs{BATCH_SIZE}_e{N_EPOCHS}_wr{WARMUP_RATIO}_mm{device_type_str}_aLR{LEARNING_RATE}_wd{WEIGHT_DECAY}_fwdIR{inp_res:.6f}_fwdOR{out_res:.6f}_fwdIN{forward_io.inp_noise}_fwdON{forward_io.out_noise}_mapW{mapping.weight_scaling_omega}_mapLOS{str(mapping.learn_out_scaling).lower()}_mapWSLC{str(mapping.weight_scaling_lr_compensation).lower()}_cgm{str(cgm).lower()}_fwdInj{str(fwd_inject).lower()}_r{LRTT_RANK_CONV}_t{TRANSFER_EVERY}_alpha{LORA_ALPHA}",
        config={
            # Model and dataset
            "model": "ResNet32-LRTT",
            "dataset": "CIFAR-10",
            "pretrained": "imagenet",
            
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
            "use_spatial_lrtt": USE_SPATIAL_LRTT,
            
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

    # Calculate evaluation frequency based on transfer_every and steps_per_epoch
    steps_per_epoch = len(train_data)
    eval_every_steps = calculate_eval_steps(TRANSFER_EVERY, steps_per_epoch)

    # Make the model
    USE_PRETRAINED = True  # Set to True to use ImageNet pretrained weights
    model = create_model(use_pretrained=USE_PRETRAINED)

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

    # C matrix monitoring for debugging forward_inject behavior
    prev_c_norms = {}  # Track previous epoch values for epoch-to-epoch changes
    print("\nRecording initial C matrix norms...")
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module'):
            try:
                if hasattr(module.analog_module, 'get_lrtt_component_weights'):
                    C, A, B = module.analog_module.get_lrtt_component_weights()
                    prev_c_norms[f"{name}_C"] = C.norm().item()
                    print(f"  {name}_C initial norm: {C.norm().item():.8f}")
                elif hasattr(module.analog_module, 'get_weights'):
                    weights, _ = module.analog_module.get_weights()
                    prev_c_norms[f"{name}_W"] = weights.norm().item()
                    print(f"  {name}_W initial norm: {weights.norm().item():.8f}")
            except Exception as e:
                print(f"  {name}: Could not access weights ({e})")

    best_accuracy = 0
    best_epoch = 0
    
    print("\nStarting LRTT training on CIFAR10...")
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
        
        # Check C matrix changes every epoch for debugging
        lrtt_max_change = 0.0
        fp_max_change = 0.0
        lrtt_changes = []
        fp_changes = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'analog_module'):
                try:
                    if hasattr(module.analog_module, 'get_lrtt_component_weights'):
                        # LRTT layer - track C matrix
                        C, A, B = module.analog_module.get_lrtt_component_weights()
                        current_norm = C.norm().item()
                        if f"{name}_C" in prev_c_norms:
                            change = abs(current_norm - prev_c_norms[f"{name}_C"])
                            lrtt_max_change = max(lrtt_max_change, change)
                            lrtt_changes.append(f"{name}_C:{change:.8f}")
                    elif hasattr(module.analog_module, 'get_weights'):
                        # FloatingPoint layer - track W matrix
                        weights, _ = module.analog_module.get_weights()
                        current_norm = weights.norm().item()
                        if f"{name}_W" in prev_c_norms:
                            change = abs(current_norm - prev_c_norms[f"{name}_W"])
                            fp_max_change = max(fp_max_change, change)
                            fp_changes.append(f"{name}_W:{change:.8f}")
                except Exception as e:
                    pass
        
        # Print separate status for LRTT C matrices and FloatingPoint layers
        lrtt_status = "🔴CHANGING!" if lrtt_max_change > 1e-6 else "🟡small" if lrtt_max_change > 1e-8 else "🟢stable"
        fp_status = "🔴CHANGING!" if fp_max_change > 1e-6 else "🟡small" if fp_max_change > 1e-8 else "🟢stable"
        
        print(f"LRTT C Matrix: {lrtt_status} (max: {lrtt_max_change:.8f}) | FloatingPoint: {fp_status} (max: {fp_max_change:.8f})")
        
        # Print LRTT statistics (less frequent to avoid clutter)  
        if (epoch + 1) % 10 == 0:
            print_lrtt_statistics(model, epoch + 1)
            
            # Detailed matrix changes every 10 epochs - separate LRTT and FloatingPoint
            if lrtt_changes or fp_changes:
                print(f"\nDetailed Matrix Changes at epoch {epoch + 1}:")
                
                if lrtt_changes:
                    print("  LRTT C Matrices:")
                    for change_info in lrtt_changes[:3]:  # Show first 3 LRTT layers
                        layer_name, change_val = change_info.split(':')
                        print(f"    {layer_name}: {change_val}")
                    if len(lrtt_changes) > 3:
                        print(f"    ... and {len(lrtt_changes) - 3} more LRTT layers")
                
                if fp_changes:
                    print("  FloatingPoint Layers:")
                    for change_info in fp_changes[:2]:  # Show first 2 FP layers
                        layer_name, change_val = change_info.split(':')
                        print(f"    {layer_name}: {change_val}")
                    if len(fp_changes) > 2:
                        print(f"    ... and {len(fp_changes) - 2} more FP layers")

    print("=" * 60)
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")
    print(f"Model weights saved to: {WEIGHT_PATH}")


if __name__ == "__main__":
    main()
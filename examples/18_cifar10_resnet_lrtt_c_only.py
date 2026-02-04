# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 18 with LRTT: ResNet32 CNN with CIFAR10 using LRTT layers.

CIFAR10 dataset on a ResNet inspired network using LRTT (Low-Rank Tensor-Train)
analog layers based on the paper: https://arxiv.org/abs/1512.03385

Modified to use C-only validation (visible weights only, no A@B contribution).
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

from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.rpu_base import cuda


# Device to use
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_LRTT_C_ONLY")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "example_18_lrtt_c_only_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 100  # Reduced for LRTT demonstration
BATCH_SIZE = 128  # Increased for better GPU utilization
LEARNING_RATE = 0.1
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# LRTT configuration parameters
LRTT_RANK_CONV = 8  # Rank for convolutional layers
LRTT_RANK_FC = 16  # Rank for fully connected layers
TRANSFER_EVERY = 100  # Transfer A⊗B to C more frequently for better convergence
LORA_ALPHA = 1  # LoRA scaling factor


def create_lrtt_config_conv():
    """Create LRTT configuration for convolutional layers.
    
    Returns:
        PythonLRTTRPUConfig: LRTT configuration for conv layers
    """
    device_config = PythonLRTTPreset.idealized(
        rank=LRTT_RANK_CONV,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA
    )
    device_config.transfer_lr = device_config.lora_alpha
    device_config.forward_inject = True
    device_config.correct_gradient_magnitudes = True
    
    # Add mapping for larger layers
    mapping = MappingParameter(
        weight_scaling_omega=0.6,
        max_input_size=512,
        max_output_size=512
    )
    
    return PythonLRTTRPUConfig(device=device_config, mapping=mapping)


def create_lrtt_config_fc():
    """Create LRTT configuration for fully connected layers.
    
    Returns:
        PythonLRTTRPUConfig: LRTT configuration for FC layers
    """
    device_config = PythonLRTTPreset.idealized(
        rank=LRTT_RANK_FC,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        forward_inject=False,
        correct_gradient_magnitudes=True
    )
    device_config.transfer_lr = device_config.lora_alpha
    
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


def create_model():
    """ResNet34 inspired analog model with LRTT layers.

    Returns:
       nn.Module: created model with LRTT
    """

    block_per_layers = (3, 4, 6, 3)
    base_channel = 16
    channel = (base_channel, 2 * base_channel, 4 * base_channel)

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
    
    # Final classification layer uses FloatingPointDevice for better stability
    l4 = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(
            channel[2], N_CLASSES,
            bias=True,  # Can use bias with FloatingPoint
            rpu_config=FloatingPointRPUConfig()
        )
    )

    model = nn.Sequential(l0, l1, l2, l3, l4)
    
    print(f"\nCreated ResNet with LRTT layers:")
    print(f"  Input layer: FloatingPointDevice")
    print(f"  Conv layers rank: {LRTT_RANK_CONV} (LRTT)")
    print(f"  Final FC layer: FloatingPointDevice")
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  C-only validation: ENABLED\n")
    
    return model


def load_images():
    """Load images for train from torchvision datasets.

    Returns:
        Dataset, Dataset: train data and validation data"""
    mean = Tensor([0.4914, 0.4822, 0.4465])
    std = Tensor([0.2470, 0.2435, 0.2616])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)

    return train_data, validation_data


def create_sgd_optimizer(model, learning_rate):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate

    Returns:
        Optimizer: created analog optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer.regroup_param_groups(model)

    return optimizer


def train_step(train_data, model, criterion, optimizer, scaler):
    """Train a single epoch.

    Args:
        train_data (DataLoader): train data set
        model (nn.Module): model to train
        criterion: criterion to compute loss
        optimizer: analog optimizer
        scaler: gradient scaler for mixed precision

    Returns:
        float: train loss of the epoch
    """
    total_loss = 0
    model.train()

    for images, labels in train_data:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        # Enable autocast for mixed precision
        with autocast():
            output = model(images)
            loss = criterion(output, labels)

        # Scale gradients and update
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
    
    epoch_loss = total_loss / len(train_data.dataset)
    return epoch_loss


def validate_c_only(validation_data, model, criterion):
    """Validate using only C tiles (no A/B contribution).
    
    This function performs forward pass using only the visible weights (C-array)
    without the LoRA components (A@B), providing a measure of the base model performance.
    
    Args:
        validation_data (DataLoader): Validation set
        model (nn.Module): Model with LRTT layers
        criterion: Loss criterion
        
    Returns:
        float, float: validation loss and accuracy using C-only
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0
    
    model.eval()
    
    with no_grad():
        for images, labels in validation_data:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # Forward pass through the model using only C tiles for LRTT layers
            x = images
            
            for layer_idx, layer in enumerate(model):
                if isinstance(layer, nn.Sequential):
                    # Process sequential blocks (l1, l2, l3)
                    for block in layer:
                        if hasattr(block, 'conv1'):
                            # ResidualBlock with LRTT layers
                            # Conv1
                            if hasattr(block.conv1, 'analog_tiles') and hasattr(block.conv1.analog_tiles, 'controller'):
                                controller = block.conv1.analog_tiles.controller
                                y = controller.tile_c.forward(x)
                            else:
                                y = block.conv1(x)
                            y = F.relu(block.bn1(y))
                            
                            # Conv2
                            if hasattr(block.conv2, 'analog_tiles') and hasattr(block.conv2.analog_tiles, 'controller'):
                                controller = block.conv2.analog_tiles.controller
                                y = controller.tile_c.forward(y)
                            else:
                                y = block.conv2(y)
                            y = block.bn2(y)
                            
                            # Skip connection
                            if block.convskip is not None:
                                if hasattr(block.convskip, 'analog_tiles') and hasattr(block.convskip.analog_tiles, 'controller'):
                                    controller = block.convskip.analog_tiles.controller
                                    x = controller.tile_c.forward(x)
                                else:
                                    x = block.convskip(x)
                            
                            x = F.relu(y + x)
                        else:
                            # Regular layers (l0, l4)
                            for sublayer in layer:
                                if isinstance(sublayer, (AnalogConv2d, AnalogLinear)):
                                    if hasattr(sublayer, 'analog_tiles') and hasattr(sublayer.analog_tiles, 'controller'):
                                        # LRTT layer - use only C tile
                                        controller = sublayer.analog_tiles.controller
                                        x = controller.tile_c.forward(x)
                                    else:
                                        # Non-LRTT analog layer
                                        x = sublayer(x)
                                else:
                                    # Regular PyTorch layers
                                    x = sublayer(x)
                else:
                    # Single layer
                    x = layer(x)
            
            # Calculate loss and accuracy
            output = x
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch_max(output.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
    
    epoch_loss = total_loss / len(validation_data.dataset)
    accuracy = predicted_ok / total_images * 100
    
    return epoch_loss, accuracy


def test_evaluation(validation_data, model, criterion):
    """Test with full model (C + A@B).
    
    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        float, float: test loss and test accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    with no_grad():
        for images, labels in validation_data:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch_max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
            
    epoch_loss = total_loss / len(validation_data.dataset)
    accuracy = predicted_ok / total_images * 100

    return epoch_loss, accuracy


def print_lrtt_statistics(model):
    """Print LRTT statistics for monitoring.
    
    Args:
        model (nn.Module): Model with LRTT layers
    """
    # Count LRTT layers and get statistics
    lrtt_count = 0
    total_transfers = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'analog_tiles') and hasattr(module.analog_tiles, 'controller'):
            controller = module.analog_tiles.controller
            lrtt_count += 1
            total_transfers += controller.num_transfers
    
    if lrtt_count > 0:
        print(f"  LRTT layers: {lrtt_count}, Total transfers: {total_transfers}")


def train_model(model, train_data, validation_data):
    """Train the model with C-only validation.

    Args:
        model (nn.Module): model to be trained
        train_data (DataLoader): train dataset
        validation_data (DataLoader): validation dataset
    """
    print(f"Training on {DEVICE}")
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = create_sgd_optimizer(model, LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    best_accuracy = 0.0
    best_c_only_accuracy = 0.0
    
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting training...")
    print("-" * 80)
    
    for epoch in range(N_EPOCHS):
        # Train
        train_loss = train_step(train_data, model, criterion, optimizer, scaler)
        
        # Validate with C-only
        val_loss_c_only, val_acc_c_only = validate_c_only(validation_data, model, criterion)
        
        # Validate with full model (C + A@B)
        val_loss_full, val_acc_full = test_evaluation(validation_data, model, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model (based on full accuracy)
        if val_acc_full > best_accuracy:
            best_accuracy = val_acc_full
            save(model.state_dict(), WEIGHT_PATH)
        
        if val_acc_c_only > best_c_only_accuracy:
            best_c_only_accuracy = val_acc_c_only
        
        # Print progress every epoch
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Train: {train_loss:.4f} | "
              f"C-only: {val_acc_c_only:.2f}% | "
              f"Full: {val_acc_full:.2f}% | "
              f"Best Full: {best_accuracy:.2f}% | "
              f"LR: {current_lr:.4f}", end="")
        
        # Print LRTT statistics
        print_lrtt_statistics(model)
    
    print("-" * 80)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Training completed!")
    print(f"Best validation accuracy (Full model): {best_accuracy:.2f}%")
    print(f"Best validation accuracy (C-only): {best_c_only_accuracy:.2f}%")
    print(f"Model weights saved to: {WEIGHT_PATH}")


def main():
    """Train ResNet on CIFAR10 with LRTT layers using C-only validation."""
    
    # Seed for reproducibility
    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)
    
    # Load datasets
    train_data, validation_data = load_images()
    
    # Create model
    model = create_model()
    model.to(DEVICE)
    
    # Train model
    train_model(model, train_data, validation_data)
    
    # Final test
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load(WEIGHT_PATH))
    criterion = nn.CrossEntropyLoss()
    
    # Test with C-only
    test_loss_c_only, test_acc_c_only = validate_c_only(validation_data, model, criterion)
    
    # Test with full model
    test_loss_full, test_acc_full = test_evaluation(validation_data, model, criterion)
    
    print(f"Final Test Accuracy (C-only): {test_acc_c_only:.2f}%")
    print(f"Final Test Accuracy (Full model): {test_acc_full:.2f}%")
    print(f"Improvement from LoRA (A@B): +{test_acc_full - test_acc_c_only:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
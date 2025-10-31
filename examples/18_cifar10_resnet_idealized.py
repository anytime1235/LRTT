# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 18 with Idealized devices: ResNet32 CNN with CIFAR10.

CIFAR10 dataset on a ResNet inspired network using IdealizedPreset and 
TikiTakaIdealizedPreset analog layers based on the paper: https://arxiv.org/abs/1512.03385
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
from time import time

from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets.configs import IdealizedPreset, TikiTakaIdealizedPreset
from aihwkit.simulator.configs import FloatingPointRPUConfig, MappingParameter
from aihwkit.simulator.rpu_base import cuda


# Device to use
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_IDEALIZED")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "example_18_idealized_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 100
BATCH_SIZE = 128  # Increased for better GPU utilization
LEARNING_RATE = 0.1
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# Device configuration
USE_TIKITAKA = True  # Set to False to use regular IdealizedPreset


def create_analog_config(use_tikitaka=USE_TIKITAKA):
    """Create analog configuration for layers.
    
    Args:
        use_tikitaka (bool): Whether to use TikiTaka variant
        
    Returns:
        RPUConfig: Configuration for analog layers
    """
    if use_tikitaka:
        # TikiTaka provides better gradient updates
        config = TikiTakaIdealizedPreset()
    else:
        # Standard idealized device
        config = IdealizedPreset()
    
    # Add mapping for larger layers
    mapping = MappingParameter(
        weight_scaling_omega=0.6,
        max_input_size=512,
        max_output_size=512
    )
    config.mapping = mapping
    
    return config


class ResidualBlockIdealized(nn.Module):
    """Residual block with Idealized analog convolutional layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        # Use Idealized analog convolutional layers
        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch, 
            kernel_size=3, padding=1, stride=stride,
            bias=False,
            rpu_config=create_analog_config()
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        
        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=False,
            rpu_config=create_analog_config()
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=False,
                rpu_config=create_analog_config()
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


def concatenate_layer_blocks_idealized(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple idealized residual blocks to form a layer.

    Returns:
       List: list of layer blocks
    """
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlockIdealized(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlockIdealized(hidden_ch, hidden_ch))
    return layers


def create_model():
    """ResNet34 inspired analog model with Idealized layers.

    Returns:
       nn.Module: created model with Idealized devices
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
        *concatenate_layer_blocks_idealized(channel[0], channel[0], block_per_layers[0], first_layer=True)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks_idealized(channel[0], channel[1], block_per_layers[1]))
    l3 = nn.Sequential(*concatenate_layer_blocks_idealized(channel[1], channel[2], block_per_layers[2]))
    
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
    
    device_type = "TikiTakaIdealizedPreset" if USE_TIKITAKA else "IdealizedPreset"
    print(f"\nCreated ResNet with Idealized layers:")
    print(f"  Input layer: FloatingPointDevice")
    print(f"  Conv layers: {device_type}")
    print(f"  Final FC layer: FloatingPointDevice")
    print(f"  Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
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
                          num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False,
                          persistent_workers=True if NUM_WORKERS > 0 else False)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE * 2, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False,
                               persistent_workers=True if NUM_WORKERS > 0 else False)

    return train_data, validation_data


def create_sgd_optimizer(model, learning_rate):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate

    Returns:
        Optimizer: created analog optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
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
        optimizer.zero_grad(set_to_none=True)  # More efficient

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


def test(validation_data, model, criterion):
    """Test the model on the validation set.

    Args:
        validation_data (DataLoader): validation data set
        model (nn.Module): model to test
        criterion: criterion to compute loss

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

            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch_max(output.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()

    epoch_loss = total_loss / len(validation_data.dataset)
    accuracy = predicted_ok / total_images * 100
    return epoch_loss, accuracy


def train_model(model, train_data, validation_data):
    """Train the model.

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
    
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting training...")
    print("-" * 80)
    
    start_time = time()
    for epoch in range(N_EPOCHS):
        epoch_start = time()
        # Train
        train_loss = train_step(train_data, model, criterion, optimizer, scaler)
        
        # Validate
        val_loss, val_accuracy = test(validation_data, model, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save(model.state_dict(), WEIGHT_PATH)
        
        # Print progress
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"Best: {best_accuracy:.2f}% | "
              f"LR: {current_lr:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    total_time = time() - start_time
    print("-" * 80)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Training completed!")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Model weights saved to: {WEIGHT_PATH}")


def main():
    """Main function to train ResNet on CIFAR10 with Idealized devices."""
    
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
    test_loss, test_accuracy = test(validation_data, model, criterion)
    
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    device_type = "TikiTakaIdealizedPreset" if USE_TIKITAKA else "IdealizedPreset"
    print(f"\nModel used: ResNet with {device_type} analog layers")
    print(f"Input/Classifier layers: FloatingPointDevice")
    print("="*80)


if __name__ == "__main__":
    main()
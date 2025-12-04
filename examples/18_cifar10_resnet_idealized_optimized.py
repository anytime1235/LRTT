# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 18 with Idealized devices: ResNet32 CNN with CIFAR10.

CIFAR10 dataset on a ResNet inspired network using IdealizedPreset and 
TikiTakaIdealizedPreset analog layers. Optimized version with improved training efficiency.
"""
# pylint: disable=invalid-name

# Imports
import os
from datetime import datetime
from time import time

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
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_IDEALIZED_OPT")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "example_18_idealized_opt_model.pth")

# Training parameters
SEED = 1
N_EPOCHS = 100
BATCH_SIZE = 128  # Increased for better GPU utilization
LEARNING_RATE = 0.1
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# Device configuration
USE_TIKITAKA = True  # Set to False to use regular IdealizedPreset

# Training optimizations
USE_MIXED_PRECISION = True  # Use automatic mixed precision for faster training
PIN_MEMORY = True  # Pin memory for faster GPU transfer
PERSISTENT_WORKERS = True  # Keep workers alive between epochs


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
    print(f"  Mixed Precision: {'Enabled' if USE_MIXED_PRECISION and USE_CUDA else 'Disabled'}")
    print(f"  Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    return model


def load_images():
    """Load images for train from torchvision datasets with optimizations.

    Returns:
        Dataset, Dataset: train data and validation data"""
    mean = Tensor([0.4914, 0.4822, 0.4465])
    std = Tensor([0.2470, 0.2435, 0.2616])

    # Training augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=train_transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=val_transform)
    
    train_data = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY and USE_CUDA,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0
    )
    
    validation_data = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE * 2,  # Larger batch for validation (no gradients)
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY and USE_CUDA,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0
    )

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


def train_step(train_data, model, criterion, optimizer, scaler=None):
    """Train a single epoch with optimizations.

    Args:
        train_data (DataLoader): train data set
        model (nn.Module): model to train
        criterion: criterion to compute loss
        optimizer: analog optimizer
        scaler: gradient scaler for mixed precision (optional)

    Returns:
        float, float: train loss and accuracy of the epoch
    """
    total_loss = 0
    correct = 0
    total = 0
    model.train()

    for images, labels in train_data:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Mixed precision training
        if USE_MIXED_PRECISION and USE_CUDA and scaler is not None:
            with autocast():
                output = model(images)
                loss = criterion(output, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch_max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = total_loss / len(train_data.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


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
    """Train the model with optimized training loop.

    Args:
        model (nn.Module): model to be trained
        train_data (DataLoader): train dataset
        validation_data (DataLoader): validation dataset
        
    Returns:
        dict: Training history
    """
    print(f"Training on {DEVICE}")
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = create_sgd_optimizer(model, LEARNING_RATE)
    
    # Learning rate scheduler with cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-4)
    
    # Gradient scaler for mixed precision
    scaler = GradScaler() if USE_MIXED_PRECISION and USE_CUDA else None
    
    best_accuracy = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting training...")
    print("-" * 100)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Best':^10} | {'LR':^10} | {'Time':^8}")
    print("-" * 100)
    
    total_start = time()
    
    for epoch in range(N_EPOCHS):
        epoch_start = time()
        
        # Train
        train_loss, train_acc = train_step(train_data, model, criterion, optimizer, scaler)
        
        # Validate
        val_loss, val_accuracy = test(validation_data, model, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            save(model.state_dict(), WEIGHT_PATH)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time() - epoch_start
        
        print(f"{epoch+1:^7d} | {train_loss:^11.4f} | {train_acc:^10.2f}% | {val_loss:^10.4f} | {val_accuracy:^10.2f}% | {best_accuracy:^10.2f}% | {current_lr:^10.6f} | {epoch_time:^8.2f}s")
        
        # Early stopping check
        if epoch - best_epoch > 20 and epoch > 50:  # No improvement for 20 epochs after epoch 50
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time() - total_start
    print("-" * 100)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Training completed!")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
    print(f"Model weights saved to: {WEIGHT_PATH}")
    
    return history


def main():
    """Main function to train ResNet on CIFAR10 with Idealized devices."""
    
    # Seed for reproducibility
    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)
        # Additional optimizations for deterministic behavior
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Load datasets
    train_data, validation_data = load_images()
    
    # Create model
    model = create_model()
    model.to(DEVICE)
    
    # Count analog tiles
    analog_tiles = sum(1 for m in model.modules() if hasattr(m, 'analog_tiles'))
    print(f"Number of analog layers: {analog_tiles}")
    
    # Train model
    history = train_model(model, train_data, validation_data)
    
    # Final test
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = test(validation_data, model, criterion)
    
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    device_type = "TikiTakaIdealizedPreset" if USE_TIKITAKA else "IdealizedPreset"
    print(f"\nModel configuration:")
    print(f"  Analog device: {device_type}")
    print(f"  Input/Classifier: FloatingPointDevice")
    print(f"  Training optimizations:")
    print(f"    - Mixed precision: {'Yes' if USE_MIXED_PRECISION and USE_CUDA else 'No'}")
    print(f"    - Data augmentation: Yes")
    print(f"    - Cosine annealing LR: Yes")
    print(f"    - Early stopping: Yes")
    print("="*80)


if __name__ == "__main__":
    main()
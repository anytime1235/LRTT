# -*- coding: utf-8 -*-
"""MNIST training with optimal transfer_every=75"""

import os
from time import time

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.rpu_base import cuda


# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 30
BATCH_SIZE = 64

# LRTT parameters - optimal settings
LRTT_RANKS = [16, 8]     # Ranks for LRTT layers (input->hidden1, hidden1->hidden2)
TRANSFER_EVERY = 75      # Optimal transfer period (from analysis)
LORA_ALPHA = 1.0         # Simple scaling


def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data


def create_lrtt_config(rank):
    """Create LRTT configuration with specified rank."""
    device_config = PythonLRTTPreset.idealized(
        rank=rank,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA
    )

    # Standard reinit mode
    device_config.reinit_mode = "standard"
    device_config.reinit_gain = 1.0
    device_config.correct_gradient_magnitudes = True
    device_config.transfer_lr = 1.0
    device_config.forward_inject = False  # Use C only for forward

    return PythonLRTTRPUConfig(device=device_config)


def create_standard_analog_config():
    """Create standard analog config for output layer (no LRTT)."""
    return SingleRPUConfig(device=IdealizedPresetDevice())


def create_analog_network(input_size, hidden_sizes, output_size):
    """Create the neural network using LRTT analog linear layers."""

    layers = []

    # Input -> Hidden1 (LRTT)
    rpu_config = create_lrtt_config(LRTT_RANKS[0])
    layers.append(AnalogLinear(input_size, hidden_sizes[0], rpu_config=rpu_config, bias=True))
    layers.append(nn.Sigmoid())

    # Hidden1 -> Hidden2 (LRTT)
    rpu_config = create_lrtt_config(LRTT_RANKS[1])
    layers.append(AnalogLinear(hidden_sizes[0], hidden_sizes[1], rpu_config=rpu_config, bias=True))
    layers.append(nn.Sigmoid())

    # Hidden2 -> Output (Standard Analog - no LRTT due to small output size)
    rpu_config = create_standard_analog_config()
    layers.append(AnalogLinear(hidden_sizes[1], output_size, rpu_config=rpu_config, bias=True))
    layers.append(nn.LogSoftmax(dim=1))

    return AnalogSequential(*layers)


def train_step(train_data, model, criterion, optimizer):
    """Train one epoch."""
    total_loss = 0
    model.train()

    for images, labels in train_data:
        images = images.view(-1, INPUT_SIZE).to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        if torch.isnan(loss):
            print("NaN loss detected!")
            return float('inf')

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(train_data.dataset)


def test_evaluation(validation_data, model, criterion):
    """Evaluate on validation set."""
    total_loss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        for images, labels in validation_data:
            images = images.view(-1, INPUT_SIZE).to(DEVICE)
            labels = labels.to(DEVICE)

            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(validation_data.dataset)
    avg_loss = total_loss / len(validation_data.dataset)

    return avg_loss, accuracy


def main():
    """Main training loop."""
    print("=" * 70)
    print("MNIST Training with Optimal LRTT Settings")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Transfer every: {TRANSFER_EVERY}")
    print(f"LRTT ranks: {LRTT_RANKS}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 70)

    # Load data
    train_data, validation_data = load_images()
    print(f"Train samples: {len(train_data.dataset)}")
    print(f"Val samples: {len(validation_data.dataset)}")

    # Create model
    model = create_analog_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model.to(DEVICE)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer and criterion
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    # Training
    best_accuracy = 0
    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(EPOCHS):
        t_start = time()

        train_loss = train_step(train_data, model, criterion, optimizer)
        val_loss, accuracy = test_evaluation(validation_data, model, criterion)

        scheduler.step()

        t_elapsed = time() - t_start

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            marker = " *"
        else:
            marker = ""

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.2f}%{marker} | "
              f"Time: {t_elapsed:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.4f}")

        # Early stopping check
        if train_loss == float('inf'):
            print("Training diverged! Stopping.")
            break

    print("-" * 70)
    print(f"Training completed!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print("=" * 70)

    return best_accuracy


if __name__ == "__main__":
    main()

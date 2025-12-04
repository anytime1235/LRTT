# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example: MNIST training with 6T1C A/B tiles and Idealized C tile.

MNIST training using LRTT (Low-Rank Tensor-Train) with:
- A/B tiles: 6T1C (6 Transistors, 1 Capacitor) devices
- C tile (visible): IdealizedPresetDevice

The 6T1C device parameters are based on experimental measurements:
- dw_min = 0.001981
- gamma_up = -0.1678 (slight saturation)
- gamma_down = +0.1410 (near-linear)
- Time constant tau = 775.1 min (retention)

The final output layer uses IdealizedPresetDevice (same as C tile type).
"""
# pylint: disable=invalid-name, redefined-outer-name

import os
from time import time

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.presets.configs import IdealizedPreset
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

# LRTT parameters
LRTT_RANK = 8  # Rank for all LRTT layers
TRANSFER_EVERY = 100  # Transfer A*B to C every N updates
LORA_ALPHA = 4.0  # LoRA scaling factor

# 6T1C retention parameters
DT_BATCH_SEC = 1.0  # Assumed time per mini-batch (seconds)
INCLUDE_RETENTION = True  # Whether to include 6T1C retention effects


def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard normalization
    ])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data


def create_6t1c_lrtt_config(rank):
    """Create LRTT configuration with 6T1C A/B and Idealized C.

    Args:
        rank (int): Rank for the LRTT decomposition

    Returns:
        PythonLRTTRPUConfig: LRTT configuration with 6T1C A/B tiles
    """
    # Use 6T1C for A/B tiles, Idealized for C tile
    device_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=rank,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        dt_batch_sec=DT_BATCH_SEC
    )

    # Stability improvements
    device_config.reinit_gain = 0.5
    device_config.correct_gradient_magnitudes = True
    device_config.transfer_lr = LORA_ALPHA

    return PythonLRTTRPUConfig(device=device_config)


def create_analog_network(input_size, hidden_sizes, output_size):
    """Create the neural network using LRTT with 6T1C A/B tiles.

    LRTT layers use:
    - A tile: 6T1C device
    - B tile: 6T1C device
    - C tile: IdealizedPresetDevice

    Final output layer uses IdealizedPresetDevice (same type as C tile).

    Args:
        input_size (int): size of the Tensor at the input.
        hidden_sizes (list): list of sizes of the hidden layers.
        output_size (int): size of the Tensor at the output.

    Returns:
        nn.Module: created analog model
    """
    print("=" * 60)
    print("Creating LRTT Network with 6T1C A/B + Idealized C")
    print("=" * 60)
    print(f"  LRTT Rank: {LRTT_RANK}")
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  6T1C dt_batch: {DT_BATCH_SEC} sec")
    print(f"  6T1C retention: {'Enabled' if INCLUDE_RETENTION else 'Disabled'}")
    print("-" * 60)
    print("  Layer structure:")
    print(f"    Layer 1: {input_size} -> {hidden_sizes[0]} (LRTT: 6T1C A/B + Ideal C)")
    print(f"    Layer 2: {hidden_sizes[0]} -> {hidden_sizes[1]} (LRTT: 6T1C A/B + Ideal C)")
    print(f"    Layer 3: {hidden_sizes[1]} -> {output_size} (IdealizedPreset - same as C tile)")
    print("=" * 60)

    model = AnalogSequential(
        # Layer 1: 784 -> 256 with LRTT (6T1C A/B + Idealized C)
        AnalogLinear(
            input_size,
            hidden_sizes[0],
            bias=False,  # LRTT doesn't support bias
            rpu_config=create_6t1c_lrtt_config(LRTT_RANK),
        ),
        nn.ReLU(),
        # Layer 2: 256 -> 128 with LRTT (6T1C A/B + Idealized C)
        AnalogLinear(
            hidden_sizes[0],
            hidden_sizes[1],
            bias=False,
            rpu_config=create_6t1c_lrtt_config(LRTT_RANK),
        ),
        nn.ReLU(),
        # Layer 3: 128 -> 10 with IdealizedPreset (same type as C tile)
        AnalogLinear(
            hidden_sizes[1],
            output_size,
            bias=False,
            rpu_config=IdealizedPreset(),
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    return model


def create_sgd_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer.regroup_param_groups(model)

    return optimizer


def train(model, train_set, val_set):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
        val_set (DataLoader): dataset of elements to use for validation.
    """
    classifier = nn.NLLLoss()
    optimizer = create_sgd_optimizer(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    time_init = time()
    for epoch_number in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        model.train()
        for batch_idx, (images, labels) in enumerate(train_set):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            output = model(images)
            loss = classifier(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(labels.view_as(pred)).sum().item()
            total_samples += images.size(0)

            # Print progress every 200 batches
            if batch_idx % 200 == 0 and batch_idx > 0:
                batch_acc = 100. * correct_predictions / total_samples
                print(f"  [Epoch {epoch_number+1:2d}] Batch {batch_idx:4d}/{len(train_set)}: "
                      f"Loss={loss.item():.4f}, Acc={batch_acc:.2f}%")

        # End of epoch
        epoch_accuracy = 100. * correct_predictions / total_samples
        avg_loss = total_loss / len(train_set)

        # Validation
        val_accuracy = test_evaluation(model, val_set, verbose=False)
        val_accuracy_c_only = validate_c_only(model, val_set)

        scheduler.step()

        print(f"Epoch {epoch_number + 1:2d}/{EPOCHS}: "
              f"Loss={avg_loss:.4f}, "
              f"Train={epoch_accuracy:.2f}%, "
              f"Val={val_accuracy:.2f}%, "
              f"Val(C-only)={val_accuracy_c_only:.2f}%, "
              f"LR={scheduler.get_last_lr()[0]:.5f}")

        # Print LRTT statistics
        print_lrtt_stats(model, epoch_number)

    print(f"\nTotal Training Time: {(time() - time_init) / 60:.2f} mins")


def print_lrtt_stats(model, epoch):
    """Print LRTT layer statistics."""
    layer_idx = 0
    for i, layer in enumerate(model):
        if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
            controller = layer.analog_module.controller
            layer_idx += 1
            if epoch == EPOCHS - 1:  # Only print detailed stats on last epoch
                print(f"    Layer {layer_idx} LRTT: "
                      f"A_updates={controller.num_a_updates}, "
                      f"B_updates={controller.num_b_updates}, "
                      f"Transfers={controller.num_transfers}")


def validate_c_only(model, val_set):
    """Validate using only C tiles (no A/B contribution).

    This tests how well the accumulated weights in C tile perform alone.

    Args:
        model (nn.Module): Trained model.
        val_set (DataLoader): Validation set.

    Returns:
        float: Validation accuracy using C-only forward pass.
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            images = images.view(images.shape[0], -1)

            # Forward pass using only C tiles for LRTT layers
            x = images
            for layer in model:
                if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
                    # LRTT layer: use only C tile
                    controller = layer.analog_module.controller
                    x = controller.tile_c.forward(x)
                elif isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
                elif isinstance(layer, nn.LogSoftmax):
                    x = torch.log_softmax(x, dim=1)
                elif hasattr(layer, 'analog_module'):
                    # Non-LRTT analog layer (final layer)
                    x = layer(x)

            _, predicted = torch.max(x.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100. * correct / total


def test_evaluation(model, val_set, verbose=True):
    """Test the trained network.

    Args:
        model (nn.Module): Trained model to be evaluated.
        val_set (DataLoader): Validation set.
        verbose (bool): Whether to print results.

    Returns:
        float: Test accuracy percentage.
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            images = images.view(images.shape[0], -1)

            output = model(images)
            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    if verbose:
        print(f"\nFinal Test Accuracy (Full model with A/B): {accuracy:.2f}%")

        # Print C-only accuracy for comparison
        c_only_accuracy = validate_c_only(model, val_set)
        print(f"Final Test Accuracy (C-only, no A/B): {c_only_accuracy:.2f}%")

        # Print final LRTT statistics
        print("\nFinal LRTT Statistics:")
        layer_idx = 0
        for layer in model:
            if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
                controller = layer.analog_module.controller
                layer_idx += 1
                print(f"  Layer {layer_idx}: "
                      f"A updates={controller.num_a_updates}, "
                      f"B updates={controller.num_b_updates}, "
                      f"Transfers={controller.num_transfers}")

    return accuracy


def print_6t1c_info():
    """Print 6T1C device information."""
    import math
    TAU_SEC = 46505.0

    if INCLUDE_RETENTION and DT_BATCH_SEC > 0:
        delta = 1 - math.exp(-DT_BATCH_SEC / TAU_SEC)
        lifetime = 1.0 / delta
    else:
        lifetime = float('inf')

    print("\n" + "=" * 60)
    print("6T1C Device Parameters (A/B tiles)")
    print("=" * 60)
    print("  Update characteristics:")
    print("    dw_min:      0.001981")
    print("    gamma_up:    -0.1678 (slight saturation)")
    print("    gamma_down:  +0.1410 (near-linear)")
    print("  Retention characteristics:")
    print(f"    Physical tau: 775.1 min (46505 sec)")
    print(f"    dt_batch:     {DT_BATCH_SEC} sec")
    print(f"    lifetime:     {lifetime:.0f}" if lifetime != float('inf') else "    lifetime:     inf (no retention)")
    print("  C tile: IdealizedPresetDevice (no non-idealities)")
    print("=" * 60)


def main():
    """Train a PyTorch analog model with LRTT (6T1C A/B) to classify MNIST."""

    print_6t1c_info()

    # Load datasets.
    train_data, validation_data = load_images()
    print(f"\nDataset loaded: {len(train_data.dataset)} train, {len(validation_data.dataset)} test")

    # Prepare the model
    model = create_analog_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # Train the model
    train(model, train_data, validation_data)

    # Final evaluation
    test_evaluation(model, validation_data, verbose=True)


if __name__ == "__main__":
    main()

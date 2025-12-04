# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""MNIST training comparing LRTT reinit modes with 6T1C A/B tiles.

Compares two reinit modes with decay_factor=1.0:
1) "decay" mode: A, B both preserved after transfer (A*=1, B*=1)
2) "hybrid" mode: A=0, B preserved after transfer (A=0, B*=1)

Both use:
- 6T1C devices for A/B tiles
- IdealizedPresetDevice for C tile
- transfer_every = 100
- rank = 8
"""
# pylint: disable=invalid-name, redefined-outer-name

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
LRTT_RANK = 8
TRANSFER_EVERY = 100  # Transfer rate
LORA_ALPHA = 4.0
DT_BATCH_SEC = 1.0
DECAY_FACTOR = 1.0  # Keep A, B values after transfer


def load_images():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data


def create_6t1c_lrtt_config(rank, reinit_mode):
    """Create LRTT configuration with specified reinit mode.

    Args:
        rank: LRTT rank
        reinit_mode: "decay" or "hybrid"

    Returns:
        PythonLRTTRPUConfig
    """
    device_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=rank,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        dt_batch_sec=DT_BATCH_SEC
    )

    # Set reinit mode and decay factor
    device_config.reinit_mode = reinit_mode
    device_config.decay_factor = DECAY_FACTOR
    device_config.reinit_gain = 0.5
    device_config.correct_gradient_magnitudes = True
    device_config.transfer_lr = LORA_ALPHA

    return PythonLRTTRPUConfig(device=device_config)


def create_model(reinit_mode):
    """Create model with specified reinit mode.

    Args:
        reinit_mode: "decay" or "hybrid"

    Returns:
        nn.Module
    """
    model = AnalogSequential(
        AnalogLinear(
            INPUT_SIZE,
            HIDDEN_SIZES[0],
            bias=False,
            rpu_config=create_6t1c_lrtt_config(LRTT_RANK, reinit_mode),
        ),
        nn.ReLU(),
        AnalogLinear(
            HIDDEN_SIZES[0],
            HIDDEN_SIZES[1],
            bias=False,
            rpu_config=create_6t1c_lrtt_config(LRTT_RANK, reinit_mode),
        ),
        nn.ReLU(),
        AnalogLinear(
            HIDDEN_SIZES[1],
            OUTPUT_SIZE,
            bias=False,
            rpu_config=IdealizedPreset(),
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    return model


def train_epoch(model, train_loader, optimizer, classifier):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE).view(images.shape[0], -1)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        output = model(images)
        loss = classifier(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader):
    """Validate model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE).view(images.shape[0], -1)
            labels = labels.to(DEVICE)

            output = model(images)
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def validate_c_only(model, val_loader):
    """Validate using only C tiles."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE).view(images.shape[0], -1)
            labels = labels.to(DEVICE)

            x = images
            for layer in model:
                if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
                    x = layer.analog_module.controller.tile_c.forward(x)
                elif isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
                elif isinstance(layer, nn.LogSoftmax):
                    x = torch.log_softmax(x, dim=1)
                elif hasattr(layer, 'analog_module'):
                    x = layer(x)

            pred = x.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def get_lrtt_stats(model):
    """Get LRTT statistics from model."""
    stats = []
    layer_idx = 0
    for layer in model:
        if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
            ctrl = layer.analog_module.controller
            stats.append({
                'layer': layer_idx,
                'a_updates': ctrl.num_a_updates,
                'b_updates': ctrl.num_b_updates,
                'transfers': ctrl.num_transfers
            })
            layer_idx += 1
    return stats


def train_model(reinit_mode, train_loader, val_loader):
    """Train model with specified reinit mode.

    Args:
        reinit_mode: "decay" or "hybrid"
        train_loader: Training data
        val_loader: Validation data

    Returns:
        dict: Training history
    """
    print(f"\n{'='*60}")
    print(f"Training with reinit_mode='{reinit_mode}', decay_factor={DECAY_FACTOR}")
    print(f"{'='*60}")

    if reinit_mode == "decay":
        print("  After transfer: A *= 1.0 (keep), B *= 1.0 (keep)")
    else:  # hybrid
        print("  After transfer: A = 0, B *= 1.0 (keep)")

    model = create_model(reinit_mode)
    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    classifier = nn.NLLLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_acc_c_only': [],
        'transfers': []
    }

    time_start = time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, classifier)
        val_acc = validate(model, val_loader)
        val_acc_c_only = validate_c_only(model, val_loader)

        scheduler.step()

        # Get transfer count
        stats = get_lrtt_stats(model)
        total_transfers = sum(s['transfers'] for s in stats)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_acc_c_only'].append(val_acc_c_only)
        history['transfers'].append(total_transfers)

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: "
              f"Loss={train_loss:.4f}, "
              f"Train={train_acc:.2f}%, "
              f"Val={val_acc:.2f}%, "
              f"Val(C)={val_acc_c_only:.2f}%, "
              f"Transfers={total_transfers}")

    elapsed = time() - time_start
    print(f"Training time: {elapsed/60:.2f} min")

    # Final stats
    print(f"\nFinal Results ({reinit_mode}):")
    print(f"  Best Val Acc: {max(history['val_acc']):.2f}%")
    print(f"  Best Val Acc (C-only): {max(history['val_acc_c_only']):.2f}%")
    print(f"  Final Val Acc: {history['val_acc'][-1]:.2f}%")
    print(f"  Total Transfers: {total_transfers}")

    return history


def compare_results(history_decay, history_hybrid):
    """Compare and print results."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nSettings: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, decay_factor={DECAY_FACTOR}")
    print("\n" + "-"*70)
    print(f"{'Metric':<25} {'Decay (A,B keep)':<20} {'Hybrid (A=0, B keep)':<20}")
    print("-"*70)

    metrics = [
        ('Best Val Acc (%)', max(history_decay['val_acc']), max(history_hybrid['val_acc'])),
        ('Final Val Acc (%)', history_decay['val_acc'][-1], history_hybrid['val_acc'][-1]),
        ('Best Val Acc C-only (%)', max(history_decay['val_acc_c_only']), max(history_hybrid['val_acc_c_only'])),
        ('Final Val Acc C-only (%)', history_decay['val_acc_c_only'][-1], history_hybrid['val_acc_c_only'][-1]),
        ('Final Train Acc (%)', history_decay['train_acc'][-1], history_hybrid['train_acc'][-1]),
        ('Final Loss', history_decay['train_loss'][-1], history_hybrid['train_loss'][-1]),
        ('Total Transfers', history_decay['transfers'][-1], history_hybrid['transfers'][-1]),
    ]

    for name, decay_val, hybrid_val in metrics:
        if isinstance(decay_val, float):
            print(f"{name:<25} {decay_val:<20.2f} {hybrid_val:<20.2f}")
        else:
            print(f"{name:<25} {decay_val:<20} {hybrid_val:<20}")

    print("-"*70)

    # Determine winner
    decay_best = max(history_decay['val_acc'])
    hybrid_best = max(history_hybrid['val_acc'])

    if decay_best > hybrid_best:
        winner = "Decay (A,B keep)"
        diff = decay_best - hybrid_best
    elif hybrid_best > decay_best:
        winner = "Hybrid (A=0, B keep)"
        diff = hybrid_best - decay_best
    else:
        winner = "Tie"
        diff = 0

    print(f"\nWinner: {winner} (diff: {diff:.2f}%)")
    print("="*70)


def main():
    print("="*70)
    print("LRTT Reinit Mode Comparison: Decay vs Hybrid")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  LRTT Rank: {LRTT_RANK}")
    print(f"  Transfer Every: {TRANSFER_EVERY}")
    print(f"  Decay Factor: {DECAY_FACTOR}")
    print(f"  LoRA Alpha: {LORA_ALPHA}")
    print(f"  A/B tiles: 6T1C")
    print(f"  C tile: IdealizedPresetDevice")
    print(f"  Output layer: IdealizedPresetDevice")
    print(f"\nReinit modes being compared:")
    print(f"  1) 'decay': After transfer, A *= {DECAY_FACTOR}, B *= {DECAY_FACTOR} (both preserved)")
    print(f"  2) 'hybrid': After transfer, A = 0, B *= {DECAY_FACTOR} (only B preserved)")

    # Load data
    train_loader, val_loader = load_images()
    print(f"\nDataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} test")

    # Train with decay mode (A, B both preserved)
    history_decay = train_model("decay", train_loader, val_loader)

    # Train with hybrid mode (A=0, B preserved)
    history_hybrid = train_model("hybrid", train_loader, val_loader)

    # Compare results
    compare_results(history_decay, history_hybrid)


if __name__ == "__main__":
    main()

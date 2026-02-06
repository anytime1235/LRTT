#!/usr/bin/env python3
"""MNIST training example using LoRA Merge with Analog C Tile.

This script demonstrates LRTT with:
- Digital A, B tiles (FloatingPointDevice)
- Analog C tile (SoftBoundsDevice, noise=0)
- Exact transfer using "set" method

Model architecture:
- AnalogLinear(784, 256): LRTT layer with digital A,B and analog C
- ReLU
- AnalogLinear(256, 10): FloatingPointRPUConfig (digital final layer)
- LogSoftmax

Training configuration:
- Optimizer: AnalogSGD
- Scheduler: StepLR (step_size=10, gamma=0.5)
- Epochs: 30
- Batch size: 64
"""

import os
os.environ["LRTT_SILENT"] = "1"  # Suppress LRTT debug output

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig

# Import our custom config
from config import create_lora_merge_config, create_lora_merge_config_decay


# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5

# LRTT parameters (higher rank for FloatingPoint A,B + forward_inject=True)
RANK = 64
TRANSFER_EVERY = 50
TRANSFER_LR = 0.1
LORA_ALPHA = 1.0
LEARNING_RATE = 0.05


def load_data():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('/tmp/mnist', download=True, train=True, transform=transform)
    val_set = datasets.MNIST('/tmp/mnist', download=True, train=False, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def create_model():
    """Create model with LRTT first layer and digital final layer.

    Architecture:
    - Layer 1: AnalogLinear(784, 256) with LRTT (digital A,B + analog C)
    - ReLU
    - Layer 2: AnalogLinear(256, 10) with FloatingPointRPUConfig (digital)
    - LogSoftmax

    Returns:
        AnalogSequential model
    """
    # Create LRTT config for first layer (decay mode with factor=1.0, like sweep)
    lrtt_config = create_lora_merge_config_decay(
        rank=RANK,
        transfer_every=TRANSFER_EVERY,
        transfer_lr=TRANSFER_LR,
        lora_alpha=LORA_ALPHA,
        decay_factor=1.0,  # No reinit, like sweep
    )

    model = AnalogSequential(
        # First layer: LRTT with digital A,B and analog C
        AnalogLinear(784, 256, bias=True, rpu_config=lrtt_config),
        nn.ReLU(),
        # Final layer: Digital (FloatingPointRPUConfig)
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    )

    model.to(DEVICE)
    return model


def verify_tile_types(model):
    """Verify that tiles are created with correct types.

    Expected:
    - A, B tiles: FloatingPointTile
    - C tile: AnalogTile (SoftBoundsDevice)
    """
    print("\n" + "=" * 60)
    print("Tile Type Verification")
    print("=" * 60)

    first_layer = model[0]
    if hasattr(first_layer, 'analog_module'):
        tile = first_layer.analog_module

        # Check individual tiles
        tile_a_type = type(tile.tile_a).__name__
        tile_b_type = type(tile.tile_b).__name__
        tile_c_type = type(tile.tile_c).__name__

        print(f"A tile type: {tile_a_type}")
        print(f"B tile type: {tile_b_type}")
        print(f"C tile type: {tile_c_type}")

        # Verify expected types
        expected_ab = "FloatingPointTile"
        expected_c = "AnalogTile"

        a_ok = expected_ab in tile_a_type
        b_ok = expected_ab in tile_b_type
        c_ok = expected_c in tile_c_type

        print(f"\nVerification:")
        print(f"  A tile is FloatingPoint: {'PASS' if a_ok else 'FAIL'}")
        print(f"  B tile is FloatingPoint: {'PASS' if b_ok else 'FAIL'}")
        print(f"  C tile is Analog: {'PASS' if c_ok else 'FAIL'}")

        # Check LRTT controller settings
        controller = tile.controller
        print(f"\nLRTT Controller Settings:")
        print(f"  rank: {controller.rank}")
        print(f"  transfer_every: {controller.transfer_every}")
        print(f"  transfer_lr: {controller.transfer_lr}")
        print(f"  transfer_method: {controller.transfer_method}")
        print(f"  update_mode: {controller.update_mode}")
        print(f"  forward_inject: {controller.forward_inject_enabled}")
        print(f"  reinit_mode: {controller.reinit_mode}")

    print("=" * 60 + "\n")


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
        target = target.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader):
    """Validate model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


def get_lrtt_stats(model):
    """Get LRTT statistics from first layer."""
    first_layer = model[0]
    if hasattr(first_layer, 'analog_module'):
        tile = first_layer.analog_module
        controller = tile.controller
        return {
            'num_a_updates': controller.num_a_updates,
            'num_b_updates': controller.num_b_updates,
            'num_transfers': controller.num_transfers,
            'transfer_counter': controller.transfer_counter,
        }
    return None


def main():
    """Main training loop."""
    print("=" * 60)
    print("LoRA Merge with Analog C Tile - MNIST Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Rank: {RANK}")
    print(f"Transfer every: {TRANSFER_EVERY}")
    print(f"Transfer LR: {TRANSFER_LR}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)

    # Load data
    train_loader, val_loader = load_data()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = create_model()
    print(f"\nModel created:")
    print(model)

    # Verify tile types
    verify_tile_types(model)

    # Setup optimizer and scheduler
    optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    # Training loop
    best_acc = 0.0
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 60)

    start_time = time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)

        # Validate
        val_acc = validate(model, val_loader)

        # Get LRTT stats
        stats = get_lrtt_stats(model)

        epoch_time = time() - epoch_start

        # Print progress
        print(f"Epoch {epoch:02d}/{EPOCHS}: "
              f"Loss={train_loss:.4f}, "
              f"Train={train_acc:.2f}%, "
              f"Val={val_acc:.2f}%, "
              f"LR={scheduler.get_last_lr()[0]:.4f}, "
              f"Time={epoch_time:.1f}s")

        if stats:
            print(f"  LRTT: A={stats['num_a_updates']}, "
                  f"B={stats['num_b_updates']}, "
                  f"Transfers={stats['num_transfers']}")

        # Learning rate scheduling
        scheduler.step()

        # Early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= 5 and best_acc < 50.0:
            print(f"\nEarly stopping: accuracy too low ({best_acc:.2f}%)")
            break

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    total_time = time() - start_time

    # Final results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Total training time: {total_time/60:.2f} minutes")

    # Final LRTT statistics
    stats = get_lrtt_stats(model)
    if stats:
        print(f"\nFinal LRTT Statistics:")
        print(f"  Total A updates: {stats['num_a_updates']}")
        print(f"  Total B updates: {stats['num_b_updates']}")
        print(f"  Total transfers: {stats['num_transfers']}")

    # Verify target accuracy
    if best_acc >= 97.0:
        print(f"\nSUCCESS: Target accuracy (97%+) achieved!")
    else:
        print(f"\nNote: Best accuracy {best_acc:.2f}% is below 97% target")
        print("Consider adjusting hyperparameters: rank, transfer_every, transfer_lr, learning_rate")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Hyperparameter sweep for optimal reinit_gain in LRTT 6T1C training.

Sweep parameters:
- rank: 32 (fixed)
- transfer_lr: [0.1, 1.0]
- transfer_every: [1, 10, 100]
- reinit_gain: [0.01, 0.1, 0.5, 1.0]
- epochs: 30

Total: 2 × 3 × 4 = 24 experiments
"""

import os
import sys
import csv
import itertools
from time import time
from datetime import datetime

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.rpu_base import cuda

# Device setup
USE_CUDA = 1 if cuda.is_compiled() else 0
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Fixed parameters
PATH_DATASET = os.path.join("data", "DATASET")
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
EPOCHS = 30
BATCH_SIZE = 64
LRTT_RANK = 32
LORA_ALPHA = 1.0

# Sweep parameters
TRANSFER_LR_LIST = [0.1, 1.0]
TRANSFER_EVERY_LIST = [1, 10, 100]
REINIT_GAIN_LIST = [0.01, 0.1, 0.5, 1.0]

# Results file
RESULTS_FILE = "sweep_reinit_gain_results.csv"


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


def create_lrtt_config(transfer_lr, transfer_every, reinit_gain):
    """Create LRTT configuration with given parameters."""
    device_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=LRTT_RANK,
        transfer_every=transfer_every,
        lora_alpha=LORA_ALPHA,
        dt_batch_sec=1.0
    )
    device_config.reinit_gain = reinit_gain
    device_config.correct_gradient_magnitudes = True
    device_config.transfer_lr = transfer_lr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.reinit_mode = "standard"

    return PythonLRTTRPUConfig(device=device_config)


def create_model(transfer_lr, transfer_every, reinit_gain):
    """Create analog network with given LRTT parameters."""
    model = AnalogSequential(
        AnalogLinear(
            INPUT_SIZE, HIDDEN_SIZE, bias=False,
            rpu_config=create_lrtt_config(transfer_lr, transfer_every, reinit_gain),
        ),
        nn.ReLU(),
        AnalogLinear(
            HIDDEN_SIZE, OUTPUT_SIZE, bias=True,
            rpu_config=FloatingPointRPUConfig(),
        ),
        nn.LogSoftmax(dim=1),
    )
    if USE_CUDA:
        model.cuda()
    return model


def validate(model, val_set):
    """Evaluate model accuracy."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE).view(images.shape[0], -1)
            labels = labels.to(DEVICE)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total


def validate_c_only(model, val_set):
    """Evaluate using C tile only."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE).view(images.shape[0], -1)
            labels = labels.to(DEVICE)
            x = images
            for layer in model:
                if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
                    controller = layer.analog_module.controller
                    x = controller.tile_c.forward(x)
                elif isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
                elif isinstance(layer, nn.LogSoftmax):
                    x = torch.log_softmax(x, dim=1)
                elif hasattr(layer, 'analog_module'):
                    x = layer(x)
            _, predicted = torch.max(x.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total


def train_and_evaluate(transfer_lr, transfer_every, reinit_gain, train_data, val_data):
    """Train model and return results."""
    model = create_model(transfer_lr, transfer_every, reinit_gain)
    classifier = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0
    best_c_only_acc = 0
    final_val_acc = 0
    final_c_only_acc = 0

    time_start = time()

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_data:
            images = images.to(DEVICE).view(images.shape[0], -1)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = classifier(output, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        val_acc = validate(model, val_data)
        c_only_acc = validate_c_only(model, val_data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if c_only_acc > best_c_only_acc:
            best_c_only_acc = c_only_acc

        final_val_acc = val_acc
        final_c_only_acc = c_only_acc

        # Progress print
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}: Val={val_acc:.2f}%, C-only={c_only_acc:.2f}%")

    train_time = time() - time_start

    # Clean up
    del model
    torch.cuda.empty_cache() if USE_CUDA else None

    return {
        'best_val_acc': best_val_acc,
        'best_c_only_acc': best_c_only_acc,
        'final_val_acc': final_val_acc,
        'final_c_only_acc': final_c_only_acc,
        'train_time_sec': train_time
    }


def main():
    print("=" * 70)
    print("LRTT 6T1C Hyperparameter Sweep for Optimal reinit_gain")
    print("=" * 70)
    print(f"Fixed: rank={LRTT_RANK}, epochs={EPOCHS}, batch_size={BATCH_SIZE}")
    print(f"Sweep: transfer_lr={TRANSFER_LR_LIST}")
    print(f"       transfer_every={TRANSFER_EVERY_LIST}")
    print(f"       reinit_gain={REINIT_GAIN_LIST}")
    print(f"Total experiments: {len(TRANSFER_LR_LIST) * len(TRANSFER_EVERY_LIST) * len(REINIT_GAIN_LIST)}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load dataset once
    print("\nLoading MNIST dataset...")
    train_data, val_data = load_images()
    print(f"Dataset loaded: {len(train_data.dataset)} train, {len(val_data.dataset)} test")

    # Prepare results file
    results = []
    fieldnames = [
        'transfer_lr', 'transfer_every', 'reinit_gain',
        'best_val_acc', 'best_c_only_acc',
        'final_val_acc', 'final_c_only_acc',
        'train_time_sec'
    ]

    # Run sweep
    all_combos = list(itertools.product(TRANSFER_LR_LIST, TRANSFER_EVERY_LIST, REINIT_GAIN_LIST))
    total_start = time()

    for idx, (transfer_lr, transfer_every, reinit_gain) in enumerate(all_combos, 1):
        print(f"\n[{idx}/{len(all_combos)}] transfer_lr={transfer_lr}, transfer_every={transfer_every}, reinit_gain={reinit_gain}")
        print("-" * 50)

        result = train_and_evaluate(transfer_lr, transfer_every, reinit_gain, train_data, val_data)
        result['transfer_lr'] = transfer_lr
        result['transfer_every'] = transfer_every
        result['reinit_gain'] = reinit_gain
        results.append(result)

        print(f"  => Best Val: {result['best_val_acc']:.2f}%, Best C-only: {result['best_c_only_acc']:.2f}%")
        print(f"  => Final Val: {result['final_val_acc']:.2f}%, Final C-only: {result['final_c_only_acc']:.2f}%")
        print(f"  => Time: {result['train_time_sec']:.1f}s")

        # Save intermediate results
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    total_time = time() - total_start

    # Print summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE - RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_FILE}")

    # Find best configuration
    print("\n" + "-" * 70)
    print("TOP 5 CONFIGURATIONS (by best_val_acc):")
    print("-" * 70)
    sorted_by_val = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
    for i, r in enumerate(sorted_by_val[:5], 1):
        print(f"{i}. transfer_lr={r['transfer_lr']}, transfer_every={r['transfer_every']}, "
              f"reinit_gain={r['reinit_gain']}")
        print(f"   Best Val: {r['best_val_acc']:.2f}%, Best C-only: {r['best_c_only_acc']:.2f}%")

    print("\n" + "-" * 70)
    print("TOP 5 CONFIGURATIONS (by best_c_only_acc):")
    print("-" * 70)
    sorted_by_c = sorted(results, key=lambda x: x['best_c_only_acc'], reverse=True)
    for i, r in enumerate(sorted_by_c[:5], 1):
        print(f"{i}. transfer_lr={r['transfer_lr']}, transfer_every={r['transfer_every']}, "
              f"reinit_gain={r['reinit_gain']}")
        print(f"   Best Val: {r['best_val_acc']:.2f}%, Best C-only: {r['best_c_only_acc']:.2f}%")

    # Summary by reinit_gain
    print("\n" + "-" * 70)
    print("AVERAGE PERFORMANCE BY reinit_gain:")
    print("-" * 70)
    for rg in REINIT_GAIN_LIST:
        rg_results = [r for r in results if r['reinit_gain'] == rg]
        avg_val = sum(r['best_val_acc'] for r in rg_results) / len(rg_results)
        avg_c = sum(r['best_c_only_acc'] for r in rg_results) / len(rg_results)
        print(f"  reinit_gain={rg}: Avg Val={avg_val:.2f}%, Avg C-only={avg_c:.2f}%")

    print("\n" + "=" * 70)
    best = sorted_by_val[0]
    print(f"OPTIMAL CONFIG: transfer_lr={best['transfer_lr']}, "
          f"transfer_every={best['transfer_every']}, reinit_gain={best['reinit_gain']}")
    print(f"                Best Val={best['best_val_acc']:.2f}%, Best C-only={best['best_c_only_acc']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

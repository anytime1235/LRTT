# -*- coding: utf-8 -*-
"""Sweep reinit_gain values to find optimal setting."""

import os
import sys
from time import time

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import lrtt_sixt1c_ab_ideal_config
from aihwkit.simulator.presets.configs import IdealizedPreset
from aihwkit.simulator.rpu_base import cuda

# Suppress verbose LRTT logs
import logging
logging.getLogger().setLevel(logging.WARNING)

# Check device
USE_CUDA = cuda.is_compiled()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# Dataset path
PATH_DATASET = os.path.join("data", "DATASET")

# Network
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training
EPOCHS = 15
BATCH_SIZE = 64

# LRTT base parameters (from previous best settings)
LRTT_RANK = 32
TRANSFER_EVERY = 100
LORA_ALPHA = 1.0
TRANSFER_LR = 0.1
LR = 0.1

# reinit_gain values to sweep
REINIT_GAINS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def load_images():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_data, val_data


def create_model(reinit_gain, update_mode="lora", reinit_mode="standard"):
    """Create model with specified reinit_gain."""
    def make_config():
        config = lrtt_sixt1c_ab_ideal_config(
            rank=LRTT_RANK,
            transfer_every=TRANSFER_EVERY,
            lora_alpha=LORA_ALPHA,
            dt_batch_sec=1.0
        )
        config.device.update_mode = update_mode
        config.device.reinit_mode = reinit_mode
        config.device.reinit_gain = reinit_gain
        config.device.correct_gradient_magnitudes = True
        config.device.transfer_lr = TRANSFER_LR
        config.device.forward_inject = False
        return config

    model = AnalogSequential(
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], bias=False, rpu_config=make_config()),
        nn.ReLU(),
        AnalogLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=False, rpu_config=make_config()),
        nn.ReLU(),
        AnalogLinear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=False, rpu_config=IdealizedPreset()),
        nn.LogSoftmax(dim=1),
    )
    if USE_CUDA:
        model.cuda()
    return model


def validate(model, val_data):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_data:
            images = images.to(DEVICE).view(-1, INPUT_SIZE)
            labels = labels.to(DEVICE)
            output = model(images)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_model(model, train_data, val_data, reinit_gain):
    """Train and return results."""
    classifier = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val = 0
    best_epoch = 0
    val_history = []

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_data:
            images = images.to(DEVICE).view(-1, INPUT_SIZE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = classifier(model(images), labels)
            loss.backward()
            optimizer.step()

        val_acc = validate(model, val_data)
        val_history.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1

        scheduler.step()

    return best_val, best_epoch, val_history


def verify_reinit_gain(reinit_gain):
    """Verify that reinit_gain is actually being used."""
    import math

    config = lrtt_sixt1c_ab_ideal_config(rank=LRTT_RANK)
    config.device.reinit_gain = reinit_gain
    config.device.reinit_mode = "standard"
    config.device.forward_inject = False

    layer = AnalogLinear(784, 256, bias=False, rpu_config=config)
    if USE_CUDA:
        layer.cuda()

    ctrl = layer.analog_module.controller

    # Force initialization by doing a forward pass
    x = torch.randn(1, 784, device=DEVICE)
    _ = layer(x)

    # Get B weights and check std
    B_weights = ctrl.tile_b.get_weights()[0]
    B_std_actual = B_weights.std().item()
    B_std_expected = reinit_gain * math.sqrt(2.0 / 784)

    return B_std_actual, B_std_expected


def main():
    print("="*70)
    print("Reinit Gain Sweep Experiment")
    print("="*70)
    print(f"Settings: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, lr={LR}")
    print(f"update_mode=lora, reinit_mode=standard, forward_inject=False")
    print(f"EPOCHS={EPOCHS}")
    print("="*70)

    # Verify reinit_gain is working
    print("\n--- Verifying reinit_gain functionality ---")
    print(f"{'reinit_gain':>12} {'B_std_actual':>14} {'B_std_expected':>16} {'Match':>8}")
    print("-"*54)
    for gain in [0.1, 1.0, 10.0]:
        actual, expected = verify_reinit_gain(gain)
        match = "OK" if abs(actual - expected) / expected < 0.3 else "MISMATCH"
        print(f"{gain:>12.2f} {actual:>14.6f} {expected:>16.6f} {match:>8}")

    # Load data
    print("\n--- Loading data ---")
    train_data, val_data = load_images()
    print(f"Train: {len(train_data.dataset)}, Val: {len(val_data.dataset)}")

    # Sweep reinit_gain
    print("\n--- Running experiments ---")
    results = {}

    for gain in REINIT_GAINS:
        print(f"\nreinit_gain = {gain}")
        model = create_model(reinit_gain=gain)

        start = time()
        best_val, best_epoch, history = train_model(model, train_data, val_data, gain)
        elapsed = time() - start

        results[gain] = {
            'best_val': best_val,
            'best_epoch': best_epoch,
            'final_val': history[-1],
            'history': history
        }

        print(f"  Best: {best_val:.2f}% (epoch {best_epoch}), Final: {history[-1]:.2f}%, Time: {elapsed:.1f}s")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'reinit_gain':>12} {'Best Val%':>12} {'Best Epoch':>12} {'Final Val%':>12}")
    print("-"*50)

    best_gain = None
    best_acc = 0
    for gain in REINIT_GAINS:
        r = results[gain]
        print(f"{gain:>12.2f} {r['best_val']:>12.2f} {r['best_epoch']:>12} {r['final_val']:>12.2f}")
        if r['best_val'] > best_acc:
            best_acc = r['best_val']
            best_gain = gain

    print("-"*50)
    print(f"OPTIMAL: reinit_gain={best_gain} with {best_acc:.2f}% accuracy")
    print("="*70)

    # Epoch-by-epoch comparison
    print("\n--- Epoch-by-epoch Val Accuracy ---")
    header = "Epoch |" + "|".join([f" {g:>6}" for g in REINIT_GAINS])
    print(header)
    print("-"*len(header))
    for ep in range(EPOCHS):
        row = f"  {ep+1:>2}  |"
        for gain in REINIT_GAINS:
            row += f" {results[gain]['history'][ep]:>6.2f}"
        print(row)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Compare LRTT update modes: lora vs reconstruction (orthogonal reinit)."""

import os
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

# Training parameters
EPOCHS = 30
BATCH_SIZE = 64

# LRTT parameters
LRTT_RANK = 32
TRANSFER_EVERY = 100
LORA_ALPHA = 1.0
TRANSFER_LR = 0.1
DT_BATCH_SEC = 1.0


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


def create_lrtt_config(rank, update_mode="lora", reinit_mode="standard"):
    """Create LRTT configuration with specified update mode.

    Args:
        rank: LRTT rank
        update_mode: 'lora' or 'reconstruction'
        reinit_mode: 'standard', 'decay', 'hybrid', or 'orthogonal'
    """
    # Use the factory function that returns PythonLRTTRPUConfig
    rpu_config = lrtt_sixt1c_ab_ideal_config(
        rank=rank,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        dt_batch_sec=DT_BATCH_SEC
    )

    # Modify update_mode and reinit_mode via device settings
    rpu_config.device.update_mode = update_mode
    rpu_config.device.reinit_mode = reinit_mode
    rpu_config.device.reinit_gain = 0.5
    rpu_config.device.correct_gradient_magnitudes = True
    rpu_config.device.transfer_lr = TRANSFER_LR
    rpu_config.device.forward_inject = False  # Disable A*B in forward pass

    return rpu_config


def create_model(update_mode="lora", reinit_mode="standard"):
    """Create analog network with specified update mode."""
    model = AnalogSequential(
        AnalogLinear(
            INPUT_SIZE, HIDDEN_SIZES[0], bias=False,
            rpu_config=create_lrtt_config(LRTT_RANK, update_mode, reinit_mode),
        ),
        nn.ReLU(),
        AnalogLinear(
            HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=False,
            rpu_config=create_lrtt_config(LRTT_RANK, update_mode, reinit_mode),
        ),
        nn.ReLU(),
        AnalogLinear(
            HIDDEN_SIZES[1], OUTPUT_SIZE, bias=False,
            rpu_config=IdealizedPreset(),
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()
    return model


def validate(model, val_set):
    """Validate model."""
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
    """Validate using only C tiles."""
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


def train_and_evaluate(model, train_data, val_data, mode_name):
    """Train model and collect results."""
    classifier = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    results = {
        'mode': mode_name,
        'train_acc': [],
        'val_acc': [],
        'val_c_only': [],
        'loss': []
    }

    print(f"\n{'='*60}")
    print(f"Training: {mode_name}")
    print(f"{'='*60}")

    time_init = time()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        for images, labels in train_data:
            images = images.to(DEVICE).view(images.shape[0], -1)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            output = model(images)
            loss = classifier(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += images.size(0)

        train_acc = 100. * correct / total
        val_acc = validate(model, val_data)
        val_c = validate_c_only(model, val_data)
        avg_loss = total_loss / len(train_data)

        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        results['val_c_only'].append(val_c)
        results['loss'].append(avg_loss)

        scheduler.step()

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: Loss={avg_loss:.4f}, "
              f"Train={train_acc:.2f}%, Val={val_acc:.2f}%, C-only={val_c:.2f}%")

    results['total_time'] = time() - time_init
    print(f"Training time: {results['total_time']:.1f}s")

    return results


def main():
    print("Loading MNIST dataset...")
    train_data, val_data = load_images()
    print(f"Dataset: {len(train_data.dataset)} train, {len(val_data.dataset)} test\n")

    # Experiment 1: LoRA mode (default)
    print("\n" + "="*70)
    print("EXPERIMENT 1: LoRA update mode (standard reinit)")
    print("="*70)
    model_lora = create_model(update_mode="lora", reinit_mode="standard")
    results_lora = train_and_evaluate(model_lora, train_data, val_data, "LoRA (standard reinit)")

    # Experiment 2: Reconstruction mode + orthogonal reinit
    print("\n" + "="*70)
    print("EXPERIMENT 2: Reconstruction update mode + orthogonal reinit")
    print("="*70)
    model_recon = create_model(update_mode="reconstruction", reinit_mode="orthogonal")
    results_recon = train_and_evaluate(model_recon, train_data, val_data, "Reconstruction (orthogonal reinit)")

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<25} {'LoRA':>15} {'Reconstruction':>15}")
    print("-"*55)
    print(f"{'Final Train Acc (%)':<25} {results_lora['train_acc'][-1]:>15.2f} {results_recon['train_acc'][-1]:>15.2f}")
    print(f"{'Final Val Acc (%)':<25} {results_lora['val_acc'][-1]:>15.2f} {results_recon['val_acc'][-1]:>15.2f}")
    print(f"{'Final C-only Acc (%)':<25} {results_lora['val_c_only'][-1]:>15.2f} {results_recon['val_c_only'][-1]:>15.2f}")
    print(f"{'Final Loss':<25} {results_lora['loss'][-1]:>15.4f} {results_recon['loss'][-1]:>15.4f}")
    print(f"{'Training Time (s)':<25} {results_lora['total_time']:>15.1f} {results_recon['total_time']:>15.1f}")
    print("-"*55)

    # Best epochs
    best_lora = max(results_lora['val_acc'])
    best_recon = max(results_recon['val_acc'])
    print(f"{'Best Val Acc (%)':<25} {best_lora:>15.2f} {best_recon:>15.2f}")

    best_c_lora = max(results_lora['val_c_only'])
    best_c_recon = max(results_recon['val_c_only'])
    print(f"{'Best C-only Acc (%)':<25} {best_c_lora:>15.2f} {best_c_recon:>15.2f}")
    print("="*70)


if __name__ == "__main__":
    main()

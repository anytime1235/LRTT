# -*- coding: utf-8 -*-
"""Sweep transfer_every with standard vs orthogonal modes."""

import os
import csv
from time import time
from datetime import datetime

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
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
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 30
BATCH_SIZE = 64

# Fixed LRTT parameters
LRTT_RANK = 32
LORA_ALPHA = 1.0
REINIT_GAIN = 0.1
TRANSFER_LR = 0.1
DT_BATCH_SEC = 1.0

# Experiment parameters
TRANSFER_EVERY_VALUES = [1, 2, 5, 10, 50, 100, 1000, 10000]
MODES = ["standard", "orthogonal"]


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


def create_lrtt_config(rank, transfer_every, mode):
    """Create LRTT configuration."""
    device_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=LORA_ALPHA,
        dt_batch_sec=DT_BATCH_SEC
    )

    device_config.reinit_gain = REINIT_GAIN
    device_config.correct_gradient_magnitudes = True
    device_config.transfer_lr = TRANSFER_LR
    device_config.forward_inject = False

    if mode == "standard":
        device_config.update_mode = "lora"
        device_config.reinit_mode = "standard"
    elif mode == "orthogonal":
        device_config.update_mode = "reconstruction"
        device_config.reinit_mode = "orthogonal"

    return PythonLRTTRPUConfig(device=device_config)


def create_model(transfer_every, mode):
    """Create the neural network."""
    model = AnalogSequential(
        AnalogLinear(
            INPUT_SIZE,
            HIDDEN_SIZE,
            bias=False,
            rpu_config=create_lrtt_config(LRTT_RANK, transfer_every, mode),
        ),
        nn.ReLU(),
        AnalogLinear(
            HIDDEN_SIZE,
            OUTPUT_SIZE,
            bias=True,
            rpu_config=FloatingPointRPUConfig(),
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    return model


def validate(model, val_set):
    """Validate the model."""
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
    return 100. * correct / total


def validate_c_only(model, val_set):
    """Validate using only C tiles."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            images = images.view(images.shape[0], -1)

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


def train_model(model, train_set, val_set, verbose=False):
    """Train the model and return metrics."""
    classifier = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0
    best_val_c_only = 0

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            output = model(images)
            loss = classifier(output, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_acc = validate(model, val_set)
        val_c_only = validate_c_only(model, val_set)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_c_only > best_val_c_only:
            best_val_c_only = val_c_only

        if verbose:
            print(f"  Epoch {epoch+1}/{EPOCHS}: Val={val_acc:.2f}%, C-only={val_c_only:.2f}%")

    # Final evaluation
    final_val_acc = validate(model, val_set)
    final_c_only = validate_c_only(model, val_set)

    # Get LRTT stats
    transfers = 0
    for layer in model:
        if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
            controller = layer.analog_module.controller
            transfers = controller.num_transfers
            break

    return {
        'final_val_acc': final_val_acc,
        'final_c_only': final_c_only,
        'best_val_acc': best_val_acc,
        'best_c_only': best_val_c_only,
        'transfers': transfers
    }


def run_experiment():
    """Run the full experiment sweep."""
    # Load data once
    print("Loading dataset...")
    train_data, val_data = load_images()
    print(f"Dataset loaded: {len(train_data.dataset)} train, {len(val_data.dataset)} test\n")

    # Results storage
    results = []

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/sweep_transfer_every_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    total_experiments = len(TRANSFER_EVERY_VALUES) * len(MODES)
    exp_num = 0

    for transfer_every in TRANSFER_EVERY_VALUES:
        for mode in MODES:
            exp_num += 1
            print(f"=" * 60)
            print(f"Experiment {exp_num}/{total_experiments}")
            print(f"  transfer_every={transfer_every}, mode={mode}")
            print(f"=" * 60)

            start_time = time()

            # Create and train model
            model = create_model(transfer_every, mode)
            metrics = train_model(model, train_data, val_data, verbose=False)

            elapsed = time() - start_time

            result = {
                'transfer_every': transfer_every,
                'mode': mode,
                'final_val_acc': metrics['final_val_acc'],
                'final_c_only': metrics['final_c_only'],
                'best_val_acc': metrics['best_val_acc'],
                'best_c_only': metrics['best_c_only'],
                'transfers': metrics['transfers'],
                'time_min': elapsed / 60
            }
            results.append(result)

            print(f"  Final Val: {metrics['final_val_acc']:.2f}%")
            print(f"  Final C-only: {metrics['final_c_only']:.2f}%")
            print(f"  Best Val: {metrics['best_val_acc']:.2f}%")
            print(f"  Transfers: {metrics['transfers']}")
            print(f"  Time: {elapsed/60:.2f} min")
            print()

            # Clear GPU memory
            del model
            if USE_CUDA:
                torch.cuda.empty_cache()

    # Save results to CSV
    csv_path = os.path.join(results_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

    # Create plots
    create_plots(results, results_dir)

    return results, results_dir


def create_plots(results, results_dir):
    """Create and save plots."""
    # Separate results by mode
    standard_results = [r for r in results if r['mode'] == 'standard']
    ortho_results = [r for r in results if r['mode'] == 'orthogonal']

    # Extract data
    transfer_every_std = [r['transfer_every'] for r in standard_results]
    transfer_every_orth = [r['transfer_every'] for r in ortho_results]

    final_val_std = [r['final_val_acc'] for r in standard_results]
    final_val_orth = [r['final_val_acc'] for r in ortho_results]

    final_c_std = [r['final_c_only'] for r in standard_results]
    final_c_orth = [r['final_c_only'] for r in ortho_results]

    best_val_std = [r['best_val_acc'] for r in standard_results]
    best_val_orth = [r['best_val_acc'] for r in ortho_results]

    # Plot 1: Final Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.semilogx(transfer_every_std, final_val_std, 'bo-', label='Standard (LoRA)', linewidth=2, markersize=8)
    plt.semilogx(transfer_every_orth, final_val_orth, 'rs--', label='Orthogonal (Recon)', linewidth=2, markersize=8)
    plt.xlabel('Transfer Every (steps)', fontsize=12)
    plt.ylabel('Final Validation Accuracy (%)', fontsize=12)
    plt.title('Final Validation Accuracy vs Transfer Frequency', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_val_accuracy.png"), dpi=150)
    plt.close()

    # Plot 2: Final C-only Accuracy
    plt.figure(figsize=(10, 6))
    plt.semilogx(transfer_every_std, final_c_std, 'bo-', label='Standard (LoRA)', linewidth=2, markersize=8)
    plt.semilogx(transfer_every_orth, final_c_orth, 'rs--', label='Orthogonal (Recon)', linewidth=2, markersize=8)
    plt.xlabel('Transfer Every (steps)', fontsize=12)
    plt.ylabel('Final C-only Accuracy (%)', fontsize=12)
    plt.title('Final C-only Accuracy vs Transfer Frequency', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_c_only_accuracy.png"), dpi=150)
    plt.close()

    # Plot 3: Best Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.semilogx(transfer_every_std, best_val_std, 'bo-', label='Standard (LoRA)', linewidth=2, markersize=8)
    plt.semilogx(transfer_every_orth, best_val_orth, 'rs--', label='Orthogonal (Recon)', linewidth=2, markersize=8)
    plt.xlabel('Transfer Every (steps)', fontsize=12)
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
    plt.title('Best Validation Accuracy vs Transfer Frequency', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "best_val_accuracy.png"), dpi=150)
    plt.close()

    # Plot 4: Combined comparison (bar chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = range(len(transfer_every_std))
    width = 0.35

    # Final accuracy bar chart
    axes[0].bar([i - width/2 for i in x], final_val_std, width, label='Standard', color='blue', alpha=0.7)
    axes[0].bar([i + width/2 for i in x], final_val_orth, width, label='Orthogonal', color='red', alpha=0.7)
    axes[0].set_xlabel('Transfer Every', fontsize=12)
    axes[0].set_ylabel('Final Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Final Validation Accuracy', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(transfer_every_std, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # C-only accuracy bar chart
    axes[1].bar([i - width/2 for i in x], final_c_std, width, label='Standard', color='blue', alpha=0.7)
    axes[1].bar([i + width/2 for i in x], final_c_orth, width, label='Orthogonal', color='red', alpha=0.7)
    axes[1].set_xlabel('Transfer Every', fontsize=12)
    axes[1].set_ylabel('Final C-only Accuracy (%)', fontsize=12)
    axes[1].set_title('Final C-only Accuracy', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(transfer_every_std, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_bar_chart.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


if __name__ == "__main__":
    results, results_dir = run_experiment()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print("\nSummary:")
    print("-" * 60)
    print(f"{'transfer_every':>14} | {'Mode':>12} | {'Final Val':>10} | {'C-only':>10} | {'Best Val':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['transfer_every']:>14} | {r['mode']:>12} | {r['final_val_acc']:>9.2f}% | {r['final_c_only']:>9.2f}% | {r['best_val_acc']:>9.2f}%")

# -*- coding: utf-8 -*-
"""Comprehensive LRTT hyperparameter sweep with plotting and logging.

Sweep parameters:
- rank: [1, 4, 8, 16, 32, 64, 128]
- transfer_lr: [0.1, 1.0]
- transfer_every: [1, 10, 50, 100, 500, 1000]
- reinit_gain: [0.1, 0.5]
- modes:
  - standard reinit + lora update
  - orthogonal reinit + reconstruction update

Total: 7 × 2 × 6 × 2 × 2 = 336 experiments

Features:
- Saves accuracy plot (PNG) for each experiment
- Saves results to CSV
- Resilient to disconnection (use with nohup)
"""

import os
import sys
import csv
import json
import itertools
import argparse
from time import time
from datetime import datetime

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

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
LORA_ALPHA = 1.0

# Sweep parameters
RANK_LIST = [1, 4, 8, 16, 32, 64, 128]
TRANSFER_LR_LIST = [0.1, 1.0]
TRANSFER_EVERY_LIST = [1, 10, 50, 100, 500, 1000]
REINIT_GAIN_LIST = [0.1, 0.5]
MODE_LIST = [
    ("standard", "lora"),        # standard reinit + lora update
    ("orthogonal", "reconstruction")  # orthogonal reinit + reconstruction update
]

# Output directory
OUTPUT_DIR = "sweep_results"


def setup_output_dir():
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, f"sweep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    return output_dir


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


def create_lrtt_config(rank, transfer_lr, transfer_every, reinit_gain, reinit_mode, update_mode):
    """Create LRTT configuration with given parameters."""
    device_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=LORA_ALPHA,
        dt_batch_sec=1.0
    )
    device_config.reinit_gain = reinit_gain
    device_config.correct_gradient_magnitudes = True
    device_config.transfer_lr = transfer_lr
    device_config.forward_inject = False
    device_config.update_mode = update_mode
    device_config.reinit_mode = reinit_mode

    return PythonLRTTRPUConfig(device=device_config)


def create_model(rank, transfer_lr, transfer_every, reinit_gain, reinit_mode, update_mode):
    """Create analog network with given LRTT parameters."""
    model = AnalogSequential(
        AnalogLinear(
            INPUT_SIZE, HIDDEN_SIZE, bias=False,
            rpu_config=create_lrtt_config(rank, transfer_lr, transfer_every, reinit_gain, reinit_mode, update_mode),
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


def save_plot(history, config, output_dir):
    """Save accuracy plot as PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(history['val_acc']) + 1)
    ax.plot(epochs, history['val_acc'], 'b-', label='Val Accuracy', linewidth=2)
    ax.plot(epochs, history['c_only_acc'], 'r--', label='C-only Accuracy', linewidth=2)
    ax.plot(epochs, history['train_acc'], 'g:', label='Train Accuracy', linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f"rank={config['rank']}, tlr={config['transfer_lr']}, te={config['transfer_every']}, "
                 f"rg={config['reinit_gain']}, {config['reinit_mode']}+{config['update_mode']}", fontsize=10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Create filename
    filename = (f"r{config['rank']}_tlr{config['transfer_lr']}_te{config['transfer_every']}_"
                f"rg{config['reinit_gain']}_{config['reinit_mode']}_{config['update_mode']}.png")
    filepath = os.path.join(output_dir, "plots", filename)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    return filepath


def train_and_evaluate(config, train_data, val_data, output_dir):
    """Train model and return results with history."""
    rank = config['rank']
    transfer_lr = config['transfer_lr']
    transfer_every = config['transfer_every']
    reinit_gain = config['reinit_gain']
    reinit_mode = config['reinit_mode']
    update_mode = config['update_mode']

    model = create_model(rank, transfer_lr, transfer_every, reinit_gain, reinit_mode, update_mode)
    classifier = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    history = {
        'train_acc': [],
        'val_acc': [],
        'c_only_acc': [],
        'loss': []
    }

    best_val_acc = 0
    best_c_only_acc = 0

    time_start = time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_data:
            images = images.to(DEVICE).view(images.shape[0], -1)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = classifier(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_acc = 100. * correct / total
        val_acc = validate(model, val_data)
        c_only_acc = validate_c_only(model, val_data)
        avg_loss = total_loss / len(train_data)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['c_only_acc'].append(c_only_acc)
        history['loss'].append(avg_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if c_only_acc > best_c_only_acc:
            best_c_only_acc = c_only_acc

    train_time = time() - time_start

    # Save plot
    plot_path = save_plot(history, config, output_dir)

    # Clean up
    del model
    if USE_CUDA:
        torch.cuda.empty_cache()

    return {
        'best_val_acc': best_val_acc,
        'best_c_only_acc': best_c_only_acc,
        'final_val_acc': history['val_acc'][-1],
        'final_c_only_acc': history['c_only_acc'][-1],
        'final_train_acc': history['train_acc'][-1],
        'train_time_sec': train_time,
        'plot_path': plot_path
    }


def run_sweep(test_mode=False):
    """Run the full hyperparameter sweep."""
    output_dir = setup_output_dir()

    # Determine parameters based on mode
    if test_mode:
        rank_list = [32]
        transfer_lr_list = [0.1]
        transfer_every_list = [100]
        reinit_gain_list = [0.1]
        mode_list = [("standard", "lora")]
        print("=" * 70)
        print("TEST MODE - Running single experiment to verify setup")
        print("=" * 70)
    else:
        rank_list = RANK_LIST
        transfer_lr_list = TRANSFER_LR_LIST
        transfer_every_list = TRANSFER_EVERY_LIST
        reinit_gain_list = REINIT_GAIN_LIST
        mode_list = MODE_LIST

    all_combos = list(itertools.product(
        rank_list, transfer_lr_list, transfer_every_list,
        reinit_gain_list, mode_list
    ))

    print("=" * 70)
    print("LRTT Comprehensive Hyperparameter Sweep")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Epochs per experiment: {EPOCHS}")
    print(f"Total experiments: {len(all_combos)}")
    print(f"Parameters:")
    print(f"  - rank: {rank_list}")
    print(f"  - transfer_lr: {transfer_lr_list}")
    print(f"  - transfer_every: {transfer_every_list}")
    print(f"  - reinit_gain: {reinit_gain_list}")
    print(f"  - modes: {mode_list}")
    print("=" * 70)
    sys.stdout.flush()

    # Load dataset once
    print("\nLoading MNIST dataset...")
    train_data, val_data = load_images()
    print(f"Dataset loaded: {len(train_data.dataset)} train, {len(val_data.dataset)} test\n")
    sys.stdout.flush()

    # Prepare results
    results = []
    fieldnames = [
        'rank', 'transfer_lr', 'transfer_every', 'reinit_gain',
        'reinit_mode', 'update_mode',
        'best_val_acc', 'best_c_only_acc',
        'final_val_acc', 'final_c_only_acc', 'final_train_acc',
        'train_time_sec', 'plot_path'
    ]

    results_file = os.path.join(output_dir, "results.csv")

    # Save config
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump({
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'lora_alpha': LORA_ALPHA,
            'rank_list': rank_list,
            'transfer_lr_list': transfer_lr_list,
            'transfer_every_list': transfer_every_list,
            'reinit_gain_list': reinit_gain_list,
            'mode_list': mode_list,
            'total_experiments': len(all_combos),
            'start_time': datetime.now().isoformat()
        }, f, indent=2)

    # Run sweep
    total_start = time()

    for idx, (rank, transfer_lr, transfer_every, reinit_gain, (reinit_mode, update_mode)) in enumerate(all_combos, 1):
        config = {
            'rank': rank,
            'transfer_lr': transfer_lr,
            'transfer_every': transfer_every,
            'reinit_gain': reinit_gain,
            'reinit_mode': reinit_mode,
            'update_mode': update_mode
        }

        print(f"\n[{idx}/{len(all_combos)}] rank={rank}, tlr={transfer_lr}, te={transfer_every}, "
              f"rg={reinit_gain}, {reinit_mode}+{update_mode}")
        print("-" * 50)
        sys.stdout.flush()

        try:
            result = train_and_evaluate(config, train_data, val_data, output_dir)
            result.update(config)
            results.append(result)

            print(f"  => Best Val: {result['best_val_acc']:.2f}%, Best C-only: {result['best_c_only_acc']:.2f}%")
            print(f"  => Final Val: {result['final_val_acc']:.2f}%, Final C-only: {result['final_c_only_acc']:.2f}%")
            print(f"  => Time: {result['train_time_sec']:.1f}s, Plot: {result['plot_path']}")
            sys.stdout.flush()

            # Save intermediate results
            with open(results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        except Exception as e:
            print(f"  => ERROR: {e}")
            sys.stdout.flush()
            continue

    total_time = time() - total_start

    # Print summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {os.path.join(output_dir, 'plots')}")

    if results:
        # Find best configuration
        print("\n" + "-" * 70)
        print("TOP 5 CONFIGURATIONS (by best_val_acc):")
        print("-" * 70)
        sorted_by_val = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
        for i, r in enumerate(sorted_by_val[:5], 1):
            print(f"{i}. rank={r['rank']}, tlr={r['transfer_lr']}, te={r['transfer_every']}, "
                  f"rg={r['reinit_gain']}, {r['reinit_mode']}+{r['update_mode']}")
            print(f"   Best Val: {r['best_val_acc']:.2f}%, Best C-only: {r['best_c_only_acc']:.2f}%")

        print("\n" + "-" * 70)
        print("TOP 5 CONFIGURATIONS (by best_c_only_acc):")
        print("-" * 70)
        sorted_by_c = sorted(results, key=lambda x: x['best_c_only_acc'], reverse=True)
        for i, r in enumerate(sorted_by_c[:5], 1):
            print(f"{i}. rank={r['rank']}, tlr={r['transfer_lr']}, te={r['transfer_every']}, "
                  f"rg={r['reinit_gain']}, {r['reinit_mode']}+{r['update_mode']}")
            print(f"   Best Val: {r['best_val_acc']:.2f}%, Best C-only: {r['best_c_only_acc']:.2f}%")

        # Summary by reinit_gain
        print("\n" + "-" * 70)
        print("AVERAGE PERFORMANCE BY reinit_gain:")
        print("-" * 70)
        for rg in reinit_gain_list:
            rg_results = [r for r in results if r['reinit_gain'] == rg]
            if rg_results:
                avg_val = sum(r['best_val_acc'] for r in rg_results) / len(rg_results)
                avg_c = sum(r['best_c_only_acc'] for r in rg_results) / len(rg_results)
                print(f"  reinit_gain={rg}: Avg Val={avg_val:.2f}%, Avg C-only={avg_c:.2f}%")

        # Summary by mode
        print("\n" + "-" * 70)
        print("AVERAGE PERFORMANCE BY MODE:")
        print("-" * 70)
        for reinit_mode, update_mode in mode_list:
            mode_results = [r for r in results if r['reinit_mode'] == reinit_mode and r['update_mode'] == update_mode]
            if mode_results:
                avg_val = sum(r['best_val_acc'] for r in mode_results) / len(mode_results)
                avg_c = sum(r['best_c_only_acc'] for r in mode_results) / len(mode_results)
                print(f"  {reinit_mode}+{update_mode}: Avg Val={avg_val:.2f}%, Avg C-only={avg_c:.2f}%")

        print("\n" + "=" * 70)
        best = sorted_by_val[0]
        print(f"OPTIMAL CONFIG: rank={best['rank']}, tlr={best['transfer_lr']}, "
              f"te={best['transfer_every']}, rg={best['reinit_gain']}, "
              f"{best['reinit_mode']}+{best['update_mode']}")
        print(f"                Best Val={best['best_val_acc']:.2f}%, Best C-only={best['best_c_only_acc']:.2f}%")
        print("=" * 70)

    sys.stdout.flush()
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='LRTT Comprehensive Hyperparameter Sweep')
    parser.add_argument('--test', action='store_true', help='Run single test experiment')
    args = parser.parse_args()

    run_sweep(test_mode=args.test)


if __name__ == "__main__":
    main()

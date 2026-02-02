# -*- coding: utf-8 -*-
"""MNIST LRTT hyperparameter search for reconstruction mode.

Target: 95%+ accuracy
Results saved to: results/hyperparam_search_results.txt
"""

import os
import sys
import torch
from torch import nn
from torchvision import datasets, transforms
from time import time
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.presets.configs import IdealizedPreset
from aihwkit.simulator.rpu_base import cuda

# Device
USE_CUDA = cuda.is_compiled()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Fixed parameters
BATCH_SIZE = 64
EPOCHS = 30

# Results file
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "hyperparam_search_results.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    """Print and save to file."""
    print(msg)
    with open(RESULTS_FILE, "a") as f:
        f.write(msg + "\n")
        f.flush()


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('data/DATASET', download=True, train=True, transform=transform)
    test_set = datasets.MNIST('data/DATASET', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def create_model(rank, transfer_every, lora_alpha, transfer_lr, lr_scale="sqrt_rank",
                 use_scalar_stabilizer=False, recon_lambda=1e-3):
    def get_config():
        rpu_config = PythonLRTTPreset.sixt1c_ab_ideal(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            dt_batch_sec=0.0,
            ab_update_mode="reconstruction",
            forward_inject=False
        )
        rpu_config.device.use_onehot = True
        rpu_config.device.use_sigma_delta = False
        rpu_config.device.reinit_gain = 0.5
        rpu_config.device.correct_gradient_magnitudes = True
        rpu_config.device.transfer_lr = transfer_lr
        rpu_config.device.transfer_lr_scale = lr_scale
        rpu_config.device.recon_lambda_a = recon_lambda
        rpu_config.device.recon_lambda_b = recon_lambda
        rpu_config.device.recon_use_scalar_stabilizer = use_scalar_stabilizer
        return rpu_config

    model = AnalogSequential(
        AnalogLinear(784, 256, bias=False, rpu_config=get_config()),
        nn.ReLU(),
        AnalogLinear(256, 128, bias=False, rpu_config=get_config()),
        nn.ReLU(),
        AnalogLinear(128, 10, bias=False, rpu_config=IdealizedPreset()),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    return model


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE).view(-1, 784)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE).view(-1, 784)
            labels = labels.to(DEVICE)
            output = model(images)
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def run_experiment(config, train_loader, test_loader):
    model = create_model(
        rank=config['rank'],
        transfer_every=config['transfer_every'],
        lora_alpha=config['lora_alpha'],
        transfer_lr=config['transfer_lr'],
        lr_scale=config.get('lr_scale', 'sqrt_rank'),
        use_scalar_stabilizer=config.get('use_scalar_stabilizer', False),
        recon_lambda=config.get('recon_lambda', 1e-3)
    )

    optimizer = AnalogSGD(model.parameters(), lr=config['lr'])
    optimizer.regroup_param_groups(model)
    criterion = nn.NLLLoss()

    best_test_acc = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0:
            log(f"  Epoch {epoch+1}: Train={train_acc:.2f}%, Test={test_acc:.2f}%")

    return best_test_acc, best_epoch


def main():
    # Clear results file
    with open(RESULTS_FILE, "w") as f:
        f.write(f"MNIST LRTT Hyperparameter Search\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write("="*70 + "\n\n")

    log(f"Using device: {DEVICE}")
    log(f"Results will be saved to: {RESULTS_FILE}")
    log("="*70)
    log("MNIST LRTT Hyperparameter Search (reconstruction mode)")
    log(f"Target: 95%+ accuracy, EPOCHS={EPOCHS}")
    log("="*70)

    train_loader, test_loader = load_mnist()

    # Hyperparameter grid
    configs = [
        {'rank': 16, 'transfer_every': 100, 'lora_alpha': 4.0, 'transfer_lr': 4.0, 'lr': 0.01},
        {'rank': 16, 'transfer_every': 50, 'lora_alpha': 4.0, 'transfer_lr': 4.0, 'lr': 0.01},
        {'rank': 32, 'transfer_every': 100, 'lora_alpha': 4.0, 'transfer_lr': 4.0, 'lr': 0.01},
        {'rank': 16, 'transfer_every': 100, 'lora_alpha': 4.0, 'transfer_lr': 4.0, 'lr': 0.05},
        {'rank': 16, 'transfer_every': 32, 'lora_alpha': 4.0, 'transfer_lr': 4.0, 'lr': 0.01},
        {'rank': 16, 'transfer_every': 100, 'lora_alpha': 8.0, 'transfer_lr': 8.0, 'lr': 0.01},
        {'rank': 16, 'transfer_every': 100, 'lora_alpha': 4.0, 'transfer_lr': 4.0, 'lr': 0.01,
         'use_scalar_stabilizer': True},
        {'rank': 16, 'transfer_every': 100, 'lora_alpha': 4.0, 'transfer_lr': 1.0, 'lr': 0.01,
         'lr_scale': 'none'},
    ]

    results = []

    for i, config in enumerate(configs):
        log(f"\n--- Config {i+1}/{len(configs)} ---")
        log(f"  {config}")

        start = time()
        best_acc, best_epoch = run_experiment(config, train_loader, test_loader)
        elapsed = time() - start

        results.append({
            'config_id': i + 1,
            'config': config,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'time': elapsed
        })

        log(f"  Best: {best_acc:.2f}% @ epoch {best_epoch} ({elapsed:.1f}s)")

        # Save intermediate results
        with open(RESULTS_FILE.replace('.txt', '_intermediate.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    log("\n" + "="*70)
    log("SUMMARY")
    log("="*70)

    results.sort(key=lambda x: x['best_acc'], reverse=True)

    log(f"{'#':>3} {'Rank':>5} {'TrEvery':>8} {'Alpha':>6} {'LR':>6} {'Best Acc':>10} {'Epoch':>6}")
    log("-"*70)

    for r in results:
        c = r['config']
        log(f"{r['config_id']:>3} {c['rank']:>5} {c['transfer_every']:>8} "
            f"{c['lora_alpha']:>6.1f} {c['lr']:>6.3f} {r['best_acc']:>9.2f}% {r['best_epoch']:>6}")

    log("="*70)

    best = results[0]
    log(f"\nBest Config: {best['config']}")
    log(f"Best Accuracy: {best['best_acc']:.2f}%")
    log(f"\nCompleted: {datetime.now()}")

    # Save final results
    with open(RESULTS_FILE.replace('.txt', '_final.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

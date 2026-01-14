# -*- coding: utf-8 -*-
"""MNIST training with reconstruction mode."""

import os
import sys
import torch
from torch import nn
from torchvision import datasets, transforms
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.presets.configs import IdealizedPreset
from aihwkit.simulator.rpu_base import cuda

# Device
USE_CUDA = cuda.is_compiled()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# Same parameters as before
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.01
RANK = 8
TRANSFER_EVERY = 100
LORA_ALPHA = 4.0


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


def create_model(ab_update_mode: str, use_onehot: bool = True, use_sigma_delta: bool = False):
    """Create LRTT model with specified ab_update_mode."""

    def get_config():
        rpu_config = PythonLRTTPreset.sixt1c_ab_ideal(
            rank=RANK,
            transfer_every=TRANSFER_EVERY,
            lora_alpha=LORA_ALPHA,
            dt_batch_sec=0.0,
            ab_update_mode=ab_update_mode,  # "auto", "projected", "reconstruction", "chain_rule"
            forward_inject=False
        )
        rpu_config.device.use_onehot = use_onehot
        rpu_config.device.use_sigma_delta = use_sigma_delta
        rpu_config.device.reinit_gain = 0.5
        rpu_config.device.correct_gradient_magnitudes = True
        rpu_config.device.transfer_lr = LORA_ALPHA
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
    total_loss = 0
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

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100. * correct / total


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


def evaluate_c_only(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE).view(-1, 784)
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

            pred = x.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def run_experiment(ab_update_mode: str, use_onehot: bool = True, use_sigma_delta: bool = False):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: ab_update_mode={ab_update_mode}")
    print(f"            use_onehot={use_onehot}, sigma_delta={use_sigma_delta}")
    print('='*60)

    train_loader, test_loader = load_mnist()
    model = create_model(ab_update_mode, use_onehot, use_sigma_delta)

    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    criterion = nn.NLLLoss()

    results = []
    start_time = time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)
        c_only_acc = evaluate_c_only(model, test_loader)

        results.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'c_only_acc': c_only_acc
        })

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: "
              f"Train={train_acc:.2f}%, Test={test_acc:.2f}%, C-only={c_only_acc:.2f}%")

    elapsed = time() - start_time

    total_transfers = 0
    for layer in model:
        if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
            total_transfers += layer.analog_module.controller.num_transfers

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Total transfers: {total_transfers}")
    print(f"Final Test Accuracy: {results[-1]['test_acc']:.2f}%")
    print(f"Final C-only Accuracy: {results[-1]['c_only_acc']:.2f}%")

    return results


def main():
    print("MNIST LRTT Training: ab_update_mode Comparison")
    print(f"Config: RANK={RANK}, TRANSFER_EVERY={TRANSFER_EVERY}, LORA_ALPHA={LORA_ALPHA}")
    print(f"Training: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}")

    # Best settings from previous test: use_onehot=True, sigma_delta=False

    # Test 1: projected (auto with forward_inject=False)
    results_projected = run_experiment("projected", use_onehot=True, use_sigma_delta=False)

    # Test 2: reconstruction
    results_reconstruction = run_experiment("reconstruction", use_onehot=True, use_sigma_delta=False)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (use_onehot=True, sigma_delta=False)")
    print("="*60)
    print(f"{'ab_update_mode':<20} {'Test Acc':>10} {'C-only':>10}")
    print("-"*60)
    print(f"{'projected':<20} {results_projected[-1]['test_acc']:>9.2f}% {results_projected[-1]['c_only_acc']:>9.2f}%")
    print(f"{'reconstruction':<20} {results_reconstruction[-1]['test_acc']:>9.2f}% {results_reconstruction[-1]['c_only_acc']:>9.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

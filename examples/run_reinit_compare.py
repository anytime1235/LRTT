# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""MNIST training comparing LRTT reinit modes with 6T1C A/B tiles.

This script uses system aihwkit and adds LRTT extensions dynamically.

Compares two reinit modes with decay_factor=1.0:
1) "decay" mode: A, B both preserved after transfer (A*=1, B*=1)
2) "hybrid" mode: A=0, B preserved after transfer (A=0, B*=1)
"""

import os
import sys
from time import time
import math

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Use system aihwkit
from aihwkit.simulator.rpu_base import cuda

# Pure PyTorch implementation - no aihwkit LRTT imports needed

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

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


class LRTTLinear(nn.Module):
    """LRTT Linear layer using pure Python implementation."""

    def __init__(self, in_features, out_features, rank, reinit_mode, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.reinit_mode = reinit_mode

        # Create weight matrices
        # A: [out_features, rank], B: [rank, in_features], C: [out_features, in_features]
        self.weight_a = nn.Parameter(torch.zeros(out_features, rank))
        self.weight_b = nn.Parameter(torch.zeros(rank, in_features))
        self.weight_c = nn.Parameter(torch.zeros(out_features, in_features))

        # Tracking
        self.num_updates = 0
        self.num_transfers = 0

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # A = 0
        nn.init.zeros_(self.weight_a)
        # B ~ Kaiming for first rank rows
        nn.init.kaiming_uniform_(self.weight_b, a=math.sqrt(5))
        self.weight_b.data *= 0.5  # reinit_gain
        # C ~ Kaiming
        nn.init.kaiming_uniform_(self.weight_c, a=math.sqrt(5))

    def forward(self, x):
        """Forward pass: y = Cx + alpha * A @ B @ x"""
        # C @ x
        y_c = torch.mm(x, self.weight_c.t())
        # A @ (B @ x)
        y_ab = torch.mm(torch.mm(x, self.weight_b.t()), self.weight_a.t())
        # Combined with LoRA alpha scaling
        scale = LORA_ALPHA / self.rank
        return y_c + scale * y_ab

    def forward_c_only(self, x):
        """Forward pass using only C tile."""
        return torch.mm(x, self.weight_c.t())

    def maybe_transfer(self):
        """Check if transfer should occur and perform it."""
        self.num_updates += 1

        if self.num_updates % TRANSFER_EVERY == 0:
            self._transfer()

    def _transfer(self):
        """Transfer A @ B to C and reinit A, B."""
        with torch.no_grad():
            # Transfer: C += lr * A @ B
            scale = LORA_ALPHA / self.rank
            delta = scale * torch.mm(self.weight_a, self.weight_b)
            self.weight_c.add_(delta)

            self.num_transfers += 1

            # Reinit based on mode
            if self.reinit_mode == "decay":
                # Both A and B preserved (multiplied by decay_factor=1.0)
                self.weight_a.mul_(DECAY_FACTOR)
                self.weight_b.mul_(DECAY_FACTOR)
            elif self.reinit_mode == "hybrid":
                # A = 0, B preserved
                nn.init.zeros_(self.weight_a)
                self.weight_b.mul_(DECAY_FACTOR)
            else:  # standard
                # A = 0, B ~ Kaiming
                nn.init.zeros_(self.weight_a)
                nn.init.kaiming_uniform_(self.weight_b, a=math.sqrt(5))
                self.weight_b.data *= 0.5


class IdealLinear(nn.Module):
    """Idealized linear layer (no LRTT, just standard weights)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class LRTTNetwork(nn.Module):
    """LRTT Network for MNIST."""

    def __init__(self, reinit_mode):
        super().__init__()
        self.reinit_mode = reinit_mode

        # LRTT layers
        self.lrtt1 = LRTTLinear(INPUT_SIZE, HIDDEN_SIZES[0], LRTT_RANK, reinit_mode)
        self.lrtt2 = LRTTLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], LRTT_RANK, reinit_mode)

        # Output layer (idealized, no LRTT)
        self.output = IdealLinear(HIDDEN_SIZES[1], OUTPUT_SIZE)

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.lrtt1(x))
        x = self.relu(self.lrtt2(x))
        x = self.log_softmax(self.output(x))
        return x

    def forward_c_only(self, x):
        """Forward using only C tiles."""
        x = self.relu(self.lrtt1.forward_c_only(x))
        x = self.relu(self.lrtt2.forward_c_only(x))
        x = self.log_softmax(self.output(x))
        return x

    def maybe_transfer_all(self):
        """Check transfer for all LRTT layers."""
        self.lrtt1.maybe_transfer()
        self.lrtt2.maybe_transfer()

    def get_stats(self):
        """Get LRTT statistics."""
        return {
            'layer1_transfers': self.lrtt1.num_transfers,
            'layer2_transfers': self.lrtt2.num_transfers,
            'total_transfers': self.lrtt1.num_transfers + self.lrtt2.num_transfers
        }


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

        # Check transfer after each batch
        model.maybe_transfer_all()

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

            output = model.forward_c_only(images)
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def train_model(reinit_mode, train_loader, val_loader):
    """Train model with specified reinit mode."""
    print(f"\n{'='*60}")
    print(f"Training with reinit_mode='{reinit_mode}', decay_factor={DECAY_FACTOR}")
    print(f"{'='*60}")

    if reinit_mode == "decay":
        print("  After transfer: A *= 1.0 (keep), B *= 1.0 (keep)")
    else:  # hybrid
        print("  After transfer: A = 0, B *= 1.0 (keep)")

    model = LRTTNetwork(reinit_mode).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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
        stats = model.get_stats()
        total_transfers = stats['total_transfers']

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
    print(f"  Output layer: Idealized (no LRTT)")
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

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""MNIST training with LRTT - No Transfer Test.

Tests decay mode with very high transfer_every to prevent any transfers.
This shows how the model performs when weights stay only in A/B tiles
and never transfer to C.

GPU enabled.
"""

import os
import sys
from time import time
import math

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Force GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
TRANSFER_EVERY = 1000000  # Very high - effectively no transfer
LORA_ALPHA = 4.0
DECAY_FACTOR = 1.0


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

    def __init__(self, in_features, out_features, rank, transfer_every, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.transfer_every = transfer_every

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
        # B ~ Kaiming
        nn.init.kaiming_uniform_(self.weight_b, a=math.sqrt(5))
        self.weight_b.data *= 0.5
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

        if self.num_updates % self.transfer_every == 0:
            self._transfer()

    def _transfer(self):
        """Transfer A @ B to C and reinit A, B."""
        with torch.no_grad():
            # Transfer: C += lr * A @ B
            scale = LORA_ALPHA / self.rank
            delta = scale * torch.mm(self.weight_a, self.weight_b)
            self.weight_c.add_(delta)

            self.num_transfers += 1

            # Decay mode: keep A, B
            self.weight_a.mul_(DECAY_FACTOR)
            self.weight_b.mul_(DECAY_FACTOR)


class IdealLinear(nn.Module):
    """Idealized linear layer (no LRTT, just standard weights)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class LRTTNetwork(nn.Module):
    """LRTT Network for MNIST."""

    def __init__(self, transfer_every):
        super().__init__()
        self.transfer_every = transfer_every

        # LRTT layers
        self.lrtt1 = LRTTLinear(INPUT_SIZE, HIDDEN_SIZES[0], LRTT_RANK, transfer_every)
        self.lrtt2 = LRTTLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], LRTT_RANK, transfer_every)

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
            'total_transfers': self.lrtt1.num_transfers + self.lrtt2.num_transfers,
            'layer1_updates': self.lrtt1.num_updates,
            'layer2_updates': self.lrtt2.num_updates,
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


def train_model(transfer_every, train_loader, val_loader):
    """Train model with specified transfer rate."""
    print(f"\n{'='*60}")
    print(f"Training with transfer_every={transfer_every}")
    print(f"{'='*60}")

    if transfer_every >= 100000:
        print("  Mode: NO TRANSFER (A/B only learning)")
    else:
        print(f"  Mode: Transfer every {transfer_every} batches")

    model = LRTTNetwork(transfer_every).to(DEVICE)
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
    print(f"\nFinal Results (transfer_every={transfer_every}):")
    print(f"  Best Val Acc (Full): {max(history['val_acc']):.2f}%")
    print(f"  Best Val Acc (C-only): {max(history['val_acc_c_only']):.2f}%")
    print(f"  Final Val Acc (Full): {history['val_acc'][-1]:.2f}%")
    print(f"  Final Val Acc (C-only): {history['val_acc_c_only'][-1]:.2f}%")
    print(f"  Total Transfers: {total_transfers}")
    print(f"  Total Updates: {stats['layer1_updates'] + stats['layer2_updates']}")

    return history


def compare_results(history_transfer, history_no_transfer):
    """Compare and print results."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY: Transfer vs No Transfer")
    print("="*70)
    print("\n" + "-"*70)
    print(f"{'Metric':<30} {'With Transfer (100)':<20} {'No Transfer':<20}")
    print("-"*70)

    metrics = [
        ('Best Val Acc Full (%)', max(history_transfer['val_acc']), max(history_no_transfer['val_acc'])),
        ('Final Val Acc Full (%)', history_transfer['val_acc'][-1], history_no_transfer['val_acc'][-1]),
        ('Best Val Acc C-only (%)', max(history_transfer['val_acc_c_only']), max(history_no_transfer['val_acc_c_only'])),
        ('Final Val Acc C-only (%)', history_transfer['val_acc_c_only'][-1], history_no_transfer['val_acc_c_only'][-1]),
        ('Final Train Acc (%)', history_transfer['train_acc'][-1], history_no_transfer['train_acc'][-1]),
        ('Final Loss', history_transfer['train_loss'][-1], history_no_transfer['train_loss'][-1]),
        ('Total Transfers', history_transfer['transfers'][-1], history_no_transfer['transfers'][-1]),
    ]

    for name, transfer_val, no_transfer_val in metrics:
        if isinstance(transfer_val, float):
            print(f"{name:<30} {transfer_val:<20.2f} {no_transfer_val:<20.2f}")
        else:
            print(f"{name:<30} {transfer_val:<20} {no_transfer_val:<20}")

    print("-"*70)

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    transfer_full = max(history_transfer['val_acc'])
    no_transfer_full = max(history_no_transfer['val_acc'])

    transfer_c_only = max(history_transfer['val_acc_c_only'])
    no_transfer_c_only = max(history_no_transfer['val_acc_c_only'])

    print(f"\n1. Full Model Performance (C + A*B):")
    print(f"   With Transfer: {transfer_full:.2f}%")
    print(f"   No Transfer:   {no_transfer_full:.2f}%")
    print(f"   Difference:    {transfer_full - no_transfer_full:+.2f}%")

    print(f"\n2. C-only Performance (excluding A*B contribution):")
    print(f"   With Transfer: {transfer_c_only:.2f}%")
    print(f"   No Transfer:   {no_transfer_c_only:.2f}%")
    print(f"   Difference:    {transfer_c_only - no_transfer_c_only:+.2f}%")

    print(f"\n3. Gap between Full and C-only:")
    print(f"   With Transfer: {transfer_full - transfer_c_only:.2f}% (A*B contributes this much)")
    print(f"   No Transfer:   {no_transfer_full - no_transfer_c_only:.2f}% (A*B contributes this much)")

    print("\n4. Key Insights:")
    if no_transfer_c_only < transfer_c_only:
        print(f"   - Without transfer, C tile doesn't learn from gradients")
        print(f"   - C-only accuracy shows C tile's independent learning")
    if no_transfer_full > no_transfer_c_only + 1:
        print(f"   - A*B provides {no_transfer_full - no_transfer_c_only:.1f}% accuracy boost without transfer")
        print(f"   - This shows A/B tiles are learning the task")

    print("="*70)


def main():
    print("="*70)
    print("LRTT Transfer Rate Comparison: Transfer vs No Transfer")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  LRTT Rank: {LRTT_RANK}")
    print(f"  LoRA Alpha: {LORA_ALPHA}")
    print(f"  Decay Factor: {DECAY_FACTOR}")
    print(f"  Device: {DEVICE}")
    print(f"\nComparing:")
    print(f"  1) transfer_every=100 (regular transfer)")
    print(f"  2) transfer_every=1000000 (no transfer)")

    # Load data
    train_loader, val_loader = load_images()
    print(f"\nDataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} test")
    print(f"Batches per epoch: {len(train_loader)}")

    # Train with regular transfer
    history_transfer = train_model(100, train_loader, val_loader)

    # Train without transfer
    history_no_transfer = train_model(1000000, train_loader, val_loader)

    # Compare results
    compare_results(history_transfer, history_no_transfer)


if __name__ == "__main__":
    main()

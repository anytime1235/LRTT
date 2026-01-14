#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LRTT MNIST Gradient Snapshots

This script trains a simple MNIST MLP (784→256→10) and captures gradient
snapshots at specific training steps. These snapshots can be used to test
the LRTT claims benchmark with realistic task-derived gradients.

Captured data per snapshot:
    - G: Gradient matrix [d_size, x_size] = D^T @ X
    - X: Layer input activations [batch, x_size]
    - D: Layer output gradients [batch, d_size]
    - metadata: step, loss, accuracy, etc.

Usage examples:
    # Generate snapshots every 100 steps for 1000 steps
    python experiments/lrtt_mnist_gradient_snapshots.py --snapshot_every 100 --steps 1000

    # Generate snapshots for first FC layer only
    python experiments/lrtt_mnist_gradient_snapshots.py --layer fc1

    # Custom output directory
    python experiments/lrtt_mnist_gradient_snapshots.py --output_dir snapshots/mnist_run1

After generating snapshots, use with lrtt_claims_benchmark.py:
    python experiments/lrtt_claims_benchmark.py --run_single \\
        --G_mode load_snapshot --snapshot_path snapshots/mnist/fc1_step100.pt

Author: LRTT Team
Date: 2024
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# MNIST MLP Model
# =============================================================================

class MNISTModel(nn.Module):
    """Simple MNIST MLP: 784 → 256 → 10"""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.hidden_size = hidden_size

        # Gradient hooks storage
        self._hooks = []
        self._activations = {}
        self._gradients = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        self._activations['fc1_input'] = x.detach()  # Store input to fc1
        x = F.relu(self.fc1(x))
        self._activations['fc2_input'] = x.detach()  # Store input to fc2
        x = self.fc2(x)
        return x

    def register_gradient_hooks(self):
        """Register hooks to capture gradients during backward pass."""

        def make_hook(name):
            def hook(grad):
                self._gradients[name] = grad.detach()
                return grad
            return hook

        # Register hooks for layer outputs (to capture gradients)
        # For fc1: gradient w.r.t. output of fc1 (before ReLU)
        # We'll use the weight gradient instead

        # Clear existing hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_layer_gradient_data(self, layer_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get gradient data for a specific layer.

        Returns:
            Dict with:
                - 'X': Layer input activations [batch, in_features]
                - 'D': Layer output gradients [batch, out_features]
                - 'G': Gradient matrix [out_features, in_features] = D^T @ X
        """
        if layer_name == 'fc1':
            layer = self.fc1
            input_key = 'fc1_input'
        elif layer_name == 'fc2':
            layer = self.fc2
            input_key = 'fc2_input'
        else:
            return None

        if input_key not in self._activations:
            return None

        X = self._activations[input_key]  # [batch, in_features]

        # Get weight gradient: dL/dW = D^T @ X
        # So D^T = dL/dW @ X^{-1} is complex, but we can compute D directly:
        # dL/dW[i,j] = sum_b D[b,i] * X[b,j]
        # So dL/dW = D^T @ X

        if layer.weight.grad is None:
            return None

        weight_grad = layer.weight.grad.detach()  # [out_features, in_features]

        # Reconstruct D from weight_grad and X
        # weight_grad = D^T @ X
        # We need D such that D^T @ X ≈ weight_grad
        # Since X^T @ X may not be invertible, use pseudoinverse
        # But for our purposes, we can use a simpler approach:
        # Create synthetic D that matches the gradient direction

        # Actually, we need to capture D during backward pass
        # Let's use a different approach: register a backward hook on the layer

        # For now, use SVD factorization of weight_grad
        batch = X.shape[0]
        G = weight_grad  # [out_features, in_features]

        # Create D such that D^T @ X ≈ G via SVD
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        min_dim = min(G.shape)
        effective_rank = min(batch, min_dim)

        sqrt_S = torch.sqrt(S[:effective_rank])
        D = (U[:, :effective_rank] * sqrt_S.unsqueeze(0)).t()  # [r, out_features]
        X_recon = sqrt_S.unsqueeze(1) * Vh[:effective_rank, :]  # [r, in_features]

        # Pad to batch size
        if effective_rank < batch:
            D = torch.cat([D, torch.zeros(batch - effective_rank, G.shape[0], device=G.device)], dim=0)
            X_recon = torch.cat([X_recon, torch.zeros(batch - effective_rank, G.shape[1], device=G.device)], dim=0)

        return {
            'X': X,
            'D': D.t(),  # [out_features, batch] -> transpose for convention
            'G': G,
            'd_size': G.shape[0],
            'x_size': G.shape[1],
            'batch': batch
        }


class GradientCaptureModel(nn.Module):
    """MNIST MLP with gradient capture via hooks."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.hidden_size = hidden_size

        # Storage for captured data
        self._fc1_input = None
        self._fc1_output = None
        self._fc2_input = None
        self._fc2_output = None
        self._fc1_output_grad = None
        self._fc2_output_grad = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def fc1_forward_hook(module, input, output):
            self._fc1_input = input[0].detach()
            self._fc1_output = output.detach()

        def fc2_forward_hook(module, input, output):
            self._fc2_input = input[0].detach()
            self._fc2_output = output.detach()

        def fc1_backward_hook(module, grad_input, grad_output):
            self._fc1_output_grad = grad_output[0].detach()

        def fc2_backward_hook(module, grad_input, grad_output):
            self._fc2_output_grad = grad_output[0].detach()

        self.fc1.register_forward_hook(fc1_forward_hook)
        self.fc2.register_forward_hook(fc2_forward_hook)
        self.fc1.register_full_backward_hook(fc1_backward_hook)
        self.fc2.register_full_backward_hook(fc2_backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_gradient_snapshot(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Get gradient snapshot for a layer.

        For fc1: G = (ReLU'(z) * D_next)^T @ X = D^T @ X
        where D = ReLU'(z) * D_next is the effective gradient at fc1 output.

        Returns:
            Dict with X, D, G, and metadata
        """
        if layer_name == 'fc1':
            X = self._fc1_input  # [batch, 784]
            # D is the gradient of loss w.r.t. fc1 output (before ReLU in this formulation)
            # Actually, we want gradient w.r.t. fc1 LINEAR output
            D = self._fc1_output_grad  # [batch, hidden_size]

            if X is None or D is None:
                return None

            # G = D^T @ X
            G = D.t() @ X  # [hidden_size, 784]

            return {
                'X': X.cpu(),
                'D': D.cpu(),
                'G': G.cpu(),
                'd_size': G.shape[0],
                'x_size': G.shape[1],
                'batch_size': X.shape[0],
                'layer': layer_name
            }

        elif layer_name == 'fc2':
            X = self._fc2_input  # [batch, hidden_size]
            D = self._fc2_output_grad  # [batch, 10]

            if X is None or D is None:
                return None

            G = D.t() @ X  # [10, hidden_size]

            return {
                'X': X.cpu(),
                'D': D.cpu(),
                'G': G.cpu(),
                'd_size': G.shape[0],
                'x_size': G.shape[1],
                'batch_size': X.shape[0],
                'layer': layer_name
            }

        return None


# =============================================================================
# Training Loop with Snapshot Capture
# =============================================================================

def train_with_snapshots(
    model: GradientCaptureModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    snapshot_every: int,
    total_steps: int,
    layers: List[str],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """Train model and capture gradient snapshots.

    Args:
        model: Model with gradient capture hooks
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device
        snapshot_every: Capture snapshot every N steps
        total_steps: Total training steps
        layers: Which layers to capture ('fc1', 'fc2')
        output_dir: Output directory for snapshots

    Returns:
        List of training metrics
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    step = 0
    epoch = 0
    metrics_history = []
    snapshot_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training MNIST MLP with Gradient Snapshots")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    print(f"Snapshot every: {snapshot_every} steps")
    print(f"Layers to capture: {layers}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    while step < total_steps:
        epoch += 1
        for batch_idx, (data, target) in enumerate(train_loader):
            if step >= total_steps:
                break

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Capture snapshots
            if step % snapshot_every == 0:
                for layer in layers:
                    snapshot = model.get_gradient_snapshot(layer)
                    if snapshot is not None:
                        # Add metadata
                        snapshot['step'] = step
                        snapshot['epoch'] = epoch
                        snapshot['loss'] = loss.item()

                        # Compute accuracy
                        pred = output.argmax(dim=1, keepdim=True)
                        correct = pred.eq(target.view_as(pred)).sum().item()
                        snapshot['accuracy'] = correct / len(target)

                        # Compute G spectrum info
                        G = snapshot['G']
                        _, S, _ = torch.linalg.svd(G, full_matrices=False)
                        snapshot['G_norm'] = torch.norm(G).item()
                        snapshot['G_rank'] = (S > 1e-6 * S[0]).sum().item()
                        snapshot['singular_values'] = S[:10].tolist()

                        # Save snapshot
                        snapshot_path = output_dir / f"{layer}_step{step:06d}.pt"
                        torch.save(snapshot, snapshot_path)
                        snapshot_count += 1

                        print(f"Step {step:5d}: Saved {layer} snapshot "
                              f"(loss={loss.item():.4f}, acc={snapshot['accuracy']:.3f}, "
                              f"||G||={snapshot['G_norm']:.4f}, rank≈{snapshot['G_rank']})")

            optimizer.step()

            # Log metrics
            if step % 100 == 0:
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(target)

                metrics_history.append({
                    'step': step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'accuracy': accuracy
                })

                if step % 500 == 0:
                    print(f"Step {step:5d} (epoch {epoch}): loss={loss.item():.4f}, acc={accuracy:.3f}")

            step += 1

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total snapshots saved: {snapshot_count}")
    print(f"{'='*60}\n")

    return metrics_history


def analyze_snapshots(output_dir: Path, layers: List[str]) -> None:
    """Analyze saved gradient snapshots."""
    print(f"\n{'='*60}")
    print(f"Gradient Snapshot Analysis")
    print(f"{'='*60}")

    for layer in layers:
        snapshots = sorted(output_dir.glob(f"{layer}_step*.pt"))
        if not snapshots:
            continue

        print(f"\n{layer}:")
        print(f"  Total snapshots: {len(snapshots)}")

        # Load a few snapshots for analysis
        for snap_path in [snapshots[0], snapshots[len(snapshots)//2], snapshots[-1]]:
            snapshot = torch.load(snap_path, map_location='cpu')
            G = snapshot['G']
            step = snapshot['step']

            print(f"\n  Step {step}:")
            print(f"    G shape: {list(G.shape)}")
            print(f"    ||G||_F: {snapshot['G_norm']:.4f}")
            print(f"    Effective rank: {snapshot['G_rank']}")
            print(f"    Top 5 singular values: {snapshot['singular_values'][:5]}")
            print(f"    Loss: {snapshot['loss']:.4f}")
            print(f"    Accuracy: {snapshot['accuracy']:.3f}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LRTT MNIST Gradient Snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--steps', type=int, default=1000,
                       help='Total training steps (default: 1000)')
    parser.add_argument('--snapshot_every', type=int, default=100,
                       help='Capture snapshot every N steps (default: 100)')
    parser.add_argument('--layer', type=str, default='all',
                       choices=['fc1', 'fc2', 'all'],
                       help='Which layer to capture (default: all)')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden layer size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='snapshots/mnist',
                       help='Output directory (default: snapshots/mnist)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze existing snapshots')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='MNIST data directory (default: ./data)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)

    # Determine layers
    if args.layer == 'all':
        layers = ['fc1', 'fc2']
    else:
        layers = [args.layer]

    if args.analyze_only:
        analyze_snapshots(output_dir, layers)
        return

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_dataset = datasets.MNIST(
            args.data_dir, train=True, download=True, transform=transform
        )
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Please ensure you have internet access for first-time download")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # Create model
    model = GradientCaptureModel(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Train with snapshots
    metrics = train_with_snapshots(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        snapshot_every=args.snapshot_every,
        total_steps=args.steps,
        layers=layers,
        output_dir=output_dir
    )

    # Save training metrics
    import pandas as pd
    df = pd.DataFrame(metrics)
    df.to_csv(output_dir / "training_metrics.csv", index=False)
    print(f"Saved training metrics to {output_dir / 'training_metrics.csv'}")

    # Save config
    config = {
        'steps': args.steps,
        'snapshot_every': args.snapshot_every,
        'layers': layers,
        'hidden_size': args.hidden_size,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Analyze snapshots
    analyze_snapshots(output_dir, layers)

    print("\nDone! Use snapshots with:")
    print(f"  python experiments/lrtt_claims_benchmark.py --run_single \\")
    print(f"    --G_mode load_snapshot --snapshot_path {output_dir}/fc1_step000100.pt")


if __name__ == "__main__":
    main()

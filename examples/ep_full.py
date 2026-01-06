# -*- coding: utf-8 -*-
"""EP-Full: Equilibrium Propagation with Full-Rank Weight Updates

Three-layer Energy-Based Model with standard EP training.
No LoRA, no transfer - direct full-rank weight updates.

Architecture:
    x (784) → h1 (256) → h2 (128) → y (10)

Energy function:
    E = 1/2||h1||^2 + 1/2||h2||^2 + 1/2||y||^2
        - x^T W1 h1 - h1^T W2 h2 - h2^T W3 y

EP update rule:
    ΔW = (η/β) [∂E(s0)/∂W - ∂E(sβ)/∂W]

No autograd - all updates are manual EP-style.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
from dataclasses import dataclass, asdict
from typing import Tuple
import wandb


@dataclass
class Config:
    """EP-Full configuration."""
    # Architecture: 784 → 256 → 128 → 10
    input_size: int = 784
    hidden_sizes: Tuple[int, int] = (256, 128)  # Two hidden layers
    output_size: int = 10

    # EP dynamics parameters
    n_iter_free: int = 50     # Free phase iterations
    n_iter_nudge: int = 8     # Nudged phase iterations
    beta: float = 1.0         # Nudging strength
    dt: float = 0.5           # Integration step size (epsilon)

    # Training parameters
    lr: float = 0.2
    batch_size: int = 64
    epochs: int = 100

    # Device (auto-detect)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EPFullModel:
    """Three-layer Energy-Based Model with EP dynamics.

    Architecture:
        x (784) → h1 (256) → h2 (128) → y (10)

    Energy function:
        E = 1/2||h1||^2 + 1/2||h2||^2 + 1/2||y||^2
            - x^T W1 h1 - h1^T W2 h2 - h2^T W3 y

    Gradients w.r.t. states:
        ∂E/∂h1 = h1 - W1^T x - W2 h2
        ∂E/∂h2 = h2 - W2^T h1 - W3 y
        ∂E/∂y = y - W3^T h2
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        h1_size, h2_size = config.hidden_sizes

        # Weight matrices (full-rank)
        # W1: [h1_size, input_size] - maps x to h1
        self.W1 = torch.empty(h1_size, config.input_size, device=config.device)
        std_1 = math.sqrt(2.0 / (config.input_size + h1_size))
        torch.nn.init.normal_(self.W1, 0, std_1)

        # W2: [h2_size, h1_size] - maps h1 to h2
        self.W2 = torch.empty(h2_size, h1_size, device=config.device)
        std_2 = math.sqrt(2.0 / (h1_size + h2_size))
        torch.nn.init.normal_(self.W2, 0, std_2)

        # W3: [output_size, h2_size] - maps h2 to y
        self.W3 = torch.empty(config.output_size, h2_size, device=config.device)
        std_3 = math.sqrt(2.0 / (h2_size + config.output_size))
        torch.nn.init.normal_(self.W3, 0, std_3)

        # Biases
        self.b1 = torch.zeros(h1_size, device=config.device)
        self.b2 = torch.zeros(h2_size, device=config.device)
        self.by = torch.zeros(config.output_size, device=config.device)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """Activation function: sigmoid (smooth, EP-compatible)"""
        return torch.sigmoid(x)

    def run_free_phase(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run free phase dynamics to equilibrium.

        Dynamics (gradient descent on E):
            dh1/dt = W1 x + W2^T h2 - h1 + b1
            dh2/dt = W2 h1 + W3^T y - h2 + b2
            dy/dt = W3 h2 - y + by

        Args:
            x: input [batch, 784]

        Returns:
            (h1_0, h2_0, y0): equilibrium states after free phase
        """
        batch_size = x.size(0)
        dt = self.config.dt
        h1_size, h2_size = self.config.hidden_sizes

        # Initialize states to zero
        h1 = torch.zeros(batch_size, h1_size, device=self.device)
        h2 = torch.zeros(batch_size, h2_size, device=self.device)
        y = torch.zeros(batch_size, self.config.output_size, device=self.device)

        # Run dynamics
        for _ in range(self.config.n_iter_free):
            # dh1/dt = W1 @ x + W2^T @ h2 - h1 + b1
            dh1 = F.linear(x, self.W1) + F.linear(h2, self.W2.t()) - h1 + self.b1

            # dh2/dt = W2 @ h1 + W3^T @ y - h2 + b2
            dh2 = F.linear(h1, self.W2) + F.linear(y, self.W3.t()) - h2 + self.b2

            # dy/dt = W3 @ h2 - y + by
            dy = F.linear(h2, self.W3) - y + self.by

            # Update with activation
            h1 = self.activation(h1 + dt * dh1)
            h2 = self.activation(h2 + dt * dh2)
            y = self.activation(y + dt * dy)

        return h1, h2, y

    def run_nudged_phase(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        h1_0: torch.Tensor,
        h2_0: torch.Tensor,
        y0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run nudged phase dynamics with output nudged toward target.

        Dynamics with nudging term:
            dh1/dt = W1 x + W2^T h2 - h1 + b1
            dh2/dt = W2 h1 + W3^T y - h2 + b2
            dy/dt = W3 h2 - y + by + β(target - y)

        Args:
            x: input [batch, 784]
            target: one-hot target [batch, 10]
            h1_0, h2_0, y0: initial states from free phase

        Returns:
            (h1_1, h2_1, y1): equilibrium states after nudged phase
        """
        dt = self.config.dt
        beta = self.config.beta

        # Start from free phase equilibrium
        h1 = h1_0.clone()
        h2 = h2_0.clone()
        y = y0.clone()

        # Run nudged dynamics
        for _ in range(self.config.n_iter_nudge):
            # dh1/dt = W1 @ x + W2^T @ h2 - h1 + b1
            dh1 = F.linear(x, self.W1) + F.linear(h2, self.W2.t()) - h1 + self.b1

            # dh2/dt = W2 @ h1 + W3^T @ y - h2 + b2
            dh2 = F.linear(h1, self.W2) + F.linear(y, self.W3.t()) - h2 + self.b2

            # dy/dt = W3 @ h2 - y + by + β(target - y)
            dy = F.linear(h2, self.W3) - y + self.by + beta * (target - y)

            # Update with activation
            h1 = self.activation(h1 + dt * dh1)
            h2 = self.activation(h2 + dt * dh2)
            y = self.activation(y + dt * dy)

        return h1, h2, y

    def ep_update(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        lr: float
    ) -> float:
        """Compute EP gradients and update weights.

        EP gradient rule:
            ΔW1 = (η/β) * (h1_1^T @ x - h1_0^T @ x)
            ΔW2 = (η/β) * (h2_1^T @ h1_1 - h2_0^T @ h1_0)
            ΔW3 = (η/β) * (y1^T @ h2_1 - y0^T @ h2_0)

        Args:
            x: input [batch, 784]
            target: one-hot target [batch, 10]
            lr: learning rate

        Returns:
            loss (MSE for monitoring)
        """
        batch_size = x.size(0)
        beta = self.config.beta

        # 1) Free phase
        h1_0, h2_0, y0 = self.run_free_phase(x)

        # 2) Nudged phase
        h1_1, h2_1, y1 = self.run_nudged_phase(x, target, h1_0, h2_0, y0)

        # 3) Compute EP gradients (average over batch)
        # ΔW1 = (h1_1^T @ x - h1_0^T @ x) / β  → [h1_size, input_size]
        dW1 = (h1_1.t() @ x - h1_0.t() @ x) / (beta * batch_size)

        # ΔW2 = (h2_1^T @ h1_1 - h2_0^T @ h1_0) / β  → [h2_size, h1_size]
        dW2 = (h2_1.t() @ h1_1 - h2_0.t() @ h1_0) / (beta * batch_size)

        # ΔW3 = (y1^T @ h2_1 - y0^T @ h2_0) / β  → [output_size, h2_size]
        dW3 = (y1.t() @ h2_1 - y0.t() @ h2_0) / (beta * batch_size)

        # Bias gradients
        db1 = (h1_1 - h1_0).mean(dim=0) / beta
        db2 = (h2_1 - h2_0).mean(dim=0) / beta
        dby = (y1 - y0).mean(dim=0) / beta

        # 4) Apply updates
        self.W1.add_(lr * dW1)
        self.W2.add_(lr * dW2)
        self.W3.add_(lr * dW3)
        self.b1.add_(lr * db1)
        self.b2.add_(lr * db2)
        self.by.add_(lr * dby)

        # Return loss for monitoring (MSE between free phase output and target)
        loss = F.mse_loss(y0, target)
        return loss.item()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass (run free phase for inference)."""
        return self.run_free_phase(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions (argmax of output)."""
        _, _, y = self.forward(x)
        return y.argmax(dim=1)


def one_hot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Convert labels to one-hot encoding."""
    return F.one_hot(labels, num_classes).float()


def evaluate(model: EPFullModel, dataloader: DataLoader) -> float:
    """Evaluate model accuracy."""
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.view(-1, 784).to(model.device)
        labels = labels.to(model.device)

        predictions = model.predict(images)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


def train_epoch(model: EPFullModel, dataloader: DataLoader, epoch: int) -> float:
    """Train for one epoch."""
    total_loss = 0
    n_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.view(-1, 784).to(model.device)
        targets = one_hot(labels).to(model.device)

        loss = model.ep_update(images, targets, model.config.lr)
        total_loss += loss
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}")

    return total_loss / n_batches


def main():
    """Main training loop."""
    print("=" * 60)
    print("EP-Full: Equilibrium Propagation with Full-Rank Updates")
    print("=" * 60)

    config = Config()

    # Initialize wandb with run name containing key parameters
    h1_size, h2_size = config.hidden_sizes
    run_name = (
        f"full_h{h1_size}-{h2_size}_"
        f"free{config.n_iter_free}_nudge{config.n_iter_nudge}_"
        f"beta{config.beta}_dt{config.dt}_"
        f"lr{config.lr}_bs{config.batch_size}_ep{config.epochs}"
    )
    wandb.init(
        project="ep_lora",
        name=run_name,
        config=asdict(config),
    )

    print(f"\nConfiguration:")
    print(f"  Architecture: {config.input_size} → {h1_size} → {h2_size} → {config.output_size}")
    print(f"  EP iterations: free={config.n_iter_free}, nudge={config.n_iter_nudge}")
    print(f"  Beta: {config.beta}, dt: {config.dt}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Device: {config.device}")
    print()

    # Load MNIST - no normalization, keep [0, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Initialize model
    print("Initializing EP-Full model...")
    model = EPFullModel(config)

    # Count parameters
    # W1: h1_size x input_size, W2: h2_size x h1_size, W3: output_size x h2_size
    total_weight_params = (
        h1_size * config.input_size +  # W1
        h2_size * h1_size +             # W2
        config.output_size * h2_size    # W3
    )
    total_bias_params = h1_size + h2_size + config.output_size
    total_params = total_weight_params + total_bias_params
    print(f"  Total trainable params: {total_params:,} ({total_weight_params:,} weights + {total_bias_params} biases)")
    print()

    # Initial evaluation
    init_acc = evaluate(model, test_loader)
    print(f"Initial Test Accuracy: {init_acc*100:.2f}%\n")
    wandb.log({"epoch": 0, "test_acc": init_acc})

    # Training loop
    for epoch in range(1, config.epochs + 1):
        print(f"{'='*40}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*40}")

        train_loss = train_epoch(model, train_loader, epoch)
        test_acc = evaluate(model, test_loader)

        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Test Accuracy: {test_acc*100:.2f}%")
        print()

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_acc": test_acc,
        })

    # Final results
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    final_acc = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")

    wandb.log({"final_test_acc": final_acc})
    wandb.finish()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""EP-LoRA-TT: Equilibrium Propagation + Low-Rank LoRA + Transfer

Single-hidden-layer Energy-Based Model with:
- EP free-phase / nudged-phase dynamics
- LoRA factorization: W = C + α(A @ B)
- Periodic transfer: C ← C + α(A @ B), then reset A, B

Energy function:
    E = 1/2||h||^2 + 1/2||y||^2 - x^T Wxh h - h^T Why y

EP update rule:
    ΔW = -(η/β) [∂E(sβ)/∂W - ∂E(s0)/∂W]

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
    """EP-LoRA-TT configuration."""
    # Architecture: 784 → 256 → 128 → 10
    input_size: int = 784
    hidden_sizes: Tuple[int, int] = (256, 128)  # Two hidden layers
    output_size: int = 10

    # LoRA parameters
    rank: int = 16
    lora_alpha: float = 2.0

    # A, B initialization mode: "zero", "kaiming", "decay"
    init_mode: str = "zero"       # A init: zero, B init: kaiming
    kaiming_scale: float = 1.0    # Kaiming std multiplier (for kaiming/decay modes)
    decay_factor: float = 0.5     # Decay multiplier for A/B after transfer (decay mode only)

    # Transfer parameters
    transfer_every: int = 1000  # Transfer every N steps

    # EP dynamics parameters
    n_iter_free: int = 50     # Free phase iterations
    n_iter_nudge: int = 8     # Nudged phase iterations
    beta: float = 1.0         # Nudging strength
    dt: float = 0.5           # Integration step size

    # Training parameters
    lr_lora: float = 0.6      # Learning rate for LoRA layers (W1, W2)
    lr_full: float = 0.2     # Learning rate for full-rank layer (W3)
    batch_size: int = 64
    epochs: int = 40

    # Device (auto-detect)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LoRAFactor:
    """LoRA factorization: W_eff = C + alpha * (A @ B)

    - C: main array [out_dim, in_dim]
    - A: fast array [out_dim, rank]
    - B: fast array [rank, in_dim]
    """

    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        rank: int,
        alpha: float = 1.0,
        device: str = "cpu",
        init_mode: str = "zero",
        kaiming_scale: float = 1.0,
        decay_factor: float = 0.5
    ):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.device = device
        self.init_mode = init_mode
        self.kaiming_scale = kaiming_scale
        self.decay_factor = decay_factor

        # Main array C - Xavier initialization
        self.C = torch.empty(out_dim, in_dim, device=device)
        std_c = math.sqrt(2.0 / (in_dim + out_dim))
        torch.nn.init.normal_(self.C, 0, std_c)

        # Initialize A and B based on init_mode
        self.A = torch.empty(out_dim, rank, device=device)
        self.B = torch.empty(rank, in_dim, device=device)
        self._init_AB()

    def _init_AB(self) -> None:
        """Initialize A and B based on init_mode.

        Modes:
        - zero: A=0, B=kaiming (standard LoRA)
        - kaiming: A=kaiming, B=kaiming (both random)
        - decay: A=0, B=kaiming (same as zero, but transfer uses decay)
        """
        std_b = math.sqrt(1.0 / self.in_dim) * self.kaiming_scale

        if self.init_mode == "zero":
            # A=0, B=kaiming
            self.A.zero_()
            torch.nn.init.normal_(self.B, 0, std_b)
        elif self.init_mode == "kaiming":
            # Both A and B use kaiming
            std_a = math.sqrt(1.0 / self.rank) * self.kaiming_scale
            torch.nn.init.normal_(self.A, 0, std_a)
            torch.nn.init.normal_(self.B, 0, std_b)
        elif self.init_mode == "decay":
            # Same as zero for initial, decay applied after transfer
            self.A.zero_()
            torch.nn.init.normal_(self.B, 0, std_b)
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def get_effective_weight(self) -> torch.Tensor:
        """Return W_eff = C + alpha * (A @ B)"""
        return self.C + self.alpha * (self.A @ self.B)

    def update(self, dA: torch.Tensor, dB: torch.Tensor, lr: float) -> None:
        """Update A and B factors (not C).

        A ← A + lr * dA
        B ← B + lr * dB
        """
        self.A.add_(lr * dA)
        self.B.add_(lr * dB)

    def transfer(self) -> None:
        """Transfer A @ B to C, then reset A and B.

        C ← C + alpha * (A @ B)

        Reset behavior depends on init_mode:
        - zero/kaiming: A ← 0 or kaiming, B ← kaiming
        - decay: A ← A * decay_factor, B ← B * decay_factor
        """
        # Transfer
        self.C.add_(self.alpha * (self.A @ self.B))

        if self.init_mode == "decay":
            # Decay mode: scale down A and B instead of reinitializing
            self.A.mul_(self.decay_factor)
            self.B.mul_(self.decay_factor)
        else:
            # zero/kaiming mode: reinitialize A and B
            self._init_AB()


class EPModel:
    """Three-layer Energy-Based Model with EP dynamics.

    Architecture:
        x (784) → h1 (256) → h2 (128) → y (10)

    Energy function:
        E = 1/2||h1||^2 + 1/2||h2||^2 + 1/2||y||^2
            - x^T W1 h1 - h1^T W2 h2 - h2^T W3 y

    LoRA applied to first two layers only:
        W1_eff = C1 + alpha * (A1 @ B1)  (x → h1)
        W2_eff = C2 + alpha * (A2 @ B2)  (h1 → h2)
        W3 = full-rank (h2 → y, no LoRA)
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        h1_size, h2_size = config.hidden_sizes

        # LoRA factors for W1: x→h1 [h1_size, input_size]
        self.lora_1 = LoRAFactor(
            out_dim=h1_size,
            in_dim=config.input_size,
            rank=config.rank,
            alpha=config.lora_alpha,
            device=config.device,
            init_mode=config.init_mode,
            kaiming_scale=config.kaiming_scale,
            decay_factor=config.decay_factor
        )

        # LoRA factors for W2: h1→h2 [h2_size, h1_size]
        self.lora_2 = LoRAFactor(
            out_dim=h2_size,
            in_dim=h1_size,
            rank=config.rank,
            alpha=config.lora_alpha,
            device=config.device,
            init_mode=config.init_mode,
            kaiming_scale=config.kaiming_scale,
            decay_factor=config.decay_factor
        )

        # Full-rank W3: h2→y [output_size, h2_size] (no LoRA)
        self.W3 = torch.empty(config.output_size, h2_size, device=config.device)
        std_w3 = math.sqrt(2.0 / (h2_size + config.output_size))
        torch.nn.init.normal_(self.W3, 0, std_w3)

        # Biases
        self.b1 = torch.zeros(h1_size, device=config.device)
        self.b2 = torch.zeros(h2_size, device=config.device)
        self.by = torch.zeros(config.output_size, device=config.device)

        # Transfer counter
        self.step_counter = 0

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """Activation function: sigmoid (smooth, EP-compatible)"""
        return torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using effective weights.

        Returns (h1, h2, y) at equilibrium after free phase.
        """
        h1, h2, y = self.run_free_phase(x)
        return h1, h2, y

    def run_free_phase(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run free phase dynamics to equilibrium.

        Dynamics (gradient descent on E):
            dh1/dt = W1 x + W2^T h2 - h1
            dh2/dt = W2 h1 + W3^T y - h2
            dy/dt = W3 h2 - y

        Returns:
            (h1_0, h2_0, y0): equilibrium states
        """
        batch_size = x.size(0)
        dt = self.config.dt
        h1_size, h2_size = self.config.hidden_sizes

        # Get effective weights
        W1 = self.lora_1.get_effective_weight()  # [h1_size, input_size]
        W2 = self.lora_2.get_effective_weight()  # [h2_size, h1_size]
        W3 = self.W3                              # [output_size, h2_size]

        # Initialize states
        h1 = torch.zeros(batch_size, h1_size, device=self.device)
        h2 = torch.zeros(batch_size, h2_size, device=self.device)
        y = torch.zeros(batch_size, self.config.output_size, device=self.device)

        # Run dynamics
        for _ in range(self.config.n_iter_free):
            # dh1/dt = W1 @ x + W2^T @ h2 - h1 + b1
            dh1 = F.linear(x, W1) + F.linear(h2, W2.t()) - h1 + self.b1

            # dh2/dt = W2 @ h1 + W3^T @ y - h2 + b2
            dh2 = F.linear(h1, W2) + F.linear(y, W3.t()) - h2 + self.b2

            # dy/dt = W3 @ h2 - y + by
            dy = F.linear(h2, W3) - y + self.by

            # Euler integration with activation
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

        Dynamics with nudging:
            dh1/dt = W1 x + W2^T h2 - h1
            dh2/dt = W2 h1 + W3^T y - h2
            dy/dt = W3 h2 - y + β(target - y)

        Args:
            x: input [batch, 784]
            target: one-hot target [batch, 10]
            h1_0, h2_0, y0: initial states from free phase

        Returns:
            (h1_1, h2_1, y1): nudged equilibrium states
        """
        dt = self.config.dt
        beta = self.config.beta

        # Get effective weights
        W1 = self.lora_1.get_effective_weight()
        W2 = self.lora_2.get_effective_weight()
        W3 = self.W3

        # Start from free phase equilibrium
        h1 = h1_0.clone()
        h2 = h2_0.clone()
        y = y0.clone()

        # Run nudged dynamics
        for _ in range(self.config.n_iter_nudge):
            # dh1/dt = W1 @ x + W2^T @ h2 - h1 + b1
            dh1 = F.linear(x, W1) + F.linear(h2, W2.t()) - h1 + self.b1

            # dh2/dt = W2 @ h1 + W3^T @ y - h2 + b2
            dh2 = F.linear(h1, W2) + F.linear(y, W3.t()) - h2 + self.b2

            # dy/dt = W3 @ h2 - y + by + β(target - y)
            dy = F.linear(h2, W3) - y + self.by + beta * (target - y)

            # Euler integration with activation
            h1 = self.activation(h1 + dt * dh1)
            h2 = self.activation(h2 + dt * dh2)
            y = self.activation(y + dt * dy)

        return h1, h2, y

    def ep_lora_update(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        lr_lora: float,
        lr_full: float
    ) -> float:
        """Compute EP gradients and update weights.

        1) Free phase: (h1_0, h2_0, y0)
        2) Nudged phase: (h1_1, h2_1, y1)
        3) EP gradients:
            dW1 = (h1_1^T @ x - h1_0^T @ x) / β
            dW2 = (h2_1^T @ h1_1 - h2_0^T @ h1_0) / β
            dW3 = (y1^T @ h2_1 - y0^T @ h2_0) / β
        4) For W1, W2: project to LoRA space and update A, B
        5) For W3: direct full-rank update

        Returns:
            loss (MSE for monitoring)
        """
        beta = self.config.beta
        batch_size = x.size(0)

        # 1) Free phase
        h1_0, h2_0, y0 = self.run_free_phase(x)

        # 2) Nudged phase
        h1_1, h2_1, y1 = self.run_nudged_phase(x, target, h1_0, h2_0, y0)

        # 3) EP gradients (average over batch)
        # dW1 = (h1_1^T @ x - h1_0^T @ x) / β  → [h1_size, input_size]
        dW1 = (h1_1.t() @ x - h1_0.t() @ x) / (beta * batch_size)

        # dW2 = (h2_1^T @ h1_1 - h2_0^T @ h1_0) / β  → [h2_size, h1_size]
        dW2 = (h2_1.t() @ h1_1 - h2_0.t() @ h1_0) / (beta * batch_size)

        # dW3 = (y1^T @ h2_1 - y0^T @ h2_0) / β  → [output_size, h2_size]
        dW3 = (y1.t() @ h2_1 - y0.t() @ h2_0) / (beta * batch_size)

        # Bias gradients
        db1 = (h1_1 - h1_0).mean(dim=0) / beta
        db2 = (h2_1 - h2_0).mean(dim=0) / beta
        dby = (y1 - y0).mean(dim=0) / beta

        # 4) Project gradients to LoRA space for W1 and W2
        # For W1: dA1 = dW1 @ B1^T, dB1 = A1^T @ dW1
        dA1 = dW1 @ self.lora_1.B.t()  # [h1_size, rank]
        dB1 = self.lora_1.A.t() @ dW1  # [rank, input_size]

        # For W2: dA2 = dW2 @ B2^T, dB2 = A2^T @ dW2
        dA2 = dW2 @ self.lora_2.B.t()  # [h2_size, rank]
        dB2 = self.lora_2.A.t() @ dW2  # [rank, h1_size]

        # Update LoRA factors (A, B only, not C)
        self.lora_1.update(dA1, dB1, lr_lora)
        self.lora_2.update(dA2, dB2, lr_lora)

        # 5) Direct full-rank update for W3 (no LoRA)
        self.W3.add_(lr_full * dW3)

        # Update biases (use lr_lora for hidden layers, lr_full for output)
        self.b1.add_(lr_lora * db1)
        self.b2.add_(lr_lora * db2)
        self.by.add_(lr_full * dby)

        # Increment counter and check for transfer
        self.step_counter += 1
        if self.step_counter >= self.config.transfer_every:
            self.transfer()
            self.step_counter = 0

        # Return loss for monitoring
        loss = F.mse_loss(y0, target)
        return loss.item()

    def transfer(self) -> None:
        """Transfer A @ B to C for LoRA layers (W1, W2 only)."""
        self.lora_1.transfer()
        self.lora_2.transfer()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions (argmax of output)."""
        _, _, y = self.forward(x)
        return y.argmax(dim=1)


def one_hot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Convert labels to one-hot encoding."""
    return F.one_hot(labels, num_classes).float()


def evaluate(model: EPModel, dataloader: DataLoader) -> float:
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


def train_epoch(model: EPModel, dataloader: DataLoader, epoch: int) -> float:
    """Train for one epoch."""
    total_loss = 0
    n_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.view(-1, 784).to(model.device)
        targets = one_hot(labels).to(model.device)

        loss = model.ep_lora_update(images, targets, model.config.lr_lora, model.config.lr_full)
        total_loss += loss
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}")

    return total_loss / n_batches


def main():
    """Main training loop."""
    print("=" * 60)
    print("EP-LoRA-TT: Equilibrium Propagation + LoRA + Transfer")
    print("=" * 60)

    config = Config()

    # Initialize wandb with run name containing key parameters
    h1_size, h2_size = config.hidden_sizes
    decay_suffix = f"_df{config.decay_factor}" if config.init_mode == "decay" else ""
    run_name = (
        f"lora_h{h1_size}-{h2_size}_r{config.rank}_alpha{config.lora_alpha}_"
        f"init{config.init_mode}_ks{config.kaiming_scale}{decay_suffix}_"
        f"tf{config.transfer_every}_"
        f"free{config.n_iter_free}_nudge{config.n_iter_nudge}_"
        f"beta{config.beta}_dt{config.dt}_"
        f"lrL{config.lr_lora}_lrF{config.lr_full}_bs{config.batch_size}_ep{config.epochs}"
    )
    wandb.init(
        project="ep_lora",
        name=run_name,
        config=asdict(config),
    )

    print(f"\nConfiguration:")
    print(f"  Architecture: {config.input_size} → {h1_size} → {h2_size} → {config.output_size}")
    print(f"  LoRA rank: {config.rank} (layers 1-2 only, layer 3 is full-rank)")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Init mode: {config.init_mode}, kaiming_scale: {config.kaiming_scale}")
    if config.init_mode == "decay":
        print(f"  Decay factor: {config.decay_factor}")
    print(f"  Transfer every: {config.transfer_every} steps")
    print(f"  EP iterations: free={config.n_iter_free}, nudge={config.n_iter_nudge}")
    print(f"  Beta: {config.beta}, dt: {config.dt}")
    print(f"  Learning rate (LoRA): {config.lr_lora}")
    print(f"  Learning rate (Full): {config.lr_full}")
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
    print("Initializing EP-LoRA-TT model...")
    model = EPModel(config)

    # Count parameters
    # W1: h1_size x input_size, W2: h2_size x h1_size, W3: output_size x h2_size
    total_weight_params = (
        h1_size * config.input_size +  # W1
        h2_size * h1_size +             # W2
        config.output_size * h2_size    # W3
    )
    total_bias_params = h1_size + h2_size + config.output_size
    total_params = total_weight_params + total_bias_params

    # LoRA params for W1 and W2 only (W3 is full-rank)
    lora_params = (
        config.rank * (h1_size + config.input_size) +  # A1, B1
        config.rank * (h2_size + h1_size)              # A2, B2
    )
    # W3 is always trained (full-rank)
    w3_params = config.output_size * h2_size

    print(f"  Total weight params: {total_weight_params:,} (+ {total_bias_params} biases)")
    print(f"  LoRA trainable params (W1, W2): {lora_params:,}")
    print(f"  Full-rank trainable params (W3): {w3_params:,}")
    print(f"  Total trainable: {lora_params + w3_params + total_bias_params:,} ({(lora_params + w3_params)/total_weight_params*100:.1f}% of weights)")
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
    total_transfers = model.step_counter + (len(train_loader) * config.epochs) // config.transfer_every
    print(f"Total transfers: {total_transfers}")

    wandb.log({"final_test_acc": final_acc, "total_transfers": total_transfers})
    wandb.finish()


if __name__ == "__main__":
    main()

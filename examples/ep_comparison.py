# -*- coding: utf-8 -*-
"""EP-Full vs EP-LoRA-TT Comparison

Train both models and plot accuracy comparison.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import models
from ep_full import EPFullModel, Config as FullConfig
from ep_lora_tt import EPModel as LoRAModel, Config as LoRAConfig


def one_hot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    return F.one_hot(labels, num_classes).float()


def evaluate(model, dataloader, device) -> float:
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)
        predictions = model.predict(images)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_ep_full(train_loader, test_loader, config):
    """Train EP-Full and return accuracy history."""
    print("\n" + "="*60)
    print("Training EP-Full")
    print("="*60)

    model = EPFullModel(config)
    accuracies = []

    # Initial accuracy
    init_acc = evaluate(model, test_loader, config.device)
    accuracies.append(init_acc)
    print(f"Initial: {init_acc*100:.2f}%")

    for epoch in range(1, config.epochs + 1):
        # Train
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784).to(config.device)
            targets = one_hot(labels).to(config.device)
            model.ep_update(images, targets, config.lr)

        # Evaluate
        acc = evaluate(model, test_loader, config.device)
        accuracies.append(acc)
        print(f"Epoch {epoch}: {acc*100:.2f}%")

    return accuracies


def train_ep_lora(train_loader, test_loader, config):
    """Train EP-LoRA-TT and return accuracy history."""
    print("\n" + "="*60)
    print("Training EP-LoRA-TT")
    print("="*60)

    model = LoRAModel(config)
    accuracies = []

    # Initial accuracy
    init_acc = evaluate(model, test_loader, config.device)
    accuracies.append(init_acc)
    print(f"Initial: {init_acc*100:.2f}%")

    for epoch in range(1, config.epochs + 1):
        # Train
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784).to(config.device)
            targets = one_hot(labels).to(config.device)
            model.ep_lora_update(images, targets, config.lr_lora, config.lr_full)

        # Evaluate
        acc = evaluate(model, test_loader, config.device)
        accuracies.append(acc)
        print(f"Epoch {epoch}: {acc*100:.2f}%")

    return accuracies


def plot_comparison(full_acc, lora_acc, epochs, save_path="ep_comparison.png"):
    """Plot accuracy comparison."""
    plt.figure(figsize=(10, 6))

    x = list(range(epochs + 1))  # 0 to epochs (including initial)

    plt.plot(x, [a*100 for a in full_acc], 'b-o', label='EP-Full', linewidth=2, markersize=6)
    plt.plot(x, [a*100 for a in lora_acc], 'r-s', label='EP-LoRA-TT (rank=8)', linewidth=2, markersize=6)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('EP-Full vs EP-LoRA-TT on MNIST', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, epochs + 0.5)
    plt.ylim(0, 100)

    # Add final accuracy text
    plt.text(epochs, full_acc[-1]*100 + 2, f'{full_acc[-1]*100:.1f}%', ha='center', fontsize=10, color='blue')
    plt.text(epochs, lora_acc[-1]*100 - 4, f'{lora_acc[-1]*100:.1f}%', ha='center', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load MNIST
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # EP-Full config
    full_config = FullConfig(device=device)

    # EP-LoRA config
    lora_config = LoRAConfig(device=device)

    # Use epochs from config (should be same for both)
    epochs = full_config.epochs

    # Train both
    full_acc = train_ep_full(train_loader, test_loader, full_config)
    lora_acc = train_ep_lora(train_loader, test_loader, lora_config)

    # Plot
    plot_comparison(full_acc, lora_acc, epochs)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'ep_full': {
            'accuracies': full_acc,
            'final': full_acc[-1]
        },
        'ep_lora': {
            'accuracies': lora_acc,
            'final': lora_acc[-1],
            'rank': lora_config.rank,
            'transfer_every': lora_config.transfer_every
        }
    }

    with open('ep_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"EP-Full Final Accuracy:     {full_acc[-1]*100:.2f}%")
    print(f"EP-LoRA-TT Final Accuracy:  {lora_acc[-1]*100:.2f}%")
    print(f"Gap: {(full_acc[-1] - lora_acc[-1])*100:.2f}%")


if __name__ == "__main__":
    main()

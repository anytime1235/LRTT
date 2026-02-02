#!/usr/bin/env python3
"""Check variance of training results to determine N_RUNS_PER_TRIAL."""

import torch
import numpy as np
from torch.utils.data import DataLoader

from regression_lrtt_scratch_decay import (
    ScratchExperimentConfig,
    train_lrtt_scratch,
    generate_target_dataset,
    DEVICE,
)

def run_experiment(config, seed):
    """Run one experiment and return final loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = generate_target_dataset("medium", config, train=True, seed=seed)
    val_dataset = generate_target_dataset("medium", config, train=False, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=config.lrtt_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.lrtt_batch_size, shuffle=False)

    model, history, _, _, _, _ = train_lrtt_scratch(
        config, train_loader, val_loader,
        seed=seed, use_wandb=False
    )

    if history:
        return history[-1].get('batch_loss', float('inf'))
    return float('inf')


def main():
    config = ScratchExperimentConfig()
    # Use default lrtt_epochs (2000) with early stopping (patience=7)
    config.log_ab_scaling = False

    n_runs = 50  # Run 50 times to measure variance
    losses = []

    print(f"Running {n_runs} experiments with same hyperparameters...")
    print(f"Device: {DEVICE}")
    print()

    for i in range(n_runs):
        seed = 42 + i * 100
        loss = run_experiment(config, seed)
        losses.append(loss)
        print(f"  Run {i+1:2d}: Loss = {loss:.6f}")

    losses = np.array(losses)

    print()
    print("="*50)
    print("VARIANCE ANALYSIS")
    print("="*50)
    print(f"Mean:   {np.mean(losses):.6f}")
    print(f"Std:    {np.std(losses):.6f}")
    print(f"Min:    {np.min(losses):.6f}")
    print(f"Max:    {np.max(losses):.6f}")
    print(f"Range:  {np.max(losses) - np.min(losses):.6f}")
    print()

    # Show how variance decreases with averaging
    print("Estimated std of mean with N runs:")
    for n in [1, 3, 5, 10, 15, 20, 30, 50]:
        std_of_mean = np.std(losses) / np.sqrt(n)
        print(f"  N={n:2d}: std_of_mean = {std_of_mean:.6f}")

    print()
    # Test multiple thresholds
    for thresh_ratio in [0.1, 0.2, 0.3, 0.5]:
        threshold = thresh_ratio * np.mean(losses)
        print(f"Recommendation: Choose N where std_of_mean < {thresh_ratio} * mean (={threshold:.6f})")
        for n in [1, 3, 5, 10, 15, 20, 30, 50]:
            std_of_mean = np.std(losses) / np.sqrt(n)
            status = "OK" if std_of_mean < threshold else ""
            print(f"  N={n:2d}: {std_of_mean:.6f} < {threshold:.6f} ? {status}")
        print()


if __name__ == "__main__":
    main()

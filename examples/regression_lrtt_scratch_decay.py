#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LRTT-LoRA scratch training experiment for 4×4 multi-output regression task.

This experiment demonstrates:
- Direct LRTT training on dataset D' from scratch (no pre-training on D)
- Comparison with fine-tuning approach
"""

import os
import json
from datetime import datetime
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from aihwkit.optim import AnalogSGD, AnalogAdam
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import wandb

# AIHWKit imports
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice, ConstantStepDevice
from aihwkit.simulator.configs import MappingParameter, SingleRPUConfig, UpdateParameters
from aihwkit.simulator.parameters import PulseType

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================================
# Configuration
# ============================================================================
class ScratchExperimentConfig:
    """Configuration for the LRTT scratch experiment."""

    # Random seeds
    seeds = [42, 1, 2, 3, 4, 5]
    primary_seed = 42

    # Data dimensions
    input_dim = 4
    output_dim = 4

    # Dataset sizes
    D_prime_train_size = 800
    D_prime_test_size = 200

    # Noise parameters
    noise_std = 0.02

    # Training hyperparameters - LRTT scratch training
    lrtt_lr = 0.1  # Very conservative LR
    lrtt_epochs = 1000
    lrtt_batch_size = 1
    lrtt_patience = 20  # Allow a bit more training than fine-tuning
    lrtt_grad_clip = 2.0  # Conservative clipping

    # LRTT configuration
    lrtt_rank = 1  # Rank-1 for minimal overfitting
    lrtt_transfer_every = 800  # Medium frequency to observe transfer effects
    lora_alpha = 2.0  # Conservative scaling

    # Reinit configuration - DECAY MODE
    reinit_mode = "decay"  # Use decay mode instead of standard
    decay_factor = 0.9  # Decay A,B weights to 90%

    # Results directory
    results_dir = "results/lrtt_scratch_decay"

# ============================================================================
# Data Generation (simplified for scratch training)
# ============================================================================

def generate_target_matrix(complexity_level: str, config: ScratchExperimentConfig, seed: int = 42) -> torch.Tensor:
    """Generate the target matrix T' with values in [-1, 1] range."""
    torch.manual_seed(seed)  # Use consistent seed for T' generation
    complexity_scales = {
        'simple': 0.5,    # Simple target function
        'medium': 0.8,    # Medium complexity target function
        'complex': 1.0    # Complex target function (full range)
    }
    scale = complexity_scales.get(complexity_level, 0.8)

    # Generate uniform random values in [-scale, scale] range
    T_prime = (torch.rand(config.output_dim, config.input_dim) * 2 - 1) * scale
    return T_prime


def generate_target_dataset(complexity_level: str, config: ScratchExperimentConfig,
                           train: bool = True, seed: int = 42) -> TensorDataset:
    """Generate dataset with target function T' for scratch training.

    No baseline needed - T' is the actual target function to learn.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed + 1000 if train else seed + 2000)
    size = config.D_prime_train_size if train else config.D_prime_test_size

    # Generate inputs from uniform distribution
    X = torch.rand(size, config.input_dim) * 2 - 1  # U([-1, 1]^4)

    # Generate target matrix T' directly based on complexity level
    T_prime = generate_target_matrix(complexity_level, config, seed)

    # Generate labels with target transition matrix
    Y = X @ T_prime.T
    Y += torch.randn_like(Y) * config.noise_std  # Add output noise

    # Print target statistics for first call only
    if train and seed == 42:
        print(f"[Target T'] Complexity level: {complexity_level}, ‖T'‖: {T_prime.norm():.3f}")

    return TensorDataset(X, Y)


# ============================================================================
# Model Definitions
# ============================================================================
class LRTTModel(nn.Module):
    """LRTT model for scratch training."""

    def __init__(self, config: ScratchExperimentConfig, pretrained_C: torch.Tensor = None):
        super().__init__()

        # Create LRTT configuration
        from aihwkit.simulator.configs import IOParameters
        from aihwkit.simulator.parameters import WeightNoiseType, BoundManagementType, NoiseManagementType

        device_config = PythonLRTTDevice(
            rank=config.lrtt_rank,
            transfer_every=config.lrtt_transfer_every,
            lora_alpha=config.lora_alpha,
            reinit_mode=config.reinit_mode,
            decay_factor=config.decay_factor,
            forward_inject=False,
            correct_gradient_magnitudes=True,
            unit_cell_devices=[
                IdealizedPresetDevice(),  # A matrix
                IdealizedPresetDevice(),  # B matrix
                IdealizedPresetDevice(),  # C matrix
            ]
        )

        device_config.transfer_lr = device_config.lora_alpha

        # I/O configuration
        forward_io = IOParameters(
            inp_res=0.007937,
            inp_bound=1.0,
            out_res=0.001961,
            out_bound=12.0,
            out_noise=0.06
        )

        backward_io = IOParameters(
            inp_res=0.007937,
            inp_bound=1.0,
            out_res=0.001961,
            out_bound=12.0,
            out_noise=0.06
        )

        mapping = MappingParameter()

        rpu_config = PythonLRTTRPUConfig(
            device=device_config,
            mapping=mapping,
            forward=forward_io,
            backward=backward_io
        )

        # Create LRTT linear layer
        self.lrtt_layer = AnalogLinear(
            config.input_dim,
            config.output_dim,
            bias=False,
            rpu_config=rpu_config
        )

        # Initialize C with pretrained weights if provided, otherwise all ones
        if pretrained_C is not None:
            self.set_C_weights(pretrained_C)
        else:
            # Set C matrix to 0.999 (safer than 1.0 for analog tiles with [-1,1] bounds)
            C_init = torch.ones(config.output_dim, config.input_dim) * 0.999
            self.set_C_weights(C_init)

    def set_C_weights(self, C: torch.Tensor):
        """Set the C matrix."""
        analog_tile = self.lrtt_layer.analog_module
        print(f"Setting C weights to shape {C.shape}, all ones? {torch.allclose(C, torch.ones_like(C))}")
        analog_tile.set_weights(C.to(DEVICE))
        # Verify what was actually set
        C_actual = analog_tile.get_weights()[0]
        print(f"After set_weights - C norm: {C_actual.norm():.4f}, all ones? {torch.allclose(C_actual, torch.ones_like(C_actual))}")
        print(f"C actual values (first 5): {C_actual.flatten()[:5]}")

    def forward(self, x):
        return self.lrtt_layer(x)

    def get_lrtt_components(self):
        """Get the C, A, B components from LRTT layer."""
        analog_tile = self.lrtt_layer.analog_module
        if hasattr(analog_tile, 'get_lrtt_component_weights'):
            C, A, B = analog_tile.get_lrtt_component_weights()
            return C, A, B
        return None, None, None


# ============================================================================
# Training Functions
# ============================================================================
def train_lrtt_scratch(config: ScratchExperimentConfig,
                      train_loader: DataLoader, val_loader: DataLoader,
                      seed: int = 42, use_wandb: bool = True) -> LRTTModel:
    """Train LRTT from scratch on D'."""

    print("\n" + "="*60)
    print("SCRATCH TRAINING: LRTT directly on D'")
    print("="*60)

    torch.manual_seed(seed)

    # Create LRTT model without pre-trained C (random initialization)
    model = LRTTModel(config, pretrained_C=None).to(DEVICE)

    # Debug: Check initial A,B,C
    C_init, A_init, B_init = model.get_lrtt_components()
    if A_init is not None and B_init is not None:
        print(f"Initial: A norm={A_init.norm():.4f}, B norm={B_init.norm():.4f}, C norm={C_init.norm():.4f}")
        print(f"A: {A_init.flatten()[:4]}, B: {B_init.flatten()[:4]}")
        print(f"C shape: {C_init.shape}, C values (first 5): {C_init.flatten()[:5]}")
        print(f"Is C all ones? {torch.allclose(C_init, torch.ones_like(C_init))}")

    # Use AnalogSGD for LRTT tiles (no momentum, simple vanilla SGD)
    optimizer = AnalogSGD(model.parameters(), lr=config.lrtt_lr, momentum=0.0)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.lrtt_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = F.mse_loss(Y_pred, Y_batch)

            loss.backward()

            # Debug: Check gradients on first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"\n  Debug Gradients (Epoch {epoch}, Batch {batch_idx}):")
                print(f"    Loss value: {loss.item():.6f}")
                print(f"    Learning rate: {config.lrtt_lr}")

                C, A, B = model.get_lrtt_components()
                print(f"    Before step - A norm: {A.norm():.6f}, B norm: {B.norm():.6f}")

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.lrtt_grad_clip)
            optimizer.step()

            # Debug: Check if A changed after optimizer step
            if epoch == 0 and batch_idx == 0:
                C, A, B = model.get_lrtt_components()
                print(f"    After step - A norm: {A.norm():.6f}, B norm: {B.norm():.6f}")

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Check for transfer event
        analog_tile = model.lrtt_layer.analog_module
        if hasattr(analog_tile, 'controller'):
            current_transfer_counter = analog_tile.controller.transfer_counter
            if epoch > 0 and current_transfer_counter % config.lrtt_transfer_every == 0:
                # Transfer just occurred
                C, A, B = model.get_lrtt_components()
                print(f"\n  *** TRANSFER DETECTED at epoch {epoch} ***")
                print(f"  Transfer counter: {current_transfer_counter}")
                print(f"  After transfer - A norm: {A.norm():.6f}, B norm: {B.norm():.6f}, C norm: {C.norm():.6f}")
                print(f"  A values (first 5): {A.flatten()[:5]}")
                print(f"  B values (first 5): {B.flatten()[:5]}")
                print(f"  A max: {A.abs().max():.6f}, A min: {A.abs().min():.6f}")
                print(f"  B max: {B.abs().max():.6f}, B min: {B.abs().min():.6f}")
                print(f"  Is A all zeros? {torch.allclose(A, torch.zeros_like(A))}")
                print(f"  Is B all zeros? {torch.allclose(B, torch.zeros_like(B))}")

        # Validation - Use full LRTT model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                Y_pred = model(X_batch)
                loss = F.mse_loss(Y_pred, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Wandb logging
        if use_wandb:
            log_dict = {
                'scratch/epoch': epoch,
                'scratch/train_loss': train_loss,
                'scratch/val_loss': val_loss
            }

            # Log component norms if available
            C, A, B = model.get_lrtt_components()
            if A is not None and B is not None:
                log_dict.update({
                    'scratch/A_norm': A.norm().item(),
                    'scratch/B_norm': B.norm().item(),
                    'scratch/delta_norm': (A @ B).norm().item()
                })

            wandb.log(log_dict)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            # Debug A,B updates
            C, A, B = model.get_lrtt_components()
            if A is not None:
                print(f"  -> A norm={A.norm():.4f}, B norm={B.norm():.4f}")

        if patience_counter >= config.lrtt_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Best validation loss: {best_val_loss:.6f}")

    # Print final a,b norms and debug info
    C, A, B = model.get_lrtt_components()
    if A is not None and B is not None:
        print(f"Final ‖A‖ = {A.norm():.4f}, ‖B‖ = {B.norm():.4f}")
        AB_product = A @ B
        print(f"‖ΔW‖_F = ‖A⊗B‖ = {AB_product.norm():.4f}")
        print(f"A shape: {A.shape}, B shape: {B.shape}, A⊗B shape: {AB_product.shape}")
        print(f"A values: {A.flatten()[:5]}")
        print(f"B values: {B.flatten()[:5]}")
        print(f"A⊗B values: {AB_product.flatten()[:5]}")

    return model


# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """Evaluate model performance on test set."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            Y_pred = model(X_batch)
            all_preds.append(Y_pred.cpu())
            all_targets.append(Y_batch.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    mse = F.mse_loss(preds, targets).item()
    rmse = np.sqrt(mse)

    # R² score
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    results = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
    }

    return results


def compare_matrices(learned_model: nn.Module, target_matrix: torch.Tensor) -> Dict[str, float]:
    """Compare learned matrix with target matrix."""
    learned_model.eval()

    # Extract learned weight matrix from the model
    with torch.no_grad():
        # Get the combined matrix (C + A⊗B)
        C, A, B = learned_model.get_lrtt_components()
        if A is not None and B is not None:
            learned_matrix = C + A @ B
        else:
            learned_matrix = C

    # Move target to same device
    target_matrix = target_matrix.to(learned_matrix.device)

    # Compute differences
    diff_matrix = learned_matrix - target_matrix
    frobenius_diff = torch.norm(diff_matrix, 'fro').item()
    mse_matrix = F.mse_loss(learned_matrix, target_matrix).item()

    # Relative error
    target_norm = torch.norm(target_matrix, 'fro').item()
    relative_error = frobenius_diff / target_norm if target_norm > 0 else float('inf')

    return {
        'frobenius_diff': frobenius_diff,
        'mse_matrix': mse_matrix,
        'relative_error': relative_error,
        'target_norm': target_norm,
        'learned_norm': torch.norm(learned_matrix, 'fro').item()
    }


def run_scratch_experiment(config: ScratchExperimentConfig, complexity_level: str = 'medium',
                          seed: int = 42, use_wandb: bool = True) -> Dict[str, any]:
    """Run LRTT experiment from scratch on target dataset."""

    print(f"\n{'='*60}")
    print(f"Running SCRATCH experiment with seed={seed}, complexity_level={complexity_level}")
    print(f"REINIT CONFIG: mode={config.reinit_mode}, decay_factor={config.decay_factor}")
    print(f"{'='*60}")

    # Initialize wandb run for this experiment
    if use_wandb:
        run_name = f"lrtt_scratch_{complexity_level}_r{config.lrtt_rank}_t{config.lrtt_transfer_every}_alpha{config.lora_alpha}_LR{config.lrtt_lr}"

        wandb.init(
            project="aihwkit-lrtt-scratch",
            name=run_name,
            config={
                'experiment_type': 'LRTT-scratch',
                'complexity_level': complexity_level,
                'seed': seed,
                'lrtt_lr': config.lrtt_lr,
                'lrtt_rank': config.lrtt_rank,
                'transfer_every': config.lrtt_transfer_every,
                'lora_alpha': config.lora_alpha,
            },
            reinit=True
        )

    # Generate target matrix and datasets
    target_matrix = generate_target_matrix(complexity_level, config, seed)
    target_train = generate_target_dataset(complexity_level, config, train=True, seed=seed)
    target_test = generate_target_dataset(complexity_level, config, train=False, seed=seed)

    if use_wandb:
        wandb.log({
            'data/complexity_level': complexity_level,
        })

    # Create data loaders
    target_train_loader = DataLoader(target_train, batch_size=config.lrtt_batch_size, shuffle=True)
    target_test_loader = DataLoader(target_test, batch_size=config.lrtt_batch_size)

    # Train LRTT from scratch on target dataset
    lrtt_model = train_lrtt_scratch(config, target_train_loader,
                                   target_test_loader, seed, use_wandb)

    # Evaluate LRTT on target dataset
    lrtt_results = evaluate_model(lrtt_model, target_test_loader)

    # Compare learned matrix with target matrix
    matrix_comparison = compare_matrices(lrtt_model, target_matrix)

    print(f"\nLRTT (scratch) on target: MSE={lrtt_results['MSE']:.6f}, R²={lrtt_results['R2']:.4f}")
    print(f"\nMatrix Comparison:")
    print(f"  Target ‖T'‖: {matrix_comparison['target_norm']:.4f}")
    print(f"  Learned ‖W‖: {matrix_comparison['learned_norm']:.4f}")
    print(f"  Frobenius diff ‖W-T'‖: {matrix_comparison['frobenius_diff']:.4f}")
    print(f"  Matrix MSE: {matrix_comparison['mse_matrix']:.6f}")
    print(f"  Relative error: {matrix_comparison['relative_error']:.4f} ({matrix_comparison['relative_error']*100:.1f}%)")

    if use_wandb:
        wandb.log({
            'lrtt/mse': lrtt_results['MSE'],
            'lrtt/rmse': lrtt_results['RMSE'],
            'lrtt/r2': lrtt_results['R2'],
            'matrix/frobenius_diff': matrix_comparison['frobenius_diff'],
            'matrix/mse': matrix_comparison['mse_matrix'],
            'matrix/relative_error': matrix_comparison['relative_error'],
        })

        wandb.summary['final_lrtt_mse'] = lrtt_results['MSE']
        wandb.summary['final_lrtt_r2'] = lrtt_results['R2']
        wandb.summary['matrix_relative_error'] = matrix_comparison['relative_error']
        wandb.finish()

    # Compile results
    results = {
        'seed': seed,
        'complexity_level': complexity_level,
        'lrtt_results': lrtt_results,
        'matrix_comparison': matrix_comparison,
        'experiment_type': 'scratch'
    }

    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================
def main(use_wandb: bool = True):
    """Main experiment runner for scratch training."""

    config = ScratchExperimentConfig()

    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments across noise levels
    all_results = []

    for complexity_level in ['low', 'medium', 'high']:
        print(f"\n{'#'*60}")
        print(f"# Running SCRATCH experiments for {complexity_level.upper()} complexity level")
        print(f"{'#'*60}")

        for seed in config.seeds[:1]:  # Start with just primary seed
            results = run_scratch_experiment(config, complexity_level, seed, use_wandb)
            all_results.append(results)

        # Print summary for this complexity level
        level_results = [r for r in all_results if r['complexity_level'] == complexity_level]
        mse_lrtt = [r['lrtt_results']['MSE'] for r in level_results]
        r2_lrtt = [r['lrtt_results']['R2'] for r in level_results]
        relative_errors = [r['matrix_comparison']['relative_error'] for r in level_results]

        print(f"\n{complexity_level.upper()} SCRATCH Summary (n={len(level_results)}):")
        print(f"  LRTT MSE: {np.mean(mse_lrtt):.6f} ± {np.std(mse_lrtt):.6f}")
        print(f"  LRTT R²: {np.mean(r2_lrtt):.4f} ± {np.std(r2_lrtt):.4f}")
        print(f"  Matrix Relative Error: {np.mean(relative_errors):.4f} ± {np.std(relative_errors):.4f} ({np.mean(relative_errors)*100:.1f}%)")

    # Save results to JSON
    results_file = os.path.join(config.results_dir, f"lrtt_scratch_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print final summary table
    print("\n" + "="*75)
    print("SCRATCH EXPERIMENT SUMMARY")
    print("="*75)
    print(f"{'Complexity':<12} {'LRTT MSE':<15} {'LRTT R²':<15} {'Matrix Error':<15}")
    print("-" * 75)

    for complexity_level in ['low', 'medium', 'high']:
        level_results = [r for r in all_results if r['complexity_level'] == complexity_level]
        if level_results:
            lrtt_mse = np.mean([r['lrtt_results']['MSE'] for r in level_results])
            lrtt_r2 = np.mean([r['lrtt_results']['R2'] for r in level_results])
            rel_error = np.mean([r['matrix_comparison']['relative_error'] for r in level_results])

            print(f"{complexity_level.upper():<12} {lrtt_mse:<15.6f} {lrtt_r2:<15.4f} {rel_error:<15.4f}")

    print("="*75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LRTT Scratch Experiment')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    main(use_wandb=not args.no_wandb)
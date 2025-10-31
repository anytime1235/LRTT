#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Baseline 4×4 training experiment without LRTT for multi-output regression task.

This experiment demonstrates:
- Direct base 4x4 analog training on dataset D' from scratch (no LRTT)
- Comparison baseline for LRTT approaches
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
import openpyxl

# AIHWKit imports
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import FloatingPointRPUConfig
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
class BaselineExperimentConfig:
    """Configuration for the baseline experiment."""

    # Random seeds
    seeds = [42, 1, 2, 3, 4, 5]
    primary_seed = 42

    # Data dimensions
    input_dim = 4
    output_dim = 4

    # Dataset sizes
    D_prime_train_size = 20
    D_prime_test_size = 20

    # Noise parameters
    noise_std = 0.02

    # Training hyperparameters - baseline training
    baseline_lr = 0.3  # Same LR as LRTT for fair comparison
    baseline_epochs = 40
    baseline_batch_size = 1
    baseline_patience = 10
    baseline_grad_clip = 2.0

    # Pulse configuration
    desired_bl = 31  # Bit length for stochastic pulsing

    # Results directory
    results_dir = "results/baseline_scratch"

# ============================================================================
# Data Generation (same as LRTT version)
# ============================================================================

def generate_target_matrix(complexity_level: str, config: BaselineExperimentConfig, seed: int = 42) -> torch.Tensor:
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


def save_dataset_to_excel(dataset: TensorDataset, target_matrix: torch.Tensor,
                         filename: str, config: BaselineExperimentConfig,
                         is_train: bool = True, final_epoch: int = None) -> None:
    """Save dataset and target matrix to Excel file for hardware experiments.

    Args:
        dataset: TensorDataset containing (X, Y) pairs
        target_matrix: Target matrix T'
        filename: Path to save Excel file
        config: Experiment configuration
    """
    X, Y = dataset.tensors

    # Create dataframes for inputs and outputs
    X_df = pd.DataFrame(X.numpy(), columns=[f'X{i+1}' for i in range(X.shape[1])])
    Y_df = pd.DataFrame(Y.numpy(), columns=[f'Y{i+1}' for i in range(Y.shape[1])])

    # Combine into single dataframe
    dataset_df = pd.concat([X_df, Y_df], axis=1)

    # Create dataframe for target matrix
    T_df = pd.DataFrame(target_matrix.numpy(),
                       columns=[f'Input_{i+1}' for i in range(target_matrix.shape[1])],
                       index=[f'Output_{i+1}' for i in range(target_matrix.shape[0])])

    # Create Excel writer and save to multiple sheets
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Save dataset
        dataset_df.to_excel(writer, sheet_name='Dataset', index=False)

        # Save target matrix
        T_df.to_excel(writer, sheet_name='Target_Matrix')

        # Save configuration info
        config_data = {
            'Parameter': ['Input Dimension', 'Output Dimension', 'Train Size', 'Test Size',
                         'Dataset Type', 'Dataset Size', 'Noise Std',
                         'Learning Rate', 'Max Epochs', 'Early Stop Patience', 'Final Epoch',
                         'Batch Size', 'Gradient Clip',
                         'Model Type'],
            'Value': [config.input_dim, config.output_dim, config.D_prime_train_size, config.D_prime_test_size,
                     'Train' if is_train else 'Test', len(dataset), config.noise_std,
                     config.baseline_lr, config.baseline_epochs, config.baseline_patience, final_epoch if final_epoch else 'N/A',
                     config.baseline_batch_size, config.baseline_grad_clip,
                     'Baseline (4x4 Analog)']
        }
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='Config', index=False)

    print(f"Dataset saved to: {filename}")


def generate_target_dataset(complexity_level: str, config: BaselineExperimentConfig,
                           train: bool = True, seed: int = 42, save_excel: bool = False,
                           final_epoch: int = None) -> TensorDataset:
    """Generate dataset with target function T' for scratch training.

    No baseline needed - T' is the actual target function to learn.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed + 1000 if train else seed + 2000)
    size = config.D_prime_train_size if train else config.D_prime_test_size

    # Generate inputs from uniform distribution
    X = torch.rand(size, config.input_dim)  # U([0, 1]^4)

    # Generate target matrix T' directly based on complexity level
    T_prime = generate_target_matrix(complexity_level, config, seed)

    # Generate labels with target transition matrix
    Y = X @ T_prime.T
    Y += torch.randn_like(Y) * config.noise_std  # Add output noise

    # Print target statistics for first call only
    if train and seed == 42:
        print(f"[Target T'] Complexity level: {complexity_level}, ‖T'‖: {T_prime.norm():.3f}")

    dataset = TensorDataset(X, Y)

    # Save to Excel if requested
    if save_excel:
        dataset_type = "train" if train else "test"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = os.path.join(
            config.results_dir,
            f"dataset_{complexity_level}_{dataset_type}_seed{seed}_{timestamp}.xlsx"
        )
        os.makedirs(config.results_dir, exist_ok=True)
        save_dataset_to_excel(dataset, T_prime, excel_filename, config, is_train=train, final_epoch=final_epoch)

    return dataset


# ============================================================================
# Model Definitions
# ============================================================================
class BaselineModel(nn.Module):
    """Baseline analog 4x4 model without LRTT."""

    def __init__(self, config: BaselineExperimentConfig):
        super().__init__()

        # Create standard analog configuration using IdealizedPresetDevice
        from aihwkit.simulator.configs import IOParameters

        # I/O configuration (same as LRTT)
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

        # Update configuration with explicit BL
        update_params = UpdateParameters(
            desired_bl=config.desired_bl,
            fixed_bl=True,
            pulse_type=PulseType.STOCHASTIC_COMPRESSED,
            update_bl_management=False,
            update_management=True
        )

        mapping = MappingParameter()

        # Create SingleRPUConfig with IdealizedPresetDevice
        rpu_config = SingleRPUConfig(
            device=IdealizedPresetDevice(),
            mapping=mapping,
            forward=forward_io,
            backward=backward_io,
            update=update_params
        )

        # Create standard analog linear layer
        self.analog_layer = AnalogLinear(
            config.input_dim,
            config.output_dim,
            bias=False,
            rpu_config=rpu_config
        )

        # Print configuration for verification
        print(f"[BL CONFIG] Using desired_bl = {rpu_config.update.desired_bl}")
        print(f"[MODEL] Using baseline 4x4 analog layer (no LRTT)")

    def forward(self, x):
        return self.analog_layer(x)

    def get_weights(self):
        """Get the weight matrix from analog layer."""
        with torch.no_grad():
            weights = self.analog_layer.get_weights()[0]
        return weights


# ============================================================================
# Training Functions
# ============================================================================
def train_baseline(config: BaselineExperimentConfig,
                   train_loader: DataLoader, val_loader: DataLoader,
                   seed: int = 42, use_wandb: bool = True) -> Tuple[BaselineModel, int]:
    """Train baseline 4x4 analog model from scratch on D'."""

    print("\n" + "="*60)
    print("BASELINE TRAINING: Standard 4x4 analog on D'")
    print("="*60)

    torch.manual_seed(seed)

    # Create baseline model
    model = BaselineModel(config).to(DEVICE)

    # Debug: Check initial weights
    W_init = model.get_weights()
    print(f"Initial: W norm={W_init.norm():.4f}")

    # Use AnalogSGD (same as LRTT for fair comparison)
    optimizer = AnalogSGD(model.parameters(), lr=config.baseline_lr, momentum=0.0)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    final_epoch = 0

    for epoch in range(config.baseline_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = F.mse_loss(Y_pred, Y_batch)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.baseline_grad_clip)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)


        # Validation
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
                'baseline/epoch': epoch,
                'baseline/train_loss': train_loss,
                'baseline/val_loss': val_loss,
                'baseline/weight_norm': model.get_weights().norm().item()
            }
            wandb.log(log_dict)

        if epoch % 1 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if patience_counter >= config.baseline_patience:
            print(f"Early stopping at epoch {epoch}")
            final_epoch = epoch
            break

        final_epoch = epoch

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Best validation loss: {best_val_loss:.6f}")

    # Print final weight norm
    W_final = model.get_weights()
    print(f"Final ‖W‖ = {W_final.norm():.4f}")

    return model, final_epoch


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


def compare_matrices(learned_model: BaselineModel, target_matrix: torch.Tensor) -> Dict[str, float]:
    """Compare learned matrix with target matrix."""
    learned_model.eval()

    # Extract learned weight matrix from the model
    learned_matrix = learned_model.get_weights()

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


def run_baseline_experiment(config: BaselineExperimentConfig, complexity_level: str = 'medium',
                            seed: int = 42, use_wandb: bool = True, save_excel: bool = False) -> Dict[str, any]:
    """Run baseline experiment from scratch on target dataset."""

    print(f"\n{'='*60}")
    print(f"Running BASELINE experiment with seed={seed}, complexity_level={complexity_level}")
    print(f"{'='*60}")

    # Initialize wandb run for this experiment
    if use_wandb:
        run_name = f"baseline_scratch_{complexity_level}_LR{config.baseline_lr}"

        wandb.init(
            project="aihwkit-baseline-scratch",
            name=run_name,
            config={
                'experiment_type': 'baseline-scratch',
                'complexity_level': complexity_level,
                'seed': seed,
                'baseline_lr': config.baseline_lr,
            },
            reinit=True
        )

    # Generate target matrix and datasets (will be saved after we know final_epoch)
    target_matrix = generate_target_matrix(complexity_level, config, seed)
    target_train = generate_target_dataset(complexity_level, config, train=True, seed=seed, save_excel=False)
    target_test = generate_target_dataset(complexity_level, config, train=False, seed=seed, save_excel=False)

    if use_wandb:
        wandb.log({
            'data/complexity_level': complexity_level,
        })

    # Create data loaders
    target_train_loader = DataLoader(target_train, batch_size=config.baseline_batch_size, shuffle=True)
    target_test_loader = DataLoader(target_test, batch_size=config.baseline_batch_size)

    # Create untrained model for baseline comparison
    torch.manual_seed(seed)
    untrained_model = BaselineModel(config).to(DEVICE)

    # Evaluate untrained model
    untrained_results = evaluate_model(untrained_model, target_test_loader)
    print(f"\nUntrained model baseline: MSE={untrained_results['MSE']:.6f}, R²={untrained_results['R2']:.4f}")

    # Train baseline from scratch on target dataset
    baseline_model, final_epoch = train_baseline(config, target_train_loader,
                                                 target_test_loader, seed, use_wandb)

    # Evaluate baseline on target dataset
    baseline_results = evaluate_model(baseline_model, target_test_loader)

    # Save datasets to Excel if requested (now we know final_epoch)
    if save_excel:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save training dataset
        train_excel_filename = os.path.join(
            config.results_dir,
            f"dataset_{complexity_level}_train_seed{seed}_{timestamp}.xlsx"
        )
        save_dataset_to_excel(target_train, target_matrix, train_excel_filename,
                            config, is_train=True, final_epoch=final_epoch)

        # Save test dataset
        test_excel_filename = os.path.join(
            config.results_dir,
            f"dataset_{complexity_level}_test_seed{seed}_{timestamp}.xlsx"
        )
        save_dataset_to_excel(target_test, target_matrix, test_excel_filename,
                            config, is_train=False, final_epoch=final_epoch)

    # Compare learned matrix with target matrix
    matrix_comparison = compare_matrices(baseline_model, target_matrix)

    print(f"\n{'='*50}")
    print("PERFORMANCE COMPARISON:")
    print(f"  Untrained:  MSE={untrained_results['MSE']:.6f}, R²={untrained_results['R2']:.4f}")
    print(f"  Trained:    MSE={baseline_results['MSE']:.6f}, R²={baseline_results['R2']:.4f}")
    print(f"  Improvement: {(untrained_results['MSE'] - baseline_results['MSE'])/untrained_results['MSE']*100:.1f}% MSE reduction")
    print(f"\nMatrix Comparison:")
    print(f"  Target ‖T'‖: {matrix_comparison['target_norm']:.4f}")
    print(f"  Learned ‖W‖: {matrix_comparison['learned_norm']:.4f}")
    print(f"  Frobenius diff ‖W-T'‖: {matrix_comparison['frobenius_diff']:.4f}")
    print(f"  Matrix MSE: {matrix_comparison['mse_matrix']:.6f}")
    print(f"  Relative error: {matrix_comparison['relative_error']:.4f} ({matrix_comparison['relative_error']*100:.1f}%)")
    print('='*50)

    if use_wandb:
        wandb.log({
            'untrained/mse': untrained_results['MSE'],
            'untrained/rmse': untrained_results['RMSE'],
            'untrained/r2': untrained_results['R2'],
            'baseline/mse': baseline_results['MSE'],
            'baseline/rmse': baseline_results['RMSE'],
            'baseline/r2': baseline_results['R2'],
            'improvement/mse_reduction_pct': (untrained_results['MSE'] - baseline_results['MSE'])/untrained_results['MSE']*100,
            'matrix/frobenius_diff': matrix_comparison['frobenius_diff'],
            'matrix/mse': matrix_comparison['mse_matrix'],
            'matrix/relative_error': matrix_comparison['relative_error'],
        })

        wandb.summary['final_baseline_mse'] = baseline_results['MSE']
        wandb.summary['final_baseline_r2'] = baseline_results['R2']
        wandb.summary['matrix_relative_error'] = matrix_comparison['relative_error']
        wandb.finish()

    # Compile results
    results = {
        'seed': seed,
        'complexity_level': complexity_level,
        'untrained_results': untrained_results,
        'baseline_results': baseline_results,
        'improvement_pct': (untrained_results['MSE'] - baseline_results['MSE'])/untrained_results['MSE']*100,
        'matrix_comparison': matrix_comparison,
        'experiment_type': 'baseline'
    }

    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================
def main(use_wandb: bool = True, save_excel: bool = False):
    """Main experiment runner for baseline training."""

    config = BaselineExperimentConfig()

    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments across complexity levels
    all_results = []

    for complexity_level in ['low', 'medium', 'high']:
        print(f"\n{'#'*60}")
        print(f"# Running BASELINE experiments for {complexity_level.upper()} complexity level")
        print(f"{'#'*60}")

        for seed in config.seeds[:1]:  # Start with just primary seed
            results = run_baseline_experiment(config, complexity_level, seed, use_wandb, save_excel)
            all_results.append(results)

        # Print summary for this complexity level
        level_results = [r for r in all_results if r['complexity_level'] == complexity_level]
        mse_baseline = [r['baseline_results']['MSE'] for r in level_results]
        r2_baseline = [r['baseline_results']['R2'] for r in level_results]
        relative_errors = [r['matrix_comparison']['relative_error'] for r in level_results]

        print(f"\n{complexity_level.upper()} BASELINE Summary (n={len(level_results)}):")
        print(f"  Baseline MSE: {np.mean(mse_baseline):.6f} ± {np.std(mse_baseline):.6f}")
        print(f"  Baseline R²: {np.mean(r2_baseline):.4f} ± {np.std(r2_baseline):.4f}")
        print(f"  Matrix Relative Error: {np.mean(relative_errors):.4f} ± {np.std(relative_errors):.4f} ({np.mean(relative_errors)*100:.1f}%)")

    # Save results to JSON
    results_file = os.path.join(config.results_dir, f"baseline_scratch_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print final summary table
    print("\n" + "="*75)
    print("BASELINE EXPERIMENT SUMMARY")
    print("="*75)
    print(f"{'Complexity':<12} {'Baseline MSE':<15} {'Baseline R²':<15} {'Matrix Error':<15}")
    print("-" * 75)

    for complexity_level in ['low', 'medium', 'high']:
        level_results = [r for r in all_results if r['complexity_level'] == complexity_level]
        if level_results:
            baseline_mse = np.mean([r['baseline_results']['MSE'] for r in level_results])
            baseline_r2 = np.mean([r['baseline_results']['R2'] for r in level_results])
            rel_error = np.mean([r['matrix_comparison']['relative_error'] for r in level_results])

            print(f"{complexity_level.upper():<12} {baseline_mse:<15.6f} {baseline_r2:<15.4f} {rel_error:<15.4f}")

    print("="*75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Baseline Scratch Experiment')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--save-excel', action='store_true', help='Save datasets to Excel files')
    args = parser.parse_args()

    main(use_wandb=not args.no_wandb, save_excel=args.save_excel)

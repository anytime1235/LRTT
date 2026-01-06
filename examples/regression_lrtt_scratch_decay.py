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
from aihwkit.simulator.configs.devices import FloatingPointDevice, ConstantStepDevice, LinearStepDevice
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
    D_prime_train_size = 80
    D_prime_test_size = 20

    # Noise parameters
    noise_std = 0.02

    # Input data type
    # Options: 'continuous', 'ternary' (0, 0.5, 1), 'binary' (0, 1)
    input_type = 'binary'  # Change this to 'ternary' or 'binary' for discrete inputs

    # Complexity levels to test
    # Options: 'simple' (0.5), 'medium' (0.8), 'complex' (1.0)
    complexity_levels = ['medium']

    # Training hyperparameters - LRTT scratch training
    lrtt_lr = 0.1  # Very conservative LR
    lrtt_epochs = 100
    lrtt_batch_size = 1
    lrtt_patience = 10  # Allow a bit more training than fine-tuning
    lrtt_grad_clip = 2.0  # Conservative clipping

    # LRTT configuration
    lrtt_rank = 1  # Rank-1 for minimal overfitting
    lrtt_transfer_every = 10  # Medium frequency to observe transfer effects
    lora_alpha = 2.0  # Conservative scaling

    # Reinit configuration - DECAY MODE
    reinit_mode = "decay"  # Use decay mode instead of standard
    decay_factor = 1.0  # Decay A,B weights to 50%

    # A matrix initialization mode
    # Options: 'zero' (LoRA-style, ΔW=0 initially), 'kaiming' (random Kaiming initialization)
    a_init_mode = 'zero'  # Change to 'zero' for original LoRA initialization

    # Device configuration
    # Use 6T1C device for A/B matrices (capacitor-based with retention decay)
    # False: IdealizedPresetDevice (idealized, noise only)
    USE_6T1C_AB = True
    dt_batch_sec = 3224  # Assumed time per mini-batch in seconds
    include_retention = True  # Include retention effects for 6T1C

    # Results directory
    results_dir = "results/lrtt_scratch_decay"

# ============================================================================
# Device Configuration
# ============================================================================

def create_6t1c_device(dt_batch_sec=1.0, include_retention=True):
    """Create 6T1C device for A/B tiles.

    6T1C Device Characteristics:
        - ~1000 conductance states per direction
        - Capacitor-based weight storage with exponential decay
        - Time constant τ ≈ 775 min (12.9 hours)

    Args:
        dt_batch_sec: Assumed time per mini-batch in seconds (for retention calculation)
        include_retention: Whether to include retention effects
    """
    import math

    # Calculate lifetime from physical τ for 6T1C
    TAU_SEC = 46505.0  # Physical time constant: 775.1 min = 46505 sec
    if include_retention and dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        lifetime = 1.0 / delta
    else:
        lifetime = 0.0  # No retention

    return LinearStepDevice(
        # Core update parameters (fitted from 6T1C data)
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,

        # Device-to-device variation
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,

        # Cycle-to-cycle variation
        dw_min_std=0.3,
        write_noise_std=0.0182,

        # LinearStepDevice specific
        mean_bound_reference=True,

        # Retention (capacitor leakage)
        lifetime=lifetime,
        lifetime_dtod=0.1 if include_retention else 0.0
    )


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

    # Generate inputs based on input_type configuration
    if config.input_type == 'ternary':
        # Generate inputs from {0, 0.5, 1}
        X = torch.randint(0, 3, (size, config.input_dim)).float() * 0.5
    elif config.input_type == 'binary':
        # Generate inputs from {0, 1}
        X = torch.randint(0, 2, (size, config.input_dim)).float()
    else:  # continuous (default)
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
        print(f"[Input Data] Type: {config.input_type}, Shape: {X.shape}")
        if config.input_type == 'ternary':
            print(f"[Input Data] Values: {0, 0.5, 1}, Sample: {X[0].tolist()[:4]}")
        elif config.input_type == 'binary':
            print(f"[Input Data] Values: {0, 1}, Sample: {X[0].tolist()[:4]}")
        else:
            print(f"[Input Data] Range: [-1, 1], Sample: {X[0].tolist()[:4]}")

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

        # Select devices for A/B tiles
        if config.USE_6T1C_AB:
            ab_device = create_6t1c_device(
                dt_batch_sec=config.dt_batch_sec,
                include_retention=config.include_retention
            )
            print(f"Using 6T1C device for A/B matrices (retention={'ON' if config.include_retention else 'OFF'})")
        else:
            ab_device = IdealizedPresetDevice()
            print("Using IdealizedPresetDevice for A/B matrices")

        device_config = PythonLRTTDevice(
            rank=config.lrtt_rank,
            transfer_every=config.lrtt_transfer_every,
            lora_alpha=config.lora_alpha,
            reinit_mode=config.reinit_mode,
            decay_factor=config.decay_factor,
            a_init_mode=config.a_init_mode,  # A initialization mode
            forward_inject=False,
            correct_gradient_magnitudes=True,
            unit_cell_devices=[
                ab_device,  # A matrix
                ab_device,  # B matrix
                FloatingPointDevice(),  # C matrix (no quantization, perfect precision)
            ]
        )

        print(f"A initialization mode: {config.a_init_mode}")

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

        mapping = MappingParameter(
            learn_out_scaling=False,  # Disable automatic output scaling
            weight_scaling_omega=0.0   # Disable weight scaling
        )

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

        # Initialize C with pretrained weights if provided, otherwise all -1
        if pretrained_C is not None:
            self.set_C_weights(pretrained_C)
        else:
            # Set C matrix to -1 (within analog tile bounds [-1,1])
            C_init = torch.ones(config.output_dim, config.input_dim) * -1.0
            self.set_C_weights(C_init)

        # Explicitly reinit A and B after setting C to ensure proper initialization
        analog_tile = self.lrtt_layer.analog_module
        if hasattr(analog_tile, 'controller'):
            print("Calling explicit reinit for A and B after C initialization...")
            analog_tile.controller.reinit()

    def set_C_weights(self, C: torch.Tensor):
        """Set the C matrix directly via tile_c to avoid quantization."""
        analog_tile = self.lrtt_layer.analog_module
        print(f"Setting C weights to shape {C.shape}")
        print(f"  Desired C values: min={C.min():.6f}, max={C.max():.6f}, mean={C.mean():.6f}")
        print(f"  All -1? {torch.allclose(C, torch.ones_like(C) * -1.0)}")

        # Access tile_c directly to bypass any LRTT-level processing
        if hasattr(analog_tile, 'tile_c'):
            print(f"  tile_c device type: {type(analog_tile.tile_c.rpu_config.device).__name__}")
            print("  Using direct tile_c.set_weights()")

            # Try to access the underlying weights tensor directly if possible
            if hasattr(analog_tile.tile_c, 'weight'):
                print("  tile_c has 'weight' attribute - trying to set directly")
                with torch.no_grad():
                    analog_tile.tile_c.weight.copy_(C.to(DEVICE))
            else:
                analog_tile.tile_c.set_weights(C.to(DEVICE), None)
        else:
            print("  Falling back to analog_tile.set_weights()")
            analog_tile.set_weights(C.to(DEVICE))

        # Verify what was actually set
        if hasattr(analog_tile, 'tile_c'):
            C_actual = analog_tile.tile_c.get_weights()[0]
        else:
            C_actual = analog_tile.get_weights()[0]

        print(f"After set_weights - C actual values:")
        print(f"  min={C_actual.min():.6f}, max={C_actual.max():.6f}, mean={C_actual.mean():.6f}")
        print(f"  All -1? {torch.allclose(C_actual, torch.ones_like(C_actual) * -1.0, atol=1e-6)}")
        print(f"  C_actual full matrix:\n{C_actual}")
        print(f"  Difference from -1: {(C_actual + 1.0).abs().max():.6f}")

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
                      seed: int = 42, use_wandb: bool = True) -> tuple:
    """Train LRTT from scratch on D'.

    Returns:
        Tuple of (model, training_history)
    """

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
        print(f"C shape: {C_init.shape}")
        print(f"C initial full matrix:\n{C_init}")
        print(f"C all -1? {torch.allclose(C_init, torch.ones_like(C_init) * -1.0)}")
        print(f"C max deviation from -1: {(C_init + 1.0).abs().max():.6f}")

        # Save a copy for comparison after first epoch
        C_init_copy = C_init.clone()

    # Use AnalogSGD for LRTT tiles (no momentum, simple vanilla SGD)
    optimizer = AnalogSGD(model.parameters(), lr=config.lrtt_lr, momentum=0.0)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training history for cell-wise tracking
    training_history = []

    # Record initial state (epoch=-1, before any training)
    C_init_log, A_init_log, B_init_log = model.get_lrtt_components()
    if A_init_log is not None and B_init_log is not None:
        initial_history = {
            'epoch': -1,  # -1 indicates initial state
            'train_loss': float('nan'),
            'val_loss': float('nan'),
            'is_transfer': False,
            'A_matrix': A_init_log.cpu().detach().numpy().copy(),
            'B_matrix': B_init_log.cpu().detach().numpy().copy(),
            'C_matrix': C_init_log.cpu().detach().numpy().copy(),
            'A_norm': A_init_log.norm().item(),
            'B_norm': B_init_log.norm().item(),
            'C_norm': C_init_log.norm().item(),
            'delta_W_norm': (A_init_log @ B_init_log).norm().item()
        }
        training_history.append(initial_history)

    # Log initial state to wandb (before any training)
    if use_wandb and A_init_log is not None and B_init_log is not None:
        log_dict = {
            'scratch/epoch': -1,  # Use -1 to indicate initial state
            'scratch/train_loss': float('nan'),
            'scratch/val_loss': float('nan'),
            'scratch/A_norm': A_init_log.norm().item(),
            'scratch/B_norm': B_init_log.norm().item(),
            'scratch/delta_norm': (A_init_log @ B_init_log).norm().item()
        }

        # Log individual A cells (A is [4,1])
        for i in range(4):
            log_dict[f'scratch/A[{i},0]'] = A_init_log[i, 0].item()

        # Log individual B cells (B is [1,4])
        for j in range(4):
            log_dict[f'scratch/B[0,{j}]'] = B_init_log[0, j].item()

        # Log individual C cells (C is [4,4])
        for i in range(4):
            for j in range(4):
                log_dict[f'scratch/C[{i},{j}]'] = C_init_log[i, j].item()

        wandb.log(log_dict)
        print(f"[WANDB] Logged initial state (epoch=-1): C range=[{C_init_log.min():.4f}, {C_init_log.max():.4f}]")

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

        # Record training history (every epoch for detailed tracking)
        C, A, B = model.get_lrtt_components()
        if A is not None and B is not None:
            # Check if transfer occurred this epoch
            analog_tile = model.lrtt_layer.analog_module
            is_transfer = False
            if hasattr(analog_tile, 'controller'):
                current_counter = analog_tile.controller.transfer_counter
                # Transfer just occurred if counter reset to 0
                if epoch > 0 and current_counter == 0:
                    is_transfer = True

            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'is_transfer': is_transfer,
                'A_matrix': A.cpu().detach().numpy().copy(),
                'B_matrix': B.cpu().detach().numpy().copy(),
                'C_matrix': C.cpu().detach().numpy().copy(),
                'A_norm': A.norm().item(),
                'B_norm': B.norm().item(),
                'C_norm': C.norm().item(),
                'delta_W_norm': (A @ B).norm().item()
            }
            training_history.append(history_entry)

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

                # Log individual A cells (A is [4,1])
                for i in range(4):
                    log_dict[f'scratch/A[{i},0]'] = A[i, 0].item()

                # Log individual B cells (B is [1,4])
                for j in range(4):
                    log_dict[f'scratch/B[0,{j}]'] = B[0, j].item()

                # Log individual C cells (C is [4,4])
                for i in range(4):
                    for j in range(4):
                        log_dict[f'scratch/C[{i},{j}]'] = C[i, j].item()

            wandb.log(log_dict)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            # Debug A,B updates
            C, A, B = model.get_lrtt_components()
            if A is not None:
                print(f"  -> A norm={A.norm():.4f}, B norm={B.norm():.4f}")

        # Special check after first epoch to see C range
        if epoch == 0:
            C_after_epoch0, _, _ = model.get_lrtt_components()
            C_diff = (C_after_epoch0 - C_init_copy).abs().max().item()
            print(f"\n[EPOCH 0 COMPLETE] C changed by max: {C_diff:.6f}")
            print(f"C range: [{C_after_epoch0.min():.4f}, {C_after_epoch0.max():.4f}]")
            if C_after_epoch0.max() > 1.0 or C_after_epoch0.min() < -1.0:
                print(f"WARNING: C exceeded [-1, 1] range even with clipping!")
            print()

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

    print(f"\nRecorded {len(training_history)} epochs of training history")

    return model, training_history


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


def save_experiment_details_to_excel(config: ScratchExperimentConfig,
                                     complexity_level: str,
                                     seed: int,
                                     target_matrix: torch.Tensor,
                                     train_dataset: TensorDataset,
                                     C_init: torch.Tensor,
                                     A_init: torch.Tensor,
                                     B_init: torch.Tensor,
                                     training_history: list,
                                     timestamp: str) -> None:
    """Save detailed experiment data to a single Excel file with multiple sheets."""
    import math

    # Create results directory if needed
    os.makedirs(config.results_dir, exist_ok=True)

    # Excel file path
    excel_path = os.path.join(
        config.results_dir,
        f"experiment_{complexity_level}_seed{seed}_{timestamp}.xlsx"
    )

    # Create Excel writer
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # 1. Hyperparameters
        lr_eff_ab = config.lrtt_lr * config.lora_alpha / math.sqrt(config.lrtt_rank)
        transfer_lr_c = config.lora_alpha

        hyperparams = {
            'Parameter': [
                'optimizer_lr', 'lora_alpha', 'rank', 'correct_gradient_magnitudes',
                'lr_eff_LoRA', 'transfer_lr_C', 'transfer_every',
                'input_type', 'complexity_level', 'seed',
                'use_6t1c_ab', 'dt_batch_sec', 'include_retention',
                'reinit_mode', 'decay_factor',
                'train_samples', 'input_dim', 'output_dim',
                'NOTE_matrix_notation', 'NOTE_code_forward'
            ],
            'Value': [
                config.lrtt_lr, config.lora_alpha, config.lrtt_rank, True,
                lr_eff_ab, transfer_lr_c, config.lrtt_transfer_every,
                config.input_type, complexity_level, seed,
                config.USE_6T1C_AB, config.dt_batch_sec if config.USE_6T1C_AB else None,
                config.include_retention if config.USE_6T1C_AB else None,
                config.reinit_mode, config.decay_factor,
                config.D_prime_train_size, config.input_dim, config.output_dim,
                'See Notation_Guide sheet',
                'y = C@x + α·(code_A)@(code_B)@x'
            ],
            'Description': [
                'Base optimizer learning rate',
                'LoRA alpha scaling factor',
                'LoRA rank',
                'Gradient magnitude correction enabled',
                'Effective LR for LoRA (lr × alpha / sqrt(rank))',
                'Transfer LR for C (= alpha)',
                'Transfer frequency (epochs)',
                'Input data type (continuous/ternary/binary)',
                'Target complexity level',
                'Random seed',
                'Use 6T1C device for A/B (LoRA matrices)',
                'Time per batch (seconds) for retention',
                'Include retention effects',
                'Reinit mode after transfer',
                'Decay factor for reinit',
                'Number of training samples',
                'Input dimension',
                'Output dimension',
                'Excel uses standard notation: B_up, A_down',
                'Code uses A@B, Excel shows as B@A (standard)'
            ]
        }
        pd.DataFrame(hyperparams).to_excel(writer, sheet_name='Hyperparameters', index=False)

        # 2. Target Matrix (T')
        target_df = pd.DataFrame(
            target_matrix.cpu().numpy(),
            columns=[f'x{i}' for i in range(target_matrix.shape[1])],
            index=[f'y{i}' for i in range(target_matrix.shape[0])]
        )
        target_df.to_excel(writer, sheet_name='Target_Matrix_T')

        # 3. Input Data (X) - training set
        X_train = train_dataset.tensors[0].cpu().numpy()
        input_df = pd.DataFrame(
            X_train,
            columns=[f'x{i}' for i in range(X_train.shape[1])]
        )
        input_df.to_excel(writer, sheet_name='Input_Data_X', index=False)

        # 4. Target Labels (Y) - training set
        Y_train = train_dataset.tensors[1].cpu().numpy()
        labels_df = pd.DataFrame(
            Y_train,
            columns=[f'y{i}' for i in range(Y_train.shape[1])]
        )
        labels_df.to_excel(writer, sheet_name='Target_Labels_Y', index=False)

        # 5. Initial B matrix (code A, up-projection)
        # Note: Code uses A @ B, but standard LoRA notation is B @ A
        # Code's A = up-projection [output, rank] → Excel's B
        B_up_df = pd.DataFrame(
            A_init.cpu().numpy(),
            columns=[f'rank{i}' for i in range(A_init.shape[1])],
            index=[f'y{i}' for i in range(A_init.shape[0])]
        )
        B_up_df.to_excel(writer, sheet_name='Initial_B_up')

        # 6. Initial A matrix (code B, down-projection)
        # Code's B = down-projection [rank, input] → Excel's A
        A_down_df = pd.DataFrame(
            B_init.cpu().numpy(),
            columns=[f'x{i}' for i in range(B_init.shape[1])],
            index=[f'rank{i}' for i in range(B_init.shape[0])]
        )
        A_down_df.to_excel(writer, sheet_name='Initial_A_down')

        # 7. Initial C matrix (Core array)
        C_df = pd.DataFrame(
            C_init.cpu().numpy(),
            columns=[f'x{i}' for i in range(C_init.shape[1])],
            index=[f'y{i}' for i in range(C_init.shape[0])]
        )
        C_df.to_excel(writer, sheet_name='Initial_C_Core')

        # 8. LoRA update ΔW (code: A⊗B, notation: B⊗A)
        # Standard notation: ΔW = B @ A (down @ up)
        # Code implementation: ΔW = A @ B
        delta_W = (A_init @ B_init).cpu().numpy()
        delta_df = pd.DataFrame(
            delta_W,
            columns=[f'x{i}' for i in range(delta_W.shape[1])],
            index=[f'y{i}' for i in range(delta_W.shape[0])]
        )
        delta_df.to_excel(writer, sheet_name='Initial_ΔW_LoRA')

        # 9. Notation guide
        notation_guide = pd.DataFrame({
            'Notation': ['Code', 'Code', 'Code', 'Excel', 'Excel', 'Standard LoRA'],
            'Matrix': ['A', 'B', 'A @ B', 'B_up', 'A_down', 'B @ A'],
            'Shape': ['[4, 1]', '[1, 4]', '[4, 4]', '[4, 1]', '[1, 4]', '[out, in]'],
            'Description': [
                'Up-projection (output side)',
                'Down-projection (input side)',
                'LoRA update',
                'Up-projection (same as code A)',
                'Down-projection (same as code B)',
                'Standard notation: down @ up'
            ]
        })
        notation_guide.to_excel(writer, sheet_name='Notation_Guide', index=False)

        # 10. Training Summary
        if training_history:
            summary_data = {
                'epoch': [h['epoch'] for h in training_history],
                'train_loss': [h['train_loss'] for h in training_history],
                'val_loss': [h['val_loss'] for h in training_history],
                'is_transfer': [h['is_transfer'] for h in training_history],
                'A_norm': [h['A_norm'] for h in training_history],
                'B_norm': [h['B_norm'] for h in training_history],
                'C_norm': [h['C_norm'] for h in training_history],
                'delta_W_norm': [h['delta_W_norm'] for h in training_history]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Training_Summary', index=False)

            # 11. Cell-wise values for all epochs (flattened format)
            # A matrix values (each row is one epoch, columns are flattened A values)
            A_history = []
            B_history = []
            C_history = []

            for h in training_history:
                A_flat = h['A_matrix'].flatten()
                B_flat = h['B_matrix'].flatten()
                C_flat = h['C_matrix'].flatten()

                A_history.append(A_flat)
                B_history.append(B_flat)
                C_history.append(C_flat)

            # A matrix history
            A_cols = [f'A[{i//A_init.shape[1]},{i%A_init.shape[1]}]' for i in range(A_flat.size)]
            A_history_df = pd.DataFrame(A_history, columns=A_cols)
            A_history_df.insert(0, 'epoch', [h['epoch'] for h in training_history])
            A_history_df.insert(1, 'is_transfer', [h['is_transfer'] for h in training_history])
            A_history_df.to_excel(writer, sheet_name='A_down_History', index=False)

            # B matrix history
            B_cols = [f'B[{i//B_init.shape[1]},{i%B_init.shape[1]}]' for i in range(B_flat.size)]
            B_history_df = pd.DataFrame(B_history, columns=B_cols)
            B_history_df.insert(0, 'epoch', [h['epoch'] for h in training_history])
            B_history_df.insert(1, 'is_transfer', [h['is_transfer'] for h in training_history])
            B_history_df.to_excel(writer, sheet_name='B_up_History', index=False)

            # C matrix history
            C_cols = [f'C[{i//C_init.shape[1]},{i%C_init.shape[1]}]' for i in range(C_flat.size)]
            C_history_df = pd.DataFrame(C_history, columns=C_cols)
            C_history_df.insert(0, 'epoch', [h['epoch'] for h in training_history])
            C_history_df.insert(1, 'is_transfer', [h['is_transfer'] for h in training_history])
            C_history_df.to_excel(writer, sheet_name='C_Core_History', index=False)

    print(f"\n✓ Detailed experiment data saved to Excel file:")
    print(f"  {excel_path}")
    print(f"  Sheets:")
    print(f"    - Hyperparameters, Target_Matrix_T, Input_Data_X, Target_Labels_Y")
    print(f"    - Initial_B_up, Initial_A_down, Initial_C_Core, Initial_ΔW_LoRA")
    print(f"    - Notation_Guide")
    print(f"    - Training_Summary (epoch-wise norms and losses)")
    print(f"    - A_down_History, B_up_History, C_Core_History (cell-wise values)")
    print(f"\n  Note: Excel uses standard LoRA notation (B@A = down@up)")
    print(f"        Code A [4,1] → Excel B_up (up-projection)")
    print(f"        Code B [1,4] → Excel A_down (down-projection)")
    print(f"        Recorded {len(training_history)} epochs")


def run_scratch_experiment(config: ScratchExperimentConfig, complexity_level: str = 'medium',
                          seed: int = 42, use_wandb: bool = True) -> Dict[str, any]:
    """Run LRTT experiment from scratch on target dataset."""

    print(f"\n{'='*60}")
    print(f"Running SCRATCH experiment with seed={seed}, complexity_level={complexity_level}")
    print(f"REINIT CONFIG: mode={config.reinit_mode}, decay_factor={config.decay_factor}")
    print(f"DEVICE CONFIG: 6T1C_AB={config.USE_6T1C_AB}, retention={config.include_retention if config.USE_6T1C_AB else 'N/A'}")
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
                'input_type': config.input_type,
                'lrtt_lr': config.lrtt_lr,
                'lrtt_rank': config.lrtt_rank,
                'transfer_every': config.lrtt_transfer_every,
                'lora_alpha': config.lora_alpha,
                'use_6t1c_ab': config.USE_6T1C_AB,
                'include_retention': config.include_retention if config.USE_6T1C_AB else None,
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

    # Get initial A, B, C before training (for CSV export)
    temp_model = LRTTModel(config, pretrained_C=None).to(DEVICE)
    C_init, A_init, B_init = temp_model.get_lrtt_components()
    del temp_model

    # Train LRTT from scratch on target dataset
    lrtt_model, training_history = train_lrtt_scratch(config, target_train_loader,
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

    # Save detailed experiment data to Excel file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_experiment_details_to_excel(
        config=config,
        complexity_level=complexity_level,
        seed=seed,
        target_matrix=target_matrix,
        train_dataset=target_train,
        C_init=C_init,
        A_init=A_init,
        B_init=B_init,
        training_history=training_history,
        timestamp=timestamp
    )

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

    # Run experiments across complexity levels
    all_results = []

    for complexity_level in config.complexity_levels:
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

    for complexity_level in config.complexity_levels:
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
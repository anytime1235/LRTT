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
    input_dim = 3
    output_dim = 3

    # Dataset sizes
    D_prime_train_size = 25
    D_prime_test_size = 25

    # Noise parameters
    noise_std = 0.02

    # Input data type
    # Options: 'continuous', 'ternary' (0, 0.5, 1), 'binary' (0, 1)
    input_type = 'binary'  # Change this to 'ternary' or 'binary' for discrete inputs

    # Complexity levels to test
    # Options: 'simple' (0.5), 'medium' (0.8), 'complex' (1.0)
    complexity_levels = ['medium']

    # Training hyperparameters - LRTT scratch training
    # Note: lrtt_lr is computed from hardware parameters below (see lrtt_lr property)
    lrtt_epochs = 1000
    lrtt_batch_size = 1
    lrtt_patience = 10  # Allow a bit more training than fine-tuning
    lrtt_grad_clip = 2.0  # Conservative clipping

    # LRTT configuration
    lrtt_rank = 1  # Rank-1 for minimal overfitting
    lrtt_transfer_every = 4  # Medium frequency to observe transfer effects
    lora_alpha = 0.0306  # Conservative scaling

    # Reinit configuration - DECAY MODE
    reinit_mode = "orthogonal_decay"  # Use decay mode instead of standard
    decay_factor = 1.0  # Decay A,B weights to 50%

    # A matrix initialization mode
    # Options: 'zero' (LoRA-style, ΔW=0 initially), 'kaiming' (random Kaiming initialization)
    a_init_mode = 'zero'  # Change to 'zero' for original LoRA initialization

    # Device configuration
    # Use 6T1C device for A/B matrices (capacitor-based with retention decay)
    # False: IdealizedPresetDevice (idealized, noise only)
    USE_6T1C_AB = True

    # Retention configuration
    # retention_ratio_at_transfer: fraction of A/B weight remaining at transfer time
    # Example: 0.9 means 90% of A/B weights remain after transfer_every steps
    #          0.5 means 50% of A/B weights remain (half decayed)
    #          1.0 means no decay (perfect retention)
    retention_ratio_at_transfer = 0.95  # 95% retention at transfer
    include_retention = True  # Enable/disable retention effects

    # Pulse/Update configuration (Hardware-realistic settings)
    desired_bl = 10  # Bit length for A/B updates (pulse train length)
    c_desired_bl = 31  # Bit length for C transfer (higher for accuracy)
    pulse_type = PulseType.STOCHASTIC_COMPRESSED  # Pulse generation type

    # Hardware mode: use fixed manual scaling factors
    # When use_manual_scaling=True, x_scaling and d_scaling are applied directly
    # as B (input) and A (gradient) factors, bypassing dynamic calculation
    use_manual_scaling = True  # Enable hardware-realistic fixed scaling mode

    # Manual scaling factors (hardware-fixed) - Global defaults
    # x_scaling: applied to input x (B factor in aihwkit)
    # d_scaling: applied to gradient d (A factor in aihwkit)
    x_scaling = None  # Input (x) scaling factor (global default)
    d_scaling = None  # Gradient (d) scaling factor (global default)

    # Separate A/B tile scaling factors (override global if set)
    # A tile update: x=XB (B projection of input), d=original gradient
    # B tile update: x=original input, d=DA (A^T projection of gradient)
    a_x_scaling = 0.2651  # A tile x scaling (None = use global x_scaling)
    a_d_scaling = 0.5359  # A tile d scaling (None = use global d_scaling)
    b_x_scaling = 1.0  # B tile x scaling (None = use global x_scaling)
    b_d_scaling = 0.7103  # B tile d scaling (None = use global d_scaling)

    # Debug logging for A/B scaling
    log_ab_scaling = True  # Enable x,d max value logging
    log_ab_scaling_every = 10  # Log every N steps

    # Output options
    save_figures = False  # Save training figures as PNG (disable to save time/space)

    # Note: dw_min is defined in SoftBoundsReferenceDevice (device characteristic)
    # Effective learning rate in hardware mode:
    # Δw = x_scaling * d_scaling * x * d * BL * dw_min (approximately)
    lrtt_lr = 0.1  # Placeholder for optimizer (actual update is hardware-controlled)

    # Results directory
    results_dir = "results/lrtt_scratch_decay"

# ============================================================================
# Device Configuration
# ============================================================================

def create_6t1c_device(retention_ratio_at_transfer=1.0, transfer_every=10, include_retention=True):
    """Create 6T1C device for A/B tiles.

    6T1C Device Characteristics:
        - ~1000 conductance states per direction
        - Capacitor-based weight storage with exponential decay

    Args:
        retention_ratio_at_transfer: Fraction of weight remaining after transfer_every steps
                                     (e.g., 0.9 = 90% retention, 0.5 = 50% retention)
        transfer_every: Number of steps between transfers
        include_retention: Whether to include retention effects
    """
    import math

    # Calculate lifetime from retention ratio
    # Weight after N steps: w(N) = w(0) * (1 - delta)^N
    # At transfer: retention_ratio = (1 - delta)^transfer_every
    # Solve for delta: delta = 1 - retention_ratio^(1/transfer_every)
    # lifetime = 1 / delta
    if include_retention and retention_ratio_at_transfer < 1.0:
        delta = 1.0 - math.pow(retention_ratio_at_transfer, 1.0 / transfer_every)
        lifetime = 1.0 / delta
        print(f"  6T1C retention: {retention_ratio_at_transfer*100:.1f}% after {transfer_every} steps → lifetime={lifetime:.1f}")
    else:
        lifetime = 0.0  # No retention
        print(f"  6T1C retention: DISABLED (perfect retention)")

    return LinearStepDevice(
        # Core update parameters (fitted from 6T1C data)
        dw_min=0.02,  #0.001981
        up_down=0.0,
        w_max=0.7,
        w_min=-0.7,
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
                retention_ratio_at_transfer=config.retention_ratio_at_transfer,
                transfer_every=config.lrtt_transfer_every,
                include_retention=config.include_retention
            )
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
            correct_gradient_magnitudes=False,
            unit_cell_devices=[
                ab_device,  # A matrix
                ab_device,  # B matrix
                FloatingPointDevice(),  # C matrix (no quantization, perfect precision)
            ],
            # Separate A/B tile scaling factors
            a_x_scaling=config.a_x_scaling,
            a_d_scaling=config.a_d_scaling,
            b_x_scaling=config.b_x_scaling,
            b_d_scaling=config.b_d_scaling,
            # Debug logging
            log_ab_scaling=config.log_ab_scaling,
            log_ab_scaling_every=config.log_ab_scaling_every,
            # Separate BL for C tile transfer
            c_desired_bl=config.c_desired_bl,
            # Exact transfer method (no pulsed update noise for C)
            transfer_method="set",
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

        # Update parameters for pulse generation
        # When use_manual_scaling=True, A and B are set directly from manual_d_scaling and manual_x_scaling
        # bypassing the dynamic calculation based on lr, dw_min, and input magnitudes
        update = UpdateParameters(
            desired_bl=config.desired_bl,              # Bit length (pulse train length)
            pulse_type=config.pulse_type,              # Stochastic pulse generation
            use_manual_scaling=config.use_manual_scaling,  # Enable hardware-realistic fixed scaling
            manual_x_scaling=config.x_scaling,         # B factor: scaling for input x
            manual_d_scaling=config.d_scaling,         # A factor: scaling for gradient d
            update_bl_management=False,                # Disable dynamic BL adjustment
            update_management=False,                   # Disable dynamic A/B scaling
        )

        rpu_config = PythonLRTTRPUConfig(
            device=device_config,
            mapping=mapping,
            forward=forward_io,
            backward=backward_io,
            update=update
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
            analog_tile.controller.reinit()

    def set_C_weights(self, C: torch.Tensor):
        """Set the C matrix directly via tile_c to avoid quantization."""
        analog_tile = self.lrtt_layer.analog_module

        # Access tile_c directly to bypass any LRTT-level processing
        if hasattr(analog_tile, 'tile_c'):
            # Try to access the underlying weights tensor directly if possible
            if hasattr(analog_tile.tile_c, 'weight'):
                with torch.no_grad():
                    analog_tile.tile_c.weight.copy_(C.to(DEVICE))
            else:
                analog_tile.tile_c.set_weights(C.to(DEVICE), None)
        else:
            analog_tile.set_weights(C.to(DEVICE))

        # Verify what was actually set
        if hasattr(analog_tile, 'tile_c'):
            C_actual = analog_tile.tile_c.get_weights()[0]
        else:
            C_actual = analog_tile.get_weights()[0]

        print(f"C initialized: min={C_actual.min():.4f}, max={C_actual.max():.4f}, mean={C_actual.mean():.4f}")

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
        Tuple of (model, training_history, epoch_history, C_init, A_init, B_init)
    """

    print("\n" + "="*60)
    print("SCRATCH TRAINING: LRTT directly on D'")
    print("="*60)

    torch.manual_seed(seed)

    # Create LRTT model without pre-trained C (random initialization)
    model = LRTTModel(config, pretrained_C=None).to(DEVICE)

    # Check initial A,B,C
    C_init, A_init, B_init = model.get_lrtt_components()
    if A_init is not None and B_init is not None:
        print(f"Initial: ‖A‖={A_init.norm():.4f}, ‖B‖={B_init.norm():.4f}, ‖C‖={C_init.norm():.4f}")

    # Use AnalogSGD for LRTT tiles (no momentum, simple vanilla SGD)
    optimizer = AnalogSGD(model.parameters(), lr=config.lrtt_lr, momentum=0.0)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training history for cell-wise tracking
    training_history = []
    epoch_history = []  # Epoch-level history (val_loss, train_loss)

    # Record initial state (step=0, before any training)
    C_init_log, A_init_log, B_init_log = model.get_lrtt_components()
    if A_init_log is not None and B_init_log is not None:
        initial_history = {
            'step': 0,  # 0 indicates initial state
            'epoch': 0,
            'batch_idx': -1,  # -1 indicates before any batch
            'batch_loss': float('nan'),
            'is_transfer': False,
            'grad_norm': 0.0,
            'grad_C_matrix': np.zeros_like(C_init_log.cpu().detach().numpy()),
            'A_matrix': A_init_log.cpu().detach().numpy().copy(),
            'B_matrix': B_init_log.cpu().detach().numpy().copy(),
            'C_matrix': C_init_log.cpu().detach().numpy().copy(),
            'A_norm': A_init_log.norm().item(),
            'B_norm': B_init_log.norm().item(),
            'C_norm': C_init_log.norm().item(),
            'delta_W_norm': (A_init_log @ B_init_log).norm().item()
        }
        training_history.append(initial_history)

    # Log initial state to wandb (before any training) as step 0
    if use_wandb and A_init_log is not None and B_init_log is not None:
        log_dict = {
            'scratch/step': 0,
            'scratch/epoch': -1,
            'scratch/batch_idx': -1,
            'scratch/batch_loss': float('nan'),
            'scratch/is_transfer_step': False,
            'scratch/A_norm_step': A_init_log.norm().item(),
            'scratch/B_norm_step': B_init_log.norm().item(),
            'scratch/C_norm_step': C_init_log.norm().item(),
            'scratch/delta_norm_step': (A_init_log @ B_init_log).norm().item()
        }

        # Log individual A cells (A is [input_dim, rank])
        for i in range(A_init_log.shape[0]):
            for r in range(A_init_log.shape[1]):
                log_dict[f'scratch/A[{i},{r}]_step'] = A_init_log[i, r].item()

        # Log individual B cells (B is [rank, input_dim])
        for r in range(B_init_log.shape[0]):
            for j in range(B_init_log.shape[1]):
                log_dict[f'scratch/B[{r},{j}]_step'] = B_init_log[r, j].item()

        # Log individual C cells (C is [input_dim, input_dim])
        for i in range(C_init_log.shape[0]):
            for j in range(C_init_log.shape[1]):
                log_dict[f'scratch/C[{i},{j}]_step'] = C_init_log[i, j].item()

        wandb.log(log_dict)

    # Global step counter for batch-wise logging (starts at 1)
    global_step = 1

    for epoch in range(config.lrtt_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = F.mse_loss(Y_pred, Y_batch)

            # Debug: Check gradients on first batch of first epoch (BEFORE backward)
            if epoch == 0 and batch_idx == 0:
                print(f"\n  Debug Gradients (Epoch {epoch}, Batch {batch_idx}):")
                print(f"    Loss value: {loss.item():.6f}")
                print(f"    Learning rate: {config.lrtt_lr}")

                C, A, B = model.get_lrtt_components()
                print(f"    Before step - A norm: {A.norm():.6f}, B norm: {B.norm():.6f}")

                # Show hardware pulse parameters (fixed scaling mode)
                import math

                # Get actual data values for reference
                x_abs_max = X_batch.abs().max().item()
                d_batch = 2 * (Y_pred - Y_batch) / Y_batch.size(0)
                d_abs_max = d_batch.abs().max().item()

                # Hardware-fixed parameters
                dw_min = 0.02  # From LinearStepDevice definition
                BL = config.desired_bl

                # Get tile-specific scaling factors
                # A tile: uses a_x_scaling, a_d_scaling (or global if None)
                # B tile: uses b_x_scaling, b_d_scaling (or global if None)
                a_x_factor = config.a_x_scaling if config.a_x_scaling is not None else config.x_scaling
                a_d_factor = config.a_d_scaling if config.a_d_scaling is not None else config.d_scaling
                b_x_factor = config.b_x_scaling if config.b_x_scaling is not None else config.x_scaling
                b_d_factor = config.b_d_scaling if config.b_d_scaling is not None else config.d_scaling

                # Format global scaling (handle None)
                global_x = f"{config.x_scaling:.4f}" if config.x_scaling is not None else "None"
                global_d = f"{config.d_scaling:.4f}" if config.d_scaling is not None else "None"
                a_x_str = f"{a_x_factor:.4f}" if a_x_factor is not None else "None"
                a_d_str = f"{a_d_factor:.4f}" if a_d_factor is not None else "None"
                b_x_str = f"{b_x_factor:.4f}" if b_x_factor is not None else "None"
                b_d_str = f"{b_d_factor:.4f}" if b_d_factor is not None else "None"

                print(f"\n    === Hardware Pulse Parameters (use_manual_scaling=True) ===")
                print(f"    dw_min: {dw_min:.6f}")
                print(f"    BL: {BL}")
                print(f"    Global: x_scaling={global_x}, d_scaling={global_d}")
                print(f"    --- Tile-Specific Scaling ---")
                print(f"    A tile (code): x_scaling={a_x_str}, d_scaling={a_d_str}")
                print(f"    B tile (code): x_scaling={b_x_str}, d_scaling={b_d_str}")
                print(f"    --- Current Batch Data ---")
                print(f"    x_abs_max: {x_abs_max:.6f}")
                print(f"    d_abs_max: {d_abs_max:.6f}")
                print(f"    ======================================\n")

            # Compute gradient norm before backward (d = ∂L/∂y)
            # MSE gradient: d = (2/N) * (y_pred - y_target)
            N = Y_pred.numel()
            grad_d = (2.0 / N) * (Y_pred - Y_batch)
            grad_norm = grad_d.norm().item()

            # Compute gradient matrix for C: ∂L/∂C = grad_d.T @ X_batch / batch_size
            # Shape: [output_dim, input_dim] - same as C
            grad_C_matrix = (grad_d.T @ X_batch) / X_batch.size(0)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.lrtt_grad_clip)
            optimizer.step()

            # Debug: Check if A changed after optimizer step
            if epoch == 0 and batch_idx == 0:
                C, A, B = model.get_lrtt_components()
                print(f"    After step - A norm: {A.norm():.6f}, B norm: {B.norm():.6f}")

            train_loss += loss.item() * X_batch.size(0)

            # Log A, B, C cell values after each batch update
            C, A, B = model.get_lrtt_components()
            if A is not None and B is not None:
                # Check if transfer occurred
                analog_tile = model.lrtt_layer.analog_module
                is_transfer_step = False
                if hasattr(analog_tile, 'controller'):
                    current_counter = analog_tile.controller.transfer_counter
                    # Transfer just occurred if counter is exactly 1 (just reset)
                    if current_counter == 1 and global_step > 1:
                        is_transfer_step = True

                # Record step-wise history for Excel export
                step_history = {
                    'step': global_step,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'batch_loss': loss.item(),
                    'is_transfer': is_transfer_step,
                    'grad_norm': grad_norm,
                    'grad_C_matrix': grad_C_matrix.cpu().detach().numpy().copy(),
                    'A_matrix': A.cpu().detach().numpy().copy(),
                    'B_matrix': B.cpu().detach().numpy().copy(),
                    'C_matrix': C.cpu().detach().numpy().copy(),
                    'A_norm': A.norm().item(),
                    'B_norm': B.norm().item(),
                    'C_norm': C.norm().item(),
                    'delta_W_norm': (A @ B).norm().item()
                }
                training_history.append(step_history)

                if use_wandb:
                    log_dict = {
                        'scratch/step': global_step,
                        'scratch/epoch': epoch,
                        'scratch/batch_idx': batch_idx,
                        'scratch/batch_loss': loss.item(),
                        'scratch/is_transfer_step': is_transfer_step,
                        'scratch/A_norm_step': A.norm().item(),
                        'scratch/B_norm_step': B.norm().item(),
                        'scratch/C_norm_step': C.norm().item(),
                        'scratch/delta_norm_step': (A @ B).norm().item()
                    }

                    # Log individual A cells (A is [input_dim, rank])
                    for i in range(A.shape[0]):
                        for r in range(A.shape[1]):
                            log_dict[f'scratch/A[{i},{r}]_step'] = A[i, r].item()

                    # Log individual B cells (B is [rank, input_dim])
                    for r in range(B.shape[0]):
                        for j in range(B.shape[1]):
                            log_dict[f'scratch/B[{r},{j}]_step'] = B[r, j].item()

                    # Log individual C cells (C is [input_dim, input_dim])
                    for i in range(C.shape[0]):
                        for j in range(C.shape[1]):
                            log_dict[f'scratch/C[{i},{j}]_step'] = C[i, j].item()

                    wandb.log(log_dict)
            else:
                if global_step <= 5:
                    print(f"  [WARNING] Step {global_step}: A or B is None, skipping log")

            global_step += 1

        train_loss /= len(train_loader.dataset)

        # Check for transfer event
        analog_tile = model.lrtt_layer.analog_module
        if hasattr(analog_tile, 'controller'):
            current_transfer_counter = analog_tile.controller.transfer_counter
            if epoch > 0 and current_transfer_counter % config.lrtt_transfer_every == 0:
                # Transfer just occurred
                C, A, B = model.get_lrtt_components()
                print(f"  [TRANSFER] Epoch {epoch}: A norm={A.norm():.4f}, B norm={B.norm():.4f}, C norm={C.norm():.4f}")

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

        # Record epoch-level history
        epoch_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Wandb logging (epoch summary - norms and losses only, no cell values)
        # Use the last step of the epoch for epoch-level metrics
        if use_wandb:
            epoch_step = global_step - 1  # Last step of this epoch
            log_dict = {
                'scratch/step': epoch_step,  # Same step as last batch
                'scratch/train_loss_epoch': train_loss,
                'scratch/val_loss_epoch': val_loss
            }

            # Log component norms if available
            C, A, B = model.get_lrtt_components()
            if A is not None and B is not None:
                log_dict.update({
                    'scratch/A_norm_epoch': A.norm().item(),
                    'scratch/B_norm_epoch': B.norm().item(),
                    'scratch/C_norm_epoch': C.norm().item(),
                    'scratch/delta_norm_epoch': (A @ B).norm().item()
                })

            wandb.log(log_dict, commit=False)  # commit=False to merge with last batch's log

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}")

        if patience_counter >= config.lrtt_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest val loss: {best_val_loss:.6f}")

    # Print final norms
    C, A, B = model.get_lrtt_components()
    if A is not None and B is not None:
        print(f"Final: ‖A‖={A.norm():.4f}, ‖B‖={B.norm():.4f}, ‖A⊗B‖={(A @ B).norm():.4f}")

    return model, training_history, epoch_history, C_init, A_init, B_init


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


def plot_all_training_figures(training_history: list,
                               epoch_history: list,
                               config: ScratchExperimentConfig,
                               complexity_level: str,
                               seed: int,
                               timestamp: str,
                               final_mse: float = None,
                               final_r2: float = None) -> dict:
    """Plot all training metrics and save as images (similar to wandb scratch tab).

    Generates multiple figures:
    1. training_cell: Single cell [0,0] tracking - grad_C, C, B@A on one graph
    2. all_norms: ||A||, ||B||, ||C||, ||B@A||
    3. loss: val_loss over epochs (with final MSE and R² annotation)
    4. A_cells: individual A matrix cell values (down-projection)
    5. B_cells: individual B matrix cell values (up-projection)
    6. C_cells: individual C matrix cell values

    Note: Uses standard LoRA notation where A=down-projection, B=up-projection.

    Args:
        training_history: List of step-wise training history dictionaries
        epoch_history: List of epoch-wise history (train_loss, val_loss)
        config: Experiment configuration
        complexity_level: Complexity level string
        seed: Random seed
        timestamp: Timestamp string for filename
        final_mse: Final MSE loss on test set
        final_r2: Final R² score on test set

    Returns:
        Dictionary of saved figure paths
    """
    if not training_history:
        print("No training history to plot")
        return {}

    os.makedirs(config.results_dir, exist_ok=True)
    saved_paths = {}

    # Extract common data
    steps = [h['step'] for h in training_history]
    is_transfer = [h['is_transfer'] for h in training_history]
    transfer_steps = [s for s, t in zip(steps, is_transfer) if t]

    def add_transfer_lines(ax):
        """Add vertical lines at transfer steps."""
        for ts in transfer_steps:
            ax.axvline(x=ts, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)

    # =========================================================================
    # Figure 1: Single Cell Tracking (grad_C, C, B@A for cell [0,0])
    # Plots a single cell to visualize training dynamics similar to memristor papers
    # =========================================================================
    cell_row, cell_col = 0, 0  # Cell to track (can be changed)

    # Extract single cell values from each step
    grad_C_cells = [h['grad_C_matrix'][cell_row, cell_col] if 'grad_C_matrix' in h else 0
                    for h in training_history]
    C_cells = [h['C_matrix'][cell_row, cell_col] for h in training_history]
    # A @ B in code notation = B @ A in standard LoRA notation
    AB_cells = [np.dot(h['A_matrix'], h['B_matrix'])[cell_row, cell_col]
                for h in training_history]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Gradient with filled area to y=0
    ax.fill_between(steps, grad_C_cells, 0, alpha=0.3, color='blue')
    ax.plot(steps, grad_C_cells, 'b-', linewidth=1, label=f'∂L/∂C[{cell_row},{cell_col}] (gradient)')
    ax.plot(steps, C_cells, 'g-', linewidth=1, label=f'C[{cell_row},{cell_col}] (core weight)')
    ax.plot(steps, AB_cells, 'r-', linewidth=1, label=f'(B@A)[{cell_row},{cell_col}] (LoRA update)')

    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title(f'Single Cell Tracking [{cell_row},{cell_col}] (complexity={complexity_level}, seed={seed})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    # Transfer lines removed - graph is already busy with 3 lines + gradient fill

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"training_cell_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['training_cell'] = path

    # =========================================================================
    # Figure 2: All Matrix Norms (A, B, C, delta_W)
    # Note: A=down-proj, B=up-proj (standard LoRA notation)
    # Code's A_norm -> B (up-proj), Code's B_norm -> A (down-proj)
    # =========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Code A = up-proj = standard B, Code B = down-proj = standard A
    code_A_norms = [h['A_norm'] for h in training_history]  # up-proj -> B
    code_B_norms = [h['B_norm'] for h in training_history]  # down-proj -> A
    C_norms = [h['C_norm'] for h in training_history]
    delta_W_norms = [h['delta_W_norm'] for h in training_history]

    axes[0].plot(steps, code_B_norms, 'cyan', linewidth=1, label='||A|| (down-proj)')
    axes[0].set_ylabel('||A|| Norm')
    axes[0].set_title(f'All Matrix Norms (complexity={complexity_level}, seed={seed})')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    add_transfer_lines(axes[0])

    axes[1].plot(steps, code_A_norms, 'purple', linewidth=1, label='||B|| (up-proj)')
    axes[1].set_ylabel('||B|| Norm')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    add_transfer_lines(axes[1])

    axes[2].plot(steps, C_norms, 'g-', linewidth=1, label='||C|| (Core)')
    axes[2].set_ylabel('||C|| Norm')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    add_transfer_lines(axes[2])

    axes[3].plot(steps, delta_W_norms, 'r-', linewidth=1, label='||B@A|| (LoRA update)')
    axes[3].set_ylabel('||B@A|| Norm')
    axes[3].set_xlabel('Step')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    add_transfer_lines(axes[3])

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"all_norms_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['all_norms'] = path

    # =========================================================================
    # Figure 3: Loss (val_loss per epoch with final MSE and R² annotation)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 4))

    if epoch_history:
        epochs = [h['epoch'] for h in epoch_history]
        val_losses = [h['val_loss'] for h in epoch_history]
        train_losses = [h['train_loss'] for h in epoch_history]

        ax.plot(epochs, val_losses, 'b-', linewidth=1, label='Val Loss (MSE)')
        ax.plot(epochs, train_losses, 'r--', linewidth=1, alpha=0.7, label='Train Loss (MSE)')
        ax.set_xlabel('Epoch')
    else:
        # Fallback to batch loss if no epoch history
        batch_losses = [h['batch_loss'] for h in training_history]
        ax.plot(steps, batch_losses, 'b-', linewidth=1, label='Batch Loss (MSE)')
        ax.set_xlabel('Step')

    ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss (complexity={complexity_level}, seed={seed})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add final MSE and R² as text annotation
    if final_mse is not None or final_r2 is not None:
        text_lines = []
        if final_mse is not None:
            text_lines.append(f'Final MSE: {final_mse:.6f}')
        if final_r2 is not None:
            text_lines.append(f'Final R²: {final_r2:.4f}')
        text_str = '\n'.join(text_lines)

        # Position text in upper left (avoiding the legend in upper right)
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(config.results_dir, f"loss_{complexity_level}_seed{seed}_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths['loss'] = path

    # =========================================================================
    # Figure 4: A matrix cell values (down-projection)
    # Note: Code's B_matrix = standard A (down-projection)
    # =========================================================================
    if training_history and 'B_matrix' in training_history[0]:
        B_shape = training_history[0]['B_matrix'].shape
        n_cells = B_shape[0] * B_shape[1]

        fig, axes = plt.subplots(n_cells, 1, figsize=(12, 3 * n_cells), sharex=True)
        if n_cells == 1:
            axes = [axes]

        cell_idx = 0
        for r in range(B_shape[0]):
            for j in range(B_shape[1]):
                values = [h['B_matrix'][r, j] for h in training_history]
                axes[cell_idx].plot(steps, values, 'cyan', linewidth=1, label=f'A[{r},{j}]')
                axes[cell_idx].set_ylabel(f'A[{r},{j}]')
                axes[cell_idx].legend(loc='upper right')
                axes[cell_idx].grid(True, alpha=0.3)
                add_transfer_lines(axes[cell_idx])
                cell_idx += 1

        axes[0].set_title(f'A Matrix Cells (down-proj) (complexity={complexity_level}, seed={seed})')
        axes[-1].set_xlabel('Step')

        plt.tight_layout()
        path = os.path.join(config.results_dir, f"A_cells_{complexity_level}_seed{seed}_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths['A_cells'] = path

    # =========================================================================
    # Figure 5: B matrix cell values (up-projection)
    # Note: Code's A_matrix = standard B (up-projection)
    # =========================================================================
    if training_history and 'A_matrix' in training_history[0]:
        A_shape = training_history[0]['A_matrix'].shape
        n_cells = A_shape[0] * A_shape[1]

        fig, axes = plt.subplots(n_cells, 1, figsize=(12, 3 * n_cells), sharex=True)
        if n_cells == 1:
            axes = [axes]

        cell_idx = 0
        for i in range(A_shape[0]):
            for r in range(A_shape[1]):
                values = [h['A_matrix'][i, r] for h in training_history]
                axes[cell_idx].plot(steps, values, 'purple', linewidth=1, label=f'B[{i},{r}]')
                axes[cell_idx].set_ylabel(f'B[{i},{r}]')
                axes[cell_idx].legend(loc='upper right')
                axes[cell_idx].grid(True, alpha=0.3)
                add_transfer_lines(axes[cell_idx])
                cell_idx += 1

        axes[0].set_title(f'B Matrix Cells (up-proj) (complexity={complexity_level}, seed={seed})')
        axes[-1].set_xlabel('Step')

        plt.tight_layout()
        path = os.path.join(config.results_dir, f"B_cells_{complexity_level}_seed{seed}_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths['B_cells'] = path

    # =========================================================================
    # Figure 6: C matrix cell values (Core weights)
    # =========================================================================
    if training_history and 'C_matrix' in training_history[0]:
        C_shape = training_history[0]['C_matrix'].shape
        n_cells = C_shape[0] * C_shape[1]

        # For large C matrices, create a grid layout instead
        n_rows = C_shape[0]
        n_cols = C_shape[1]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_rows):
            for j in range(n_cols):
                values = [h['C_matrix'][i, j] for h in training_history]
                axes[i, j].plot(steps, values, 'g-', linewidth=1)
                axes[i, j].set_title(f'C[{i},{j}]', fontsize=10)
                axes[i, j].grid(True, alpha=0.3)
                add_transfer_lines(axes[i, j])

        fig.suptitle(f'C Matrix Cells (Core) (complexity={complexity_level}, seed={seed})', fontsize=12)
        axes[-1, n_cols // 2].set_xlabel('Step')

        plt.tight_layout()
        path = os.path.join(config.results_dir, f"C_cells_{complexity_level}_seed{seed}_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths['C_cells'] = path

    # Print summary
    print(f"\nTraining figures saved to {config.results_dir}/:")
    for name, path in saved_paths.items():
        print(f"  - {os.path.basename(path)}")

    return saved_paths


def save_experiment_details_to_excel(config: ScratchExperimentConfig,
                                     complexity_level: str,
                                     seed: int,
                                     target_matrix: torch.Tensor,
                                     train_dataset: TensorDataset,
                                     C_init: torch.Tensor,
                                     A_init: torch.Tensor,
                                     B_init: torch.Tensor,
                                     training_history: list,
                                     timestamp: str,
                                     final_r2: float = None) -> None:
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

        # 1. Hyperparameters (simplified and reorganized)
        # Note: Scaling parameters use Excel notation (swapped from code)
        # Code A tile → Excel B tile (up-projection)
        # Code B tile → Excel A tile (down-projection)
        param_names = [
            'loss_function',
            'rank',
            'transfer_lr',
            'transfer_every',
            'input_type',
            'complexity',
            'seed',
            'use_6t1c',
            'retention',
            'include_retention',
            'desired_bl',
            'dw_min',
            'a_x_scaling',
            'a_d_scaling',
            'b_x_scaling',
            'b_d_scaling',
            'train_samples',
            'input_dim',
            'output_dim',
            'total_steps',
        ]

        # Calculate total steps from training history
        total_steps = training_history[-1]['step'] if training_history else 0

        param_values = [
            'MSE (Mean Squared Error)',
            config.lrtt_rank,
            config.lora_alpha,  # transfer_lr = lora_alpha
            config.lrtt_transfer_every,
            config.input_type,
            complexity_level,
            seed,
            config.USE_6T1C_AB,
            config.retention_ratio_at_transfer if config.USE_6T1C_AB else None,
            config.include_retention if config.USE_6T1C_AB else None,
            config.desired_bl,
            0.02,  # dw_min from LinearStepDevice
            config.b_x_scaling,  # Excel A = Code B (down-projection)
            config.b_d_scaling,  # Excel A = Code B
            config.a_x_scaling,  # Excel B = Code A (up-projection)
            config.a_d_scaling,  # Excel B = Code A
            config.D_prime_train_size,
            config.input_dim,
            config.output_dim,
            total_steps,
        ]
        param_descs = [
            '(1/N) * Σ(y_pred - y_target)²',
            'LoRA rank',
            'Transfer learning rate (= lora_alpha)',
            'Transfer frequency (steps)',
            'Input data type (continuous/ternary/binary)',
            'Target complexity level',
            'Random seed',
            'Use 6T1C device for A/B matrices',
            'Retention ratio at transfer (e.g., 0.95 = 95%)',
            'Include retention effects',
            'Bit length (pulse train length)',
            'Minimum weight update step',
            'A tile (down-proj) x scaling',
            'A tile (down-proj) d scaling',
            'B tile (up-proj) x scaling',
            'B tile (up-proj) d scaling',
            'Number of training samples',
            'Input dimension',
            'Output dimension',
            'Total training steps',
        ]

        hyperparams = {
            'Parameter': param_names,
            'Value': param_values,
            'Description': param_descs
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
            # Add final_r2 column (only last row has the value)
            n_steps = len(training_history)
            r2_column = [None] * n_steps
            if final_r2 is not None:
                r2_column[-1] = final_r2  # Put R² in the last row

            summary_data = {
                'step': [h['step'] for h in training_history],
                'epoch': [h['epoch'] for h in training_history],
                'batch_idx': [h['batch_idx'] for h in training_history],
                'batch_loss': [h['batch_loss'] for h in training_history],
                'is_transfer': [h['is_transfer'] for h in training_history],
                'grad_norm': [h.get('grad_norm', None) for h in training_history],
                'A_norm': [h['A_norm'] for h in training_history],
                'B_norm': [h['B_norm'] for h in training_history],
                'C_norm': [h['C_norm'] for h in training_history],
                'delta_W_norm': [h['delta_W_norm'] for h in training_history],
                'final_r2': r2_column
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Training_Summary', index=False)

            # 11. Cell-wise values for all steps (flattened format)
            # A matrix values (each row is one step, columns are flattened A values)
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

            # A matrix history (step-wise)
            A_cols = [f'A[{i//A_init.shape[1]},{i%A_init.shape[1]}]' for i in range(A_flat.size)]
            A_history_df = pd.DataFrame(A_history, columns=A_cols)
            A_history_df.insert(0, 'step', [h['step'] for h in training_history])
            A_history_df.insert(1, 'epoch', [h['epoch'] for h in training_history])
            A_history_df.insert(2, 'is_transfer', [h['is_transfer'] for h in training_history])
            A_history_df.to_excel(writer, sheet_name='A_down_History', index=False)

            # B matrix history (step-wise)
            B_cols = [f'B[{i//B_init.shape[1]},{i%B_init.shape[1]}]' for i in range(B_flat.size)]
            B_history_df = pd.DataFrame(B_history, columns=B_cols)
            B_history_df.insert(0, 'step', [h['step'] for h in training_history])
            B_history_df.insert(1, 'epoch', [h['epoch'] for h in training_history])
            B_history_df.insert(2, 'is_transfer', [h['is_transfer'] for h in training_history])
            B_history_df.to_excel(writer, sheet_name='B_up_History', index=False)

            # C matrix history (step-wise)
            C_cols = [f'C[{i//C_init.shape[1]},{i%C_init.shape[1]}]' for i in range(C_flat.size)]
            C_history_df = pd.DataFrame(C_history, columns=C_cols)
            C_history_df.insert(0, 'step', [h['step'] for h in training_history])
            C_history_df.insert(1, 'epoch', [h['epoch'] for h in training_history])
            C_history_df.insert(2, 'is_transfer', [h['is_transfer'] for h in training_history])
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
    if config.USE_6T1C_AB and config.include_retention:
        print(f"DEVICE CONFIG: 6T1C_AB=True, retention={config.retention_ratio_at_transfer*100:.1f}% at transfer")
    else:
        print(f"DEVICE CONFIG: 6T1C_AB={config.USE_6T1C_AB}, retention={'OFF' if not config.include_retention else 'N/A'}")
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
                'retention_ratio_at_transfer': config.retention_ratio_at_transfer if config.USE_6T1C_AB else None,
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

    # Train LRTT from scratch on target dataset (also returns initial A, B, C)
    lrtt_model, training_history, epoch_history, C_init, A_init, B_init = train_lrtt_scratch(
        config, target_train_loader, target_test_loader, seed, use_wandb)

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
        timestamp=timestamp,
        final_r2=lrtt_results['R2']
    )

    # Plot and save all training figures (like wandb scratch tab)
    if config.save_figures:
        plot_all_training_figures(
            training_history=training_history,
            epoch_history=epoch_history,
            config=config,
            complexity_level=complexity_level,
            seed=seed,
            timestamp=timestamp,
            final_mse=lrtt_results['MSE'],
            final_r2=lrtt_results['R2']
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
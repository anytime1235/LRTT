#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LRTT Figure 3 Toy Experiment: Update Mechanism Trace

This script replicates TTv2 paper Figure 3 (5x5 toy) style experiments
to observe the update/transfer dynamics of LRTTController.

Key features:
- A, B tiles: 6T1C preset (LinearStepDevice with retention disabled by default)
- C tile: IdealizedPreset or EcramPreset (configurable)
- forward_inject=False (fixed) -> uses ab_weight_update_reconstruction path
- Logs norms, read errors, transfer events, single element traces

Read Error Analysis:
- After reinit (step 0): A weights are ~0.0001, analog MVM can't detect them -> read_err=1.0
- Accumulating (step 1-24): A grows via gradient updates -> read_err decreases to ~5%
- After transfer (step 25): A is reinitialized -> read_err jumps back to 1.0
- This cycle repeats, showing physically meaningful analog readout limitations
- Symmetric error formula: ||est-true|| / max(||est||,||true||) handles near-zero gracefully

Usage examples:
    # Single run with default parameters
    python experiments/lrtt_fig3_toy_trace.py --run_single

    # Single run with specific parameters
    python experiments/lrtt_fig3_toy_trace.py --run_single \
        --rank 4 --transfer_every 25 --transfer_lr_base 0.1 \
        --use_onehot --use_sigma_delta --reinit_mode orthogonal \
        --gradient_pattern ones --steps 500

    # Sweep over multiple parameter combinations
    python experiments/lrtt_fig3_toy_trace.py --sweep --sweep_subset small

    # With noisy device config
    python experiments/lrtt_fig3_toy_trace.py --run_single --device noisy

Author: LRTT Team
Date: 2024
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# AIHWKit imports
from aihwkit.simulator.configs import (
    SingleRPUConfig,
    UnitCellRPUConfig,
    IOParameters,
    UpdateParameters,
    PulseType,
)
from aihwkit.simulator.configs.devices import (
    ConstantStepDevice,
    LinearStepDevice,
    IdealDevice,
)
from aihwkit.simulator.presets.devices import (
    IdealizedPresetDevice,
    PCMPresetDevice,
    ReRamESPresetDevice,
)
from aihwkit.simulator.tiles.analog import AnalogTile

# LRTT imports
from aihwkit.simulator.tiles.lrtt_controller import LRTTController
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice, PythonLRTTPreset


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for LRTT Figure 3 toy experiment."""

    # Matrix dimensions
    d_size: int = 3
    x_size: int = 3

    # LRTT parameters
    rank: int = 1
    transfer_every: int = 25
    transfer_lr_base: float = 0.1
    transfer_lr_scale: str = "sqrt_rank"  # "none" or "sqrt_rank"
    lora_alpha: float = 1.0
    reinit_mode: str = "orthogonal"  # "standard" or "orthogonal"
    reinit_gain: float = 0.1
    decay_factor: float = 0.9

    # Transfer mode
    use_onehot: bool = True
    use_sigma_delta: bool = False  # Disable ΣΔ, use simple pulsed update

    # Experiment parameters
    steps: int = 500
    lr_update: float = 0.01
    gradient_pattern: str = "ones"  # "ones", "eye", "rand"
    seed: int = 42

    # Device configuration
    # ab_io_noise: Controls IO noise for A/B tile forward/backward passes
    #   - "off": No IO noise (default, focus on device update dynamics)
    #   - "standard": Standard IO noise (0.02 out/inp noise)
    # Note: A/B tiles ALWAYS use 6T1C device regardless of this setting
    ab_io_noise: str = "off"  # "off", "standard"
    c_device_type: str = "ideal"  # "ideal", "pcm", "rram", "ecram"

    # One-hot read averaging
    read_n_avg: int = 1

    # Sigma-delta parameters
    transfer_burst_limit: int = 10

    # Read error logging (for paper figures: compare analog readout vs true)
    # When True, always compute A_hat/B_hat via one-hot read for logging,
    # even when use_onehot=False for transfer. This allows "measured vs true" comparison.
    log_read_errors: bool = True

    # Output
    output_dir: str = "results/lrtt_fig3_toy"
    tag: str = ""

    def __post_init__(self):
        """Generate tag if not provided."""
        if not self.tag:
            # Generate descriptive tag with all key parameters
            onehot_str = "onehot" if self.use_onehot else "direct"
            sd_str = "sd" if self.use_sigma_delta else "nosd"
            c_dev_str = self.c_device_type[:4]  # ideal, ecra, pcm, rram
            self.tag = (
                f"rank{self.rank}_"
                f"te{self.transfer_every}_"
                f"tlr{self.transfer_lr_base}_"
                f"lr{self.lr_update}_"
                f"{onehot_str}_{sd_str}_"
                f"{self.reinit_mode}_"
                f"C{c_dev_str}_"
                f"{self.gradient_pattern}"
            )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Device Creation Helpers
# =============================================================================

def get_c_device_from_type(device_type: str):
    """Get C tile device based on type string.

    Args:
        device_type: One of "ideal", "pcm", "rram", "ecram"

    Returns:
        PulsedDevice for C tile
    """
    if device_type == "ideal":
        return IdealizedPresetDevice(
            dw_min=0.001,
            dw_min_std=0.0,
            dw_min_dtod=0.0,
        )
    elif device_type == "pcm":
        return PCMPresetDevice()
    elif device_type == "rram":
        return ReRamESPresetDevice()
    elif device_type == "ecram":
        # ECRAM-like device (LinearStep with better characteristics)
        return LinearStepDevice(
            dw_min=0.001,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=0.0,
            gamma_down=0.0,
            mult_noise=False,
            dw_min_dtod=0.05,
            dw_min_std=0.1,
        )
    else:
        return IdealizedPresetDevice()


def create_tile_config(device, noisy: bool = False):
    """Create RPU config for a tile using the device from PythonLRTTPreset."""
    from aihwkit.simulator.configs import NoiseManagementType, BoundManagementType

    if noisy:
        forward = IOParameters(
            out_noise=0.02,
            inp_noise=0.02,
            out_bound=1.0,
            inp_bound=1.0,
            noise_management=NoiseManagementType.ABS_MAX,
            bound_management=BoundManagementType.NONE,
        )
        backward = IOParameters(
            out_noise=0.02,
            inp_noise=0.02,
            out_bound=1.0,
            inp_bound=1.0,
            noise_management=NoiseManagementType.ABS_MAX,
            bound_management=BoundManagementType.NONE,
        )
    else:
        forward = IOParameters(
            out_noise=0.0,
            inp_noise=0.0,
            out_bound=1.0,
            inp_bound=1.0,
            noise_management=NoiseManagementType.NONE,
            bound_management=BoundManagementType.NONE,
        )
        backward = IOParameters(
            out_noise=0.0,
            inp_noise=0.0,
            out_bound=1.0,
            inp_bound=1.0,
            noise_management=NoiseManagementType.NONE,
            bound_management=BoundManagementType.NONE,
        )

    return SingleRPUConfig(
        device=device,
        forward=forward,
        backward=backward,
        update=UpdateParameters(
            pulse_type=PulseType.STOCHASTIC_COMPRESSED,
        ),
    )


def create_tiles(config: ExperimentConfig, torch_device: torch.device):
    """Create A, B, C tiles for the experiment using PythonLRTTPreset.

    Uses PythonLRTTPreset.sixt1c_ab() to get properly configured devices
    for A/B tiles (6T1C) and C tile, ensuring consistency with LRTT examples.
    """
    # Get C device based on config
    c_device = get_c_device_from_type(config.c_device_type)

    # Create LRTT device config using PythonLRTTPreset (canonical source)
    # This ensures A/B tiles use the same 6T1C configuration as other LRTT examples
    lrtt_device = PythonLRTTPreset.sixt1c_ab(
        rank=config.rank,
        transfer_every=config.transfer_every,
        lora_alpha=config.lora_alpha,
        dt_batch_sec=0.0,           # No retention for toy experiment
        include_retention=False,
        c_device=c_device,
        reinit_mode=config.reinit_mode,
        decay_factor=config.decay_factor,
    )

    # Extract devices from PythonLRTTDevice.unit_cell_devices
    # [0] = A tile device (6T1C), [1] = B tile device (6T1C), [2] = C tile device
    a_device = lrtt_device.unit_cell_devices[0]
    b_device = lrtt_device.unit_cell_devices[1]
    c_device_final = lrtt_device.unit_cell_devices[2]

    noisy = config.ab_io_noise == "standard"

    # Create configs using the devices from PythonLRTTPreset
    a_config = create_tile_config(a_device, noisy=noisy)
    b_config = create_tile_config(b_device, noisy=noisy)
    c_config = create_tile_config(c_device_final, noisy=noisy)

    # Create tiles using AnalogTile (which has get_weights/set_weights methods)
    # A: [d_size, rank]
    tile_a = AnalogTile(
        out_size=config.d_size,
        in_size=config.rank,
        rpu_config=a_config,
        bias=False,
    )

    # B: [rank, x_size]
    tile_b = AnalogTile(
        out_size=config.rank,
        in_size=config.x_size,
        rpu_config=b_config,
        bias=False,
    )

    # C: [d_size, x_size]
    tile_c = AnalogTile(
        out_size=config.d_size,
        in_size=config.x_size,
        rpu_config=c_config,
        bias=False,
    )

    # Move to device
    if torch_device.type == 'cuda':
        tile_a = tile_a.cuda(torch_device)
        tile_b = tile_b.cuda(torch_device)
        tile_c = tile_c.cuda(torch_device)

    # Set learning rates
    tile_a.set_learning_rate(config.lr_update)
    tile_b.set_learning_rate(config.lr_update)
    tile_c.set_learning_rate(config.lr_update)

    return tile_a, tile_b, tile_c


# =============================================================================
# Weight Reading Helpers
# =============================================================================

def read_A_true(tile_a, d_size: int, rank: int, device: torch.device) -> torch.Tensor:
    """Read A matrix [d_size, rank] from tile_a using get_weights()."""
    weights, _ = tile_a.get_weights()
    return weights.to(device)  # [d_size, rank]


def read_B_true(tile_b, rank: int, x_size: int, device: torch.device) -> torch.Tensor:
    """Read B matrix [rank, x_size] from tile_b using get_weights()."""
    weights, _ = tile_b.get_weights()
    return weights.to(device)  # [rank, x_size]


def read_C_true(tile_c, d_size: int, x_size: int, device: torch.device) -> torch.Tensor:
    """Read C matrix [d_size, x_size] from tile_c using get_weights()."""
    weights, _ = tile_c.get_weights()
    return weights.to(device)  # [d_size, x_size]


# =============================================================================
# Gradient Pattern Generation
# =============================================================================

def create_gradient_inputs(
    pattern: str,
    d_size: int,
    x_size: int,
    device: torch.device,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create (X, D) inputs for gradient computation.

    G = D^T @ X is the target gradient for reconstruction.

    Returns:
        X: [batch, x_size]
        D: [batch, d_size]
        G: [d_size, x_size] = D^T @ X
    """
    torch.manual_seed(seed)

    if pattern == "ones":
        # batch=1, all ones -> G is all-ones outer product
        X = torch.ones(1, x_size, device=device)
        D = torch.ones(1, d_size, device=device)
    elif pattern == "eye":
        # batch=d_size=x_size, identity matrices -> G = I (approximately)
        batch = min(d_size, x_size)
        X = torch.eye(batch, x_size, device=device)
        D = torch.eye(batch, d_size, device=device)
    elif pattern == "rand":
        # Random but fixed pattern
        batch = 8
        X = torch.randn(batch, x_size, device=device)
        D = torch.randn(batch, d_size, device=device)
    else:
        raise ValueError(f"Unknown gradient pattern: {pattern}")

    # Compute target gradient G = D^T @ X
    G = D.t() @ X  # [d_size, x_size]

    return X, D, G


# =============================================================================
# Metrics Computation
# =============================================================================

@dataclass
class StepMetrics:
    """Metrics recorded at each update step."""
    step: int
    transfer_flag: int = 0

    # True weights norms
    norm_A: float = 0.0
    norm_B: float = 0.0
    norm_C: float = 0.0
    norm_AB: float = 0.0

    # Single element traces (legacy)
    a00: float = 0.0
    b00: float = 0.0
    c00: float = 0.0
    ab00: float = 0.0

    # === All A elements [3x1] ===
    A_0_0: float = 0.0
    A_1_0: float = 0.0
    A_2_0: float = 0.0

    # === All B elements [1x3] ===
    B_0_0: float = 0.0
    B_0_1: float = 0.0
    B_0_2: float = 0.0

    # === All C elements [3x3] ===
    C_0_0: float = 0.0
    C_0_1: float = 0.0
    C_0_2: float = 0.0
    C_1_0: float = 0.0
    C_1_1: float = 0.0
    C_1_2: float = 0.0
    C_2_0: float = 0.0
    C_2_1: float = 0.0
    C_2_2: float = 0.0

    # One-hot read values (NaN if not applicable)
    norm_Ahat: float = float('nan')
    norm_Bhat: float = float('nan')
    norm_ABhat: float = float('nan')
    read_err_A: float = float('nan')
    read_err_B: float = float('nan')
    a00_hat: float = float('nan')
    b00_hat: float = float('nan')
    ab00_hat: float = float('nan')

    # Reconstruction quality
    cos_sim_AB_negG: float = float('nan')

    # === NEW: Reconstruction & Transfer Metrics ===
    # 1. Reconstruction Relative Error: ||AB - (-G)||_F / ||G||_F
    recon_rel_err: float = float('nan')

    # 2. SVD Optimal Gap Ratio: ||AB - (-G)||_F / ||G - G_r^SVD||_F
    #    Measures how close AB is to optimal rank-r approximation of -G
    svd_gap_ratio: float = float('nan')

    # 3. Transfer Fidelity Error (only at transfer steps):
    #    ε_tx = ||ΔC - η·Ã·B̃||_F / max(||ΔC||_F, ||η·Ã·B̃||_F)
    transfer_fidelity_err: float = float('nan')

    # Supporting metrics for transfer fidelity
    norm_eta_AB_hat: float = float('nan')  # ||η·Ã·B̃||_F (expected transfer)
    norm_deltaC: float = float('nan')       # ||ΔC||_F (actual transfer)

    # Transfer event metrics (NaN if no transfer)
    deltaC_norm: float = float('nan')
    nonzero_ranks: int = 0
    max_reps: int = 0
    residual_max: float = float('nan')

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_frobenius_norm(tensor: torch.Tensor) -> float:
    """Compute Frobenius norm."""
    return torch.norm(tensor, p='fro').item()


def compute_svd_optimal_error(G: torch.Tensor, rank: int) -> float:
    """Compute ||G - G_r^SVD||_F, the error of optimal rank-r SVD approximation.

    Args:
        G: Target matrix [d_size, x_size]
        rank: Rank for truncated SVD

    Returns:
        Frobenius norm of (G - G_r^SVD)
    """
    # SVD: G = U @ S @ V^T
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)

    # Truncate to rank-r
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Optimal rank-r approximation: G_r = U_r @ diag(S_r) @ Vh_r
    G_r = U_r @ torch.diag(S_r) @ Vh_r

    # Error: ||G - G_r||_F = sqrt(sum of squared singular values beyond rank r)
    # This equals sqrt(S[rank]^2 + S[rank+1]^2 + ...)
    return torch.norm(G - G_r, p='fro').item()


def compute_cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> float:
    """Compute cosine similarity between flattened tensors."""
    a_flat = A.flatten()
    b_flat = B.flatten()

    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).item()


def compute_relative_error(estimate: torch.Tensor, true: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute relative Frobenius error with symmetric normalization.

    Uses symmetric normalization: ||est - true||_F / (max(||est||, ||true||) + eps)

    This handles the case where true ≈ 0 (e.g., after reinit) more gracefully:
    - If both are ~0: error ≈ 0 (both unmeasurable, no error)
    - If true ~0 but est ≠ 0: error = ||est|| / ||est|| = 1 (phantom reading)
    - If true ≠ 0 but est ~0: error = ||true|| / ||true|| = 1 (can't read)
    - If both ≠ 0: standard relative error behavior

    Note: For analog readout error analysis, error ≈ 1 when weights are too small
    to be detected by analog MVM is physically meaningful.
    """
    diff_norm = torch.norm(estimate - true, p='fro').item()
    est_norm = torch.norm(estimate, p='fro').item()
    true_norm = torch.norm(true, p='fro').item()
    max_norm = max(est_norm, true_norm)
    return diff_norm / (max_norm + eps)


# =============================================================================
# Main Experiment Runner
# =============================================================================

class LRTTToyExperiment:
    """LRTT Figure 3 style toy experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Create tiles
        self.tile_a, self.tile_b, self.tile_c = create_tiles(config, self.device)

        # Create LRTT controller
        self._create_controller()

        # Create gradient inputs
        self.X, self.D, self.G = create_gradient_inputs(
            config.gradient_pattern,
            config.d_size,
            config.x_size,
            self.device,
            config.seed
        )

        # Metrics storage
        self.metrics_history: List[StepMetrics] = []

    def _create_controller(self):
        """Create LRTT controller with configured parameters."""
        config = self.config

        # Compute effective transfer_lr
        if config.transfer_lr_scale == "sqrt_rank":
            transfer_lr = config.transfer_lr_base / np.sqrt(config.rank)
        else:
            transfer_lr = config.transfer_lr_base

        self.controller = LRTTController(
            tile_a=self.tile_a,
            tile_b=self.tile_b,
            tile_c=self.tile_c,
            rank=config.rank,
            d_size=config.d_size,
            x_size=config.x_size,
            transfer_lr=transfer_lr,
            transfer_every=config.transfer_every,
            units_in_mbatch=False,
            lora_alpha=config.lora_alpha,
            reinit_gain=config.reinit_gain,
            reinit_mode=config.reinit_mode,
            decay_factor=config.decay_factor,
            correct_gradient_magnitudes=False,
            rank_chunk=config.rank,
            ab_bl_mgmt={},
            transfer_bl_mgmt={},
            forward_inject=False,  # FIXED: always False
            use_onehot=config.use_onehot,
            use_sigma_delta=config.use_sigma_delta,
        )

        # Apply controller config options that are set post-construction
        self.controller.read_n_avg = config.read_n_avg
        self.controller.transfer_burst_limit = config.transfer_burst_limit

        # Initialize A/B tiles with reinit (sets up initial weights)
        self.controller.reinit()

    def _read_true_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read true weights from tiles."""
        A_true = read_A_true(self.tile_a, self.config.d_size, self.config.rank, self.device)
        B_true = read_B_true(self.tile_b, self.config.rank, self.config.x_size, self.device)
        C_true = read_C_true(self.tile_c, self.config.d_size, self.config.x_size, self.device)
        return A_true, B_true, C_true

    def _read_onehot_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Read weights via one-hot read method.

        This method always performs one-hot read for logging purposes,
        regardless of whether use_onehot is True for transfer.
        This allows comparing "analog readout vs true" for both modes.
        """
        # Use controller's one-hot read method
        if hasattr(self.controller, '_read_ab_onehot_symmetric'):
            A_hat, B_hat = self.controller._read_ab_onehot_symmetric()
            return A_hat, B_hat
        else:
            # Fallback: manual one-hot read not implemented
            return None, None

    def _compute_step_metrics(self, step: int, transfer_occurred: bool,
                               C_before: Optional[torch.Tensor] = None,
                               transfer_stats: Optional[Dict] = None,
                               A_hat_before: Optional[torch.Tensor] = None,
                               B_hat_before: Optional[torch.Tensor] = None) -> StepMetrics:
        """Compute all metrics for current step."""
        metrics = StepMetrics(step=step, transfer_flag=int(transfer_occurred))

        # Read true weights
        A_true, B_true, C_true = self._read_true_weights()
        AB_true = A_true @ B_true

        # True weight norms
        metrics.norm_A = compute_frobenius_norm(A_true)
        metrics.norm_B = compute_frobenius_norm(B_true)
        metrics.norm_C = compute_frobenius_norm(C_true)
        metrics.norm_AB = compute_frobenius_norm(AB_true)

        # Single element traces (legacy)
        metrics.a00 = A_true[0, 0].item() if A_true.numel() > 0 else 0.0
        metrics.b00 = B_true[0, 0].item() if B_true.numel() > 0 else 0.0
        metrics.c00 = C_true[0, 0].item() if C_true.numel() > 0 else 0.0
        metrics.ab00 = AB_true[0, 0].item() if AB_true.numel() > 0 else 0.0

        # === All A elements [d_size x rank] = [3 x 1] ===
        if A_true.shape[0] >= 3 and A_true.shape[1] >= 1:
            metrics.A_0_0 = A_true[0, 0].item()
            metrics.A_1_0 = A_true[1, 0].item()
            metrics.A_2_0 = A_true[2, 0].item()

        # === All B elements [rank x x_size] = [1 x 3] ===
        if B_true.shape[0] >= 1 and B_true.shape[1] >= 3:
            metrics.B_0_0 = B_true[0, 0].item()
            metrics.B_0_1 = B_true[0, 1].item()
            metrics.B_0_2 = B_true[0, 2].item()

        # === All C elements [d_size x x_size] = [3 x 3] ===
        if C_true.shape[0] >= 3 and C_true.shape[1] >= 3:
            metrics.C_0_0 = C_true[0, 0].item()
            metrics.C_0_1 = C_true[0, 1].item()
            metrics.C_0_2 = C_true[0, 2].item()
            metrics.C_1_0 = C_true[1, 0].item()
            metrics.C_1_1 = C_true[1, 1].item()
            metrics.C_1_2 = C_true[1, 2].item()
            metrics.C_2_0 = C_true[2, 0].item()
            metrics.C_2_1 = C_true[2, 1].item()
            metrics.C_2_2 = C_true[2, 2].item()

        # Cosine similarity: AB vs -G
        metrics.cos_sim_AB_negG = compute_cosine_similarity(AB_true, -self.G)

        # === NEW: Reconstruction & SVD Metrics ===
        # 1. Reconstruction Relative Error: ||AB - (-G)||_F / ||G||_F
        G_norm = compute_frobenius_norm(self.G)
        recon_err_norm = compute_frobenius_norm(AB_true - (-self.G))
        if G_norm > 1e-10:
            metrics.recon_rel_err = recon_err_norm / G_norm
        else:
            metrics.recon_rel_err = float('nan')

        # 2. SVD Optimal Gap Ratio: ||AB - (-G)||_F / ||G - G_r^SVD||_F
        # Note: If G is already rank-r (e.g., ones matrix is rank-1), SVD error ≈ 0
        # In this case, use ||G||_F as denominator instead (measures how close AB is to -G)
        svd_opt_err = compute_svd_optimal_error(-self.G, self.config.rank)
        if svd_opt_err > 1e-6:  # SVD optimal has meaningful error
            metrics.svd_gap_ratio = recon_err_norm / svd_opt_err
        elif G_norm > 1e-10:
            # G is already (near) rank-r, use reconstruction relative error instead
            # Ratio < 1 means AB is better than G itself (impossible), = 1 means perfect
            metrics.svd_gap_ratio = recon_err_norm / G_norm
        else:
            metrics.svd_gap_ratio = float('nan')

        # One-hot read metrics (always compute for comparison, even when use_onehot=False for transfer)
        # This allows comparing "analog readout vs true" for both modes
        if self.config.log_read_errors:
            A_hat, B_hat = self._read_onehot_weights()
            if A_hat is not None and B_hat is not None:
                AB_hat = A_hat @ B_hat

                metrics.norm_Ahat = compute_frobenius_norm(A_hat)
                metrics.norm_Bhat = compute_frobenius_norm(B_hat)
                metrics.norm_ABhat = compute_frobenius_norm(AB_hat)

                metrics.read_err_A = compute_relative_error(A_hat, A_true)
                metrics.read_err_B = compute_relative_error(B_hat, B_true)

                metrics.a00_hat = A_hat[0, 0].item() if A_hat.numel() > 0 else float('nan')
                metrics.b00_hat = B_hat[0, 0].item() if B_hat.numel() > 0 else float('nan')
                metrics.ab00_hat = AB_hat[0, 0].item() if AB_hat.numel() > 0 else float('nan')

        # Transfer event metrics
        if transfer_occurred and C_before is not None:
            delta_C = C_true - C_before
            metrics.deltaC_norm = compute_frobenius_norm(delta_C)
            metrics.norm_deltaC = metrics.deltaC_norm

            if transfer_stats is not None:
                metrics.nonzero_ranks = transfer_stats.get('nonzero_ranks', 0)
                metrics.max_reps = transfer_stats.get('max_reps', 0)
                metrics.residual_max = transfer_stats.get('residual_max', float('nan'))

            # 3. Transfer Fidelity Error: ε_tx = ||ΔC - η·Ã·B̃||_F / max(||ΔC||, ||η·Ã·B̃||)
            # Use A_hat_before, B_hat_before if provided
            if A_hat_before is not None and B_hat_before is not None:
                # η = transfer_lr (effective)
                eta = self.controller.transfer_lr
                eta_AB_hat = eta * (A_hat_before @ B_hat_before)
                norm_eta_AB_hat = compute_frobenius_norm(eta_AB_hat)
                metrics.norm_eta_AB_hat = norm_eta_AB_hat

                # Transfer fidelity error
                transfer_diff = delta_C - eta_AB_hat
                transfer_diff_norm = compute_frobenius_norm(transfer_diff)
                max_norm = max(metrics.deltaC_norm, norm_eta_AB_hat)

                if max_norm > 1e-10:
                    metrics.transfer_fidelity_err = transfer_diff_norm / max_norm
                else:
                    metrics.transfer_fidelity_err = 0.0 if transfer_diff_norm < 1e-10 else float('nan')

        return metrics

    def run(self) -> pd.DataFrame:
        """Run the experiment and return metrics DataFrame."""
        print(f"\n{'='*60}")
        print(f"Running LRTT Toy Experiment")
        print(f"{'='*60}")
        print(f"Config: rank={self.config.rank}, transfer_every={self.config.transfer_every}")
        print(f"        use_onehot={self.config.use_onehot}, use_sigma_delta={self.config.use_sigma_delta}")
        print(f"        reinit_mode={self.config.reinit_mode}, pattern={self.config.gradient_pattern}")
        print(f"        steps={self.config.steps}, device={self.device}")
        print(f"{'='*60}\n")

        self.metrics_history = []

        # Initial metrics (step 0)
        metrics = self._compute_step_metrics(0, False)
        self.metrics_history.append(metrics)

        # Main training loop
        for step in range(1, self.config.steps + 1):
            # AB update (reconstruction mode since forward_inject=False)
            self.controller.ab_weight_update(self.X, self.D, lr=self.config.lr_update)

            # Check for transfer
            transfer_occurred = False
            C_before = None
            transfer_stats = None
            A_hat_before = None
            B_hat_before = None

            if self.controller.should_transfer():
                # Read C before transfer
                C_before = read_C_true(self.tile_c, self.config.d_size, self.config.x_size, self.device).clone()

                # Read A_hat, B_hat BEFORE transfer for transfer fidelity calculation
                A_hat_before, B_hat_before = self._read_onehot_weights()
                if A_hat_before is not None:
                    A_hat_before = A_hat_before.clone()
                if B_hat_before is not None:
                    B_hat_before = B_hat_before.clone()

                # Perform transfer (reinit is called internally at end of transfer)
                self.controller.ab_weight_transfer(
                    use_onehot=self.config.use_onehot,
                    use_sigma_delta=self.config.use_sigma_delta
                )
                # NOTE: Do NOT call reinit() here - ab_weight_transfer() already calls it internally

                transfer_occurred = True

                # Try to get transfer stats (if available)
                if hasattr(self.controller, '_last_transfer_stats'):
                    transfer_stats = self.controller._last_transfer_stats

            # Compute metrics
            metrics = self._compute_step_metrics(
                step, transfer_occurred, C_before, transfer_stats,
                A_hat_before, B_hat_before
            )
            self.metrics_history.append(metrics)

            # Progress logging
            if step % 100 == 0 or step == self.config.steps:
                print(f"Step {step:4d}/{self.config.steps}: "
                      f"||A||={metrics.norm_A:.4f}, ||B||={metrics.norm_B:.4f}, "
                      f"||C||={metrics.norm_C:.4f}, ||AB||={metrics.norm_AB:.4f}, "
                      f"cos(AB,-G)={metrics.cos_sim_AB_negG:.4f}")

        # Convert to DataFrame
        df = pd.DataFrame([m.to_dict() for m in self.metrics_history])

        # Add config columns
        df['rank'] = self.config.rank
        df['transfer_every'] = self.config.transfer_every
        df['transfer_lr_base'] = self.config.transfer_lr_base
        df['transfer_lr_scale'] = self.config.transfer_lr_scale
        df['use_onehot'] = self.config.use_onehot
        df['use_sigma_delta'] = self.config.use_sigma_delta
        df['reinit_mode'] = self.config.reinit_mode
        df['read_n_avg'] = self.config.read_n_avg
        df['gradient_pattern'] = self.config.gradient_pattern

        return df


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_norm_traces(df: pd.DataFrame, output_path: Path, title_suffix: str = ""):
    """Plot norm traces (Figure 3 style)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = df['step'].values
    transfer_steps = df[df['transfer_flag'] == 1]['step'].values

    # Plot 1: Individual norms
    ax = axes[0, 0]
    ax.plot(steps, df['norm_A'], label='||A||_F')
    ax.plot(steps, df['norm_B'], label='||B||_F')
    ax.plot(steps, df['norm_C'], label='||C||_F')
    for ts in transfer_steps:
        ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Individual Weight Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: AB norm and C norm
    ax = axes[0, 1]
    ax.plot(steps, df['norm_AB'], label='||AB||_F')
    ax.plot(steps, df['norm_C'], label='||C||_F', alpha=0.7)
    for ts in transfer_steps:
        ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Product Norm ||AB||_F vs ||C||_F')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cosine similarity AB vs -G
    ax = axes[1, 0]
    ax.plot(steps, df['cos_sim_AB_negG'], label='cos(AB, -G)')
    for ts in transfer_steps:
        ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Reconstruction Quality: cos(AB, -G)')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Delta C norm at transfer steps
    ax = axes[1, 1]
    transfer_df = df[df['transfer_flag'] == 1]
    if len(transfer_df) > 0:
        ax.bar(transfer_df['step'], transfer_df['deltaC_norm'], width=max(1, len(steps)//100))
        ax.set_xlabel('Update Step')
        ax.set_ylabel('||ΔC||_F')
        ax.set_title('Transfer Magnitude ||ΔC||_F')
    else:
        ax.text(0.5, 0.5, 'No transfers occurred', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'LRTT Norm Traces {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_element_traces(df: pd.DataFrame, output_path: Path, title_suffix: str = ""):
    """Plot single element traces (a00, b00, c00, ab00)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = df['step'].values
    transfer_steps = df[df['transfer_flag'] == 1]['step'].values

    # Plot 1: a00
    ax = axes[0, 0]
    ax.plot(steps, df['a00'], label='A[0,0]')
    if 'a00_hat' in df.columns and not df['a00_hat'].isna().all():
        ax.plot(steps, df['a00_hat'], '--', label='A_hat[0,0]', alpha=0.7)
    for ts in transfer_steps:
        ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Value')
    ax.set_title('A[0,0] Element Trace')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: b00
    ax = axes[0, 1]
    ax.plot(steps, df['b00'], label='B[0,0]')
    if 'b00_hat' in df.columns and not df['b00_hat'].isna().all():
        ax.plot(steps, df['b00_hat'], '--', label='B_hat[0,0]', alpha=0.7)
    for ts in transfer_steps:
        ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Value')
    ax.set_title('B[0,0] Element Trace')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: c00
    ax = axes[1, 0]
    ax.plot(steps, df['c00'], label='C[0,0]')
    for ts in transfer_steps:
        ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Value')
    ax.set_title('C[0,0] Element Trace')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: ab00
    ax = axes[1, 1]
    ax.plot(steps, df['ab00'], label='(AB)[0,0]')
    if 'ab00_hat' in df.columns and not df['ab00_hat'].isna().all():
        ax.plot(steps, df['ab00_hat'], '--', label='(AB)_hat[0,0]', alpha=0.7)
    for ts in transfer_steps:
        ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Value')
    ax.set_title('(AB)[0,0] Element Trace')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'LRTT Element Traces {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_read_errors(df: pd.DataFrame, output_path: Path, title_suffix: str = ""):
    """Plot one-hot read errors (analog readout vs true).

    Read Error Interpretation:
    - error=1.0 after reinit: A weights too small for analog MVM to detect
    - error decreasing: weights growing, becoming readable
    - error spike at transfer: reinit resets A to near-zero
    """
    if df['read_err_A'].isna().all():
        print(f"  Skipping read error plot (no read error data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = df['step'].values
    transfer_steps = df[df['transfer_flag'] == 1]['step'].values

    # Plot 1: Read errors
    ax = axes[0]
    ax.plot(steps, df['read_err_A'], label='A read error', color='tab:blue')
    ax.plot(steps, df['read_err_B'], label='B read error', color='tab:orange')
    for ts in transfer_steps:
        ax.axvline(x=ts, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Unreadable threshold')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Relative Error: ||est-true|| / max(||est||,||true||)')
    ax.set_title('Analog Read Errors (error=1.0 means weights too small to detect)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Plot 2: Norm comparison (true vs hat)
    ax = axes[1]
    ax.plot(steps, df['norm_A'], label='||A||_F (true)', color='tab:blue')
    ax.plot(steps, df['norm_Ahat'], '--', label='||A_hat||_F (read)', color='tab:blue', alpha=0.7)
    ax.plot(steps, df['norm_B'], label='||B||_F (true)', color='tab:orange')
    ax.plot(steps, df['norm_Bhat'], '--', label='||B_hat||_F (read)', color='tab:orange', alpha=0.7)
    for ts in transfer_steps:
        ax.axvline(x=ts, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('True vs Analog Read Norms (gap = detection limit)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'LRTT Read Analysis {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_all_elements(df: pd.DataFrame, output_path: Path, title_suffix: str = ""):
    """Plot all A[3x1], B[1x3], C[3x3] element traces."""

    steps = df['step'].values
    transfer_steps = df[df['transfer_flag'] == 1]['step'].values

    # === Plot 1: A elements [3x1] ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    a_cols = ['A_0_0', 'A_1_0', 'A_2_0']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['A[0,0]', 'A[1,0]', 'A[2,0]']

    for col, color, label in zip(a_cols, colors, labels):
        if col in df.columns:
            ax.plot(steps, df[col], label=label, color=color, linewidth=1.5)

    for ts in transfer_steps:
        ax.axvline(x=ts, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Update Step', fontsize=12)
    ax.set_ylabel('Weight Value', fontsize=12)
    ax.set_title(f'A [3x1] Element Traces {title_suffix}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    a_path = output_path.parent / (output_path.stem + '_A.png')
    plt.savefig(a_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {a_path}")

    # === Plot 2: B elements [1x3] ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    b_cols = ['B_0_0', 'B_0_1', 'B_0_2']
    labels = ['B[0,0]', 'B[0,1]', 'B[0,2]']

    for col, color, label in zip(b_cols, colors, labels):
        if col in df.columns:
            ax.plot(steps, df[col], label=label, color=color, linewidth=1.5)

    for ts in transfer_steps:
        ax.axvline(x=ts, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Update Step', fontsize=12)
    ax.set_ylabel('Weight Value', fontsize=12)
    ax.set_title(f'B [1x3] Element Traces {title_suffix}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    b_path = output_path.parent / (output_path.stem + '_B.png')
    plt.savefig(b_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {b_path}")

    # === Plot 3: C elements [3x3] - 3x3 subplots ===
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    c_elements = [
        ['C_0_0', 'C_0_1', 'C_0_2'],
        ['C_1_0', 'C_1_1', 'C_1_2'],
        ['C_2_0', 'C_2_1', 'C_2_2'],
    ]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            col = c_elements[i][j]

            if col in df.columns:
                ax.plot(steps, df[col], color='tab:purple', linewidth=1.5)

            for ts in transfer_steps:
                ax.axvline(x=ts, color='red', linestyle='--', alpha=0.5, linewidth=1)

            ax.set_title(f'C[{i},{j}]', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

            if i == 2:
                ax.set_xlabel('Step', fontsize=10)
            if j == 0:
                ax.set_ylabel('Value', fontsize=10)

    plt.suptitle(f'C [3x3] Element Traces {title_suffix}', fontsize=14)
    plt.tight_layout()
    c_path = output_path.parent / (output_path.stem + '_C.png')
    plt.savefig(c_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {c_path}")

    # === Plot 4: Combined A, B, C overview ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # A elements
    ax = axes[0]
    for col, color, label in zip(a_cols, colors, labels):
        if col in df.columns:
            ax.plot(steps, df[col], label=label, color=color, linewidth=1.5)
    for ts in transfer_steps:
        ax.axvline(x=ts, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_ylabel('A Value', fontsize=11)
    ax.set_title('A [3x1] Elements', fontsize=12)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # B elements
    ax = axes[1]
    b_labels = ['B[0,0]', 'B[0,1]', 'B[0,2]']
    for col, color, label in zip(b_cols, colors, b_labels):
        if col in df.columns:
            ax.plot(steps, df[col], label=label, color=color, linewidth=1.5)
    for ts in transfer_steps:
        ax.axvline(x=ts, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_ylabel('B Value', fontsize=11)
    ax.set_title('B [1x3] Elements', fontsize=12)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # C elements (all 9)
    ax = axes[2]
    c_colors = plt.cm.viridis(np.linspace(0, 1, 9))
    c_flat = ['C_0_0', 'C_0_1', 'C_0_2', 'C_1_0', 'C_1_1', 'C_1_2', 'C_2_0', 'C_2_1', 'C_2_2']
    c_labels = ['C[0,0]', 'C[0,1]', 'C[0,2]', 'C[1,0]', 'C[1,1]', 'C[1,2]', 'C[2,0]', 'C[2,1]', 'C[2,2]']
    for col, color, label in zip(c_flat, c_colors, c_labels):
        if col in df.columns:
            ax.plot(steps, df[col], label=label, color=color, linewidth=1.2)
    for ts in transfer_steps:
        ax.axvline(x=ts, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Update Step', fontsize=11)
    ax.set_ylabel('C Value', fontsize=11)
    ax.set_title('C [3x3] Elements', fontsize=12)
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.suptitle(f'All Weight Element Traces {title_suffix}', fontsize=14)
    plt.tight_layout()
    combined_path = output_path.parent / (output_path.stem + '_combined.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {combined_path}")


def plot_reconstruction_metrics(df: pd.DataFrame, output_path: Path, title_suffix: str = ""):
    """Plot reconstruction and transfer fidelity metrics.

    Three metrics:
    1. Reconstruction Relative Error: ||AB - (-G)||_F / ||G||_F
    2. SVD Optimal Gap Ratio: ||AB - (-G)||_F / ||G - G_r^SVD||_F
    3. Transfer Fidelity Error: ε_tx = ||ΔC - η·Ã·B̃||_F / max(||ΔC||, ||η·Ã·B̃||)
    """
    # Check if new metrics exist
    if 'recon_rel_err' not in df.columns or df['recon_rel_err'].isna().all():
        print(f"  Skipping reconstruction metrics plot (no data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = df['step'].values
    transfer_steps = df[df['transfer_flag'] == 1]['step'].values

    # Plot 1: Reconstruction Relative Error
    ax = axes[0, 0]
    ax.plot(steps, df['recon_rel_err'], label='||AB - (-G)||_F / ||G||_F', color='tab:blue')
    for ts in transfer_steps:
        ax.axvline(x=ts, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Relative Error')
    ax.set_title('Reconstruction Relative Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 2: SVD Optimal Gap Ratio
    ax = axes[0, 1]
    svd_gap = df['svd_gap_ratio'].values
    # Filter out NaN for better visualization
    valid_mask = ~np.isnan(svd_gap)
    if valid_mask.any():
        ax.plot(steps[valid_mask], svd_gap[valid_mask], label='||AB - (-G)||_F / ||G - G_r^SVD||_F',
                color='tab:orange', marker='o', markersize=2, linestyle='-')
        for ts in transfer_steps:
            ax.axvline(x=ts, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.7, label='Optimal (ratio=1)')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Gap Ratio')
        ax.set_title('SVD Optimal Gap Ratio (1.0 = optimal rank-r approx)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No valid SVD gap ratio data', ha='center', va='center', transform=ax.transAxes)

    # Plot 3: Transfer Fidelity Error (only at transfer steps)
    ax = axes[1, 0]
    transfer_df = df[df['transfer_flag'] == 1]
    if len(transfer_df) > 0 and not transfer_df['transfer_fidelity_err'].isna().all():
        valid_transfer = transfer_df[~transfer_df['transfer_fidelity_err'].isna()]
        if len(valid_transfer) > 0:
            ax.bar(valid_transfer['step'], valid_transfer['transfer_fidelity_err'],
                   width=max(1, len(steps)//100), color='tab:green', alpha=0.7,
                   label='||ΔC - η·Ã·B̃||_F / max(||ΔC||, ||η·Ã·B̃||)')
            ax.set_xlabel('Update Step (transfer events)')
            ax.set_ylabel('Fidelity Error')
            ax.set_title('Transfer Fidelity Error (0 = perfect transfer)')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
        else:
            ax.text(0.5, 0.5, 'No valid transfer fidelity data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No transfer events', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # Plot 4: Transfer magnitudes comparison
    ax = axes[1, 1]
    if len(transfer_df) > 0:
        valid_transfer = transfer_df[~transfer_df['norm_deltaC'].isna() & ~transfer_df['norm_eta_AB_hat'].isna()]
        if len(valid_transfer) > 0:
            width = max(1, len(steps)//200)
            x_pos = valid_transfer['step'].values
            ax.bar(x_pos - width/2, valid_transfer['norm_deltaC'], width=width,
                   label='||ΔC||_F (actual)', color='tab:blue', alpha=0.7)
            ax.bar(x_pos + width/2, valid_transfer['norm_eta_AB_hat'], width=width,
                   label='||η·Ã·B̃||_F (expected)', color='tab:orange', alpha=0.7)
            ax.set_xlabel('Update Step (transfer events)')
            ax.set_ylabel('Frobenius Norm')
            ax.set_title('Transfer Magnitude: Actual vs Expected')
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No valid transfer magnitude data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No transfer events', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'LRTT Reconstruction & Transfer Metrics {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Result Saving
# =============================================================================

def save_results(df: pd.DataFrame, config: ExperimentConfig, output_dir: Path):
    """Save experiment results."""
    run_dir = output_dir / f"run_{config.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {run_dir}")

    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved: {config_path}")

    # Save full trace CSV
    csv_path = run_dir / "trace.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save additional CSV files for analysis
    # 1. Transfer events only
    transfer_df = df[df['transfer_flag'] == 1]
    if len(transfer_df) > 0:
        transfer_cols = ['step', 'norm_A', 'norm_B', 'norm_AB',
                        'norm_Ahat', 'norm_Bhat', 'norm_ABhat',
                        'read_err_A', 'read_err_B',
                        'recon_rel_err', 'svd_gap_ratio',
                        'transfer_fidelity_err', 'norm_eta_AB_hat', 'norm_deltaC',
                        'norm_C', 'deltaC_norm']
        transfer_cols = [c for c in transfer_cols if c in df.columns]
        transfer_df[transfer_cols].to_csv(run_dir / "transfer_events.csv", index=False)
        print(f"  Saved: {run_dir / 'transfer_events.csv'}")

    # 2. Weight elements (A, B, C)
    element_cols = ['step', 'transfer_flag']
    element_cols += [c for c in df.columns if c.startswith('A_') or c.startswith('B_') or c.startswith('C_')]
    if len(element_cols) > 2:
        df[element_cols].to_csv(run_dir / "weight_elements.csv", index=False)
        print(f"  Saved: {run_dir / 'weight_elements.csv'}")

    # 3. Reconstruction metrics
    recon_cols = ['step', 'transfer_flag', 'norm_AB', 'cos_sim_AB_negG',
                  'recon_rel_err', 'svd_gap_ratio',
                  'transfer_fidelity_err', 'norm_eta_AB_hat', 'norm_deltaC']
    recon_cols = [c for c in recon_cols if c in df.columns]
    df[recon_cols].to_csv(run_dir / "reconstruction_metrics.csv", index=False)
    print(f"  Saved: {run_dir / 'reconstruction_metrics.csv'}")

    # 4. Read errors
    read_cols = ['step', 'transfer_flag',
                 'norm_A', 'norm_Ahat', 'read_err_A',
                 'norm_B', 'norm_Bhat', 'read_err_B',
                 'norm_AB', 'norm_ABhat']
    read_cols = [c for c in read_cols if c in df.columns]
    df[read_cols].to_csv(run_dir / "read_errors.csv", index=False)
    print(f"  Saved: {run_dir / 'read_errors.csv'}")

    # Generate title suffix for plots
    title_suffix = f"(r={config.rank}, te={config.transfer_every}, {config.reinit_mode})"

    # Save plots
    plot_norm_traces(df, run_dir / "trace.png", title_suffix)
    plot_element_traces(df, run_dir / "trace_elements.png", title_suffix)
    plot_read_errors(df, run_dir / "trace_read_errors.png", title_suffix)
    plot_all_elements(df, run_dir / "all_elements.png", title_suffix)
    plot_reconstruction_metrics(df, run_dir / "trace_reconstruction.png", title_suffix)

    return run_dir


# =============================================================================
# Sweep Runner
# =============================================================================

def get_sweep_configs(subset: str = "full") -> List[Tuple]:
    """Get parameter combinations for sweep.

    Returns tuples of:
        (rank, transfer_every, transfer_lr_base, transfer_lr_scale,
         use_onehot, use_sigma_delta, reinit_mode, gradient_pattern,
         lr_update, c_device_type)
    """

    if subset == "minimal":
        # Minimal sweep for quick testing
        return list(product(
            [4],            # rank
            [25],           # transfer_every
            [0.1],          # transfer_lr_base
            ["sqrt_rank"],  # transfer_lr_scale
            [True, False],  # use_onehot
            [True],         # use_sigma_delta
            ["orthogonal"], # reinit_mode
            ["ones"],       # gradient_pattern
            [0.01],         # lr_update
            ["ideal"],      # c_device_type
        ))

    elif subset == "small":
        # Small sweep for testing
        return list(product(
            [2, 4],             # rank
            [5, 25],            # transfer_every
            [0.1],              # transfer_lr_base
            ["sqrt_rank"],      # transfer_lr_scale
            [True, False],      # use_onehot
            [True],             # use_sigma_delta
            ["orthogonal"],     # reinit_mode
            ["ones"],           # gradient_pattern
            [0.01],             # lr_update
            ["ideal", "ecram"], # c_device_type
        ))

    elif subset == "main":
        # Main sweep: all requested parameter combinations
        # 1) use_onehot: True/False
        # 2) rank: 1,2,3,4,5
        # 3) transfer_every, transfer_lr, lr_update variations
        # 4) C tile: Idealized, EcRAM
        return list(product(
            [1, 2, 3, 4, 5],        # rank
            [10, 25, 50, 100],      # transfer_every
            [0.05, 0.1, 0.5],       # transfer_lr_base
            ["sqrt_rank"],          # transfer_lr_scale
            [True, False],          # use_onehot
            [True],                 # use_sigma_delta (True when onehot)
            ["orthogonal"],         # reinit_mode
            ["ones"],               # gradient_pattern
            [0.005, 0.01, 0.05],    # lr_update
            ["ideal", "ecram"],     # c_device_type
        ))

    elif subset == "rank_study":
        # Focus on rank variation with key parameters
        return list(product(
            [1, 2, 3, 4, 5],        # rank
            [25],                   # transfer_every
            [0.1],                  # transfer_lr_base
            ["sqrt_rank"],          # transfer_lr_scale
            [True, False],          # use_onehot
            [True],                 # use_sigma_delta
            ["orthogonal"],         # reinit_mode
            ["ones"],               # gradient_pattern
            [0.01],                 # lr_update
            ["ideal", "ecram"],     # c_device_type
        ))

    elif subset == "transfer_study":
        # Focus on transfer parameters
        return list(product(
            [4],                        # rank
            [5, 10, 25, 50, 100],       # transfer_every
            [0.01, 0.05, 0.1, 0.5, 1.0], # transfer_lr_base
            ["sqrt_rank"],              # transfer_lr_scale
            [True, False],              # use_onehot
            [True],                     # use_sigma_delta
            ["orthogonal"],             # reinit_mode
            ["ones"],                   # gradient_pattern
            [0.01],                     # lr_update
            ["ideal"],                  # c_device_type
        ))

    elif subset == "lr_study":
        # Focus on learning rate variations
        return list(product(
            [4],                            # rank
            [25],                           # transfer_every
            [0.1],                          # transfer_lr_base
            ["sqrt_rank"],                  # transfer_lr_scale
            [True, False],                  # use_onehot
            [True],                         # use_sigma_delta
            ["orthogonal"],                 # reinit_mode
            ["ones"],                       # gradient_pattern
            [0.001, 0.005, 0.01, 0.05, 0.1], # lr_update
            ["ideal", "ecram"],             # c_device_type
        ))

    elif subset == "c_device_study":
        # Focus on C device comparison
        return list(product(
            [2, 4],                     # rank
            [25, 50],                   # transfer_every
            [0.1],                      # transfer_lr_base
            ["sqrt_rank"],              # transfer_lr_scale
            [True, False],              # use_onehot
            [True],                     # use_sigma_delta
            ["orthogonal"],             # reinit_mode
            ["ones"],                   # gradient_pattern
            [0.01],                     # lr_update
            ["ideal", "ecram", "pcm", "rram"],  # c_device_type
        ))

    elif subset == "onehot_compare":
        # Compare one-hot vs direct read
        return list(product(
            [2, 4],             # rank
            [25],               # transfer_every
            [0.1, 0.5],         # transfer_lr_base
            ["sqrt_rank"],      # transfer_lr_scale
            [True, False],      # use_onehot
            [True, False],      # use_sigma_delta
            ["orthogonal"],     # reinit_mode
            ["ones", "eye"],    # gradient_pattern
            [0.01],             # lr_update
            ["ideal"],          # c_device_type
        ))

    elif subset == "reinit_compare":
        # Compare reinit modes
        return list(product(
            [2, 4],             # rank
            [25, 100],          # transfer_every
            [0.1],              # transfer_lr_base
            ["sqrt_rank"],      # transfer_lr_scale
            [True],             # use_onehot
            [True],             # use_sigma_delta
            ["orthogonal", "standard", "decay", "hybrid"],  # reinit_mode
            ["ones"],           # gradient_pattern
            [0.01],             # lr_update
            ["ideal"],          # c_device_type
        ))

    else:  # full - comprehensive sweep
        return list(product(
            [1, 2, 3, 4, 5],         # rank
            [5, 10, 25, 50, 100],    # transfer_every
            [0.05, 0.1, 0.5, 1.0],   # transfer_lr_base
            ["sqrt_rank"],           # transfer_lr_scale (fixed for simplicity)
            [True, False],           # use_onehot
            [True],                  # use_sigma_delta
            ["orthogonal"],          # reinit_mode
            ["ones"],                # gradient_pattern
            [0.005, 0.01, 0.05],     # lr_update
            ["ideal", "ecram"],      # c_device_type
        ))


def run_sweep(args):
    """Run parameter sweep."""
    sweep_configs = get_sweep_configs(args.sweep_subset)
    total_configs = len(sweep_configs)

    print(f"\n{'='*70}")
    print(f"LRTT Parameter Sweep")
    print(f"{'='*70}")
    print(f"Subset: {args.sweep_subset}")
    print(f"Total configurations: {total_configs}")
    print(f"Steps per run: {args.steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    start_time = time.time()

    for i, config_tuple in enumerate(sweep_configs):
        # Unpack configuration tuple
        (rank, te, tlr, tlr_scale, onehot, sigma_delta,
         reinit, pattern, lr_update, c_device) = config_tuple

        print(f"\n[{i+1}/{total_configs}] Running experiment:")
        print(f"  rank={rank}, transfer_every={te}, transfer_lr={tlr}, lr={lr_update}")
        print(f"  onehot={onehot}, sigma_delta={sigma_delta}, reinit={reinit}")
        print(f"  C_device={c_device}, pattern={pattern}")

        config = ExperimentConfig(
            rank=rank,
            transfer_every=te,
            transfer_lr_base=tlr,
            transfer_lr_scale=tlr_scale,
            use_onehot=onehot,
            use_sigma_delta=sigma_delta,
            reinit_mode=reinit,
            gradient_pattern=pattern,
            steps=args.steps,
            lr_update=lr_update,
            seed=args.seed,
            ab_io_noise=args.ab_io_noise,
            c_device_type=c_device,
            output_dir=args.output_dir,
        )

        try:
            experiment = LRTTToyExperiment(config)
            df = experiment.run()
            run_dir = save_results(df, config, output_dir)
            all_results.append({
                'config': config.to_dict(),
                'success': True,
                'dir': str(run_dir),
                'tag': config.tag
            })
            print(f"  SUCCESS: Saved to {run_dir}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                'config': config.to_dict(),
                'success': False,
                'error': str(e),
                'tag': config.tag
            })

        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total_configs - i - 1)
        print(f"  Progress: {i+1}/{total_configs} ({100*(i+1)/total_configs:.1f}%) "
              f"| Elapsed: {elapsed/60:.1f}m | Remaining: {remaining/60:.1f}m")

    # Save sweep summary
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r['success'])
    failed = total_configs - successful

    print(f"\n{'='*70}")
    print(f"Sweep Complete!")
    print(f"{'='*70}")
    print(f"Total runs: {total_configs}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per run: {total_time/total_configs:.1f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {summary_path}")
    print(f"{'='*70}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LRTT Figure 3 Toy Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single run with default parameters
    python experiments/lrtt_fig3_toy_trace.py --run_single

    # Single run with specific parameters
    python experiments/lrtt_fig3_toy_trace.py --run_single \\
        --rank 4 --transfer_every 25 --transfer_lr_base 0.1 \\
        --use_onehot --use_sigma_delta --reinit_mode orthogonal

    # Parameter sweep
    python experiments/lrtt_fig3_toy_trace.py --sweep --sweep_subset small

    # With noisy device
    python experiments/lrtt_fig3_toy_trace.py --run_single --device noisy
        """
    )

    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--run_single', action='store_true', help='Run single experiment')
    mode_group.add_argument('--sweep', action='store_true', help='Run parameter sweep')

    # Sweep options
    parser.add_argument('--sweep_subset', type=str, default='small',
                        choices=['minimal', 'small', 'main', 'rank_study', 'transfer_study',
                                 'lr_study', 'c_device_study', 'onehot_compare', 'reinit_compare', 'full'],
                        help='Sweep subset to run (default: small)')

    # LRTT parameters
    parser.add_argument('--rank', type=int, default=1, help='LoRA rank (default: 1)')
    parser.add_argument('--transfer_every', type=int, default=25, help='Transfer frequency (default: 25)')
    parser.add_argument('--transfer_lr_base', type=float, default=0.1, help='Base transfer LR (default: 0.1)')
    parser.add_argument('--transfer_lr_scale', type=str, default='sqrt_rank',
                        choices=['none', 'sqrt_rank'], help='Transfer LR scaling (default: sqrt_rank)')
    parser.add_argument('--reinit_mode', type=str, default='orthogonal',
                        choices=['standard', 'orthogonal', 'decay', 'hybrid'],
                        help='Reinit mode (default: orthogonal)')
    parser.add_argument('--reinit_gain', type=float, default=0.1, help='Reinit gain (default: 0.1)')

    # Transfer mode
    parser.add_argument('--use_onehot', action='store_true', help='Use one-hot read for transfer')
    parser.add_argument('--no_onehot', action='store_true', help='Disable one-hot read')
    parser.add_argument('--use_sigma_delta', action='store_true', help='Use sigma-delta for transfer')
    parser.add_argument('--no_sigma_delta', action='store_true', help='Disable sigma-delta')

    # Experiment parameters
    parser.add_argument('--steps', type=int, default=500, help='Number of update steps (default: 500)')
    parser.add_argument('--lr_update', type=float, default=0.01, help='Update learning rate (default: 0.01)')
    parser.add_argument('--gradient_pattern', type=str, default='ones',
                        choices=['ones', 'eye', 'rand'], help='Gradient pattern (default: ones)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    # IO Noise configuration (A/B tiles always use 6T1C device)
    parser.add_argument('--ab_io_noise', type=str, default='off',
                        choices=['off', 'standard'],
                        help='A/B tile IO noise: off=no IO noise, standard=0.02 noise (default: off). '
                             'Note: A/B tiles always use 6T1C device regardless of this setting.')
    parser.add_argument('--c_device', type=str, default='ideal',
                        choices=['ideal', 'pcm', 'rram', 'ecram'],
                        help='Device type for C tile (default: ideal)')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/lrtt_fig3_toy',
                        help='Output directory (default: results/lrtt_fig3_toy)')
    parser.add_argument('--tag', type=str, default='', help='Run tag (auto-generated if empty)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle one-hot flags
    if args.no_onehot:
        use_onehot = False
    elif args.use_onehot:
        use_onehot = True
    else:
        use_onehot = True  # default

    # Handle sigma-delta flags
    if args.no_sigma_delta:
        use_sigma_delta = False
    elif args.use_sigma_delta:
        use_sigma_delta = True
    else:
        use_sigma_delta = False  # default: disabled

    if args.sweep:
        run_sweep(args)
    else:
        # Single run
        config = ExperimentConfig(
            rank=args.rank,
            transfer_every=args.transfer_every,
            transfer_lr_base=args.transfer_lr_base,
            transfer_lr_scale=args.transfer_lr_scale,
            use_onehot=use_onehot,
            use_sigma_delta=use_sigma_delta,
            reinit_mode=args.reinit_mode,
            reinit_gain=args.reinit_gain,
            gradient_pattern=args.gradient_pattern,
            steps=args.steps,
            lr_update=args.lr_update,
            seed=args.seed,
            ab_io_noise=args.ab_io_noise,
            c_device_type=args.c_device,
            output_dir=args.output_dir,
            tag=args.tag,
        )

        experiment = LRTTToyExperiment(config)
        df = experiment.run()

        output_dir = Path(args.output_dir)
        save_results(df, config, output_dir)

        print("\nExperiment complete!")


if __name__ == "__main__":
    main()
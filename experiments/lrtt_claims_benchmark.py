#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LRTT Claims Benchmark: Experimental Verification of LR-TT Properties

This script experimentally verifies three core claims about LR-TT:

Claim A (Expressiveness + Optimality):
    - LR-TT can learn rank-r approximations AB ≈ -G
    - Converged AB approaches optimal truncated SVD approximation error
    - Measured by: recon_rel_err = ||AB - (-G)||_F / ||G||_F
    - Measured by: svd_gap_ratio = ||AB - (-G)||_F / ||G - G_r^SVD||_F → 1

Claim B (SGD Equivalence):
    - Transfer step ΔC ≈ η_tx * AB ≈ -η_tx * G_r
    - Equivalent to low-rank approximated SGD step
    - Measured by: err_to_sgd = ||ΔC - (-η_tx * G_r^SVD)||_F / max(||ΔC||, ||η_tx * G_r^SVD||)

Claim C (Analog Read/Write Reality):
    - Separate algorithmic error from analog implementation error
    - Algorithmic: ||AB_true - (-G_r^SVD)||_F / ||G||_F
    - Read error: ||AB_hat - AB_true||_F / max(||AB_hat||, ||AB_true||)
    - Write/transfer error: ||ΔC - η_tx * AB_hat||_F / max(||ΔC||, ||η_tx * AB_hat||)

Usage examples:
    # Basic run with default settings (use_onehot=True, use_sigma_delta=True)
    python experiments/lrtt_claims_benchmark.py --run_single

    # Run with specific G mode
    python experiments/lrtt_claims_benchmark.py --run_single --G_mode lowrank --true_rank 2 --rank 2

    # Run Claim A sweep across ranks
    python experiments/lrtt_claims_benchmark.py --sweep_rank --G_mode decay --ranks 1,2,4,8

    # Run with direct transfer for comparison
    python experiments/lrtt_claims_benchmark.py --run_single --baseline_direct_transfer

    # Load MNIST gradient snapshot
    python experiments/lrtt_claims_benchmark.py --run_single --G_mode load_snapshot --snapshot_path snapshots/mnist_fc1_step100.pt

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# AIHWKit imports
from aihwkit.simulator.configs import SingleRPUConfig, IOParameters, UpdateParameters, PulseType
from aihwkit.simulator.configs import NoiseManagementType, BoundManagementType
from aihwkit.simulator.configs.devices import ConstantStepDevice, LinearStepDevice, IdealDevice
from aihwkit.simulator.presets.devices import IdealizedPresetDevice, PCMPresetDevice, ReRamESPresetDevice
from aihwkit.simulator.tiles.analog import AnalogTile

# LRTT imports
from aihwkit.simulator.tiles.lrtt_controller import LRTTController
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice, PythonLRTTPreset


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for LRTT Claims Benchmark."""

    # Matrix dimensions
    d_size: int = 16
    x_size: int = 16

    # LRTT parameters
    rank: int = 4
    transfer_every: int = 10000  # Very large for Claim A (no transfer during updates)
    transfer_lr_base: float = 0.1
    transfer_lr_scale: str = "sqrt_rank"
    lora_alpha: float = 1.0
    reinit_mode: str = "standard"  # Default as specified
    reinit_gain: float = 0.1
    decay_factor: float = 0.9

    # Transfer mode - DEFAULTS as specified
    use_onehot: bool = True      # Default: True
    use_sigma_delta: bool = True  # Default: True

    # Training parameters
    steps: int = 500
    lr_update: float = 0.01

    # G generation parameters
    G_mode: str = "lowrank"  # lowrank, decay, flat, rand_full, identity, load_npy, load_snapshot
    true_rank: int = 2        # For lowrank mode
    decay_rate: float = 0.5   # For decay mode (geometric decay)
    G_scale: float = 1.0      # Scale factor for G
    snapshot_path: str = ""   # Path for load_npy/load_snapshot modes

    # Experiment control
    seed: int = 42
    log_every: int = 10       # Metrics logging frequency

    # Comparison options
    baseline_direct_transfer: bool = False  # Also run direct transfer for comparison

    # Device configuration
    ab_io_noise: str = "off"
    c_device_type: str = "ideal"

    # Read settings
    read_n_avg: int = 1
    transfer_burst_limit: int = 10

    # Output
    output_dir: str = "results/lrtt_claims_benchmark"
    tag: str = ""

    def __post_init__(self):
        if not self.tag:
            onehot_str = "onehot" if self.use_onehot else "direct"
            sd_str = "sd" if self.use_sigma_delta else "nosd"
            self.tag = (
                f"{self.G_mode}_r{self.rank}_"
                f"{onehot_str}_{sd_str}_"
                f"d{self.d_size}x{self.x_size}"
            )

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# G Generation and Factorization
# =============================================================================

def generate_target_G(
    mode: str,
    d_size: int,
    x_size: int,
    seed: int = 42,
    true_rank: int = 2,
    decay_rate: float = 0.5,
    scale: float = 1.0,
    load_path: str = ""
) -> torch.Tensor:
    """Generate target gradient G with specified spectral properties.

    Args:
        mode: Generation mode:
            - "lowrank": Exactly low-rank G with rank=true_rank
            - "decay": Full-rank G with geometrically decaying singular values
            - "flat": Full-rank G with nearly uniform singular values (negative control)
            - "rand_full": Random full-rank G
            - "identity": Identity-like G (scaled to match dimensions)
            - "load_npy": Load from .npy file
            - "load_snapshot": Load from PyTorch snapshot
        d_size: Output dimension
        x_size: Input dimension
        seed: Random seed
        true_rank: Rank for lowrank mode
        decay_rate: Decay rate for decay mode (σ_i = decay_rate^i)
        scale: Scale factor for G
        load_path: Path for load_npy/load_snapshot modes

    Returns:
        G: Target gradient [d_size, x_size]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    min_dim = min(d_size, x_size)

    if mode == "lowrank":
        # Generate exactly rank-r matrix using SVD construction
        assert true_rank <= min_dim, f"true_rank ({true_rank}) must be <= min(d,x) ({min_dim})"

        U = torch.randn(d_size, true_rank)
        U, _ = torch.linalg.qr(U)  # Orthogonalize

        V = torch.randn(x_size, true_rank)
        V, _ = torch.linalg.qr(V)  # Orthogonalize

        # Random singular values
        S = torch.rand(true_rank) * scale + 0.1 * scale
        S = S.sort(descending=True).values  # Sorted descending

        G = U @ torch.diag(S) @ V.t()

    elif mode == "decay":
        # Full-rank with geometrically decaying singular values
        U = torch.randn(d_size, min_dim)
        U, _ = torch.linalg.qr(U)

        V = torch.randn(x_size, min_dim)
        V, _ = torch.linalg.qr(V)

        # Geometric decay: σ_i = scale * decay_rate^i
        S = torch.tensor([scale * (decay_rate ** i) for i in range(min_dim)])

        G = U @ torch.diag(S) @ V.t()

    elif mode == "flat":
        # Flat spectrum (negative control - hard to approximate with low rank)
        U = torch.randn(d_size, min_dim)
        U, _ = torch.linalg.qr(U)

        V = torch.randn(x_size, min_dim)
        V, _ = torch.linalg.qr(V)

        # Nearly uniform singular values
        S = torch.ones(min_dim) * scale + torch.randn(min_dim) * 0.01 * scale

        G = U @ torch.diag(S) @ V.t()

    elif mode == "rand_full":
        # Random full-rank matrix
        G = torch.randn(d_size, x_size) * scale

    elif mode == "identity":
        # Identity-like matrix
        G = torch.eye(d_size, x_size) * scale

    elif mode == "load_npy":
        # Load from numpy file
        if not load_path:
            raise ValueError("load_path required for load_npy mode")
        G = torch.from_numpy(np.load(load_path)).float()
        if G.shape != (d_size, x_size):
            raise ValueError(f"Loaded G shape {G.shape} != expected ({d_size}, {x_size})")

    elif mode == "load_snapshot":
        # Load from PyTorch snapshot (expects dict with 'G' key)
        if not load_path:
            raise ValueError("load_path required for load_snapshot mode")
        snapshot = torch.load(load_path, map_location='cpu')
        if isinstance(snapshot, dict):
            G = snapshot.get('G', snapshot.get('gradient', None))
            if G is None:
                raise ValueError("Snapshot must contain 'G' or 'gradient' key")
        else:
            G = snapshot
        G = G.float()
        # Update dimensions from loaded G
        d_size, x_size = G.shape

    else:
        raise ValueError(f"Unknown G mode: {mode}")

    return G * scale if mode not in ["load_npy", "load_snapshot"] else G


def factorize_G_to_XD(G: torch.Tensor, batch: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factorize G into X, D such that D.T @ X = G.

    Uses SVD to create a valid factorization.

    Args:
        G: Target gradient [d_size, x_size]
        batch: Optional batch size (default: min(d_size, x_size))

    Returns:
        X: [batch, x_size]
        D: [batch, d_size]
    """
    d_size, x_size = G.shape
    min_dim = min(d_size, x_size)

    if batch is None:
        batch = min_dim

    # SVD: G = U @ S @ V^T
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)

    # Use top-batch components
    effective_rank = min(batch, len(S))

    # Create X and D such that D.T @ X = G
    # D = U @ sqrt(S), X = sqrt(S) @ V^T
    # Then D.T @ X = sqrt(S) @ U.T @ U @ sqrt(S) @ V^T = S @ V^T (if U orthogonal)

    # More direct construction:
    # Let D = U[:, :r], X = (S[:r].unsqueeze(1) * Vh[:r, :])
    # Then D.T @ X = U[:, :r].T @ (U[:, :r] @ diag(S[:r]) @ Vh[:r, :]) doesn't work directly

    # Correct approach: use identity to ensure D.T @ X = G
    # Set D.T = U @ diag(sqrt(S)), X = diag(sqrt(S)) @ V^T
    # Then D.T @ X = U @ diag(sqrt(S)) @ diag(sqrt(S)) @ V^T = U @ S @ V^T = G

    sqrt_S = torch.sqrt(S[:effective_rank])

    # D: [batch, d_size] = (U[:, :r] @ diag(sqrt_S)).T
    D = (U[:, :effective_rank] * sqrt_S.unsqueeze(0)).t()  # [r, d_size]

    # X: [batch, x_size] = diag(sqrt_S) @ Vh[:r, :]
    X = sqrt_S.unsqueeze(1) * Vh[:effective_rank, :]  # [r, x_size]

    # Pad to batch size if needed
    if effective_rank < batch:
        pad_d = torch.zeros(batch - effective_rank, d_size, device=G.device, dtype=G.dtype)
        pad_x = torch.zeros(batch - effective_rank, x_size, device=G.device, dtype=G.dtype)
        D = torch.cat([D, pad_d], dim=0)
        X = torch.cat([X, pad_x], dim=0)

    # Verification
    reconstructed = D.t() @ X
    rel_err = torch.norm(reconstructed - G) / (torch.norm(G) + 1e-10)
    assert rel_err < 1e-5, f"Factorization error too large: {rel_err:.6e}"

    return X, D


def truncated_svd(G: torch.Tensor, rank: int) -> torch.Tensor:
    """Compute optimal rank-r approximation of G via truncated SVD.

    Args:
        G: Input matrix [d_size, x_size]
        rank: Target rank

    Returns:
        G_r: Rank-r approximation [d_size, x_size]
    """
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)

    effective_rank = min(rank, len(S))
    U_r = U[:, :effective_rank]
    S_r = S[:effective_rank]
    Vh_r = Vh[:effective_rank, :]

    G_r = U_r @ torch.diag(S_r) @ Vh_r
    return G_r


def compute_svd_optimal_error(G: torch.Tensor, rank: int) -> float:
    """Compute ||G - G_r^SVD||_F."""
    G_r = truncated_svd(G, rank)
    return torch.norm(G - G_r, p='fro').item()


# =============================================================================
# Tile Creation (based on toy_example.py)
# =============================================================================

def get_c_device_from_type(device_type: str):
    """Get C tile device based on type string."""
    if device_type == "ideal":
        return IdealizedPresetDevice(dw_min=0.001, dw_min_std=0.0, dw_min_dtod=0.0)
    elif device_type == "pcm":
        return PCMPresetDevice()
    elif device_type == "rram":
        return ReRamESPresetDevice()
    elif device_type == "ecram":
        return LinearStepDevice(
            dw_min=0.001, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=0.0, gamma_down=0.0, mult_noise=False,
            dw_min_dtod=0.05, dw_min_std=0.1
        )
    else:
        return IdealizedPresetDevice()


def create_tile_config(device, noisy: bool = False):
    """Create RPU config for a tile."""
    if noisy:
        io_params = IOParameters(
            out_noise=0.02, inp_noise=0.02, out_bound=1.0, inp_bound=1.0,
            noise_management=NoiseManagementType.ABS_MAX,
            bound_management=BoundManagementType.NONE
        )
    else:
        io_params = IOParameters(
            out_noise=0.0, inp_noise=0.0, out_bound=1.0, inp_bound=1.0,
            noise_management=NoiseManagementType.NONE,
            bound_management=BoundManagementType.NONE
        )

    return SingleRPUConfig(
        device=device,
        forward=io_params,
        backward=io_params,
        update=UpdateParameters(pulse_type=PulseType.STOCHASTIC_COMPRESSED)
    )


def create_tiles(config: BenchmarkConfig, torch_device: torch.device):
    """Create A, B, C tiles using PythonLRTTPreset.sixt1c_ab()."""
    c_device = get_c_device_from_type(config.c_device_type)

    lrtt_device = PythonLRTTPreset.sixt1c_ab(
        rank=config.rank,
        transfer_every=config.transfer_every,
        lora_alpha=config.lora_alpha,
        dt_batch_sec=0.0,
        include_retention=False,
        c_device=c_device,
        reinit_mode=config.reinit_mode,
        decay_factor=config.decay_factor
    )

    a_device = lrtt_device.unit_cell_devices[0]
    b_device = lrtt_device.unit_cell_devices[1]
    c_device_final = lrtt_device.unit_cell_devices[2]

    noisy = config.ab_io_noise == "standard"

    a_config = create_tile_config(a_device, noisy=noisy)
    b_config = create_tile_config(b_device, noisy=noisy)
    c_config = create_tile_config(c_device_final, noisy=noisy)

    tile_a = AnalogTile(out_size=config.d_size, in_size=config.rank, rpu_config=a_config, bias=False)
    tile_b = AnalogTile(out_size=config.rank, in_size=config.x_size, rpu_config=b_config, bias=False)
    tile_c = AnalogTile(out_size=config.d_size, in_size=config.x_size, rpu_config=c_config, bias=False)

    if torch_device.type == 'cuda':
        tile_a = tile_a.cuda(torch_device)
        tile_b = tile_b.cuda(torch_device)
        tile_c = tile_c.cuda(torch_device)

    tile_a.set_learning_rate(config.lr_update)
    tile_b.set_learning_rate(config.lr_update)
    tile_c.set_learning_rate(config.lr_update)

    return tile_a, tile_b, tile_c


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_relative_error(estimate: torch.Tensor, true: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute symmetric relative error: ||est - true|| / max(||est||, ||true||)."""
    diff_norm = torch.norm(estimate - true, p='fro').item()
    est_norm = torch.norm(estimate, p='fro').item()
    true_norm = torch.norm(true, p='fro').item()
    return diff_norm / (max(est_norm, true_norm) + eps)


def compute_frobenius_norm(tensor: torch.Tensor) -> float:
    """Compute Frobenius norm."""
    return torch.norm(tensor, p='fro').item()


@dataclass
class ClaimMetrics:
    """Metrics for all three claims at a single step."""
    step: int

    # === Claim A: Reconstruction Quality ===
    # True weights (from get_weights)
    norm_A_true: float = 0.0
    norm_B_true: float = 0.0
    norm_AB_true: float = 0.0

    # Reconstruction error: ||AB_true - (-G)||_F / ||G||_F
    recon_rel_err_true: float = float('nan')

    # One-hot read estimates
    norm_A_hat: float = float('nan')
    norm_B_hat: float = float('nan')
    norm_AB_hat: float = float('nan')

    # Reconstruction error with one-hot read
    recon_rel_err_hat: float = float('nan')

    # SVD optimal error: ||(-G) - (-G)_r^SVD||_F
    svd_opt_err: float = float('nan')

    # SVD gap ratio: ||AB_true - (-G)||_F / max(svd_opt_err, eps)
    svd_gap_ratio_true: float = float('nan')
    svd_gap_ratio_hat: float = float('nan')

    # Cosine similarity: cos(AB_true, -G)
    cosine_sim_true: float = float('nan')

    # === Claim B: Transfer Equivalence (only at transfer steps) ===
    transfer_occurred: bool = False

    # C weight changes
    norm_C_before: float = float('nan')
    norm_C_after: float = float('nan')
    norm_deltaC: float = float('nan')

    # Expected values
    norm_expected_sgd: float = float('nan')        # ||−η_tx * G_r^SVD||
    norm_expected_from_true: float = float('nan')  # ||η_tx * AB_true||
    norm_expected_from_hat: float = float('nan')   # ||η_tx * AB_hat||

    # Transfer fidelity errors
    err_to_sgd: float = float('nan')       # ||ΔC - (-η_tx * G_r^SVD)|| / max(...)
    err_to_true: float = float('nan')      # ||ΔC - η_tx * AB_true|| / max(...)
    err_to_hat: float = float('nan')       # ||ΔC - η_tx * AB_hat|| / max(...)

    # === Claim C: Error Separation ===
    # Algorithmic approximation error
    alg_err: float = float('nan')  # ||AB_true - (-G_r^SVD)||_F / ||G||_F

    # Read error (A, B separately)
    read_err_A: float = float('nan')
    read_err_B: float = float('nan')
    read_err_AB: float = float('nan')  # ||AB_hat - AB_true|| / max(...)

    # Write/transfer error
    tx_err: float = float('nan')  # ||ΔC - η_tx * AB_hat|| / max(...)

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Main Experiment Class
# =============================================================================

class LRTTClaimsBenchmark:
    """LRTT Claims Benchmark Experiment."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Generate target G
        self.G = generate_target_G(
            mode=config.G_mode,
            d_size=config.d_size,
            x_size=config.x_size,
            seed=config.seed,
            true_rank=config.true_rank,
            decay_rate=config.decay_rate,
            scale=config.G_scale,
            load_path=config.snapshot_path
        ).to(self.device)

        # Update dimensions if loaded from file
        if config.G_mode in ["load_npy", "load_snapshot"]:
            config.d_size, config.x_size = self.G.shape

        # Factorize G to X, D
        self.X, self.D = factorize_G_to_XD(self.G)
        self.X = self.X.to(self.device)
        self.D = self.D.to(self.device)

        # Compute SVD optimal approximation
        self.G_r_svd = truncated_svd(-self.G, config.rank).to(self.device)
        self.svd_opt_err = compute_svd_optimal_error(-self.G, config.rank)

        # Print G spectrum info
        self._print_G_spectrum()

        # Create tiles
        self.tile_a, self.tile_b, self.tile_c = create_tiles(config, self.device)

        # Create controller
        self._create_controller()

        # Metrics storage
        self.metrics_history: List[ClaimMetrics] = []

    def _print_G_spectrum(self):
        """Print spectrum information about G."""
        U, S, Vh = torch.linalg.svd(self.G, full_matrices=False)
        print(f"\n{'='*60}")
        print(f"Target G Spectrum Analysis")
        print(f"{'='*60}")
        print(f"G shape: {self.G.shape}")
        print(f"G norm: {torch.norm(self.G):.4f}")
        print(f"G mode: {self.config.G_mode}")
        print(f"Rank for approximation: {self.config.rank}")
        print(f"\nSingular values (top 10):")
        for i, s in enumerate(S[:10]):
            cumulative = (S[:i+1]**2).sum() / (S**2).sum() * 100
            print(f"  σ_{i}: {s:.4f} (cumulative energy: {cumulative:.1f}%)")
        print(f"\nSVD optimal rank-{self.config.rank} error: {self.svd_opt_err:.6f}")
        print(f"{'='*60}\n")

    def _create_controller(self):
        """Create LRTT controller."""
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
            forward_inject=False,  # FIXED: always False for reconstruction update
            use_onehot=config.use_onehot,
            use_sigma_delta=config.use_sigma_delta,
        )

        self.controller.read_n_avg = config.read_n_avg
        self.controller.transfer_burst_limit = config.transfer_burst_limit

        # Note: Stabilizer/clip functions in the controller have bugs
        # We disable them and rely on orthogonal mode or lower learning rates for stability
        self.controller.recon_use_scalar_stabilizer = False
        self.controller.recon_use_clip_norm = False
        # Use small L2 regularization coefficients to help stabilize
        self.controller.recon_lambda_a = 1e-4
        self.controller.recon_lambda_b = 1e-4

        self.controller.reinit()

    def _read_true_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read true weights from tiles using get_weights()."""
        A_true = self.tile_a.get_weights()[0].to(self.device)
        B_true = self.tile_b.get_weights()[0].to(self.device)
        C_true = self.tile_c.get_weights()[0].to(self.device)
        return A_true, B_true, C_true

    def _read_onehot_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Read A, B via one-hot differential read."""
        if hasattr(self.controller, '_read_ab_onehot_symmetric'):
            A_hat, B_hat = self.controller._read_ab_onehot_symmetric()
            return A_hat, B_hat
        return None, None

    def _compute_claim_a_metrics(self, step: int) -> ClaimMetrics:
        """Compute Claim A metrics (reconstruction quality)."""
        metrics = ClaimMetrics(step=step)

        # Read true weights
        A_true, B_true, C_true = self._read_true_weights()
        AB_true = A_true @ B_true

        # True weight norms
        metrics.norm_A_true = compute_frobenius_norm(A_true)
        metrics.norm_B_true = compute_frobenius_norm(B_true)
        metrics.norm_AB_true = compute_frobenius_norm(AB_true)

        # Reconstruction error (true)
        neg_G = -self.G
        G_norm = compute_frobenius_norm(self.G)
        recon_err_true = compute_frobenius_norm(AB_true - neg_G)

        if G_norm > 1e-10:
            metrics.recon_rel_err_true = recon_err_true / G_norm

        # SVD gap ratio (true)
        metrics.svd_opt_err = self.svd_opt_err
        if self.svd_opt_err > 1e-6:
            metrics.svd_gap_ratio_true = recon_err_true / self.svd_opt_err
        elif G_norm > 1e-10:
            # G is already near rank-r
            metrics.svd_gap_ratio_true = recon_err_true / G_norm

        # Cosine similarity
        ab_flat = AB_true.flatten()
        neg_g_flat = neg_G.flatten()
        ab_norm = torch.norm(ab_flat)
        neg_g_norm = torch.norm(neg_g_flat)
        if ab_norm > 1e-10 and neg_g_norm > 1e-10:
            metrics.cosine_sim_true = (torch.dot(ab_flat, neg_g_flat) / (ab_norm * neg_g_norm)).item()

        # One-hot read estimates
        A_hat, B_hat = self._read_onehot_weights()
        if A_hat is not None and B_hat is not None:
            AB_hat = A_hat @ B_hat

            metrics.norm_A_hat = compute_frobenius_norm(A_hat)
            metrics.norm_B_hat = compute_frobenius_norm(B_hat)
            metrics.norm_AB_hat = compute_frobenius_norm(AB_hat)

            # Reconstruction error (hat)
            recon_err_hat = compute_frobenius_norm(AB_hat - neg_G)
            if G_norm > 1e-10:
                metrics.recon_rel_err_hat = recon_err_hat / G_norm

            # SVD gap ratio (hat)
            if self.svd_opt_err > 1e-6:
                metrics.svd_gap_ratio_hat = recon_err_hat / self.svd_opt_err

            # === Claim C: Error separation ===
            # Algorithmic error: ||AB_true - (-G_r^SVD)||_F / ||G||_F
            alg_err_norm = compute_frobenius_norm(AB_true - self.G_r_svd)
            if G_norm > 1e-10:
                metrics.alg_err = alg_err_norm / G_norm

            # Read errors
            metrics.read_err_A = compute_relative_error(A_hat, A_true)
            metrics.read_err_B = compute_relative_error(B_hat, B_true)
            metrics.read_err_AB = compute_relative_error(AB_hat, AB_true)

        return metrics

    def _compute_claim_b_metrics(
        self,
        metrics: ClaimMetrics,
        C_before: torch.Tensor,
        C_after: torch.Tensor,
        A_true_before: torch.Tensor,
        B_true_before: torch.Tensor,
        A_hat_before: Optional[torch.Tensor],
        B_hat_before: Optional[torch.Tensor]
    ) -> ClaimMetrics:
        """Compute Claim B metrics (transfer equivalence)."""
        metrics.transfer_occurred = True

        # ΔC = C_after - C_before
        deltaC = C_after - C_before

        metrics.norm_C_before = compute_frobenius_norm(C_before)
        metrics.norm_C_after = compute_frobenius_norm(C_after)
        metrics.norm_deltaC = compute_frobenius_norm(deltaC)

        eta_tx = self.controller.transfer_lr

        # Expected SGD: -η_tx * G_r^SVD
        expected_sgd = -eta_tx * self.G_r_svd
        metrics.norm_expected_sgd = compute_frobenius_norm(expected_sgd)

        # Expected from true: η_tx * AB_true
        AB_true_before = A_true_before @ B_true_before
        expected_from_true = eta_tx * AB_true_before
        metrics.norm_expected_from_true = compute_frobenius_norm(expected_from_true)

        # Transfer fidelity errors
        metrics.err_to_sgd = compute_relative_error(deltaC, expected_sgd)
        metrics.err_to_true = compute_relative_error(deltaC, expected_from_true)

        if A_hat_before is not None and B_hat_before is not None:
            AB_hat_before = A_hat_before @ B_hat_before
            expected_from_hat = eta_tx * AB_hat_before
            metrics.norm_expected_from_hat = compute_frobenius_norm(expected_from_hat)

            metrics.err_to_hat = compute_relative_error(deltaC, expected_from_hat)

            # Claim C: Write/transfer error
            metrics.tx_err = metrics.err_to_hat

        return metrics

    def run_claim_a_experiment(self) -> pd.DataFrame:
        """Run Claim A experiment: reconstruction quality over steps."""
        print(f"\n{'='*60}")
        print(f"Running Claim A Experiment: Reconstruction Quality")
        print(f"{'='*60}")
        print(f"Steps: {self.config.steps}, Rank: {self.config.rank}")
        print(f"G mode: {self.config.G_mode}, SVD optimal error: {self.svd_opt_err:.6f}")
        print(f"{'='*60}\n")

        self.metrics_history = []

        # Initial metrics
        metrics = self._compute_claim_a_metrics(step=0)
        self.metrics_history.append(metrics)

        for step in range(1, self.config.steps + 1):
            # AB update (reconstruction mode)
            self.controller.ab_weight_update(self.X, self.D, lr=self.config.lr_update)

            # Log metrics periodically
            if step % self.config.log_every == 0 or step == self.config.steps:
                metrics = self._compute_claim_a_metrics(step=step)
                self.metrics_history.append(metrics)

                if step % 50 == 0 or step == self.config.steps:
                    print(f"Step {step:4d}: recon_rel_err={metrics.recon_rel_err_true:.4f}, "
                          f"svd_gap_ratio={metrics.svd_gap_ratio_true:.4f}, "
                          f"cos_sim={metrics.cosine_sim_true:.4f}")

        return pd.DataFrame([m.to_dict() for m in self.metrics_history])

    def run_claim_b_experiment(self) -> pd.DataFrame:
        """Run Claim B experiment: transfer equivalence after convergence."""
        print(f"\n{'='*60}")
        print(f"Running Claim B Experiment: Transfer Equivalence")
        print(f"{'='*60}")

        # First, run reconstruction updates until convergence
        print("Phase 1: Running reconstruction updates until convergence...")
        for step in range(1, self.config.steps + 1):
            self.controller.ab_weight_update(self.X, self.D, lr=self.config.lr_update)

        # Compute final reconstruction metrics
        final_metrics = self._compute_claim_a_metrics(step=self.config.steps)
        print(f"After {self.config.steps} steps:")
        print(f"  recon_rel_err_true = {final_metrics.recon_rel_err_true:.6f}")
        print(f"  svd_gap_ratio_true = {final_metrics.svd_gap_ratio_true:.6f}")

        # Phase 2: Single transfer
        print("\nPhase 2: Performing transfer...")

        # Read states before transfer
        A_true_before, B_true_before, C_before = self._read_true_weights()
        A_hat_before, B_hat_before = self._read_onehot_weights()
        if A_hat_before is not None:
            A_hat_before = A_hat_before.clone()
        if B_hat_before is not None:
            B_hat_before = B_hat_before.clone()

        # Perform transfer
        self.controller.ab_weight_transfer(
            use_onehot=self.config.use_onehot,
            use_sigma_delta=self.config.use_sigma_delta
        )

        # Read C after transfer
        _, _, C_after = self._read_true_weights()

        # Compute Claim B metrics
        metrics = self._compute_claim_b_metrics(
            metrics=final_metrics,
            C_before=C_before,
            C_after=C_after,
            A_true_before=A_true_before,
            B_true_before=B_true_before,
            A_hat_before=A_hat_before,
            B_hat_before=B_hat_before
        )

        print(f"\nTransfer Results:")
        print(f"  ||ΔC|| = {metrics.norm_deltaC:.6f}")
        print(f"  ||expected_sgd|| = {metrics.norm_expected_sgd:.6f}")
        print(f"  err_to_sgd = {metrics.err_to_sgd:.6f}")
        print(f"  err_to_true = {metrics.err_to_true:.6f}")
        print(f"  err_to_hat = {metrics.err_to_hat:.6f}")

        return pd.DataFrame([metrics.to_dict()])

    def run_claim_c_experiment(self) -> pd.DataFrame:
        """Run Claim C experiment: error separation with optional direct transfer comparison."""
        print(f"\n{'='*60}")
        print(f"Running Claim C Experiment: Error Separation")
        print(f"{'='*60}")

        results = []

        # Run reconstruction updates
        print("Phase 1: Running reconstruction updates...")
        for step in range(1, self.config.steps + 1):
            self.controller.ab_weight_update(self.X, self.D, lr=self.config.lr_update)

        # Compute metrics before transfer
        metrics = self._compute_claim_a_metrics(step=self.config.steps)

        # Read states before transfer
        A_true, B_true, C_before = self._read_true_weights()
        A_hat, B_hat = self._read_onehot_weights()

        # Store A, B for comparison if doing baseline
        if self.config.baseline_direct_transfer:
            A_true_backup = A_true.clone()
            B_true_backup = B_true.clone()
            C_before_backup = C_before.clone()

        # ---- One-hot + ΣΔ transfer (default) ----
        print(f"\nPhase 2a: One-hot + ΣΔ transfer (use_onehot={self.config.use_onehot}, "
              f"use_sigma_delta={self.config.use_sigma_delta})...")

        A_hat_before = A_hat.clone() if A_hat is not None else None
        B_hat_before = B_hat.clone() if B_hat is not None else None

        self.controller.ab_weight_transfer(
            use_onehot=self.config.use_onehot,
            use_sigma_delta=self.config.use_sigma_delta
        )

        _, _, C_after = self._read_true_weights()

        metrics = self._compute_claim_b_metrics(
            metrics=metrics,
            C_before=C_before,
            C_after=C_after,
            A_true_before=A_true,
            B_true_before=B_true,
            A_hat_before=A_hat_before,
            B_hat_before=B_hat_before
        )

        onehot_result = {
            'mode': 'onehot_sd' if self.config.use_sigma_delta else 'onehot_simple',
            **metrics.to_dict()
        }
        results.append(onehot_result)

        print(f"\nOne-hot Results:")
        print(f"  alg_err = {metrics.alg_err:.6f}")
        print(f"  read_err_AB = {metrics.read_err_AB:.6f}")
        print(f"  tx_err = {metrics.tx_err:.6f}")

        # ---- Direct transfer comparison (if enabled) ----
        if self.config.baseline_direct_transfer:
            print("\nPhase 2b: Direct transfer comparison...")

            # Reset tiles to the state BEFORE one-hot transfer
            self.tile_a.set_weights(A_true_backup)
            self.tile_b.set_weights(B_true_backup)
            self.tile_c.set_weights(C_before_backup)

            # Also reset controller state that was modified during one-hot transfer
            self.controller.transfer_counter = self.config.steps  # Reset to trigger transfer

            # Compute metrics BEFORE direct transfer (using restored state)
            metrics_direct = self._compute_claim_a_metrics(step=self.config.steps)

            # Direct transfer (use_onehot=False)
            self.controller.ab_weight_transfer(use_onehot=False, use_sigma_delta=False)

            _, _, C_after_direct = self._read_true_weights()

            # Compute transfer metrics
            metrics_direct = self._compute_claim_b_metrics(
                metrics=metrics_direct,
                C_before=C_before_backup,
                C_after=C_after_direct,
                A_true_before=A_true_backup,
                B_true_before=B_true_backup,
                A_hat_before=A_hat_before,  # Same read as before
                B_hat_before=B_hat_before
            )

            direct_result = {
                'mode': 'direct',
                **metrics_direct.to_dict()
            }
            results.append(direct_result)

            print(f"\nDirect Transfer Results:")
            print(f"  alg_err = {metrics_direct.alg_err:.6f}")
            print(f"  read_err_AB = {metrics_direct.read_err_AB:.6f}")
            print(f"  tx_err = {metrics_direct.tx_err:.6f}")

            print(f"\n=== Comparison: One-hot vs Direct ===")
            print(f"  err_to_sgd: one-hot={metrics.err_to_sgd:.6f}, direct={metrics_direct.err_to_sgd:.6f}")
            print(f"  tx_err: one-hot={metrics.tx_err:.6f}, direct={metrics_direct.tx_err:.6f}")

        return pd.DataFrame(results)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_claim_a_results(df: pd.DataFrame, output_path: Path, config: BenchmarkConfig):
    """Plot Claim A results: reconstruction quality over steps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = df['step'].values

    # Plot 1: Reconstruction relative error
    ax = axes[0, 0]
    ax.semilogy(steps, df['recon_rel_err_true'], label='||AB_true - (-G)|| / ||G||', color='tab:blue')
    if 'recon_rel_err_hat' in df.columns:
        valid = ~df['recon_rel_err_hat'].isna()
        if valid.any():
            ax.semilogy(steps[valid], df['recon_rel_err_hat'][valid], '--',
                       label='||AB_hat - (-G)|| / ||G||', color='tab:orange', alpha=0.7)
    ax.axhline(y=config.G_scale * 0.01, color='gray', linestyle=':', alpha=0.5, label='1% of ||G||')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reconstruction Relative Error')
    ax.set_title('Claim A: Reconstruction Error (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: SVD gap ratio
    ax = axes[0, 1]
    valid = ~df['svd_gap_ratio_true'].isna()
    if valid.any():
        ax.plot(steps[valid], df['svd_gap_ratio_true'][valid], label='||AB - (-G)|| / ||G - G_r^SVD||',
                color='tab:blue')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Optimal (ratio=1)')
    ax.set_xlabel('Step')
    ax.set_ylabel('SVD Gap Ratio')
    ax.set_title('Claim A: SVD Optimality Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 3: Cosine similarity
    ax = axes[1, 0]
    ax.plot(steps, df['cosine_sim_true'], label='cos(AB_true, -G)', color='tab:blue')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect alignment')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('AB Direction Alignment with -G')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Plot 4: Norms
    ax = axes[1, 1]
    ax.plot(steps, df['norm_AB_true'], label='||AB||_F (true)', color='tab:blue')
    if 'norm_AB_hat' in df.columns:
        valid = ~df['norm_AB_hat'].isna()
        if valid.any():
            ax.plot(steps[valid], df['norm_AB_hat'][valid], '--',
                   label='||AB||_F (one-hot)', color='tab:orange', alpha=0.7)
    ax.axhline(y=compute_frobenius_norm(truncated_svd(-torch.zeros(config.d_size, config.x_size), config.rank)),
               color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('AB Product Norm Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Claim A Results: {config.G_mode} (rank={config.rank})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_claim_c_comparison(df: pd.DataFrame, output_path: Path, config: BenchmarkConfig):
    """Plot Claim C error separation comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    modes = df['mode'].values
    x = np.arange(len(modes))
    width = 0.25

    # Error bars
    alg_err = df['alg_err'].values
    read_err = df['read_err_AB'].values
    tx_err = df['tx_err'].values

    ax.bar(x - width, alg_err, width, label='Algorithmic Error', color='tab:blue', alpha=0.8)
    ax.bar(x, read_err, width, label='Read Error (AB)', color='tab:orange', alpha=0.8)
    ax.bar(x + width, tx_err, width, label='Transfer Error', color='tab:green', alpha=0.8)

    ax.set_xlabel('Transfer Mode')
    ax.set_ylabel('Relative Error')
    ax.set_title(f'Claim C: Error Separation\n(G mode: {config.G_mode}, rank={config.rank})')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (a, r, t) in enumerate(zip(alg_err, read_err, tx_err)):
        ax.annotate(f'{a:.3f}', (i - width, a + 0.01), ha='center', fontsize=8)
        ax.annotate(f'{r:.3f}', (i, r + 0.01), ha='center', fontsize=8)
        ax.annotate(f'{t:.3f}', (i + width, t + 0.01), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Sweep Functions
# =============================================================================

def run_rank_sweep(args) -> pd.DataFrame:
    """Run sweep over different ranks."""
    ranks = [int(r) for r in args.ranks.split(',')]
    all_results = []

    for rank in ranks:
        print(f"\n{'='*40}")
        print(f"Rank Sweep: rank={rank}")
        print(f"{'='*40}")

        config = BenchmarkConfig(
            d_size=args.d_size,
            x_size=args.x_size,
            rank=rank,
            steps=args.steps,
            lr_update=args.lr_update,
            G_mode=args.G_mode,
            true_rank=args.true_rank,
            decay_rate=args.decay_rate,
            use_onehot=not args.no_onehot,
            use_sigma_delta=not args.no_sigma_delta,
            reinit_mode=args.reinit_mode,
            seed=args.seed,
            output_dir=args.output_dir,
            log_every=args.log_every,
            baseline_direct_transfer=args.baseline_direct_transfer
        )

        benchmark = LRTTClaimsBenchmark(config)
        df = benchmark.run_claim_a_experiment()

        # Get final metrics
        final = df.iloc[-1].to_dict()
        final['rank'] = rank
        final['svd_opt_err'] = benchmark.svd_opt_err
        all_results.append(final)

    return pd.DataFrame(all_results)


def run_g_mode_sweep(args) -> pd.DataFrame:
    """Run sweep over different G generation modes."""
    modes = ['lowrank', 'decay', 'flat', 'rand_full']
    all_results = []

    for mode in modes:
        print(f"\n{'='*40}")
        print(f"G Mode Sweep: mode={mode}")
        print(f"{'='*40}")

        config = BenchmarkConfig(
            d_size=args.d_size,
            x_size=args.x_size,
            rank=args.rank,
            steps=args.steps,
            lr_update=args.lr_update,
            G_mode=mode,
            true_rank=args.true_rank,
            decay_rate=args.decay_rate,
            use_onehot=not args.no_onehot,
            use_sigma_delta=not args.no_sigma_delta,
            reinit_mode=args.reinit_mode,
            seed=args.seed,
            output_dir=args.output_dir,
            log_every=args.log_every
        )

        benchmark = LRTTClaimsBenchmark(config)
        df = benchmark.run_claim_a_experiment()

        final = df.iloc[-1].to_dict()
        final['G_mode'] = mode
        final['svd_opt_err'] = benchmark.svd_opt_err
        all_results.append(final)

    return pd.DataFrame(all_results)


# =============================================================================
# Result Saving
# =============================================================================

def save_results(
    df: pd.DataFrame,
    config: BenchmarkConfig,
    output_dir: Path,
    experiment_type: str
) -> Path:
    """Save experiment results."""
    run_dir = output_dir / f"{experiment_type}_{config.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {run_dir}")

    # Save config
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Save trace CSV
    df.to_csv(run_dir / "trace.csv", index=False)

    # Save summary
    summary = {
        'experiment_type': experiment_type,
        'final_metrics': df.iloc[-1].to_dict() if len(df) > 0 else {},
        'svd_opt_err': config.true_rank if hasattr(config, 'svd_opt_err') else None,
        'timestamp': datetime.now().isoformat()
    }
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved config.json, trace.csv, summary.json")

    return run_dir


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LRTT Claims Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--run_single', action='store_true',
                           help='Run single experiment with all claims')
    mode_group.add_argument('--claim_a_only', action='store_true',
                           help='Run only Claim A experiment')
    mode_group.add_argument('--claim_b_only', action='store_true',
                           help='Run only Claim B experiment')
    mode_group.add_argument('--claim_c_only', action='store_true',
                           help='Run only Claim C experiment')
    mode_group.add_argument('--sweep_rank', action='store_true',
                           help='Sweep over ranks')
    mode_group.add_argument('--sweep_g_mode', action='store_true',
                           help='Sweep over G generation modes')

    # Matrix dimensions
    parser.add_argument('--d_size', type=int, default=16, help='Output dimension')
    parser.add_argument('--x_size', type=int, default=16, help='Input dimension')

    # LRTT parameters
    parser.add_argument('--rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--ranks', type=str, default='1,2,4,8',
                       help='Comma-separated ranks for sweep')
    parser.add_argument('--reinit_mode', type=str, default='standard',
                       choices=['standard', 'orthogonal', 'decay', 'hybrid'],
                       help='Reinit mode (default: standard)')

    # Transfer mode (DEFAULTS as specified)
    parser.add_argument('--no_onehot', action='store_true',
                       help='Disable one-hot read (default: use_onehot=True)')
    parser.add_argument('--no_sigma_delta', action='store_true',
                       help='Disable sigma-delta (default: use_sigma_delta=True)')

    # G generation
    parser.add_argument('--G_mode', type=str, default='lowrank',
                       choices=['lowrank', 'decay', 'flat', 'rand_full', 'identity',
                               'load_npy', 'load_snapshot'],
                       help='Target G generation mode')
    parser.add_argument('--true_rank', type=int, default=2,
                       help='True rank for lowrank mode')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                       help='Decay rate for decay mode')
    parser.add_argument('--G_scale', type=float, default=1.0,
                       help='Scale factor for G')
    parser.add_argument('--snapshot_path', type=str, default='',
                       help='Path for load_npy/load_snapshot modes')

    # Training
    parser.add_argument('--steps', type=int, default=500, help='Number of update steps')
    parser.add_argument('--lr_update', type=float, default=0.01, help='Update learning rate')
    parser.add_argument('--log_every', type=int, default=10, help='Logging frequency')

    # Comparison
    parser.add_argument('--baseline_direct_transfer', action='store_true',
                       help='Also run direct transfer for comparison (Claim C)')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/lrtt_claims_benchmark',
                       help='Output directory')
    parser.add_argument('--tag', type=str, default='', help='Run tag')

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = BenchmarkConfig(
        d_size=args.d_size,
        x_size=args.x_size,
        rank=args.rank,
        steps=args.steps,
        lr_update=args.lr_update,
        G_mode=args.G_mode,
        true_rank=args.true_rank,
        decay_rate=args.decay_rate,
        G_scale=args.G_scale,
        snapshot_path=args.snapshot_path,
        use_onehot=not args.no_onehot,
        use_sigma_delta=not args.no_sigma_delta,
        reinit_mode=args.reinit_mode,
        seed=args.seed,
        output_dir=args.output_dir,
        log_every=args.log_every,
        baseline_direct_transfer=args.baseline_direct_transfer,
        tag=args.tag
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_single:
        # Run all claims
        benchmark = LRTTClaimsBenchmark(config)

        # Claim A
        df_a = benchmark.run_claim_a_experiment()
        run_dir = save_results(df_a, config, output_dir, "claim_a")
        plot_claim_a_results(df_a, run_dir / "claim_a_plots.png", config)

        # Reset for Claim B/C
        benchmark = LRTTClaimsBenchmark(config)

        # Claim B
        df_b = benchmark.run_claim_b_experiment()
        save_results(df_b, config, output_dir, "claim_b")

        # Reset for Claim C
        benchmark = LRTTClaimsBenchmark(config)

        # Claim C
        df_c = benchmark.run_claim_c_experiment()
        run_dir = save_results(df_c, config, output_dir, "claim_c")
        if len(df_c) > 1:
            plot_claim_c_comparison(df_c, run_dir / "claim_c_comparison.png", config)

    elif args.claim_a_only:
        benchmark = LRTTClaimsBenchmark(config)
        df = benchmark.run_claim_a_experiment()
        run_dir = save_results(df, config, output_dir, "claim_a")
        plot_claim_a_results(df, run_dir / "claim_a_plots.png", config)

    elif args.claim_b_only:
        benchmark = LRTTClaimsBenchmark(config)
        df = benchmark.run_claim_b_experiment()
        save_results(df, config, output_dir, "claim_b")

    elif args.claim_c_only:
        benchmark = LRTTClaimsBenchmark(config)
        df = benchmark.run_claim_c_experiment()
        run_dir = save_results(df, config, output_dir, "claim_c")
        if len(df) > 1:
            plot_claim_c_comparison(df, run_dir / "claim_c_comparison.png", config)

    elif args.sweep_rank:
        df = run_rank_sweep(args)
        df.to_csv(output_dir / "rank_sweep_results.csv", index=False)
        print(f"\nRank Sweep Summary:")
        print(df[['rank', 'recon_rel_err_true', 'svd_gap_ratio_true', 'svd_opt_err']])

    elif args.sweep_g_mode:
        df = run_g_mode_sweep(args)
        df.to_csv(output_dir / "g_mode_sweep_results.csv", index=False)
        print(f"\nG Mode Sweep Summary:")
        print(df[['G_mode', 'recon_rel_err_true', 'svd_gap_ratio_true', 'svd_opt_err']])

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""CIFAR-10 ResNet18 Digital LRTT training with 2-stage warm-start.

Pure PyTorch implementation without aihwkit.
This is the digital baseline for comparing with analog LRTT.

Two-stage training approach:
- Stage 1 (Warmstart): Train standard ResNet (full weights) for initial convergence
- Stage 2 (LRTT): Transfer weights to LRTT model, train with A/B updates only

Structure:
- conv1: Standard Conv2d (no LRTT)
- layer1-4: LRTT Conv2d (configurable per sublayer)
- fc: Standard Linear (no LRTT)

Key features:
- No analog noise or device non-idealities
- Exact matrix operations for A, B, C
- Same LRTT algorithm as analog version but fully digital
"""
# pylint: disable=invalid-name

import os
import math
from time import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

# Logging
import wandb

# ==============================================================================
# Configuration
# ==============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_DIGITAL_LRTT")
os.makedirs(RESULTS, exist_ok=True)

# Training - Stage 1 (Warmstart with standard ResNet)
N_EPOCHS_STAGE1 = 0            # Warmstart epochs (set to 0 to skip)
LEARNING_RATE_STAGE1 = 0.1
WARMUP_RATIO_STAGE1 = 0.0
LR_SCHEDULE_STAGE1 = "constant"
LR_MILESTONES_STAGE1 = [30, 40]
LR_GAMMA_STAGE1 = 0.1

# Training - Stage 2 (LRTT)
N_EPOCHS_STAGE2 = 300           # LRTT training epochs
LEARNING_RATE_STAGE2 = 0.1
WARMUP_RATIO_STAGE2 = 0.0
LR_SCHEDULE_STAGE2 = "cosine"
LR_MILESTONES_STAGE2 = [150, 200]
LR_GAMMA_STAGE2 = 0.1

# Common
SEED = 1
BATCH_SIZE = 8
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NESTEROV = True
N_CLASSES = 10
NUM_WORKERS = 4

# ==============================================================================
# LRTT Configuration
# ==============================================================================

# --- Per-layer LRTT configuration ---
# Format: "layer.block.sublayer": {"use_lrtt": bool, "rank": int}
# sublayer can be: "conv1", "conv2", "downsample"
# If a layer is not specified, it uses LRTT_RANK_DEFAULT.

LAYER_CONFIG = {
    # =========================================================================
    # layer1 (64 channels): 64->64, 3x3 conv
    # C params per layer: 64*64*3*3 = 36,864
    # =========================================================================
    "layer1.0.conv1": {"use_lrtt": True, "rank": 16},    # A+B: 64*16 + 16*576 = 10,240
    "layer1.0.conv2": {"use_lrtt": True, "rank": 16},    # A+B: 64*16 + 16*576 = 10,240
    "layer1.1.conv1": {"use_lrtt": True, "rank": 16},    # A+B: 64*16 + 16*576 = 10,240
    "layer1.1.conv2": {"use_lrtt": True, "rank": 16},    # A+B: 64*16 + 16*576 = 10,240

    # =========================================================================
    # layer2 (128 channels): 64->128 (first), 128->128 (rest), 3x3 conv
    # =========================================================================
    "layer2.0.conv1": {"use_lrtt": True, "rank": 32},    # 64->128: C=73,728, A+B: 128*32 + 32*576 = 22,528
    "layer2.0.conv2": {"use_lrtt": True, "rank": 32},    # 128->128: C=147,456, A+B: 128*32 + 32*1152 = 40,960
    "layer2.0.downsample": {"use_lrtt": False},  # 1x1: 64->128, C=8,192
    "layer2.1.conv1": {"use_lrtt": True, "rank": 32},    # 128->128: C=147,456, A+B: 128*32 + 32*1152 = 40,960
    "layer2.1.conv2": {"use_lrtt": True, "rank": 32},    # 128->128: C=147,456, A+B: 128*32 + 32*1152 = 40,960

    # =========================================================================
    # layer3 (256 channels): 128->256 (first), 256->256 (rest), 3x3 conv
    # =========================================================================
    "layer3.0.conv1": {"use_lrtt": True, "rank": 64},    # 128->256: C=294,912, A+B: 256*64 + 64*1152 = 90,112
    "layer3.0.conv2": {"use_lrtt": True, "rank": 64},    # 256->256: C=589,824, A+B: 256*64 + 64*2304 = 163,840
    "layer3.0.downsample": {"use_lrtt": False},  # 1x1: 128->256, C=32,768
    "layer3.1.conv1": {"use_lrtt": True, "rank": 64},    # 256->256: C=589,824, A+B: 256*64 + 64*2304 = 163,840
    "layer3.1.conv2": {"use_lrtt": True, "rank": 64},    # 256->256: C=589,824, A+B: 256*64 + 64*2304 = 163,840

    # =========================================================================
    # layer4 (512 channels): 256->512 (first), 512->512 (rest), 3x3 conv
    # =========================================================================
    "layer4.0.conv1": {"use_lrtt": True, "rank": 128},   # 256->512: C=1,179,648, A+B: 512*128 + 128*2304 = 360,448
    "layer4.0.conv2": {"use_lrtt": True, "rank": 128},   # 512->512: C=2,359,296, A+B: 512*128 + 128*4608 = 655,360
    "layer4.0.downsample": {"use_lrtt": False},  # 1x1: 256->512, C=131,072
    "layer4.1.conv1": {"use_lrtt": True, "rank": 128},   # 512->512: C=2,359,296, A+B: 512*128 + 128*4608 = 655,360
    "layer4.1.conv2": {"use_lrtt": True, "rank": 128},   # 512->512: C=2,359,296, A+B: 512*128 + 128*4608 = 655,360
}

# --- Default LRTT parameters (used when not specified in LAYER_CONFIG) ---
LRTT_RANK_DEFAULT = 16          # Default LoRA rank
TRANSFER_EVERY = 256            # A @ B -> C transfer period (iteration units)
LORA_ALPHA = 2.0                # LoRA scale factor
TRANSFER_LR = LORA_ALPHA        # Transfer learning rate

# Forward mode
FORWARD_INJECT = False          # True: y = Cx + alpha * A(Bx)
                                # False: y = Cx (A@B accumulated via transfers)

# Reinit mode after transfer
REINIT_MODE = "standard"        # "standard": A=0, B=Kaiming
                                # "decay": A *= decay_factor, B *= decay_factor
                                # "hybrid": A=0, B *= decay_factor
                                # "orthogonal": A=0, B=orthogonal (fixed, B frozen)
REINIT_DECAY_FACTOR = 0.9       # decay factor for "decay" mode
REINIT_GAIN = 0.1               # Kaiming gain for B initialization

# Gradient magnitude correction
CORRECT_GRADIENT_MAGNITUDES = True  # lr *= 1/sqrt(rank)

# Weight clamping (to match analog behavior)
USE_WEIGHT_CLAMP = False        # Clamp weights to [-w_max, w_max]
W_MAX = 1.0                     # Max weight value

# Transfer counter mode (match aihwkit default)
UNITS_IN_MBATCH = False         # True: count by batch size, False: count by 1 (iteration)

# Transfer analysis settings
TRANSFER_ANALYSIS = True        # Enable detailed transfer analysis
TRANSFER_ANALYSIS_CELLS = 3     # Number of cells to track per matrix
TRANSFER_ANALYSIS_LAYERS = ["layer1.0.conv1", "layer4.1.conv2"]  # Layers to analyze (empty = all)
TRANSFER_ANALYSIS_WANDB = True  # Log transfer analysis to wandb as plots


def get_layer_config(layer_name: str) -> dict:
    """Get LRTT config for a specific layer.

    Args:
        layer_name: e.g., "layer1.0.conv1", "layer2.0.downsample"

    Returns:
        dict with 'use_lrtt' and 'rank' keys
    """
    config = {"use_lrtt": True, "rank": LRTT_RANK_DEFAULT}

    # Check from most specific to least specific
    # e.g., "layer1.0.conv1" -> "layer1.0" -> "layer1"
    parts = layer_name.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in LAYER_CONFIG:
            layer_cfg = LAYER_CONFIG[prefix]
            if "use_lrtt" in layer_cfg:
                config["use_lrtt"] = layer_cfg["use_lrtt"]
            if "rank" in layer_cfg:
                config["rank"] = layer_cfg["rank"]
            break

    return config


# ==============================================================================
# Digital LRTT Conv2d Layer
# ==============================================================================

class DigitalLRTTConv2d(nn.Module):
    """Digital LRTT Conv2d layer.

    Implements LRTT for convolution:
    - C: [out_channels, in_channels, kH, kW] - main weight (frozen for SGD if forward_inject=False)
    - A: [out_channels, rank] - left factor
    - B: [rank, in_channels * kH * kW] - right factor

    Forward:
    - If forward_inject=True: y = conv(x, C) + alpha * conv(x, A@B)
    - If forward_inject=False: y = conv(x, C)

    Update (called manually, not via autograd):
    - XB = B @ x_col  (projection)
    - DA = A^T @ d_col  (projection)
    - A -= lr_eff * d_col^T @ XB
    - B -= lr_eff * DA^T @ x_col (if not frozen)

    Transfer (every transfer_every steps):
    - C += transfer_lr * A @ B
    - Reinit A, B
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        rank: int = 16,
        lora_alpha: float = 1.0,
        transfer_every: int = 1000,
        transfer_lr: float = 1.0,
        forward_inject: bool = False,
        reinit_mode: str = "standard",
        reinit_decay_factor: float = 0.9,
        reinit_gain: float = 0.1,
        correct_gradient_magnitudes: bool = True,
        use_weight_clamp: bool = True,
        w_max: float = 1.0,
        units_in_mbatch: bool = False,
        bias: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.transfer_every = transfer_every
        self.transfer_lr = transfer_lr
        self.forward_inject = forward_inject
        self.reinit_mode = reinit_mode
        self.reinit_decay_factor = reinit_decay_factor
        self.reinit_gain = reinit_gain
        self.correct_gradient_magnitudes = correct_gradient_magnitudes
        self.use_weight_clamp = use_weight_clamp
        self.w_max = w_max
        self.units_in_mbatch = units_in_mbatch

        # Current learning rate (set externally)
        self.current_lr = 0.1

        # Weight dimensions for A, B
        self.weight_rows = out_channels
        self.weight_cols = in_channels * kernel_size * kernel_size

        # Main weight matrix C (as buffer - not updated by optimizer)
        # C is only updated via transfer
        self.register_buffer('C', torch.empty(out_channels, in_channels, kernel_size, kernel_size))

        # LoRA matrices A and B (buffers - manually updated)
        self.register_buffer('A', torch.zeros(self.weight_rows, rank))
        self.register_buffer('B', torch.zeros(rank, self.weight_cols))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('bias', None)

        # Counters
        self.transfer_counter = 0
        self.num_transfers = 0
        self.num_a_updates = 0
        self.num_b_updates = 0

        # B frozen flag (for orthogonal mode)
        self._b_frozen = False

        # Storage for backward
        self._last_input = None
        self._last_output = None

        # Transfer analysis
        self.layer_name = ""  # Set externally for logging
        self.transfer_history = []  # List of analysis dicts
        self._tracked_cells_A = None  # Fixed cell indices for A
        self._tracked_cells_B = None  # Fixed cell indices for B
        self._tracked_cells_C = None  # Fixed cell indices for C

        self._init_weights()

    def _init_weights(self):
        """Initialize C with Kaiming, A=0, B based on reinit_mode."""
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
        self.A.zero_()

        if self.reinit_mode == "orthogonal":
            self._init_orthogonal_b()
            self._b_frozen = True
        else:
            nn.init.kaiming_normal_(self.B, a=0, mode='fan_out')
            self.B.mul_(self.reinit_gain)

    def _init_orthogonal_b(self):
        """Initialize B with orthogonal rows (B @ B^T ≈ I)."""
        with torch.no_grad():
            if self.weight_cols >= self.rank:
                temp = torch.randn(self.rank, self.weight_cols, device=self.B.device)
                q, _ = torch.linalg.qr(temp.t())
                self.B.copy_(q.t() * self.reinit_gain)
            else:
                nn.init.kaiming_normal_(self.B, a=0, mode='fan_out')
                self.B.mul_(self.reinit_gain)

    def reinit_ab(self):
        """Reinitialize A and B after transfer."""
        with torch.no_grad():
            if self.reinit_mode == "standard":
                self.A.zero_()
                nn.init.kaiming_normal_(self.B, a=0, mode='fan_out')
                self.B.mul_(self.reinit_gain)
            elif self.reinit_mode == "decay":
                self.A.mul_(self.reinit_decay_factor)
                self.B.mul_(self.reinit_decay_factor)
            elif self.reinit_mode == "hybrid":
                self.A.zero_()
                self.B.mul_(self.reinit_decay_factor)
            elif self.reinit_mode == "orthogonal":
                self.A.zero_()
                # B stays frozen

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Store input for LRTT update
        if self.training:
            self._last_input = x.detach()

        # Standard convolution with C
        y = F.conv2d(x, self.C, self.bias, self.stride, self.padding)

        # Add A @ B contribution if forward_inject
        if self.forward_inject:
            AB = self.A @ self.B
            AB_weight = AB.view(self.out_channels, self.in_channels,
                               self.kernel_size, self.kernel_size)
            y_ab = F.conv2d(x, AB_weight, None, self.stride, self.padding)
            y = y + self.lora_alpha * y_ab

        return y

    def lrtt_update(self, d: Tensor):
        """LRTT update for A and B.

        Args:
            d: Gradient w.r.t. output, [batch, out_channels, H_out, W_out]

        This implements the LoRA chain rule update:
            A -= lr_eff * D^T @ (B @ X)
            B -= lr_eff * (A^T @ D)^T @ X
        """
        if self._last_input is None:
            return

        x = self._last_input
        batch_size = x.size(0)

        # Effective learning rate
        lr_eff = self.current_lr * self.lora_alpha
        if self.correct_gradient_magnitudes:
            lr_eff /= math.sqrt(self.rank)

        with torch.no_grad():
            # im2col: unfold input
            # x: [N, C_in, H, W] -> x_unfold: [N, C_in*k*k, L]
            x_unfold = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
            N, CKK, L = x_unfold.shape

            # x_col: [N*L, C_in*k*k]
            x_col = x_unfold.permute(0, 2, 1).reshape(N * L, CKK)

            # d: [N, C_out, H_out, W_out] -> d_col: [N*L, C_out]
            d_col = d.permute(0, 2, 3, 1).reshape(N * L, self.out_channels)

            # 1) Projections (matches aihwkit tile.forward/backward)
            # XB = B @ x_col^T -> [rank, N*L] -> transpose -> [N*L, rank]
            XB = x_col @ self.B.t()  # [N*L, rank]

            # DA = A^T @ d_col^T -> [rank, N*L] -> transpose -> [N*L, rank]
            DA = d_col @ self.A  # [N*L, rank]

            # 2) A update: A -= lr_eff * d_col^T @ XB
            # This is: A -= lr_eff * (d^T @ (B@x))
            grad_A = d_col.t() @ XB  # [out_channels, rank]
            self.A.sub_(lr_eff * grad_A)
            self.num_a_updates += 1

            # 3) B update: B -= lr_eff * DA^T @ x_col (if not frozen)
            # This is: B -= lr_eff * (A^T @ d)^T @ x
            if not self._b_frozen:
                grad_B = DA.t() @ x_col  # [rank, C_in*k*k]
                self.B.sub_(lr_eff * grad_B)
                self.num_b_updates += 1

            # Clamp A, B if needed
            if self.use_weight_clamp:
                self.A.clamp_(-self.w_max, self.w_max)
                self.B.clamp_(-self.w_max, self.w_max)

        # Update transfer counter
        if self.units_in_mbatch:
            self.transfer_counter += batch_size
        else:
            self.transfer_counter += 1

        # Clear stored input
        self._last_input = None

    def check_transfer(self, analyze: bool = False, n_cells: int = 5, log_wandb: bool = False):
        """Check and perform transfer if needed."""
        if self.transfer_counter >= self.transfer_every:
            self.transfer(analyze=analyze, n_cells=n_cells, log_wandb=log_wandb)

    def transfer(self, analyze: bool = False, n_cells: int = 5, log_wandb: bool = False):
        """Transfer A @ B to C.

        Args:
            analyze: If True, collect detailed statistics before transfer
            n_cells: Number of cells to track
            log_wandb: If True, log analysis to wandb in real-time
        """
        with torch.no_grad():
            # Analyze before transfer if enabled
            if analyze:
                self._analyze_transfer(n_cells)
                # Log to wandb if enabled
                if log_wandb and self.transfer_history:
                    self.log_to_wandb_realtime(self.transfer_history[-1])

            AB = self.A @ self.B
            AB_weight = AB.view(self.out_channels, self.in_channels,
                               self.kernel_size, self.kernel_size)

            self.C.add_(self.transfer_lr * AB_weight)

            if self.use_weight_clamp:
                self.C.clamp_(-self.w_max, self.w_max)

            self.reinit_ab()
            self.num_transfers += 1
            self.transfer_counter = 0

    def _analyze_transfer(self, n_cells: int = 5):
        """Analyze A, B, C matrices before transfer.

        Collects:
        - Frobenius norm of A, B, AB, C
        - Cosine similarity with previous A, B, C (if available)
        - Individual cell values for tracking
        """
        A_flat = self.A.flatten()
        B_flat = self.B.flatten()
        C_flat = self.C.flatten()
        AB = self.A @ self.B
        AB_flat = AB.flatten()

        # Initialize tracked cell indices (fixed across transfers)
        if self._tracked_cells_A is None:
            n_A = min(n_cells, A_flat.numel())
            n_B = min(n_cells, B_flat.numel())
            n_C = min(n_cells, C_flat.numel())
            # Use evenly spaced indices
            self._tracked_cells_A = torch.linspace(0, A_flat.numel() - 1, n_A).long()
            self._tracked_cells_B = torch.linspace(0, B_flat.numel() - 1, n_B).long()
            self._tracked_cells_C = torch.linspace(0, C_flat.numel() - 1, n_C).long()

        # Compute norms
        A_norm = torch.norm(self.A, p='fro').item()
        B_norm = torch.norm(self.B, p='fro').item()
        AB_norm = torch.norm(AB, p='fro').item()
        C_norm = torch.norm(self.C, p='fro').item()

        # Compute per-row/column norms
        A_row_norms = torch.norm(self.A, dim=1)  # [out_channels]
        B_col_norms = torch.norm(self.B, dim=0)  # [weight_cols]

        # Get tracked cell values
        A_cells = A_flat[self._tracked_cells_A].cpu().tolist()
        B_cells = B_flat[self._tracked_cells_B].cpu().tolist()
        C_cells = C_flat[self._tracked_cells_C].cpu().tolist()

        # Compute cosine similarity with previous if available
        cos_sim_A = None
        cos_sim_B = None
        cos_sim_C = None
        if len(self.transfer_history) > 0:
            prev = self.transfer_history[-1]
            prev_A = torch.tensor(prev['A_cells'], device=self.A.device)
            prev_B = torch.tensor(prev['B_cells'], device=self.B.device)
            prev_C = torch.tensor(prev['C_cells'], device=self.C.device)
            curr_A = A_flat[self._tracked_cells_A]
            curr_B = B_flat[self._tracked_cells_B]
            curr_C = C_flat[self._tracked_cells_C]

            # Cosine similarity for A
            if torch.norm(prev_A) > 1e-8 and torch.norm(curr_A) > 1e-8:
                cos_sim_A = (torch.dot(prev_A, curr_A) /
                            (torch.norm(prev_A) * torch.norm(curr_A))).item()
            # Cosine similarity for B
            if torch.norm(prev_B) > 1e-8 and torch.norm(curr_B) > 1e-8:
                cos_sim_B = (torch.dot(prev_B, curr_B) /
                            (torch.norm(prev_B) * torch.norm(curr_B))).item()
            # Cosine similarity for C (measures how much C changed)
            if torch.norm(prev_C) > 1e-8 and torch.norm(curr_C) > 1e-8:
                cos_sim_C = (torch.dot(prev_C, curr_C) /
                            (torch.norm(prev_C) * torch.norm(curr_C))).item()

        analysis = {
            'transfer_idx': self.num_transfers,
            'A_norm': A_norm,
            'B_norm': B_norm,
            'AB_norm': AB_norm,
            'C_norm': C_norm,
            'A_row_norm_mean': A_row_norms.mean().item(),
            'A_row_norm_std': A_row_norms.std().item(),
            'B_col_norm_mean': B_col_norms.mean().item(),
            'B_col_norm_std': B_col_norms.std().item(),
            'A_mean': self.A.mean().item(),
            'A_std': self.A.std().item(),
            'B_mean': self.B.mean().item(),
            'B_std': self.B.std().item(),
            'AB_mean': AB.mean().item(),
            'AB_std': AB.std().item(),
            'C_mean': self.C.mean().item(),
            'C_std': self.C.std().item(),
            'cos_sim_A': cos_sim_A,
            'cos_sim_B': cos_sim_B,
            'cos_sim_C': cos_sim_C,
            'A_cells': A_cells,
            'B_cells': B_cells,
            'C_cells': C_cells,
        }

        self.transfer_history.append(analysis)

    def get_transfer_analysis_summary(self) -> str:
        """Get a formatted summary of transfer analysis."""
        if not self.transfer_history:
            return "No transfer history available."

        lines = [f"\n{'='*80}"]
        lines.append(f"Transfer Analysis: {self.layer_name} (rank={self.rank})")
        lines.append(f"{'='*80}")
        lines.append(f"{'Idx':>4} | {'||A||':>7} {'||B||':>7} {'||AB||':>8} {'||C||':>8} | "
                    f"{'cos(A)':>7} {'cos(B)':>7} {'cos(C)':>7}")
        lines.append("-" * 80)

        for h in self.transfer_history[-10:]:  # Show last 10
            cos_a = f"{h['cos_sim_A']:.4f}" if h['cos_sim_A'] is not None else "   N/A"
            cos_b = f"{h['cos_sim_B']:.4f}" if h['cos_sim_B'] is not None else "   N/A"
            cos_c = f"{h['cos_sim_C']:.4f}" if h['cos_sim_C'] is not None else "   N/A"
            lines.append(
                f"{h['transfer_idx']:>4} | {h['A_norm']:>7.4f} {h['B_norm']:>7.4f} "
                f"{h['AB_norm']:>8.5f} {h['C_norm']:>8.4f} | "
                f"{cos_a:>7} {cos_b:>7} {cos_c:>7}"
            )

        # Cell tracking
        lines.append("-" * 80)
        lines.append("Tracked cells (last transfer):")
        last = self.transfer_history[-1]
        lines.append(f"  A cells: {[f'{v:.4f}' for v in last['A_cells']]}")
        lines.append(f"  B cells: {[f'{v:.4f}' for v in last['B_cells']]}")
        lines.append(f"  C cells: {[f'{v:.4f}' for v in last['C_cells']]}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def log_to_wandb_realtime(self, analysis: dict):
        """Log single transfer analysis to wandb as scalar values (for real-time plotting).

        Uses commit=False to avoid incrementing wandb's global step.
        The step will be incremented by the main training loop's wandb.log() call.
        """
        layer_name = self.layer_name.replace(".", "_")
        transfer_idx = analysis['transfer_idx']

        log_dict = {
            # Norms
            f"transfer/{layer_name}/norm_A": analysis['A_norm'],
            f"transfer/{layer_name}/norm_B": analysis['B_norm'],
            f"transfer/{layer_name}/norm_AB": analysis['AB_norm'],
            f"transfer/{layer_name}/norm_C": analysis['C_norm'],
            # Std
            f"transfer/{layer_name}/std_A": analysis['A_std'],
            f"transfer/{layer_name}/std_B": analysis['B_std'],
            f"transfer/{layer_name}/std_C": analysis['C_std'],
            f"transfer/{layer_name}/std_AB": analysis['AB_std'],
            # Mean
            f"transfer/{layer_name}/mean_A": analysis['A_mean'],
            f"transfer/{layer_name}/mean_B": analysis['B_mean'],
            f"transfer/{layer_name}/mean_C": analysis['C_mean'],
            f"transfer/{layer_name}/mean_AB": analysis['AB_mean'],
            # Transfer index for x-axis
            f"transfer/{layer_name}/transfer_idx": transfer_idx,
        }

        # Cosine similarities (may be None for first transfer)
        if analysis['cos_sim_A'] is not None:
            log_dict[f"transfer/{layer_name}/cos_A"] = analysis['cos_sim_A']
        if analysis['cos_sim_B'] is not None:
            log_dict[f"transfer/{layer_name}/cos_B"] = analysis['cos_sim_B']
        if analysis['cos_sim_C'] is not None:
            log_dict[f"transfer/{layer_name}/cos_C"] = analysis['cos_sim_C']

        # Cell values
        for i, val in enumerate(analysis['A_cells']):
            log_dict[f"transfer/{layer_name}/A_cell_{i}"] = val
        for i, val in enumerate(analysis['B_cells']):
            log_dict[f"transfer/{layer_name}/B_cell_{i}"] = val
        for i, val in enumerate(analysis['C_cells']):
            log_dict[f"transfer/{layer_name}/C_cell_{i}"] = val

        # commit=False: Don't increment global step, let main training loop handle it
        wandb.log(log_dict, commit=False)

    def get_effective_weight(self) -> Tensor:
        """Get effective weight: C + alpha * A @ B."""
        AB = self.A @ self.B
        AB_weight = AB.view(self.out_channels, self.in_channels,
                           self.kernel_size, self.kernel_size)
        return self.C + self.lora_alpha * AB_weight


# ==============================================================================
# Standard ResNet18 (for Stage 1 warmstart)
# ==============================================================================

class BasicBlockStandard(nn.Module):
    """Standard ResNet BasicBlock (no LRTT)."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                        stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetStandard(nn.Module):
    """Standard ResNet18 for CIFAR-10 (no LRTT)."""

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ==============================================================================
# ResNet18 with Digital LRTT (for Stage 2)
# ==============================================================================

class BasicBlockLRTT(nn.Module):
    """ResNet BasicBlock with Digital LRTT Conv2d."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_name="", base_lrtt_config=None):
        """
        Args:
            in_planes: Input channels
            planes: Output channels
            stride: Stride for first conv
            block_name: e.g., "layer1.0" for config lookup
            base_lrtt_config: Base LRTT config (without rank, use_lrtt)
        """
        super().__init__()

        if base_lrtt_config is None:
            base_lrtt_config = {}

        # Get per-sublayer config
        conv1_cfg = get_layer_config(f"{block_name}.conv1")
        conv2_cfg = get_layer_config(f"{block_name}.conv2")
        ds_cfg = get_layer_config(f"{block_name}.downsample")

        # conv1
        if conv1_cfg["use_lrtt"]:
            self.conv1 = DigitalLRTTConv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1,
                rank=conv1_cfg["rank"], **base_lrtt_config
            )
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)

        # conv2
        if conv2_cfg["use_lrtt"]:
            self.conv2 = DigitalLRTTConv2d(
                planes, planes, kernel_size=3, stride=1, padding=1,
                rank=conv2_cfg["rank"], **base_lrtt_config
            )
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                  padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut/downsample
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if ds_cfg["use_lrtt"]:
                self.shortcut = nn.Sequential(
                    DigitalLRTTConv2d(
                        in_planes, self.expansion * planes, kernel_size=1,
                        stride=stride, padding=0, rank=ds_cfg["rank"], **base_lrtt_config
                    ),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                            stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetLRTT(nn.Module):
    """ResNet18 for CIFAR-10 with Digital LRTT.

    Structure:
    - conv1: Standard Conv2d (no LRTT)
    - layer1-4: LRTT Conv2d (configurable per sublayer)
    - fc: Standard Linear (no LRTT)
    """

    def __init__(self, block, num_blocks, num_classes=10, base_lrtt_config=None):
        super().__init__()
        self.in_planes = 64
        self.base_lrtt_config = base_lrtt_config or {}

        # First conv layer (digital, no LRTT)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet layers with LRTT
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, layer_name="layer1")
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, layer_name="layer2")
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, layer_name="layer3")
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, layer_name="layer4")

        # Final FC layer (digital, no LRTT)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, layer_name):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_idx, stride in enumerate(strides):
            block_name = f"{layer_name}.{block_idx}"
            layers.append(block(self.in_planes, planes, stride,
                               block_name=block_name, base_lrtt_config=self.base_lrtt_config))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_lrtt_layers(self):
        """Get all LRTT layers."""
        layers = []
        for module in self.modules():
            if isinstance(module, DigitalLRTTConv2d):
                layers.append(module)
        return layers

    def set_lr(self, lr: float):
        """Set learning rate for all LRTT layers."""
        for layer in self.get_lrtt_layers():
            layer.current_lr = lr

    def check_transfers(self, analyze: bool = False, analyze_layers: list = None,
                        n_cells: int = 5, log_wandb: bool = False):
        """Check and perform transfers for all LRTT layers.

        Args:
            analyze: If True, collect transfer statistics
            analyze_layers: List of layer names to analyze (None = all)
            n_cells: Number of cells to track per matrix
            log_wandb: If True, log analysis to wandb in real-time
        """
        for name, module in self.named_modules():
            if isinstance(module, DigitalLRTTConv2d):
                # Decide whether to analyze this layer
                do_analyze = analyze
                do_log_wandb = log_wandb
                if analyze_layers and name not in analyze_layers:
                    do_analyze = False
                    do_log_wandb = False
                module.check_transfer(analyze=do_analyze, n_cells=n_cells, log_wandb=do_log_wandb)

    def get_transfer_analysis(self, layers: list = None) -> str:
        """Get transfer analysis summary for specified layers."""
        lines = []
        for name, module in self.named_modules():
            if isinstance(module, DigitalLRTTConv2d):
                if layers is None or name in layers:
                    if module.transfer_history:
                        lines.append(module.get_transfer_analysis_summary())
        return "\n".join(lines) if lines else "No transfer analysis available."

def build_standard_model():
    """Build standard ResNet18 (for Stage 1)."""
    model = ResNetStandard(BasicBlockStandard, [2, 2, 2, 2], num_classes=N_CLASSES)

    if USE_CUDA:
        model = model.to(DEVICE)

    print("\n" + "="*70)
    print("Stage 1: Standard ResNet18 Model (Warmstart)")
    print("="*70)
    print(f"  - All layers: Standard Conv2d/Linear")
    print(f"  - Epochs: {N_EPOCHS_STAGE1}")
    print(f"  - Learning rate: {LEARNING_RATE_STAGE1} ({LR_SCHEDULE_STAGE1})")
    print("="*70 + "\n")

    return model


def build_lrtt_model():
    """Build ResNet18 with Digital LRTT (for Stage 2)."""
    # Base config (rank is set per-layer via LAYER_CONFIG)
    base_lrtt_config = {
        'lora_alpha': LORA_ALPHA,
        'transfer_every': TRANSFER_EVERY,
        'transfer_lr': TRANSFER_LR,
        'forward_inject': FORWARD_INJECT,
        'reinit_mode': REINIT_MODE,
        'reinit_decay_factor': REINIT_DECAY_FACTOR,
        'reinit_gain': REINIT_GAIN,
        'correct_gradient_magnitudes': CORRECT_GRADIENT_MAGNITUDES,
        'use_weight_clamp': USE_WEIGHT_CLAMP,
        'w_max': W_MAX,
        'units_in_mbatch': UNITS_IN_MBATCH,
        'bias': False,
    }

    model = ResNetLRTT(BasicBlockLRTT, [2, 2, 2, 2], num_classes=N_CLASSES, base_lrtt_config=base_lrtt_config)

    if USE_CUDA:
        model = model.to(DEVICE)

    # Set layer names for analysis
    for name, module in model.named_modules():
        if isinstance(module, DigitalLRTTConv2d):
            module.layer_name = name

    # Count and summarize LRTT layers
    lrtt_layers = model.get_lrtt_layers()
    num_lrtt = len(lrtt_layers)
    num_standard = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d) and not isinstance(m, DigitalLRTTConv2d))

    print("\n" + "="*70)
    print("Stage 2: Digital LRTT Model")
    print("="*70)
    print(f"  - conv1: Standard Conv2d (no LRTT)")
    print(f"  - fc: Standard Linear (no LRTT)")
    print(f"  - Total LRTT layers: {num_lrtt}")
    print(f"  - Total Standard Conv layers: {num_standard}")
    print(f"  - Forward inject: {FORWARD_INJECT}")
    print(f"  - Transfer every: {TRANSFER_EVERY} steps")
    print(f"  - LoRA alpha: {LORA_ALPHA}")
    print(f"  - Reinit mode: {REINIT_MODE}")
    print(f"  - Weight clamp: {USE_WEIGHT_CLAMP} (w_max={W_MAX})")
    print(f"  - Epochs: {N_EPOCHS_STAGE2}")
    print(f"  - Learning rate: {LEARNING_RATE_STAGE2} ({LR_SCHEDULE_STAGE2})")
    print("-"*70)
    print("  Per-layer configuration:")
    for name, module in model.named_modules():
        if isinstance(module, DigitalLRTTConv2d):
            print(f"    {name}: LRTT (rank={module.rank})")
    print("="*70 + "\n")

    # Print parameter summary
    total_c_params = 0
    total_ab_params = 0
    for layer in lrtt_layers:
        c_params = layer.out_channels * layer.in_channels * layer.kernel_size * layer.kernel_size
        ab_params = layer.out_channels * layer.rank + layer.rank * layer.weight_cols
        total_c_params += c_params
        total_ab_params += ab_params

    if total_c_params > 0:
        print(f"Parameter Summary (LRTT layers only):")
        print(f"  - Full C params: {total_c_params:,}")
        print(f"  - LRTT A+B params: {total_ab_params:,}")
        print(f"  - Ratio: {100 * total_ab_params / total_c_params:.1f}%")
        print("="*70 + "\n")

    return model


def transfer_weights_to_lrtt(standard_model, lrtt_model):
    """Transfer weights from standard ResNet to LRTT ResNet.

    Copies:
    - conv1, bn1, linear: directly
    - layer1-4 conv weights: to C matrix of DigitalLRTTConv2d
    - layer1-4 bn weights: directly
    """
    standard_state = standard_model.state_dict()
    lrtt_state = lrtt_model.state_dict()

    transferred = 0
    for key in standard_state:
        if key in lrtt_state:
            # Direct copy for matching keys
            lrtt_state[key].copy_(standard_state[key])
            transferred += 1
        else:
            # Handle conv weight -> C buffer mapping
            # Standard: "layer1.0.conv1.weight" -> LRTT: "layer1.0.conv1.C"
            if '.weight' in key and 'conv' in key and 'bn' not in key:
                c_key = key.replace('.weight', '.C')
                if c_key in lrtt_state:
                    lrtt_state[c_key].copy_(standard_state[key])
                    transferred += 1
                    print(f"  Transferred: {key} -> {c_key}")

    lrtt_model.load_state_dict(lrtt_state)
    print(f"  Total transferred: {transferred} tensors\n")


# ==============================================================================
# Data loading
# ==============================================================================

def load_images():
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(PATH_DATASET, train=True, download=True,
                                     transform=transform_train)
    val_dataset = datasets.CIFAR10(PATH_DATASET, train=False, download=True,
                                   transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader


# ==============================================================================
# Training functions
# ==============================================================================

def train_one_epoch_standard(model, train_loader, optimizer, criterion):
    """Train standard model for one epoch."""
    model.train()

    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output.data, 1)
        epoch_total += labels.size(0)
        epoch_correct += (predicted == labels).sum().item()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}',
                         'Acc': f'{100 * epoch_correct / epoch_total:.2f}%'})

    train_loss = epoch_loss / len(train_loader.dataset)
    train_acc = 100 * epoch_correct / epoch_total

    return train_loss, train_acc


def train_one_epoch_lrtt(model, train_loader, optimizer, criterion, lr,
                         analyze_transfers: bool = False,
                         analyze_layers: list = None,
                         n_cells: int = 5,
                         log_wandb: bool = False):
    """Train LRTT model for one epoch with A/B updates.

    Args:
        model: LRTT model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        lr: Current learning rate
        analyze_transfers: If True, collect transfer statistics
        analyze_layers: List of layer names to analyze (None = all)
        n_cells: Number of cells to track per matrix
        log_wandb: If True, log transfer analysis to wandb in real-time
    """
    model.train()
    model.set_lr(lr)

    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    # Register hooks to capture output gradients
    grad_outputs = {}
    hooks = []

    def make_hook(name):
        def hook(module, grad_input, grad_output):
            grad_outputs[name] = grad_output[0].detach()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, DigitalLRTTConv2d):
            h = module.register_full_backward_hook(make_hook(name))
            hooks.append(h)

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        grad_outputs.clear()

        # Forward
        output = model(images)
        loss = criterion(output, labels)

        # Backward (populates grad_outputs via hooks)
        loss.backward()

        # LRTT A/B updates using captured gradients
        for name, module in model.named_modules():
            if isinstance(module, DigitalLRTTConv2d):
                if name in grad_outputs:
                    module.lrtt_update(grad_outputs[name])

        # Standard optimizer step (updates conv1, fc, and BN params)
        optimizer.step()

        # Check for transfers (with optional analysis and wandb logging)
        model.check_transfers(
            analyze=analyze_transfers,
            analyze_layers=analyze_layers,
            n_cells=n_cells,
            log_wandb=log_wandb
        )

        epoch_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output.data, 1)
        epoch_total += labels.size(0)
        epoch_correct += (predicted == labels).sum().item()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}',
                         'Acc': f'{100 * epoch_correct / epoch_total:.2f}%'})

    # Remove hooks
    for h in hooks:
        h.remove()

    train_loss = epoch_loss / len(train_loader.dataset)
    train_acc = 100 * epoch_correct / epoch_total

    return train_loss, train_acc


def evaluate(model, val_loader, criterion):
    """Evaluate model."""
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            output = model(images)
            loss = criterion(output, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100 * val_correct / val_total

    return val_loss, val_acc


def apply_lr_schedule(optimizer, epoch, total_epochs, base_lr, schedule="cosine",
                      warmup_ratio=0.0, milestones=None, gamma=0.1, min_lr=1e-5):
    """Apply learning rate schedule."""
    warmup_epochs = int(total_epochs * warmup_ratio)

    if schedule == "constant":
        lr = base_lr
    elif schedule == "cosine":
        if epoch <= warmup_epochs:
            lr = base_lr * (epoch / warmup_epochs) if warmup_epochs > 0 else base_lr
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == "multistep":
        lr = base_lr
        for milestone in (milestones or []):
            if epoch > milestone:
                lr *= gamma
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# ==============================================================================
# Main training loop
# ==============================================================================

def train():
    """Main training function with 2-stage warmstart."""
    torch.manual_seed(SEED)

    # Initialize wandb
    # Summarize per-layer ranks: layer1=16, layer2=32, layer3=64, layer4=128 -> "r16-32-64-128"
    layer_rank_summary = {}
    for k, v in LAYER_CONFIG.items():
        if v.get('use_lrtt', True) and 'rank' in v:
            # Extract layer number (e.g., "layer1.0.conv1" -> "layer1")
            layer_num = k.split('.')[0]
            rank = v['rank']
            if layer_num not in layer_rank_summary:
                layer_rank_summary[layer_num] = rank
    # Create compact rank string: "r16-32-64-128"
    rank_str = "r" + "-".join(str(layer_rank_summary.get(f"layer{i}", 0)) for i in range(1, 5))

    run_name = (f"digital_2stage_s1e{N_EPOCHS_STAGE1}_s2e{N_EPOCHS_STAGE2}"
                f"_{rank_str}_a{LORA_ALPHA}_te{TRANSFER_EVERY}"
                f"_ri{REINIT_MODE}_clamp{int(USE_WEIGHT_CLAMP)}")

    wandb.init(
        project="cifar10-resnet18-digital-2stage-lrtt",
        name=run_name,
        config={
            "stage1_epochs": N_EPOCHS_STAGE1,
            "stage1_lr": LEARNING_RATE_STAGE1,
            "stage1_schedule": LR_SCHEDULE_STAGE1,
            "stage2_epochs": N_EPOCHS_STAGE2,
            "stage2_lr": LEARNING_RATE_STAGE2,
            "stage2_schedule": LR_SCHEDULE_STAGE2,
            "batch_size": BATCH_SIZE,
            "lrtt_rank_default": LRTT_RANK_DEFAULT,
            "layer_config": LAYER_CONFIG,
            "lora_alpha": LORA_ALPHA,
            "transfer_every": TRANSFER_EVERY,
            "transfer_lr": TRANSFER_LR,
            "forward_inject": FORWARD_INJECT,
            "reinit_mode": REINIT_MODE,
            "reinit_decay_factor": REINIT_DECAY_FACTOR,
            "reinit_gain": REINIT_GAIN,
            "correct_gradient_magnitudes": CORRECT_GRADIENT_MAGNITUDES,
            "use_weight_clamp": USE_WEIGHT_CLAMP,
            "w_max": W_MAX,
            "units_in_mbatch": UNITS_IN_MBATCH,
        }
    )

    # Load data
    train_loader, val_loader = load_images()
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    total_epoch = 0

    # =========================================================================
    # Stage 1: Warmstart with standard ResNet
    # =========================================================================
    if N_EPOCHS_STAGE1 > 0:
        print("\n" + "="*70)
        print("STAGE 1: Warmstart Training (Standard ResNet)")
        print("="*70 + "\n")

        standard_model = build_standard_model()

        optimizer = torch.optim.SGD(
            standard_model.parameters(),
            lr=LEARNING_RATE_STAGE1,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=NESTEROV
        )

        for epoch in range(N_EPOCHS_STAGE1):
            total_epoch += 1

            lr = apply_lr_schedule(
                optimizer, epoch + 1, N_EPOCHS_STAGE1, LEARNING_RATE_STAGE1,
                schedule=LR_SCHEDULE_STAGE1, warmup_ratio=WARMUP_RATIO_STAGE1,
                milestones=LR_MILESTONES_STAGE1, gamma=LR_GAMMA_STAGE1
            )

            train_loss, train_acc = train_one_epoch_standard(
                standard_model, train_loader, optimizer, criterion
            )
            val_loss, val_acc = evaluate(standard_model, val_loader, criterion)

            print(f"[S1 Epoch {epoch+1:03d}/{N_EPOCHS_STAGE1}] "
                  f"Loss={train_loss:.4f} TrainAcc={train_acc:.2f}% "
                  f"ValAcc={val_acc:.2f}% LR={lr:.6f}")

            wandb.log({
                "epoch": total_epoch,
                "stage": 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": lr,
            })

            if val_acc > best_acc:
                best_acc = val_acc

        print(f"\nStage 1 Complete! Best Val Acc: {best_acc:.2f}%\n")

    # =========================================================================
    # Stage 2: LRTT Training
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 2: LRTT Training")
    print("="*70 + "\n")

    lrtt_model = build_lrtt_model()

    # Transfer weights from Stage 1 if it was run
    if N_EPOCHS_STAGE1 > 0:
        print("Transferring weights from Stage 1 to Stage 2...")
        transfer_weights_to_lrtt(standard_model, lrtt_model)
        del standard_model  # Free memory

    # Optimizer for Stage 2 (only updates conv1, fc, and BN)
    optimizer = torch.optim.SGD(
        lrtt_model.parameters(),
        lr=LEARNING_RATE_STAGE2,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV
    )

    for epoch in range(N_EPOCHS_STAGE2):
        total_epoch += 1

        lr = apply_lr_schedule(
            optimizer, epoch + 1, N_EPOCHS_STAGE2, LEARNING_RATE_STAGE2,
            schedule=LR_SCHEDULE_STAGE2, warmup_ratio=WARMUP_RATIO_STAGE2,
            milestones=LR_MILESTONES_STAGE2, gamma=LR_GAMMA_STAGE2
        )

        train_loss, train_acc = train_one_epoch_lrtt(
            lrtt_model, train_loader, optimizer, criterion, lr,
            analyze_transfers=TRANSFER_ANALYSIS,
            analyze_layers=TRANSFER_ANALYSIS_LAYERS if TRANSFER_ANALYSIS_LAYERS else None,
            n_cells=TRANSFER_ANALYSIS_CELLS,
            log_wandb=TRANSFER_ANALYSIS_WANDB
        )
        val_loss, val_acc = evaluate(lrtt_model, val_loader, criterion)

        print(f"[S2 Epoch {epoch+1:03d}/{N_EPOCHS_STAGE2}] "
              f"Loss={train_loss:.4f} TrainAcc={train_acc:.2f}% "
              f"ValAcc={val_acc:.2f}% LR={lr:.6f}")

        # Log LRTT statistics periodically
        if epoch % 10 == 0 or epoch == N_EPOCHS_STAGE2 - 1:
            lrtt_layers = lrtt_model.get_lrtt_layers()
            if lrtt_layers:
                sample_layer = lrtt_layers[0]
                print(f"  LRTT stats: Transfers={sample_layer.num_transfers}, "
                      f"A_updates={sample_layer.num_a_updates}, "
                      f"B_updates={sample_layer.num_b_updates}")

        # Print transfer analysis periodically
        if TRANSFER_ANALYSIS and (epoch % 10 == 0 or epoch == N_EPOCHS_STAGE2 - 1):
            analysis_layers = TRANSFER_ANALYSIS_LAYERS if TRANSFER_ANALYSIS_LAYERS else None
            print(lrtt_model.get_transfer_analysis(layers=analysis_layers))

        wandb.log({
            "epoch": total_epoch,
            "stage": 2,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "learning_rate": lr,
        })

        if val_acc > best_acc:
            best_acc = val_acc

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")

    # Final LRTT statistics
    print("\nFinal LRTT Statistics:")
    for i, layer in enumerate(lrtt_model.get_lrtt_layers()):
        print(f"  Layer {i}: Transfers={layer.num_transfers}, "
              f"A_updates={layer.num_a_updates}, B_updates={layer.num_b_updates}")
    print("="*70 + "\n")

    wandb.log({"best_val_accuracy": best_acc})
    wandb.finish()

    # Save model
    save_path = os.path.join(RESULTS, "final_model.pth")
    torch.save(lrtt_model.state_dict(), save_path)
    print(f"Model saved to: {save_path}\n")

    return lrtt_model, best_acc


# ==============================================================================
# Main entry point
# ==============================================================================

def main():
    """Main function."""
    print("\n" + "="*70)
    print("CIFAR-10 ResNet18: Digital LRTT 2-Stage Training")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Stage 1: {N_EPOCHS_STAGE1} epochs, LR={LEARNING_RATE_STAGE1} ({LR_SCHEDULE_STAGE1})")
    print(f"Stage 2: {N_EPOCHS_STAGE2} epochs, LR={LEARNING_RATE_STAGE2} ({LR_SCHEDULE_STAGE2})")
    print(f"LRTT: default_rank={LRTT_RANK_DEFAULT}, alpha={LORA_ALPHA}, transfer_every={TRANSFER_EVERY}")
    print(f"Forward inject: {FORWARD_INJECT}, Reinit mode: {REINIT_MODE}")
    print(f"Weight clamp: {USE_WEIGHT_CLAMP} (w_max={W_MAX})")
    print("="*70 + "\n")

    t0 = time()
    model, final_acc = train()

    print(f"Total Time: {(time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""CIFAR-10 ResNet18 LRTT training with 2-stage warm-start.

Two-stage training approach:
- Stage 1 (FullAnalog): Train C matrix only (A/B disabled) for initial convergence
- Stage 2 (LRTT): Load C weights, initialize A=0/B=Kaiming, train with LRTT

Key features:
- In-memory weight transfer (no file I/O between stages)
- Prevents potential bugs from save/load process
- Consistent setup between stages
- Single script execution

Based on 03_mnist_training_lrtt_warmup.py pattern.
"""
# pylint: disable=invalid-name

import os
from time import time

import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

from aihwkit.nn import AnalogConv2d
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset, PythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType, UpdateParameters
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice

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
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_2STAGE_WARMSTART")
os.makedirs(RESULTS, exist_ok=True)

# Training - Stage 1 (FullAnalog)
N_EPOCHS_STAGE1 = 1  # FullAnalog warm-start epochs
LEARNING_RATE_STAGE1 = 0.1
WARMUP_RATIO_STAGE1 = 0.04

# Training - Stage 2 (LRTT)
N_EPOCHS_STAGE2 = 299  # LRTT fine-tuning epochs
LEARNING_RATE_STAGE2 = 0.1  # Lower LR for fine-tuning (BN/conv1/fc already converged)
WARMUP_RATIO_STAGE2 = 0.0  # No warmup for stage 2

# Common
SEED = 1
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NESTEROV = True
N_CLASSES = 10
NUM_WORKERS = 4

# LRTT configuration
LRTT_RANK_CONV = 32
LRTT_RANK_FC = 32
TRANSFER_EVERY = 100
LORA_ALPHA = 2.0
TRANSFER_LR = LORA_ALPHA


# ==============================================================================
# FullAnalog tile for Stage 1
# ==============================================================================

from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from aihwkit.exceptions import TileError
from dataclasses import dataclass

class LRTTSimulatorTileFullAnalog(LRTTSimulatorTile):
    """Regular LRTT tile for fullanalog training (C-only update)."""

    def _hook_tile_updates(self) -> None:
        """Override parent's hook to update only C matrix for fullanalog training."""
        # Store original update methods
        if hasattr(self, 'tile_a'):
            self.tile_a._orig_update = self.tile_a.update
        if hasattr(self, 'tile_b'):
            self.tile_b._orig_update = self.tile_b.update
        self.tile_c._orig_update = self.tile_c.update

        # Track if we've already handled this batch
        self._update_handled = False

        parent_tile = self

        # Hook tile_c.update() to update C only
        def tile_c_update_wrapper(x_input, d_input, bias=False, in_trans=False,
                                 out_trans=False, non_blocking=False):
            if bias:
                raise TileError("LRTT does not support bias")

            # Prevent double updates
            if parent_tile._update_handled:
                return None
            parent_tile._update_handled = True

            # Update C tile directly
            parent_tile.tile_c._orig_update(x_input, d_input)
            return None

        # Hook tile_a and tile_b to do nothing (no A, B updates for fullanalog)
        def noop_update(*args, **kwargs):
            return None

        # Replace update methods
        if hasattr(self, 'tile_a'):
            self.tile_a.update = noop_update
        if hasattr(self, 'tile_b'):
            self.tile_b.update = noop_update
        self.tile_c.update = tile_c_update_wrapper


@dataclass
class PythonLRTTDeviceFullAnalog(PythonLRTTDevice):
    """Regular LRTT device for fullanalog training."""

    def get_default_tile_module_class(self):
        """Return the fullanalog regular LRTT tile class."""
        return LRTTSimulatorTileFullAnalog


def create_fullanalog_config(rank=1):
    """Create FullAnalog LRTT configuration (C-only training)."""
    device_config = PythonLRTTDeviceFullAnalog(
        rank=rank,  # Minimal rank (not used in forward)
        transfer_every=10000000000,  # Very large to avoid transfers
        lora_alpha=1.0,
        forward_inject=False,  # Use only C matrix in forward pass
        correct_gradient_magnitudes=False,
        unit_cell_devices=[
            IdealizedPresetDevice(),
            IdealizedPresetDevice(),
            IdealizedPresetDevice()
        ]
    )

    # Add mapping configuration (same as rlrtt_scratch)
    mapping = MappingParameter(
        weight_scaling_omega=0.0,  #0.6
        learn_out_scaling=False,
        weight_scaling_lr_compensation=False,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=False,
        max_input_size=512,
        max_output_size=512
    )

    # Add forward/backward IO configuration (same as rlrtt_scratch)
    forward_io = IOParameters(
        inp_res=0.0,   #0.007937
        inp_bound=1.0,
        inp_noise=0.0,
        inp_sto_round=False,
        out_res=0.00,   #0.001961
        out_bound=12.0,
        out_noise=0.0,     #0.06
        w_noise=0.0,
        w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=True,   #False
        max_bm_factor=1000,
    )

    # Update parameters - disable BL management for debugging
    update_params = UpdateParameters(
        desired_bl=127,
        update_bl_management=True,
        update_management=True,
    )

    return PythonLRTTRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io, update=update_params)


# ==============================================================================
# LRTT configuration for Stage 2
# ==============================================================================

def create_lrtt_config(rank, is_conv=True):
    """Create LRTT configuration for Stage 2."""
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        transfer_lr=TRANSFER_LR,
        forward_inject=False,  # Use C matrix only in forward (A⊗B accumulated via transfers)
        correct_gradient_magnitudes=False,
        reinit_gain=0.1,
        reinit_mode="standard",  # A=0, B=Kaiming
        unit_cell_devices=[
            IdealizedPresetDevice(),
            IdealizedPresetDevice(),
            IdealizedPresetDevice()
        ]
    )

    # Add mapping configuration (same as rlrtt_scratch)
    mapping = MappingParameter(
        weight_scaling_omega=0.0,    #0.6
        learn_out_scaling=False,
        weight_scaling_lr_compensation=False,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=False,
        max_input_size=512,
        max_output_size=512
    )

    # Add forward/backward IO configuration (same as rlrtt_scratch)
    forward_io = IOParameters(
        inp_res=0.00, #0.07937
        inp_bound=1.0,
        inp_noise=0.0,
        inp_sto_round=False,
        out_res=0.00,  #0.001961
        out_bound=12.0,
        out_noise=0.0,  #0.06
        w_noise=0.0,
        w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=False,   #False
        max_bm_factor=1000,
    )

    # Update parameters - disable BL management for debugging
    update_params = UpdateParameters(
        desired_bl=127,
        update_bl_management=True,
        update_management=True,
    )

    return PythonLRTTRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io, update=update_params)


# ==============================================================================
# ResNet18 model builders
# ==============================================================================

class BasicBlock(nn.Module):
    """ResNet BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, rpu_config=None, use_analog=True):
        super().__init__()

        if use_analog:
            self.conv1 = AnalogConv2d(in_planes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False, rpu_config=rpu_config)
            self.conv2 = AnalogConv2d(planes, planes, kernel_size=3, stride=1,
                                     padding=1, bias=False, rpu_config=rpu_config)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                  padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if use_analog:
                self.shortcut = nn.Sequential(
                    AnalogConv2d(in_planes, self.expansion * planes, kernel_size=1,
                               stride=stride, bias=False, rpu_config=rpu_config),
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


class ResNet(nn.Module):
    """ResNet18 for CIFAR-10."""

    def __init__(self, block, num_blocks, num_classes=10, rpu_config=None,
                 conv1_use_analog=False, fc_use_analog=False):
        super().__init__()
        self.in_planes = 64

        # First conv layer (digital for both stages)
        if conv1_use_analog:
            self.conv1 = AnalogConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                     bias=False, rpu_config=rpu_config)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        # ResNet layers (analog)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, rpu_config=rpu_config)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, rpu_config=rpu_config)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, rpu_config=rpu_config)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, rpu_config=rpu_config)

        # Final FC layer (digital for both stages)
        if fc_use_analog:
            # Note: AnalogLinear not imported, using Conv2d hack
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, rpu_config):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rpu_config, use_analog=True))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def build_fullanalog_model():
    """Build FullAnalog model for Stage 1."""
    fullanalog_config = create_fullanalog_config(rank=1)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=N_CLASSES,
                   rpu_config=fullanalog_config,
                   conv1_use_analog=False, fc_use_analog=False)

    if USE_CUDA:
        model = model.to(DEVICE)

    print("\n" + "="*70)
    print("Stage 1: FullAnalog Model")
    print("="*70)
    print("  - All ResNet blocks: FullAnalog (C-only training)")
    print("  - conv1: Digital (FloatingPoint)")
    print("  - fc: Digital (FloatingPoint)")
    print(f"  - Total epochs: {N_EPOCHS_STAGE1}")
    print(f"  - Learning rate: {LEARNING_RATE_STAGE1} (CONSTANT - no decay)")
    print("="*70 + "\n")

    return model


def build_lrtt_model():
    """Build LRTT model for Stage 2 (without weight loading)."""
    lrtt_config_conv = create_lrtt_config(LRTT_RANK_CONV, is_conv=True)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=N_CLASSES,
                   rpu_config=lrtt_config_conv,
                   conv1_use_analog=False, fc_use_analog=False)

    if USE_CUDA:
        model = model.to(DEVICE)

    print("\n" + "="*70)
    print("Stage 2: LRTT Model")
    print("="*70)
    print(f"  - All ResNet blocks: LRTT (rank={LRTT_RANK_CONV})")
    print("  - conv1: Digital (FloatingPoint)")
    print("  - fc: Digital (FloatingPoint)")
    print(f"  - Transfer every: {TRANSFER_EVERY} steps")
    print(f"  - LoRA alpha: {LORA_ALPHA}")
    print(f"  - Total epochs: {N_EPOCHS_STAGE2}")
    print(f"  - Learning rate: {LEARNING_RATE_STAGE2} (COSINE decay)")
    print(f"  - Warmup ratio: {WARMUP_RATIO_STAGE2}")
    print("="*70 + "\n")

    return model


# ==============================================================================
# Weight transfer from FullAnalog to LRTT
# ==============================================================================

@torch.no_grad()
def transfer_weights_fullanalog_to_lrtt(fullanalog_model, lrtt_model):
    """Transfer weights from FullAnalog model to LRTT model.

    Transfers:
    1. Analog C matrices (FullAnalog → LRTT C tiles)
    2. BatchNorm parameters (weight, bias, running_mean, running_var)
    3. Digital layers (conv1, fc)

    LRTT A, B matrices are reinitialized (A=0, B=Kaiming).
    """
    print("\n" + "="*70)
    print("Transferring weights: FullAnalog → LRTT")
    print("="*70)

    transferred = 0

    # Get state dicts
    fullanalog_dict = fullanalog_model.state_dict()
    lrtt_dict = lrtt_model.state_dict()

    # Transfer BatchNorm and digital layer parameters (skip analog_module entirely)
    for name in lrtt_dict.keys():
        if name in fullanalog_dict:
            # Only transfer if it's NOT analog_module related
            if 'analog_module' not in name:
                # Check if it's a tensor (not dict or other types)
                if isinstance(fullanalog_dict[name], torch.Tensor):
                    lrtt_dict[name].copy_(fullanalog_dict[name])
                    transferred += 1

    # Load state dict
    lrtt_model.load_state_dict(lrtt_dict, strict=False)

    print(f"✓ Transferred {transferred} non-analog parameters (BatchNorm, conv1, fc)")

    # Transfer analog C matrices
    analog_transferred = 0
    for (fa_name, fa_module), (lrtt_name, lrtt_module) in zip(
        fullanalog_model.named_modules(), lrtt_model.named_modules()
    ):
        if isinstance(fa_module, AnalogConv2d) and isinstance(lrtt_module, AnalogConv2d):
            if hasattr(fa_module, 'analog_module') and hasattr(lrtt_module, 'analog_module'):
                # Get FullAnalog weights
                if hasattr(fa_module.analog_module, 'get_lrtt_component_weights'):
                    C_fa, _, _ = fa_module.analog_module.get_lrtt_component_weights()
                else:
                    C_fa = fa_module.analog_module.get_weights()[0]

                # Set LRTT C matrix
                if hasattr(lrtt_module.analog_module, 'set_lrtt_component_weights'):
                    C_lrtt, A_lrtt, B_lrtt = lrtt_module.analog_module.get_lrtt_component_weights()
                    lrtt_module.analog_module.set_lrtt_component_weights(
                        C_fa.to(C_lrtt.device), A_lrtt, B_lrtt
                    )
                    analog_transferred += 1
                    print(f"  ✓ {fa_name}: C matrix transferred")

    print(f"✓ Transferred {analog_transferred} analog C matrices")

    # Reinitialize LRTT A, B matrices
    reinit_count = 0
    for name, module in lrtt_model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            module.analog_module.controller.reinit()
            reinit_count += 1

    print(f"✓ Reinitialized {reinit_count} LRTT layers (A=0, B=Kaiming)")
    print("="*70 + "\n")

    return lrtt_model


# ==============================================================================
# Data loading
# ==============================================================================

def load_images():
    """Load CIFAR-10 dataset."""
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Validation transforms (no augmentation)
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
# Training and evaluation
# ==============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
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


def apply_constant_lr(optimizer, base_lr):
    """Apply constant learning rate (no decay)."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr
    return base_lr


def apply_warmup_cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_ratio=0.0, min_lr=1e-5):
    """Apply learning rate warmup + cosine annealing (epoch-based)."""
    import math

    warmup_epochs = int(total_epochs * warmup_ratio)

    if epoch <= warmup_epochs:
        # Linear warmup
        lr = base_lr * (epoch / warmup_epochs) if warmup_epochs > 0 else base_lr
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# ==============================================================================
# Main training loop
# ==============================================================================

def train_2stage():
    """Two-stage training: FullAnalog → LRTT."""
    torch.manual_seed(SEED)

    # Initialize wandb
    wandb.init(
        project="cifar10-resnet18-2stage-warmstart",
        name=f"2stage_s1e{N_EPOCHS_STAGE1}_s2e{N_EPOCHS_STAGE2}_r{LRTT_RANK_CONV}_a{LORA_ALPHA}",
        config={
            "stage1_epochs": N_EPOCHS_STAGE1,
            "stage1_lr": LEARNING_RATE_STAGE1,
            "stage1_warmup": WARMUP_RATIO_STAGE1,
            "stage2_epochs": N_EPOCHS_STAGE2,
            "stage2_lr": LEARNING_RATE_STAGE2,
            "stage2_warmup": WARMUP_RATIO_STAGE2,
            "batch_size": BATCH_SIZE,
            "lrtt_rank": LRTT_RANK_CONV,
            "lora_alpha": LORA_ALPHA,
            "transfer_every": TRANSFER_EVERY,
        }
    )

    # Load data
    train_loader, val_loader = load_images()
    criterion = nn.CrossEntropyLoss()

    # ========================================================================
    # Stage 1: FullAnalog training
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: FullAnalog Training")
    print("="*70 + "\n")

    model_stage1 = build_fullanalog_model()
    optimizer_stage1 = AnalogSGD(model_stage1.parameters(), lr=LEARNING_RATE_STAGE1,
                                 momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
    optimizer_stage1.regroup_param_groups(model_stage1)

    # Handle N_EPOCHS_STAGE1 = 0 case (skip Stage 1, use random init)
    if N_EPOCHS_STAGE1 == 0:
        print("N_EPOCHS_STAGE1 = 0: Skipping Stage 1 training (random initialization)")
        val_loss, val_acc = evaluate(model_stage1, val_loader, criterion)
        print(f"Initial Val Accuracy (random): {val_acc:.2f}%\n")
    else:
        for epoch in range(N_EPOCHS_STAGE1):
            # Apply constant LR for Stage 1 (no decay - keep weights "fluid")
            lr = apply_constant_lr(optimizer_stage1, LEARNING_RATE_STAGE1)

            # Train
            train_loss, train_acc = train_one_epoch(model_stage1, train_loader, optimizer_stage1, criterion)

            # Validate
            val_loss, val_acc = evaluate(model_stage1, val_loader, criterion)

            print(f"[Stage1 {epoch+1:02d}/{N_EPOCHS_STAGE1}] "
                  f"Loss={train_loss:.4f} TrainAcc={train_acc:.2f}% "
                  f"ValAcc={val_acc:.2f}% LR={lr:.6f} (constant)")

            wandb.log({
                "stage": 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": lr,
            })

        print(f"\n✓ Stage 1 Complete: Val Accuracy = {val_acc:.2f}%\n")

    # ========================================================================
    # Weight Transfer: FullAnalog → LRTT
    # ========================================================================
    model_stage2 = build_lrtt_model()

    # ========================================================================
    # DEBUG: Compare forward pass BEFORE transfer
    # ========================================================================
    print("\n" + "="*70)
    print("DEBUG: Comparing Forward Pass Before/After Transfer")
    print("="*70)

    # Get a sample batch for comparison
    sample_images, sample_labels = next(iter(val_loader))
    sample_images = sample_images.to(DEVICE)

    # Forward pass with FullAnalog model
    model_stage1.eval()
    with torch.no_grad():
        output_fullanalog = model_stage1(sample_images)

    # Forward pass with LRTT model BEFORE transfer (random init)
    model_stage2.eval()
    with torch.no_grad():
        output_lrtt_before = model_stage2(sample_images)

    print(f"FullAnalog output mean: {output_fullanalog.mean().item():.6f}, std: {output_fullanalog.std().item():.6f}")
    print(f"LRTT (before transfer) output mean: {output_lrtt_before.mean().item():.6f}, std: {output_lrtt_before.std().item():.6f}")

    # Transfer weights
    model_stage2 = transfer_weights_fullanalog_to_lrtt(model_stage1, model_stage2)

    # Forward pass with LRTT model AFTER transfer
    model_stage2.eval()
    with torch.no_grad():
        output_lrtt_after = model_stage2(sample_images)

    print(f"\nFullAnalog output mean: {output_fullanalog.mean().item():.6f}, std: {output_fullanalog.std().item():.6f}")
    print(f"LRTT (after transfer) output mean: {output_lrtt_after.mean().item():.6f}, std: {output_lrtt_after.std().item():.6f}")

    # Compare outputs
    diff = (output_fullanalog - output_lrtt_after).abs()
    print(f"\nOutput difference (FullAnalog vs LRTT after transfer):")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Relative diff: {(diff.mean() / output_fullanalog.abs().mean()).item():.6f}")

    # Check if predictions match
    pred_fullanalog = output_fullanalog.argmax(dim=1)
    pred_lrtt = output_lrtt_after.argmax(dim=1)
    pred_match = (pred_fullanalog == pred_lrtt).float().mean().item() * 100
    print(f"  Prediction match: {pred_match:.2f}%")
    print("="*70 + "\n")

    # Verify transfer
    print("Verifying weight transfer...")
    val_loss_after_transfer, val_acc_after_transfer = evaluate(model_stage2, val_loader, criterion)
    print(f"✓ LRTT model after transfer: Val Accuracy = {val_acc_after_transfer:.2f}%")
    print(f"  (Should match Stage 1 final accuracy: {val_acc:.2f}%)\n")

    wandb.log({
        "stage1_final_accuracy": val_acc,
        "stage2_initial_accuracy": val_acc_after_transfer,
        "transfer_accuracy_match": abs(val_acc - val_acc_after_transfer) < 0.5,
    })

    # Clean up Stage 1 model
    del model_stage1, optimizer_stage1
    torch.cuda.empty_cache() if USE_CUDA else None

    # ========================================================================
    # Stage 2: LRTT training
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: LRTT Training")
    print("="*70 + "\n")

    optimizer_stage2 = AnalogSGD(model_stage2.parameters(), lr=LEARNING_RATE_STAGE2,
                                 momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
    optimizer_stage2.regroup_param_groups(model_stage2)

    for epoch in range(N_EPOCHS_STAGE2):
        # Apply LR schedule
        lr = apply_warmup_cosine_lr(optimizer_stage2, epoch + 1, N_EPOCHS_STAGE2,
                                    LEARNING_RATE_STAGE2, WARMUP_RATIO_STAGE2)

        # Train
        train_loss, train_acc = train_one_epoch(model_stage2, train_loader, optimizer_stage2, criterion)

        # Validate
        val_loss, val_acc = evaluate(model_stage2, val_loader, criterion)

        print(f"[Stage2 {epoch+1:02d}/{N_EPOCHS_STAGE2}] "
              f"Loss={train_loss:.4f} TrainAcc={train_acc:.2f}% "
              f"ValAcc={val_acc:.2f}% LR={lr:.6f} (cosine)")

        # Log LRTT statistics
        if epoch % 5 == 0 or epoch == N_EPOCHS_STAGE2 - 1:
            for name, module in model_stage2.named_modules():
                if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
                    ctrl = module.analog_module.controller
                    print(f"  {name}: Transfers={ctrl.num_transfers}, "
                          f"A_updates={ctrl.num_a_updates}, B_updates={ctrl.num_b_updates}")

        wandb.log({
            "stage": 2,
            "epoch": N_EPOCHS_STAGE1 + epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "learning_rate": lr,
        })

    print(f"\n✓ Stage 2 Complete: Val Accuracy = {val_acc:.2f}%\n")

    # Final statistics
    print("\n" + "="*70)
    print("Final LRTT Statistics")
    print("="*70)
    for name, module in model_stage2.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            ctrl = module.analog_module.controller
            print(f"  {name}:")
            print(f"    Transfers: {ctrl.num_transfers}")
            print(f"    A updates: {ctrl.num_a_updates}")
            print(f"    B updates: {ctrl.num_b_updates}")
    print("="*70 + "\n")

    wandb.log({
        "final_val_accuracy": val_acc,
    })

    wandb.finish()

    return model_stage2, val_acc


# ==============================================================================
# Main entry point
# ==============================================================================

def main():
    """Main function."""
    print("\n" + "="*70)
    print("CIFAR-10 ResNet18: 2-Stage LRTT Training with Warm-Start")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Stage 1: {N_EPOCHS_STAGE1} epochs (FullAnalog)")
    print(f"Stage 2: {N_EPOCHS_STAGE2} epochs (LRTT)")
    print(f"LRTT Rank: {LRTT_RANK_CONV}")
    print(f"LoRA Alpha: {LORA_ALPHA}")
    print(f"Transfer Every: {TRANSFER_EVERY} steps")
    print("="*70 + "\n")

    t0 = time()
    model, final_acc = train_2stage()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Final Validation Accuracy: {final_acc:.2f}%")
    print(f"Total Time: {(time() - t0) / 60:.2f} min")
    print(f"{'='*70}\n")

    # Save final model
    save_path = os.path.join(RESULTS, "final_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}\n")


if __name__ == "__main__":
    main()

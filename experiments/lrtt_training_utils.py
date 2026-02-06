#!/usr/bin/env python3
"""Shared LRTT Training Utilities.

Provides common functions for LRTT training experiments:
- Data loading (MNIST)
- Model creation with configurable reinit modes
- Training and validation loops
"""

import os
os.environ["LRTT_SILENT"] = "1"

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional, Callable, Dict, Any

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


# Default device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# SoftBounds config (no noise)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    """Convert lifetime to dt_batch_sec for sixt1c_ab preset."""
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    dt = -TAU_SEC * math.log(1 - delta)
    return dt


def load_data(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset.

    Args:
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, val_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('/tmp/mnist', download=True, train=True, transform=transform)
    val_set = datasets.MNIST('/tmp/mnist', download=True, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


def create_model(
    rank: int,
    te: int,
    lifetime: float,
    lr: float,
    tlr: float,
    reinit_mode: str = "decay",
    decay_factor: float = 1.0,
    device: torch.device = DEVICE
) -> AnalogSequential:
    """Create LRTT model with configurable reinit mode.

    Args:
        rank: LoRA rank
        te: Transfer every N steps
        lifetime: Lifetime parameter for A/B tiles
        lr: Learning rate (used for optimizer, not in model)
        tlr: Transfer learning rate
        reinit_mode: Reinit strategy - "decay", "hybrid", "standard"
        decay_factor: Decay factor for "decay" and "hybrid" modes
        device: Target device

    Returns:
        AnalogSequential model
    """
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)

    # Calculate lifetime for A/B tiles
    TAU_SEC = 46505.0
    if dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        ab_lifetime = 1.0 / delta
    else:
        ab_lifetime = 0.0

    # A/B tiles: 6T1C LinearStepDevice (sixt1c original params)
    ab_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime,
        lifetime_dtod=0.1,
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: SoftBounds with NO NOISE
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    # Create PythonLRTTDevice with specified reinit mode
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode=reinit_mode,
        decay_factor=decay_factor,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    model = AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    )
    model.to(device)
    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = DEVICE,
    step_callback: Optional[Callable[[int, nn.Module, torch.optim.Optimizer], None]] = None
) -> int:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Target device
        step_callback: Optional callback called after each step with (step, model, optimizer)

    Returns:
        Number of steps executed
    """
    model.train()
    step = 0
    for data, target in train_loader:
        data = data.to(device, non_blocking=True).view(data.shape[0], -1)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        step += 1

        if step_callback is not None:
            step_callback(step, model, optimizer)

    return step


def validate(model: nn.Module, val_loader: DataLoader, device: torch.device = DEVICE) -> float:
    """Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Target device

    Returns:
        Validation accuracy (0-100)
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True).view(data.shape[0], -1)
            target = target.to(device, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def get_lrtt_tiles(model: AnalogSequential) -> Dict[str, Any]:
    """Extract LRTT tiles from model.

    Args:
        model: AnalogSequential model

    Returns:
        Dict mapping layer names to their LRTT tile objects
    """
    tiles = {}
    for name, module in model.named_modules():
        # Check for analog_module (LRTT uses this instead of analog_tile)
        if hasattr(module, 'analog_module'):
            tile = module.analog_module
            if hasattr(tile, 'controller') and hasattr(tile, 'get_lrtt_component_weights'):
                tiles[name] = tile
    return tiles


def get_lrtt_component_weights(tile) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get LRTT component weights from a tile.

    Args:
        tile: LRTTSimulatorTile

    Returns:
        Tuple of (C, A, B) weight tensors
    """
    if hasattr(tile, 'get_lrtt_component_weights'):
        return tile.get_lrtt_component_weights()
    else:
        # Fallback for non-LRTT tiles
        weights = tile.get_weights()[0]
        return weights, None, None

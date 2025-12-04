# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 18 with LRTT warmup: ResNet32 CNN with CIFAR10 using LRTT layers.

CIFAR10 dataset on a ResNet inspired network using LRTT (Low-Rank Tensor-Train)
analog layers with warm-start training strategy:

- Phase A (Warmup): Full-rank training with IdealizedPreset for initial epochs
- Phase B (LRTT): Training with automatic merge-and-reinit controlled by transfer_every
- Proper weight transfer from warm-start to LRTT's C tile
- C-only validation to monitor merge quality
"""
# pylint: disable=invalid-name

# Imports
import os
from datetime import datetime
from time import time
import copy

# Imports from PyTorch.
import torch
from torch import nn, Tensor, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from torchvision import datasets, transforms
import pandas as pd

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.presets.configs import IdealizedPreset
from aihwkit.simulator.rpu_base import cuda


# Device to use
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Training parameters
SEED = 1
N_EPOCHS = 100  # Total epochs
WARM_EPOCHS = 5  # Phase-A: full-rank warm-start epochs
BATCH_SIZE = 128  # Increased for better GPU utilization
LEARNING_RATE = 0.1
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# LRTT configuration parameters
LRTT_RANK_CONV = 8  # Rank for convolutional layers
LRTT_RANK_FC = 16  # Rank for fully connected layers
TRANSFER_EVERY = 100  # Transfer A⊗B to C every N updates
LORA_ALPHA = 1  # LoRA scaling factor
VALIDATE_C_ONLY_EVERY = 1  # Validate with C-only every N epochs

# Learning rate schedule
LR_WARMSTART = 0.1  # Learning rate for warm-start phase
LR_LRTT = 0.05  # Learning rate for LRTT phase (reduced for stability)

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results_cifar10_lrtt_warmup")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "model_weight.pth")

# Excel file path with parameters in filename
EXCEL_PATH = os.path.join(RESULTS, f"cifar10_lrtt_warmup_te{TRANSFER_EVERY}_r{LRTT_RANK_CONV}_a{LORA_ALPHA}_warm{WARM_EPOCHS}.xlsx")


def create_lrtt_config_conv():
    """Create LRTT configuration for convolutional layers.
    
    Returns:
        PythonLRTTRPUConfig: LRTT configuration for conv layers
    """
    device_config = PythonLRTTPreset.idealized(
        rank=LRTT_RANK_CONV,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA
    )
    device_config.transfer_lr = device_config.lora_alpha
    device_config.forward_inject = True
    device_config.correct_gradient_magnitudes = True
    device_config.reinit_gain = 0.5  # For stability
    
    # Add mapping for larger layers
    mapping = MappingParameter(
        weight_scaling_omega=0.6,
        max_input_size=512,
        max_output_size=512
    )
    
    return PythonLRTTRPUConfig(device=device_config, mapping=mapping)


def create_idealized_config_conv():
    """Create IdealizedPreset configuration for warm-start phase.
    
    Returns:
        IdealizedPreset: Configuration for warm-start conv layers
    """
    config = IdealizedPreset()
    
    # Add mapping for larger layers
    mapping = MappingParameter(
        weight_scaling_omega=0.6,
        max_input_size=512,
        max_output_size=512
    )
    config.mapping = mapping
    
    return config


class ResidualBlockWarmup(nn.Module):
    """Residual block for warm-start phase with IdealizedPreset."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        # Use IdealizedPreset for warm-start (no bias to match LRTT)
        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch, 
            kernel_size=3, padding=1, stride=stride,
            bias=False,
            rpu_config=create_idealized_config_conv()
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        
        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=False,
            rpu_config=create_idealized_config_conv()
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=False,
                rpu_config=create_idealized_config_conv()
            )
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass"""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


class ResidualBlockLRTT(nn.Module):
    """Residual block with LRTT analog convolutional layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        # Use LRTT analog convolutional layers
        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch, 
            kernel_size=3, padding=1, stride=stride,
            bias=False,
            rpu_config=create_lrtt_config_conv()
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        
        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=False,
            rpu_config=create_lrtt_config_conv()
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=False,
                rpu_config=create_lrtt_config_conv()
            )
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass"""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks(in_ch, hidden_ch, num_layer, first_layer=False, use_lrtt=False):
    """Concatenate multiple residual blocks to form a layer.
    
    Args:
        in_ch: Input channels
        hidden_ch: Hidden channels
        num_layer: Number of blocks
        first_layer: Whether this is the first layer
        use_lrtt: Whether to use LRTT blocks (vs warmup blocks)

    Returns:
       List: list of layer blocks
    """
    layers = []
    BlockClass = ResidualBlockLRTT if use_lrtt else ResidualBlockWarmup
    
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(BlockClass(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(BlockClass(hidden_ch, hidden_ch))
    return layers


def create_warmup_model():
    """Create ResNet34 model for warm-start phase with IdealizedPreset.

    Returns:
       nn.Module: warm-start model with IdealizedPreset
    """

    block_per_layers = (3, 4, 6, 3)
    base_channel = 16
    channel = (base_channel, 2 * base_channel, 4 * base_channel)

    # Input layer uses FloatingPointDevice for better stability
    l0 = nn.Sequential(
        AnalogConv2d(
            3, channel[0],
            kernel_size=3, stride=1, padding=1,
            bias=True,
            rpu_config=FloatingPointRPUConfig()
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    # Use IdealizedPreset for warm-start
    l1 = nn.Sequential(
        *concatenate_layer_blocks(channel[0], channel[0], block_per_layers[0], first_layer=True, use_lrtt=False)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks(channel[0], channel[1], block_per_layers[1], use_lrtt=False))
    l3 = nn.Sequential(*concatenate_layer_blocks(channel[1], channel[2], block_per_layers[2], use_lrtt=False))
    
    # Final classification layer uses FloatingPointDevice
    l4 = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(
            channel[2], N_CLASSES,
            bias=True,
            rpu_config=FloatingPointRPUConfig()
        )
    )

    model = nn.Sequential(l0, l1, l2, l3, l4)
    
    print(f"\nCreated Warm-start ResNet:")
    print(f"  Input layer: FloatingPointDevice")
    print(f"  Conv layers: IdealizedPreset (full-rank)")
    print(f"  Final FC layer: FloatingPointDevice")
    print(f"  Warm-start epochs: {WARM_EPOCHS}\n")
    
    return model


@torch.no_grad()
def replace_with_lrtt(warmup_model):
    """Replace IdealizedPreset layers with LRTT layers after warm-start.
    
    Copies weights from warm-start model to LRTT C tiles.
    
    Args:
        warmup_model: Trained warm-start model
        
    Returns:
        nn.Module: New model with LRTT layers
    """
    
    block_per_layers = (3, 4, 6, 3)
    base_channel = 16
    channel = (base_channel, 2 * base_channel, 4 * base_channel)
    
    # Keep the same input layer (FloatingPoint)
    l0 = warmup_model[0]
    
    # Create new LRTT layers
    l1 = nn.Sequential(
        *concatenate_layer_blocks(channel[0], channel[0], block_per_layers[0], first_layer=True, use_lrtt=True)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks(channel[0], channel[1], block_per_layers[1], use_lrtt=True))
    l3 = nn.Sequential(*concatenate_layer_blocks(channel[1], channel[2], block_per_layers[2], use_lrtt=True))
    
    # Keep the same output layer (FloatingPoint)
    l4 = warmup_model[4]
    
    # Create new model
    lrtt_model = nn.Sequential(l0, l1, l2, l3, l4)
    
    if USE_CUDA:
        lrtt_model.cuda()
    
    # Transfer weights from warmup to LRTT C tiles
    print("Transferring weights from warm-start to LRTT C tiles...")
    
    # Copy the state dict from warmup model
    warmup_state = warmup_model.state_dict()
    lrtt_state = lrtt_model.state_dict()
    
    # Transfer matching weights (BatchNorm and other compatible layers)
    transferred_count = 0
    for key in warmup_state:
        if key in lrtt_state and warmup_state[key].shape == lrtt_state[key].shape:
            lrtt_state[key] = warmup_state[key].clone()
            transferred_count += 1
    
    # Load the modified state dict
    lrtt_model.load_state_dict(lrtt_state, strict=False)
    
    # Now transfer analog tile weights specifically
    # Following the pattern from 03_mnist_training_lrtt_warmup.py
    for warmup_layer, lrtt_layer in zip([warmup_model[1], warmup_model[2], warmup_model[3]], 
                                        [l1, l2, l3]):
        for warmup_block, lrtt_block in zip(warmup_layer, lrtt_layer):
            # Transfer conv1 weights
            if hasattr(warmup_block, 'conv1') and hasattr(lrtt_block, 'conv1'):
                try:
                    # Get weights from warmup analog module
                    if hasattr(warmup_block.conv1, 'analog_module'):
                        w_full = warmup_block.conv1.analog_module.get_weights()[0].detach().clone()
                        
                        # Set weights to LRTT C tile
                        if hasattr(lrtt_block.conv1, 'analog_module'):
                            try:
                                lrtt_block.conv1.analog_module.tile_c.set_weights(w_full)
                            except Exception:
                                # Fallback: try through controller
                                if hasattr(lrtt_block.conv1.analog_module, 'controller'):
                                    lrtt_block.conv1.analog_module.controller.tile_c.set_weights(w_full)
                except Exception:
                    pass  # Skip if transfer fails
            
            # Transfer conv2 weights  
            if hasattr(warmup_block, 'conv2') and hasattr(lrtt_block, 'conv2'):
                try:
                    # Get weights from warmup analog module
                    if hasattr(warmup_block.conv2, 'analog_module'):
                        w_full = warmup_block.conv2.analog_module.get_weights()[0].detach().clone()
                        
                        # Set weights to LRTT C tile
                        if hasattr(lrtt_block.conv2, 'analog_module'):
                            try:
                                lrtt_block.conv2.analog_module.tile_c.set_weights(w_full)
                            except Exception:
                                # Fallback: try through controller
                                if hasattr(lrtt_block.conv2.analog_module, 'controller'):
                                    lrtt_block.conv2.analog_module.controller.tile_c.set_weights(w_full)
                except Exception:
                    pass  # Skip if transfer fails
            
            # Transfer convskip weights if exists
            if hasattr(warmup_block, 'convskip') and warmup_block.convskip is not None:
                if hasattr(lrtt_block, 'convskip') and lrtt_block.convskip is not None:
                    try:
                        # Get weights from warmup analog module
                        if hasattr(warmup_block.convskip, 'analog_module'):
                            w_full = warmup_block.convskip.analog_module.get_weights()[0].detach().clone()
                            
                            # Set weights to LRTT C tile
                            if hasattr(lrtt_block.convskip, 'analog_module'):
                                try:
                                    lrtt_block.convskip.analog_module.tile_c.set_weights(w_full)
                                except Exception:
                                    # Fallback: try through controller
                                    if hasattr(lrtt_block.convskip.analog_module, 'controller'):
                                        lrtt_block.convskip.analog_module.controller.tile_c.set_weights(w_full)
                    except Exception:
                        pass  # Skip if transfer fails
    
    print(f"Weight transfer complete. Transferred {transferred_count} state dict entries.")
    print(f"Analog tile weights transferred to LRTT C tiles (following MNIST warmup pattern).")
    print(f"LRTT A and B matrices will be initialized fresh.")
    print(f"  Conv layers rank: {LRTT_RANK_CONV}")
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}\n")
    
    return lrtt_model


def load_images():
    """Load images for train from torchvision datasets.

    Returns:
        Dataset, Dataset: train data and validation data"""
    mean = Tensor([0.4914, 0.4822, 0.4465])
    std = Tensor([0.2470, 0.2435, 0.2616])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)

    return train_data, validation_data


def create_sgd_optimizer(model, learning_rate):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate

    Returns:
        Optimizer: created analog optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer.regroup_param_groups(model)

    return optimizer


def set_forward_inject(model, enabled):
    """Set forward_inject state for all LRTT layers.
    
    Args:
        model (nn.Module): Model with LRTT layers
        enabled (bool): Whether to enable forward_inject
        
    Returns:
        list: Original states for restoration
    """
    original_states = []
    for module in model.modules():
        if hasattr(module, 'analog_tiles') and hasattr(module.analog_tiles, 'controller'):
            controller = module.analog_tiles.controller
            original_states.append((controller, controller.forward_inject_enabled))
            controller.forward_inject_enabled = enabled
    return original_states


def restore_forward_inject(original_states):
    """Restore original forward_inject states."""
    for controller, original_state in original_states:
        controller.forward_inject_enabled = original_state


def train_step(train_data, model, criterion, optimizer):
    """Train network for one epoch.

    Args:
        train_data (DataLoader): Training data
        model (nn.Module): Model to train
        criterion: Loss function
        optimizer: Optimizer

    Returns:
        float, float: epoch loss and accuracy
    """
    total_loss = 0
    correct = 0
    total = 0

    model.train()

    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch_max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = total_loss / len(train_data.dataset)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def test_evaluation(validation_data, model, criterion, c_only=False):
    """Test trained network with option for C-only validation.

    Args:
        validation_data (DataLoader): Validation data
        model (nn.Module): Model to evaluate
        criterion: Loss function
        c_only (bool): If True, use C-only mode (no A@B contribution)

    Returns:
        float, float: test loss and test accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()
    
    # Temporarily disable forward_inject if C-only mode
    original_states = None
    if c_only:
        original_states = set_forward_inject(model, enabled=False)

    try:
        with no_grad():
            for images, labels in validation_data:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                pred = model(images)
                loss = criterion(pred, labels)
                total_loss += loss.item() * images.size(0)

                _, predicted = torch_max(pred.data, 1)
                total_images += labels.size(0)
                predicted_ok += (predicted == labels).sum().item()
    finally:
        # Restore original forward_inject states
        if original_states:
            restore_forward_inject(original_states)
            
    epoch_loss = total_loss / len(validation_data.dataset)
    accuracy = predicted_ok / total_images * 100

    return epoch_loss, accuracy


def print_lrtt_statistics(model):
    """Print LRTT statistics for monitoring."""
    lrtt_count = 0
    total_transfers = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'analog_tiles') and hasattr(module.analog_tiles, 'controller'):
            controller = module.analog_tiles.controller
            lrtt_count += 1
            total_transfers += controller.num_transfers
    
    if lrtt_count > 0:
        return f" | LRTT: {lrtt_count} layers, {total_transfers} transfers"
    return ""


def save_results_to_excel(results_df, params_dict, excel_path):
    """Save training results and parameters to Excel file.
    
    Args:
        results_df: DataFrame with training results
        params_dict: Dictionary with training parameters
        excel_path: Path to save Excel file
    """
    # Create parameters DataFrame
    params_df = pd.DataFrame(list(params_dict.items()), columns=['Parameter', 'Value'])
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Training_Results', index=False)
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
    
    print(f"\nResults saved to: {excel_path}")


def main():
    """Train ResNet on CIFAR10 with warm-start LRTT strategy."""
    
    # Seed for reproducibility
    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    # Load datasets
    train_data, validation_data = load_images()
    
    # Lists to store all results
    results = []

    # Phase A: Warm-start with full-rank IdealizedPreset
    print("="*80)
    print("PHASE A: WARM-START TRAINING (Full-rank IdealizedPreset)")
    print("="*80)
    
    warmup_model = create_warmup_model()
    if USE_CUDA:
        warmup_model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_sgd_optimizer(warmup_model, LR_WARMSTART)
    
    best_warmup_acc = 0
    
    for epoch in range(WARM_EPOCHS):
        epoch_start = time()
        train_loss, train_acc = train_step(train_data, warmup_model, criterion, optimizer)
        val_loss, val_acc = test_evaluation(validation_data, warmup_model, criterion)
        epoch_time = time() - epoch_start
        
        if val_acc > best_warmup_acc:
            best_warmup_acc = val_acc
        
        # Store warmup results
        result_row = {
            'epoch': epoch + 1,
            'phase': 'warmup',
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_accuracy_c_only': None,  # Not applicable for warmup
            'learning_rate': LR_WARMSTART,
            'epoch_time': epoch_time,
            'num_transfers': 0  # No transfers in warmup
        }
        results.append(result_row)
        
        print(f"Warmup Epoch {epoch+1}/{WARM_EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
              f"Time={epoch_time:.1f}s")
    
    print(f"\nWarm-start complete. Best accuracy: {best_warmup_acc:.2f}%")
    
    # Phase B: Replace with LRTT and continue training
    print("\n" + "="*80)
    print("PHASE B: LRTT TRAINING (Low-Rank with automatic transfers)")
    print("="*80)
    
    model = replace_with_lrtt(warmup_model)
    
    # Create new optimizer for LRTT phase with reduced learning rate
    optimizer = create_sgd_optimizer(model, LR_LRTT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS-WARM_EPOCHS, eta_min=1e-4
    )
    
    best_accuracy = best_warmup_acc
    best_c_only = 0
    best_epoch = 0
    
    print(f"\nStarting LRTT training from epoch {WARM_EPOCHS+1}...")
    print("-" * 80)
    
    for epoch in range(WARM_EPOCHS, N_EPOCHS):
        epoch_start = time()
        
        # Train
        train_loss, train_acc = train_step(train_data, model, criterion, optimizer)
        
        # Validate with full model (C + A@B)
        val_loss, val_acc = test_evaluation(validation_data, model, criterion, c_only=False)
        
        # Periodically validate with C-only
        c_only_acc = None
        if (epoch + 1) % VALIDATE_C_ONLY_EVERY == 0 or epoch == WARM_EPOCHS:
            _, c_only_acc = test_evaluation(validation_data, model, criterion, c_only=True)
            if c_only_acc > best_c_only:
                best_c_only = c_only_acc
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch + 1
            save(model.state_dict(), WEIGHT_PATH)
        
        epoch_time = time() - epoch_start
        
        # Get LRTT statistics
        total_transfers = 0
        for module in model.modules():
            if hasattr(module, 'analog_tiles') and hasattr(module.analog_tiles, 'controller'):
                total_transfers += module.analog_tiles.controller.num_transfers
        
        # Store LRTT results
        result_row = {
            'epoch': epoch + 1,
            'phase': 'lrtt',
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_accuracy_c_only': c_only_acc,
            'learning_rate': current_lr,
            'epoch_time': epoch_time,
            'num_transfers': total_transfers
        }
        results.append(result_row)
        
        # Print progress
        lrtt_stats = print_lrtt_statistics(model)
        
        if c_only_acc is not None:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train={train_acc:.2f}%, "
                  f"Val={val_acc:.2f}%, "
                  f"C-only={c_only_acc:.2f}%, "
                  f"Best={best_accuracy:.2f}%, "
                  f"LR={current_lr:.6f}, "
                  f"Time={epoch_time:.1f}s"
                  f"{lrtt_stats}")
        else:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train={train_acc:.2f}%, "
                  f"Val={val_acc:.2f}%, "
                  f"Best={best_accuracy:.2f}%, "
                  f"LR={current_lr:.6f}, "
                  f"Time={epoch_time:.1f}s"
                  f"{lrtt_stats}")
    
    print("-" * 80)
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
    print(f"Best C-only accuracy: {best_c_only:.2f}%")
    print(f"Improvement from warm-start: +{best_accuracy - best_warmup_acc:.2f}%")
    print(f"Model weights saved to: {WEIGHT_PATH}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    model.load_state_dict(torch.load(WEIGHT_PATH))
    
    final_loss_c, final_acc_c = test_evaluation(validation_data, model, criterion, c_only=True)
    final_loss_full, final_acc_full = test_evaluation(validation_data, model, criterion, c_only=False)
    
    print(f"Final C-only accuracy: {final_acc_c:.2f}%")
    print(f"Final Full model accuracy: {final_acc_full:.2f}%")
    print(f"LoRA contribution: +{final_acc_full - final_acc_c:.2f}%")
    print("="*80)
    
    # Add final test results to last row
    if results:
        results[-1]['test_accuracy'] = final_acc_full
        results[-1]['test_accuracy_c_only'] = final_acc_c
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Parameters to save
    params_dict = {
        'seed': SEED,
        'total_epochs': N_EPOCHS,
        'warmup_epochs': WARM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr_warmstart': LR_WARMSTART,
        'lr_lrtt': LR_LRTT,
        'lrtt_rank_conv': LRTT_RANK_CONV,
        'lrtt_rank_fc': LRTT_RANK_FC,
        'transfer_every': TRANSFER_EVERY,
        'lora_alpha': LORA_ALPHA,
        'validate_c_only_every': VALIDATE_C_ONLY_EVERY,
        'best_warmup_accuracy': best_warmup_acc,
        'best_val_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'best_c_only_accuracy': best_c_only,
        'final_test_accuracy': final_acc_full,
        'final_test_accuracy_c_only': final_acc_c,
        'lora_contribution': final_acc_full - final_acc_c,
        'improvement_from_warmup': best_accuracy - best_warmup_acc
    }
    
    save_results_to_excel(results_df, params_dict, EXCEL_PATH)


if __name__ == "__main__":
    main()
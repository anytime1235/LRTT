# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example: ViT with SPT+LSA for CIFAR10 using TTv2 layers.

Based on the paper settings (C.1.4):
- SPT (Shifted Patch Tokenization) + LSA (Locality Self-Attention)
- 4 Transformer blocks
- ~4.3M trainable parameters, 18 linear layers
- All linear and conv layers are analog, normalization layers are FP
- 40 epochs, batch size 8, no image augmentation
- ReduceLROnPlateau scheduler

Paper TTv2 parameters:
- ns = 1, λA = 0.075
- nstates = 200 (dw_min = 0.01)
- γ0 = 10000 (auto_granularity)
- ρ = 0.01 (chopper prob, for c-TTv2 only)

Reference: https://github.com/kentaroy47/vision-transformers-cifar10
"""
# pylint: disable=invalid-name

import os
import math
import json

import torch
from torch import nn, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets, transforms

from tqdm import tqdm
import wandb

from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogLinear, AnalogConv2d
from aihwkit.simulator.configs import UnitCellRPUConfig, MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "VITSPTLSA_TTV2_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "cifar10_vitsptlsa_ttv2_scratch_model_weight.pth")

# Training parameters (from paper)
SEED = 1
N_EPOCHS = 40  # Paper: 40 epochs
BATCH_SIZE = 8  # Paper: batch size 8
LEARNING_RATE = 1e-2  # Initial LR (will be reduced on plateau)
LR_REDUCTION_FACTOR = 0.1  # Paper: reduce LR by 0.1 on plateau
LR_PATIENCE = 5  # Patience for ReduceLROnPlateau
WEIGHT_DECAY = 5e-5
N_CLASSES = 10
NUM_WORKERS = 4
IMAGE_SIZE = 32  # CIFAR-10 native size (no resize for this model)

# ViT model configuration (SPT+LSA from paper)
PATCH_SIZE = 4
EMBED_DIM = 288
DEPTH = 4
NUM_HEADS = 8
MLP_RATIO = 4.0
DROPOUT = 0.0

# TTv2 configuration parameters
N_STATES = 10000  # ~10000 states (idealized)
DW_MIN = 0.0002  # step size for ~10000 states
AUTO_GRANULARITY = 10000  # Paper: γ0 = 10000
FAST_LR = 0.075  # Fast tile learning rate (paper: λA = 0.075)
IN_CHOP_PROB = 0.0  # TTv2: no chopping (c-TTv2 uses 0.01)

# Layer configuration
USE_ANALOG_FOR_ALL_LINEAR = True
USE_ANALOG_FOR_ALL_CONV = True


def create_ttv2_config():
    """Create TTv2 configuration for linear/conv layers (paper settings)."""

    # Paper: nstates = 200 -> dw_min = 0.01
    unit_devices = [
        IdealizedPresetDevice(
            w_max=1.0,
            w_min=-1.0,
            dw_min=DW_MIN,
            dw_min_dtod=0.3,
            dw_min_std=0.3,
            up_down=0.0,
            up_down_dtod=0.0,
        ),
        IdealizedPresetDevice(
            w_max=1.0,
            w_min=-1.0,
            dw_min=DW_MIN,
            dw_min_dtod=0.3,
            dw_min_std=0.3,
            up_down=0.0,
            up_down_dtod=0.0,
        ),
    ]

    # ChoppedTransferCompound for TTv2
    device_config = ChoppedTransferCompound(
        unit_cell_devices=unit_devices,
        transfer_forward=PresetIOParameters(
            noise_management=NoiseManagementType.NONE,
            bound_management=BoundManagementType.NONE
        ),
        transfer_update=PresetUpdateParameters(
            desired_bl=1,
            update_bl_management=False,
            update_management=False
        ),
        units_in_mbatch=False,
        in_chop_prob=IN_CHOP_PROB,  # 0.0 for TTv2, 0.01 for c-TTv2
        fast_lr=FAST_LR,
        auto_scale=True,
        auto_granularity=AUTO_GRANULARITY,
    )

    mapping = MappingParameter(
        weight_scaling_omega=1.0,
        learn_out_scaling=False,
        weight_scaling_lr_compensation=True,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=True,
        max_input_size=1024,
        max_output_size=1024
    )

    forward_io = IOParameters(
        inp_res=0.007937,
        inp_bound=1.0,
        inp_noise=0.0,
        inp_sto_round=False,
        out_res=0.001961,
        out_bound=12.0,
        out_noise=0.06,
        w_noise=0.0,
        w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=False,
        max_bm_factor=1000,
    )

    config = UnitCellRPUConfig(
        device=device_config,
        mapping=mapping,
        forward=forward_io,
        backward=forward_io,
        update=PresetUpdateParameters(desired_bl=31)
    )

    return config


class ShiftedPatchTokenization(nn.Module):
    """Shifted Patch Tokenization (SPT) module."""

    def __init__(self, in_channels=3, embed_dim=256, patch_size=4, use_analog=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels * 5

        # Patch embedding projection - digital
        self.proj = nn.Conv2d(
            self.in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def shift_features(self, x):
        B, C, H, W = x.shape
        shift = self.patch_size // 2

        x_orig = x
        x_tl = F.pad(x, (shift, 0, shift, 0))[:, :, :H, :W]
        x_tr = F.pad(x, (0, shift, shift, 0))[:, :, :H, shift:]
        x_bl = F.pad(x, (shift, 0, 0, shift))[:, :, shift:, :W]
        x_br = F.pad(x, (0, shift, 0, shift))[:, :, shift:, shift:]

        x = torch.cat([x_orig, x_tl, x_tr, x_bl, x_br], dim=1)
        return x

    def forward(self, x):
        x = self.shift_features(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LocalitySelfAttention(nn.Module):
    """Locality Self-Attention (LSA) module."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, use_analog=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * math.sqrt(self.head_dim))

        if use_analog:
            self.qkv = AnalogLinear(
                embed_dim, embed_dim * 3,
                bias=True,
                rpu_config=create_ttv2_config()
            )
            self.proj = AnalogLinear(
                embed_dim, embed_dim,
                bias=True,
                rpu_config=create_ttv2_config()
            )
        else:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.register_buffer('mask', None)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.temperature

        if self.mask is None or self.mask.shape[-1] != N:
            mask = torch.eye(N, device=x.device, dtype=torch.bool)
            self.mask = mask

        attn = attn.masked_fill(self.mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        attn = torch.nan_to_num(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class MLP(nn.Module):
    """MLP block with analog linear layers."""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0, use_analog=True):
        super().__init__()

        if use_analog:
            self.fc1 = AnalogLinear(
                in_features, hidden_features,
                bias=True,
                rpu_config=create_ttv2_config()
            )
            self.fc2 = AnalogLinear(
                hidden_features, out_features,
                bias=True,
                rpu_config=create_ttv2_config()
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with LSA and MLP."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.0, use_analog=True):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.attn = LocalitySelfAttention(
            embed_dim, num_heads, dropout,
            use_analog=use_analog
        )

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            embed_dim, mlp_hidden_dim, embed_dim, dropout,
            use_analog=use_analog
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT_SPT_LSA(nn.Module):
    """Vision Transformer with SPT and LSA for CIFAR-10."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=256, depth=4, num_heads=4,
                 mlp_ratio=2.0, dropout=0.0, use_analog=True):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = ShiftedPatchTokenization(
            in_channels, embed_dim, patch_size,
            use_analog=use_analog
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout,
                use_analog=use_analog
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_linear_layers(model):
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, AnalogLinear, nn.Conv2d, AnalogConv2d)):
            count += 1
    return count


def create_model():
    """Create ViT-SPT-LSA model with TTv2 layers."""

    model = ViT_SPT_LSA(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        num_classes=N_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        use_analog=USE_ANALOG_FOR_ALL_LINEAR
    )

    num_params = count_parameters(model)
    num_linear = count_linear_layers(model)

    print(f"\nCreated ViT-SPT-LSA model with TTv2:")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Num patches: {(IMAGE_SIZE // PATCH_SIZE) ** 2}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Depth: {DEPTH} transformer blocks")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  MLP ratio: {MLP_RATIO}")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Linear/Conv layers: {num_linear}")
    print(f"  Device states: {N_STATES} (dw_min={DW_MIN:.4f})")
    print(f"  Auto granularity: {AUTO_GRANULARITY}")
    print(f"  Fast LR: {FAST_LR}")
    print(f"  Chop prob: {IN_CHOP_PROB} (TTv2)\n")

    return model


def load_images():
    """Load CIFAR-10 images without augmentation."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)

    train_data = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False
    )
    validation_data = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False
    )

    return train_data, validation_data


def create_optimizer(model, learning_rate, weight_decay):
    """Create analog-aware optimizer."""
    optimizer = AnalogSGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
    )
    optimizer.regroup_param_groups(model)
    return optimizer


def test_evaluation(validation_data, model, criterion):
    """Evaluate model on validation set."""
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    pbar = tqdm(validation_data, desc="Validating", leave=False)

    with no_grad():
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch_max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()

            current_acc = 100 * predicted_ok / total_images
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        epoch_loss = total_loss / len(validation_data.dataset)
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

    return model, epoch_loss, error, accuracy


def main():
    """Train ViT-SPT-LSA with TTv2 on CIFAR-10."""
    manual_seed(SEED)

    # Initialize wandb
    wandb.init(
        project="cifar10_vitsptlsa_ttv2_scratch",
        name=f"vitsptlsa_ttv2_bs{BATCH_SIZE}_e{N_EPOCHS}_lr{LEARNING_RATE}_states{N_STATES}",
        config={
            "model": "ViT-SPT-LSA",
            "optimizer": "TTv2",
            "dataset": "CIFAR-10",
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
            "mlp_ratio": MLP_RATIO,
            "dropout": DROPOUT,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lr_reduction_factor": LR_REDUCTION_FACTOR,
            "lr_patience": LR_PATIENCE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            # TTv2 specific
            "n_states": N_STATES,
            "dw_min": DW_MIN,
            "auto_granularity": AUTO_GRANULARITY,
            "fast_lr": FAST_LR,
            "in_chop_prob": IN_CHOP_PROB,
            "use_analog_linear": USE_ANALOG_FOR_ALL_LINEAR,
            "augmentation": False,
            "device": str(DEVICE),
            "use_cuda": USE_CUDA,
        }
    )

    # Load data
    train_data, validation_data = load_images()
    print(f"Training samples: {len(train_data.dataset)}")
    print(f"Validation samples: {len(validation_data.dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {len(train_data)}")

    # Create model
    model = create_model()

    if USE_CUDA:
        model = model.to(DEVICE)
    print(f"Model moved to {DEVICE}")

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_REDUCTION_FACTOR,
        patience=LR_PATIENCE
    )

    best_accuracy = 0
    best_epoch = 0
    epoch_history = []  # Track epoch-wise results for plotting

    print(f"\n{'='*60}")
    print(f"Starting training: {N_EPOCHS} epochs, batch_size={BATCH_SIZE}")
    print(f"LR schedule: ReduceLROnPlateau (factor={LR_REDUCTION_FACTOR}, patience={LR_PATIENCE})")
    print(f"No image augmentation (as per paper)")
    print(f"{'='*60}\n")

    epoch_pbar = tqdm(range(N_EPOCHS), desc="Overall Progress", position=0)

    global_step = 0

    for epoch in epoch_pbar:
        model.train()

        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        batch_pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, (images, labels) in enumerate(batch_pbar):
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

            current_acc = 100 * epoch_correct / epoch_total
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

            global_step += 1

        train_loss = epoch_loss / len(train_data.dataset)
        train_acc = 100 * epoch_correct / epoch_total

        model.eval()
        _, val_loss, val_error, val_accuracy = test_evaluation(validation_data, model, criterion)
        model.train()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc / 100,
            "eval/loss": val_loss,
            "eval/accuracy": val_accuracy / 100,
            "eval/error": val_error,
            "learning_rate": current_lr,
        }, step=global_step)

        # Save epoch history for plotting
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_error": val_error,
            "learning_rate": current_lr,
        })

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            save(model.state_dict(), WEIGHT_PATH)

        epoch_pbar.set_postfix({
            'Train': f'{train_acc:.2f}%',
            'Val': f'{val_accuracy:.2f}%',
            'Best': f'{best_accuracy:.2f}%',
            'LR': f'{current_lr:.2e}'
        })

        if (epoch + 1) % 5 == 0:
            tqdm.write(f"Epoch {epoch + 1:3d}: "
                      f"Train Loss {train_loss:.4f} (Acc {train_acc:.2f}%) | "
                      f"Val Loss {val_loss:.4f} (Acc {val_accuracy:.2f}%) | "
                      f"LR {current_lr:.2e}")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")
    print(f"Best validation error: {100 - best_accuracy:.2f}%")
    print(f"Model weights saved to: {WEIGHT_PATH}")
    print(f"{'='*60}")

    # Save epoch history to JSON for plotting
    history_path = os.path.join(RESULTS, "epoch_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            "method": "TTv2",
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch + 1,
            "history": epoch_history
        }, f, indent=2)
    print(f"Epoch history saved to: {history_path}")

    print(f"\n{'='*60}")
    print("Comparison with paper results:")
    print(f"  Paper FP baseline: 29.3% error")
    print(f"  Paper TTv2 (no noise): 36.1% error")
    print(f"  Paper c-TTv2 (no noise): 35.9% error")
    print(f"  Our TTv2 result: {100 - best_accuracy:.1f}% error")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    main()

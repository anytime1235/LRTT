# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""ViT-Tiny for CIFAR10 using TikiTaka TTv2 (TransferCompound) layers.

Model: ViT-Tiny (standard ViT, no SPT/LSA)
- Standard patch embedding (no shifted patch tokenization)
- Standard multi-head self-attention (no locality self-attention)
- embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0
- patch_size=4 for CIFAR-10 (32x32 -> 64 patches)
- First layer (patch_embed) and last layer (head) are digital
- Transformer blocks use TTv2 analog layers
"""
# pylint: disable=invalid-name

import os
import gc
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

from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.configs import TikiTakaIdealizedPreset
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import SoftBoundsDevice, LinearStepDevice, FloatingPointDevice
from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "VITTINY_TTV2_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "cifar10_vittiny_ttv2_scratch_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 200
BATCH_SIZE = 8  # Larger batch sizes cause CUDA errors with ChoppedTransferCompound
LEARNING_RATE = 1e-2
LR_REDUCTION_FACTOR = 0.1
LR_PATIENCE = 3
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 5e-5
OPTIMIZER = "AnalogSGD"  # "AnalogSGD", "AnalogAdam"
N_CLASSES = 10
NUM_WORKERS = 4
IMAGE_SIZE = 32

# ViT-Tiny model configuration
PATCH_SIZE = 4
EMBED_DIM = 192
DEPTH = 12
NUM_HEADS = 3
MLP_RATIO = 4.0
DROPOUT = 0.0

# TikiTaka TTv2 configuration
TRANSFER_EVERY = 1.0
UNITS_IN_MBATCH = True
FAST_LR = 0.5
AUTO_GRANULARITY = 10000

# Device configuration
DEVICE_A = "6t1c"
DEVICE_C = "softbounds"


def _create_device(device_type):
    """Create device based on type string."""
    if device_type == "6t1c":
        return LinearStepDevice(
            dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
            dw_min_dtod=0.1, up_down_dtod=0.01,
            w_max_dtod=0.05, w_min_dtod=0.05,
            gamma_up_dtod=0.05, gamma_down_dtod=0.05,
            dw_min_std=0.3, write_noise_std=0,
            mean_bound_reference=True,
        )
    elif device_type == "softbounds":
        return SoftBoundsDevice(
            w_max=1.0, w_min=-1.0, w_max_dtod=0.0, w_min_dtod=0.0,
            dw_min=0.001, dw_min_dtod=0.0, dw_min_std=0.0,
            up_down=0.0, up_down_dtod=0.0,
            mult_noise=True, write_noise_std=0.0,
        )
    elif device_type == "floating_point":
        return FloatingPointDevice()
    else:
        return IdealizedPresetDevice(w_max=1.0, w_min=-1.0)


def create_ttv2_config():
    """Create TikiTaka TTv2 configuration (ChoppedTransferCompound with in_chop_prob=0.0)."""
    unit_devices = [_create_device(DEVICE_A), _create_device(DEVICE_C)]

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
        transfer_every=TRANSFER_EVERY,
        units_in_mbatch=UNITS_IN_MBATCH,
        in_chop_prob=0.0,  # TTv2: no chopping
        fast_lr=FAST_LR,
        auto_scale=True,
        auto_granularity=AUTO_GRANULARITY,
    )

    mapping = MappingParameter(
        weight_scaling_omega=1.0, learn_out_scaling=False,
        weight_scaling_lr_compensation=True, digital_bias=True,
        weight_scaling_columnwise=False, out_scaling_columnwise=True,
        max_input_size=1024, max_output_size=1024
    )

    forward_io = IOParameters(
        inp_res=0.007937, inp_bound=1.0, inp_noise=0.0, inp_sto_round=False,
        out_res=0.001961, out_bound=12.0, out_noise=0.06,
        w_noise=0.0, w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=False, max_bm_factor=1000,
    )

    config = TikiTakaIdealizedPreset()
    config.device = device_config
    config.mapping = mapping
    config.forward = forward_io
    config.backward = forward_io
    return config


class PatchEmbedding(nn.Module):
    """Standard patch embedding - Digital."""
    def __init__(self, in_channels=3, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with TTv2 analog layers."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = AnalogLinear(embed_dim, embed_dim * 3, bias=True, rpu_config=create_ttv2_config())
        self.proj = AnalogLinear(embed_dim, embed_dim, bias=True, rpu_config=create_ttv2_config())
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """MLP block with TTv2 analog layers."""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = AnalogLinear(in_features, hidden_features, bias=True, rpu_config=create_ttv2_config())
        self.fc2 = AnalogLinear(hidden_features, out_features, bias=True, rpu_config=create_ttv2_config())
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
    """Transformer block with TTv2 analog layers."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT_Tiny_TTv2(nn.Module):
    """Vision Transformer Tiny with TTv2 analog layers."""
    def __init__(self, image_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

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


def count_analog_layers(model):
    return sum(1 for m in model.modules() if isinstance(m, AnalogLinear))


def create_model():
    """Create ViT-Tiny model with TTv2 analog layers."""
    model = ViT_Tiny_TTv2(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=3,
        num_classes=N_CLASSES, embed_dim=EMBED_DIM, depth=DEPTH,
        num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
    )
    print(f"\nCreated ViT-Tiny (TTv2): {count_parameters(model):,} params, {count_analog_layers(model)} analog layers")
    print(f"  TTv2 config: transfer_every={TRANSFER_EVERY}, fast_lr={FAST_LR}, auto_granularity={AUTO_GRANULARITY}")
    return model


def load_images():
    """Load CIFAR-10 images."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)

    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(SEED)

    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_CUDA, generator=g)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
    return train_data, validation_data


def create_optimizer(model, learning_rate, weight_decay):
    if OPTIMIZER == "AnalogSGD":
        return AnalogSGD(model.parameters(), lr=learning_rate)
    else:
        return AnalogAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def test_evaluation(validation_data, model, criterion):
    total_loss = 0
    predicted_ok = 0
    total_images = 0
    model.eval()

    with no_grad():
        for images, labels in validation_data:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch_max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()

    return model, total_loss / len(validation_data.dataset), predicted_ok / total_images * 100


def main():
    """Train ViT-Tiny with TTv2 on CIFAR-10."""
    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    wandb.init(
        project="cifar10_vittiny_ttv2_scratch",
        name=f"vittiny_ttv2_te{TRANSFER_EVERY}_bs{BATCH_SIZE}",
        config={
            "model": "ViT-Tiny-TTv2", "dataset": "CIFAR-10",
            "embed_dim": EMBED_DIM, "depth": DEPTH, "num_heads": NUM_HEADS,
            "transfer_every": TRANSFER_EVERY, "fast_lr": FAST_LR, "auto_granularity": AUTO_GRANULARITY,
            "device_a": DEVICE_A, "device_c": DEVICE_C,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER, "seed": SEED,
        }
    )

    train_data, validation_data = load_images()
    model = create_model()
    if USE_CUDA:
        model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_REDUCTION_FACTOR, patience=LR_PATIENCE)

    best_accuracy = 0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")

    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for images, labels in train_data:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

        train_loss = epoch_loss / len(train_data.dataset)
        train_acc = 100 * epoch_correct / epoch_total

        model.eval()
        _, val_loss, val_accuracy = test_evaluation(validation_data, model, criterion)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch + 1, "train/loss": train_loss, "train/accuracy": train_acc / 100,
            "eval/loss": val_loss, "eval/accuracy": val_accuracy / 100, "learning_rate": current_lr,
        })

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Val {val_accuracy:.2f}% | Best {best_accuracy:.2f}%")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"\nBest accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")

    # Memory cleanup
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU cache cleared")

    wandb.finish()


if __name__ == "__main__":
    main()

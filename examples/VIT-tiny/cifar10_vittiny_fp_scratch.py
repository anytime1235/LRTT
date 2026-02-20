# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""ViT-Tiny for CIFAR10 using Digital/FP layers.

Digital floating-point baseline for comparison with analog methods.

Model: ViT-Tiny (standard ViT, no SPT/LSA)
- Standard patch embedding (no shifted patch tokenization)
- Standard multi-head self-attention (no locality self-attention)
- embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0
- patch_size=4 for CIFAR-10 (32x32 -> 64 patches)
"""
# pylint: disable=invalid-name

import os
import math
import json
import gc

import torch
from torch import nn, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets, transforms

from tqdm import tqdm
import wandb


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "VITTINY_FP_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "cifar10_vittiny_fp_scratch_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 1e-2
LR_REDUCTION_FACTOR = 0.1
LR_PATIENCE = 3
WEIGHT_DECAY = 5e-5
OPTIMIZER = "SGD"  # "SGD", "Adam"
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

# Freeze option: freeze attention (qkv, proj) and MLP (fc1, fc2) layers
FREEZE_TRANSFORMER = False


def freeze_transformer_layers(model):
    """Freeze transformer block layers (qkv, proj, fc1, fc2).

    Only embeddings and layer norms remain trainable.
    """
    frozen_count = 0
    frozen_params = 0

    for block in model.blocks:
        # Freeze attention: qkv and proj
        for name in ['qkv', 'proj']:
            layer = getattr(block.attn, name)
            for param in layer.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_count += 1

        # Freeze MLP: fc1 and fc2
        for name in ['fc1', 'fc2']:
            layer = getattr(block.mlp, name)
            for param in layer.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_count += 1

    return frozen_count, frozen_params


class PatchEmbedding(nn.Module):
    """Standard patch embedding (Conv2d projection)."""

    def __init__(self, in_channels=3, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

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
    """MLP block with linear layers."""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()

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
    """Transformer block with standard MHSA and MLP."""

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


class ViT_Tiny(nn.Module):
    """Vision Transformer Tiny for CIFAR-10 (Digital/FP version)."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=192, depth=12, num_heads=3,
                 mlp_ratio=4.0, dropout=0.0):
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
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            count += 1
    return count


def create_model():
    """Create ViT-Tiny model with digital FP layers."""

    model = ViT_Tiny(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        num_classes=N_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    )

    # Freeze transformer layers if requested
    if FREEZE_TRANSFORMER:
        frozen_count, frozen_params = freeze_transformer_layers(model)
        print(f"  Frozen {frozen_count} layers ({frozen_params:,} params)")

    num_params = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    num_linear = count_linear_layers(model)

    print(f"\nCreated ViT-Tiny model (Digital FP):")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Num patches: {(IMAGE_SIZE // PATCH_SIZE) ** 2}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Depth: {DEPTH} transformer blocks")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  MLP ratio: {MLP_RATIO}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Linear/Conv layers: {num_linear}")
    print(f"  Transformer frozen: {FREEZE_TRANSFORMER}")
    print(f"  Mode: Full Precision (FP32)\n")

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

    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(SEED)

    train_data = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False,
        generator=g
    )
    validation_data = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False
    )

    return train_data, validation_data


def create_optimizer(model, learning_rate, weight_decay):
    """Create optimizer."""
    if OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    elif OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}. Choose from: SGD, Adam")
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
    """Train ViT-Tiny with Digital FP on CIFAR-10."""
    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    # Initialize wandb
    wandb.init(
        project="cifar10_vittiny_fp_scratch",
        name=f"vittiny_fp_bs{BATCH_SIZE}_e{N_EPOCHS}_lr{LEARNING_RATE}",
        config={
            "model": "ViT-Tiny",
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
            "optimizer": OPTIMIZER,
            "seed": SEED,
            "freeze_transformer": FREEZE_TRANSFORMER,
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
    epoch_history = []

    print(f"\n{'='*60}")
    print(f"Starting training: {N_EPOCHS} epochs, batch_size={BATCH_SIZE}")
    print(f"LR schedule: ReduceLROnPlateau (factor={LR_REDUCTION_FACTOR}, patience={LR_PATIENCE})")
    print(f"No image augmentation")
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

    history_path = os.path.join(RESULTS, "epoch_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            "method": "FP",
            "model": "ViT-Tiny",
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch + 1,
            "freeze_transformer": FREEZE_TRANSFORMER,
            "history": epoch_history
        }, f, indent=2)
    print(f"Epoch history saved to: {history_path}")

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

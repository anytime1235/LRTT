# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ViT-SPT-LSA with LRTT on CIFAR-10.

Usage:
    # Single process
    python optuna_vitsptlsa_lrtt.py --n-trials 50

    # Parallel execution (run in multiple terminals)
    python optuna_vitsptlsa_lrtt.py --study-name my_sweep --n-trials 20 &
    python optuna_vitsptlsa_lrtt.py --study-name my_sweep --n-trials 20 &
    python optuna_vitsptlsa_lrtt.py --study-name my_sweep --n-trials 20 &
"""

import os
import math
import json
import argparse
from datetime import datetime

import torch
from torch import nn, device, no_grad, manual_seed
from torch import max as torch_max
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets, transforms
from tqdm import tqdm

import optuna
from optuna.trial import TrialState

from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import WeightNoiseType, BoundManagementType, NoiseManagementType
from aihwkit.simulator.presets.devices import IdealizedPresetDevice

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Fixed parameters
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
RESULTS = os.path.join(os.getcwd(), "results", "optuna_vitsptlsa_lrtt")
os.makedirs(RESULTS, exist_ok=True)

N_CLASSES = 10
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

# Fixed model architecture (from paper)
EMBED_DIM = 288
DEPTH = 4
NUM_HEADS = 8
MLP_RATIO = 4.0
DROPOUT = 0.0


def create_lrtt_config(rank, transfer_every, lora_alpha, transfer_lr_scale=1.0, dw_min=0.0002, dw_min_dtod=0.3, dw_min_std=0.3):
    """Create LRTT configuration with given hyperparameters."""
    unit_devices = [
        IdealizedPresetDevice(
            w_max=1.0, w_min=-1.0,
            dw_min=dw_min,
            dw_min_dtod=dw_min_dtod,
            dw_min_std=dw_min_std,
            up_down=0.0, up_down_dtod=0.0,
        ),
        IdealizedPresetDevice(
            w_max=1.0, w_min=-1.0,
            dw_min=dw_min,
            dw_min_dtod=dw_min_dtod,
            dw_min_std=dw_min_std,
            up_down=0.0, up_down_dtod=0.0,
        ),
        IdealizedPresetDevice(
            w_max=1.0, w_min=-1.0,
            dw_min=dw_min,
            dw_min_dtod=dw_min_dtod,
            dw_min_std=dw_min_std,
            up_down=0.0, up_down_dtod=0.0,
        ),
    ]

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        transfer_lr_scale=transfer_lr_scale,
        forward_inject=False,
        unit_cell_devices=unit_devices
    )
    device_config.transfer_lr = lora_alpha

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

    return PythonLRTTRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io)


# Global config holder for model creation
_current_config = {}


def get_current_lrtt_config():
    """Get current LRTT config from global holder."""
    return create_lrtt_config(
        rank=_current_config['rank'],
        transfer_every=_current_config['transfer_every'],
        lora_alpha=_current_config['lora_alpha'],
        transfer_lr_scale=_current_config.get('transfer_lr_scale', 1.0),
        dw_min=_current_config.get('dw_min', 0.0002),
        dw_min_dtod=_current_config.get('dw_min_dtod', 0.3),
        dw_min_std=_current_config.get('dw_min_std', 0.3),
    )


# Import model components (simplified inline versions)
import torch.nn.functional as F


class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels * 5
        self.proj = nn.Conv2d(self.in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def shift_features(self, x):
        B, C, H, W = x.shape
        shift = self.patch_size // 2
        x_orig = x
        x_tl = F.pad(x, (shift, 0, shift, 0))[:, :, :H, :W]
        x_tr = F.pad(x, (0, shift, shift, 0))[:, :, :H, shift:]
        x_bl = F.pad(x, (shift, 0, 0, shift))[:, :, shift:, :W]
        x_br = F.pad(x, (0, shift, 0, shift))[:, :, shift:, shift:]
        return torch.cat([x_orig, x_tl, x_tr, x_bl, x_br], dim=1)

    def forward(self, x):
        x = self.shift_features(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LocalitySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * math.sqrt(self.head_dim))
        self.qkv = AnalogLinear(embed_dim, embed_dim * 3, bias=True, rpu_config=get_current_lrtt_config())
        self.proj = AnalogLinear(embed_dim, embed_dim, bias=True, rpu_config=get_current_lrtt_config())
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
            self.mask = torch.eye(N, device=x.device, dtype=torch.bool)
        attn = attn.masked_fill(self.mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        attn = torch.nan_to_num(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = AnalogLinear(in_features, hidden_features, bias=True, rpu_config=get_current_lrtt_config())
        self.fc2 = AnalogLinear(hidden_features, out_features, bias=True, rpu_config=get_current_lrtt_config())
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
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = LocalitySelfAttention(embed_dim, num_heads, dropout)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT_SPT_LSA(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.patch_embed = ShiftedPatchTokenization(in_channels, embed_dim, patch_size)
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


def load_data(batch_size):
    """Load CIFAR-10 data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
    return train_loader, val_loader


def evaluate(model, val_loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch_max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, 100.0 * correct / total


def objective(trial):
    """Optuna objective function."""
    global _current_config

    # Hyperparameters to tune
    rank_exp = trial.suggest_int('rank_exp', 0, 7)  # 2^0 ~ 2^7
    rank = 2 ** rank_exp  # 1, 2, 4, 8, 16, 32, 64, 128
    transfer_every = trial.suggest_int('transfer_every', 1, 50000, log=True)
    lora_alpha = trial.suggest_float('lora_alpha', 0., 10.0, log=True)
    transfer_lr_scale = trial.suggest_float('transfer_lr_scale', 0.1, 10.0, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e0, log=True)
    batch_size = 8  # Fixed
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)

    # Early stopping settings (no max epoch limit)
    max_epochs = 200  # Safety limit
    early_stop_patience = 7  # Stop if no improvement for N epochs

    # Set current config
    _current_config = {
        'rank': rank,
        'transfer_every': transfer_every,
        'lora_alpha': lora_alpha,
        'transfer_lr_scale': transfer_lr_scale,
    }

    manual_seed(SEED)

    # Load data
    train_loader, val_loader = load_data(batch_size)

    # Create model
    model = ViT_SPT_LSA(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=N_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Optimizer and scheduler
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    optimizer.regroup_param_groups(model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        # Check improvement
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Report intermediate value
        trial.report(val_accuracy, epoch)

        # Early stopping if no improvement
        if epochs_without_improvement >= early_stop_patience:
            break

        # Prune if not promising (Optuna's pruning)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_accuracy


def main():
    """Run Optuna hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="Optuna sweep for ViT-SPT-LSA LRTT")
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (same name = shared study for parallel execution)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials to run (default: 50)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (default: None)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database path (default: auto-generated)')
    args = parser.parse_args()

    # Generate or use provided study name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.study_name:
        study_name = args.study_name
        # Use fixed storage for shared study
        storage = args.storage or f"sqlite:///{RESULTS}/optuna_{study_name}.db"
    else:
        study_name = f"vitsptlsa_lrtt_{timestamp}"
        storage = args.storage or f"sqlite:///{RESULTS}/optuna_{timestamp}.db"

    # load_if_exists=True allows multiple workers to share the same study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        load_if_exists=True,  # Enable parallel execution
    )

    print(f"Starting Optuna study: {study_name}")
    print(f"Database: {storage}")
    print(f"Device: {DEVICE}")
    print(f"Trials: {args.n_trials}")
    print(f"(Run multiple instances with same --study-name for parallel execution)")

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 60)
    print("OPTUNA STUDY COMPLETED")
    print("=" * 60)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"  Pruned: {len(pruned_trials)}")
    print(f"  Complete: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (accuracy): {trial.value:.2f}%")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save results
    results_path = os.path.join(RESULTS, f"best_params_{study_name}.json")
    with open(results_path, 'w') as f:
        json.dump({
            "best_accuracy": trial.value,
            "best_params": trial.params,
            "n_trials": len(study.trials),
            "n_pruned": len(pruned_trials),
            "n_complete": len(complete_trials),
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

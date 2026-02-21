# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ViT-Tiny with LRTT on input layer only.

This version:
- Applies LRTT to patch embedding layer (first input layer) only
- Freezes transformer layers: qkv, proj, fc1, fc2
- Trainable: patch_embed (LRTT), cls_token, pos_embed, layer norms, classification head

Usage:
    # Run with SGD, all tuning disabled, batch size 128
    python optuna_vittiny_lrtt_inputonly.py --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --batch-size 128 --n-trials 50

    # Run trials (results are automatically saved and can be resumed)
    python optuna_vittiny_lrtt_inputonly.py --n-trials 50

    # Visualize results without running new trials
    python optuna_vittiny_lrtt_inputonly.py --visualize

Results are stored in:
    - SQLite DB: results/optuna_vittiny_lrtt_inputonly/optuna_<study_name>.db
    - JSON summary: results/optuna_vittiny_lrtt_inputonly/best_params_*.json
    - Visualization: results/optuna_vittiny_lrtt_inputonly/visualization_*.png
"""

import os
import math
import json
import argparse
import gc

import torch
from torch import nn, device, no_grad, manual_seed
from torch import max as torch_max
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets, transforms
from tqdm import tqdm

import optuna
from optuna.trial import TrialState
from optuna_integration import BoTorchSampler
import matplotlib.pyplot as plt

from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn import AnalogConv2d
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.devices import SoftBoundsDevice, LinearStepDevice, FloatingPointDevice


class ConfigAwareBoTorchSampler(BoTorchSampler):
    """BoTorchSampler that respects OPT_CONFIG for fixed parameters."""
    def sample_relative(self, study, trial, search_space):
        params = super().sample_relative(study, trial, search_space)
        # Force reinit_mode if fixed in config
        if OPT_CONFIG['reinit_mode'] is not None and 'reinit_mode' in params:
            params['reinit_mode'] = OPT_CONFIG['reinit_mode']
        # Force optimizer if fixed in config
        if 'optimizer' in params:
            params['optimizer'] = OPT_CONFIG['optimizer']
        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name == 'reinit_mode' and OPT_CONFIG['reinit_mode'] is not None:
            return OPT_CONFIG['reinit_mode']
        if param_name == 'optimizer':
            return OPT_CONFIG['optimizer']
        return super().sample_independent(study, trial, param_name, param_distribution)


DEFAULT_STUDY_NAME = "vittiny_lrtt_inputonly_main"

USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
RESULTS = os.path.join(os.getcwd(), "results", "optuna_vittiny_lrtt_inputonly")
os.makedirs(RESULTS, exist_ok=True)

N_CLASSES = 10
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

# ViT-Tiny architecture
EMBED_DIM = 192
DEPTH = 12
NUM_HEADS = 3
MLP_RATIO = 4.0
DROPOUT = 0.0

# Fixed batch size (can be overridden by --batch-size)
BATCH_SIZE = 128

# Global configuration (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogSGD',
    'tune_wd': True,
    'tune_momentum': True,
    'tune_nesterov': True,
    'reinit_mode': None,  # None = tune, or 'standard'/'decay'/'hybrid' = fixed
}


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config."""
    opt = OPT_CONFIG['optimizer'].lower().replace('analog', '')
    suffix = opt

    # Add reinit_mode if fixed
    if OPT_CONFIG['reinit_mode'] is not None:
        suffix += f"_{OPT_CONFIG['reinit_mode']}"

    if opt == 'sgd':
        if not OPT_CONFIG['tune_wd']:
            suffix += "_nowd"
        if not OPT_CONFIG['tune_momentum']:
            suffix += "_nomom"
        if not OPT_CONFIG['tune_nesterov']:
            suffix += "_nonest"
    else:  # adam
        if not OPT_CONFIG['tune_wd']:
            suffix += "_nowd"

    return suffix


def _create_6t1c_device(tau_sec=46505.0, dt_batch_sec=1.0):
    delta = 1 - math.exp(-dt_batch_sec / tau_sec)
    lifetime = 1.0 / delta
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3, write_noise_std=0,
        mean_bound_reference=True, lifetime=lifetime, lifetime_dtod=0.1, reset=0.0, reset_dtod=0.0,
    )


def create_lrtt_config(rank, transfer_every, lora_alpha, transfer_lr_scale, reinit_mode, tau_sec):
    ab_device = _create_6t1c_device(tau_sec=tau_sec)
    c_device = SoftBoundsDevice(
        w_max=1.0, w_min=-1.0, w_max_dtod=0.0, w_min_dtod=0.0,
        dw_min=0.001, dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0, mult_noise=True, write_noise_std=0.0,
    )
    unit_devices = [ab_device, ab_device, c_device]

    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=transfer_every, lora_alpha=lora_alpha,
        transfer_lr_scale=transfer_lr_scale, forward_inject=False,
        reinit_mode=reinit_mode,
        unit_cell_devices=unit_devices
    )
    device_config.transfer_lr = lora_alpha

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

    return PythonLRTTRPUConfig(device=device_config, mapping=mapping, forward=forward_io, backward=forward_io)


class AnalogPatchEmbedding(nn.Module):
    """Patch embedding using AnalogConv2d with LRTT."""
    def __init__(self, in_channels=3, embed_dim=192, patch_size=4, rpu_config=None):
        super().__init__()
        self.proj = AnalogConv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            bias=True, rpu_config=rpu_config
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention (frozen, no analog)."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_dropout(self.proj(x))


class MLP(nn.Module):
    """Standard MLP (frozen, no analog)."""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        return self.dropout(self.fc2(x))


class TransformerBlock(nn.Module):
    """Transformer block with standard (non-analog) layers."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class ViT_Tiny_LRTT_InputOnly(nn.Module):
    """ViT-Tiny with LRTT on input layer only.

    - patch_embed: AnalogConv2d with LRTT (trainable)
    - transformer blocks: standard nn.Linear (frozen)
    - cls_token, pos_embed, norms, head: trainable
    """
    def __init__(self, rpu_config, image_size=32, patch_size=4, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # Analog patch embedding with LRTT
        self.patch_embed = AnalogPatchEmbedding(3, embed_dim, patch_size, rpu_config)

        # Positional embeddings (trainable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks (will be frozen)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output layers (trainable)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_dropout(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def freeze_transformer_layers(model):
    """Freeze transformer block layers (qkv, proj, fc1, fc2).

    Only patch_embed (LRTT), cls_token, pos_embed, layer norms, and head remain trainable.
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


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_CUDA),
            DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=USE_CUDA))


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch_max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / total, 100.0 * correct / total


def objective(trial):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # LRTT Hyperparameters (for patch_embed only)
    rank_exp = trial.suggest_int('rank_exp', 0, 5)  # max 2^5=32 < min(192,48)=48
    rank = 2 ** rank_exp
    transfer_every = trial.suggest_int('transfer_every', 1, 5000, log=True)
    lora_alpha = trial.suggest_float('lora_alpha', 0.1, 10.0, log=True)
    transfer_lr_scale = trial.suggest_float('transfer_lr_scale', 1.0, 1.0, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e0, log=True)
    batch_size = BATCH_SIZE  # Fixed batch size from command line
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    tau_sec = trial.suggest_float('tau_sec', 1.0, 1e9, log=True)

    # reinit_mode: use fixed value if set, otherwise tune
    if OPT_CONFIG['reinit_mode'] is not None:
        reinit_mode = OPT_CONFIG['reinit_mode']
    else:
        reinit_mode = trial.suggest_categorical('reinit_mode', ['standard', 'decay', 'hybrid'])

    # optimizer: always use config value
    optimizer_name = OPT_CONFIG['optimizer']

    max_epochs = 2000
    early_stop_patience = 7

    manual_seed(SEED)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting (LRTT Input Only)")
    print(f"{'='*70}")
    print(f"  rank={rank}, transfer_every={transfer_every}, lora_alpha={lora_alpha:.2e}")
    print(f"  transfer_lr_scale={transfer_lr_scale:.4f}, lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  tau_sec={tau_sec:.1f}, reinit_mode={reinit_mode}, optimizer={optimizer_name}")
    print(f"  batch_size={batch_size}")
    print(f"  LRTT: patch_embed only | Frozen: qkv, proj, fc1, fc2")
    print(f"{'='*70}")

    train_loader, val_loader = load_data(batch_size)

    model = None
    try:
        rpu_config = create_lrtt_config(rank, transfer_every, lora_alpha, transfer_lr_scale, reinit_mode, tau_sec)
        model = ViT_Tiny_LRTT_InputOnly(
            rpu_config, IMAGE_SIZE, PATCH_SIZE, N_CLASSES,
            EMBED_DIM, DEPTH, NUM_HEADS, MLP_RATIO, DROPOUT
        ).to(DEVICE)

        # Freeze transformer layers
        frozen_count, frozen_params = freeze_transformer_layers(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Frozen {frozen_count} layers ({frozen_params:,} params)")
        print(f"  Params: total={total_params:,}, trainable={trainable_params:,}")

        if optimizer_name == "AnalogAdam":
            optimizer = AnalogAdam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = AnalogSGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            val_loss, val_accuracy = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)

            improved = ""
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                epochs_without_improvement = 0
                improved = " ★"
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Trial {trial.number}] Epoch {epoch+1:3d} | "
                  f"Val Acc: {val_accuracy:6.2f}% | Best: {best_accuracy:6.2f}% | "
                  f"Loss: {val_loss:.4f} | LR: {current_lr:.2e} | "
                  f"No imp: {epochs_without_improvement}/{early_stop_patience}{improved}")

            trial.report(val_accuracy, epoch)

            if epochs_without_improvement >= early_stop_patience:
                break

            if trial.should_prune():
                print(f"[Trial {trial.number}] Pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()

        print(f"\n[Trial {trial.number}] Finished - Best Accuracy: {best_accuracy:.2f}% (Epoch {epoch+1})")
        print(f"{'='*70}\n")
        return best_accuracy

    finally:
        # Delete training loop variables that hold references to model/tensors
        try:
            del loss
        except NameError:
            pass
        try:
            del images
        except NameError:
            pass
        try:
            del labels
        except NameError:
            pass
        # Delete in reverse dependency order: scheduler -> optimizer -> model
        # optimizer holds references to analog tiles via param_groups
        if 'scheduler' in dir():
            del scheduler
        if 'optimizer' in dir():
            del optimizer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        tqdm.write(f"[Trial {trial.number}] GPU cache cleared")


def visualize_study(study, save_dir):
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    trial_numbers = [t.number for t in complete_trials]
    accuracies = [t.value for t in complete_trials]

    axes[0].scatter(trial_numbers, accuracies, alpha=0.6)
    axes[0].plot(trial_numbers, [max(accuracies[:i+1]) for i in range(len(accuracies))], 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Optimization History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    try:
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys())[::-1], list(importances.values())[::-1])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Parameter Importance')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Not enough trials', ha='center', va='center', transform=axes[1].transAxes)

    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, accuracies, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Learning Rate vs Accuracy')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualization.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved.")


def print_study_summary(study):
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        accuracies = [t.value for t in complete_trials]
        print(f"Best: {max(accuracies):.2f}%, Mean: {sum(accuracies)/len(accuracies):.2f}%")
        print(f"Best params: {study.best_params}")


def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for ViT-Tiny LRTT (Input Only)")
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AnalogSGD', choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer type (default: AnalogSGD)')
    parser.add_argument('--no-wd', action='store_true',
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--no-momentum', action='store_true',
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--no-nesterov', action='store_true',
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--reinit-mode', type=str, default=None,
                        choices=['standard', 'decay', 'hybrid'],
                        help='Fix reinit mode (default: tune all three)')
    args = parser.parse_args()

    # Debug: print parsed args
    print(f"DEBUG args: no_wd={args.no_wd}, no_momentum={args.no_momentum}, no_nesterov={args.no_nesterov}, reinit_mode={args.reinit_mode}")

    # Update global config
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    OPT_CONFIG['reinit_mode'] = args.reinit_mode

    # Debug: print OPT_CONFIG
    print(f"DEBUG OPT_CONFIG: {OPT_CONFIG}")

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"vittiny_lrtt_inputonly_bs{BATCH_SIZE}_{get_study_name_suffix()}"
    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=ConfigAwareBoTorchSampler(), pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    print(f"\n{'='*60}")
    print(f"Study: {study_name}")
    print(f"Database: {storage}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"New trials: {args.n_trials}")
    print(f"LRTT: patch_embed only | Frozen: qkv, proj, fc1, fc2")
    print(f"{'='*60}")

    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,), show_progress_bar=False)

    print_study_summary(study)
    visualize_study(study, RESULTS)

    if study.best_trial:
        with open(os.path.join(RESULTS, f"best_params_{study_name}.json"), 'w') as f:
            json.dump({"best_accuracy": study.best_value, "best_params": study.best_params}, f, indent=2)


if __name__ == "__main__":
    main()

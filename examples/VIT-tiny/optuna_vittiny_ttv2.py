# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ViT-Tiny with TTv2 on CIFAR-10.

Usage:
    python optuna_vittiny_ttv2.py --n-trials 50
    python optuna_vittiny_ttv2.py --visualize
"""

import os
import json
import argparse
import gc

import torch
from torch import nn, device, no_grad, manual_seed
from torch import max as torch_max
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import datasets, transforms

import optuna
from optuna.trial import TrialState
from optuna_integration import BoTorchSampler

# Contextual dynamic-space GP: keeps the GP fitting all completed trials across
# mid-study suggest-range edits. See examples/optuna_contextual_sampler.py.
import os.path as _osp, sys as _sys
for _p in (_osp.dirname(_osp.abspath(__file__)),
           _osp.join(_osp.dirname(_osp.abspath(__file__)), '..')):
    if _osp.isfile(_osp.join(_p, 'optuna_contextual_sampler.py')) and _p not in _sys.path:
        _sys.path.insert(0, _p)
from optuna_contextual_sampler import ContextualBoTorchMixin, ContextualBoTorchSampler
import matplotlib.pyplot as plt

from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.configs import TikiTakaIdealizedPreset
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import SoftBoundsDevice, LinearStepDevice
from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters


class SGDOnlyBoTorchSampler(ContextualBoTorchMixin, BoTorchSampler):
    """BoTorchSampler that forces optimizer to 'AnalogSGD'."""
    def sample_relative(self, study, trial, search_space):
        params = super().sample_relative(study, trial, search_space)
        if 'optimizer' in params:
            params['optimizer'] = 'AnalogSGD'
        return self._postprocess(params)

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name == 'optimizer':
            return 'AnalogSGD'
        return super().sample_independent(study, trial, param_name, param_distribution)


DEFAULT_STUDY_NAME = "vittiny_ttv2_main"

USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
RESULTS = os.path.join(os.getcwd(), "results", "optuna_vittiny_ttv2")
os.makedirs(RESULTS, exist_ok=True)

N_CLASSES = 10
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

EMBED_DIM = 192
DEPTH = 12
NUM_HEADS = 3
MLP_RATIO = 4.0
DROPOUT = 0.0

# Fixed batch size for study naming
BATCH_SIZE = 64

# Global configuration (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogSGD',
    'tune_wd': True,
    'tune_momentum': True,
    'tune_nesterov': True,
}


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config."""
    opt = OPT_CONFIG['optimizer'].lower().replace('analog', '')
    suffix = opt

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


def _create_device(device_type):
    if device_type == "6t1c":
        return LinearStepDevice(
            dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
            dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
            gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3, write_noise_std=0,
            mean_bound_reference=True,
        )
    elif device_type == "softbounds":
        return SoftBoundsDevice(
            w_max=1.0, w_min=-1.0, w_max_dtod=0.0, w_min_dtod=0.0,
            dw_min=0.001, dw_min_dtod=0.0, dw_min_std=0.0,
            up_down=0.0, up_down_dtod=0.0, mult_noise=True, write_noise_std=0.0,
        )
    else:
        return SoftBoundsDevice(w_max=1.0, w_min=-1.0)


def create_ttv2_config(transfer_every, fast_lr, auto_granularity=10000):
    unit_devices = [_create_device("6t1c"), _create_device("softbounds")]

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
        transfer_every=transfer_every,
        units_in_mbatch=True,
        in_chop_prob=0.0,  # TTv2: no chopping
        fast_lr=fast_lr,
        auto_scale=True,
        auto_granularity=auto_granularity,
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
    def __init__(self, in_channels=3, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, rpu_config):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = AnalogLinear(embed_dim, embed_dim * 3, bias=True, rpu_config=rpu_config)
        self.proj = AnalogLinear(embed_dim, embed_dim, bias=True, rpu_config=rpu_config)
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
    def __init__(self, in_features, hidden_features, out_features, dropout, rpu_config):
        super().__init__()
        self.fc1 = AnalogLinear(in_features, hidden_features, bias=True, rpu_config=rpu_config)
        self.fc2 = AnalogLinear(hidden_features, out_features, bias=True, rpu_config=rpu_config)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, rpu_config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, rpu_config)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout, rpu_config)

    def forward(self, x):
        return x + self.mlp(self.ln_2(x + self.attn(self.ln_1(x))))


class ViT_Tiny_TTv2(nn.Module):
    def __init__(self, rpu_config, image_size=32, patch_size=4, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(3, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, rpu_config)
            for _ in range(depth)
        ])
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

    transfer_every = trial.suggest_int('transfer_every', 1, 300000, log=True)
    fast_lr = trial.suggest_float('fast_lr', 0.1, 1.0, log=True)
    auto_granularity = 10000  # Fixed
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e0, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 8, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AnalogSGD', 'AnalogAdam'])

    max_epochs = 2000
    early_stop_patience = 7

    manual_seed(SEED)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting")
    print(f"{'='*70}")
    print(f"  transfer_every={transfer_every}, fast_lr={fast_lr:.4f}")
    print(f"  lr={learning_rate:.2e}, wd={weight_decay:.2e}, optimizer={optimizer_name}")
    print(f"{'='*70}")

    train_loader, val_loader = load_data(batch_size)

    model = None
    try:
        rpu_config = create_ttv2_config(transfer_every, fast_lr)
        model = ViT_Tiny_TTv2(
            rpu_config, IMAGE_SIZE, PATCH_SIZE, N_CLASSES,
            EMBED_DIM, DEPTH, NUM_HEADS, MLP_RATIO, DROPOUT
        ).to(DEVICE)

        if optimizer_name == "AnalogAdam":
            optimizer = AnalogAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = AnalogSGD(model.parameters(), lr=learning_rate)

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
        print(f"[Trial {trial.number}] GPU cache cleared")


def visualize_study(study, save_dir):
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    trial_numbers = [t.number for t in complete_trials]
    accuracies = [t.value for t in complete_trials]

    axes[0].scatter(trial_numbers, accuracies, alpha=0.6)
    axes[0].plot(trial_numbers, [max(accuracies[:i+1]) for i in range(len(accuracies))], 'r-', linewidth=2)
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Optimization History')

    try:
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys())[::-1], list(importances.values())[::-1])
        axes[1].set_title('Parameter Importance')
    except:
        pass

    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, accuracies, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('LR vs Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualization.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for ViT-Tiny TTv2")
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
    args = parser.parse_args()

    # Update global config
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"vittiny_ttv2_bs{BATCH_SIZE}_{get_study_name_suffix()}"
    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        visualize_study(study, RESULTS)
        return

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=SGDOnlyBoTorchSampler(consider_running_trials=True), pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,), show_progress_bar=False)

    if study.best_trial:
        with open(os.path.join(RESULTS, f"best_params_{study_name}.json"), 'w') as f:
            json.dump({"best_accuracy": study.best_value, "best_params": study.best_params}, f, indent=2)


if __name__ == "__main__":
    main()

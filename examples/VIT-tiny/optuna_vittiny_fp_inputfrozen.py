# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ViT-Tiny with Digital FP on CIFAR-10 (Frozen layers).

This version freezes:
- Transformer layers: qkv, proj, fc1, fc2
- Patch embedding layer (first input layer)

Only trainable: cls_token, pos_embed, layer norms, classification head

Usage:
    # Run trials with SGD, all tuning disabled, batch size 128
    python optuna_vittiny_fp_inputfrozen.py --optimizer sgd --no-wd --no-momentum --no-nesterov --batch-size 128 --n-trials 50

    # Run trials (results are automatically saved and can be resumed)
    python optuna_vittiny_fp_inputfrozen.py --n-trials 50

    # Resume existing study and add more trials
    python optuna_vittiny_fp_inputfrozen.py --n-trials 30  # continues from saved results

    # Parallel execution (run in multiple terminals, same study name)
    python optuna_vittiny_fp_inputfrozen.py --n-trials 20 &
    python optuna_vittiny_fp_inputfrozen.py --n-trials 20 &
    python optuna_vittiny_fp_inputfrozen.py --n-trials 20 &

    # Visualize results without running new trials
    python optuna_vittiny_fp_inputfrozen.py --visualize

    # Use a different study name (for separate experiments)
    python optuna_vittiny_fp_inputfrozen.py --n-trials 50 --study-name vittiny_fp_inputfrozen_custom

    # Real-time dashboard (install: pip install optuna-dashboard)
    optuna-dashboard sqlite:///results/optuna_vittiny_fp_inputfrozen/optuna_vittiny_fp_inputfrozen_main.db

    # Reset study (delete DB to start fresh)
    rm results/optuna_vittiny_fp_inputfrozen/optuna_vittiny_fp_inputfrozen_main.db

Results are stored in:
    - SQLite DB: results/optuna_vittiny_fp_inputfrozen/optuna_<study_name>.db
    - JSON summary: results/optuna_vittiny_fp_inputfrozen/best_params_*.json
    - Visualization: results/optuna_vittiny_fp_inputfrozen/visualization_*.png
"""

import os
import math
import json
import argparse
import gc
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
from optuna_integration import BoTorchSampler
import matplotlib.pyplot as plt

# Default study name for persistence
DEFAULT_STUDY_NAME = "vittiny_fp_inputfrozen_main"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Fixed parameters
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
RESULTS = os.path.join(os.getcwd(), "results", "optuna_vittiny_fp_inputfrozen")
os.makedirs(RESULTS, exist_ok=True)

N_CLASSES = 10
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

# Fixed model architecture (ViT-Tiny)
EMBED_DIM = 192
DEPTH = 12
NUM_HEADS = 3
MLP_RATIO = 4.0
DROPOUT = 0.0

# Fixed batch size (can be overridden by --batch-size)
BATCH_SIZE = 128

# Global configuration (set by argparse)
OPT_CONFIG = {
    'optimizer': 'sgd',
    'tune_wd': True,
    'tune_momentum': True,
    'tune_nesterov': True,
}


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config."""
    opt = OPT_CONFIG['optimizer']
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


def freeze_layers(model):
    """Freeze transformer block layers (qkv, proj, fc1, fc2) and patch embedding.

    Only cls_token, pos_embed, layer norms, and classification head remain trainable.
    """
    frozen_count = 0
    frozen_params = 0

    # Freeze patch embedding (first input layer)
    for param in model.patch_embed.parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    frozen_count += 1
    print(f"  Frozen: patch_embed (Conv2d input layer)")

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


import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e0, log=True)
    batch_size = BATCH_SIZE  # Fixed batch size from command line

    # Weight decay based on config
    if OPT_CONFIG['tune_wd']:
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    else:
        weight_decay = 0

    # Optimizer selection
    opt_type = OPT_CONFIG['optimizer']

    # SGD-specific: momentum and nesterov
    if opt_type == 'sgd':
        if OPT_CONFIG['tune_momentum']:
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
        else:
            momentum = 0

        if OPT_CONFIG['tune_nesterov'] and momentum > 0:
            nesterov = trial.suggest_categorical('nesterov', [True, False])
        else:
            nesterov = False
    else:
        momentum = 0
        nesterov = False

    max_epochs = 2000
    early_stop_patience = 7

    manual_seed(SEED)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting (Digital FP - FROZEN)")
    print(f"{'='*70}")
    print(f"  optimizer={opt_type}, lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, batch_size={batch_size}")
    print(f"  Frozen: patch_embed + qkv + proj + fc1 + fc2")
    print(f"{'='*70}")

    train_loader, val_loader = load_data(batch_size)

    model = None
    try:
        model = ViT_Tiny(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_classes=N_CLASSES,
            embed_dim=EMBED_DIM,
            depth=DEPTH,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO,
            dropout=DROPOUT,
        ).to(DEVICE)

        # Freeze transformer layers and patch embedding
        frozen_count, frozen_params = freeze_layers(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Frozen {frozen_count} layers ({frozen_params:,} params)")
        print(f"  Params: total={total_params:,}, trainable={trainable_params:,}")

        if opt_type == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=learning_rate, weight_decay=weight_decay)
        else:  # sgd
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=learning_rate, weight_decay=weight_decay,
                                        momentum=momentum, nesterov=nesterov)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
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
        if 'optimizer' in dir():
            del optimizer
        if 'scheduler' in dir():
            del scheduler
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[Trial {trial.number}] GPU cache cleared")


def visualize_study(study, save_dir):
    """Generate visualization plots for the study."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(complete_trials) == 0:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    trial_numbers = [t.number for t in complete_trials]
    accuracies = [t.value for t in complete_trials]
    ax.scatter(trial_numbers, accuracies, alpha=0.6)
    ax.plot(trial_numbers, [max(accuracies[:i+1]) for i in range(len(accuracies))],
            'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Optimization History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    try:
        importances = optuna.importance.get_param_importances(study)
        param_names = list(importances.keys())
        values = list(importances.values())
        ax.barh(param_names[::-1], values[::-1])
        ax.set_xlabel('Importance (fANOVA)')
        ax.set_title('Parameter Importance')
    except Exception as e:
        ax.text(0.5, 0.5, f'Not enough trials\n({e})', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Parameter Importance (unavailable)')

    ax = axes[2]
    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    ax.scatter(lrs, accuracies, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Learning Rate vs Accuracy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "visualization.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_path}")
    plt.close()

    all_trials_data = []
    for t in study.trials:
        trial_data = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            "duration_seconds": (t.datetime_complete - t.datetime_start).total_seconds()
                               if t.datetime_complete and t.datetime_start else None,
        }
        all_trials_data.append(trial_data)

    history_path = os.path.join(save_dir, "all_trials.json")
    with open(history_path, 'w') as f:
        json.dump({
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number if study.best_trial else None,
            "best_value": study.best_value if study.best_trial else None,
            "best_params": study.best_params if study.best_trial else None,
            "trials": all_trials_data,
        }, f, indent=2)
    print(f"Trial history saved to: {history_path}")


def print_study_summary(study):
    """Print summary of the study."""
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)

    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    running_trials = [t for t in study.trials if t.state == TrialState.RUNNING]

    print(f"Study name: {study.study_name}")
    print(f"Total trials: {len(study.trials)}")
    print(f"  - Complete: {len(complete_trials)}")
    print(f"  - Pruned: {len(pruned_trials)}")
    print(f"  - Running: {len(running_trials)}")

    if complete_trials:
        accuracies = [t.value for t in complete_trials]
        print(f"\nAccuracy statistics:")
        print(f"  - Best: {max(accuracies):.2f}%")
        print(f"  - Mean: {sum(accuracies)/len(accuracies):.2f}%")
        print(f"  - Min: {min(accuracies):.2f}%")

        print(f"\nBest trial (#{study.best_trial.number}):")
        print(f"  Accuracy: {study.best_value:.2f}%")
        print("  Params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")


def main():
    """Run Optuna hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="Optuna sweep for ViT-Tiny FP (Frozen layers)")
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials to run (default: 50)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (default: None)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database path (default: auto-generated)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize existing results without running new trials')
    parser.add_argument('--new-study', action='store_true',
                        help='Start a new study (ignore existing results)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer type (default: sgd)')
    parser.add_argument('--no-wd', action='store_true',
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--no-momentum', action='store_true',
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--no-nesterov', action='store_true',
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    args = parser.parse_args()

    # Update global batch size
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    # Update global config
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov

    os.makedirs(RESULTS, exist_ok=True)

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"vittiny_fp_inputfrozen_bs{BATCH_SIZE}_{get_study_name_suffix()}"
    storage = args.storage or f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            print_study_summary(study)
            visualize_study(study, RESULTS)
            print(f"\nTo run dashboard: optuna-dashboard {storage}")
        except Exception as e:
            print(f"Error loading study: {e}")
            print(f"No existing study found with name '{study_name}'")
        return

    if args.new_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"Deleted existing study: {study_name}")
        except:
            pass

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=BoTorchSampler(),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    existing_trials = len(study.trials)
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if existing_trials > 0:
        print(f"\nResuming study '{study_name}' with {existing_trials} existing trials ({len(completed_trials)} completed)")
        if completed_trials:
            print(f"Current best: {study.best_value:.2f}%")
        else:
            print("No completed trials yet")

    print(f"\n{'='*60}")
    print(f"Study: {study_name}")
    print(f"Database: {storage}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"New trials: {args.n_trials}")
    print(f"Frozen layers: patch_embed + qkv + proj + fc1 + fc2")
    print(f"{'='*60}")
    print(f"(Run multiple instances for parallel execution)")
    print(f"(Use --visualize to see results)")
    print(f"(Use optuna-dashboard {storage} for real-time monitoring)\n")

    def delete_failed_trial_callback(study, trial):
        if trial.state == TrialState.FAIL:
            print(f"[Trial {trial.number}] Failed - removing from database")
            try:
                study._storage.delete_trial(trial._trial_id)
            except Exception as e:
                print(f"[Trial {trial.number}] Could not delete: {e}")

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   catch=(Exception,), show_progress_bar=True,
                   callbacks=[delete_failed_trial_callback])

    print_study_summary(study)
    visualize_study(study, RESULTS)

    if study.best_trial:
        results_path = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(results_path, 'w') as f:
            json.dump({
                "study_name": study_name,
                "best_accuracy": study.best_value,
                "best_params": study.best_params,
                "best_trial_number": study.best_trial.number,
                "n_trials": len(study.trials),
            }, f, indent=2)
        print(f"\nBest params saved to: {results_path}")


if __name__ == "__main__":
    main()

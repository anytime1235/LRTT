# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ViT-SPT-LSA with Digital FP on CIFAR-10.

Usage:
    # Run trials (results are automatically saved and can be resumed)
    python optuna_vitsptlsa_fp.py --n-trials 50

    # Resume existing study and add more trials
    python optuna_vitsptlsa_fp.py --n-trials 30  # continues from saved results

    # Parallel execution (run in multiple terminals, same study name)
    python optuna_vitsptlsa_fp.py --n-trials 20 &
    python optuna_vitsptlsa_fp.py --n-trials 20 &
    python optuna_vitsptlsa_fp.py --n-trials 20 &

    # Visualize results without running new trials
    python optuna_vitsptlsa_fp.py --visualize

    # Use a different study name (for separate experiments)
    python optuna_vitsptlsa_fp.py --n-trials 50 --study-name vitsptlsa_fp_main

    # Real-time dashboard (install: pip install optuna-dashboard)
    optuna-dashboard sqlite:///results/optuna_vitsptlsa_fp/optuna_vitsptlsa_fp_main.db

    # Remote dashboard access (with localtunnel):
    # 1. Start dashboard: optuna-dashboard sqlite:///results/optuna_vitsptlsa_fp/optuna_vitsptlsa_fp_main.db --host 0.0.0.0 --port 8081
    # 2. Start tunnel: npx localtunnel --port 8081
    # 3. Tunnel password: run `curl -s ifconfig.me` to get server's public IP

    # Reset study (delete DB to start fresh)
    rm results/optuna_vitsptlsa_fp/optuna_vitsptlsa_fp_main.db

Results are stored in:
    - SQLite DB: results/optuna_vitsptlsa_fp/optuna_vitsptlsa_fp_main.db
    - JSON summary: results/optuna_vitsptlsa_fp/best_params_*.json
    - Visualization: results/optuna_vitsptlsa_fp/visualization_*.png
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
import matplotlib.pyplot as plt

# Default study name for persistence
DEFAULT_STUDY_NAME = "vitsptlsa_fp_main"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Fixed parameters
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
RESULTS = os.path.join(os.getcwd(), "results", "optuna_vitsptlsa_fp")
os.makedirs(RESULTS, exist_ok=True)

N_CLASSES = 10
IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_WORKERS = 4  # H200 GPU with good storage can handle parallel data loading
SEED = 42

# Fixed model architecture (from paper)
EMBED_DIM = 288
DEPTH = 4
NUM_HEADS = 8
MLP_RATIO = 4.0
DROPOUT = 0.0


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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameters to tune (narrowed based on top 10 results)
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True)  # top10: 7.5e-5 ~ 4.7e-4
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # top10: 32, 128
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)  # top10: 4e-6 ~ 8e-4
    optimizer_name = 'Adam'  # Fixed to Adam

    max_epochs = 2000
    early_stop_patience = 7

    manual_seed(SEED)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting")
    print(f"{'='*70}")
    print(f"  lr={learning_rate:.2e}, batch_size={batch_size}, wd={weight_decay:.2e}, optimizer={optimizer_name}")
    print(f"{'='*70}")

    train_loader, val_loader = load_data(batch_size)

    model = None
    try:
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

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
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
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    parser = argparse.ArgumentParser(description="Optuna sweep for ViT-SPT-LSA FP")
    parser.add_argument('--study-name', type=str, default=DEFAULT_STUDY_NAME,
                        help=f'Study name (default: {DEFAULT_STUDY_NAME})')
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
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    study_name = args.study_name
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
    print(f"New trials: {args.n_trials}")
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

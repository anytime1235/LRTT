#!/usr/bin/env python3
"""MNIST Bayesian Hyperparameter Sweep for LRTT Decay Mode.

Based on sweep_mnist_bayesian.py with decay mode modifications.

Key changes from standard mode:
- reinit_mode: "decay" (instead of "standard")
- reinit_gain: 0.1 (fixed)
- decay_factor: 1.0 (fixed)
- write_noise_std: 0.0 (fixed)
- decay_ratio: search parameter [0.5, 0.99] -> controls 6T1C lifetime

Search parameters:
- rank: [1, 4, 8, 16, 32, 64]
- transfer_every: [1, 5, 10, 20, 50, 100, 500, 1000, 2000]
- learning_rate: [0.001, 1.0] (log scale)
- transfer_lr: [0.001, 10.0] (log scale)
- decay_ratio: [0.5, 0.99] (retention ratio at transfer)

Usage:
    python sweep_mnist_decay.py --n-workers 4
"""

import os
import sys
import math
import argparse
from datetime import datetime
from time import time
from typing import Dict, Any, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add LRTT src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LRTT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(LRTT_ROOT, "src"))

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Logging will be disabled.")
    WANDB_AVAILABLE = False

# ============================================================================
# Configuration (same as bayesian sweep)
# ============================================================================

# Model architecture
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
BATCH_SIZE = 64
EPOCHS = 30

# Dataset
PATH_DATASET = '/tmp/mnist'

# Fixed parameters for decay mode (as requested)
REINIT_MODE = "decay"
REINIT_GAIN = 0.1        # Fixed
DECAY_FACTOR = 1.0       # Fixed
WRITE_NOISE_STD = 0.0    # Fixed
A_INIT_MODE = "zero"     # Fixed (only affects first init in decay mode)
LORA_ALPHA = 1.0

# Search space
RANK_LIST = [1, 4, 8, 16, 32, 64]
TRANSFER_EVERY_LIST = [1, 5, 10, 20, 50, 100, 500, 1000, 2000]

# Hyperparameter ranges
LEARNING_RATE_MIN = 0.001
LEARNING_RATE_MAX = 1.0
TRANSFER_LR_MIN = 0.001
TRANSFER_LR_MAX = 10.0
DECAY_RATIO_MIN = 0.5    # 50% retention at transfer
DECAY_RATIO_MAX = 0.99   # 99% retention at transfer

# Bayesian optimization settings (same as bayesian sweep)
MIN_TRIALS_PER_TE = 5
ADDITIONAL_TRIALS = 45
TOTAL_TRIALS_PER_RANK = MIN_TRIALS_PER_TE * len(TRANSFER_EVERY_LIST) + ADDITIONAL_TRIALS  # 90

# GPU settings
USE_CUDA = torch.cuda.is_available()
N_GPUS = torch.cuda.device_count() if USE_CUDA else 0

# Global worker GPU assignment
WORKER_GPU_ID = None


def get_device():
    """Get the appropriate device for this worker."""
    global WORKER_GPU_ID
    if USE_CUDA:
        if WORKER_GPU_ID is not None and WORKER_GPU_ID < N_GPUS:
            return torch.device(f"cuda:{WORKER_GPU_ID}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_data():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def calculate_lifetime(decay_ratio: float, transfer_every: int) -> float:
    """Calculate 6T1C lifetime from decay ratio.

    Decay model: w(N) = w(0) * (1 - delta)^N
    At transfer: decay_ratio = (1 - delta)^transfer_every
    Solve: delta = 1 - decay_ratio^(1/transfer_every)
    lifetime = 1 / delta
    """
    if decay_ratio >= 1.0:
        return 0.0  # No decay

    delta = 1.0 - math.pow(decay_ratio, 1.0 / transfer_every)
    if delta <= 0:
        return 0.0

    return 1.0 / delta


def create_decay_config(rank: int, transfer_every: int, transfer_lr: float,
                        decay_ratio: float) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with decay mode."""

    # Calculate lifetime from decay ratio
    lifetime = calculate_lifetime(decay_ratio, transfer_every)

    # Create base config (same as bayesian sweep)
    device_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=LORA_ALPHA,
        dt_batch_sec=1.0,
        include_retention=True,  # Enable retention for decay
    )

    # Decay mode settings (fixed as requested)
    device_config.reinit_mode = REINIT_MODE
    device_config.decay_factor = DECAY_FACTOR
    device_config.reinit_gain = REINIT_GAIN
    device_config.a_init_mode = A_INIT_MODE
    device_config.write_noise_std = WRITE_NOISE_STD

    # Tunable parameters
    device_config.transfer_lr = transfer_lr
    device_config.lifetime = lifetime

    # Fixed settings (same as bayesian sweep)
    device_config.correct_gradient_magnitudes = False
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    return PythonLRTTRPUConfig(device=device_config)


def create_model(rank: int, transfer_every: int, transfer_lr: float,
                 decay_ratio: float) -> nn.Module:
    """Create analog network with decay mode.

    Architecture matches bayesian sweep:
    - Layer 1: AnalogLinear with LRTT decay config
    - Layer 2: AnalogLinear with FloatingPointRPUConfig (standard, no LRTT)
    - LogSoftmax output
    """
    device = get_device()
    rpu_config = create_decay_config(rank, transfer_every, transfer_lr, decay_ratio)

    model = AnalogSequential(
        AnalogLinear(
            INPUT_SIZE, HIDDEN_SIZE, bias=False,
            rpu_config=rpu_config,
        ),
        nn.ReLU(),
        AnalogLinear(
            HIDDEN_SIZE, OUTPUT_SIZE, bias=True,
            rpu_config=FloatingPointRPUConfig(),  # Same as bayesian sweep
        ),
        nn.LogSoftmax(dim=1),
    )
    model.to(device)
    return model


def validate(model, val_data, device):
    """Evaluate accuracy using full model (C + A@B)."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_data:
            images = images.to(device).view(images.shape[0], -1)
            labels = labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def validate_c_only(model, val_data, device):
    """Evaluate accuracy using C matrix only (without A@B contribution).

    In decay mode with forward_inject=False, this should be same as validate().
    """
    model.eval()
    correct = 0
    total = 0

    # Get the first layer (LRTT layer)
    lrtt_layer = model[0]  # First AnalogLinear

    with torch.no_grad():
        for images, labels in val_data:
            images = images.to(device).view(images.shape[0], -1)
            labels = labels.to(device)

            # Get C matrix from LRTT layer
            analog_tile = lrtt_layer.analog_module
            if hasattr(analog_tile, 'tile_c'):
                C = analog_tile.tile_c.get_weights()[0]
            else:
                C = analog_tile.get_weights()[0]

            # Ensure C is on the same device as images
            C = C.to(device)

            # Forward with C only for first layer
            h = torch.mm(images, C.T)
            h = torch.relu(h)

            # Continue with rest of model (second layer + logsoftmax)
            # Second layer is FloatingPoint, so use standard forward
            second_layer = model[2]  # Second AnalogLinear
            output = second_layer(h)
            output = torch.log_softmax(output, dim=1)

            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def train_and_evaluate(config: Dict, train_data, val_data,
                       trial: Optional[optuna.Trial] = None,
                       patience: int = 5) -> Dict[str, Any]:
    """Train model and return results with early stopping.

    Same structure as bayesian sweep's train_and_evaluate.
    """
    rank = config['rank']
    transfer_every = config['transfer_every']
    transfer_lr = config['transfer_lr']
    decay_ratio = config['decay_ratio']
    learning_rate = config['learning_rate']

    device = get_device()
    model = create_model(rank, transfer_every, transfer_lr, decay_ratio)
    classifier = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0
    best_c_only_acc = 0
    epochs_without_improvement = 0
    epochs_trained = 0

    time_start = time()

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_data:
            images = images.to(device).view(images.shape[0], -1)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = classifier(output, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        epochs_trained = epoch + 1

        val_acc = validate(model, val_data, device)
        c_only_acc = validate_c_only(model, val_data, device)

        # Track best results
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = True
        if c_only_acc > best_c_only_acc:
            best_c_only_acc = c_only_acc

        # Early stopping check
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break  # Early stop

        # Optuna pruning
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                del model
                if USE_CUDA:
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned()

    train_time = time() - time_start

    # Clean up
    del model
    if USE_CUDA:
        torch.cuda.empty_cache()

    return {
        'best_val_acc': best_val_acc,
        'best_c_only_acc': best_c_only_acc,
        'train_time_sec': train_time,
        'epochs_trained': epochs_trained,
    }


class LRTTDecayObjective:
    """Optuna objective for LRTT decay mode optimization."""

    def __init__(self, rank: int, train_data, val_data,
                 global_trial_counter: Optional[Dict] = None,
                 sweep_group: str = None, use_wandb: bool = True):
        self.rank = rank
        self.train_data = train_data
        self.val_data = val_data
        self.global_trial_counter = global_trial_counter
        self.sweep_group = sweep_group
        self.use_wandb = use_wandb and WANDB_AVAILABLE

    def __call__(self, trial: optuna.Trial) -> float:
        # Sample hyperparameters
        transfer_every = trial.suggest_categorical(
            "transfer_every", TRANSFER_EVERY_LIST
        )
        transfer_lr = trial.suggest_float(
            "transfer_lr", TRANSFER_LR_MIN, TRANSFER_LR_MAX, log=True
        )
        decay_ratio = trial.suggest_float(
            "decay_ratio", DECAY_RATIO_MIN, DECAY_RATIO_MAX
        )
        learning_rate = trial.suggest_float(
            "learning_rate", LEARNING_RATE_MIN, LEARNING_RATE_MAX, log=True
        )

        # Track global trial number
        global_trial_num = 0
        if self.global_trial_counter is not None:
            self.global_trial_counter['count'] += 1
            global_trial_num = self.global_trial_counter['count']

        # Calculate lifetime
        lifetime = calculate_lifetime(decay_ratio, transfer_every)
        trial_id = f"r{self.rank}_te{transfer_every}_t{trial.number}"

        # Create individual wandb run for this trial
        trial_run = None
        if self.use_wandb:
            trial_run = wandb.init(
                project="lrtt-mnist-decay-sweep",
                name=trial_id,
                group=self.sweep_group,
                job_type="trial",
                config={
                    "rank": self.rank,
                    "transfer_every": transfer_every,
                    "transfer_lr": transfer_lr,
                    "decay_ratio": decay_ratio,
                    "lifetime": lifetime,
                    "learning_rate": learning_rate,
                    "trial_number": trial.number,
                    "global_trial_num": global_trial_num,
                    # Fixed params
                    "reinit_mode": REINIT_MODE,
                    "reinit_gain": REINIT_GAIN,
                    "decay_factor": DECAY_FACTOR,
                    "write_noise_std": WRITE_NOISE_STD,
                },
                reinit=True
            )

        try:
            # Build config dict
            config = {
                'rank': self.rank,
                'transfer_every': transfer_every,
                'transfer_lr': transfer_lr,
                'decay_ratio': decay_ratio,
                'learning_rate': learning_rate,
            }

            # Train and evaluate
            result = train_and_evaluate(
                config, self.train_data, self.val_data,
                trial=trial, patience=5
            )

            # Store user attributes
            trial.set_user_attr("rank", self.rank)
            trial.set_user_attr("transfer_every", transfer_every)
            trial.set_user_attr("decay_ratio", decay_ratio)
            trial.set_user_attr("best_val_acc", result['best_val_acc'])

            # Log to individual trial run
            if trial_run is not None:
                wandb.log({
                    "best_val_acc": result['best_val_acc'],
                    "best_c_only_acc": result['best_c_only_acc'],
                    "epochs_trained": result['epochs_trained'],
                    "train_time_sec": result['train_time_sec'],
                    "early_stopped": result['epochs_trained'] < EPOCHS,
                })
                wandb.summary["best_val_acc"] = result['best_val_acc']
                wandb.summary["best_c_only_acc"] = result['best_c_only_acc']
                wandb.finish()

            print(f"    Trial {trial.number}: te={transfer_every}, decay={decay_ratio:.3f}, "
                  f"Val={result['best_val_acc']:.2f}%, C-only={result['best_c_only_acc']:.2f}%, "
                  f"tlr={transfer_lr:.4f}, lr={learning_rate:.4f}")

            # Minimize negative accuracy
            return -result['best_val_acc']

        except optuna.TrialPruned:
            print(f"    Trial {trial.number} pruned")
            if trial_run is not None:
                wandb.log({"pruned": 1})
                wandb.finish()
            raise

        except Exception as e:
            print(f"    Trial {trial.number} failed: {e}")
            if trial_run is not None:
                wandb.log({"error": str(e), "failed": 1})
                wandb.finish()
            return 0.0


def enqueue_balanced_trials(study: optuna.Study, min_trials_per_te: int):
    """Enqueue trials to ensure minimum coverage per transfer_every."""
    import random
    random.seed(42)

    for te in TRANSFER_EVERY_LIST:
        for _ in range(min_trials_per_te):
            study.enqueue_trial({
                "transfer_every": te,
                "transfer_lr": 10 ** random.uniform(
                    np.log10(TRANSFER_LR_MIN), np.log10(TRANSFER_LR_MAX)
                ),
                "decay_ratio": random.uniform(DECAY_RATIO_MIN, DECAY_RATIO_MAX),
                "learning_rate": 10 ** random.uniform(
                    np.log10(LEARNING_RATE_MIN), np.log10(LEARNING_RATE_MAX)
                ),
            })


def run_rank_optimization(rank: int, results_dir: str,
                          train_data, val_data,
                          global_trial_counter: Dict,
                          sweep_group: str = None,
                          use_wandb: bool = True) -> Dict[str, Any]:
    """Run Bayesian optimization for a single rank."""

    print(f"\n[Rank {RANK_LIST.index(rank)+1}/{len(RANK_LIST)}] rank={rank}")

    # Create study with SQLite storage
    study_name = f"rank_{rank}"
    storage_path = os.path.join(results_dir, "studies", f"{study_name}.db")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path}",
        engine_kwargs={"connect_args": {"timeout": 60}}
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=MIN_TRIALS_PER_TE * len(TRANSFER_EVERY_LIST)),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        load_if_exists=True
    )

    # Enqueue balanced trials for Phase 1
    if len(study.trials) == 0:
        enqueue_balanced_trials(study, MIN_TRIALS_PER_TE)

    # Create objective
    objective = LRTTDecayObjective(
        rank=rank,
        train_data=train_data,
        val_data=val_data,
        global_trial_counter=global_trial_counter,
        sweep_group=sweep_group,
        use_wandb=use_wandb
    )

    # Run optimization
    remaining_trials = TOTAL_TRIALS_PER_RANK - len(study.trials)
    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=True,
            gc_after_trial=True
        )

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = -best_trial.value  # Convert back to accuracy

    print(f"\n  Best for rank={rank}:")
    print(f"    Accuracy: {best_value:.2f}%")
    print(f"    transfer_every: {best_params['transfer_every']}")
    print(f"    transfer_lr: {best_params['transfer_lr']:.6f}")
    print(f"    decay_ratio: {best_params['decay_ratio']:.4f}")
    print(f"    learning_rate: {best_params['learning_rate']:.6f}")

    return {
        'rank': rank,
        'best_val_acc': best_value,
        'best_params': best_params,
        'n_trials': len(study.trials)
    }


def main():
    parser = argparse.ArgumentParser(description='LRTT Decay Mode Bayesian Sweep')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(LRTT_ROOT, "sweep_decay_results", f"decay_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("LRTT Decay Mode Bayesian Hyperparameter Sweep")
    print("="*60)
    print(f"\nFixed parameters:")
    print(f"  - reinit_mode: {REINIT_MODE}")
    print(f"  - reinit_gain: {REINIT_GAIN}")
    print(f"  - decay_factor: {DECAY_FACTOR}")
    print(f"  - write_noise_std: {WRITE_NOISE_STD}")
    print(f"  - a_init_mode: {A_INIT_MODE}")
    print(f"\nSearch space:")
    print(f"  - rank: {RANK_LIST}")
    print(f"  - transfer_every: {TRANSFER_EVERY_LIST}")
    print(f"  - learning_rate: [{LEARNING_RATE_MIN}, {LEARNING_RATE_MAX}] (log)")
    print(f"  - transfer_lr: [{TRANSFER_LR_MIN}, {TRANSFER_LR_MAX}] (log)")
    print(f"  - decay_ratio: [{DECAY_RATIO_MIN}, {DECAY_RATIO_MAX}]")
    print(f"\nTrials per rank: {TOTAL_TRIALS_PER_RANK}")
    print(f"  - Phase 1 (balanced): {MIN_TRIALS_PER_TE} × {len(TRANSFER_EVERY_LIST)} = {MIN_TRIALS_PER_TE * len(TRANSFER_EVERY_LIST)}")
    print(f"  - Phase 2 (Bayesian): {ADDITIONAL_TRIALS}")
    print(f"Total trials: {len(RANK_LIST) * TOTAL_TRIALS_PER_RANK}")
    print(f"\nResults directory: {results_dir}")
    print(f"Workers: {args.n_workers}")
    print(f"GPUs available: {N_GPUS}")

    # Load data once
    print("\nLoading MNIST dataset...")
    train_data, val_data = load_data()
    print("Dataset loaded.")

    # Setup wandb
    sweep_group = f"decay_sweep_{timestamp}"
    use_wandb = WANDB_AVAILABLE and not args.no_wandb

    # Initialize summary wandb run
    if use_wandb:
        summary_run = wandb.init(
            project="lrtt-mnist-decay-sweep",
            name=f"{sweep_group}_summary",
            group=sweep_group,
            job_type="summary",
            config={
                "reinit_mode": REINIT_MODE,
                "reinit_gain": REINIT_GAIN,
                "decay_factor": DECAY_FACTOR,
                "write_noise_std": WRITE_NOISE_STD,
                "a_init_mode": A_INIT_MODE,
                "rank_list": RANK_LIST,
                "transfer_every_list": TRANSFER_EVERY_LIST,
                "learning_rate_range": [LEARNING_RATE_MIN, LEARNING_RATE_MAX],
                "transfer_lr_range": [TRANSFER_LR_MIN, TRANSFER_LR_MAX],
                "decay_ratio_range": [DECAY_RATIO_MIN, DECAY_RATIO_MAX],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "total_trials_per_rank": TOTAL_TRIALS_PER_RANK,
                "n_workers": args.n_workers,
            }
        )
        print(f"\nWandB Summary Run: {wandb.run.url}")
        print(f"WandB Group: {sweep_group}")
        wandb.finish()

    # Global trial counter
    global_trial_counter = {'count': 0}

    # Run optimization for each rank
    all_results = []

    for rank in RANK_LIST:
        result = run_rank_optimization(
            rank, results_dir, train_data, val_data,
            global_trial_counter,
            sweep_group=sweep_group, use_wandb=use_wandb
        )
        all_results.append(result)

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Rank':>6} | {'Best Acc%':>10} | {'te':>6} | {'decay':>8} | {'tlr':>10} | {'lr':>10}")
    print("-"*70)

    for r in all_results:
        p = r['best_params']
        print(f"{r['rank']:>6} | {r['best_val_acc']:>10.2f} | {p['transfer_every']:>6} | "
              f"{p['decay_ratio']:>8.4f} | {p['transfer_lr']:>10.4f} | {p['learning_rate']:>10.4f}")

    # Find global best
    best_result = max(all_results, key=lambda x: x['best_val_acc'])
    print(f"\n🏆 GLOBAL BEST: rank={best_result['rank']}, "
          f"te={best_result['best_params']['transfer_every']}, "
          f"decay={best_result['best_params']['decay_ratio']:.4f} → "
          f"{best_result['best_val_acc']:.2f}%")

    # Save results
    results_file = os.path.join(results_dir, "final_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Final summary run
    if use_wandb:
        wandb.init(
            project="lrtt-mnist-decay-sweep",
            name=f"{sweep_group}_summary",
            group=sweep_group,
            job_type="summary",
            resume="allow",
            id=f"{sweep_group}_summary".replace("_", "-")[:64],
        )
        for result in all_results:
            rank = result['rank']
            wandb.log({
                f"summary/rank_{rank}_best_acc": result['best_val_acc'],
                f"summary/rank_{rank}_best_te": result['best_params']['transfer_every'],
                f"summary/rank_{rank}_best_decay": result['best_params']['decay_ratio'],
            })
        wandb.log({
            "summary/global_best_acc": best_result['best_val_acc'],
            "summary/global_best_rank": best_result['rank'],
        })
        wandb.summary["global_best_acc"] = best_result['best_val_acc']
        wandb.summary["global_best_rank"] = best_result['rank']
        wandb.summary["global_best_te"] = best_result['best_params']['transfer_every']
        wandb.summary["global_best_decay"] = best_result['best_params']['decay_ratio']
        wandb.finish()

    print(f"\nWandB Group '{sweep_group}' contains:")
    print(f"  - 1 summary run")
    print(f"  - {global_trial_counter['count']} individual trial runs")


if __name__ == "__main__":
    main()

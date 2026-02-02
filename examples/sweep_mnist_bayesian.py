# -*- coding: utf-8 -*-
"""MNIST LRTT Bayesian Hyperparameter Search with Optuna.

Hybrid search strategy:
- For each rank, create ONE Optuna study
- Phase 1: Ensure minimum trials per transfer_every (balanced exploration)
- Phase 2: Additional trials with full Bayesian optimization (exploitation)

Search Configuration:
- rank: [1, 4, 8, 16, 32, 64]
- transfer_every: [1, 5, 10, 20, 50, 100, 500, 1000, 2000] (categorical, Bayesian)
- transfer_lr: [0.001, 10.0] (log scale, Bayesian)
- reinit_gain: [0.01, 1.0] (log scale, Bayesian)
- learning_rate: [0.001, 1.0] (log scale, Bayesian)

Fixed Settings:
- update_mode: "lora"
- transfer_mode: "off"
- correct_gradient_magnitudes: False
- forward_inject: False

Parallelization:
- Multi-GPU support with --n-workers
- Each worker runs on a separate GPU (round-robin assignment)
- Optuna handles trial distribution via SQLite storage
"""

import os
import sys
import csv
import json
import argparse
from time import time
from datetime import datetime
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")

# Set multiprocessing start method for CUDA compatibility
if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Logging will be disabled.")
    print("Install with: pip install wandb")
    WANDB_AVAILABLE = False

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.rpu_base import cuda

# Device setup
USE_CUDA = 1 if cuda.is_compiled() else 0
N_GPUS = torch.cuda.device_count() if USE_CUDA else 0
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Worker-specific GPU assignment (set per-worker)
WORKER_GPU_ID = None  # Will be set in worker process

# Fixed parameters
PATH_DATASET = os.path.join("data", "DATASET")
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
EPOCHS = 30
BATCH_SIZE = 64
LORA_ALPHA = 1.0

# Search space - Grid parameters
RANK_LIST = [1, 4, 8, 16, 32, 64]
TRANSFER_EVERY_LIST = [1, 5, 10, 20, 50, 100, 500, 1000, 2000]

# Bayesian search ranges
TRANSFER_LR_MIN = 0.001
TRANSFER_LR_MAX = 10.0
REINIT_GAIN_MIN = 0.01
REINIT_GAIN_MAX = 1.0
LEARNING_RATE_MIN = 0.001
LEARNING_RATE_MAX = 1.0

# Trials configuration
MIN_TRIALS_PER_TE = 5      # Minimum trials per transfer_every (Phase 1)
ADDITIONAL_TRIALS = 45     # Additional Bayesian trials per rank (Phase 2)
# Total per rank = 9 te × 5 min + 45 additional = 90 trials

# Output directory
OUTPUT_DIR = "sweep_bayesian_results"


def setup_output_dir():
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, f"bayesian_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "studies"), exist_ok=True)
    return output_dir


def load_images():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    return train_data, validation_data


def create_lrtt_config(rank, transfer_every, transfer_lr, reinit_gain):
    """Create LRTT configuration with given parameters."""
    device_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=LORA_ALPHA,
        dt_batch_sec=1.0,
        include_retention=False,  # Disable retention
    )

    # Fixed settings as requested
    device_config.reinit_gain = reinit_gain
    device_config.correct_gradient_magnitudes = False  # As requested
    device_config.transfer_lr = transfer_lr
    device_config.forward_inject = False
    device_config.update_mode = "lora"  # Fixed
    device_config.reinit_mode = "standard"
    device_config.transfer_mode = "off"  # Fixed

    return PythonLRTTRPUConfig(device=device_config)


def get_device():
    """Get the appropriate device for this worker."""
    global WORKER_GPU_ID
    if USE_CUDA:
        if WORKER_GPU_ID is not None and WORKER_GPU_ID < N_GPUS:
            return torch.device(f"cuda:{WORKER_GPU_ID}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def create_model(rank, transfer_every, transfer_lr, reinit_gain):
    """Create analog network with given LRTT parameters."""
    device = get_device()
    model = AnalogSequential(
        AnalogLinear(
            INPUT_SIZE, HIDDEN_SIZE, bias=False,
            rpu_config=create_lrtt_config(rank, transfer_every, transfer_lr, reinit_gain),
        ),
        nn.ReLU(),
        AnalogLinear(
            HIDDEN_SIZE, OUTPUT_SIZE, bias=True,
            rpu_config=FloatingPointRPUConfig(),
        ),
        nn.LogSoftmax(dim=1),
    )
    model.to(device)
    return model


def validate(model, val_set, device):
    """Evaluate model accuracy."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(device).view(images.shape[0], -1)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total


def validate_c_only(model, val_set, device):
    """Evaluate using C tile only."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(device).view(images.shape[0], -1)
            labels = labels.to(device)
            x = images
            for layer in model:
                if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
                    controller = layer.analog_module.controller
                    x = controller.tile_c.forward(x)
                elif isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
                elif isinstance(layer, nn.LogSoftmax):
                    x = torch.log_softmax(x, dim=1)
                elif hasattr(layer, 'analog_module'):
                    x = layer(x)
            _, predicted = torch.max(x.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total


def train_and_evaluate(config, train_data, val_data, trial=None, patience=5):
    """Train model and return results with early stopping.

    Args:
        config: Training configuration dict
        train_data: Training data loader
        val_data: Validation data loader
        trial: Optuna trial object for pruning (optional)
        patience: Early stopping patience (epochs without improvement)

    Returns:
        dict with best_val_acc, best_c_only_acc, train_time_sec, epochs_trained
    """
    rank = config['rank']
    transfer_every = config['transfer_every']
    transfer_lr = config['transfer_lr']
    reinit_gain = config['reinit_gain']
    learning_rate = config['learning_rate']

    device = get_device()
    model = create_model(rank, transfer_every, transfer_lr, reinit_gain)
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

        # Optuna pruning: report intermediate value and check if should prune
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                # Clean up before raising
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


class RankObjective:
    """Optuna objective for a specific rank with transfer_every as categorical param."""

    def __init__(self, rank, transfer_every_list, train_data, val_data,
                 global_trial_counter=None, n_gpus=1):
        self.rank = rank
        self.transfer_every_list = transfer_every_list
        self.train_data = train_data
        self.val_data = val_data
        self.global_trial_counter = global_trial_counter
        self.n_gpus = n_gpus

    def __call__(self, trial: optuna.Trial) -> float:
        global WORKER_GPU_ID

        # Assign GPU based on trial number (round-robin)
        if self.n_gpus > 0:
            WORKER_GPU_ID = trial.number % self.n_gpus
        # Bayesian search parameters (including transfer_every)
        transfer_every = trial.suggest_categorical("transfer_every", self.transfer_every_list)
        transfer_lr = trial.suggest_float("transfer_lr", TRANSFER_LR_MIN, TRANSFER_LR_MAX, log=True)
        reinit_gain = trial.suggest_float("reinit_gain", REINIT_GAIN_MIN, REINIT_GAIN_MAX, log=True)
        learning_rate = trial.suggest_float("learning_rate", LEARNING_RATE_MIN, LEARNING_RATE_MAX, log=True)

        config = {
            'rank': self.rank,
            'transfer_every': transfer_every,
            'transfer_lr': transfer_lr,
            'reinit_gain': reinit_gain,
            'learning_rate': learning_rate,
        }

        # Increment global trial counter
        if self.global_trial_counter is not None:
            self.global_trial_counter['count'] += 1
            global_trial_num = self.global_trial_counter['count']
        else:
            global_trial_num = trial.number

        try:
            result = train_and_evaluate(config, self.train_data, self.val_data, trial=trial, patience=5)

            # Store additional info
            trial.set_user_attr("best_val_acc", result['best_val_acc'])
            trial.set_user_attr("best_c_only_acc", result['best_c_only_acc'])
            trial.set_user_attr("train_time_sec", result['train_time_sec'])
            trial.set_user_attr("epochs_trained", result['epochs_trained'])
            trial.set_user_attr("rank", self.rank)
            trial.set_user_attr("transfer_every", transfer_every)

            # Log to wandb with clear identification
            if WANDB_AVAILABLE:
                # Create unique trial identifier
                trial_id = f"r{self.rank}_te{transfer_every}_t{trial.number}"

                wandb.log({
                    # Trial identification
                    "trial/id": trial_id,
                    "trial/global_num": global_trial_num,
                    "trial/local_num": trial.number,

                    # Configuration
                    "trial/rank": self.rank,
                    "trial/transfer_every": transfer_every,
                    "trial/transfer_lr": transfer_lr,
                    "trial/reinit_gain": reinit_gain,
                    "trial/learning_rate": learning_rate,

                    # Results
                    "trial/best_val_acc": result['best_val_acc'],
                    "trial/best_c_only_acc": result['best_c_only_acc'],
                    "trial/train_time_sec": result['train_time_sec'],
                    "trial/epochs_trained": result['epochs_trained'],
                    "trial/early_stopped": result['epochs_trained'] < EPOCHS,

                    # Per-rank tracking (for filtering in wandb)
                    f"rank_{self.rank}/val_acc": result['best_val_acc'],
                    f"rank_{self.rank}/c_only_acc": result['best_c_only_acc'],
                    f"rank_{self.rank}/te_{transfer_every}_val": result['best_val_acc'],
                })

            print(f"    Trial {trial.number}: te={transfer_every}, Val={result['best_val_acc']:.2f}%, "
                  f"C-only={result['best_c_only_acc']:.2f}%, "
                  f"tlr={transfer_lr:.4f}, rg={reinit_gain:.4f}, lr={learning_rate:.4f}")

            # Maximize val_acc (return negative for minimization)
            return -result['best_val_acc']

        except optuna.TrialPruned:
            print(f"    Trial {trial.number} pruned (poor intermediate results)")
            if WANDB_AVAILABLE:
                trial_id = f"r{self.rank}_te{transfer_every}_t{trial.number}"
                wandb.log({
                    "trial/id": trial_id,
                    "trial/global_num": global_trial_num,
                    "trial/rank": self.rank,
                    "trial/transfer_every": transfer_every,
                    "trial/pruned": 1,
                })
            raise  # Re-raise to let Optuna handle it

        except Exception as e:
            print(f"    Trial {trial.number} failed: {e}")
            if WANDB_AVAILABLE:
                trial_id = f"r{self.rank}_te{transfer_every}_t{trial.number}"
                wandb.log({
                    "trial/id": trial_id,
                    "trial/global_num": global_trial_num,
                    "trial/rank": self.rank,
                    "trial/transfer_every": transfer_every,
                    "trial/error": str(e),
                    "trial/failed": 1,
                })
            return 0.0  # Return 0% accuracy for failed trials


def enqueue_balanced_trials(study, transfer_every_list, min_trials_per_te):
    """Enqueue trials to ensure minimum coverage per transfer_every."""
    import random
    random.seed(42)

    for te in transfer_every_list:
        for _ in range(min_trials_per_te):
            # Enqueue with fixed transfer_every, random other params
            study.enqueue_trial({
                "transfer_every": te,
                "transfer_lr": 10 ** random.uniform(np.log10(TRANSFER_LR_MIN), np.log10(TRANSFER_LR_MAX)),
                "reinit_gain": 10 ** random.uniform(np.log10(REINIT_GAIN_MIN), np.log10(REINIT_GAIN_MAX)),
                "learning_rate": 10 ** random.uniform(np.log10(LEARNING_RATE_MIN), np.log10(LEARNING_RATE_MAX)),
            })


def run_bayesian_search(test_mode=False, wandb_project="lrtt-mnist-bayesian", n_workers=1,
                        min_trials_per_te=None, additional_trials=None):
    """Run Bayesian hyperparameter search for each rank.

    Args:
        test_mode: Run minimal test configuration
        wandb_project: WandB project name
        n_workers: Number of parallel workers (ideally = number of GPUs)
        min_trials_per_te: Minimum trials per transfer_every (Phase 1)
        additional_trials: Additional Bayesian trials per rank (Phase 2)
    """
    output_dir = setup_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine optimal number of workers
    if n_workers > 1 and N_GPUS > 0:
        n_workers = min(n_workers, N_GPUS)
        print(f"\nUsing {n_workers} parallel workers on {N_GPUS} GPUs")
    elif n_workers > 1 and N_GPUS == 0:
        print(f"\nWarning: No GPU available, using {n_workers} CPU workers")

    effective_n_jobs = n_workers if n_workers > 1 else 1

    # Determine parameters based on mode
    if test_mode:
        rank_list = [32]
        transfer_every_list = [10, 100]
        min_trials_per_te = 2
        additional_trials = 4
        print("=" * 70)
        print("TEST MODE - Running minimal search to verify setup")
        print("=" * 70)
    else:
        rank_list = RANK_LIST
        transfer_every_list = TRANSFER_EVERY_LIST
        min_trials_per_te = min_trials_per_te if min_trials_per_te is not None else MIN_TRIALS_PER_TE
        additional_trials = additional_trials if additional_trials is not None else ADDITIONAL_TRIALS

    n_te = len(transfer_every_list)
    phase1_trials = n_te * min_trials_per_te  # Balanced exploration
    total_trials_per_rank = phase1_trials + additional_trials
    total_trials = len(rank_list) * total_trials_per_rank

    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=f"bayesian_search_{timestamp}",
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "lora_alpha": LORA_ALPHA,
                "rank_list": rank_list,
                "transfer_every_list": transfer_every_list,
                "transfer_lr_range": [TRANSFER_LR_MIN, TRANSFER_LR_MAX],
                "reinit_gain_range": [REINIT_GAIN_MIN, REINIT_GAIN_MAX],
                "learning_rate_range": [LEARNING_RATE_MIN, LEARNING_RATE_MAX],
                "min_trials_per_te": min_trials_per_te,
                "additional_trials": additional_trials,
                "total_trials_per_rank": total_trials_per_rank,
                "total_trials": total_trials,
                "test_mode": test_mode,
                "n_workers": effective_n_jobs,
                "n_gpus": N_GPUS,
                "search_strategy": "hybrid (balanced + bayesian)",
                "fixed_settings": {
                    "update_mode": "lora",
                    "transfer_mode": "off",
                    "correct_gradient_magnitudes": False,
                    "forward_inject": False,
                    "include_retention": False,
                },
            },
            tags=["bayesian", "mnist", "lrtt", "6t1c", "hybrid"] + (["test"] if test_mode else []) + ([f"{effective_n_jobs}workers"] if effective_n_jobs > 1 else []),
        )
        print(f"\nWandB initialized: {wandb.run.url}")

    print("=" * 70)
    print("MNIST LRTT Hybrid Bayesian Hyperparameter Search")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Epochs per trial: {EPOCHS}")
    print(f"\nSearch Strategy: Hybrid (Balanced + Bayesian)")
    print(f"  - Phase 1: {min_trials_per_te} trials per transfer_every (balanced)")
    print(f"  - Phase 2: {additional_trials} additional Bayesian trials")
    print(f"  - Total per rank: {total_trials_per_rank} trials")
    print(f"\nGrid Parameters:")
    print(f"  - rank: {rank_list} ({len(rank_list)} values)")
    print(f"  - transfer_every: {transfer_every_list} ({n_te} values, categorical)")
    print(f"\nBayesian Search Parameters:")
    print(f"  - transfer_lr: [{TRANSFER_LR_MIN}, {TRANSFER_LR_MAX}] (log scale)")
    print(f"  - reinit_gain: [{REINIT_GAIN_MIN}, {REINIT_GAIN_MAX}] (log scale)")
    print(f"  - learning_rate: [{LEARNING_RATE_MIN}, {LEARNING_RATE_MAX}] (log scale)")
    print(f"\nFixed Settings:")
    print(f"  - update_mode: lora")
    print(f"  - transfer_mode: off")
    print(f"  - correct_gradient_magnitudes: False")
    print(f"\nParallelization:")
    print(f"  - Workers: {effective_n_jobs}")
    print(f"  - GPUs available: {N_GPUS}")
    if effective_n_jobs > 1:
        print(f"  - GPU assignment: round-robin (trial_num % {N_GPUS})")
    print(f"\nTotal trials: {total_trials} ({len(rank_list)} ranks × {total_trials_per_rank} trials)")
    print("=" * 70)
    sys.stdout.flush()

    # Load dataset once
    print("\nLoading MNIST dataset...")
    train_data, val_data = load_images()
    print(f"Dataset loaded: {len(train_data.dataset)} train, {len(val_data.dataset)} test\n")
    sys.stdout.flush()

    # Prepare results storage
    all_results = []
    best_per_rank = []
    best_per_rank_te = []  # Best per (rank, transfer_every)

    fieldnames = [
        'rank', 'transfer_every', 'transfer_lr', 'reinit_gain', 'learning_rate',
        'best_val_acc', 'best_c_only_acc', 'train_time_sec', 'epochs_trained', 'trial_number'
    ]

    results_file = os.path.join(output_dir, "all_results.csv")
    best_rank_file = os.path.join(output_dir, "best_per_rank.csv")
    best_rank_te_file = os.path.join(output_dir, "best_per_rank_te.csv")

    # Save config
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump({
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'lora_alpha': LORA_ALPHA,
            'rank_list': rank_list,
            'transfer_every_list': transfer_every_list,
            'transfer_lr_range': [TRANSFER_LR_MIN, TRANSFER_LR_MAX],
            'reinit_gain_range': [REINIT_GAIN_MIN, REINIT_GAIN_MAX],
            'learning_rate_range': [LEARNING_RATE_MIN, LEARNING_RATE_MAX],
            'min_trials_per_te': min_trials_per_te,
            'additional_trials': additional_trials,
            'total_trials_per_rank': total_trials_per_rank,
            'total_trials': total_trials,
            'search_strategy': 'hybrid',
            'n_workers': effective_n_jobs,
            'n_gpus': N_GPUS,
            'fixed_settings': {
                'update_mode': 'lora',
                'transfer_mode': 'off',
                'correct_gradient_magnitudes': False,
                'forward_inject': False,
                'include_retention': False,
            },
            'start_time': datetime.now().isoformat()
        }, f, indent=2)

    # Run Bayesian search for each rank
    total_start = time()
    global_trial_counter = {'count': 0}

    for rank_idx, rank in enumerate(rank_list, 1):
        print(f"\n{'='*70}")
        print(f"[Rank {rank_idx}/{len(rank_list)}] rank={rank}")
        print(f"  Phase 1: {phase1_trials} balanced trials ({min_trials_per_te} per te)")
        print(f"  Phase 2: {additional_trials} Bayesian trials")
        print(f"{'='*70}")
        sys.stdout.flush()

        # Log rank start to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                "rank/index": rank_idx,
                "rank/value": rank,
                "rank/status": "started",
            })

        # Create study for this rank
        study_name = f"rank_{rank}"
        study_path = os.path.join(output_dir, "studies", f"{study_name}.db")

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=42, n_startup_trials=phase1_trials),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
            storage=f"sqlite:///{study_path}",
            load_if_exists=True,
        )

        # Phase 1: Enqueue balanced trials (ensure each transfer_every gets minimum coverage)
        print(f"\n  Phase 1: Enqueueing {phase1_trials} balanced trials...")
        enqueue_balanced_trials(study, transfer_every_list, min_trials_per_te)

        # Create objective
        objective = RankObjective(
            rank, transfer_every_list, train_data, val_data,
            global_trial_counter, n_gpus=N_GPUS
        )

        # Run optimization (Phase 1 + Phase 2)
        # n_jobs > 1 enables parallel trial execution
        study.optimize(
            objective,
            n_trials=total_trials_per_rank,
            n_jobs=effective_n_jobs,
            show_progress_bar=effective_n_jobs == 1,  # Disable progress bar for parallel
            gc_after_trial=True,
        )

        # Collect results from this study
        rank_results = []
        pruned_count = 0
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {
                    'rank': rank,
                    'transfer_every': trial.params['transfer_every'],
                    'transfer_lr': trial.params['transfer_lr'],
                    'reinit_gain': trial.params['reinit_gain'],
                    'learning_rate': trial.params['learning_rate'],
                    'best_val_acc': trial.user_attrs.get('best_val_acc', 0),
                    'best_c_only_acc': trial.user_attrs.get('best_c_only_acc', 0),
                    'train_time_sec': trial.user_attrs.get('train_time_sec', 0),
                    'epochs_trained': trial.user_attrs.get('epochs_trained', EPOCHS),
                    'trial_number': trial.number,
                }
                all_results.append(result)
                rank_results.append(result)
            elif trial.state == optuna.trial.TrialState.PRUNED:
                pruned_count += 1

        # Get best trial for this rank (overall)
        best_trial = study.best_trial
        best_for_rank = {
            'rank': rank,
            'transfer_every': best_trial.params['transfer_every'],
            'transfer_lr': best_trial.params['transfer_lr'],
            'reinit_gain': best_trial.params['reinit_gain'],
            'learning_rate': best_trial.params['learning_rate'],
            'best_val_acc': best_trial.user_attrs.get('best_val_acc', 0),
            'best_c_only_acc': best_trial.user_attrs.get('best_c_only_acc', 0),
            'train_time_sec': best_trial.user_attrs.get('train_time_sec', 0),
            'trial_number': best_trial.number,
        }
        best_per_rank.append(best_for_rank)

        # Get best per (rank, transfer_every)
        for te in transfer_every_list:
            te_results = [r for r in rank_results if r['transfer_every'] == te]
            if te_results:
                best_te = max(te_results, key=lambda x: x['best_val_acc'])
                best_per_rank_te.append(best_te)

        # Print summary for this rank
        print(f"\n  Best for rank={rank}:")
        print(f"    transfer_every={best_for_rank['transfer_every']}")
        print(f"    Val Acc: {best_for_rank['best_val_acc']:.2f}%, C-only: {best_for_rank['best_c_only_acc']:.2f}%")
        print(f"    transfer_lr={best_for_rank['transfer_lr']:.4f}, "
              f"reinit_gain={best_for_rank['reinit_gain']:.4f}, "
              f"lr={best_for_rank['learning_rate']:.4f}")

        # Print trials per transfer_every
        te_counts = {}
        for r in rank_results:
            te = r['transfer_every']
            te_counts[te] = te_counts.get(te, 0) + 1
        print(f"\n  Trials per transfer_every: {dict(sorted(te_counts.items()))}")
        if pruned_count > 0:
            print(f"  Pruned trials: {pruned_count}")
        early_stopped = sum(1 for r in rank_results if r['epochs_trained'] < EPOCHS)
        if early_stopped > 0:
            print(f"  Early stopped trials: {early_stopped}/{len(rank_results)}")
        sys.stdout.flush()

        # Log best result for this rank to wandb
        if WANDB_AVAILABLE:
            best_id = f"r{rank}_te{best_for_rank['transfer_every']}_best"
            wandb.log({
                # Best for this rank
                "rank_best/id": best_id,
                "rank_best/rank": rank,
                "rank_best/transfer_every": best_for_rank['transfer_every'],
                "rank_best/transfer_lr": best_for_rank['transfer_lr'],
                "rank_best/reinit_gain": best_for_rank['reinit_gain'],
                "rank_best/learning_rate": best_for_rank['learning_rate'],
                "rank_best/val_acc": best_for_rank['best_val_acc'],
                "rank_best/c_only_acc": best_for_rank['best_c_only_acc'],

                # Summary per rank (for easy comparison)
                f"summary/rank_{rank}_best_val": best_for_rank['best_val_acc'],
                f"summary/rank_{rank}_best_c_only": best_for_rank['best_c_only_acc'],
                f"summary/rank_{rank}_best_te": best_for_rank['transfer_every'],
            })

            # Log best per (rank, transfer_every) for this rank
            for te_result in [r for r in best_per_rank_te if r['rank'] == rank]:
                te = te_result['transfer_every']
                wandb.log({
                    f"te_best/r{rank}_te{te}_val": te_result['best_val_acc'],
                    f"te_best/r{rank}_te{te}_c_only": te_result['best_c_only_acc'],
                })

        # Save intermediate results
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        with open(best_rank_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(best_per_rank)

        with open(best_rank_te_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(best_per_rank_te)

    total_time = time() - total_start

    # Print final summary
    print("\n" + "=" * 70)
    print("BAYESIAN SEARCH COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Total trials: {len(all_results)}")
    print(f"Results saved to: {output_dir}")

    # Best per rank
    print("\n" + "-" * 70)
    print("BEST CONFIGURATION PER RANK:")
    print("-" * 70)
    sorted_by_rank = sorted(best_per_rank, key=lambda x: x['rank'])
    for r in sorted_by_rank:
        print(f"  rank={r['rank']:2d}: te={r['transfer_every']:4d}, "
              f"Val={r['best_val_acc']:.2f}%, C-only={r['best_c_only_acc']:.2f}%")
        print(f"          tlr={r['transfer_lr']:.4f}, rg={r['reinit_gain']:.4f}, lr={r['learning_rate']:.4f}")

    # Top 10 overall
    print("\n" + "-" * 70)
    print("TOP 10 CONFIGURATIONS (by best_val_acc):")
    print("-" * 70)
    sorted_results = sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"{i:2d}. rank={r['rank']:2d}, te={r['transfer_every']:4d}, "
              f"Val={r['best_val_acc']:.2f}%, C-only={r['best_c_only_acc']:.2f}%")
        print(f"    tlr={r['transfer_lr']:.4f}, rg={r['reinit_gain']:.4f}, lr={r['learning_rate']:.4f}")

    # Top 10 by C-only accuracy
    print("\n" + "-" * 70)
    print("TOP 10 CONFIGURATIONS (by best_c_only_acc):")
    print("-" * 70)
    sorted_by_c = sorted(all_results, key=lambda x: x['best_c_only_acc'], reverse=True)
    for i, r in enumerate(sorted_by_c[:10], 1):
        print(f"{i:2d}. rank={r['rank']:2d}, te={r['transfer_every']:4d}, "
              f"Val={r['best_val_acc']:.2f}%, C-only={r['best_c_only_acc']:.2f}%")

    # Overall best
    print("\n" + "=" * 70)
    best_overall = sorted_results[0]
    print(f"OPTIMAL CONFIG: rank={best_overall['rank']}, te={best_overall['transfer_every']}")
    print(f"                transfer_lr={best_overall['transfer_lr']:.4f}, "
          f"reinit_gain={best_overall['reinit_gain']:.4f}, lr={best_overall['learning_rate']:.4f}")
    print(f"                Val={best_overall['best_val_acc']:.2f}%, C-only={best_overall['best_c_only_acc']:.2f}%")
    print("=" * 70)

    # Create summary heatmaps
    create_heatmaps(best_per_rank_te, rank_list, transfer_every_list, output_dir)

    # Log final summary to wandb
    if WANDB_AVAILABLE:
        # Log overall best configuration
        wandb.log({
            "final/best_rank": best_overall['rank'],
            "final/best_transfer_every": best_overall['transfer_every'],
            "final/best_transfer_lr": best_overall['transfer_lr'],
            "final/best_reinit_gain": best_overall['reinit_gain'],
            "final/best_learning_rate": best_overall['learning_rate'],
            "final/best_val_acc": best_overall['best_val_acc'],
            "final/best_c_only_acc": best_overall['best_c_only_acc'],
            "final/total_time_min": total_time / 60,
            "final/total_trials": len(all_results),
        })

        # Log best by C-only accuracy
        best_c_only = sorted_by_c[0]
        wandb.log({
            "final/best_c_only_rank": best_c_only['rank'],
            "final/best_c_only_transfer_every": best_c_only['transfer_every'],
            "final/best_c_only_val_acc": best_c_only['best_val_acc'],
            "final/best_c_only_c_only_acc": best_c_only['best_c_only_acc'],
        })

        # Log heatmap images
        for img_name in ["heatmap_val_acc.png", "heatmap_c_only_acc.png", "heatmap_trials_count.png"]:
            img_path = os.path.join(output_dir, img_name)
            if os.path.exists(img_path):
                wandb.log({img_name.replace(".png", ""): wandb.Image(img_path)})

        # Create wandb table for best per rank
        columns = ["rank", "transfer_every", "transfer_lr", "reinit_gain",
                   "learning_rate", "best_val_acc", "best_c_only_acc"]
        table = wandb.Table(columns=columns)
        for r in sorted_by_rank:
            table.add_data(
                r['rank'], r['transfer_every'], r['transfer_lr'],
                r['reinit_gain'], r['learning_rate'],
                r['best_val_acc'], r['best_c_only_acc']
            )
        wandb.log({"best_per_rank_table": table})

        # Finish wandb run
        wandb.finish()
        print("\nWandB logging complete.")

    sys.stdout.flush()
    return output_dir


def create_heatmaps(best_per_rank_te, rank_list, transfer_every_list, output_dir):
    """Create heatmaps for best accuracy per (rank, transfer_every)."""
    # Create matrices
    val_acc_matrix = np.zeros((len(rank_list), len(transfer_every_list)))
    c_only_matrix = np.zeros((len(rank_list), len(transfer_every_list)))

    for result in best_per_rank_te:
        if result['rank'] in rank_list and result['transfer_every'] in transfer_every_list:
            rank_idx = rank_list.index(result['rank'])
            te_idx = transfer_every_list.index(result['transfer_every'])
            val_acc_matrix[rank_idx, te_idx] = result['best_val_acc']
            c_only_matrix[rank_idx, te_idx] = result['best_c_only_acc']

    # Validation accuracy heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(val_acc_matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(transfer_every_list)))
    ax.set_xticklabels(transfer_every_list, rotation=45)
    ax.set_yticks(range(len(rank_list)))
    ax.set_yticklabels(rank_list)
    ax.set_xlabel('transfer_every')
    ax.set_ylabel('rank')
    ax.set_title('Best Validation Accuracy (%) per (rank, transfer_every)')
    plt.colorbar(im, ax=ax)
    for i in range(len(rank_list)):
        for j in range(len(transfer_every_list)):
            if val_acc_matrix[i, j] > 0:
                ax.text(j, i, f'{val_acc_matrix[i, j]:.1f}',
                       ha='center', va='center', color='white', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_val_acc.png"), dpi=150)
    plt.close(fig)

    # C-only accuracy heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(c_only_matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(transfer_every_list)))
    ax.set_xticklabels(transfer_every_list, rotation=45)
    ax.set_yticks(range(len(rank_list)))
    ax.set_yticklabels(rank_list)
    ax.set_xlabel('transfer_every')
    ax.set_ylabel('rank')
    ax.set_title('Best C-only Accuracy (%) per (rank, transfer_every)')
    plt.colorbar(im, ax=ax)
    for i in range(len(rank_list)):
        for j in range(len(transfer_every_list)):
            if c_only_matrix[i, j] > 0:
                ax.text(j, i, f'{c_only_matrix[i, j]:.1f}',
                       ha='center', va='center', color='white', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_c_only_acc.png"), dpi=150)
    plt.close(fig)

    print(f"\nHeatmaps saved to: {output_dir}")


def main():
    global WANDB_AVAILABLE

    parser = argparse.ArgumentParser(description='MNIST LRTT Hybrid Bayesian Hyperparameter Search')
    parser.add_argument('--test', action='store_true', help='Run minimal test')
    parser.add_argument('--min-trials-per-te', type=int, default=5,
                        help='Minimum trials per transfer_every (default: 5)')
    parser.add_argument('--additional-trials', type=int, default=45,
                        help='Additional Bayesian trials per rank (default: 45)')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, set to num GPUs for full parallelization)')
    parser.add_argument('--wandb-project', type=str, default='lrtt-mnist-bayesian',
                        help='WandB project name (default: lrtt-mnist-bayesian)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    args = parser.parse_args()

    if args.no_wandb:
        WANDB_AVAILABLE = False
        print("WandB logging disabled.")

    run_bayesian_search(
        test_mode=args.test,
        wandb_project=args.wandb_project,
        n_workers=args.n_workers,
        min_trials_per_te=args.min_trials_per_te,
        additional_trials=args.additional_trials
    )


if __name__ == "__main__":
    main()

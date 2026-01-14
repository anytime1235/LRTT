#!/usr/bin/env python3
"""Optuna-based hyperparameter tuning for A/B tile scaling factors.

Searches for optimal a_x_scaling, a_d_scaling, b_d_scaling values
to maximize R² score. b_x_scaling is fixed at 1.0.

Usage:
    python tune_scaling_optuna.py --n-trials 50
    python tune_scaling_optuna.py --n-trials 100
"""

import argparse
import optuna
from optuna.trial import Trial
import torch
import numpy as np
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from torch.utils.data import DataLoader, TensorDataset

# Silence optuna logs during trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Fixed Default Values (used when parameter is not in search space)
# ============================================================================
FIXED_A_X_SCALING = 0.2651
FIXED_A_D_SCALING = 0.5359
FIXED_B_D_SCALING = 0.7103
FIXED_LORA_ALPHA = 1.0
FIXED_TRANSFER_EVERY = 5
FIXED_DESIRED_BL = 7

# ============================================================================
# Search Space Configuration (modify here!)
# Remove key to use fixed value above
# ============================================================================
DEFAULT_SEARCH_SPACE = {
    'lora_alpha': (0.0, 1.0),    # lora_alpha range (transfer LR)
    'transfer_every': (1, 10),   # transfer_every range (int)
    'desired_bl': (1, 10),       # desired_bl range (int, pulse train length)
}

# Number of runs per trial (average over multiple seeds for noisy environments)
N_RUNS_PER_TRIAL = 30


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Import after defining suppress_stdout
from regression_lrtt_scratch_decay import (
    ScratchExperimentConfig,
    train_lrtt_scratch,
    generate_target_matrix,
    generate_target_dataset,
    DEVICE,
)


class TuningConfig(ScratchExperimentConfig):
    """Config for tuning - inherits from ScratchExperimentConfig."""
    pass


def create_config(a_x_scaling, a_d_scaling, b_d_scaling, lora_alpha=2.0, transfer_every=10, desired_bl=7, epochs=50):
    """Create a config with specified scaling factors."""
    config = TuningConfig()

    # Override scaling factors
    config.x_scaling = None  # Not used (tile-specific scaling)
    config.d_scaling = None  # Not used (tile-specific scaling)

    # Tile-specific scaling
    config.a_x_scaling = a_x_scaling
    config.a_d_scaling = a_d_scaling
    config.b_x_scaling = 1.0  # Fixed
    config.b_d_scaling = b_d_scaling

    # LRTT parameters
    config.lora_alpha = lora_alpha
    config.lrtt_transfer_every = transfer_every
    config.desired_bl = desired_bl

    # Disable verbose logging during tuning
    config.log_ab_scaling = False

    # Reduce epochs for faster tuning
    config.lrtt_epochs = epochs

    return config


def objective(trial: Trial, epochs: int = 50, search_space: dict = None) -> float:
    """Optuna objective function - returns R² score to maximize."""

    # Default search space
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    # Sample hyperparameters (if key exists in search_space, tune it; otherwise use fixed value)
    a_x_scaling = trial.suggest_float("a_x_scaling", *search_space['a_x']) if 'a_x' in search_space else FIXED_A_X_SCALING
    a_d_scaling = trial.suggest_float("a_d_scaling", *search_space['a_d']) if 'a_d' in search_space else FIXED_A_D_SCALING
    b_d_scaling = trial.suggest_float("b_d_scaling", *search_space['b_d']) if 'b_d' in search_space else FIXED_B_D_SCALING
    lora_alpha = trial.suggest_float("lora_alpha", *search_space['lora_alpha']) if 'lora_alpha' in search_space else FIXED_LORA_ALPHA
    transfer_every = trial.suggest_int("transfer_every", *search_space['transfer_every']) if 'transfer_every' in search_space else FIXED_TRANSFER_EVERY
    desired_bl = trial.suggest_int("desired_bl", *search_space['desired_bl']) if 'desired_bl' in search_space else FIXED_DESIRED_BL

    # Create config with sampled parameters
    config = create_config(a_x_scaling, a_d_scaling, b_d_scaling, lora_alpha, transfer_every, desired_bl, epochs)

    # Run multiple times with different seeds and average
    losses = []
    for run_idx in range(N_RUNS_PER_TRIAL):
        seed = 42 + run_idx * 100  # Different seed for each run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate data using generate_target_dataset (returns TensorDataset)
        complexity_level = "medium"
        train_dataset = generate_target_dataset(complexity_level, config, train=True, seed=seed)
        val_dataset = generate_target_dataset(complexity_level, config, train=False, seed=seed)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config.lrtt_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.lrtt_batch_size, shuffle=False)

        try:
            # Train model with suppressed output
            with suppress_stdout():
                model, history, _, _, _ = train_lrtt_scratch(
                    config, train_loader, val_loader,
                    seed=seed, use_wandb=False
                )

            # Get final validation loss
            if history:
                final_val_loss = history[-1].get('batch_loss', float('inf'))
                losses.append(final_val_loss)

        except Exception as e:
            pass  # Skip failed runs

    # Return average loss (negative for maximize)
    if losses:
        avg_loss = np.mean(losses)
        return -avg_loss
    return -float('inf')


def run_tuning(n_trials: int, epochs: int = 50, study_name: str = None, save_results: bool = True, search_space: dict = None):
    """Run Optuna hyperparameter tuning."""

    # Default search space
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    if study_name is None:
        study_name = f"scaling_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Tuning for LRTT")
    print(f"{'='*60}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Runs per trial: {N_RUNS_PER_TRIAL} (averaged)")
    print(f"Epochs per run: {epochs}")
    print(f"Fixed parameters:")
    print(f"  b_x_scaling = 1.0")
    if 'a_x' not in search_space:
        print(f"  a_x_scaling = {FIXED_A_X_SCALING}")
    if 'a_d' not in search_space:
        print(f"  a_d_scaling = {FIXED_A_D_SCALING}")
    if 'b_d' not in search_space:
        print(f"  b_d_scaling = {FIXED_B_D_SCALING}")
    print(f"Search space:")
    if 'a_x' in search_space:
        print(f"  a_x_scaling:    [{search_space['a_x'][0]}, {search_space['a_x'][1]}]")
    if 'a_d' in search_space:
        print(f"  a_d_scaling:    [{search_space['a_d'][0]}, {search_space['a_d'][1]}]")
    if 'b_d' in search_space:
        print(f"  b_d_scaling:    [{search_space['b_d'][0]}, {search_space['b_d'][1]}]")
    if 'lora_alpha' in search_space:
        print(f"  lora_alpha:     [{search_space['lora_alpha'][0]}, {search_space['lora_alpha'][1]}]")
    if 'transfer_every' in search_space:
        print(f"  transfer_every: [{search_space['transfer_every'][0]}, {search_space['transfer_every'][1]}]")
    if 'desired_bl' in search_space:
        print(f"  desired_bl:     [{search_space['desired_bl'][0]}, {search_space['desired_bl'][1]}]")
    print(f"{'='*60}\n")

    # Create study (maximize R²)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # Callback to print progress
    def print_callback(study, trial):
        if trial.value is not None and trial.value > -float('inf'):
            loss = -trial.value  # Convert back to positive loss
            parts = [f"  Trial {trial.number:3d}: Loss={loss:.6f} |"]
            if 'a_x_scaling' in trial.params:
                parts.append(f"a_x={trial.params['a_x_scaling']:.3f},")
            if 'a_d_scaling' in trial.params:
                parts.append(f"a_d={trial.params['a_d_scaling']:.3f},")
            if 'b_d_scaling' in trial.params:
                parts.append(f"b_d={trial.params['b_d_scaling']:.3f},")
            if 'lora_alpha' in trial.params:
                parts.append(f"alpha={trial.params['lora_alpha']:.2f},")
            if 'transfer_every' in trial.params:
                parts.append(f"t_every={trial.params['transfer_every']},")
            if 'desired_bl' in trial.params:
                parts.append(f"bl={trial.params['desired_bl']}")
            print(" ".join(parts).rstrip(","))

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, epochs, search_space),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[print_callback]
    )

    # Print results
    print(f"\n{'='*60}")
    print("TUNING RESULTS")
    print(f"{'='*60}")

    best_loss = -study.best_value  # Convert back to positive loss
    print(f"\nBest trial (#{study.best_trial.number}):")
    print(f"  Loss: {best_loss:.6f}")
    print(f"  Parameters:")
    print(f"    a_x_scaling    = {study.best_params.get('a_x_scaling', FIXED_A_X_SCALING):.4f}{'' if 'a_x_scaling' in study.best_params else ' (fixed)'}")
    print(f"    a_d_scaling    = {study.best_params.get('a_d_scaling', FIXED_A_D_SCALING):.4f}{'' if 'a_d_scaling' in study.best_params else ' (fixed)'}")
    print(f"    b_x_scaling    = 1.0 (fixed)")
    print(f"    b_d_scaling    = {study.best_params.get('b_d_scaling', FIXED_B_D_SCALING):.4f}{'' if 'b_d_scaling' in study.best_params else ' (fixed)'}")
    print(f"    lora_alpha     = {study.best_params.get('lora_alpha', FIXED_LORA_ALPHA):.4f}{'' if 'lora_alpha' in study.best_params else ' (fixed)'}")
    print(f"    transfer_every = {study.best_params.get('transfer_every', FIXED_TRANSFER_EVERY)}{'' if 'transfer_every' in study.best_params else ' (fixed)'}")
    print(f"    desired_bl     = {study.best_params.get('desired_bl', FIXED_DESIRED_BL)}{'' if 'desired_bl' in study.best_params else ' (fixed)'}")

    # Top 5 trials
    print(f"\nTop 5 trials (lowest loss):")
    completed_trials = [t for t in study.trials if t.value is not None and t.value > -float('inf')]
    completed_trials.sort(key=lambda t: t.value, reverse=True)  # Higher value = lower loss
    for i, t in enumerate(completed_trials[:5]):
        loss = -t.value
        parts = [f"  {i+1}. Trial #{t.number}: Loss={loss:.6f} |"]
        if 'a_x_scaling' in t.params:
            parts.append(f"a_x={t.params['a_x_scaling']:.3f},")
        if 'a_d_scaling' in t.params:
            parts.append(f"a_d={t.params['a_d_scaling']:.3f},")
        if 'b_d_scaling' in t.params:
            parts.append(f"b_d={t.params['b_d_scaling']:.3f},")
        if 'lora_alpha' in t.params:
            parts.append(f"alpha={t.params['lora_alpha']:.2f},")
        if 'transfer_every' in t.params:
            parts.append(f"t_every={t.params['transfer_every']},")
        if 'desired_bl' in t.params:
            parts.append(f"bl={t.params['desired_bl']}")
        print(" ".join(parts).rstrip(","))

    # Save results
    if save_results:
        results_dir = Path("results/optuna_tuning")
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "study_name": study_name,
            "n_trials": n_trials,
            "epochs_per_trial": epochs,
            "best_loss": best_loss,
            "best_params": study.best_params,
            "fixed_params": {"b_x_scaling": 1.0},
            "all_trials": [
                {
                    "number": t.number,
                    "loss": -t.value if t.value is not None and t.value > -float('inf') else None,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ]
        }

        results_file = results_dir / f"{study_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    print(f"\n{'='*60}")
    print("Copy these values to regression_lrtt_scratch_decay.py:")
    print(f"{'='*60}")
    print(f"    a_x_scaling = {study.best_params.get('a_x_scaling', FIXED_A_X_SCALING):.4f}")
    print(f"    a_d_scaling = {study.best_params.get('a_d_scaling', FIXED_A_D_SCALING):.4f}")
    print(f"    b_x_scaling = 1.0")
    print(f"    b_d_scaling = {study.best_params.get('b_d_scaling', FIXED_B_D_SCALING):.4f}")
    print(f"    lora_alpha = {study.best_params.get('lora_alpha', FIXED_LORA_ALPHA):.4f}")
    print(f"    lrtt_transfer_every = {study.best_params.get('transfer_every', FIXED_TRANSFER_EVERY)}")
    print(f"    desired_bl = {study.best_params.get('desired_bl', FIXED_DESIRED_BL)}")
    print(f"{'='*60}\n")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuning for LRTT hyperparameters")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials (default: 50)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per trial (default: 50)")
    parser.add_argument("--study-name", type=str, default=None, help="Study name")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--use-default", action="store_true", help="Use DEFAULT_SEARCH_SPACE (ignore other args)")
    args = parser.parse_args()

    # Use DEFAULT_SEARCH_SPACE directly
    search_space = DEFAULT_SEARCH_SPACE

    print(f"Using device: {DEVICE}")

    study = run_tuning(
        n_trials=args.n_trials,
        epochs=args.epochs,
        study_name=args.study_name,
        save_results=not args.no_save,
        search_space=search_space,
    )

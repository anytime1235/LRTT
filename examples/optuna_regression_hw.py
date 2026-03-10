#!/usr/bin/env python3
"""Optuna-based hyperparameter tuning for A/B tile scaling factors.

Searches for optimal a_x_scaling, a_d_scaling, b_d_scaling values
to maximize R² score. b_x_scaling is fixed at 1.0.

Usage:
    python optuna_regression_hw.py --n-trials 50
    python optuna_regression_hw.py --n-trials 100

Dashboard:
    pip install optuna-dashboard
    optuna-dashboard sqlite:///results/optuna_tuning/<study_name>.db

    # Remote dashboard access (with localtunnel):
    # 1. Start dashboard: optuna-dashboard sqlite:///results/optuna_tuning/<study_name>.db --host 0.0.0.0 --port 8081
    # 2. Start tunnel: npx localtunnel --port 8081
    # 3. Tunnel password: run `curl -s ifconfig.me` to get server's public IP
"""

import argparse
import optuna
from optuna.trial import Trial
from optuna.integration import BoTorchSampler
import torch
import numpy as np
import json
import sys
import os
import gc
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from torch.utils.data import DataLoader, TensorDataset

# Global semaphore for limiting concurrent GPU operations
GPU_SEMAPHORE = None

# Silence optuna logs during trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Fixed Default Values (used when parameter is not in search space)
# ============================================================================
FIXED_A_X_SCALING = 0.2651
FIXED_A_D_SCALING = 0.5359
FIXED_B_D_SCALING = 0.7103
FIXED_LORA_ALPHA = 1.0
FIXED_TRANSFER_EVERY = 1000
FIXED_DESIRED_BL = 10
FIXED_LRTT_RANK = 2
FIXED_C_DW_MIN = 0.0008
FIXED_C_DESIRED_BL = 31
FIXED_A_LIFETIME_PER_BATCH = 11.72  # Batch 단위 lifetime (내부적으로 pulse 단위로 변환됨) - A device
FIXED_B_LIFETIME_PER_BATCH = 10000000  # B device lifetime (사실상 decay 없음)
FIXED_LRTT_LR = 0.01   # Learning rate (used when use_manual_scaling=False)
FIXED_BATCH_SIZE = 1   # Batch size for training
FIXED_TRANSFER_RANK_SCHEDULE = 'all'  # 'all' or 'round_robin'
FIXED_TRANSFER_RANKS_PER_STEP = 1     # Ranks per transfer step (only used in round_robin)

# ============================================================================
# Batch Size 조정 공식
# ============================================================================
# batch_size를 K배 증가시킬 때, 동일한 학습 dynamics 유지를 위해:
#
#   new_transfer_every = old_transfer_every / K
#   new_lifetime = old_lifetime / K
#   new_lr = old_lr * sqrt(K)  (Linear Scaling Rule, optional)
#
# 이유:
#   - transfer_every, lifetime은 batch 단위로 정의됨
#   - batch_size 증가 → epoch당 batch 수 감소 → transfer 주기가 epoch 기준으로 늘어남
#   - 파라미터를 K로 나누면 epoch 기준 동일한 주기 유지
#
# 예시 (batch_size 1→4):
#   transfer_every: 378 → 95 (378/4)
#   lifetime: 7370 → 1842 (7370/4)
# ============================================================================

# ============================================================================
# Search Space Configuration (modify here!)
# Remove key to use fixed value above
# ============================================================================
DEFAULT_SEARCH_SPACE = {
    'a_x': (0.0, 1.0),           # a_x_scaling range
    'a_d': (0.0, 1.0),           # a_d_scaling range
    'b_d': (0.0, 1.0),           # b_d_scaling range
    'lora_alpha': (0.0, 30.0),   # lora_alpha range (transfer LR)
    'transfer_every': (1, 30), # transfer_every range (int)
    'desired_bl': (1, 10),      # desired_bl range (int, pulse train length)
    #'lrtt_rank': (1, 4),         # lrtt_rank range (int, log scale)
    #'c_dw_min': (0.0002, 0.2),  # c_dw_min range (float, log scale)
    #'c_desired_bl': (1, 20),    # c_desired_bl range (int, pulse train length for C transfer)
    #'lifetime': (1.0, 1e9), # lifetime range (batch units, log scale)
    #'lrtt_lr': (0.00001, 1.0),   # learning rate range (log scale, used when use_manual_scaling=False)
}

# Number of runs per trial (average over multiple seeds for noisy environments)
N_RUNS_PER_TRIAL = 20


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Use StringIO instead of devnull to avoid closed file issues
    try:
        yield
    finally:
        sys.stdout = old_stdout


# Import after defining suppress_stdout
from regression_lrtt_scratch_decay_hw import (
    ScratchExperimentConfig,
    train_lrtt_scratch,
    generate_target_matrix,
    generate_target_dataset,
    DEVICE,
)


class TuningConfig(ScratchExperimentConfig):
    """Config for tuning - inherits from ScratchExperimentConfig."""
    pass


def create_config(a_x_scaling, a_d_scaling, b_d_scaling, lora_alpha=2.0, transfer_every=10, desired_bl=7, lrtt_rank=1, c_dw_min=0.0002, c_desired_bl=10, lifetime=7370, b_lifetime=10000000, lrtt_lr=0.01, batch_size=1, transfer_rank_schedule='all', transfer_ranks_per_step=1):
    """Create a config with specified scaling factors.

    Note: batch_size 변경 시 transfer_every, lifetime도 함께 조정 필요.
    상단의 'Batch Size 조정 공식' 참고.
    """
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
    config.lrtt_rank = lrtt_rank
    config.c_dw_min = c_dw_min
    config.c_desired_bl = c_desired_bl
    config.a_lifetime = lifetime  # Batch 단위 lifetime (A device)
    config.b_lifetime = b_lifetime  # Batch 단위 lifetime (B device)
    config.lrtt_lr = lrtt_lr    # Learning rate
    config.lrtt_batch_size = batch_size  # Batch size
    config.transfer_rank_schedule = transfer_rank_schedule
    config.transfer_ranks_per_step = transfer_ranks_per_step
    # use_manual_scaling is inherited from ScratchExperimentConfig

    # Disable verbose logging during tuning
    config.log_ab_scaling = False

    # Use lrtt_epochs from ScratchExperimentConfig (same as regression_lrtt_scratch_decay.py)

    return config


def run_single_training(config, seed: int) -> float:
    """Run a single training with semaphore-controlled GPU access."""
    global GPU_SEMAPHORE

    # Acquire semaphore before using GPU
    if GPU_SEMAPHORE is not None:
        GPU_SEMAPHORE.acquire()

    try:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate data and train with suppressed output
        complexity_level = "medium"
        with suppress_stdout():
            train_dataset = generate_target_dataset(complexity_level, config, train=True, seed=seed)
            val_dataset = generate_target_dataset(complexity_level, config, train=False, seed=seed)

            # Create DataLoaders (pin_memory for faster CPU→GPU transfer)
            train_loader = DataLoader(train_dataset, batch_size=config.lrtt_batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=config.lrtt_batch_size, shuffle=False, pin_memory=True)

            # Train model (no history collection for speed)
            model, history, epoch_history, _, _, _ = train_lrtt_scratch(
                config, train_loader, val_loader,
                seed=seed, use_wandb=False, collect_history=False
            )

        # Get final validation loss
        if epoch_history:
            final_val_loss = epoch_history[-1].get('val_loss', float('inf'))
        else:
            final_val_loss = float('inf')

        # Cleanup
        del model, history, epoch_history
        gc.collect()
        torch.cuda.empty_cache()

        return final_val_loss

    except Exception as e:
        import traceback
        traceback.print_exc()
        return float('inf')
    finally:
        # Release semaphore
        if GPU_SEMAPHORE is not None:
            GPU_SEMAPHORE.release()


def objective(trial: Trial, search_space: dict = None) -> float:
    """Optuna objective function - returns R² score to maximize."""
    try:
        return _objective_inner(trial, search_space)
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def _objective_inner(trial: Trial, search_space: dict = None) -> float:
    """Inner objective function."""

    # Default search space
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    # Sample hyperparameters
    # All params use suggest_* so they appear in trial_params for TPE to reference prior trials.
    # Fixed params use suggest with min=max. Scaling params only registered when use_manual_scaling=True.
    manual = ScratchExperimentConfig.use_manual_scaling

    # Scaling params: suggest when manual=True, plain fixed when manual=False
    if manual:
        a_x_scaling = trial.suggest_float("a_x_scaling", *search_space['a_x']) if 'a_x' in search_space else trial.suggest_float("a_x_scaling", FIXED_A_X_SCALING, FIXED_A_X_SCALING)
        a_d_scaling = trial.suggest_float("a_d_scaling", *search_space['a_d']) if 'a_d' in search_space else trial.suggest_float("a_d_scaling", FIXED_A_D_SCALING, FIXED_A_D_SCALING)
        b_d_scaling = trial.suggest_float("b_d_scaling", *search_space['b_d']) if 'b_d' in search_space else trial.suggest_float("b_d_scaling", FIXED_B_D_SCALING, FIXED_B_D_SCALING)
    else:
        a_x_scaling = FIXED_A_X_SCALING
        a_d_scaling = FIXED_A_D_SCALING
        b_d_scaling = FIXED_B_D_SCALING

    lora_alpha = trial.suggest_float("lora_alpha", *search_space['lora_alpha']) if 'lora_alpha' in search_space else trial.suggest_float("lora_alpha", FIXED_LORA_ALPHA, FIXED_LORA_ALPHA)
    transfer_every = trial.suggest_int("transfer_every", *search_space['transfer_every']) if 'transfer_every' in search_space else trial.suggest_int("transfer_every", FIXED_TRANSFER_EVERY, FIXED_TRANSFER_EVERY)
    desired_bl = trial.suggest_int("desired_bl", *search_space['desired_bl']) if 'desired_bl' in search_space else trial.suggest_int("desired_bl", FIXED_DESIRED_BL, FIXED_DESIRED_BL)
    lrtt_rank = trial.suggest_int("lrtt_rank", *search_space['lrtt_rank'], log=True) if 'lrtt_rank' in search_space else trial.suggest_int("lrtt_rank", FIXED_LRTT_RANK, FIXED_LRTT_RANK, log=True)
    c_dw_min = trial.suggest_float("c_dw_min", *search_space['c_dw_min'], log=True) if 'c_dw_min' in search_space else trial.suggest_float("c_dw_min", FIXED_C_DW_MIN, FIXED_C_DW_MIN, log=True)
    c_desired_bl = trial.suggest_int("c_desired_bl", *search_space['c_desired_bl']) if 'c_desired_bl' in search_space else trial.suggest_int("c_desired_bl", FIXED_C_DESIRED_BL, FIXED_C_DESIRED_BL)
    lifetime = trial.suggest_float("lifetime", *search_space['lifetime'], log=True) if 'lifetime' in search_space else trial.suggest_float("lifetime", FIXED_A_LIFETIME_PER_BATCH, FIXED_A_LIFETIME_PER_BATCH, log=True)
    b_lifetime = trial.suggest_float("b_lifetime", *search_space['b_lifetime'], log=True) if 'b_lifetime' in search_space else trial.suggest_float("b_lifetime", FIXED_B_LIFETIME_PER_BATCH, FIXED_B_LIFETIME_PER_BATCH, log=True)
    lrtt_lr = trial.suggest_float("lrtt_lr", *search_space['lrtt_lr'], log=True) if 'lrtt_lr' in search_space else trial.suggest_float("lrtt_lr", FIXED_LRTT_LR, FIXED_LRTT_LR, log=True)
    batch_size = trial.suggest_int("batch_size", *search_space['batch_size']) if 'batch_size' in search_space else trial.suggest_int("batch_size", FIXED_BATCH_SIZE, FIXED_BATCH_SIZE)
    transfer_rank_schedule = trial.suggest_categorical("transfer_rank_schedule", search_space['transfer_rank_schedule']) if 'transfer_rank_schedule' in search_space else trial.suggest_categorical("transfer_rank_schedule", [FIXED_TRANSFER_RANK_SCHEDULE])
    transfer_ranks_per_step = trial.suggest_int("transfer_ranks_per_step", *search_space['transfer_ranks_per_step']) if 'transfer_ranks_per_step' in search_space else trial.suggest_int("transfer_ranks_per_step", FIXED_TRANSFER_RANKS_PER_STEP, FIXED_TRANSFER_RANKS_PER_STEP)

    # Print all trial parameters at start
    params = []
    manual = ScratchExperimentConfig.use_manual_scaling
    if manual: params.append(f"a_x={a_x_scaling:.3f}")
    if manual: params.append(f"a_d={a_d_scaling:.3f}")
    if manual: params.append(f"b_d={b_d_scaling:.3f}")
    params.append(f"alpha={lora_alpha:.2f}")
    params.append(f"t_every={transfer_every}")
    params.append(f"bl={desired_bl}")
    params.append(f"rank={lrtt_rank}")
    params.append(f"c_dw={c_dw_min:.5f}")
    params.append(f"c_bl={c_desired_bl}")
    params.append(f"life={lifetime:.1f}")
    params.append(f"b_life={b_lifetime:.1f}")
    params.append(f"lr={lrtt_lr:.5f}")
    if 'batch_size' in search_space: params.append(f"bs={batch_size}")
    params.append(f"rank_sched={transfer_rank_schedule}")
    params.append(f"ranks_per_step={transfer_ranks_per_step}")
    print(f"[Trial {trial.number:3d}] START | {', '.join(params)}", flush=True)

    # Record config-level parameters as user_attrs (not in search space but important for reproducibility)
    cfg = ScratchExperimentConfig
    trial.set_user_attr("reinit_mode", cfg.reinit_mode)
    trial.set_user_attr("b_init_mode", cfg.b_init_mode)
    trial.set_user_attr("c_init_value", cfg.c_init_value)
    trial.set_user_attr("c_device_type", cfg.c_device_type)
    trial.set_user_attr("transfer_method", cfg.transfer_method)
    # transfer_rank_schedule and transfer_ranks_per_step are now in trial.params via suggest_*
    trial.set_user_attr("input_dim", cfg.input_dim)
    trial.set_user_attr("output_dim", cfg.output_dim)
    trial.set_user_attr("D_prime_train_size", cfg.D_prime_train_size)
    trial.set_user_attr("input_type", cfg.input_type)
    trial.set_user_attr("use_manual_scaling", cfg.use_manual_scaling)
    trial.set_user_attr("n_runs_per_trial", N_RUNS_PER_TRIAL)

    # Create config with sampled parameters
    config = create_config(a_x_scaling, a_d_scaling, b_d_scaling, lora_alpha, transfer_every, desired_bl, lrtt_rank, c_dw_min, c_desired_bl, lifetime, b_lifetime, lrtt_lr, batch_size, transfer_rank_schedule, transfer_ranks_per_step)

    # Run multiple times with different seeds in parallel
    seeds = [42 + run_idx * 100 for run_idx in range(N_RUNS_PER_TRIAL)]

    # Use ThreadPoolExecutor for parallel runs within a trial
    losses = []
    with ThreadPoolExecutor(max_workers=N_RUNS_PER_TRIAL) as executor:
        futures = {executor.submit(run_single_training, config, seed): seed for seed in seeds}
        for future in as_completed(futures):
            loss = future.result()
            if loss < float('inf') and not np.isnan(loss):
                losses.append(loss)

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()

    # Return average loss (negative for maximize)
    if losses:
        avg_loss = np.mean(losses)
        return -avg_loss
    return -1e10  # Finite fallback (BoTorchSampler crashes on -inf/NaN)


def run_tuning(n_trials: int, study_name: str = None, save_results: bool = True, search_space: dict = None, max_concurrent: int = 25, n_jobs: int = 1):
    """Run Optuna hyperparameter tuning.

    Parallelization:
        - n_jobs controls how many trials run concurrently
        - max_concurrent limits total GPU operations via semaphore
        - Set n_jobs=-1 to run all trials in parallel (limited by semaphore)

    Example:
        python optuna_regression.py --n-trials 50 --max-concurrent 25
        → 50 trials start, but only 10 GPU ops run at any time

    Args:
        n_trials: Number of trials
        study_name: Name for the study
        save_results: Whether to save results
        search_space: Parameter search space
        max_concurrent: Maximum concurrent GPU operations (semaphore limit)
        n_jobs: Number of parallel trials (1 = all)
    """
    global GPU_SEMAPHORE
    GPU_SEMAPHORE = threading.Semaphore(max_concurrent)

    # Default search space
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    if study_name is None:
        study_name = f"scaling_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create results directory for SQLite DB
    results_dir = Path("results/optuna_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / f"{study_name}.db"

    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Tuning for LRTT")
    print(f"{'='*60}")
    print(f"Study name: {study_name}")
    print(f"SQLite DB: {db_path}")
    print(f"Number of trials: {n_trials}")
    print(f"Runs per trial: {N_RUNS_PER_TRIAL} (parallel)")
    print(f"Max concurrent GPU ops: {max_concurrent} (semaphore)")
    print(f"Epochs per run: {ScratchExperimentConfig.lrtt_epochs} (from config, with early stopping)")
    print(f"Fixed parameters:")
    print(f"  b_x_scaling = 1.0")
    if 'a_x' not in search_space:
        print(f"  a_x_scaling = {FIXED_A_X_SCALING}")
    if 'a_d' not in search_space:
        print(f"  a_d_scaling = {FIXED_A_D_SCALING}")
    if 'b_d' not in search_space:
        print(f"  b_d_scaling = {FIXED_B_D_SCALING}")
    if 'lora_alpha' not in search_space:
        print(f"  lora_alpha = {FIXED_LORA_ALPHA}")
    if 'transfer_every' not in search_space:
        print(f"  transfer_every = {FIXED_TRANSFER_EVERY}")
    if 'desired_bl' not in search_space:
        print(f"  desired_bl = {FIXED_DESIRED_BL}")
    if 'lrtt_rank' not in search_space:
        print(f"  lrtt_rank = {FIXED_LRTT_RANK}")
    if 'c_dw_min' not in search_space:
        print(f"  c_dw_min = {FIXED_C_DW_MIN}")
    if 'c_desired_bl' not in search_space:
        print(f"  c_desired_bl = {FIXED_C_DESIRED_BL}")
    if 'lifetime' not in search_space:
        print(f"  a_lifetime = {FIXED_A_LIFETIME_PER_BATCH}")
    if 'b_lifetime' not in search_space:
        print(f"  b_lifetime = {FIXED_B_LIFETIME_PER_BATCH}")
    if 'lrtt_lr' not in search_space:
        print(f"  lrtt_lr = {FIXED_LRTT_LR}")
    if 'batch_size' not in search_space:
        print(f"  batch_size = {FIXED_BATCH_SIZE}")
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
    if 'lrtt_rank' in search_space:
        print(f"  lrtt_rank:      [{search_space['lrtt_rank'][0]}, {search_space['lrtt_rank'][1]}] (log)")
    if 'c_dw_min' in search_space:
        print(f"  c_dw_min:       [{search_space['c_dw_min'][0]}, {search_space['c_dw_min'][1]}] (log)")
    if 'c_desired_bl' in search_space:
        print(f"  c_desired_bl:   [{search_space['c_desired_bl'][0]}, {search_space['c_desired_bl'][1]}]")
    if 'lifetime' in search_space:
        print(f"  a_lifetime:     [{search_space['lifetime'][0]}, {search_space['lifetime'][1]}] (log)")
    if 'b_lifetime' in search_space:
        print(f"  b_lifetime:     [{search_space['b_lifetime'][0]}, {search_space['b_lifetime'][1]}] (log)")
    if 'lrtt_lr' in search_space:
        print(f"  lrtt_lr:        [{search_space['lrtt_lr'][0]}, {search_space['lrtt_lr'][1]}] (log)")
    print(f"{'='*60}\n")

    # Create study with SQLite storage (maximize R²)
    # This allows multiple processes to share the same study
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,  # Resume existing study if DB exists
        direction="maximize",
        sampler=BoTorchSampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # Show existing progress
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Resuming study: {n_existing} trials already completed")

    # Callback to print progress
    # Log file for real-time monitoring
    log_file = results_dir / f"{study_name}.log"
    print(f"Trial log: {log_file} (use 'tail -f {log_file}' to monitor)")

    def print_callback(study, trial):
        if trial.value is not None and trial.value > -float('inf'):
            loss = -trial.value  # Convert back to positive loss
            p = trial.params
            parts = [f"  Trial {trial.number:3d}: Loss={loss:.6f} |"]
            for key, fmt in [
                ("a_x_scaling", ".3f"), ("a_d_scaling", ".3f"), ("b_d_scaling", ".3f"),
                ("lora_alpha", ".2f"), ("transfer_every", "d"), ("desired_bl", "d"),
                ("lrtt_rank", "d"), ("c_dw_min", ".5f"), ("c_desired_bl", "d"),
                ("lifetime", ".1f"), ("b_lifetime", ".1f"), ("lrtt_lr", ".5f"),
                ("batch_size", "d"), ("transfer_ranks_per_step", "d"),
            ]:
                if key in p:
                    parts.append(f"{key}={p[key]:{fmt}},")
            if "transfer_rank_schedule" in p:
                parts.append(f"rank_sched={p['transfer_rank_schedule']},")
            msg = " ".join(parts).rstrip(",")

            # Add best trial info
            best = study.best_trial
            best_loss = -best.value
            best_msg = f"    [Best: Trial {best.number}, Loss={best_loss:.6f}]"

            # Write to log file first (always works)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
                f.write(best_msg + '\n')
            # Try to print (may fail with n_jobs > 1)
            try:
                print(msg, flush=True)
                print(best_msg, flush=True)
            except (ValueError, OSError):
                pass  # stdout closed in subprocess

    # Run optimization
    # n_jobs=n_trials to start all trials at once (semaphore limits actual GPU usage)
    actual_n_jobs = n_trials if n_jobs == -1 else n_jobs
    study.optimize(
        lambda trial: objective(trial, search_space),
        n_trials=n_trials,
        n_jobs=actual_n_jobs,
        show_progress_bar=True,
        callbacks=[print_callback]
    )

    # Restore stdout if closed (can happen with tqdm/multiprocessing)
    import sys
    if sys.stdout.closed:
        sys.stdout = open('/dev/tty', 'w')

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
    print(f"    lrtt_rank      = {study.best_params.get('lrtt_rank', FIXED_LRTT_RANK)}{'' if 'lrtt_rank' in study.best_params else ' (fixed)'}")
    print(f"    c_dw_min       = {study.best_params.get('c_dw_min', FIXED_C_DW_MIN):.6f}{'' if 'c_dw_min' in study.best_params else ' (fixed)'}")
    print(f"    c_desired_bl   = {study.best_params.get('c_desired_bl', FIXED_C_DESIRED_BL)}{'' if 'c_desired_bl' in study.best_params else ' (fixed)'}")
    print(f"    a_lifetime     = {study.best_params.get('lifetime', FIXED_A_LIFETIME_PER_BATCH):.4f}{'' if 'lifetime' in study.best_params else ' (fixed)'}")
    print(f"    b_lifetime     = {study.best_params.get('b_lifetime', FIXED_B_LIFETIME_PER_BATCH):.4f}{'' if 'b_lifetime' in study.best_params else ' (fixed)'}")
    print(f"    lrtt_lr        = {study.best_params.get('lrtt_lr', FIXED_LRTT_LR):.6f}{'' if 'lrtt_lr' in study.best_params else ' (fixed)'}")

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
            parts.append(f"bl={t.params['desired_bl']},")
        if 'lrtt_rank' in t.params:
            parts.append(f"rank={t.params['lrtt_rank']},")
        if 'c_dw_min' in t.params:
            parts.append(f"c_dw={t.params['c_dw_min']:.5f},")
        if 'c_desired_bl' in t.params:
            parts.append(f"c_bl={t.params['c_desired_bl']},")
        if 'lifetime' in t.params:
            parts.append(f"life={t.params['lifetime']:.1f},")
        if 'b_lifetime' in t.params:
            parts.append(f"b_life={t.params['b_lifetime']:.1f},")
        if 'lrtt_lr' in t.params:
            parts.append(f"lr={t.params['lrtt_lr']:.5f}")
        print(" ".join(parts).rstrip(","))

    # Save results
    if save_results:
        results_dir = Path("results/optuna_tuning")
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "study_name": study_name,
            "n_trials": n_trials,
            "max_epochs": ScratchExperimentConfig.lrtt_epochs,
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
    print(f"    lrtt_rank = {study.best_params.get('lrtt_rank', FIXED_LRTT_RANK)}")
    print(f"    c_dw_min = {study.best_params.get('c_dw_min', FIXED_C_DW_MIN):.6f}")
    print(f"    c_desired_bl = {study.best_params.get('c_desired_bl', FIXED_C_DESIRED_BL)}")
    print(f"    a_lifetime = {study.best_params.get('lifetime', FIXED_A_LIFETIME_PER_BATCH):.4f}")
    print(f"    b_lifetime = {study.best_params.get('b_lifetime', FIXED_B_LIFETIME_PER_BATCH):.4f}")
    print(f"    lrtt_lr = {study.best_params.get('lrtt_lr', FIXED_LRTT_LR):.6f}")
    print(f"{'='*60}\n")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuning for LRTT hyperparameters")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials (default: 50)")
    parser.add_argument("--max-concurrent", type=int, default=25, help="Max concurrent GPU operations (default: 25)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trials (default: 1 = sequential for TPE)")
    parser.add_argument("--study-name", type=str, default=None, help="Study name")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--use-default", action="store_true", help="Use DEFAULT_SEARCH_SPACE (ignore other args)")
    parser.add_argument("--transfer-every", type=int, default=None, help="Override FIXED_TRANSFER_EVERY")
    args = parser.parse_args()

    # Override FIXED_TRANSFER_EVERY if specified
    if args.transfer_every is not None:
        FIXED_TRANSFER_EVERY = args.transfer_every
        print(f"Using transfer_every={FIXED_TRANSFER_EVERY}")

    # Use DEFAULT_SEARCH_SPACE directly
    search_space = DEFAULT_SEARCH_SPACE

    # Auto-generate study name with transfer_every if not specified
    study_name = args.study_name
    if study_name is None and args.transfer_every is not None:
        study_name = f"transfer_every_{args.transfer_every}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Using device: {DEVICE}")

    study = run_tuning(
        n_trials=args.n_trials,
        study_name=study_name,
        save_results=not args.no_save,
        search_space=search_space,
        max_concurrent=args.max_concurrent,
        n_jobs=args.n_jobs,
    )

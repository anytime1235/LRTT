#!/usr/bin/env python3
"""LRTT Hybrid Reinit Sweep with Hyperparameter Search.

Sweep configuration:
- Device: SoftBoundsDevice (C tile, noise=0), LinearStepDevice (A/B tiles, sixt1c)
- Mode: hybrid (A=0 hard reset, B=B*decay_factor with decay_factor=1.0)
- Lifetime: 1000, 10000, 46505 (sixt1c), 100000
- TE: 1, 10, 50, 100, 500, 1000
- Rank: 1, 4, 8, 16, 32, 64
- Hyperparameter search: max 10 trials per config, early stopping
- Best config per (rank, te, lifetime) trains to 30 epochs
"""

import os
os.environ["LRTT_SILENT"] = "1"

import math
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time
from datetime import datetime
import json

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset, PythonLRTTDevice

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEARCH_EPOCHS = 10       # Epochs for hyperparameter search trials
FULL_EPOCHS = 30         # Epochs for best config training
EARLY_STOP_PATIENCE = 3  # Early stopping patience for search trials
MAX_TRIALS_PER_CONFIG = 10

# Sweep parameters
RANKS = [1, 4, 8, 16, 32, 64]
TES = [1, 10, 50, 100, 500, 1000]
LIFETIMES = [1000, 10000, 46505, 100000]  # 10^3, 10^4, sixt1c, 10^5

# Hyperparameter search ranges (log-uniform)
LR_RANGE = (0.001, 1.0)
TLR_RANGE = (0.001, 10.0)

# SoftBounds config (no noise)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}


def log_uniform(low, high):
    """Sample from log-uniform distribution."""
    return math.exp(random.uniform(math.log(low), math.log(high)))


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    """Convert lifetime to dt_batch_sec for sixt1c_ab preset."""
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    dt = -TAU_SEC * math.log(1 - delta)
    return dt


def load_data():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('/tmp/mnist', download=True, train=True, transform=transform)
    val_set = datasets.MNIST('/tmp/mnist', download=True, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


def create_model(rank: int, te: int, lifetime: float, lr: float, tlr: float):
    """Create LRTT model with hybrid reinit (A=0, B unchanged)."""
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)

    # Calculate lifetime for A/B tiles
    TAU_SEC = 46505.0
    if dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        ab_lifetime = 1.0 / delta
    else:
        ab_lifetime = 0.0

    # A/B tiles: 6T1C LinearStepDevice (sixt1c original params)
    ab_device = LinearStepDevice(
        # Core update parameters (from 6T1C measurements)
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        # Device-to-device variation (sixt1c original)
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        # Cycle-to-cycle variation (sixt1c original)
        dw_min_std=0.3,
        write_noise_std=0.0,  # No write noise
        # LinearStepDevice specific
        mean_bound_reference=True,
        # Retention
        lifetime=ab_lifetime,
        lifetime_dtod=0.1,
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: SoftBounds with NO NOISE
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    # Create PythonLRTTDevice with HYBRID reinit mode
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="hybrid",      # A=0 hard reset, B=B*decay_factor
        decay_factor=1.0,          # B unchanged
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    model = AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    )
    model.to(DEVICE)
    return model


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    for data, target in train_loader:
        data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
        target = target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def validate(model, val_loader):
    """Validate model."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def run_search_trial(rank, te, lifetime, lr, tlr, train_loader, val_loader, trial_id, use_wandb, sweep_group):
    """Run single hyperparameter search trial with early stopping."""
    model = create_model(rank, te, lifetime, lr, tlr)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    if use_wandb:
        wandb.init(
            project="lrtt-hybrid-sweep",
            group=sweep_group,
            name=f"search_r{rank}_te{te}_lt{lifetime}_t{trial_id}",
            tags=[f"rank={rank}", f"te={te}", f"lifetime={lifetime}",
                  "hybrid", "search", "bias=True"],
            config={"rank": rank, "te": te, "lifetime": lifetime,
                    "lr": lr, "tlr": tlr, "trial": trial_id,
                    "mode": "hybrid", "phase": "search"},
            reinit=True
        )

    best_acc = 0.0
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(1, SEARCH_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        scheduler.step()
        epochs_trained = epoch

        if use_wandb:
            wandb.log({"epoch": epoch, "val_acc": val_acc, "best_acc": max(best_acc, val_acc)})

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping for search (low accuracy = not converging)
        if epoch >= 3 and best_acc < 50.0:
            break

        if patience_counter >= EARLY_STOP_PATIENCE:
            break

    if use_wandb:
        wandb.log({"final_best_acc": best_acc, "epochs_trained": epochs_trained})
        wandb.finish()

    del model
    torch.cuda.empty_cache()

    return {
        'lr': lr, 'tlr': tlr,
        'best_acc': best_acc,
        'epochs_trained': epochs_trained,
    }


def run_full_training(rank, te, lifetime, lr, tlr, train_loader, val_loader, use_wandb, sweep_group):
    """Run full 30 epoch training with best hyperparameters."""
    model = create_model(rank, te, lifetime, lr, tlr)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    if use_wandb:
        wandb.init(
            project="lrtt-hybrid-sweep",
            group=sweep_group,
            name=f"full_r{rank}_te{te}_lt{lifetime}",
            tags=[f"rank={rank}", f"te={te}", f"lifetime={lifetime}",
                  "hybrid", "full_training", "bias=True", "best_config"],
            config={"rank": rank, "te": te, "lifetime": lifetime,
                    "lr": lr, "tlr": tlr,
                    "mode": "hybrid", "phase": "full_training"},
            reinit=True
        )

    best_acc = 0.0
    acc_history = []

    for epoch in range(1, FULL_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        scheduler.step()
        best_acc = max(best_acc, val_acc)
        acc_history.append(val_acc)

        if use_wandb:
            wandb.log({"epoch": epoch, "val_acc": val_acc, "best_acc": best_acc})

    if use_wandb:
        wandb.log({"final_best_acc": best_acc})
        wandb.finish()

    del model
    torch.cuda.empty_cache()

    return {
        'best_acc': best_acc,
        'final_acc': acc_history[-1],
        'acc_history': acc_history,
    }


def get_completed_configs():
    """Get set of completed full training configs from wandb."""
    if not WANDB_AVAILABLE:
        return set()
    try:
        api = wandb.Api()
        runs = api.runs('spk/lrtt-hybrid-sweep')
        completed = set()
        for run in runs:
            if 'full_' in run.name and run.state == 'finished':
                # Extract: full_r{rank}_te{te}_lt{lifetime}
                completed.add(run.name.replace('full_', ''))
        return completed
    except Exception as e:
        print(f"Warning: Could not fetch completed configs: {e}")
        return set()


def main():
    results_dir = "/data/LRTT/experiments/hybrid_results"
    os.makedirs(results_dir, exist_ok=True)

    total_configs = len(RANKS) * len(TES) * len(LIFETIMES)
    print(f"Hybrid Sweep: {total_configs} configs x max {MAX_TRIALS_PER_CONFIG} trials")
    print(f"Mode: hybrid reinit (A=0, B unchanged)")

    # Check completed configs from wandb
    completed_configs = get_completed_configs()
    print(f"Already completed: {len(completed_configs)} configs")
    print(f"Remaining: {total_configs - len(completed_configs)} configs")

    train_loader, val_loader = load_data()
    use_wandb = WANDB_AVAILABLE
    sweep_group = "hybrid_sweep_resume"  # Fixed group name for continuity

    # Load existing results if available
    results_file = os.path.join(results_dir, "results_partial.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results")
    else:
        all_results = []

    config_idx = 0

    for lifetime in LIFETIMES:
        for rank in RANKS:
            for te in TES:
                config_idx += 1
                config_key = f"r{rank}_te{te}_lt{lifetime}"

                # Skip if already completed
                if config_key in completed_configs:
                    print(f"\n[{config_idx}/{total_configs}] {config_key} - SKIPPED (already done)")
                    continue

                print(f"\n[{config_idx}/{total_configs}] {config_key}")
                print("-" * 50)

                # Hyperparameter search phase
                search_results = []
                for trial in range(MAX_TRIALS_PER_CONFIG):
                    lr = log_uniform(*LR_RANGE)
                    tlr = log_uniform(*TLR_RANGE)

                    result = run_search_trial(
                        rank, te, lifetime, lr, tlr,
                        train_loader, val_loader, trial, use_wandb, sweep_group
                    )
                    search_results.append(result)
                    print(f"  Trial {trial}: lr={lr:.6f}, tlr={tlr:.6f} -> {result['best_acc']:.2f}%")

                    # Early termination if we find a very good config
                    if result['best_acc'] > 96.5 and trial >= 4:
                        print(f"  Found good config, stopping search early")
                        break

                # Find best hyperparameters
                best_trial = max(search_results, key=lambda x: x['best_acc'])
                best_lr, best_tlr = best_trial['lr'], best_trial['tlr']
                print(f"  Best search: lr={best_lr:.6f}, tlr={best_tlr:.6f} -> {best_trial['best_acc']:.2f}%")

                # Full training with best hyperparameters
                print(f"  Running full {FULL_EPOCHS} epoch training...")
                full_result = run_full_training(
                    rank, te, lifetime, best_lr, best_tlr,
                    train_loader, val_loader, use_wandb, sweep_group
                )
                print(f"  Full training: {full_result['best_acc']:.2f}%")

                # Save config results
                config_result = {
                    'rank': rank, 'te': te, 'lifetime': lifetime,
                    'best_lr': best_lr, 'best_tlr': best_tlr,
                    'search_best_acc': best_trial['best_acc'],
                    'full_best_acc': full_result['best_acc'],
                    'full_final_acc': full_result['final_acc'],
                    'num_trials': len(search_results),
                    'all_search_results': search_results,
                }
                all_results.append(config_result)

                # Save intermediate results
                with open(os.path.join(results_dir, "results_partial.json"), 'w') as f:
                    json.dump(all_results, f, indent=2)

    # Save final results
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)

    global_best = max(all_results, key=lambda x: x['full_best_acc'])
    print(f"\nBest config: r{global_best['rank']}_te{global_best['te']}_lt{global_best['lifetime']}")
    print(f"  lr={global_best['best_lr']:.6f}, tlr={global_best['best_tlr']:.6f}")
    print(f"  Accuracy: {global_best['full_best_acc']:.2f}%")
    print(f"\nResults saved: {results_file}")

    # Per-lifetime summary
    for lt in LIFETIMES:
        lt_results = [r for r in all_results if r['lifetime'] == lt]
        if lt_results:
            best_lt = max(lt_results, key=lambda x: x['full_best_acc'])
            print(f"\nLifetime {lt}: best = r{best_lt['rank']}_te{best_lt['te']} -> {best_lt['full_best_acc']:.2f}%")


if __name__ == "__main__":
    main()

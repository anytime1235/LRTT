#!/usr/bin/env python3
"""LRTT SoftBounds Lifetime Sweep.

Sweep configuration:
- Device: SoftBoundsDevice (noise=0)
- Mode: decay (fixed)
- Lifetime: 1000, 10000, 46505 (sixt1c), 100000
- TE: 1, 10, 50, 100, 500, 1000
- Rank: 1, 4, 8, 16, 32, 64
- Runs per config: 5
- Early stopping: patience=5

Hyperparameters (lr, tlr) from wandb sweep results.
"""

import os
os.environ["LRTT_SILENT"] = "1"

import math
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
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
RUNS_PER_CONFIG = 5

# Sweep parameters
RANKS = [1, 4, 8, 16, 32, 64]
TES = [1, 10, 50, 100, 500, 1000]
LIFETIMES = [1000, 10000, 46505, 100000]  # 10^3, 10^4, sixt1c, 10^5

# SoftBounds config (no noise)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

# Hyperparameters from wandb sweep (best lr, tlr per rank/te)
HYPERPARAMETERS = {
    1: {
        1: {"lr": 0.089054, "tlr": 0.001277},
        10: {"lr": 0.040615, "tlr": 1.692439},
        50: {"lr": 0.674107, "tlr": 0.011775},
        100: {"lr": 0.023799, "tlr": 0.038241},
        500: {"lr": 0.013959, "tlr": 0.048825},
        1000: {"lr": 0.007858, "tlr": 4.472098},
    },
    4: {
        1: {"lr": 0.089054, "tlr": 0.001277},
        10: {"lr": 0.001735, "tlr": 0.008158},
        50: {"lr": 0.674107, "tlr": 0.011775},
        100: {"lr": 0.493706, "tlr": 0.011245},
        500: {"lr": 0.818908, "tlr": 9.649071},
        1000: {"lr": 0.007858, "tlr": 4.472098},
    },
    8: {
        1: {"lr": 0.089054, "tlr": 0.001277},
        10: {"lr": 0.001735, "tlr": 0.008158},
        50: {"lr": 0.092537, "tlr": 3.202363},
        100: {"lr": 0.493706, "tlr": 0.011245},
        500: {"lr": 0.013959, "tlr": 0.048825},
        1000: {"lr": 0.003435, "tlr": 0.011312},
    },
    16: {
        1: {"lr": 0.089054, "tlr": 0.001277},
        10: {"lr": 0.045302, "tlr": 7.80662},
        50: {"lr": 0.008838, "tlr": 0.008245},
        100: {"lr": 0.023799, "tlr": 0.038241},
        500: {"lr": 0.006316, "tlr": 0.53314},
        1000: {"lr": 0.003435, "tlr": 0.011312},
    },
    32: {
        1: {"lr": 0.089054, "tlr": 0.001277},
        10: {"lr": 0.040615, "tlr": 1.692439},
        50: {"lr": 0.008838, "tlr": 0.008245},
        100: {"lr": 0.023799, "tlr": 0.038241},
        500: {"lr": 0.013959, "tlr": 0.048825},
        1000: {"lr": 0.003435, "tlr": 0.011312},
    },
    64: {
        1: {"lr": 0.089054, "tlr": 0.001277},
        10: {"lr": 0.040615, "tlr": 1.692439},
        50: {"lr": 0.008838, "tlr": 0.008245},
        100: {"lr": 0.493706, "tlr": 0.011245},
        500: {"lr": 0.006316, "tlr": 0.53314},
        1000: {"lr": 0.002874, "tlr": 0.359582},
    },
}


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
    """Create LRTT model with SoftBounds C tile and noise-free A/B tiles."""
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

    # Create PythonLRTTDevice directly with custom devices
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
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


def run_single_experiment(rank, te, lifetime, lr, tlr, train_loader, val_loader, run_id):
    """Run single training experiment with early stopping."""
    model = create_model(rank, te, lifetime, lr, tlr)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        scheduler.step()
        epochs_trained = epoch

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping based on low accuracy (clearly not converging)
        if epoch >= 5 and best_acc < 50.0:
            break

        if patience_counter >= EARLY_STOP_PATIENCE:
            break

    del model
    torch.cuda.empty_cache()

    return {
        'best_acc': best_acc,
        'epochs_trained': epochs_trained,
        'early_stopped': epochs_trained < EPOCHS
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/data/LRTT/experiments/sweep_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    total_configs = len(RANKS) * len(TES) * len(LIFETIMES)
    print(f"Sweep: {total_configs} configs x {RUNS_PER_CONFIG} runs = {total_configs * RUNS_PER_CONFIG} total")

    train_loader, val_loader = load_data()
    use_wandb = WANDB_AVAILABLE
    sweep_group = f"sweep_{timestamp}"

    all_results = []
    config_idx = 0

    for lifetime in LIFETIMES:
        for rank in RANKS:
            for te in TES:
                config_idx += 1
                hp = HYPERPARAMETERS[rank][te]
                lr, tlr = hp['lr'], hp['tlr']

                run_accs = []
                for run in range(RUNS_PER_CONFIG):
                    # Individual wandb run per experiment
                    if use_wandb:
                        wandb.init(
                            project="lrtt-softbounds-sweep",
                            group=sweep_group,
                            name=f"r{rank}_te{te}_lt{lifetime}_run{run}",
                            tags=[f"rank={rank}", f"te={te}", f"lifetime={lifetime}",
                                  "SoftBounds", "decay", "bias=True"],
                            config={"rank": rank, "te": te, "lifetime": lifetime,
                                    "lr": lr, "tlr": tlr, "run": run,
                                    "device": "SoftBounds", "mode": "decay"},
                            reinit=True
                        )

                    result = run_single_experiment(
                        rank, te, lifetime, lr, tlr,
                        train_loader, val_loader, run
                    )
                    run_accs.append(result['best_acc'])

                    if use_wandb:
                        wandb.log({"best_acc": result['best_acc'],
                                   "epochs": result['epochs_trained'],
                                   "early_stopped": result['early_stopped']})
                        wandb.finish()

                mean_acc = sum(run_accs) / len(run_accs)
                std_acc = (sum((x - mean_acc)**2 for x in run_accs) / len(run_accs)) ** 0.5

                config_result = {
                    'rank': rank, 'te': te, 'lifetime': lifetime,
                    'lr': lr, 'tlr': tlr,
                    'mean_acc': mean_acc, 'std_acc': std_acc,
                    'max_acc': max(run_accs), 'min_acc': min(run_accs),
                    'all_accs': run_accs,
                }
                all_results.append(config_result)

                print(f"[{config_idx}/{total_configs}] r{rank}_te{te}_lt{lifetime}: {mean_acc:.2f}±{std_acc:.2f}%")

    # Save results
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    global_best = max(all_results, key=lambda x: x['mean_acc'])
    print(f"\nBEST: r{global_best['rank']}_te{global_best['te']}_lt{global_best['lifetime']} = {global_best['mean_acc']:.2f}%")
    print(f"Saved: {results_file}")


if __name__ == "__main__":
    main()

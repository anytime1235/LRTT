#!/usr/bin/env python3
"""Rank=8 Transfer Every Sweep - DECAY mode.

Fixed rank=8, sweep 20 transfer_every values (1-10000) with 3 trials each.
Saves best result per TE.
"""

import os
os.environ["LRTT_SILENT"] = "1"

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
import json

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

# Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
RANK = 8
LIFETIME = 46505  # sixt1c value (fixed)

# SoftBounds config (no noise)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
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


def create_model(te: int, lifetime: float, lr: float, tlr: float):
    """Create LRTT model with decay reinit (A and B both decay)."""
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)

    # Calculate lifetime for A/B tiles
    TAU_SEC = 46505.0
    if dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        ab_lifetime = 1.0 / delta
    else:
        ab_lifetime = 0.0

    # A/B tiles: 6T1C LinearStepDevice
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime, lifetime_dtod=0.1, reset=0.0, reset_dtod=0.0,
    )

    # C tile: SoftBounds with NO NOISE
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    # Create PythonLRTTDevice with DECAY reinit mode
    device_config = PythonLRTTDevice(
        rank=RANK,
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


def run_single_trial(te, lr, tlr, train_loader, val_loader):
    """Run single training trial with early stopping."""
    model = create_model(te, LIFETIME, lr, tlr)
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
    # Load configs
    with open('/root/rank8_te20_sweep_configs.json', 'r') as f:
        sweep_config = json.load(f)

    results_dir = "/root/results/rank8_te20_decay"
    os.makedirs(results_dir, exist_ok=True)

    decay_configs = sweep_config['decay']['configs']
    total_configs = len(decay_configs)

    print("=" * 70)
    print("RANK=8 TRANSFER_EVERY SWEEP - DECAY MODE")
    print("=" * 70)
    print(f"Total TE values: {total_configs}")
    print(f"TE range: 1 - 10000")
    print(f"Trials per TE: 3")
    print(f"Total runs: {total_configs * 3}")
    print(f"Rank: {RANK}")
    print(f"Lifetime: {LIFETIME} (sixt1c)")
    print()

    train_loader, val_loader = load_data()
    all_results = []

    for idx, config in enumerate(decay_configs, 1):
        te = config['te']
        trials = config['trials']

        print(f"[{idx}/{total_configs}] TE={te}")
        print(f"  Center: lr={config['lr_center']:.5f}, tlr={config['tlr_center']:.5f}")
        print("-" * 50)

        trial_results = []
        for trial_idx, trial_params in enumerate(trials):
            lr = trial_params['lr']
            tlr = trial_params['tlr']

            result = run_single_trial(te, lr, tlr, train_loader, val_loader)
            trial_results.append({
                'trial': trial_idx,
                'lr': lr,
                'tlr': tlr,
                'best_acc': result['best_acc'],
                'epochs': result['epochs_trained'],
                'early_stopped': result['early_stopped']
            })

            print(f"    Trial {trial_idx}: lr={lr:.5f}, tlr={tlr:.5f} -> {result['best_acc']:.2f}%")

        # Find best trial
        best_trial = max(trial_results, key=lambda x: x['best_acc'])

        config_result = {
            'te': te,
            'best_acc': best_trial['best_acc'],
            'best_lr': best_trial['lr'],
            'best_tlr': best_trial['tlr'],
            'all_trials': trial_results
        }
        all_results.append(config_result)

        print(f"  -> Best: {best_trial['best_acc']:.2f}% (trial {best_trial['trial']})")
        print()

        # Save intermediate results
        with open(f"{results_dir}/results_partial.json", 'w') as f:
            json.dump(all_results, f, indent=2)

    # Save final results
    with open(f"{results_dir}/results_final.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE - DECAY MODE")
    print("=" * 70)

    global_best = max(all_results, key=lambda x: x['best_acc'])
    print(f"\nBest: TE={global_best['te']}, Acc={global_best['best_acc']:.2f}%")
    print(f"      lr={global_best['best_lr']:.5f}, tlr={global_best['best_tlr']:.5f}")
    print(f"\nResults saved: {results_dir}/results_final.json")


if __name__ == "__main__":
    main()

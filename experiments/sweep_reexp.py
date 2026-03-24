#!/usr/bin/env python3
"""Re-experiment: Rank x TE sweep for Decay and Hybrid (Reset) modes.

Model/train/eval code identical to sweep_rank8_nn_decay.py / sweep_rank8_nn_hybrid.py.
Only changes: rank from config, lifetime=0, unified script for both modes.

Usage:
  # Step 1: Fixed HP (lr=0.3, tlr=0.005)
  python generate_reexp_config.py
  python sweep_reexp.py --config reexp_sweep_configs.json --mode decay
  python sweep_reexp.py --config reexp_sweep_configs.json --mode hybrid

  # Step 1b: tlr = 0.009/sqrt(rank), rank=1 excluded
  python generate_reexp_config.py --tlr_rule sqrt_rank
  python sweep_reexp.py --config reexp_sweep_configs_sqrt_rank.json --mode decay
  python sweep_reexp.py --config reexp_sweep_configs_sqrt_rank.json --mode hybrid
"""

import os
os.environ["LRTT_SILENT"] = "1"

import argparse
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
LIFETIME = 0  # no retention

# SoftBounds config (no noise) — identical to original
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


def create_model(rank: int, te: int, reinit_mode: str, tlr: float):
    """Create LRTT model — identical to original except rank/lifetime parameterized."""
    if LIFETIME > 0:
        dt_batch_sec = lifetime_to_dt_batch_sec(LIFETIME)
        TAU_SEC = 46505.0
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        ab_lifetime = 1.0 / delta
    else:
        ab_lifetime = 0.0

    # A/B tiles: 6T1C LinearStepDevice — identical to original
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime, lifetime_dtod=0.1 if ab_lifetime > 0 else 0.0,
        reset=0.0, reset_dtod=0.0,
    )

    # C tile: SoftBounds with NO NOISE — identical to original
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    # PythonLRTTDevice — reinit_mode is the only mode difference
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=1.0,
        reinit_mode=reinit_mode,  # "decay" or "hybrid"
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Disable output noise — identical to original
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

    # Model architecture — identical to original
    model = AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    )
    model.to(DEVICE)
    return model


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch — identical to original."""
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
    """Validate model — identical to original."""
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


def run_single_trial(rank, te, reinit_mode, lr, tlr, train_loader, val_loader):
    """Run single training trial with early stopping — identical to original."""
    model = create_model(rank, te, reinit_mode, tlr)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config JSON from generate_reexp_config.py")
    parser.add_argument("--mode", type=str, required=True, choices=["decay", "hybrid"],
                        help="decay or hybrid (reset)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        sweep_config = json.load(f)

    reinit_mode = args.mode  # "decay" or "hybrid"
    configs = sweep_config[args.mode]['configs']
    total_configs = len(configs)
    trials_per = sweep_config['metadata']['trials_per_te']

    if args.output_dir is None:
        args.output_dir = f"results/reexp_{args.mode}"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"RE-EXPERIMENT: RANK x TE SWEEP - {args.mode.upper()} MODE")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Ranks: {sweep_config['metadata']['ranks']}")
    print(f"HP: lr={sweep_config['metadata']['lr']}, tlr={sweep_config['metadata']['tlr']}")
    print(f"Total cells: {total_configs}, Trials/cell: {trials_per}")
    print(f"Total runs: {total_configs * trials_per}")
    print(f"Lifetime: {LIFETIME}")
    print(f"Device: {DEVICE}")
    print()

    train_loader, val_loader = load_data()
    all_results = []

    for idx, config in enumerate(configs, 1):
        te = config['te']
        rank = config['rank']
        trials = config['trials']

        print(f"[{idx}/{total_configs}] Rank={rank}, TE={te}, "
              f"lr={config['lr_base']:.4f}, tlr={config['tlr_base']:.6f}")

        trial_results = []
        for trial_idx, trial_params in enumerate(trials):
            lr = trial_params['lr']
            tlr = trial_params['tlr']

            result = run_single_trial(rank, te, reinit_mode, lr, tlr,
                                      train_loader, val_loader)
            trial_results.append({
                'trial': trial_idx,
                'lr': lr,
                'tlr': tlr,
                'best_acc': result['best_acc'],
                'epochs': result['epochs_trained'],
                'early_stopped': result['early_stopped']
            })

            print(f"  Trial {trial_idx}: {result['best_acc']:.2f}% "
                  f"({result['epochs_trained']}ep)")

        accs = [t['best_acc'] for t in trial_results]
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs))**0.5

        config_result = {
            'rank': rank,
            'te': te,
            'lr': config['lr_base'],
            'tlr': config['tlr_base'],
            'mean_acc': round(mean_acc, 2),
            'std_acc': round(std_acc, 2),
            'best_acc': round(max(accs), 2),
            'all_trials': trial_results
        }
        all_results.append(config_result)
        print(f"  -> Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print()

        # Save intermediate
        with open(f"{args.output_dir}/results_partial.json", 'w') as f:
            json.dump(all_results, f, indent=2)

    # Save final
    with open(f"{args.output_dir}/results_final.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("=" * 70)
    print(f"SWEEP COMPLETE - {args.mode.upper()} MODE")
    print("=" * 70)
    global_best = max(all_results, key=lambda x: x['best_acc'])
    print(f"Best: Rank={global_best['rank']}, TE={global_best['te']}, "
          f"Acc={global_best['best_acc']:.2f}%")
    print(f"Results: {args.output_dir}/results_final.json")


if __name__ == "__main__":
    main()

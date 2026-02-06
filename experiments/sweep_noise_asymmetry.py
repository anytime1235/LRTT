#!/usr/bin/env python3
"""Sweep C-tile noise and asymmetry for best rank configs.

Experiment design:
- up_down (asymmetry): 0%, 10%, 25%, 50% -> 0.0, 0.048, 0.111, 0.200
- noise_scale (GokmenVlasov): 0%, 10%, 50%, 100% -> 0.0, 0.1, 0.5, 1.0
- 4x4 = 16 combinations, exclude (0,0) -> 15 combinations
- 6 ranks x 15 combinations x 2 modes (hybrid, decay) = 180 experiments
- 5 runs per experiment = 900 total runs
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
RUNS_PER_CONFIG = 1

# Asymmetry levels (UP/DOWN ratio -> up_down parameter)
# 0% -> 1.00 ratio -> 0.000
# 10% -> 1.10 ratio -> 0.048
# 25% -> 1.25 ratio -> 0.111
# 50% -> 1.50 ratio -> 0.200
UP_DOWN_VALUES = [0.0, 0.048, 0.111, 0.200]
UP_DOWN_LABELS = ['0%', '10%', '25%', '50%']

# Noise levels (GokmenVlasov scale)
# 0% -> scale 0.0
# 10% -> scale 0.1
# 50% -> scale 0.5
# 100% -> scale 1.0
NOISE_SCALES = [0.0, 0.1, 0.5, 1.0]
NOISE_LABELS = ['0%', '10%', '50%', '100%']

# Best configs per rank - HYBRID mode
HYBRID_CONFIGS = {
    1: {'te': 500, 'lifetime': 100000, 'lr': 0.2559987802145917, 'tlr': 0.06477913614978935},
    4: {'te': 100, 'lifetime': 100000, 'lr': 0.34183447068119827, 'tlr': 0.003910713393630559},
    8: {'te': 50, 'lifetime': 100000, 'lr': 0.7010916941927409, 'tlr': 0.004173727844750237},
    16: {'te': 500, 'lifetime': 46505, 'lr': 0.2334611631608765, 'tlr': 0.004401532304629191},
    32: {'te': 500, 'lifetime': 100000, 'lr': 0.6023546288776297, 'tlr': 0.001898086405681223},
    64: {'te': 500, 'lifetime': 46505, 'lr': 0.2520116659530396, 'tlr': 0.0013358069303925648},
}

# Best configs per rank - DECAY mode
DECAY_CONFIGS = {
    1: {'te': 50, 'lifetime': 1000, 'lr': 0.674107, 'tlr': 0.011775},
    4: {'te': 100, 'lifetime': 46505, 'lr': 0.493706, 'tlr': 0.011245},
    8: {'te': 100, 'lifetime': 1000, 'lr': 0.493706, 'tlr': 0.011245},
    16: {'te': 1, 'lifetime': 10000, 'lr': 0.089054, 'tlr': 0.001277},
    32: {'te': 1, 'lifetime': 10000, 'lr': 0.089054, 'tlr': 0.001277},
    64: {'te': 1, 'lifetime': 100000, 'lr': 0.089054, 'tlr': 0.001277},
}

RANKS = [1, 4, 8, 16, 32, 64]


def get_softbounds_config_gokmen(noise_scale, up_down):
    """
    GokmenVlasov ratio-based noise scaling for C-tile.

    Args:
        noise_scale: 0.0 (no noise), 0.1 (10%), 0.5 (50%), 1.0 (100% = GokmenVlasov)
        up_down: asymmetry parameter (0.0, 0.048, 0.111, 0.200)
    """
    return {
        'dw_min': 0.001,
        'w_max': 1.0,
        'w_min': -1.0,
        'mult_noise': True,
        # Asymmetry (experimental variable)
        'up_down': up_down,
        'up_down_dtod': 0.0,  # No D2D variation in asymmetry for clean comparison
        # C2C noise
        'dw_min_std': 0.3 * noise_scale,
        'write_noise_std': 0.0,
        # D2D noise (proportional scaling)
        'dw_min_dtod': 0.3 * noise_scale,
        'w_max_dtod': 0.3 * noise_scale,
        'w_min_dtod': 0.3 * noise_scale,
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


def create_model(rank: int, te: int, lifetime: float, lr: float, tlr: float,
                 reinit_mode: str, noise_scale: float, up_down: float):
    """Create LRTT model with specified noise and asymmetry."""
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
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime,
        lifetime_dtod=0.1,
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: SoftBounds with noise and asymmetry
    c_config = get_softbounds_config_gokmen(noise_scale, up_down)
    c_device = SoftBoundsDevice(**c_config)

    # Create PythonLRTTDevice
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode=reinit_mode,
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


def run_training(rank, config, reinit_mode, noise_scale, up_down,
                 run_id, train_loader, val_loader, use_wandb):
    """Run single training."""
    te = config['te']
    lifetime = config['lifetime']
    lr = config['lr']
    tlr = config['tlr']

    model = create_model(rank, te, lifetime, lr, tlr, reinit_mode, noise_scale, up_down)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    # Find labels
    up_down_idx = UP_DOWN_VALUES.index(up_down) if up_down in UP_DOWN_VALUES else -1
    noise_idx = NOISE_SCALES.index(noise_scale) if noise_scale in NOISE_SCALES else -1
    up_down_label = UP_DOWN_LABELS[up_down_idx] if up_down_idx >= 0 else f"{up_down}"
    noise_label = NOISE_LABELS[noise_idx] if noise_idx >= 0 else f"{noise_scale}"

    run_name = f"{reinit_mode}_r{rank}_asymm{up_down_label}_noise{noise_label}_run{run_id}"

    if use_wandb:
        wandb.init(
            project="lrtt-noise-asymmetry-sweep",
            name=run_name,
            tags=[f"rank={rank}", f"mode={reinit_mode}",
                  f"asymm={up_down_label}", f"noise={noise_label}",
                  f"te={te}", f"lifetime={lifetime}"],
            config={
                "rank": rank, "te": te, "lifetime": lifetime,
                "lr": lr, "tlr": tlr, "run": run_id,
                "reinit_mode": reinit_mode,
                "up_down": up_down, "up_down_label": up_down_label,
                "noise_scale": noise_scale, "noise_label": noise_label,
                "dw_min_std": 0.3 * noise_scale,
            },
            reinit=True
        )

    best_acc = 0.0
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(1, EPOCHS + 1):
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

        # Early stopping
        if epoch >= 5 and best_acc < 50.0:
            print(f"      Early stopped at epoch {epoch} (low accuracy)")
            break

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"      Early stopped at epoch {epoch}")
            break

    if use_wandb:
        wandb.log({"best_acc": best_acc, "epochs_trained": epochs_trained})
        wandb.finish()

    del model
    torch.cuda.empty_cache()

    return best_acc


def main():
    print("="*70)
    print("Noise & Asymmetry Sweep for C-tile")
    print("="*70)

    # Generate experiment combinations (exclude 0,0)
    experiments = []
    for up_down_idx, up_down in enumerate(UP_DOWN_VALUES):
        for noise_idx, noise_scale in enumerate(NOISE_SCALES):
            if up_down == 0.0 and noise_scale == 0.0:
                continue  # Skip baseline (already have results)
            experiments.append({
                'up_down': up_down,
                'up_down_label': UP_DOWN_LABELS[up_down_idx],
                'noise_scale': noise_scale,
                'noise_label': NOISE_LABELS[noise_idx],
            })

    total_experiments = len(RANKS) * len(experiments) * 2  # 2 modes
    total_runs = total_experiments * RUNS_PER_CONFIG

    print(f"Asymmetry levels: {UP_DOWN_LABELS} -> {UP_DOWN_VALUES}")
    print(f"Noise levels: {NOISE_LABELS} -> {NOISE_SCALES}")
    print(f"Combinations (excl. 0,0): {len(experiments)}")
    print(f"Ranks: {RANKS}")
    print(f"Modes: hybrid, decay")
    print(f"Runs per config: {RUNS_PER_CONFIG}")
    print(f"Total experiments: {total_experiments}")
    print(f"Total runs: {total_runs}")
    print()

    train_loader, val_loader = load_data()
    use_wandb = WANDB_AVAILABLE

    all_results = []
    exp_count = 0

    for mode in ['hybrid', 'decay']:
        configs = HYBRID_CONFIGS if mode == 'hybrid' else DECAY_CONFIGS

        for rank in RANKS:
            config = configs[rank]

            for exp in experiments:
                exp_count += 1
                up_down = exp['up_down']
                noise_scale = exp['noise_scale']
                up_down_label = exp['up_down_label']
                noise_label = exp['noise_label']

                print(f"\n[{exp_count}/{total_experiments}] {mode} rank={rank} asymm={up_down_label} noise={noise_label}")
                print(f"  Config: te={config['te']}, lifetime={config['lifetime']}, lr={config['lr']:.4f}, tlr={config['tlr']:.4f}")
                print("-"*50)

                results = []
                for run_id in range(RUNS_PER_CONFIG):
                    acc = run_training(
                        rank, config, mode, noise_scale, up_down,
                        run_id, train_loader, val_loader, use_wandb
                    )
                    results.append(acc)
                    print(f"    Run {run_id}: {acc:.2f}%")

                best = max(results)
                avg = sum(results) / len(results)
                print(f"  -> Best: {best:.2f}%, Avg: {avg:.2f}%")

                all_results.append({
                    'mode': mode,
                    'rank': rank,
                    'te': config['te'],
                    'lifetime': config['lifetime'],
                    'lr': config['lr'],
                    'tlr': config['tlr'],
                    'up_down': up_down,
                    'up_down_label': up_down_label,
                    'noise_scale': noise_scale,
                    'noise_label': noise_label,
                    'dw_min_std': 0.3 * noise_scale,
                    'results': results,
                    'best': best,
                    'avg': avg,
                })

                # Save intermediate results
                with open('/data/LRTT_transformer/experiments/noise_asymmetry_results.json', 'w') as f:
                    json.dump(all_results, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for mode in ['hybrid', 'decay']:
        print(f"\n{mode.upper()} MODE:")
        mode_results = [r for r in all_results if r['mode'] == mode]

        for rank in RANKS:
            rank_results = [r for r in mode_results if r['rank'] == rank]
            if rank_results:
                best_config = max(rank_results, key=lambda x: x['best'])
                print(f"  Rank {rank}: Best at asymm={best_config['up_down_label']}, "
                      f"noise={best_config['noise_label']} -> {best_config['best']:.2f}%")

    print("\n" + "="*70)
    print("COMPLETED")
    print("="*70)
    print(f"Results saved to: /data/LRTT_transformer/experiments/noise_asymmetry_results.json")


if __name__ == "__main__":
    main()

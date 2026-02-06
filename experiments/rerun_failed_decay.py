#!/usr/bin/env python3
"""Re-run failed decay configurations with nearby hyperparameters.

Failed configs (best < 90%):
- rank=4, te=10: Use params from te=100
- rank=8, te=10: Use params from te=100
- rank=8, te=50: Use params from te=100
- rank=64, te=100: Use params from te=10

All runs use lifetime=46505 (sixt1c default).
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
LIFETIME = 46505  # sixt1c default
RUNS_PER_CONFIG = 5

# Failed configurations with suggested hyperparameters from nearby TE
FAILED_CONFIGS = [
    {'rank': 4, 'te': 10, 'lr': 0.493706, 'tlr': 0.011245, 'source': 'te=100'},
    {'rank': 8, 'te': 10, 'lr': 0.493706, 'tlr': 0.011245, 'source': 'te=100'},
    {'rank': 8, 'te': 50, 'lr': 0.493706, 'tlr': 0.011245, 'source': 'te=100'},
    {'rank': 64, 'te': 100, 'lr': 0.040615, 'tlr': 1.692439, 'source': 'te=10'},
]

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


def create_model(rank: int, te: int, lifetime: float, lr: float, tlr: float):
    """Create LRTT model with decay reinit mode."""
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

    # C tile: SoftBounds with NO NOISE
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    # Create PythonLRTTDevice with DECAY reinit mode
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,  # Same as sweep_softbounds_lifetime.py
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


def run_training(config, run_id, train_loader, val_loader, use_wandb):
    """Run single training."""
    rank = config['rank']
    te = config['te']
    lr = config['lr']
    tlr = config['tlr']

    model = create_model(rank, te, LIFETIME, lr, tlr)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    run_name = f"rerun_r{rank}_te{te}_run{run_id}"

    if use_wandb:
        wandb.init(
            project="lrtt-softbounds-sweep",
            name=run_name,
            tags=[f"rank={rank}", f"te={te}", f"lifetime={LIFETIME}",
                  "decay", "rerun", "bias=True", f"source={config['source']}"],
            config={"rank": rank, "te": te, "lifetime": LIFETIME,
                    "lr": lr, "tlr": tlr, "run": run_id,
                    "mode": "decay", "rerun": True, "source": config['source']},
            reinit=True
        )

    best_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        scheduler.step()

        if use_wandb:
            wandb.log({"epoch": epoch, "val_acc": val_acc, "best_acc": max(best_acc, val_acc)})

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping based on low accuracy (clearly not converging)
        if epoch >= 5 and best_acc < 50.0:
            print(f"    Early stopped at epoch {epoch} (low accuracy)")
            break

        if patience_counter >= patience:
            print(f"    Early stopped at epoch {epoch}")
            break

    if use_wandb:
        wandb.log({"best_acc": best_acc, "early_stopped": patience_counter >= patience})
        wandb.finish()

    del model
    torch.cuda.empty_cache()

    return best_acc


def main():
    print("="*70)
    print("Re-running Failed Decay Configurations")
    print("="*70)
    print(f"Lifetime: {LIFETIME} (sixt1c default)")
    print(f"Runs per config: {RUNS_PER_CONFIG}")
    print(f"Total configs: {len(FAILED_CONFIGS)}")
    print()

    train_loader, val_loader = load_data()
    use_wandb = WANDB_AVAILABLE

    for config in FAILED_CONFIGS:
        rank = config['rank']
        te = config['te']
        lr = config['lr']
        tlr = config['tlr']

        print(f"\n[rank={rank}, te={te}] Using lr={lr:.6f}, tlr={tlr:.6f} (from {config['source']})")
        print("-"*50)

        results = []
        for run_id in range(RUNS_PER_CONFIG):
            acc = run_training(config, run_id, train_loader, val_loader, use_wandb)
            results.append(acc)
            print(f"  Run {run_id}: {acc:.2f}%")

        print(f"  -> Best: {max(results):.2f}%, Avg: {sum(results)/len(results):.2f}%")

    print("\n" + "="*70)
    print("COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Re-run failed hybrid configurations with nearby TE hyperparameters.

Failed configs (accuracy < 90%) and nearby HP sources:
- Rank 4, TE=50: Try TE=10 (95.09%) and TE=100 (96.45%) HP
- Rank 16, TE=1: Try TE=10 (90.07%) HP
- Rank 16, TE=50: Try TE=10 (90.07%) and TE=100 (94.91%) HP
- Rank 32, TE=1: Try TE=10 (94.26%) HP
- Rank 32, TE=50: Try TE=10 (94.26%) and TE=100 (95.90%) HP
- Rank 64, TE=1,10,50,100: All use TE=500 (95.86%) HP

All runs use lifetime=46505 (sixt1c default).
Settings identical to sweep_hybrid_reinit.py: reinit_mode="hybrid", decay_factor=1.0
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
# Configuration (identical to sweep_hybrid_reinit.py)
# =============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LIFETIME = 46505  # sixt1c default
RUNS_PER_CONFIG = 10

# Failed configurations with NEARBY TE hyperparameters
FAILED_CONFIGS = [
    # Rank 4, TE=50: Try from TE=10 and TE=100
    {'rank': 4, 'te': 50, 'lr': 0.388981, 'tlr': 0.001048, 'source': 'te=10'},
    {'rank': 4, 'te': 50, 'lr': 0.341834, 'tlr': 0.003911, 'source': 'te=100'},

    # Rank 16, TE=1: Try from TE=10
    {'rank': 16, 'te': 1, 'lr': 0.140737, 'tlr': 0.085730, 'source': 'te=10'},

    # Rank 16, TE=50: Try from TE=10 and TE=100
    {'rank': 16, 'te': 50, 'lr': 0.140737, 'tlr': 0.085730, 'source': 'te=10'},
    {'rank': 16, 'te': 50, 'lr': 0.187206, 'tlr': 0.002161, 'source': 'te=100'},

    # Rank 32, TE=1: Try from TE=10
    {'rank': 32, 'te': 1, 'lr': 0.169914, 'tlr': 0.001749, 'source': 'te=10'},

    # Rank 32, TE=50: Try from TE=10 and TE=100
    {'rank': 32, 'te': 50, 'lr': 0.169914, 'tlr': 0.001749, 'source': 'te=10'},
    {'rank': 32, 'te': 50, 'lr': 0.993416, 'tlr': 0.001219, 'source': 'te=100'},

    # Rank 64: All failed TEs use TE=500 HP
    {'rank': 64, 'te': 1, 'lr': 0.252012, 'tlr': 0.001336, 'source': 'te=500'},
    {'rank': 64, 'te': 10, 'lr': 0.252012, 'tlr': 0.001336, 'source': 'te=500'},
    {'rank': 64, 'te': 50, 'lr': 0.252012, 'tlr': 0.001336, 'source': 'te=500'},
    {'rank': 64, 'te': 100, 'lr': 0.252012, 'tlr': 0.001336, 'source': 'te=500'},
]

# SoftBounds config (no noise) - identical to sweep_hybrid_reinit.py
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


def run_training(config, run_id, train_loader, val_loader, use_wandb):
    """Run single training with early stopping."""
    rank = config['rank']
    te = config['te']
    lr = config['lr']
    tlr = config['tlr']
    source = config['source']

    model = create_model(rank, te, LIFETIME, lr, tlr)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    run_name = f"rerun_hybrid_r{rank}_te{te}_from{source}_run{run_id}"

    if use_wandb:
        wandb.init(
            project="lrtt-hybrid-sweep",
            name=run_name,
            tags=[f"rank={rank}", f"te={te}", f"lifetime={LIFETIME}",
                  "hybrid", "rerun", "bias=True", f"source={source}"],
            config={"rank": rank, "te": te, "lifetime": LIFETIME,
                    "lr": lr, "tlr": tlr, "run": run_id,
                    "mode": "hybrid", "rerun": True, "source": source},
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

        # Early stopping based on low accuracy at epoch 5 (clearly not converging)
        if epoch >= 5 and best_acc < 50.0:
            print(f"    Early stopped at epoch {epoch} (low accuracy)")
            break

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"    Early stopped at epoch {epoch}")
            break

    if use_wandb:
        wandb.log({"best_acc": best_acc, "epochs_trained": epochs_trained,
                   "early_stopped": epochs_trained < EPOCHS})
        wandb.finish()

    del model
    torch.cuda.empty_cache()

    return best_acc


def main():
    print("="*70)
    print("Re-running Failed Hybrid Configurations")
    print("Using nearby TE hyperparameters from successful configs")
    print("="*70)
    print(f"Lifetime: {LIFETIME} (sixt1c default)")
    print(f"Mode: hybrid reinit (A=0, B unchanged, decay_factor=1.0)")
    print(f"Runs per config: {RUNS_PER_CONFIG}")
    print(f"Total configs: {len(FAILED_CONFIGS)}")
    print()

    train_loader, val_loader = load_data()
    use_wandb = WANDB_AVAILABLE

    all_results = []

    for config in FAILED_CONFIGS:
        rank = config['rank']
        te = config['te']
        lr = config['lr']
        tlr = config['tlr']
        source = config['source']

        print(f"\n[rank={rank}, te={te}] Using HP from {source}: lr={lr:.6f}, tlr={tlr:.6f}")
        print("-"*50)

        results = []
        for run_id in range(RUNS_PER_CONFIG):
            acc = run_training(config, run_id, train_loader, val_loader, use_wandb)
            results.append(acc)
            print(f"  Run {run_id}: {acc:.2f}%")

        best = max(results)
        avg = sum(results) / len(results)
        print(f"  -> Best: {best:.2f}%, Avg: {avg:.2f}%")

        all_results.append({
            'rank': rank, 'te': te, 'source': source,
            'lr': lr, 'tlr': tlr,
            'results': results, 'best': best, 'avg': avg,
            'status': 'SUCCESS' if best >= 90 else 'FAILED'
        })

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in all_results:
        status = "OK" if r['status'] == 'SUCCESS' else "XX"
        print(f"{status} rank={r['rank']}, te={r['te']} (from {r['source']}): Best={r['best']:.2f}%, Avg={r['avg']:.2f}%")

    print("\n" + "="*70)
    print("COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()

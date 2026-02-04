#!/usr/bin/env python3
"""
Lifetime sweep: 10^3, 10^2, 10^1 with rank=32, te=1, SoftBoundsDevice
"""

import os
os.environ["LRTT_SILENT"] = "1"

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time
import math

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset

# ============================================================================
# CONFIG
# ============================================================================
RANK = 32
TE = 1
LR = 0.089054      # wandb 최적
TLR = 0.001277     # wandb 최적

LIFETIMES = [1000, 100, 10]  # 10^3, 10^2, 10^1

BATCH_SIZE = 64
EPOCHS = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# SoftBoundsDevice (NO NOISE)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001,
    'w_max': 1.0,
    'w_min': -1.0,
    'dw_min_dtod': 0.0,
    'dw_min_std': 0.0,
    'up_down': 0.0,
    'up_down_dtod': 0.0,
    'w_max_dtod': 0.0,
    'w_min_dtod': 0.0,
    'write_noise_std': 0.0,
    'mult_noise': True,
}


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    """lifetime을 dt_batch_sec로 변환

    sixt1c_ab에서: delta = 1 - exp(-dt/TAU), lifetime = 1/delta
    따라서: dt = -TAU * ln(1 - 1/lifetime)
    """
    TAU_SEC = 46505.0
    if lifetime <= 1:
        return float('inf')
    delta = 1.0 / lifetime
    dt = -TAU_SEC * math.log(1 - delta)
    return dt


def create_model(lifetime: float):
    """Create LRTT model with specific lifetime"""

    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)

    device_config = PythonLRTTPreset.sixt1c_ab(
        rank=RANK,
        transfer_every=TE,
        lora_alpha=1.0,
        dt_batch_sec=dt_batch_sec,
        include_retention=True,
        c_device=c_device,
        reinit_mode="decay",
        decay_factor=1.0,
    )

    device_config.transfer_lr = TLR
    device_config.reinit_gain = 0.1
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    model = AnalogSequential(
        AnalogLinear(784, 256, bias=False, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    )

    return model, device_config


def train_and_evaluate(lifetime: float):
    """Train and evaluate with specific lifetime"""

    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)

    print(f"\n{'='*70}")
    print(f"LIFETIME = {lifetime} (10^{int(math.log10(lifetime))})")
    print(f"dt_batch_sec = {dt_batch_sec:.2f}")
    print(f"rank={RANK}, te={TE}, lr={LR}, tlr={TLR}")
    print(f"{'='*70}")

    model, device_config = create_model(lifetime)
    model.to(DEVICE)

    # Verify lifetime
    actual_lt = device_config.unit_cell_devices[0].lifetime
    print(f"Actual lifetime in config: {actual_lt:.2f}")

    # DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform)
    val_set = datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Optimizer
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    # Training
    best_val_acc = 0.0
    start_time = time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
                target = target.to(DEVICE, non_blocking=True)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        val_acc = 100.0 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        scheduler.step()
        print(f"  Epoch {epoch}/{EPOCHS}: {val_acc:.2f}% (best={best_val_acc:.2f}%)")

    total_time = time() - start_time

    return {
        'lifetime': lifetime,
        'dt_batch_sec': dt_batch_sec,
        'best_val_acc': best_val_acc,
        'train_time': total_time,
    }


def main():
    print("=" * 70)
    print("LIFETIME SWEEP: SoftBoundsDevice, rank=32, te=1")
    print(f"Lifetimes: {LIFETIMES}")
    print(f"lr={LR}, tlr={TLR}")
    print("=" * 70)

    results = []

    for lifetime in LIFETIMES:
        result = train_and_evaluate(lifetime)
        results.append(result)
        print(f"\n>>> lifetime={lifetime}: best_acc={result['best_val_acc']:.2f}%\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'lifetime':<12} {'dt_batch':<12} {'best_acc':<12} {'time':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['lifetime']:<12} {r['dt_batch_sec']:<12.2f} {r['best_val_acc']:<12.2f} {r['train_time']:<12.1f}s")

    # 이전 실험 비교 (lifetime=46505, acc=96.81%)
    print("\n" + "-" * 50)
    print("비교: 이전 실험 (lifetime=46505): 96.81%")
    print("-" * 50)


if __name__ == "__main__":
    main()

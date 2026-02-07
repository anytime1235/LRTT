#!/usr/bin/env python3
"""Single trial: rank=4 best decay config with SoftBounds classifier layer.

Based on sweep_softbounds_lifetime.py with identical device settings.
Change: classifier layer (256->10) uses SoftBoundsDevice instead of FloatingPointRPUConfig.

Config: rank=4, te=100, lifetime=46505, lr=0.493706, tlr=0.011245
"""

import os
os.environ["LRTT_SILENT"] = "1"

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =============================================================================
# Configuration — identical to sweep_softbounds_lifetime.py
# =============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5

# Rank 4 best config from decay sweep
RANK = 4
TE = 100
LIFETIME = 46505
LR = 0.493706
TLR = 0.011245

# SoftBounds config (no noise) — same as C tile
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    dt = -TAU_SEC * math.log(1 - delta)
    return dt


def load_data():
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


def create_model():
    dt_batch_sec = lifetime_to_dt_batch_sec(LIFETIME)

    TAU_SEC = 46505.0
    if dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        ab_lifetime = 1.0 / delta
    else:
        ab_lifetime = 0.0

    # A/B tiles: 6T1C LinearStepDevice (identical to sweep_softbounds_lifetime.py)
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )

    # C tile: SoftBounds (no noise)
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    # LRTT layer (784 -> 256)
    device_config = PythonLRTTDevice(
        rank=RANK, transfer_every=TE, lora_alpha=1.0,
        reinit_gain=0.1, reinit_mode="decay", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = TLR
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    lrtt_rpu = PythonLRTTRPUConfig(device=device_config)

    # Classifier layer: SoftBounds (same as C tile) instead of FloatingPoint
    classifier_rpu = SingleRPUConfig(device=SoftBoundsDevice(**SOFTBOUNDS_CONFIG))

    model = AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=lrtt_rpu),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=classifier_rpu),
        nn.LogSoftmax(dim=1),
    )
    model.to(DEVICE)
    return model


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
        target = target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, val_loader):
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


def main():
    print("=" * 70)
    print("Single Trial: Rank=4 Decay Config + SoftBounds Classifier")
    print("=" * 70)
    print(f"rank={RANK}, te={TE}, lifetime={LIFETIME}")
    print(f"lr={LR}, tlr={TLR}")
    print(f"Classifier: SoftBounds (same as C tile)")
    print(f"Baseline (FloatingPoint classifier): 97.34%")
    print()

    train_loader, val_loader = load_data()

    if WANDB_AVAILABLE:
        wandb.init(
            project="lrtt-softbounds-sweep",
            name="trial_r4_te100_softbound_classifier",
            tags=["rank=4", "te=100", f"lifetime={LIFETIME}",
                  "SoftBounds", "decay", "softbound_classifier", "trial"],
            config={"rank": RANK, "te": TE, "lifetime": LIFETIME,
                    "lr": LR, "tlr": TLR,
                    "classifier": "SoftBounds", "mode": "decay"},
        )

    model = create_model()
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        print(f"  Epoch {epoch:2d}: loss={train_loss:.4f}  val_acc={val_acc:.2f}%  best={best_acc:.2f}%{marker}")

        if WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "train_loss": train_loss,
                       "val_acc": val_acc, "best_acc": best_acc})

        if epoch >= 5 and best_acc < 50.0:
            print(f"  Early stopped (low accuracy)")
            break

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopped (patience={EARLY_STOP_PATIENCE})")
            break

    print(f"\nFinal best accuracy: {best_acc:.2f}%")
    print(f"Baseline (FloatingPoint classifier): 97.34%")
    print(f"Difference: {best_acc - 97.34:+.2f}%")

    if WANDB_AVAILABLE:
        wandb.log({"final_best_acc": best_acc})
        wandb.finish()


if __name__ == "__main__":
    main()

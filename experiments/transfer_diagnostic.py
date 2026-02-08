#!/usr/bin/env python3
"""Transfer Diagnostic: Track per-transfer metrics over 30 epochs.

Investigates whether LRTT transfers become meaningless in later epochs —
specifically, whether C stops changing despite transfers being triggered.

Runs a single 30-epoch training with fixed hyperparameters (from sweep_hybrid_reinit.py
best config for rank=4, te=100, lifetime=46505) and monkey-patches
controller.ab_weight_transfer() to record per-transfer metrics for A, B, C tiles.

Metrics tracked per transfer:
- a_norm: ‖A[:, :rank]‖_F
- b_norm: ‖B[:rank, :]‖_F
- ab_magnitude: ‖tlr × A @ B‖_F  (intended transfer signal)
- c_norm: ‖C‖_F before transfer
- delta_c_norm: ‖C_after - C_before‖_F  (actual change in C)
- delta_ratio: delta_c_norm / ab_magnitude  (transfer efficiency)
- unchanged_elem_ratio: fraction of C elements where |ΔC| < 1e-7
"""

import os

os.environ["LRTT_SILENT"] = "1"

import csv
import json
import math
from dataclasses import dataclass, fields
from time import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30

# Fixed hyperparameters
RANK = 4
TE = 100
LIFETIME = 46505
LR = 0.4937
TLR = 0.01125
REINIT_MODE = "decay"

# SoftBounds config (no noise)
SOFTBOUNDS_CONFIG = {
    "dw_min": 0.001,
    "w_max": 1.0,
    "w_min": -1.0,
    "dw_min_dtod": 0.0,
    "dw_min_std": 0.0,
    "up_down": 0.0,
    "up_down_dtod": 0.0,
    "w_max_dtod": 0.0,
    "w_min_dtod": 0.0,
    "write_noise_std": 0.0,
    "mult_noise": True,
}


# =============================================================================
# Transfer Tracker
# =============================================================================


@dataclass
class TransferRecord:
    """Single transfer event record."""

    epoch: int
    batch_idx: int
    layer_name: str
    a_norm: float
    b_norm: float
    ab_magnitude: float
    c_norm: float
    delta_c_norm: float
    delta_ratio: float
    unchanged_elem_ratio: float
    # Signal vs noise metrics
    cosine_sim: float  # cosine(AB_intended, delta_C): 1=perfect signal, 0=pure noise
    sign_agree_all: float  # fraction of all elements with matching sign
    sign_agree_changed: float  # fraction of CHANGED elements with matching sign
    signal_ratio_changed: float  # among changed elements: |projection onto AB| / |delta_c|

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "layer_name": self.layer_name,
            "a_norm": self.a_norm,
            "b_norm": self.b_norm,
            "ab_magnitude": self.ab_magnitude,
            "c_norm": self.c_norm,
            "delta_c_norm": self.delta_c_norm,
            "delta_ratio": self.delta_ratio,
            "unchanged_elem_ratio": self.unchanged_elem_ratio,
            "cosine_sim": self.cosine_sim,
            "sign_agree_all": self.sign_agree_all,
            "sign_agree_changed": self.sign_agree_changed,
            "signal_ratio_changed": self.signal_ratio_changed,
        }


class TransferTracker:
    """Collects per-transfer diagnostic metrics."""

    def __init__(self) -> None:
        self.records: list[TransferRecord] = []
        self.current_epoch: int = 0
        self.current_batch: int = 0

    def add(self, record: TransferRecord) -> None:
        self.records.append(record)

    def epoch_summary(self, epoch: int) -> dict:
        """Compute summary statistics for a given epoch."""
        epoch_records = [r for r in self.records if r.epoch == epoch]
        if not epoch_records:
            return {"epoch": epoch, "n_transfers": 0}

        def _stats(values: list[float]) -> dict:
            return {
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
            }

        return {
            "epoch": epoch,
            "n_transfers": len(epoch_records),
            "a_norm": _stats([r.a_norm for r in epoch_records]),
            "b_norm": _stats([r.b_norm for r in epoch_records]),
            "ab_magnitude": _stats([r.ab_magnitude for r in epoch_records]),
            "c_norm": _stats([r.c_norm for r in epoch_records]),
            "delta_c_norm": _stats([r.delta_c_norm for r in epoch_records]),
            "delta_ratio": _stats([r.delta_ratio for r in epoch_records]),
            "unchanged_elem_ratio": _stats([r.unchanged_elem_ratio for r in epoch_records]),
            "cosine_sim": _stats([r.cosine_sim for r in epoch_records]),
            "sign_agree_all": _stats([r.sign_agree_all for r in epoch_records]),
            "sign_agree_changed": _stats([r.sign_agree_changed for r in epoch_records]),
            "signal_ratio_changed": _stats([r.signal_ratio_changed for r in epoch_records]),
        }

    def all_records_as_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self.records]


# =============================================================================
# Monkey-patching
# =============================================================================


def patch_controller(controller, tracker: TransferTracker, layer_name: str) -> None:
    """Monkey-patch controller.ab_weight_transfer to record metrics."""
    original_transfer = controller.ab_weight_transfer
    rank = controller.rank
    tlr = controller.transfer_lr

    def tracked_transfer(method=None):
        # --- Before transfer: read A, B, C ---
        A = controller.tile_a.get_weights()[0]  # [d_size, rank_padded]
        B = controller.tile_b.get_weights()[0]  # [rank_padded, x_size]
        C_before = controller.tile_c.get_weights()[0].clone()  # [d_size, x_size]

        # Only use active rank columns/rows
        A_lr = A[:, :rank]  # [d_size, rank]
        B_lr = B[:rank, :]  # [rank, x_size]

        a_norm = torch.linalg.norm(A_lr).item()
        b_norm = torch.linalg.norm(B_lr).item()
        ab_product = tlr * (A_lr @ B_lr)
        ab_magnitude = torch.linalg.norm(ab_product).item()
        c_norm = torch.linalg.norm(C_before).item()

        # --- Perform actual transfer ---
        original_transfer(method=method)

        # --- After transfer: measure C change ---
        C_after = controller.tile_c.get_weights()[0]
        delta_c = C_after - C_before
        delta_c_norm = torch.linalg.norm(delta_c).item()

        # Transfer efficiency
        delta_ratio = delta_c_norm / ab_magnitude if ab_magnitude > 1e-12 else 0.0

        # Fraction of unchanged elements
        changed_mask = delta_c.abs() >= 1e-7
        unchanged = 1.0 - changed_mask.float().mean().item()

        # --- Signal vs noise analysis ---
        ab_flat = ab_product.flatten()
        dc_flat = delta_c.flatten()

        # Cosine similarity: how aligned is actual change with intended signal
        cos_denom = torch.linalg.norm(ab_flat) * torch.linalg.norm(dc_flat)
        cosine_sim = (torch.dot(ab_flat, dc_flat) / cos_denom).item() if cos_denom > 1e-12 else 0.0

        # Sign agreement (all elements where both are nonzero)
        both_nonzero = (ab_flat.abs() > 1e-12) & (dc_flat.abs() > 1e-12)
        if both_nonzero.sum() > 0:
            sign_agree_all = (
                (ab_flat[both_nonzero].sign() == dc_flat[both_nonzero].sign()).float().mean().item()
            )
        else:
            sign_agree_all = 0.0

        # Sign agreement among CHANGED elements only
        changed_flat = changed_mask.flatten()
        changed_and_signal = changed_flat & (ab_flat.abs() > 1e-12)
        if changed_and_signal.sum() > 0:
            sign_agree_changed = (
                (ab_flat[changed_and_signal].sign() == dc_flat[changed_and_signal].sign())
                .float()
                .mean()
                .item()
            )
        else:
            sign_agree_changed = 0.0

        # Signal ratio among changed elements:
        # What fraction of delta_c energy is explained by AB direction?
        if changed_flat.sum() > 0 and ab_magnitude > 1e-12:
            dc_changed = dc_flat[changed_flat]
            ab_changed = ab_flat[changed_flat]
            proj = torch.dot(dc_changed, ab_changed) / torch.linalg.norm(ab_changed)
            signal_ratio_changed = abs(proj.item()) / torch.linalg.norm(dc_changed).item()
        else:
            signal_ratio_changed = 0.0

        record = TransferRecord(
            epoch=tracker.current_epoch,
            batch_idx=tracker.current_batch,
            layer_name=layer_name,
            a_norm=a_norm,
            b_norm=b_norm,
            ab_magnitude=ab_magnitude,
            c_norm=c_norm,
            delta_c_norm=delta_c_norm,
            delta_ratio=delta_ratio,
            unchanged_elem_ratio=unchanged,
            cosine_sim=cosine_sim,
            sign_agree_all=sign_agree_all,
            sign_agree_changed=sign_agree_changed,
            signal_ratio_changed=signal_ratio_changed,
        )
        tracker.add(record)

    controller.ab_weight_transfer = tracked_transfer


# =============================================================================
# Model creation (from sweep_hybrid_reinit.py)
# =============================================================================


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    """Convert lifetime to dt_batch_sec for sixt1c_ab preset."""
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    dt = -TAU_SEC * math.log(1 - delta)
    return dt


def create_model() -> AnalogSequential:
    """Create LRTT model with hybrid reinit (A=0, B unchanged)."""
    dt_batch_sec = lifetime_to_dt_batch_sec(LIFETIME)

    # Calculate lifetime for A/B tiles
    TAU_SEC = 46505.0
    if dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        ab_lifetime = 1.0 / delta
    else:
        ab_lifetime = 0.0

    # A/B tiles: 6T1C LinearStepDevice
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

    # PythonLRTTDevice with configurable reinit
    device_config = PythonLRTTDevice(
        rank=RANK,
        transfer_every=TE,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode=REINIT_MODE,
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = TLR
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


# =============================================================================
# Data loading
# =============================================================================


def load_data() -> tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform)
    val_set = datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader


# =============================================================================
# Training & Validation
# =============================================================================


def train_epoch(
    model: AnalogSequential,
    train_loader: DataLoader,
    optimizer: AnalogSGD,
    criterion: nn.Module,
    tracker: TransferTracker,
    epoch: int,
) -> None:
    """Train for one epoch, updating tracker batch counter."""
    model.train()
    tracker.current_epoch = epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        tracker.current_batch = batch_idx
        data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
        target = target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def validate(model: AnalogSequential, val_loader: DataLoader) -> float:
    """Validate model, return accuracy percentage."""
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


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 70)
    print("LRTT Transfer Diagnostic")
    print("=" * 70)
    print(
        f"Config: rank={RANK}, te={TE}, lifetime={LIFETIME}, lr={LR}, tlr={TLR}, reinit={REINIT_MODE}"
    )
    print(f"Epochs: {EPOCHS}, batch_size={BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print()

    # Load data
    train_loader, val_loader = load_data()
    n_batches = len(train_loader)
    expected_transfers_per_epoch = n_batches // TE
    print(f"Batches/epoch: {n_batches}")
    print(f"Expected transfers/epoch: ~{expected_transfers_per_epoch}")
    print()

    # Create model
    model = create_model()

    # Set up tracker and patch controllers
    tracker = TransferTracker()
    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LRTTSimulatorTile):
            patch_controller(module.controller, tracker, layer_name=name)
            patched_count += 1
            print(f"Patched: {name}")

    print(f"Total patched tiles: {patched_count}")
    print()

    # Optimizer & scheduler
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    # Training loop
    best_acc = 0.0
    epoch_acc: list[tuple[float, float]] = []  # (val_acc, best_acc) per epoch

    for epoch in range(1, EPOCHS + 1):
        t0 = time()
        train_epoch(model, train_loader, optimizer, criterion, tracker, epoch)
        val_acc = validate(model, val_loader)
        scheduler.step()
        best_acc = max(best_acc, val_acc)
        epoch_acc.append((val_acc, best_acc))
        elapsed = time() - t0

        n = sum(1 for r in tracker.records if r.epoch == epoch)
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | acc={val_acc:.2f}% (best={best_acc:.2f}%) "
            f"| transfers={n} | {elapsed:.1f}s"
        )

    # --- Save per-transfer CSV ---
    transfer_csv = f"/root/LRTT/experiments/transfer_diagnostic_{REINIT_MODE}_transfers.csv"
    fieldnames = [f.name for f in fields(TransferRecord)]
    with open(transfer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in tracker.records:
            writer.writerow(rec.to_dict())

    # --- Save per-epoch summary CSV ---
    epoch_csv = f"/root/LRTT/experiments/transfer_diagnostic_{REINIT_MODE}_epochs.csv"
    metric_keys = [
        "a_norm",
        "b_norm",
        "ab_magnitude",
        "c_norm",
        "delta_c_norm",
        "delta_ratio",
        "unchanged_elem_ratio",
        "cosine_sim",
        "sign_agree_all",
        "sign_agree_changed",
        "signal_ratio_changed",
    ]
    epoch_fieldnames = ["epoch", "val_acc", "best_acc", "n_transfers"]
    for key in metric_keys:
        for stat in ["mean", "min", "max"]:
            epoch_fieldnames.append(f"{key}_{stat}")

    with open(epoch_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_fieldnames)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            summary = tracker.epoch_summary(epoch)
            va, ba = epoch_acc[epoch - 1]
            row: dict = {
                "epoch": epoch,
                "val_acc": va,
                "best_acc": ba,
                "n_transfers": summary["n_transfers"],
            }
            for key in metric_keys:
                stats = summary.get(key, {})
                for stat in ["mean", "min", "max"]:
                    row[f"{key}_{stat}"] = stats.get(stat, "")
            writer.writerow(row)

    print()
    print(f"Done. {len(tracker.records)} transfers recorded.")
    print(f"  Per-transfer CSV: {transfer_csv}")
    print(f"  Per-epoch CSV:    {epoch_csv}")
    print(f"  Final accuracy: {val_acc:.2f}%, Best: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

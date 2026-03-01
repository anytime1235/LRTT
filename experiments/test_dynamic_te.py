#!/usr/bin/env python3
"""Dynamic TE A/B Test with Transfer Diagnostics.

Based on transfer_diagnostic.py — same model, same hyperparameters.
Sweeps multiple dynamic_te_power values (+ baseline) and saves per-transfer
diagnostic CSVs for each condition.

Config from sweep_hybrid_reinit.py best for rank=4, te=100, lifetime=46505:
  lr=0.4937, tlr=0.01125, reinit_mode="hybrid"
"""

import os

os.environ["LRTT_SILENT"] = "1"

import csv
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

RANK = 4
TE = 100
LIFETIME = 46505
LR = 0.4937
TLR = 0.01125
REINIT_MODE = "hybrid"

SOFTBOUNDS_CONFIG = {
    "dw_min": 0.001, "w_max": 1.0, "w_min": -1.0,
    "dw_min_dtod": 0.0, "dw_min_std": 0.0, "up_down": 0.0,
    "up_down_dtod": 0.0, "w_max_dtod": 0.0, "w_min_dtod": 0.0,
    "write_noise_std": 0.0, "mult_noise": True,
}

# Sweep these power values (None = baseline/static TE)
POWER_SWEEP = [None, 0.5, 1.0, 2.0]

OUTPUT_DIR = "/data/LRTT_transformer/experiments/dynamic_te_results"


# =============================================================================
# Transfer Tracker (from transfer_diagnostic.py)
# =============================================================================

@dataclass
class TransferRecord:
    condition: str
    epoch: int
    batch_idx: int
    transfer_idx: int
    current_te: int
    current_lr: float
    a_norm: float
    b_norm: float
    ab_magnitude: float
    c_norm: float
    delta_c_norm: float
    delta_ratio: float
    unchanged_elem_ratio: float
    cosine_sim: float

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class TransferTracker:
    def __init__(self):
        self.records: list[TransferRecord] = []
        self.current_epoch: int = 0
        self.current_batch: int = 0

    def add(self, record: TransferRecord):
        self.records.append(record)


def patch_controller(controller, tracker: TransferTracker, condition: str):
    """Monkey-patch controller.ab_weight_transfer to record diagnostics."""
    original_transfer = controller.ab_weight_transfer
    rank = controller.rank
    tlr = controller.transfer_lr
    transfer_count = [0]

    def tracked_transfer(method=None):
        A = controller.tile_a.get_weights()[0][:, :rank]
        B = controller.tile_b.get_weights()[0][:rank, :]
        C_before = controller.tile_c.get_weights()[0].clone()

        a_norm = torch.linalg.norm(A).item()
        b_norm = torch.linalg.norm(B).item()
        ab_product = tlr * (A @ B)
        ab_magnitude = torch.linalg.norm(ab_product).item()
        c_norm = torch.linalg.norm(C_before).item()

        original_transfer(method=method)

        C_after = controller.tile_c.get_weights()[0]
        delta_c = C_after - C_before
        delta_c_norm = torch.linalg.norm(delta_c).item()

        delta_ratio = delta_c_norm / ab_magnitude if ab_magnitude > 1e-12 else 0.0

        changed_mask = delta_c.abs() >= 1e-7
        unchanged = 1.0 - changed_mask.float().mean().item()

        ab_flat = ab_product.flatten()
        dc_flat = delta_c.flatten()
        cos_denom = torch.linalg.norm(ab_flat) * torch.linalg.norm(dc_flat)
        cosine_sim = (torch.dot(ab_flat, dc_flat) / cos_denom).item() if cos_denom > 1e-12 else 0.0

        transfer_count[0] += 1

        tracker.add(TransferRecord(
            condition=condition,
            epoch=tracker.current_epoch,
            batch_idx=tracker.current_batch,
            transfer_idx=transfer_count[0],
            current_te=controller.transfer_every,
            current_lr=controller.lr_peak if controller.lr_peak else 0.0,
            a_norm=a_norm,
            b_norm=b_norm,
            ab_magnitude=ab_magnitude,
            c_norm=c_norm,
            delta_c_norm=delta_c_norm,
            delta_ratio=delta_ratio,
            unchanged_elem_ratio=unchanged,
            cosine_sim=cosine_sim,
        ))

    controller.ab_weight_transfer = tracked_transfer


# =============================================================================
# Model creation
# =============================================================================

def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_model(dynamic_te: bool = False, power: float = 1.0) -> AnalogSequential:
    dt_batch_sec = lifetime_to_dt_batch_sec(LIFETIME)
    TAU_SEC = 46505.0
    ab_lifetime = 1.0 / (1 - math.exp(-dt_batch_sec / TAU_SEC)) if dt_batch_sec > 0 else 0.0

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
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=RANK, transfer_every=TE,
        lora_alpha=1.0, reinit_gain=0.1,
        reinit_mode=REINIT_MODE, decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
        dynamic_te=dynamic_te,
        dynamic_te_power=power,
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

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform)
    val_set = datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


# =============================================================================
# Training
# =============================================================================

def get_controller(model):
    for module in model.modules():
        if isinstance(module, LRTTSimulatorTile):
            return module.controller
    return None


def run_experiment(condition: str, dynamic_te: bool, power: float,
                   train_loader, val_loader, tracker: TransferTracker):
    """Run single 30-epoch experiment with diagnostic tracking."""
    print(f"\n{'='*60}")
    print(f"  {condition}")
    print(f"{'='*60}")

    model = create_model(dynamic_te=dynamic_te, power=power)
    controller = get_controller(model)

    te_max = controller.dynamic_te_max
    print(f"  dynamic_te={dynamic_te}, power={power}, te_max={te_max}")

    # Patch for diagnostics
    patch_controller(controller, tracker, condition)

    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    results = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time()
        tracker.current_epoch = epoch
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            tracker.current_batch = batch_idx
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

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
        val_acc = 100.0 * correct / total
        best_acc = max(best_acc, val_acc)

        current_te = controller.transfer_every
        current_lr = optimizer.param_groups[0]['lr']
        n_epoch_transfers = sum(1 for r in tracker.records
                                if r.condition == condition and r.epoch == epoch)
        elapsed = time() - t0

        results.append({
            'epoch': epoch, 'val_acc': val_acc, 'best_acc': best_acc,
            'te': current_te, 'lr': current_lr,
            'n_transfers': n_epoch_transfers,
        })

        print(f"  Epoch {epoch:2d}/{EPOCHS} | acc={val_acc:.2f}% (best={best_acc:.2f}%) "
              f"| TE={current_te:>5d} | xfers={n_epoch_transfers:>2d} | lr={current_lr:.5f} | {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Dynamic TE Sweep with Transfer Diagnostics")
    print("=" * 60)
    print(f"Config: rank={RANK}, te={TE}, lifetime={LIFETIME}")
    print(f"  lr={LR}, tlr={TLR}, reinit={REINIT_MODE}")
    print(f"  power sweep: {POWER_SWEEP}")
    print(f"Device: {DEVICE}")

    train_loader, val_loader = load_data()
    n_batches = len(train_loader)
    print(f"Batches/epoch: {n_batches}")

    tracker = TransferTracker()
    all_results = {}

    for power in POWER_SWEEP:
        if power is None:
            condition = "BASELINE"
            dynamic_te = False
            p = 1.0  # Ignored when dynamic_te=False, but must be positive for validation
        else:
            condition = f"POWER_{power}"
            dynamic_te = True
            p = power

        results = run_experiment(condition, dynamic_te, p, train_loader, val_loader, tracker)
        all_results[condition] = results

    # === Save per-transfer CSV ===
    transfer_csv = os.path.join(OUTPUT_DIR, "transfer_diagnostics.csv")
    fieldnames = [f.name for f in fields(TransferRecord)]
    with open(transfer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in tracker.records:
            writer.writerow(rec.to_dict())

    # === Save per-epoch CSV ===
    epoch_csv = os.path.join(OUTPUT_DIR, "epoch_summary.csv")
    with open(epoch_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "epoch", "val_acc", "best_acc", "te", "lr",
            "n_transfers", "mean_ab_mag", "mean_delta_c", "mean_delta_ratio",
            "mean_cosine_sim", "mean_unchanged_ratio",
        ])
        writer.writeheader()
        for condition, results in all_results.items():
            for r in results:
                epoch = r['epoch']
                epoch_recs = [rec for rec in tracker.records
                              if rec.condition == condition and rec.epoch == epoch]
                if epoch_recs:
                    mean_ab = sum(rec.ab_magnitude for rec in epoch_recs) / len(epoch_recs)
                    mean_dc = sum(rec.delta_c_norm for rec in epoch_recs) / len(epoch_recs)
                    mean_dr = sum(rec.delta_ratio for rec in epoch_recs) / len(epoch_recs)
                    mean_cs = sum(rec.cosine_sim for rec in epoch_recs) / len(epoch_recs)
                    mean_ur = sum(rec.unchanged_elem_ratio for rec in epoch_recs) / len(epoch_recs)
                else:
                    mean_ab = mean_dc = mean_dr = mean_cs = mean_ur = 0.0

                writer.writerow({
                    "condition": condition,
                    "epoch": epoch,
                    "val_acc": r['val_acc'],
                    "best_acc": r['best_acc'],
                    "te": r['te'],
                    "lr": r['lr'],
                    "n_transfers": r['n_transfers'],
                    "mean_ab_mag": f"{mean_ab:.4f}",
                    "mean_delta_c": f"{mean_dc:.4f}",
                    "mean_delta_ratio": f"{mean_dr:.4f}",
                    "mean_cosine_sim": f"{mean_cs:.4f}",
                    "mean_unchanged_ratio": f"{mean_ur:.4f}",
                })

    # === Print comparison table ===
    conditions = list(all_results.keys())
    print(f"\n{'='*80}")
    print("  COMPARISON TABLE")
    print(f"{'='*80}")
    header = f"{'Epoch':>5}"
    for c in conditions:
        short = c.replace("POWER_", "p=").replace("BASELINE", "static")
        header += f" | {short:>10}"
    header += f" | {'TE(p=1.0)':>9}"
    print(header)
    print("-" * 80)

    for epoch in range(EPOCHS):
        row = f"{epoch+1:5d}"
        for c in conditions:
            row += f" | {all_results[c][epoch]['val_acc']:10.2f}"
        # Show TE for power=1.0
        p1_key = "POWER_1.0"
        if p1_key in all_results:
            row += f" | {all_results[p1_key][epoch]['te']:>9d}"
        print(row)

    print("-" * 80)
    row_best = f"{'BEST':>5}"
    row_final = f"{'FINAL':>5}"
    for c in conditions:
        b = max(r['best_acc'] for r in all_results[c])
        f_acc = all_results[c][-1]['val_acc']
        row_best += f" | {b:10.2f}"
        row_final += f" | {f_acc:10.2f}"
    print(row_best)
    print(row_final)

    # === Print diagnostic summary (early/mid/late) ===
    print(f"\n{'='*80}")
    print("  TRANSFER DIAGNOSTIC SUMMARY (mean per phase)")
    print(f"{'='*80}")

    for condition in conditions:
        recs = [r for r in tracker.records if r.condition == condition]
        n = len(recs)
        if n == 0:
            continue
        chunk = max(1, n // 3)
        phases = [("early", recs[:chunk]), ("mid", recs[chunk:2*chunk]), ("late", recs[2*chunk:])]

        short = condition.replace("POWER_", "p=").replace("BASELINE", "static")
        print(f"\n  {short} ({n} transfers)")
        print(f"  {'Phase':>6} | {'AB_mag':>10} | {'delta_C':>10} | {'ratio':>10} | {'cos_sim':>10} | {'unchanged':>10} | {'TE':>6}")
        print(f"  " + "-" * 72)
        for phase_name, phase_recs in phases:
            if not phase_recs:
                continue
            m_ab = sum(r.ab_magnitude for r in phase_recs) / len(phase_recs)
            m_dc = sum(r.delta_c_norm for r in phase_recs) / len(phase_recs)
            m_dr = sum(r.delta_ratio for r in phase_recs) / len(phase_recs)
            m_cs = sum(r.cosine_sim for r in phase_recs) / len(phase_recs)
            m_ur = sum(r.unchanged_elem_ratio for r in phase_recs) / len(phase_recs)
            m_te = sum(r.current_te for r in phase_recs) / len(phase_recs)
            print(f"  {phase_name:>6} | {m_ab:10.2f} | {m_dc:10.4f} | {m_dr:10.4f} | {m_cs:10.4f} | {m_ur:10.4f} | {m_te:6.0f}")

    print(f"\nCSVs saved to: {OUTPUT_DIR}/")
    print(f"  - transfer_diagnostics.csv ({len(tracker.records)} records)")
    print(f"  - epoch_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()

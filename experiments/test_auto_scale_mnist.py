#!/usr/bin/env python3
"""MNIST A/B test: auto_scale and transfer_ema_scale individual/combined effects.

Uses IDENTICAL architecture & device config as sweep_softbounds_lifetime.py:
  - Model: MLP 784→256→10 (AnalogLinear + FloatingPoint output)
  - A/B tiles: LinearStepDevice (6T1C params, lifetime=46505)
  - C tile: SoftBoundsDevice (no noise)
  - Optimizer: AnalogSGD

Runs 4 conditions × 10 epochs on MNIST:
  - BASELINE:          auto_scale=False, transfer_ema=False
  - AUTO_SCALE_ONLY:   auto_scale=True,  transfer_ema=False
  - TRANSFER_EMA_ONLY: auto_scale=False, transfer_ema=True
  - BOTH:              auto_scale=True,  transfer_ema=True

Usage:
    python experiments/test_auto_scale_mnist.py
"""

import os
os.environ["LRTT_SILENT"] = "1"

import math
import json
import time
from datetime import datetime
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

# =============================================================================
# Configuration — sweep_softbounds_lifetime.py와 동일
# =============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
SEED = 42

# Fixed config: rank=4, te=100, lifetime=46505 (sixt1c)
RANK = 4
TE = 100
LIFETIME = 46505
# Hyperparams from sweep_softbounds_lifetime.py table (rank=4, te=100)
LR = 0.493706
TLR = 0.011245

# SoftBounds C tile config (no noise) — sweep_softbounds_lifetime.py 동일
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


# =============================================================================
# LRTT Config — sweep_softbounds_lifetime.py 동일 + auto_scale/transfer_ema 옵션
# =============================================================================

def create_lrtt_config(
    auto_scale: bool = False,
    transfer_ema_scale: bool = False,
    auto_scale_momentum: float = 0.99,
    transfer_ema_momentum: float = 0.99,
):
    dt_batch_sec = lifetime_to_dt_batch_sec(LIFETIME)

    TAU_SEC = 46505.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    ab_lifetime = 1.0 / delta if delta > 0 else 0.0

    # A/B tiles: 6T1C LinearStepDevice — sweep_softbounds_lifetime.py 동일
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

    # C tile: SoftBounds (no noise) — sweep_softbounds_lifetime.py 동일
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=RANK,
        transfer_every=TE,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
        # --- 테스트 대상 ---
        auto_scale=auto_scale,
        auto_scale_momentum=auto_scale_momentum,
        transfer_ema_scale=transfer_ema_scale,
        transfer_ema_momentum=transfer_ema_momentum,
        transfer_ema_target_norm=0.0,  # auto-calibrate
    )
    device_config.transfer_lr = TLR
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    return rpu_config


# =============================================================================
# Model & Data — sweep_softbounds_lifetime.py 동일
# =============================================================================

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


def create_model(rpu_config):
    model = AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    )
    return model.to(DEVICE)


# =============================================================================
# Diagnostic: collect EMA stats from all LRTT layers
# =============================================================================

def collect_lrtt_stats(model) -> Dict[str, Any]:
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, LRTTSimulatorTile):
            ctrl = module.controller
            layer_stats = {
                "num_transfers": ctrl.num_transfers,
                "num_a_updates": ctrl.num_a_updates,
            }
            if ctrl._ema_m_x is not None:
                layer_stats["ema_m_x"] = ctrl._ema_m_x.item()
                layer_stats["ema_m_d"] = ctrl._ema_m_d.item()
                layer_stats["ema_product"] = ctrl._ema_m_x.item() * ctrl._ema_m_d.item()
            if ctrl._ema_ab_norm is not None:
                layer_stats["ema_ab_norm"] = ctrl._ema_ab_norm.item()
                layer_stats["transfer_ema_target_norm"] = ctrl.transfer_ema_target_norm
            stats[name] = layer_stats
    return stats


# =============================================================================
# Train & Evaluate
# =============================================================================

def train_and_evaluate(condition_name, rpu_config, train_loader, val_loader):
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = create_model(rpu_config)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    # --- Train ---
    all_step_losses = []
    epoch_val_accs = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for data, target in train_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        all_step_losses.extend(epoch_losses)
        scheduler.step()

        # Validate
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
        epoch_val_accs.append(val_acc)
        print(f"    Epoch {epoch}/{EPOCHS}: train_loss={np.mean(epoch_losses):.4f}, val_acc={val_acc:.2f}%")

    train_time = time.time() - t0

    # --- Collect EMA stats ---
    lrtt_stats = collect_lrtt_stats(model)

    # --- Loss stability analysis ---
    window = 50
    window_means = [np.mean(all_step_losses[i:i+window]) for i in range(0, len(all_step_losses), window)]
    loss_trajectory_std = np.std(window_means) if len(window_means) > 1 else 0.0

    tail_losses = all_step_losses[-100:] if len(all_step_losses) >= 100 else all_step_losses
    tail_std = np.std(tail_losses)

    results = {
        "condition": condition_name,
        "best_val_acc": max(epoch_val_accs),
        "final_val_acc": epoch_val_accs[-1],
        "final_train_loss": np.mean(all_step_losses[-50:]) if len(all_step_losses) >= 50 else np.mean(all_step_losses),
        "train_loss_mean": np.mean(all_step_losses),
        "train_loss_std": np.std(all_step_losses),
        "loss_trajectory_std": loss_trajectory_std,
        "tail_loss_std": tail_std,
        "train_time_sec": train_time,
        "total_steps": len(all_step_losses),
        "epoch_val_accs": epoch_val_accs,
        "lrtt_stats": lrtt_stats,
    }

    # Print layer-wise EMA stats
    if lrtt_stats:
        print(f"\n    Layer-wise EMA Stats ({len(lrtt_stats)} layers):")
        for layer_name, ls in lrtt_stats.items():
            short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
            info = f"      {short_name}: transfers={ls['num_transfers']}"
            if "ema_product" in ls:
                info += f", ema(x*d)={ls['ema_product']:.6f}"
            if "ema_ab_norm" in ls:
                info += f", ema_ab_norm={ls['ema_ab_norm']:.6f}, target={ls['transfer_ema_target_norm']:.6f}"
            print(info)

    del model
    torch.cuda.empty_cache()
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Architecture: MLP 784→256→10 (same as sweep_softbounds_lifetime.py)")
    print(f"LRTT: rank={RANK}, te={TE}, lifetime={LIFETIME}, lr={LR}, tlr={TLR}")
    print(f"Epochs: {EPOCHS}, batch_size={BATCH_SIZE}, seed={SEED}")

    train_loader, val_loader = load_data()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # === 4 Conditions ===
    CONDITIONS = [
        ("BASELINE",          False, False),
        ("AUTO_SCALE_ONLY",   True,  False),
        ("TRANSFER_EMA_ONLY", False, True),
        ("BOTH",              True,  True),
    ]

    all_results = {}
    for name, auto_scale, transfer_ema in CONDITIONS:
        rpu_config = create_lrtt_config(auto_scale=auto_scale, transfer_ema_scale=transfer_ema)
        results = train_and_evaluate(name, rpu_config, train_loader, val_loader)
        all_results[name] = results

    # === Comparison Summary ===
    print("\n" + "=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)

    cond_names = [c[0] for c in CONDITIONS]
    header = f"{'Metric':<20}" + "".join(f"{n:>18}" for n in cond_names)
    print(f"\n  {header}")
    print(f"  {'-'*20}" + "".join(f"{'-'*18}" for _ in cond_names))

    metrics = [
        ("Best Val Acc (%)", "best_val_acc", False),
        ("Final Val Acc (%)", "final_val_acc", False),
        ("Final Train Loss", "final_train_loss", False),
        ("Train Loss Mean", "train_loss_mean", False),
        ("Loss Traj Std", "loss_trajectory_std", False),
        ("Tail Loss Std", "tail_loss_std", False),
        ("Train Time (s)", "train_time_sec", True),
    ]

    for metric_name, key, is_time in metrics:
        values = [all_results[n][key] for n in cond_names]
        if is_time:
            row = f"  {metric_name:<20}" + "".join(f"{v:>17.1f}s" for v in values)
        else:
            row = f"  {metric_name:<20}" + "".join(f"{v:>18.4f}" for v in values)
        print(row)

    # === Delta from BASELINE ===
    baseline = all_results["BASELINE"]
    print(f"\n  Delta from BASELINE:")
    print(f"  {'-'*20}" + "".join(f"{'-'*18}" for _ in cond_names[1:]))

    for metric_name, key, is_time in metrics:
        base_val = baseline[key]
        deltas = []
        for name in cond_names[1:]:
            delta = all_results[name][key] - base_val
            if is_time and base_val > 0:
                pct = (delta / base_val) * 100
                deltas.append(f"{pct:+17.1f}%")
            else:
                deltas.append(f"{delta:+18.4f}")
        row = f"  {metric_name:<20}" + "".join(deltas)
        print(row)

    # === NaN/Inf Check ===
    print(f"\n  NaN/Inf Check:")
    any_nan = False
    for cond_name in cond_names:
        has_nan = False
        for layer_name, ls in all_results[cond_name]["lrtt_stats"].items():
            for k, v in ls.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    print(f"    WARNING: [{cond_name}] {layer_name}.{k} = {v}")
                    has_nan = True
                    any_nan = True
        if not has_nan:
            print(f"    [{cond_name}] CLEAN")

    # === EMA Summary ===
    print(f"\n  EMA Statistics Summary:")
    for cond_name in cond_names:
        ema_products = []
        ema_ab_norms = []
        for _, ls in all_results[cond_name]["lrtt_stats"].items():
            if "ema_product" in ls:
                ema_products.append(ls["ema_product"])
            if "ema_ab_norm" in ls:
                ema_ab_norms.append(ls["ema_ab_norm"])

        print(f"\n    [{cond_name}]")
        if ema_products:
            print(f"      EMA(x*d): mean={np.mean(ema_products):.6f}, std={np.std(ema_products):.6f}")
        else:
            print(f"      EMA(x*d): N/A (auto_scale=False)")
        if ema_ab_norms:
            print(f"      EMA(||AB||): mean={np.mean(ema_ab_norms):.6f}, std={np.std(ema_ab_norms):.6f}")
        else:
            print(f"      EMA(||AB||): N/A (transfer_ema=False)")

    # === Save Results ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/data/LRTT/experiments/auto_scale_mnist_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Clean for JSON
    save_results = {}
    for cond_name in cond_names:
        r = dict(all_results[cond_name])
        for layer_name in list(r["lrtt_stats"].keys()):
            for k, v in list(r["lrtt_stats"][layer_name].items()):
                if isinstance(v, torch.Tensor):
                    r["lrtt_stats"][layer_name][k] = v.item()
        save_results[cond_name] = r

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n  Results saved to: {out_dir}/results.json")


if __name__ == "__main__":
    main()

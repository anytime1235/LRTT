#!/usr/bin/env python3
"""Per-tile auto_scale 검증: BASELINE vs GLOBAL vs PER_TILE vs PER_TILE_CAP10 비교.

핵심 검증 항목:
1. per_tile의 tile별 amplification이 이상적 값(A≈0.86x, B≈5.3x)에 근접하는지
2. 포화율이 BASELINE 수준으로 돌아오는지
3. 성능(val_acc)이 BASELINE을 유지하거나 개선되는지

4 conditions × 5 epochs × MNIST. 약 4분 소요.

Usage:
    cd /data/LRTT_transformer && python experiments/diagnose_auto_scale_pertile.py
"""

import os
os.environ["LRTT_SILENT"] = "1"

import csv
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

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
EPOCHS = 5
SEED = 42

RANK = 4
TE = 100
LIFETIME = 46505
LR = 0.493706
TLR = 0.011245

SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

# Sampling interval for update hook (every N steps)
UPDATE_SAMPLE_EVERY = 10

# Conditions: (name, auto_scale, auto_scale_mode, auto_scale_max_amplification, transfer_ema_scale)
CONDITIONS = [
    ("BASELINE",       False, "per_tile",   0.0,  False),  # auto_scale=OFF
    ("CALIBRATE",      True,  "calibrate",  0.0,  False),  # calibrate-then-freeze
    ("CAL+TEMS",       True,  "calibrate",  0.0,  True),   # calibrate + transfer_ema_scale
    ("TEMS_ONLY",      False, "per_tile",   0.0,  True),   # transfer_ema_scale only (no auto_scale)
]


# =============================================================================
# Config / Model / Data
# =============================================================================

def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(auto_scale: bool = False, auto_scale_mode: str = "per_tile",
                        auto_scale_max_amplification: float = 0.0,
                        transfer_ema_scale: bool = False):
    dt_batch_sec = lifetime_to_dt_batch_sec(LIFETIME)
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    ab_lifetime = 1.0 / delta if delta > 0 else 0.0

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
        rank=RANK,
        transfer_every=TE,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
        auto_scale=auto_scale,
        auto_scale_momentum=0.99,
        auto_scale_mode=auto_scale_mode,
        auto_scale_max_amplification=auto_scale_max_amplification,
        transfer_ema_scale=transfer_ema_scale,
        transfer_ema_momentum=0.99,
        transfer_ema_target_norm=0.0,
    )
    device_config.transfer_lr = TLR
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    return PythonLRTTRPUConfig(device=device_config)


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
# Hook Records
# =============================================================================

update_records: List[dict] = []
transfer_records: List[dict] = []


# =============================================================================
# Update Hook: measures per-tile amplification, saturation, and mismatch
# =============================================================================

def make_update_hook(ctrl, layer_name: str, condition: str):
    """Hook _ab_weight_update_lora to measure per-tile amplification and saturation."""
    original_fn = ctrl._ab_weight_update_lora
    step_counter = [0]

    def hooked_update(x, d, lr, in_trans=False, out_trans=False):
        step_counter[0] += 1
        should_log = (step_counter[0] % UPDATE_SAMPLE_EVERY == 1)

        if should_log:
            with torch.no_grad():
                # Normalize to [batch, feat]
                x_flat = x.t() if in_trans else x
                d_flat = d.t() if out_trans else d

                # --- What auto_scale tracks: original x, d ---
                max_x = x_flat.abs().amax().item()
                max_d = d_flat.abs().amax().item()
                product_tracked = max_x * max_d

                # --- What tile_a actually receives: (XB, d) ---
                XB = ctrl.tile_b.forward(x_flat)
                max_XB = XB.abs().amax().item()
                product_a = max_XB * max_d

                # --- What tile_b actually receives: (x, DA) ---
                DA = ctrl.tile_a.backward(d_flat)
                max_DA = DA.abs().amax().item()
                product_b = max_x * max_DA

                # --- A/B tile weight stats ---
                A_w = ctrl.tile_a.get_weights()[0].to(ctrl.device)
                B_w = ctrl.tile_b.get_weights()[0].to(ctrl.device)
                w_max = 1.0

                A_saturated_frac = (A_w.abs() > 0.9 * w_max).float().mean().item()
                B_saturated_frac = (B_w.abs() > 0.9 * w_max).float().mean().item()

        # Call original
        original_fn(x, d, lr, in_trans, out_trans)

        if should_log:
            # Read EMA state after update
            ema_m_x = ctrl._ema_m_x.item() if ctrl._ema_m_x is not None else None
            ema_m_d = ctrl._ema_m_d.item() if ctrl._ema_m_d is not None else None
            ema_m_xb = ctrl._ema_m_xb.item() if ctrl._ema_m_xb is not None else None
            ema_m_da = ctrl._ema_m_da.item() if ctrl._ema_m_da is not None else None

            # Compute lr_base
            lr_base = lr * ctrl.lora_alpha
            if ctrl.correct_gradient_magnitudes:
                lr_base /= math.sqrt(ctrl.rank)

            # Compute per-tile effective lr
            lr_eff_a = lr_base
            lr_eff_b = lr_base
            if ctrl.auto_scale and ctrl._auto_scale_steps > ctrl.auto_scale_warmup:
                cap = ctrl.auto_scale_max_amplification
                if ctrl.auto_scale_mode == "calibrate":
                    # Use frozen products (constant after warmup)
                    if ctrl._frozen_product_a is not None:
                        fp_a = ctrl._frozen_product_a.item()
                        fp_b = ctrl._frozen_product_b.item()
                        if fp_a > 0:
                            lr_eff_a = lr_base / fp_a
                        if fp_b > 0:
                            lr_eff_b = lr_base / fp_b
                        if cap > 0:
                            lr_eff_a = min(lr_eff_a, lr_base * cap)
                            lr_eff_b = min(lr_eff_b, lr_base * cap)
                elif ctrl.auto_scale_mode == "per_tile":
                    if ema_m_xb is not None and ema_m_d is not None:
                        prod_a_ema = ema_m_xb * ema_m_d
                        if prod_a_ema > 0:
                            lr_eff_a = lr_base / prod_a_ema
                    if ema_m_x is not None and ema_m_da is not None:
                        prod_b_ema = ema_m_x * ema_m_da
                        if prod_b_ema > 0:
                            lr_eff_b = lr_base / prod_b_ema
                    if cap > 0:
                        lr_eff_a = min(lr_eff_a, lr_base * cap)
                        lr_eff_b = min(lr_eff_b, lr_base * cap)
                else:  # global
                    if ema_m_x is not None and ema_m_d is not None:
                        ema_product = ema_m_x * ema_m_d
                        if ema_product > 0:
                            lr_eff_a = lr_eff_b = lr_base / ema_product
                        if cap > 0:
                            lr_eff_a = min(lr_eff_a, lr_base * cap)
                            lr_eff_b = lr_eff_a

            amp_a = lr_eff_a / lr_base if lr_base > 1e-12 else float('nan')
            amp_b = lr_eff_b / lr_base if lr_base > 1e-12 else float('nan')

            record = {
                "condition": condition,
                "layer": layer_name,
                "step": step_counter[0],
                # Products
                "product_tracked": product_tracked,
                "product_a": product_a,
                "product_b": product_b,
                # Per-tile LR
                "lr_base": lr_base,
                "lr_eff_a": lr_eff_a,
                "lr_eff_b": lr_eff_b,
                "amp_a": amp_a,
                "amp_b": amp_b,
                # EMA state
                "ema_m_x": ema_m_x,
                "ema_m_d": ema_m_d,
                "ema_m_xb": ema_m_xb,
                "ema_m_da": ema_m_da,
                # Saturation
                "A_saturated_frac": A_saturated_frac,
                "B_saturated_frac": B_saturated_frac,
            }
            update_records.append(record)

    return hooked_update


# =============================================================================
# Transfer Hook
# =============================================================================

def make_transfer_hook(ctrl, layer_name: str, condition: str):
    """Hook _ab_weight_transfer_onehot_off to capture A/B saturation at transfer time."""
    original_fn = ctrl._ab_weight_transfer_onehot_off

    def hooked_transfer_off():
        with torch.no_grad():
            A_w = ctrl.tile_a.get_weights()[0].to(ctrl.device)
            B_w = ctrl.tile_b.get_weights()[0].to(ctrl.device)
            w_max = 1.0
            A_sat = (A_w.abs() > 0.9 * w_max).float().mean().item()
            B_sat = (B_w.abs() > 0.9 * w_max).float().mean().item()
            A_F = A_w.norm().item()
            B_F = B_w.norm().item()
            AB_F = (A_w @ B_w).norm().item()

            # Read transfer_ema state
            ema_ab_norm = ctrl._ema_ab_norm.item() if (ctrl._ema_ab_norm is not None and ctrl._ema_ab_norm.numel() > 0) else 0.0
            tems_target = ctrl.transfer_ema_target_norm

        original_fn()

        record = {
            "condition": condition,
            "layer": layer_name,
            "transfer_idx": ctrl.num_transfers - 1,
            "A_sat_pre": A_sat,
            "B_sat_pre": B_sat,
            "A_F_pre": A_F,
            "B_F_pre": B_F,
            "AB_F_pre": AB_F,
            "ema_ab_norm": ema_ab_norm,
            "tems_target": tems_target,
        }
        transfer_records.append(record)

    return hooked_transfer_off


# =============================================================================
# Install / Remove Hooks
# =============================================================================

def install_hooks(model, condition: str):
    hooks_installed = []
    for name, module in model.named_modules():
        if isinstance(module, LRTTSimulatorTile):
            ctrl = module.controller
            short_name = name.split(".")[-1] if "." in name else name

            ctrl._orig_transfer_off = ctrl._ab_weight_transfer_onehot_off
            ctrl._orig_update_lora = ctrl._ab_weight_update_lora

            ctrl._ab_weight_transfer_onehot_off = make_transfer_hook(ctrl, short_name, condition)
            ctrl._ab_weight_update_lora = make_update_hook(ctrl, short_name, condition)

            hooks_installed.append(short_name)
    return hooks_installed


def remove_hooks(model):
    for _, module in model.named_modules():
        if isinstance(module, LRTTSimulatorTile):
            ctrl = module.controller
            if hasattr(ctrl, '_orig_transfer_off'):
                ctrl._ab_weight_transfer_onehot_off = ctrl._orig_transfer_off
                del ctrl._orig_transfer_off
            if hasattr(ctrl, '_orig_update_lora'):
                ctrl._ab_weight_update_lora = ctrl._orig_update_lora
                del ctrl._orig_update_lora


# =============================================================================
# Train Loop
# =============================================================================

def train_condition(condition_name: str, rpu_config, train_loader, val_loader):
    global transfer_records, update_records

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

    hooks = install_hooks(model, condition_name)
    print(f"  Hooks installed on: {hooks}")

    t0 = time.time()
    epoch_val_accs = []

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

        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
                target = target.to(DEVICE, non_blocking=True)
                output = model(data)
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        val_acc = 100.0 * correct / total
        epoch_val_accs.append(val_acc)

        n_transfers = sum(1 for r in transfer_records if r["condition"] == condition_name)
        n_updates = sum(1 for r in update_records if r["condition"] == condition_name)
        print(f"    Epoch {epoch}/{EPOCHS}: loss={np.mean(epoch_losses):.4f}, "
              f"val_acc={val_acc:.2f}%, transfers={n_transfers}, update_samples={n_updates}")

    train_time = time.time() - t0
    remove_hooks(model)
    del model
    torch.cuda.empty_cache()

    return {
        "best_val_acc": max(epoch_val_accs),
        "final_val_acc": epoch_val_accs[-1],
        "train_time": train_time,
    }


# =============================================================================
# CSV Output
# =============================================================================

def save_csv(records: List[dict], path: str, label: str):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records)} {label} records to {path}")


# =============================================================================
# Console Summary
# =============================================================================

def print_summary(all_results: dict):
    print("\n" + "=" * 80)
    print("  PER-TILE AUTO_SCALE VERIFICATION")
    print("=" * 80)

    # Per-condition summary
    for cond_name, *_ in CONDITIONS:
        cond_updates = [r for r in update_records if r["condition"] == cond_name]
        cond_transfers = [r for r in transfer_records if r["condition"] == cond_name]

        if not cond_updates:
            continue

        print(f"\n--- {cond_name} ---")
        print(f"  Val accuracy: best={all_results[cond_name]['best_val_acc']:.2f}%, "
              f"final={all_results[cond_name]['final_val_acc']:.2f}%")
        print(f"  Update samples: {len(cond_updates)}, Transfers: {len(cond_transfers)}")

        # Amplification
        amp_a_vals = [r["amp_a"] for r in cond_updates if not math.isnan(r["amp_a"])]
        amp_b_vals = [r["amp_b"] for r in cond_updates if not math.isnan(r["amp_b"])]
        if amp_a_vals:
            print(f"\n  Per-tile amplification:")
            print(f"    tile_a: mean={np.mean(amp_a_vals):.4f}, "
                  f"std={np.std(amp_a_vals):.4f}, "
                  f"min={np.min(amp_a_vals):.4f}, max={np.max(amp_a_vals):.4f}")
            print(f"    tile_b: mean={np.mean(amp_b_vals):.4f}, "
                  f"std={np.std(amp_b_vals):.4f}, "
                  f"min={np.min(amp_b_vals):.4f}, max={np.max(amp_b_vals):.4f}")

        # Saturation
        a_sat = [r["A_saturated_frac"] for r in cond_updates]
        b_sat = [r["B_saturated_frac"] for r in cond_updates]
        print(f"\n  AB Tile Saturation (|w|>0.9·w_max):")
        print(f"    A: mean={np.mean(a_sat)*100:.2f}%, max={np.max(a_sat)*100:.2f}%")
        print(f"    B: mean={np.mean(b_sat)*100:.2f}%, max={np.max(b_sat)*100:.2f}%")

        # Transfer saturation + ||AB||_F
        if cond_transfers:
            a_sat_t = [r["A_sat_pre"] for r in cond_transfers]
            b_sat_t = [r["B_sat_pre"] for r in cond_transfers]
            ab_f = [r["AB_F_pre"] for r in cond_transfers if "AB_F_pre" in r]
            print(f"\n  Transfer-time stats:")
            print(f"    A sat: mean={np.mean(a_sat_t)*100:.2f}%, max={np.max(a_sat_t)*100:.2f}%")
            print(f"    B sat: mean={np.mean(b_sat_t)*100:.2f}%, max={np.max(b_sat_t)*100:.2f}%")
            if ab_f:
                print(f"    ||AB||_F: mean={np.mean(ab_f):.4f}, std={np.std(ab_f):.4f}, "
                      f"min={np.min(ab_f):.4f}, max={np.max(ab_f):.4f}")

    # =================================================================
    # CROSS-CONDITION COMPARISON TABLE
    # =================================================================
    print("\n" + "=" * 80)
    print("  CROSS-CONDITION COMPARISON")
    print("=" * 80)

    header = f"{'Condition':<18} {'Val(best)':<10} {'Val(final)':<10} {'amp_a':<10} {'amp_b':<10} {'A_sat%':<10} {'B_sat%':<10}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for cond_name, *_ in CONDITIONS:
        cond_updates = [r for r in update_records if r["condition"] == cond_name]
        if not cond_updates:
            continue

        res = all_results[cond_name]
        amp_a = np.mean([r["amp_a"] for r in cond_updates if not math.isnan(r["amp_a"])])
        amp_b = np.mean([r["amp_b"] for r in cond_updates if not math.isnan(r["amp_b"])])
        a_sat = np.mean([r["A_saturated_frac"] for r in cond_updates]) * 100
        b_sat = np.mean([r["B_saturated_frac"] for r in cond_updates]) * 100

        print(f"  {cond_name:<18} {res['best_val_acc']:<10.2f} {res['final_val_acc']:<10.2f} "
              f"{amp_a:<10.4f} {amp_b:<10.4f} {a_sat:<10.2f} {b_sat:<10.2f}")

    # =================================================================
    # KEY FINDINGS
    # =================================================================
    print("\n" + "=" * 80)
    print("  KEY FINDINGS")
    print("=" * 80)

    def _get_cond_stats(cond_name):
        updates = [r for r in update_records if r["condition"] == cond_name]
        if not updates:
            return None
        amp_a = [r["amp_a"] for r in updates if not math.isnan(r["amp_a"])]
        amp_b = [r["amp_b"] for r in updates if not math.isnan(r["amp_b"])]
        # Epoch-binned amp_b to show temporal stability
        n = len(updates)
        bin_sz = max(1, n // 5)
        amp_b_per_epoch = []
        for i in range(5):
            chunk = updates[i*bin_sz : (i+1)*bin_sz if i < 4 else n]
            vals = [r["amp_b"] for r in chunk if not math.isnan(r["amp_b"])]
            amp_b_per_epoch.append(np.mean(vals) if vals else float('nan'))
        return {
            "a_sat": np.mean([r["A_saturated_frac"] for r in updates]) * 100,
            "b_sat": np.mean([r["B_saturated_frac"] for r in updates]) * 100,
            "amp_a": np.mean(amp_a) if amp_a else float('nan'),
            "amp_b": np.mean(amp_b) if amp_b else float('nan'),
            "amp_b_std": np.std(amp_b) if amp_b else float('nan'),
            "amp_b_max": np.max(amp_b) if amp_b else float('nan'),
            "amp_b_e1": amp_b_per_epoch[0],
            "amp_b_e5": amp_b_per_epoch[4],
        }

    stats = {name: _get_cond_stats(name) for name, *_ in CONDITIONS}
    stats = {k: v for k, v in stats.items() if v is not None}

    # 1. Temporal stability: amp_b drift from epoch 1 → 5
    print(f"\n  1. Temporal stability (amp_b drift epoch 1 → 5):")
    for name, s in stats.items():
        drift = s["amp_b_e5"] - s["amp_b_e1"]
        print(f"     {name:<18} e1={s['amp_b_e1']:.2f}x → e5={s['amp_b_e5']:.2f}x  "
              f"drift={drift:+.2f}x  std={s['amp_b_std']:.2f}  max={s['amp_b_max']:.2f}x")

    # 2. Saturation
    print(f"\n  2. Saturation:")
    for name, s in stats.items():
        print(f"     {name:<18} A={s['a_sat']:.2f}%, B={s['b_sat']:.2f}%")

    # 3. Accuracy
    print(f"\n  3. Accuracy:")
    for name, s in stats.items():
        res = all_results[name]
        print(f"     {name:<18} best={res['best_val_acc']:.2f}%, final={res['final_val_acc']:.2f}%")

    # 4. Transfer ||AB||_F stability
    print(f"\n  4. Transfer ||AB||_F stability:")
    for name, *_ in CONDITIONS:
        cond_transfers = [r for r in transfer_records if r["condition"] == name]
        if cond_transfers:
            ab_f = [r["AB_F_pre"] for r in cond_transfers if "AB_F_pre" in r]
            if ab_f:
                # Split into first/last half
                mid = len(ab_f) // 2
                first_half = np.mean(ab_f[:mid]) if mid > 0 else float('nan')
                second_half = np.mean(ab_f[mid:]) if mid > 0 else float('nan')
                print(f"     {name:<18} mean={np.mean(ab_f):.4f}  std={np.std(ab_f):.4f}  "
                      f"1st_half={first_half:.4f}  2nd_half={second_half:.4f}")

    # 5. CALIBRATE vs CAL+TEMS (key question: does transfer_ema_scale help?)
    if "CALIBRATE" in stats and "CAL+TEMS" in stats and "BASELINE" in stats:
        cal = stats["CALIBRATE"]
        cal_tems = stats["CAL+TEMS"]
        base = stats["BASELINE"]

        print(f"\n  5. CALIBRATE vs CAL+TEMS (does transfer_ema_scale help?):")
        print(f"     final_acc:    BASELINE={all_results['BASELINE']['final_val_acc']:.2f}%  "
              f"CALIBRATE={all_results['CALIBRATE']['final_val_acc']:.2f}%  "
              f"CAL+TEMS={all_results['CAL+TEMS']['final_val_acc']:.2f}%")
        print(f"     best_acc:     BASELINE={all_results['BASELINE']['best_val_acc']:.2f}%  "
              f"CALIBRATE={all_results['CALIBRATE']['best_val_acc']:.2f}%  "
              f"CAL+TEMS={all_results['CAL+TEMS']['best_val_acc']:.2f}%")
        print(f"     A_sat:        BASELINE={base['a_sat']:.2f}%  "
              f"CALIBRATE={cal['a_sat']:.2f}%  CAL+TEMS={cal_tems['a_sat']:.2f}%")
        print(f"     B_sat:        BASELINE={base['b_sat']:.2f}%  "
              f"CALIBRATE={cal['b_sat']:.2f}%  CAL+TEMS={cal_tems['b_sat']:.2f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    global transfer_records, update_records

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/data/AIMC_LoRA_results/auto_scale_pertile_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Per-Tile Auto-Scale Diagnostics")
    print(f"Device: {DEVICE}")
    print(f"LRTT: rank={RANK}, te={TE}, lr={LR}, tlr={TLR}")
    print(f"Epochs: {EPOCHS}, batch_size={BATCH_SIZE}, seed={SEED}")
    print(f"Output: {out_dir}")

    train_loader, val_loader = load_data()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    all_results = {}
    for cond_name, auto_scale, mode, cap, tems in CONDITIONS:
        rpu_config = create_lrtt_config(
            auto_scale=auto_scale,
            auto_scale_mode=mode,
            auto_scale_max_amplification=cap,
            transfer_ema_scale=tems,
        )
        results = train_condition(cond_name, rpu_config, train_loader, val_loader)
        all_results[cond_name] = results

    # Save CSVs
    print(f"\n{'='*60}")
    print(f"  Saving results...")
    print(f"{'='*60}")
    save_csv(update_records, os.path.join(out_dir, "pertile_update_records.csv"), "update")
    save_csv(transfer_records, os.path.join(out_dir, "pertile_transfer_records.csv"), "transfer")

    # Print summary
    print_summary(all_results)

    print(f"\n\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()

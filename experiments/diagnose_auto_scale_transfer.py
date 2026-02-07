#!/usr/bin/env python3
"""Transfer 전후 진단: auto_scale + transfer_ema가 one-hot transfer를 올바르게 관리하는지 검증.

매 transfer마다 A, B, C, A@B를 per-rank/per-norm으로 분리 분석하고,
매 N step마다 update의 입력/EMA/LR 상태를 기록한다.

4 conditions × 5 epochs × MNIST. 약 3~4분 소요.

Usage:
    cd /data/LRTT_transformer && python experiments/diagnose_auto_scale_transfer.py
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
# Configuration (test_auto_scale_mnist.py와 동일)
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

# Update hook sampling interval (steps)
UPDATE_SAMPLE_EVERY = 10

CONDITIONS = [
    ("BASELINE",          False, False),
    ("AUTO_SCALE_ONLY",   True,  False),
    ("TRANSFER_EMA_ONLY", False, True),
    ("BOTH",              True,  True),
]


# =============================================================================
# Config / Model / Data (from test_auto_scale_mnist.py)
# =============================================================================

def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(auto_scale: bool = False, transfer_ema_scale: bool = False):
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

transfer_records: List[dict] = []
update_records: List[dict] = []


# =============================================================================
# Monkey-patch Factory: Transfer Hook
# =============================================================================

def make_transfer_hook(ctrl, layer_name: str, condition: str):
    """Create a hooked version of _ab_weight_transfer_onehot_off for the given controller.

    We hook _ab_weight_transfer_onehot_off specifically since transfer_mode="off".
    """
    original_fn = ctrl._ab_weight_transfer_onehot_off

    def hooked_transfer_off():
        """Wraps _ab_weight_transfer_onehot_off to capture before/after C and A/B state."""
        with torch.no_grad():
            # ---- BEFORE transfer: capture state ----
            # get_weights() may return CPU tensor; move to controller device
            C_before = ctrl.tile_c.get_weights()[0].to(ctrl.device).clone()

            # Read A, B via one-hot (same as controller does internally)
            if ctrl._transfer_vec_a is None or ctrl._transfer_vec_a.device != ctrl.device:
                ctrl._transfer_vec_a = torch.eye(
                    ctrl.rank, dtype=ctrl.dtype, device=ctrl.device
                )
            I = ctrl._transfer_vec_a

            # Batch differential read (matching _read_ab_onehot_symmetric fast path)
            if ctrl.differential_read:
                A_p = ctrl.tile_a.forward(I)
                A_m = ctrl.tile_a.forward(-I)
                B_p = ctrl.tile_b.backward(I)
                B_m = ctrl.tile_b.backward(-I)
                A_cols = (0.5 * (A_p - A_m)).T   # [d_size, rank]
                B_rows = 0.5 * (B_p - B_m)       # [rank, x_size]
            else:
                A_cols = ctrl.tile_a.forward(I).T
                B_rows = ctrl.tile_b.backward(I)

            # Per-rank norms
            rank = ctrl.rank
            a_rank_norms = [A_cols[:, k].norm().item() for k in range(rank)]
            b_rank_norms = [B_rows[k, :].norm().item() for k in range(rank)]
            ab_rank_products = [a_rank_norms[k] * b_rank_norms[k] for k in range(rank)]

            A_F = A_cols.norm().item()
            B_F = B_rows.norm().item()

            # ||A@B||_F via Gram
            G_A = A_cols.t() @ A_cols
            G_B = B_rows @ B_rows.t()
            AB_F = torch.sqrt((G_A * G_B).sum() + 1e-12).item()

            C_before_F = C_before.norm().item()

            # EMA state (before transfer call, which may update EMA)
            ema_ab_norm_pre = ctrl._ema_ab_norm.item() if ctrl._ema_ab_norm is not None else None
            target_norm = ctrl.transfer_ema_target_norm

            # Compute expected lr_eff for this transfer
            # (matches _ab_weight_transfer_onehot_off logic)
            lr_abs_base = abs(ctrl.transfer_lr)
            ema_scale = 1.0  # will be set by transfer if transfer_ema_scale enabled

        # ---- Call original transfer ----
        original_fn()

        with torch.no_grad():
            # ---- AFTER transfer: capture state ----
            # Note: num_transfers was already incremented inside original_fn
            C_after = ctrl.tile_c.get_weights()[0].to(ctrl.device)
            C_after_F = C_after.norm().item()

            actual_delta_C = C_after - C_before
            actual_delta_C_F = actual_delta_C.norm().item()

            # EMA state after transfer
            ema_ab_norm_post = ctrl._ema_ab_norm.item() if ctrl._ema_ab_norm is not None else None

            # Compute scale_factor from post-EMA state
            # _compute_transfer_ema_scale updates EMA first, then returns target/updated_ema
            scale_factor = 1.0
            if ctrl.transfer_ema_scale and ema_ab_norm_post is not None and target_norm > 0:
                if ema_ab_norm_post > 1e-12:
                    scale_factor = target_norm / ema_ab_norm_post

            lr_effective = lr_abs_base * scale_factor
            M_total = max(1, int(ctrl.transfer_micro_steps))

            # Expected ΔC = lr_eff * A@B (ideal, no pulsed quantization)
            expected_delta_C = lr_effective * (A_cols @ B_rows)  # [d_size, x_size]
            expected_delta_C_F = expected_delta_C.norm().item()

            # Transfer accuracy: ||actual - expected|| / ||expected||
            if expected_delta_C_F > 1e-12:
                transfer_error = (actual_delta_C - expected_delta_C).norm().item()
                transfer_accuracy = transfer_error / expected_delta_C_F
            else:
                transfer_accuracy = float('nan')

            # Relative ΔC: ||expected_ΔC|| / ||C_before||
            relative_delta_C = expected_delta_C_F / C_before_F if C_before_F > 1e-12 else float('nan')

        record = {
            "condition": condition,
            "layer": layer_name,
            "transfer_idx": ctrl.num_transfers - 1,  # 0-indexed (already incremented)
            # A matrix
            "A_F": A_F,
            **{f"a_rank{k}_norm": a_rank_norms[k] for k in range(rank)},
            # B matrix
            "B_F": B_F,
            **{f"b_rank{k}_norm": b_rank_norms[k] for k in range(rank)},
            # A@B product
            "AB_F": AB_F,
            **{f"ab_rank{k}_product": ab_rank_products[k] for k in range(rank)},
            # C before/after
            "C_before_F": C_before_F,
            "C_after_F": C_after_F,
            # Deltas
            "actual_delta_C_F": actual_delta_C_F,
            "expected_delta_C_F": expected_delta_C_F,
            "transfer_accuracy": transfer_accuracy,
            "relative_delta_C": relative_delta_C,
            # EMA state
            "ema_ab_norm_pre": ema_ab_norm_pre,
            "ema_ab_norm_post": ema_ab_norm_post,
            "target_norm": target_norm,
            "scale_factor": scale_factor,
            # LR
            "transfer_lr_base": lr_abs_base,
            "transfer_lr_effective": lr_effective,
            "micro_steps": M_total,
        }
        transfer_records.append(record)

    return hooked_transfer_off


# =============================================================================
# Monkey-patch Factory: Update Hook
# =============================================================================

def make_update_hook(ctrl, layer_name: str, condition: str):
    """Create a hooked version of _ab_weight_update_lora for the given controller."""
    original_fn = ctrl._ab_weight_update_lora
    step_counter = [0]  # mutable closure

    def hooked_update(x, d, lr, in_trans=False, out_trans=False):
        step_counter[0] += 1
        should_log = (step_counter[0] % UPDATE_SAMPLE_EVERY == 1)

        if should_log:
            with torch.no_grad():
                # Normalize x, d to [batch, feat]
                x_flat = x.t() if in_trans else x
                d_flat = d.t() if out_trans else d
                max_x = x_flat.abs().amax().item()
                max_d = d_flat.abs().amax().item()

        # Call original
        original_fn(x, d, lr, in_trans, out_trans)

        if should_log:
            # Read EMA state after update
            ema_m_x = ctrl._ema_m_x.item() if ctrl._ema_m_x is not None else None
            ema_m_d = ctrl._ema_m_d.item() if ctrl._ema_m_d is not None else None
            ema_product = (ema_m_x * ema_m_d) if (ema_m_x is not None and ema_m_d is not None) else None

            # Compute effective lr (same logic as _ab_weight_update_lora)
            lr_base = lr * ctrl.lora_alpha
            if ctrl.correct_gradient_magnitudes:
                lr_base /= math.sqrt(ctrl.rank)

            lr_effective = lr_base
            if ctrl.auto_scale and ema_product is not None and ema_product > 0:
                if ctrl._auto_scale_steps > ctrl.auto_scale_warmup:
                    lr_effective = lr_base / ema_product

            amplification = lr_effective / lr_base if lr_base > 1e-12 else float('nan')

            record = {
                "condition": condition,
                "layer": layer_name,
                "step": step_counter[0],
                "max_x": max_x,
                "max_d": max_d,
                "xd_product": max_x * max_d,
                "ema_m_x": ema_m_x,
                "ema_m_d": ema_m_d,
                "ema_product": ema_product,
                "lr_base": lr_base,
                "lr_effective": lr_effective,
                "amplification": amplification,
            }
            update_records.append(record)

    return hooked_update


# =============================================================================
# Install / Remove Hooks
# =============================================================================

def install_hooks(model, condition: str):
    """Install monkey-patch hooks on all LRTT controllers."""
    hooks_installed = []
    for name, module in model.named_modules():
        if isinstance(module, LRTTSimulatorTile):
            ctrl = module.controller
            short_name = name.split(".")[-1] if "." in name else name

            # Save originals
            ctrl._orig_transfer_off = ctrl._ab_weight_transfer_onehot_off
            ctrl._orig_update_lora = ctrl._ab_weight_update_lora

            # Install hooks
            ctrl._ab_weight_transfer_onehot_off = make_transfer_hook(ctrl, short_name, condition)
            ctrl._ab_weight_update_lora = make_update_hook(ctrl, short_name, condition)

            hooks_installed.append(short_name)

    return hooks_installed


def remove_hooks(model):
    """Restore original methods."""
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
    """Train one condition and collect diagnostics."""
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

    # Install hooks
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

        # Count transfers so far for this condition
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

def save_transfer_csv(records: List[dict], path: str):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records)} transfer records to {path}")


def save_update_csv(records: List[dict], path: str):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {len(records)} update records to {path}")


# =============================================================================
# Console Summary
# =============================================================================

def print_summary(all_results: dict):
    """Print cross-condition analysis."""
    print("\n" + "=" * 80)
    print("  TRANSFER DIAGNOSTICS SUMMARY")
    print("=" * 80)

    baseline_transfers = [r for r in transfer_records if r["condition"] == "BASELINE"]

    for cond_name, _, _ in CONDITIONS:
        cond_transfers = [r for r in transfer_records if r["condition"] == cond_name]
        cond_updates = [r for r in update_records if r["condition"] == cond_name]

        if not cond_transfers:
            print(f"\n=== CONDITION: {cond_name} === (no transfer records)")
            continue

        print(f"\n=== CONDITION: {cond_name} ===")
        print(f"  Transfers: {len(cond_transfers)}, Update samples: {len(cond_updates)}")
        print(f"  Val accuracy: best={all_results[cond_name]['best_val_acc']:.2f}%, "
              f"final={all_results[cond_name]['final_val_acc']:.2f}%")

        # A@B norm stats
        ab_norms = [r["AB_F"] for r in cond_transfers]
        print(f"\n  ||A@B||_F: mean={np.mean(ab_norms):.4f}, std={np.std(ab_norms):.4f}, "
              f"min={np.min(ab_norms):.4f}, max={np.max(ab_norms):.4f}")

        # Trend: linear regression on AB_F
        if len(ab_norms) > 2:
            x_idx = np.arange(len(ab_norms))
            slope = np.polyfit(x_idx, ab_norms, 1)[0]
            print(f"    trend: slope={slope:.6f}/transfer "
                  f"({'growing' if slope > 0 else 'shrinking'})")

        # Relative ΔC
        rel_deltas = [r["relative_delta_C"] for r in cond_transfers
                      if not math.isnan(r["relative_delta_C"])]
        if rel_deltas:
            print(f"\n  relative_ΔC (||expected_ΔC||/||C||): "
                  f"mean={np.mean(rel_deltas):.6f}, std={np.std(rel_deltas):.6f}")

        # Transfer accuracy
        accs = [r["transfer_accuracy"] for r in cond_transfers
                if not math.isnan(r["transfer_accuracy"])]
        if accs:
            print(f"\n  transfer_accuracy (0=perfect, 1=100% error): "
                  f"mean={np.mean(accs):.4f}, std={np.std(accs):.4f}")

        # EMA / scale info
        if cond_transfers[0].get("ema_ab_norm_post") is not None:
            ema_vals = [r["ema_ab_norm_post"] for r in cond_transfers
                        if r["ema_ab_norm_post"] is not None]
            scale_vals = [r["scale_factor"] for r in cond_transfers]
            target = cond_transfers[-1]["target_norm"]
            print(f"\n  EMA state:")
            print(f"    ema_ab_norm: mean={np.mean(ema_vals):.4f}, final={ema_vals[-1]:.4f}")
            print(f"    target_norm: {target:.4f}")
            print(f"    scale_factor: mean={np.mean(scale_vals):.4f}, "
                  f"std={np.std(scale_vals):.4f}, final={scale_vals[-1]:.4f}")

        # Auto-scale LR stats
        if cond_updates:
            amps = [r["amplification"] for r in cond_updates
                    if not math.isnan(r["amplification"])]
            if amps:
                print(f"\n  auto_scale LR:")
                print(f"    amplification: mean={np.mean(amps):.4f}, "
                      f"std={np.std(amps):.4f}, final={amps[-1]:.4f}")

        # Transfer LR
        lr_effs = [r["transfer_lr_effective"] for r in cond_transfers]
        print(f"\n  Transfer LR:")
        print(f"    base={cond_transfers[0]['transfer_lr_base']:.6f}")
        print(f"    effective: mean={np.mean(lr_effs):.6f}, final={lr_effs[-1]:.6f}")

        # Per-rank balance
        rank = RANK
        print(f"\n  Per-rank ||a_k||*||b_k|| (last 10 transfers avg):")
        last_n = cond_transfers[-10:]
        for k in range(rank):
            vals = [r[f"ab_rank{k}_product"] for r in last_n]
            total = sum(np.mean([r[f"ab_rank{j}_product"] for r in last_n]) for j in range(rank))
            frac = np.mean(vals) / total if total > 1e-12 else 0
            print(f"    rank {k}: mean={np.mean(vals):.4f} ({frac*100:.1f}% of total)")

    # === Cross-condition analysis ===
    print("\n" + "=" * 80)
    print("  CROSS-CONDITION ANALYSIS")
    print("=" * 80)

    if baseline_transfers:
        base_ab_mean = np.mean([r["AB_F"] for r in baseline_transfers])
        base_rel_mean = np.mean([r["relative_delta_C"] for r in baseline_transfers
                                 if not math.isnan(r["relative_delta_C"])])

        for cond_name, _, _ in CONDITIONS[1:]:
            cond_t = [r for r in transfer_records if r["condition"] == cond_name]
            if not cond_t:
                continue

            cond_ab_mean = np.mean([r["AB_F"] for r in cond_t])
            cond_rel_vals = [r["relative_delta_C"] for r in cond_t
                            if not math.isnan(r["relative_delta_C"])]
            cond_rel_mean = np.mean(cond_rel_vals) if cond_rel_vals else float('nan')

            ab_ratio = cond_ab_mean / base_ab_mean if base_ab_mean > 1e-12 else float('nan')
            rel_ratio = cond_rel_mean / base_rel_mean if base_rel_mean > 1e-12 else float('nan')

            print(f"\n  {cond_name} vs BASELINE:")
            print(f"    ||A@B||_F ratio: {ab_ratio:.4f}x "
                  f"({'larger' if ab_ratio > 1 else 'smaller'})")
            print(f"    relative_ΔC ratio: {rel_ratio:.4f}x "
                  f"({'larger' if rel_ratio > 1 else 'smaller'})")

    # Key questions
    print("\n" + "-" * 80)
    print("  KEY FINDINGS:")
    print("-" * 80)

    # Q1: auto_scale → AB growth?
    base_t = [r for r in transfer_records if r["condition"] == "BASELINE"]
    auto_t = [r for r in transfer_records if r["condition"] == "AUTO_SCALE_ONLY"]
    if base_t and auto_t:
        base_slope = np.polyfit(range(len(base_t)), [r["AB_F"] for r in base_t], 1)[0] if len(base_t) > 2 else 0
        auto_slope = np.polyfit(range(len(auto_t)), [r["AB_F"] for r in auto_t], 1)[0] if len(auto_t) > 2 else 0
        slope_ratio = auto_slope / base_slope if abs(base_slope) > 1e-12 else float('nan')
        print(f"\n  Q1. auto_scale → AB growth acceleration?")
        print(f"      BASELINE AB slope: {base_slope:.6f}/transfer")
        print(f"      AUTO_SCALE AB slope: {auto_slope:.6f}/transfer")
        if not math.isnan(slope_ratio):
            print(f"      Ratio: {slope_ratio:.2f}x → "
                  f"{'YES, accelerates' if slope_ratio > 1.2 else 'NO, similar growth'}")

    # Q2: transfer_ema → stabilizes relative_ΔC?
    ema_t = [r for r in transfer_records if r["condition"] == "TRANSFER_EMA_ONLY"]
    if base_t and ema_t:
        base_rel = [r["relative_delta_C"] for r in base_t if not math.isnan(r["relative_delta_C"])]
        ema_rel = [r["relative_delta_C"] for r in ema_t if not math.isnan(r["relative_delta_C"])]
        if base_rel and ema_rel:
            base_cv = np.std(base_rel) / np.mean(base_rel) if np.mean(base_rel) > 1e-12 else float('nan')
            ema_cv = np.std(ema_rel) / np.mean(ema_rel) if np.mean(ema_rel) > 1e-12 else float('nan')
            print(f"\n  Q2. transfer_ema → stabilizes relative_ΔC?")
            print(f"      BASELINE CV(relative_ΔC): {base_cv:.4f}")
            print(f"      TRANSFER_EMA CV(relative_ΔC): {ema_cv:.4f}")
            if not math.isnan(base_cv) and not math.isnan(ema_cv):
                print(f"      → {'YES, more stable' if ema_cv < base_cv else 'NO, not more stable'} "
                      f"(CV ratio: {ema_cv/base_cv:.2f}x)")

    # Q3: Pulsed update accuracy
    all_accs = [r["transfer_accuracy"] for r in transfer_records
                if not math.isnan(r["transfer_accuracy"])]
    if all_accs:
        print(f"\n  Q3. Pulsed update accuracy?")
        print(f"      Overall: mean={np.mean(all_accs):.4f}, "
              f"median={np.median(all_accs):.4f}, max={np.max(all_accs):.4f}")
        print(f"      → {'GOOD' if np.mean(all_accs) < 0.3 else 'CONCERNING'}: "
              f"avg {np.mean(all_accs)*100:.1f}% error")

    # Q4: Per-rank imbalance
    print(f"\n  Q4. Per-rank imbalance?")
    for cond_name, _, _ in CONDITIONS:
        cond_t = [r for r in transfer_records if r["condition"] == cond_name]
        if not cond_t:
            continue
        last = cond_t[-10:]
        rank_means = [np.mean([r[f"ab_rank{k}_product"] for r in last]) for k in range(RANK)]
        total = sum(rank_means)
        if total > 1e-12:
            fracs = [v / total for v in rank_means]
            max_frac = max(fracs)
            dominant = fracs.index(max_frac)
            gini = _gini(rank_means)
            print(f"      {cond_name}: dominant=rank{dominant} ({max_frac*100:.1f}%), "
                  f"Gini={gini:.3f} ({'balanced' if gini < 0.2 else 'imbalanced'})")


def _gini(values):
    """Compute Gini coefficient for a list of non-negative values."""
    vals = sorted(values)
    n = len(vals)
    if n == 0 or sum(vals) < 1e-12:
        return 0.0
    cumsum = np.cumsum(vals)
    return (2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * cumsum[-1])) - (n + 1) / n


# =============================================================================
# Main
# =============================================================================

def main():
    global transfer_records, update_records

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/data/AIMC_LoRA_results/transfer_diagnostics_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Transfer Diagnostics Script")
    print(f"Device: {DEVICE}")
    print(f"LRTT: rank={RANK}, te={TE}, lr={LR}, tlr={TLR}")
    print(f"Epochs: {EPOCHS}, batch_size={BATCH_SIZE}, seed={SEED}")
    print(f"Output: {out_dir}")

    train_loader, val_loader = load_data()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    all_results = {}
    for cond_name, auto_scale, transfer_ema in CONDITIONS:
        # Reset records for this condition (accumulate across conditions)
        rpu_config = create_lrtt_config(
            auto_scale=auto_scale,
            transfer_ema_scale=transfer_ema,
        )
        results = train_condition(cond_name, rpu_config, train_loader, val_loader)
        all_results[cond_name] = results

    # Save CSVs
    print(f"\n{'='*60}")
    print(f"  Saving results...")
    print(f"{'='*60}")
    save_transfer_csv(transfer_records, os.path.join(out_dir, "transfer_diagnostics.csv"))
    save_update_csv(update_records, os.path.join(out_dir, "update_diagnostics.csv"))

    # Print summary
    print_summary(all_results)

    print(f"\n\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()

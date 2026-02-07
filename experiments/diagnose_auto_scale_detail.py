#!/usr/bin/env python3
"""auto_scale 심층 진단: auto_scale이 추적하는 값과 실제 타일 입력의 불일치를 수치적으로 검증.

핵심 질문:
1. auto_scale이 추적하는 max|x|·max|d|와 실제 타일 입력(max|XB|·max|d|, max|x|·max|DA|)의 불일치 정도
2. auto_scale ON일 때 AB 가중치 포화(|w|>0.9·w_max) 증가 여부
3. 이상적인 amplification은 tile별로 얼마인가

2 conditions × 5 epochs × MNIST. 약 2분 소요.

Usage:
    cd /data/LRTT_transformer && python experiments/diagnose_auto_scale_detail.py
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

# Conditions: (name, auto_scale)
# transfer_ema_scale=False for all (isolate auto_scale effect)
CONDITIONS = [
    ("BASELINE",    False),
    ("AUTO_SCALE",  True),
]


# =============================================================================
# Config / Model / Data
# =============================================================================

def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(auto_scale: bool = False):
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
        transfer_ema_scale=False,  # OFF for all conditions
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
# Update Hook: measures mismatch between auto_scale tracking and actual tile inputs
# =============================================================================

def make_update_hook(ctrl, layer_name: str, condition: str):
    """Hook _ab_weight_update_lora to measure auto_scale mismatch and tile saturation."""
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
                XB = ctrl.tile_b.forward(x_flat)  # [batch, rank]
                max_XB = XB.abs().amax().item()
                product_a = max_XB * max_d

                # --- What tile_b actually receives: (x, DA) ---
                DA = ctrl.tile_a.backward(d_flat)  # [batch, rank]
                max_DA = DA.abs().amax().item()
                product_b = max_x * max_DA

                # --- Mismatch ratios ---
                mismatch_a = product_tracked / product_a if product_a > 1e-12 else float('nan')
                mismatch_b = product_tracked / product_b if product_b > 1e-12 else float('nan')

                # --- A/B tile weight stats ---
                A_w = ctrl.tile_a.get_weights()[0].to(ctrl.device)
                B_w = ctrl.tile_b.get_weights()[0].to(ctrl.device)

                w_max = 1.0  # Tile bound

                A_w_absmax = A_w.abs().max().item()
                A_w_mean_abs = A_w.abs().mean().item()
                B_w_absmax = B_w.abs().max().item()
                B_w_mean_abs = B_w.abs().mean().item()

                # Saturation: fraction of weights near bounds
                A_saturated_frac = (A_w.abs() > 0.9 * w_max).float().mean().item()
                B_saturated_frac = (B_w.abs() > 0.9 * w_max).float().mean().item()

        # Call original
        original_fn(x, d, lr, in_trans, out_trans)

        if should_log:
            # Read EMA state after update
            ema_m_x = ctrl._ema_m_x.item() if ctrl._ema_m_x is not None else None
            ema_m_d = ctrl._ema_m_d.item() if ctrl._ema_m_d is not None else None
            ema_product = (ema_m_x * ema_m_d) if (ema_m_x is not None and ema_m_d is not None) else None

            # Compute effective lr (same logic as controller)
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
                # What auto_scale tracks
                "max_x": max_x,
                "max_d": max_d,
                "product_tracked": product_tracked,
                # What tile_a actually sees
                "max_XB": max_XB,
                "product_a": product_a,
                # What tile_b actually sees
                "max_DA": max_DA,
                "product_b": product_b,
                # Mismatch ratios
                "mismatch_a": mismatch_a,
                "mismatch_b": mismatch_b,
                # LR state
                "lr_base": lr_base,
                "lr_effective": lr_effective,
                "amplification": amplification,
                "ema_m_x": ema_m_x,
                "ema_m_d": ema_m_d,
                "ema_product": ema_product,
                # A tile weight stats
                "A_w_absmax": A_w_absmax,
                "A_w_mean_abs": A_w_mean_abs,
                "A_saturated_frac": A_saturated_frac,
                # B tile weight stats
                "B_w_absmax": B_w_absmax,
                "B_w_mean_abs": B_w_mean_abs,
                "B_saturated_frac": B_saturated_frac,
            }
            update_records.append(record)

    return hooked_update


# =============================================================================
# Transfer Hook: measures A/B/C weight before/after transfer and reinit
# =============================================================================

def make_transfer_hook(ctrl, layer_name: str, condition: str):
    """Hook _ab_weight_transfer_onehot_off to capture before/after state."""
    original_fn = ctrl._ab_weight_transfer_onehot_off

    def hooked_transfer_off():
        with torch.no_grad():
            # --- BEFORE transfer ---
            C_before = ctrl.tile_c.get_weights()[0].to(ctrl.device).clone()
            A_w_pre = ctrl.tile_a.get_weights()[0].to(ctrl.device).clone()
            B_w_pre = ctrl.tile_b.get_weights()[0].to(ctrl.device).clone()

            # Read A, B via one-hot (what transfer uses)
            if ctrl._transfer_vec_a is None or ctrl._transfer_vec_a.device != ctrl.device:
                ctrl._transfer_vec_a = torch.eye(
                    ctrl.rank, dtype=ctrl.dtype, device=ctrl.device
                )
            I = ctrl._transfer_vec_a

            if ctrl.differential_read:
                A_p = ctrl.tile_a.forward(I)
                A_m = ctrl.tile_a.forward(-I)
                B_p = ctrl.tile_b.backward(I)
                B_m = ctrl.tile_b.backward(-I)
                A_cols = (0.5 * (A_p - A_m)).T
                B_rows = 0.5 * (B_p - B_m)
            else:
                A_cols = ctrl.tile_a.forward(I).T
                B_rows = ctrl.tile_b.backward(I)

            A_F_pre = A_cols.norm().item()
            B_F_pre = B_rows.norm().item()

            # ||A@B||_F via Gram
            G_A = A_cols.t() @ A_cols
            G_B = B_rows @ B_rows.t()
            AB_F_pre = torch.sqrt((G_A * G_B).sum() + 1e-12).item()

            C_F_pre = C_before.norm().item()

            # Per-rank norms
            rank = ctrl.rank
            a_rank_norms_pre = [A_cols[:, k].norm().item() for k in range(rank)]
            b_rank_norms_pre = [B_rows[k, :].norm().item() for k in range(rank)]

            # A/B weight saturation before transfer
            w_max = 1.0
            A_sat_pre = (A_w_pre.abs() > 0.9 * w_max).float().mean().item()
            B_sat_pre = (B_w_pre.abs() > 0.9 * w_max).float().mean().item()

        # --- Call original transfer ---
        original_fn()

        with torch.no_grad():
            # --- AFTER transfer (and reinit) ---
            C_after = ctrl.tile_c.get_weights()[0].to(ctrl.device)
            A_w_post = ctrl.tile_a.get_weights()[0].to(ctrl.device)
            B_w_post = ctrl.tile_b.get_weights()[0].to(ctrl.device)

            C_F_post = C_after.norm().item()
            delta_C_F = (C_after - C_before).norm().item()

            A_F_post = A_w_post.norm().item()
            B_F_post = B_w_post.norm().item()

            # Per-rank norms after reinit
            if ctrl.differential_read:
                A_p2 = ctrl.tile_a.forward(I)
                A_m2 = ctrl.tile_a.forward(-I)
                B_p2 = ctrl.tile_b.backward(I)
                B_m2 = ctrl.tile_b.backward(-I)
                A_cols_post = (0.5 * (A_p2 - A_m2)).T
                B_rows_post = 0.5 * (B_p2 - B_m2)
            else:
                A_cols_post = ctrl.tile_a.forward(I).T
                B_rows_post = ctrl.tile_b.backward(I)

            a_rank_norms_post = [A_cols_post[:, k].norm().item() for k in range(rank)]
            b_rank_norms_post = [B_rows_post[k, :].norm().item() for k in range(rank)]

        record = {
            "condition": condition,
            "layer": layer_name,
            "transfer_idx": ctrl.num_transfers - 1,
            # Before transfer
            "A_F_pre": A_F_pre,
            "B_F_pre": B_F_pre,
            "AB_F_pre": AB_F_pre,
            "C_F_pre": C_F_pre,
            "A_sat_pre": A_sat_pre,
            "B_sat_pre": B_sat_pre,
            # After transfer + reinit
            "A_F_post": A_F_post,
            "B_F_post": B_F_post,
            "C_F_post": C_F_post,
            "delta_C_F": delta_C_F,
            # Per-rank before
            **{f"a_rank{k}_pre": a_rank_norms_pre[k] for k in range(rank)},
            **{f"b_rank{k}_pre": b_rank_norms_pre[k] for k in range(rank)},
            # Per-rank after
            **{f"a_rank{k}_post": a_rank_norms_post[k] for k in range(rank)},
            **{f"b_rank{k}_post": b_rank_norms_post[k] for k in range(rank)},
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
    print("  AUTO_SCALE MAGNITUDE MISMATCH ANALYSIS")
    print("=" * 80)

    for cond_name, _ in CONDITIONS:
        cond_updates = [r for r in update_records if r["condition"] == cond_name]
        cond_transfers = [r for r in transfer_records if r["condition"] == cond_name]

        if not cond_updates:
            continue

        print(f"\n--- {cond_name} ---")
        print(f"  Val accuracy: best={all_results[cond_name]['best_val_acc']:.2f}%, "
              f"final={all_results[cond_name]['final_val_acc']:.2f}%")
        print(f"  Update samples: {len(cond_updates)}, Transfers: {len(cond_transfers)}")

        # --- Mismatch analysis ---
        mismatch_a_vals = [r["mismatch_a"] for r in cond_updates
                           if not math.isnan(r["mismatch_a"])]
        mismatch_b_vals = [r["mismatch_b"] for r in cond_updates
                           if not math.isnan(r["mismatch_b"])]

        if mismatch_a_vals:
            print(f"\n  What auto_scale tracks vs what tiles actually see:")
            print(f"    tile_a inputs: (XB, d)")
            print(f"      product_tracked / product_a = {np.mean(mismatch_a_vals):.4f} "
                  f"(std={np.std(mismatch_a_vals):.4f})")
            print(f"      min={np.min(mismatch_a_vals):.4f}, max={np.max(mismatch_a_vals):.4f}")
        if mismatch_b_vals:
            print(f"    tile_b inputs: (x, DA)")
            print(f"      product_tracked / product_b = {np.mean(mismatch_b_vals):.4f} "
                  f"(std={np.std(mismatch_b_vals):.4f})")
            print(f"      min={np.min(mismatch_b_vals):.4f}, max={np.max(mismatch_b_vals):.4f}")

        # --- Product magnitudes ---
        prod_tracked = [r["product_tracked"] for r in cond_updates]
        prod_a = [r["product_a"] for r in cond_updates]
        prod_b = [r["product_b"] for r in cond_updates]
        print(f"\n  Product magnitudes:")
        print(f"    tracked (max|x|·max|d|): mean={np.mean(prod_tracked):.6f}")
        print(f"    tile_a  (max|XB|·max|d|): mean={np.mean(prod_a):.6f}")
        print(f"    tile_b  (max|x|·max|DA|): mean={np.mean(prod_b):.6f}")

        # --- Amplification ---
        amps = [r["amplification"] for r in cond_updates
                if not math.isnan(r["amplification"])]
        if amps:
            print(f"\n  LR amplification:")
            print(f"    mean={np.mean(amps):.4f}, std={np.std(amps):.4f}")
            print(f"    min={np.min(amps):.4f}, max={np.max(amps):.4f}, final={amps[-1]:.4f}")

        # --- Saturation ---
        a_sat = [r["A_saturated_frac"] for r in cond_updates]
        b_sat = [r["B_saturated_frac"] for r in cond_updates]
        print(f"\n  AB Tile Saturation (|w|>0.9·w_max):")
        print(f"    A: mean={np.mean(a_sat)*100:.2f}%, max={np.max(a_sat)*100:.2f}%, "
              f"final={a_sat[-1]*100:.2f}%")
        print(f"    B: mean={np.mean(b_sat)*100:.2f}%, max={np.max(b_sat)*100:.2f}%, "
              f"final={b_sat[-1]*100:.2f}%")

        # --- Weight magnitudes ---
        a_absmax = [r["A_w_absmax"] for r in cond_updates]
        b_absmax = [r["B_w_absmax"] for r in cond_updates]
        print(f"\n  Weight |max|:")
        print(f"    A: mean={np.mean(a_absmax):.4f}, max={np.max(a_absmax):.4f}, "
              f"final={a_absmax[-1]:.4f}")
        print(f"    B: mean={np.mean(b_absmax):.4f}, max={np.max(b_absmax):.4f}, "
              f"final={b_absmax[-1]:.4f}")

        # --- Time trend (first quarter vs last quarter) ---
        n = len(cond_updates)
        q1 = n // 4
        if q1 > 0:
            first_q_a_sat = np.mean([r["A_saturated_frac"] for r in cond_updates[:q1]])
            last_q_a_sat = np.mean([r["A_saturated_frac"] for r in cond_updates[-q1:]])
            first_q_b_sat = np.mean([r["B_saturated_frac"] for r in cond_updates[:q1]])
            last_q_b_sat = np.mean([r["B_saturated_frac"] for r in cond_updates[-q1:]])
            print(f"\n  Time trend (first 25% → last 25%):")
            print(f"    A saturation: {first_q_a_sat*100:.2f}% → {last_q_a_sat*100:.2f}%")
            print(f"    B saturation: {first_q_b_sat*100:.2f}% → {last_q_b_sat*100:.2f}%")

            if mismatch_a_vals:
                first_q_mm_a = np.mean(mismatch_a_vals[:q1])
                last_q_mm_a = np.mean(mismatch_a_vals[-q1:])
                first_q_mm_b = np.mean(mismatch_b_vals[:q1])
                last_q_mm_b = np.mean(mismatch_b_vals[-q1:])
                print(f"    mismatch_a: {first_q_mm_a:.4f} → {last_q_mm_a:.4f}")
                print(f"    mismatch_b: {first_q_mm_b:.4f} → {last_q_mm_b:.4f}")

    # =================================================================
    # CROSS-CONDITION COMPARISON
    # =================================================================
    print("\n" + "=" * 80)
    print("  CROSS-CONDITION COMPARISON")
    print("=" * 80)

    base_updates = [r for r in update_records if r["condition"] == "BASELINE"]
    auto_updates = [r for r in update_records if r["condition"] == "AUTO_SCALE"]

    if base_updates and auto_updates:
        # Saturation comparison
        base_a_sat = np.mean([r["A_saturated_frac"] for r in base_updates])
        auto_a_sat = np.mean([r["A_saturated_frac"] for r in auto_updates])
        base_b_sat = np.mean([r["B_saturated_frac"] for r in base_updates])
        auto_b_sat = np.mean([r["B_saturated_frac"] for r in auto_updates])

        print(f"\n  Saturation comparison:")
        print(f"    BASELINE:   A_sat={base_a_sat*100:.2f}%, B_sat={base_b_sat*100:.2f}%")
        print(f"    AUTO_SCALE: A_sat={auto_a_sat*100:.2f}%, B_sat={auto_b_sat*100:.2f}%")
        if base_a_sat > 1e-6:
            print(f"    → auto_scale changes A saturation by {auto_a_sat/base_a_sat:.2f}x")
        if base_b_sat > 1e-6:
            print(f"    → auto_scale changes B saturation by {auto_b_sat/base_b_sat:.2f}x")

    # =================================================================
    # IDEAL AUTO_SCALE COMPARISON
    # =================================================================
    print("\n" + "=" * 80)
    print("  IDEAL AUTO_SCALE COMPARISON")
    print("=" * 80)

    if base_updates:
        # From BASELINE, compute what the ideal amplification would be for each tile
        prod_tracked_vals = [r["product_tracked"] for r in base_updates]
        prod_a_vals = [r["product_a"] for r in base_updates]
        prod_b_vals = [r["product_b"] for r in base_updates]

        mean_tracked = np.mean(prod_tracked_vals)
        mean_prod_a = np.mean(prod_a_vals)
        mean_prod_b = np.mean(prod_b_vals)

        print(f"\n  Based on BASELINE measurements:")
        print(f"    mean product_tracked (max|x|·max|d|):  {mean_tracked:.6f}")
        print(f"    mean product_a (max|XB|·max|d|):       {mean_prod_a:.6f}")
        print(f"    mean product_b (max|x|·max|DA|):       {mean_prod_b:.6f}")

        if mean_tracked > 1e-12:
            # Current auto_scale: amplification = 1 / product_tracked
            current_amp = 1.0 / mean_tracked
            # Ideal for tile_a: amplification = 1 / product_a
            ideal_a = 1.0 / mean_prod_a if mean_prod_a > 1e-12 else float('nan')
            # Ideal for tile_b: amplification = 1 / product_b
            ideal_b = 1.0 / mean_prod_b if mean_prod_b > 1e-12 else float('nan')

            print(f"\n  Amplification (1/product):")
            print(f"    Current auto_scale: {current_amp:.2f}x (based on max|x|·max|d|)")
            print(f"    Ideal for tile_a:   {ideal_a:.2f}x (based on max|XB|·max|d|)")
            print(f"    Ideal for tile_b:   {ideal_b:.2f}x (based on max|x|·max|DA|)")

            if not math.isnan(ideal_a):
                print(f"\n  Over-correction:")
                print(f"    tile_a: current/ideal = {current_amp/ideal_a:.2f}x "
                      f"({'over' if current_amp > ideal_a else 'under'}-corrects)")
            if not math.isnan(ideal_b):
                print(f"    tile_b: current/ideal = {current_amp/ideal_b:.2f}x "
                      f"({'over' if current_amp > ideal_b else 'under'}-corrects)")

    # =================================================================
    # TRANSFER ANALYSIS
    # =================================================================
    print("\n" + "=" * 80)
    print("  TRANSFER-TIME AB/C WEIGHT ANALYSIS")
    print("=" * 80)

    for cond_name, _ in CONDITIONS:
        cond_trans = [r for r in transfer_records if r["condition"] == cond_name]
        if not cond_trans:
            continue

        print(f"\n--- {cond_name} ({len(cond_trans)} transfers) ---")

        # A/B norms before transfer
        a_f_pre = [r["A_F_pre"] for r in cond_trans]
        b_f_pre = [r["B_F_pre"] for r in cond_trans]
        ab_f_pre = [r["AB_F_pre"] for r in cond_trans]
        delta_c = [r["delta_C_F"] for r in cond_trans]

        print(f"  Before transfer:")
        print(f"    ||A||_F: mean={np.mean(a_f_pre):.4f}, std={np.std(a_f_pre):.4f}")
        print(f"    ||B||_F: mean={np.mean(b_f_pre):.4f}, std={np.std(b_f_pre):.4f}")
        print(f"    ||A@B||_F: mean={np.mean(ab_f_pre):.4f}, std={np.std(ab_f_pre):.4f}")

        # After reinit
        a_f_post = [r["A_F_post"] for r in cond_trans]
        b_f_post = [r["B_F_post"] for r in cond_trans]
        print(f"  After reinit:")
        print(f"    ||A||_F: mean={np.mean(a_f_post):.4f}, std={np.std(a_f_post):.4f}")
        print(f"    ||B||_F: mean={np.mean(b_f_post):.4f}, std={np.std(b_f_post):.4f}")

        # Delta C
        print(f"  Transfer effect:")
        print(f"    ||ΔC||_F: mean={np.mean(delta_c):.6f}, std={np.std(delta_c):.6f}")

        # AB growth trend
        if len(ab_f_pre) > 2:
            slope = np.polyfit(range(len(ab_f_pre)), ab_f_pre, 1)[0]
            print(f"    ||A@B||_F trend: slope={slope:.6f}/transfer "
                  f"({'growing' if slope > 0 else 'shrinking'})")

        # Saturation at transfer time
        a_sat_vals = [r["A_sat_pre"] for r in cond_trans]
        b_sat_vals = [r["B_sat_pre"] for r in cond_trans]
        print(f"  Saturation at transfer:")
        print(f"    A: mean={np.mean(a_sat_vals)*100:.2f}%, max={np.max(a_sat_vals)*100:.2f}%")
        print(f"    B: mean={np.mean(b_sat_vals)*100:.2f}%, max={np.max(b_sat_vals)*100:.2f}%")

        # Per-rank balance (last 10 transfers)
        rank = RANK
        last_n = cond_trans[-10:]
        print(f"  Per-rank norms (last {len(last_n)} transfers, before transfer):")
        for k in range(rank):
            a_k = np.mean([r[f"a_rank{k}_pre"] for r in last_n])
            b_k = np.mean([r[f"b_rank{k}_pre"] for r in last_n])
            print(f"    rank {k}: ||a_k||={a_k:.4f}, ||b_k||={b_k:.4f}, "
                  f"product={a_k*b_k:.6f}")

    # =================================================================
    # KEY FINDINGS & PRESCRIPTION
    # =================================================================
    print("\n" + "=" * 80)
    print("  KEY FINDINGS")
    print("=" * 80)

    if base_updates and auto_updates:
        # Q1: Mismatch degree
        mm_a = [r["mismatch_a"] for r in base_updates if not math.isnan(r["mismatch_a"])]
        mm_b = [r["mismatch_b"] for r in base_updates if not math.isnan(r["mismatch_b"])]
        if mm_a and mm_b:
            print(f"\n  Q1. Mismatch degree (from BASELINE):")
            print(f"      product_tracked / product_a = {np.mean(mm_a):.4f}")
            print(f"      product_tracked / product_b = {np.mean(mm_b):.4f}")
            if np.mean(mm_a) > 1.5 or np.mean(mm_b) > 1.5:
                print(f"      → SIGNIFICANT mismatch: auto_scale uses wrong normalization basis")
            else:
                print(f"      → Moderate mismatch")

        # Q2: Saturation evidence
        base_a = np.mean([r["A_saturated_frac"] for r in base_updates])
        auto_a = np.mean([r["A_saturated_frac"] for r in auto_updates])
        base_b = np.mean([r["B_saturated_frac"] for r in base_updates])
        auto_b = np.mean([r["B_saturated_frac"] for r in auto_updates])
        print(f"\n  Q2. Saturation evidence:")
        print(f"      BASELINE → AUTO_SCALE:")
        print(f"        A: {base_a*100:.2f}% → {auto_a*100:.2f}%")
        print(f"        B: {base_b*100:.2f}% → {auto_b*100:.2f}%")
        if auto_a > base_a * 1.5 or auto_b > base_b * 1.5:
            print(f"      → YES, auto_scale increases saturation significantly")
        else:
            print(f"      → No significant saturation increase")

        # Q3: Time trend
        n = len(auto_updates)
        q1_len = n // 4
        if q1_len > 0:
            first_amp = np.mean([r["amplification"] for r in auto_updates[:q1_len]
                                if not math.isnan(r["amplification"])])
            last_amp = np.mean([r["amplification"] for r in auto_updates[-q1_len:]
                               if not math.isnan(r["amplification"])])
            print(f"\n  Q3. Time trend:")
            print(f"      Amplification: {first_amp:.2f}x (early) → {last_amp:.2f}x (late)")
            if last_amp > first_amp * 1.2:
                print(f"      → Amplification GROWS over time (unstable)")
            elif last_amp < first_amp * 0.8:
                print(f"      → Amplification SHRINKS over time")
            else:
                print(f"      → Amplification relatively STABLE")

        # Q4: Prescription
        print(f"\n  Q4. Prescription:")
        if mm_a and mm_b:
            mean_mm_a = np.mean(mm_a)
            mean_mm_b = np.mean(mm_b)
            if mean_mm_a > 2.0 or mean_mm_b > 2.0:
                print(f"      auto_scale tracks max|x|·max|d| but tiles see different products.")
                print(f"      Options:")
                print(f"        a) Per-tile EMA: track max|XB|·max|d| for A, max|x|·max|DA| for B")
                print(f"        b) Amplification cap: clamp amplification to prevent over-correction")
                print(f"        c) Use geometric mean of tile-specific products")
            else:
                print(f"      Mismatch is moderate. Consider:")
                print(f"        a) Amplification cap to limit extreme values")
                print(f"        b) Slower momentum (0.999) for more stability")


# =============================================================================
# Main
# =============================================================================

def main():
    global transfer_records, update_records

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/data/AIMC_LoRA_results/auto_scale_detail_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Auto-Scale Detail Diagnostics")
    print(f"Device: {DEVICE}")
    print(f"LRTT: rank={RANK}, te={TE}, lr={LR}, tlr={TLR}")
    print(f"Epochs: {EPOCHS}, batch_size={BATCH_SIZE}, seed={SEED}")
    print(f"Output: {out_dir}")

    train_loader, val_loader = load_data()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    all_results = {}
    for cond_name, auto_scale in CONDITIONS:
        rpu_config = create_lrtt_config(auto_scale=auto_scale)
        results = train_condition(cond_name, rpu_config, train_loader, val_loader)
        all_results[cond_name] = results

    # Save CSVs
    print(f"\n{'='*60}")
    print(f"  Saving results...")
    print(f"{'='*60}")
    save_csv(update_records, os.path.join(out_dir, "auto_scale_update_detail.csv"), "update")
    save_csv(transfer_records, os.path.join(out_dir, "auto_scale_transfer_detail.csv"), "transfer")

    # Print summary
    print_summary(all_results)

    print(f"\n\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()

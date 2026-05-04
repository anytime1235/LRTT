#!/usr/bin/env python3
"""Reset-mode ablation sweep, 5x5 (AF, UNR) grid, fixed best HP.

Adds three new method variants on top of the existing sweep_5x5_fixed_hp.py:

  - tikitaka_reset:    TikiTaka v1 with TransferCompound(with_reset_prob=1.0)
                       → transferred columns of the fast tile are zeroed each
                       transfer (TikiTaka-native hard reset). Uses plain
                       TransferCompound rather than ChoppedTransferCompound
                       because the chopped device's C++ checkSupported()
                       (rpu_chopped_transfer_device.cpp:82) hard-rejects any
                       with_reset_prob>0. The baseline tikitaka_v1 uses
                       ChoppedTransferCompound with chopper fully disabled
                       (in_chop_prob=out_chop_prob=0, no_buffer=True), which
                       is bit-identical at runtime to plain TransferCompound,
                       so this swap toggles only the reset axis.
  - lrtt_v1_reset:     LRTT v1 with reinit_mode="standard" (instead of "decay")
                       → A and B are reinitialized (A=Kaiming, B=Kaiming) every
                       transfer, wiping accumulated AF-biased state.
  - lrtt_v2_noreset:   LRTT v2 with selector_reset_b_on_advance=False
                       → B is NOT zeroed when the row selector advances.
                       The PythonLRTTDevice docstring marks this as "Required
                       for coordinate consistency" — this variant tests
                       empirically whether v2 can still train without reset.

Best HPs are reused from sweep_5x5_fixed_hp.py. The intent is to test whether
adding a hard-reset to v1 / TikiTaka closes the AF-sensitivity gap with v2,
and whether removing reset from v2 breaks training entirely.

Output: /root/LRTT/results/sweep_5x5_reset_modes/
  tikitaka_reset.json
  lrtt_v1_reset.json
  lrtt_v2_noreset.json

Run:
  PYTHONPATH=/root/LRTT/src \
    LD_LIBRARY_PATH=/root/.venv310/lib/python3.10/site-packages/aihwkit.libs:/root/.venv310/lib/python3.10/site-packages/torch/lib \
    /root/.venv310/bin/python /root/LRTT/experiments/sweep_5x5_reset_modes.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("LRTT_SILENT", "1")
sys.path.insert(0, "/root/LRTT/experiments")

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig, UnitCellRPUConfig,
)
from aihwkit.simulator.configs.compounds import (
    ChoppedTransferCompound, TransferCompound,
)
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs.utils import (
    BoundManagementType, IOParameters,
    NoiseManagementType, UpdateParameters,
)

# Reuse data loader / hyperparams from existing modules
import hp_search_tikitaka_3hp_gamma_af as tt3_mod
import hp_search_v1_decay_gamma_af_2stage as v1_mod
import hp_search_v2_rank8_gamma_af_2stage as v2_mod


AF_GRID  = [0.0, 1.0, 2.0, 5.0, 10.0]
UNR_GRID = [0.0, 1.0, 3.0, 5.0, 10.0]

BEST_HP = {
    "tikitaka_reset":  {"transfer_lr": 1.0, "fast_lr": 0.3, "classifier_lr": 1.0},
    "lrtt_v1_reset":   {"lr": 0.1, "tlr": 0.001, "clr": 0.3},
    "lrtt_v2_noreset": {"lr": 1.0, "tlr": 0.1, "clr": 1.0},
}

OUT_DIR = "/root/LRTT/results/sweep_5x5_reset_modes"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# TikiTaka with hard reset (with_reset_prob=1.0)
# ---------------------------------------------------------------------------
def _create_rpu_tikitaka_reset(transfer_lr, fast_lr, af_ratio, unr):
    lt_param = tt3_mod._ab_lifetime_param(tt3_mod.LIFETIME_PHYS)
    fast = tt3_mod._make_fast_device(lt_param, af_ratio, unr)
    slow = tt3_mod._make_slow_device()
    # NOTE: plain TransferCompound (NOT ChoppedTransferCompound). The chopped
    # device's C++ checkSupported() rejects with_reset_prob>0 unconditionally.
    # The baseline tikitaka_v1 runs ChoppedTransferCompound with the chopper
    # fully disabled, which is bit-identical at runtime to TransferCompound,
    # so this swap leaves the reset toggle as the only differing axis.
    rpu = UnitCellRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[fast, slow],
            transfer_every=tt3_mod.TE,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=tt3_mod.GAMMA,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=False,
            with_reset_prob=1.0,                 # ← reset mode (fast tile col)
            random_selection=False,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=tt3_mod.DESIRED_BL,
                update_bl_management=True,
                update_management=True,
            ),
        )
    )
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return rpu


def run_tikitaka_reset(transfer_lr, fast_lr, classifier_lr, af_ratio, unr):
    torch.manual_seed(tt3_mod.SEED)
    train_loader, val_loader = tt3_mod.build_loaders(smoke=False)
    rpu = _create_rpu_tikitaka_reset(transfer_lr, fast_lr, af_ratio, unr)
    model = AnalogSequential(
        AnalogLinear(784, tt3_mod.HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(tt3_mod.HIDDEN, tt3_mod.OUT, bias=True,
                     rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    optimizer = AnalogSGD(model.parameters(), lr=tt3_mod.LR)
    optimizer.regroup_param_groups(model)
    for pg in optimizer.param_groups:
        if not pg.get("analog", False):
            pg["lr"] = classifier_lr
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience = 0
    for epoch in range(1, tt3_mod.EPOCHS + 1):
        model.train()
        for d, t in train_loader:
            d = d.to(DEVICE, non_blocking=True).view(d.shape[0], -1)
            t = t.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(d), t)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for d, t in val_loader:
                d = d.to(DEVICE, non_blocking=True).view(d.shape[0], -1)
                t = t.to(DEVICE, non_blocking=True)
                correct += model(d).argmax(dim=1).eq(t).sum().item()
                total += t.size(0)
        acc = 100.0 * correct / total
        scheduler.step()
        if acc > best_acc:
            best_acc = acc; patience = 0
        else:
            patience += 1
        if epoch >= 5 and best_acc < 50.0:
            break
        if patience >= tt3_mod.EARLY_STOP_PATIENCE:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc


# ---------------------------------------------------------------------------
# LRTT v1 with hard reset (reinit_mode="standard")
# ---------------------------------------------------------------------------
def _create_model_v1_reset(tlr, af_ratio, unr):
    lt_param = v1_mod._ab_lifetime_param(v1_mod.LIFETIME_PHYS)
    a = v1_mod._make_ab_device(lt_param, af_ratio, unr)
    b = v1_mod._make_ab_device(lt_param, af_ratio, unr)
    c = v1_mod._make_c_device()
    dev = PythonLRTTDevice(
        rank=v1_mod.RANK, transfer_every=v1_mod.TE,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode="standard",                  # ← reset mode (vs "decay")
        decay_factor=1.0,
        unit_cell_devices=[a, b, c],
    )
    dev.transfer_lr = tlr
    dev.forward_inject = False
    dev.update_mode = "lora"
    dev.transfer_method = "onehot"
    dev.transfer_mode = "off"
    rpu = PythonLRTTRPUConfig(device=dev)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return AnalogSequential(
        AnalogLinear(784, v1_mod.HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(v1_mod.HIDDEN, v1_mod.OUT, bias=True,
                     rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)


def run_lrtt_v1_reset(lr, tlr, clr, af_ratio, unr):
    torch.manual_seed(v1_mod.SEED)
    train_loader, val_loader = v1_mod.build_loaders(smoke=False)
    model = _create_model_v1_reset(tlr, af_ratio, unr)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    for pg in optimizer.param_groups:
        if not pg.get("analog", False):
            pg["lr"] = clr
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience = 0
    for epoch in range(1, v1_mod.EPOCHS + 1):
        model.train()
        for d, t in train_loader:
            d = d.to(DEVICE, non_blocking=True).view(d.shape[0], -1)
            t = t.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(d), t)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for d, t in val_loader:
                d = d.to(DEVICE, non_blocking=True).view(d.shape[0], -1)
                t = t.to(DEVICE, non_blocking=True)
                correct += model(d).argmax(dim=1).eq(t).sum().item()
                total += t.size(0)
        acc = 100.0 * correct / total
        scheduler.step()
        if acc > best_acc:
            best_acc = acc; patience = 0
        else:
            patience += 1
        if epoch >= 5 and best_acc < 50.0:
            break
        if patience >= v1_mod.EARLY_STOP_PATIENCE:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc


# ---------------------------------------------------------------------------
# LRTT v2 WITHOUT reset (selector_reset_b_on_advance=False)
# ---------------------------------------------------------------------------
def _create_model_v2_noreset(tlr, af_ratio, unr):
    lt_param = v2_mod._ab_lifetime_param(v2_mod.LIFETIME_PHYS)
    a = v2_mod._make_ab_device(lt_param, af_ratio, unr)
    b = v2_mod._make_ab_device(lt_param, af_ratio, unr)
    c = v2_mod._make_c_device()
    dev = PythonLRTTDevice(
        rank=v2_mod.RANK, transfer_every=v2_mod.TE,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode="standard", decay_factor=1.0,
        b_init_mode="zero",
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        forward_inject=False,
        selector_axis="row", selector_policy="shuffled_cycle",
        selector_seed=v2_mod.SEED,
        selector_reset_b_on_advance=False,       # ← ablation: no reset
        cap_stabilizer_enabled=True, cap_rho=1.0,
        cap_compensate_transfer=True,
        unit_cell_devices=[a, b, c],
    )
    dev.transfer_lr = tlr
    dev.transfer_mode = "off"

    rpu = PythonLRTTRPUConfig(device=dev)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6

    return AnalogSequential(
        AnalogLinear(784, v2_mod.HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(v2_mod.HIDDEN, v2_mod.OUT, bias=True,
                     rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)


def run_lrtt_v2_noreset(lr, tlr, classifier_lr, af_ratio, update_noise_ratio):
    torch.manual_seed(v2_mod.SEED)
    train_loader, val_loader = v2_mod.build_loaders(smoke=False)
    model = _create_model_v2_noreset(tlr, af_ratio, update_noise_ratio)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    for pg in optimizer.param_groups:
        if not pg.get("analog", False):
            pg["lr"] = classifier_lr
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience = 0
    for epoch in range(1, v2_mod.EPOCHS + 1):
        model.train()
        for d, t in train_loader:
            d = d.to(DEVICE, non_blocking=True).view(d.shape[0], -1)
            t = t.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(d), t)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for d, t in val_loader:
                d = d.to(DEVICE, non_blocking=True).view(d.shape[0], -1)
                t = t.to(DEVICE, non_blocking=True)
                correct += model(d).argmax(dim=1).eq(t).sum().item()
                total += t.size(0)
        acc = 100.0 * correct / total
        scheduler.step()
        if acc > best_acc:
            best_acc = acc; patience = 0
        else:
            patience += 1
        if epoch >= 5 and best_acc < 50.0:
            break
        if patience >= v2_mod.EARLY_STOP_PATIENCE:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc


# ---------------------------------------------------------------------------
# Sweep driver (mirrors sweep_5x5_fixed_hp.run_method)
# ---------------------------------------------------------------------------
def run_one(method, hp, af, unr):
    if method == "tikitaka_reset":
        return run_tikitaka_reset(hp["transfer_lr"], hp["fast_lr"],
                                  hp["classifier_lr"], af, unr)
    if method == "lrtt_v1_reset":
        return run_lrtt_v1_reset(hp["lr"], hp["tlr"], hp["clr"], af, unr)
    if method == "lrtt_v2_noreset":
        return run_lrtt_v2_noreset(hp["lr"], hp["tlr"], hp["clr"], af, unr)
    raise ValueError(method)


def run_method(method, out_dir):
    cache_path = f"{out_dir}/{method}_cache.json"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    print(f"\n{'='*70}\n{method.upper()}  best HP={BEST_HP[method]}\n{'='*70}",
          flush=True)
    print(f"Cache: {cache_path} ({len(cache)} entries)", flush=True)

    grid = []
    t0 = time.time()
    n_new = 0
    for af in AF_GRID:
        for unr in UNR_GRID:
            k = f"af{af}_unr{unr}"
            if k in cache:
                acc = cache[k]
                src = "cache"
            else:
                t_start = time.time()
                acc = run_one(method, BEST_HP[method], af, unr)
                cache[k] = acc
                tmp = cache_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(cache, f, indent=2)
                os.replace(tmp, cache_path)
                n_new += 1
                src = f"new ({time.time()-t_start:.0f}s)"
            grid.append({
                "af_ratio": af, "update_noise_ratio": unr,
                "acc": round(float(acc), 2), "source": src,
            })
            elapsed = (time.time() - t0) / 60
            print(f"  AF={af:>4}, UNR={unr:>4} → {acc:6.2f}%  [{src}]  "
                  f"elapsed={elapsed:.1f}min  new={n_new}", flush=True)

    out_path = f"{out_dir}/{method}.json"
    with open(out_path, "w") as f:
        json.dump({
            "method": method,
            "best_hp": BEST_HP[method],
            "af_grid": AF_GRID,
            "unr_grid": UNR_GRID,
            "wall_seconds": round(time.time() - t0, 1),
            "n_new_trials": n_new,
            "grid": grid,
        }, f, indent=2)
    print(f"\n  Saved {out_path}  (wall={(time.time()-t0)/60:.1f}min, "
          f"{n_new} new trials)", flush=True)
    return grid


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--methods", nargs="+",
        default=["tikitaka_reset", "lrtt_v1_reset", "lrtt_v2_noreset"],
        choices=["tikitaka_reset", "lrtt_v1_reset", "lrtt_v2_noreset"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"5x5 reset-mode sweep → {OUT_DIR}", flush=True)
    print(f"AF  ∈ {AF_GRID}\nUNR ∈ {UNR_GRID}", flush=True)

    t_total = time.time()
    all_results = {}
    for method in args.methods:
        all_results[method] = run_method(method, OUT_DIR)

    with open(f"{OUT_DIR}/all_methods.json", "w") as f:
        json.dump({
            "af_grid": AF_GRID,
            "unr_grid": UNR_GRID,
            "best_hp": BEST_HP,
            "results": all_results,
            "wall_seconds_total": round(time.time() - t_total, 1),
        }, f, indent=2)

    print(f"\n{'='*70}\nALL DONE — wall={((time.time()-t_total)/60):.1f}min\n"
          f"Output: {OUT_DIR}\n{'='*70}", flush=True)


if __name__ == "__main__":
    main()

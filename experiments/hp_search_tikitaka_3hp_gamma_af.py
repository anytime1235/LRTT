#!/usr/bin/env python3
"""TikiTaka v1 — small HP search varying transfer_lr, fast_lr, classifier_lr.

Existing TikiTaka script fixed transfer_lr=fast_lr=1.0, classifier_lr=0.1
(coupled with analog lr=0.1). This script decouples those 3 HPs to find a
TikiTaka HP that exploits a faster FP classifier (analogous to v1/v2 best HP).

HP grid (12 cells):
  transfer_lr   ∈ {1.0, 3.0}
  fast_lr       ∈ {0.3, 1.0}
  classifier_lr ∈ {0.1, 0.3, 1.0}
  lr fixed      = 0.1

Screening grid (9 (AF, UNR) points per cell):
  AF  ∈ {0, 1, 5}
  UNR ∈ {0, 1, 3}

Total: 12 × 9 = 108 trials, ~4.5 hours on A100.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

os.environ.setdefault("LRTT_SILENT", "1")

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig, UnitCellRPUConfig,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.utils import (
    BoundManagementType, IOParameters,
    NoiseManagementType, UpdateParameters,
)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
SEED = 42
HIDDEN = 256
OUT = 10
TAU_SEC = 46505.0

LR = 0.1                    # fixed analog lr
LIFETIME_PHYS = 1000
DESIRED_BL = 31
TE = 10
GAMMA = 0.0

TRANSFER_LR_GRID  = [1.0, 3.0]
FAST_LR_GRID      = [0.3, 1.0]
CLASSIFIER_LR_GRID = [0.1, 0.3, 1.0]

AF_GRID  = [0.0, 1.0, 5.0]
UNR_GRID = [0.0, 1.0, 3.0]


def _ab_lifetime_param(lt):
    if lt is None or lt <= 0:
        return 0.0
    dt = -TAU_SEC * math.log(1 - 1.0 / lt)
    return 1.0 / (1 - math.exp(-dt / TAU_SEC))


_TRAIN_DS = None
_VAL_DS = None


def build_loaders(smoke=False):
    global _TRAIN_DS, _VAL_DS
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if _TRAIN_DS is None:
        _TRAIN_DS = datasets.MNIST("/tmp/mnist", download=True, train=True,
                                    transform=transform)
        _VAL_DS = datasets.MNIST("/tmp/mnist", download=True, train=False,
                                  transform=transform)
    if smoke:
        train_ds = Subset(_TRAIN_DS, range(1024))
        val_ds = Subset(_VAL_DS, range(1024))
        bs, nw = 128, 0
    else:
        train_ds, val_ds = _TRAIN_DS, _VAL_DS
        bs, nw = BATCH_SIZE, 4
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                   pin_memory=(DEVICE.type == "cuda")),
        DataLoader(val_ds, batch_size=bs, shuffle=False,
                   num_workers=max(0, nw - 2),
                   pin_memory=(DEVICE.type == "cuda"))
    )


def _make_fast_device(lt_param, af_ratio, unr):
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0,
        w_max=1.0, w_min=-1.0,
        gamma_up=af_ratio, gamma_down=af_ratio,
        mult_noise=False,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3 * unr, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lt_param, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


def _make_slow_device():
    return LinearStepDevice(
        dw_min=2.0 / 1024.0, w_max=1.0, w_min=-1.0,            # 10-bit
        gamma_up=0.0, gamma_down=0.0,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0,
    )


def create_rpu(transfer_lr, fast_lr, af_ratio, unr):
    lt_param = _ab_lifetime_param(LIFETIME_PHYS)
    fast = _make_fast_device(lt_param, af_ratio, unr)
    slow = _make_slow_device()
    rpu = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[fast, slow],
            transfer_every=TE,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=GAMMA,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=False,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=DESIRED_BL,
                update_bl_management=True,
                update_management=True,
            ),
            no_buffer=True, in_chop_prob=0.0, out_chop_prob=0.0,
            auto_scale=False, auto_momentum=0.99,
        )
    )
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return rpu


def run_trial(transfer_lr, fast_lr, classifier_lr, af_ratio, unr, *, smoke=False):
    torch.manual_seed(SEED)
    train_loader, val_loader = build_loaders(smoke=smoke)
    rpu = create_rpu(transfer_lr, fast_lr, af_ratio, unr)
    model = AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(HIDDEN, OUT, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    # Assign separate classifier_lr to FP param groups
    for pg in optimizer.param_groups:
        if not pg.get("analog", False):
            pg["lr"] = classifier_lr
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience = 0
    epochs = 2 if smoke else EPOCHS
    for epoch in range(1, epochs + 1):
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
        if not smoke and epoch >= 5 and best_acc < 50.0:
            break
        if not smoke and patience >= EARLY_STOP_PATIENCE:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc


def cache_key(transfer_lr, fast_lr, classifier_lr, af, unr):
    return (f"tlr{transfer_lr}_flr{fast_lr}_clr{classifier_lr}"
            f"_af{af}_unr{unr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("=== smoke ===")
        acc = run_trial(transfer_lr=1.0, fast_lr=1.0, classifier_lr=1.0,
                        af_ratio=1.0, unr=1.0, smoke=True)
        print(f"  acc={acc:.2f}%")
        return

    output_dir = args.out or "/root/LRTT/results/hp_search_tikitaka_3hp_gamma_af"
    os.makedirs(output_dir, exist_ok=True)
    cache_path = f"{output_dir}/trial_cache.json"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"[resume] loaded {len(cache)} cached trials")

    cells = [(t, f, c)
             for t in TRANSFER_LR_GRID
             for f in FAST_LR_GRID
             for c in CLASSIFIER_LR_GRID]
    n_cells = len(cells)
    print("=" * 80)
    print(f"TikiTaka v1 — 3-HP search (transfer_lr × fast_lr × classifier_lr)")
    print(f"Cells: {n_cells} (analog lr fixed = {LR})")
    print(f"Screening: {len(AF_GRID)*len(UNR_GRID)} (AF×UNR) per cell")
    print(f"Total trials: {n_cells*len(AF_GRID)*len(UNR_GRID)}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    t_total = time.time()
    cells_records = []
    for ci, (t_lr, f_lr, c_lr) in enumerate(cells, 1):
        cell_t0 = time.time()
        screening_records = []
        accs = []
        print(f"\n[cell {ci}/{n_cells}] tlr={t_lr}, flr={f_lr}, clr={c_lr}", flush=True)
        for af in AF_GRID:
            for unr in UNR_GRID:
                k = cache_key(t_lr, f_lr, c_lr, af, unr)
                if k in cache:
                    acc = cache[k]
                    src = "cache"
                else:
                    t0 = time.time()
                    acc = run_trial(t_lr, f_lr, c_lr, af, unr)
                    cache[k] = acc
                    with open(cache_path, "w") as f:
                        json.dump(cache, f, indent=2)
                    src = f"new ({time.time()-t0:.0f}s)"
                accs.append(acc)
                screening_records.append({
                    "af_ratio": af, "update_noise_ratio": unr,
                    "acc": round(acc, 2),
                })
                print(f"  AF={af:>3}, UNR={unr:>3} → {acc:6.2f}%  [{src}]",
                      flush=True)
        mean_acc = sum(accs) / len(accs)
        cells_records.append({
            "transfer_lr": t_lr, "fast_lr": f_lr, "classifier_lr": c_lr,
            "mean_acc": round(mean_acc, 2),
            "screening_records": screening_records,
            "wall_seconds": round(time.time() - cell_t0, 1),
        })
        elapsed = (time.time() - t_total) / 60
        print(f"  ↳ cell mean = {mean_acc:.2f}%  (cell wall={(time.time()-cell_t0)/60:.1f}min, "
              f"total elapsed={elapsed:.1f}min)", flush=True)

    cells_records.sort(key=lambda r: -r["mean_acc"])
    best = cells_records[0]
    final = {
        "method": "tikitaka_v1_3hp_gamma_af",
        "lr": LR, "lifetime_phys": LIFETIME_PHYS, "te": TE,
        "best_transfer_lr": best["transfer_lr"],
        "best_fast_lr": best["fast_lr"],
        "best_classifier_lr": best["classifier_lr"],
        "best_mean_acc": best["mean_acc"],
        "n_cells": n_cells,
        "n_trials": n_cells * len(AF_GRID) * len(UNR_GRID),
        "wall_seconds_total": round(time.time() - t_total, 1),
        "cells_evaluated": cells_records,
    }
    with open(f"{output_dir}/results_final.json", "w") as f:
        json.dump(final, f, indent=2)

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*80}\nDONE — wall={elapsed:.1f}min")
    print(f"Best HP: tlr={best['transfer_lr']}, flr={best['fast_lr']}, "
          f"clr={best['classifier_lr']} → mean acc {best['mean_acc']:.2f}%")
    print(f"\nTop 5 cells:")
    print(f"  {'tlr':>5} {'flr':>5} {'clr':>5} {'mean':>7}")
    for c in cells_records[:5]:
        print(f"  {c['transfer_lr']:>5} {c['fast_lr']:>5} "
              f"{c['classifier_lr']:>5} {c['mean_acc']:>7.2f}")


if __name__ == "__main__":
    main()

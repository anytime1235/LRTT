#!/usr/bin/env python3
"""LRTT-v1 (decay/lora/onehot) γ-AF / UNR 2-stage sweep, rank=8.

Same 2-stage structure as v2 with classifier_lr decoupled.
HP grid: analog_lr × tlr × classifier_lr = 3 × 3 × 3 = 27 cells.
TLR range tuned for v1 best (~8e-4).
"""
from __future__ import annotations

import os
os.environ.setdefault("LRTT_SILENT", "1")

import argparse
import json
import math
import time

import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
SEED = 42
HIDDEN = 256
OUT = 10
TAU_SEC = 46505.0

RANK = 8
TE = 10
LIFETIME_PHYS = 1000

LR_GRID            = [1.0, 0.3, 0.1]                  # v1 best ≈ 0.187 (covered)
TLR_GRID           = [0.01, 0.001, 0.0001]            # v1 best tlr ≈ 8e-4 (covered)
CLASSIFIER_LR_GRID = [1.0, 0.3, 0.1]

AF_GRID  = [0.0, 1.0, 5.0]
UNR_GRID = [0.0, 1.0, 3.0]
SCREEN_AF  = [0.0, 1.0, 5.0]
SCREEN_UNR = [0.0, 1.0, 3.0]


def _ab_lifetime_param(lt):
    if lt is None or lt <= 0:
        return 0.0
    dt = -TAU_SEC * math.log(1 - 1.0/lt)
    return 1.0 / (1 - math.exp(-dt/TAU_SEC))


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
                   num_workers=max(0, nw-2),
                   pin_memory=(DEVICE.type == "cuda"))
    )


def _make_ab_device(lt_param, af_ratio, unr):
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


def _make_c_device():
    return LinearStepDevice(
        dw_min=2.0/1024.0, w_max=1.0, w_min=-1.0,
        gamma_up=0.0, gamma_down=0.0,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0,
    )


def create_model(tlr, af_ratio, unr):
    lt_param = _ab_lifetime_param(LIFETIME_PHYS)
    a = _make_ab_device(lt_param, af_ratio, unr)
    b = _make_ab_device(lt_param, af_ratio, unr)
    c = _make_c_device()
    dev = PythonLRTTDevice(
        rank=RANK, transfer_every=TE,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode="decay",                 # ← v1
        decay_factor=1.0,
        unit_cell_devices=[a, b, c],
    )
    dev.transfer_lr = tlr
    dev.forward_inject = False
    dev.update_mode = "lora"                 # ← v1
    dev.transfer_method = "onehot"           # ← v1
    dev.transfer_mode = "off"
    rpu = PythonLRTTRPUConfig(device=dev)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(HIDDEN, OUT, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)


def run_trial(lr, tlr, clr, af_ratio, unr, *, smoke=False):
    torch.manual_seed(SEED)
    train_loader, val_loader = build_loaders(smoke=smoke)
    model = create_model(tlr, af_ratio, unr)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    for pg in optimizer.param_groups:
        if not pg.get("analog", False):
            pg["lr"] = clr
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


def cache_key(lr, tlr, clr, af, unr):
    return f"lr{lr}_tlr{tlr}_clr{clr}_af{af}_unr{unr}"


def load_or_run(cache, lr, tlr, clr, af, unr):
    k = cache_key(lr, tlr, clr, af, unr)
    if k in cache:
        return cache[k]
    acc = run_trial(lr, tlr, clr, af, unr)
    cache[k] = acc
    return acc


def save_cache(cache, path):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("=== smoke ===")
        acc = run_trial(lr=0.3, tlr=0.001, clr=0.1, af_ratio=1.0, unr=1.0, smoke=True)
        print(f"  acc={acc:.2f}%")
        return

    output_dir = args.out or "/root/LRTT/results/hp_search_v1_decay_rank8_gamma_af_noise_10bitC"
    os.makedirs(output_dir, exist_ok=True)
    cache_path = f"{output_dir}/trial_cache.json"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"[resume] loaded {len(cache)} cached trials")

    cells = [(lr, tlr, clr)
             for lr in LR_GRID for tlr in TLR_GRID for clr in CLASSIFIER_LR_GRID]
    n_cells = len(cells)
    print("=" * 80)
    print(f"LRTT-v1 (decay/lora/onehot) γ-AF/UNR 2-stage sweep")
    print(f"Stage 1: {n_cells} HP cells × {len(SCREEN_AF)*len(SCREEN_UNR)} screening pts")
    print(f"Stage 2: best HP × {len(AF_GRID)*len(UNR_GRID)} full grid")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Stage 1
    print("\n=== STAGE 1 — HP SCREENING ===")
    t0 = time.time()
    cell_means = {}
    cell_records = {}
    for ci, (lr, tlr, clr) in enumerate(cells, 1):
        accs = []; records = []
        cell_t0 = time.time()
        for af in SCREEN_AF:
            for unr in SCREEN_UNR:
                acc = load_or_run(cache, lr, tlr, clr, af, unr)
                accs.append(acc)
                records.append({"af_ratio": af, "update_noise_ratio": unr,
                                "acc": round(acc, 2)})
        cell_means[(lr, tlr, clr)] = sum(accs) / len(accs)
        cell_records[(lr, tlr, clr)] = records
        save_cache(cache, cache_path)
        print(f"  [{ci}/{n_cells}] lr={lr:>4}, tlr={tlr:>9.5f}, clr={clr:>4}  "
              f"mean={cell_means[(lr, tlr, clr)]:.2f}%  "
              f"min={min(accs):.1f}, max={max(accs):.1f}  "
              f"({(time.time()-cell_t0)/60:.1f}min, "
              f"total {(time.time()-t0)/60:.1f}min)")

    stage1_elapsed = time.time() - t0
    sorted_cells = sorted(cell_means.items(), key=lambda kv: -kv[1])
    best_cell = sorted_cells[0][0]
    print(f"\nBest: lr={best_cell[0]}, tlr={best_cell[1]}, clr={best_cell[2]} "
          f"(mean={cell_means[best_cell]:.2f}%)")
    print("Top 5:")
    for i, ((lr, tlr, clr), m) in enumerate(sorted_cells[:5], 1):
        print(f"  {i}. lr={lr:>4}, tlr={tlr:>9.5f}, clr={clr:>4}  mean={m:.2f}%")

    with open(f"{output_dir}/stage1_summary.json", "w") as f:
        json.dump({
            "stage": 1,
            "cells_evaluated": [
                {"lr": lr_, "tlr": tlr_, "classifier_lr": clr_,
                 "mean_acc": round(m, 2),
                 "screening_records": cell_records[(lr_, tlr_, clr_)]}
                for (lr_, tlr_, clr_), m in cell_means.items()
            ],
            "wall_seconds": round(stage1_elapsed, 1),
        }, f, indent=2)

    # Stage 2
    lr, tlr, clr = best_cell
    print(f"\n=== STAGE 2 — FULL GRID at lr={lr}, tlr={tlr}, clr={clr} ===")
    t1 = time.time()
    full = []
    new_count = 0
    for af in AF_GRID:
        for unr in UNR_GRID:
            k = cache_key(lr, tlr, clr, af, unr)
            cached_now = k in cache
            acc = load_or_run(cache, lr, tlr, clr, af, unr)
            full.append({"af_ratio": af, "update_noise_ratio": unr,
                         "acc": round(acc, 2), "from_cache": cached_now})
            if not cached_now:
                new_count += 1
                save_cache(cache, cache_path)

    stage2_elapsed = time.time() - t1
    total = time.time() - t0
    best_full = max(full, key=lambda r: r["acc"])

    final = {
        "method": "lrtt_v1_decay_lora_onehot_gamma_af_2stage",
        "rank": RANK, "te": TE, "lifetime_phys": LIFETIME_PHYS,
        "best_lr": lr, "best_tlr": tlr, "best_classifier_lr": clr,
        "stage1_mean_at_best": round(cell_means[best_cell], 2),
        "best_acc": best_full["acc"],
        "best_af_ratio": best_full["af_ratio"],
        "best_update_noise_ratio": best_full["update_noise_ratio"],
        "n_trials_stage1": n_cells * len(SCREEN_AF) * len(SCREEN_UNR),
        "n_trials_stage2_new": new_count,
        "wall_seconds_stage1": round(stage1_elapsed, 1),
        "wall_seconds_stage2": round(stage2_elapsed, 1),
        "wall_seconds_total": round(total, 1),
        "all_trials": full,
        "cell_means_stage1": [
            {"lr": lr_, "tlr": tlr_, "classifier_lr": clr_, "mean_acc": round(m, 2)}
            for (lr_, tlr_, clr_), m in sorted_cells
        ],
    }
    with open(f"{output_dir}/results_final.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nDone. Total {total/60:.1f} min")
    print(f"Best @ AF={best_full['af_ratio']}, UNR={best_full['update_noise_ratio']}: "
          f"{best_full['acc']:.2f}%")

    grid = {(r["af_ratio"], r["update_noise_ratio"]): r["acc"] for r in full}
    print("\nFinal 5×5 grid (val acc %):")
    print("  AF \\ UNR " + "".join(f" {n:>6}" for n in UNR_GRID))
    for af in AF_GRID:
        row = f"  af={af:>4} "
        for unr in UNR_GRID:
            row += f" {grid[(af, unr)]:6.2f}"
        print(row)


if __name__ == "__main__":
    main()

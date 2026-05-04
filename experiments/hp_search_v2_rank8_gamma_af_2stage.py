#!/usr/bin/env python3
"""LRTT-v2 γ-AF / update-noise 2-stage sweep, rank=8 (Option C: 32 HP cells).

Stage 1 — HP screening:
  32 (lr, tlr) cells × 9 sparse (AF, UNR) screening points = 288 trials
  Pick best (lr, tlr) cell by mean acc across the 9 screening points.

Stage 2 — full sweep at best HP:
  Best (lr, tlr) cell × full 5×5 (AF × UNR) grid = 25 points
  (9 already cached from stage 1 → 16 new trials)

HP search grid (Option C):
  LR_GRID  = [3.0, 1.0, 0.3, 0.1]                              # 4 values
  TLR_GRID = [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01]              # 8 values, log-spaced
  LIFETIME = 1000  (fixed)

(AF, UNR) grids (kept identical to prior sweeps):
  AF_GRID  = [0.0, 1.0, 5.0]                        # γ_up = γ_down magnitude
  UNR_GRID = [0.0, 1.0, 3.0]                          # scales dw_min_std

Sparse screening points (Stage 1):
  SCREEN_AF  = [0.0, 1.0, 5.0]    (small / nominal-ish / large)
  SCREEN_UNR = [0.0, 1.0, 3.0]    (none / nominal / stress)
  → 3 × 3 = 9 points

6T1C device: up_down=0 fixed; gamma_up = gamma_down = af_ratio (swept).
C-tile: 10-bit idealized LinearStepDevice.

Total: 288 (Stage 1) + 16 (Stage 2 new) = 304 trials, ~4–5 h.

Usage:
  python hp_search_v2_rank8_gamma_af_2stage.py
  python hp_search_v2_rank8_gamma_af_2stage.py --smoke
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

# 3-axis HP grid (analog_lr × tlr × classifier_lr) = 27 cells
LR_GRID            = [3.0, 1.0, 0.3]                  # analog tile lr
TLR_GRID           = [10.0, 1.0, 0.1]                 # transfer lr
CLASSIFIER_LR_GRID = [1.0, 0.3, 0.1]                  # FP classifier lr (decoupled)

# Full (AF, UNR) grid
AF_GRID  = [0.0, 1.0, 5.0]
UNR_GRID = [0.0, 1.0, 3.0]

# Stage 1 sparse screening
SCREEN_AF  = [0.0, 1.0, 5.0]
SCREEN_UNR = [0.0, 1.0, 3.0]


def _ab_lifetime_param(lifetime_phys):
    if lifetime_phys is None or lifetime_phys <= 0:
        return 0.0
    dt_batch_sec = -TAU_SEC * math.log(1 - 1.0 / lifetime_phys)
    return 1.0 / (1 - math.exp(-dt_batch_sec / TAU_SEC))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_TRAIN_DS = None
_VAL_DS = None


def build_loaders(smoke=False, smoke_train_n=1024, smoke_val_n=1024,
                  smoke_batch=128):
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
        train_ds = Subset(_TRAIN_DS, range(min(smoke_train_n, len(_TRAIN_DS))))
        val_ds = Subset(_VAL_DS, range(min(smoke_val_n, len(_VAL_DS))))
        bs, nw = smoke_batch, 0
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


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------
def _make_ab_device(ab_lifetime_param, af_ratio, update_noise_ratio):
    return LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,                                       # FIXED at 0
        w_max=1.0, w_min=-1.0,
        gamma_up=af_ratio,                                 # ← swept (γ-AF response slope)
        gamma_down=af_ratio,                               # ← swept (same value)
        mult_noise=False,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3 * update_noise_ratio,               # ← noise sweep
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime_param,
        lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


def _make_c_device():
    return LinearStepDevice(
        dw_min=2.0/1024.0, w_max=1.0, w_min=-1.0,          # 10-bit
        gamma_up=0.0, gamma_down=0.0,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0,
    )


def create_model(rank, te, tlr, lifetime_phys, af_ratio, update_noise_ratio):
    ab_lifetime_param = _ab_lifetime_param(lifetime_phys)
    a = _make_ab_device(ab_lifetime_param, af_ratio, update_noise_ratio)
    b = _make_ab_device(ab_lifetime_param, af_ratio, update_noise_ratio)
    c = _make_c_device()
    dev = PythonLRTTDevice(
        rank=rank, transfer_every=te,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode="standard", decay_factor=1.0,
        b_init_mode="zero",
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        forward_inject=False,
        selector_axis="row", selector_policy="shuffled_cycle",
        selector_seed=SEED, selector_reset_b_on_advance=True,
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

    model = AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(HIDDEN, OUT, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    return model


def run_trial(lr, tlr, classifier_lr, af_ratio, update_noise_ratio, *, smoke=False):
    torch.manual_seed(SEED)
    train_loader, val_loader = build_loaders(smoke=smoke)
    model = create_model(RANK, TE, tlr, LIFETIME_PHYS, af_ratio,
                         update_noise_ratio)
    # Decouple analog tile lr from FP classifier lr
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    # Set classifier (digital/FP) param groups to classifier_lr
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
        for data, target in train_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
                target = target.to(DEVICE, non_blocking=True)
                correct += model(data).argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
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


# ---------------------------------------------------------------------------
# Cache helper
# ---------------------------------------------------------------------------
def cache_key(lr, tlr, clr, af_ratio, unr):
    return f"lr{lr}_tlr{tlr}_clr{clr}_af{af_ratio}_unr{unr}"


def load_or_run(cache, lr, tlr, clr, af_ratio, unr):
    k = cache_key(lr, tlr, clr, af_ratio, unr)
    if k in cache:
        return cache[k]
    acc = run_trial(lr, tlr, clr, af_ratio, unr)
    cache[k] = acc
    return acc


def save_cache(cache, path):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("=== smoke ===")
        acc = run_trial(lr=1.0, tlr=10.0, classifier_lr=0.1,
                        af_ratio=1.0, update_noise_ratio=1.0, smoke=True)
        print(f"  acc={acc:.2f}%")
        return

    output_dir = args.out or "/root/LRTT/results/hp_search_v2_rank8_gamma_af_noise_10bitC"
    os.makedirs(output_dir, exist_ok=True)

    cache_path = f"{output_dir}/trial_cache.json"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"[resume] loaded {len(cache)} cached trials")

    n_cells_total = len(LR_GRID) * len(TLR_GRID) * len(CLASSIFIER_LR_GRID)
    print("=" * 80)
    print("LRTT-v2 γ-AF / UNR  2-stage sweep")
    print(f"Stage 1: HP screening  {n_cells_total} cells × "
          f"{len(SCREEN_AF)*len(SCREEN_UNR)} pts = "
          f"{n_cells_total*len(SCREEN_AF)*len(SCREEN_UNR)} trials")
    print(f"Stage 2: full grid     1 cell × {len(AF_GRID)*len(UNR_GRID)} pts")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # ---------------- Stage 1: HP screening ----------------
    print("\n" + "=" * 80)
    print("STAGE 1 — HP SCREENING (3 axes: analog_lr × tlr × classifier_lr)")
    print("=" * 80)
    t0 = time.time()
    cell_means = {}      # (lr, tlr, clr) → mean acc across 9 screening points
    cell_records = {}    # (lr, tlr, clr) → list of {af, unr, acc}

    cells = [(lr, tlr, clr)
             for lr in LR_GRID for tlr in TLR_GRID for clr in CLASSIFIER_LR_GRID]
    n_cells = len(cells)

    for ci, (lr, tlr, clr) in enumerate(cells, 1):
        accs = []
        records = []
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
        cell_elapsed = (time.time() - cell_t0) / 60
        total_elapsed = (time.time() - t0) / 60
        print(f"  [{ci}/{n_cells}] lr={lr:>4}, tlr={tlr:>5}, clr={clr:>4} → "
              f"mean={cell_means[(lr, tlr, clr)]:.2f}%  "
              f"min={min(accs):.1f}, max={max(accs):.1f}  "
              f"({cell_elapsed:.1f}min cell, {total_elapsed:.1f}min total)")

    stage1_elapsed = time.time() - t0

    # Save stage 1 summary
    stage1_summary = {
        "stage": 1,
        "cells_evaluated": [
            {"lr": lr, "tlr": tlr, "classifier_lr": clr,
             "mean_acc": round(m, 2),
             "screening_records": cell_records[(lr, tlr, clr)]}
            for (lr, tlr, clr), m in cell_means.items()
        ],
        "wall_seconds": round(stage1_elapsed, 1),
    }
    with open(f"{output_dir}/stage1_summary.json", "w") as f:
        json.dump(stage1_summary, f, indent=2)

    # Pick best HP
    best_cell = max(cell_means, key=cell_means.get)
    print(f"\nBest cell from Stage 1: lr={best_cell[0]}, tlr={best_cell[1]}  "
          f"(mean={cell_means[best_cell]:.2f}%)")

    # Print top 5
    sorted_cells = sorted(cell_means.items(), key=lambda kv: -kv[1])
    print("\nTop 5 cells (by mean acc across 9 screening points):")
    for i, ((lr, tlr, clr), m) in enumerate(sorted_cells[:5], 1):
        print(f"  {i}. lr={lr:>4}, tlr={tlr:>5}, clr={clr:>4}  → mean={m:.2f}%")

    # ---------------- Stage 2: full sweep at best HP ----------------
    lr, tlr, clr = best_cell
    print("\n" + "=" * 80)
    print(f"STAGE 2 — FULL SWEEP at best HP (lr={lr}, tlr={tlr}, clr={clr})")
    print("=" * 80)
    t1 = time.time()

    full_records = []
    new_count = 0
    for af in AF_GRID:
        for unr in UNR_GRID:
            k = cache_key(lr, tlr, clr, af, unr)
            cached_now = k in cache
            acc = load_or_run(cache, lr, tlr, clr, af, unr)
            full_records.append({"af_ratio": af, "update_noise_ratio": unr,
                                  "acc": round(acc, 2),
                                  "from_cache": cached_now})
            if not cached_now:
                new_count += 1
                save_cache(cache, cache_path)

    stage2_elapsed = time.time() - t1
    total_elapsed = time.time() - t0

    # Build final result
    best_full = max(full_records, key=lambda r: r["acc"])
    final = {
        "method": "lrtt_v2_selector_blockwise_gamma_af_2stage",
        "rank": RANK, "te": TE, "lifetime_phys": LIFETIME_PHYS,
        "best_lr": lr,
        "best_tlr": tlr,
        "best_classifier_lr": clr,
        "stage1_mean_at_best": round(cell_means[best_cell], 2),
        "best_acc": best_full["acc"],
        "best_af_ratio": best_full["af_ratio"],
        "best_update_noise_ratio": best_full["update_noise_ratio"],
        "n_trials_stage1": n_cells * len(SCREEN_AF) * len(SCREEN_UNR),
        "n_trials_stage2_new": new_count,
        "wall_seconds_stage1": round(stage1_elapsed, 1),
        "wall_seconds_stage2": round(stage2_elapsed, 1),
        "wall_seconds_total": round(total_elapsed, 1),
        "all_trials": full_records,
        "cell_means_stage1": [
            {"lr": lr_, "tlr": tlr_, "classifier_lr": clr_, "mean_acc": round(m, 2)}
            for (lr_, tlr_, clr_), m in sorted_cells
        ],
    }

    with open(f"{output_dir}/results_final.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nDone. Total: {total_elapsed/60:.1f} min "
          f"(stage1 {stage1_elapsed/60:.1f}, stage2 {stage2_elapsed/60:.1f})")
    print(f"Best @ AF={best_full['af_ratio']}, UNR={best_full['update_noise_ratio']}: "
          f"{best_full['acc']:.2f}%")

    # Print final 5×5 grid
    print("\n" + "=" * 80)
    print(f"FINAL GRID @ best HP (lr={lr}, tlr={tlr}, clr={clr})")
    print("=" * 80)
    grid = {(r["af_ratio"], r["update_noise_ratio"]): r["acc"]
            for r in full_records}
    print("  AF \\ UNR " + "".join(f" {n:>6}" for n in UNR_GRID))
    for af in AF_GRID:
        row = f"  af={af:>4} "
        for unr in UNR_GRID:
            row += f" {grid[(af, unr)]:6.2f}"
        print(row)


if __name__ == "__main__":
    main()

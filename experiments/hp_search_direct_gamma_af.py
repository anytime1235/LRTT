#!/usr/bin/env python3
"""Direct (SingleRPU + 6T1C) γ-AF / UNR sweep — fixed HP.

HP fixed: lr=0.1.
γ-AF sweep: gamma_up=gamma_down=af_ratio (up_down=0).
Total 5×5 = 25 trials, ~30 min.
"""
from __future__ import annotations

import os
os.environ.setdefault("LRTT_SILENT", "1")

import argparse
import json
import math
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice

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

LR = 0.1
LIFETIME_PHYS = 1000
DESIRED_BL = 31

AF_GRID  = [0.0, 1.0, 5.0]
UNR_GRID = [0.0, 1.0, 3.0]


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


def _make_6t1c_device(lt_param, af_ratio, unr):
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


def create_rpu(af_ratio, unr):
    lt_param = _ab_lifetime_param(LIFETIME_PHYS)
    dev = _make_6t1c_device(lt_param, af_ratio, unr)
    rpu = SingleRPUConfig(device=dev)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.update.desired_bl = DESIRED_BL
    rpu.mapping.weight_scaling_omega = 0.6
    return rpu


def run_trial(af_ratio, unr, *, smoke=False):
    torch.manual_seed(SEED)
    train_loader, val_loader = build_loaders(smoke=smoke)
    rpu = create_rpu(af_ratio, unr)
    model = AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(HIDDEN, OUT, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("=== smoke ===")
        acc = run_trial(af_ratio=1.0, unr=1.0, smoke=True)
        print(f"  acc={acc:.2f}%")
        return

    output_dir = args.out or "/root/LRTT/results/hp_search_direct_rank8_gamma_af_noise_10bitC"
    os.makedirs(output_dir, exist_ok=True)
    cache_path = f"{output_dir}/trial_cache.json"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"[resume] {len(cache)} cached")

    print(f"Direct γ-AF/UNR sweep (fixed lr={LR}, lifetime={LIFETIME_PHYS})")
    print(f"Grid: AF ∈ {AF_GRID} × UNR ∈ {UNR_GRID}")
    print(f"Output: {output_dir}")

    t0 = time.time()
    full = []
    for af in AF_GRID:
        for unr in UNR_GRID:
            k = f"af{af}_unr{unr}"
            if k in cache:
                acc = cache[k]
                cn = True
            else:
                acc = run_trial(af, unr)
                cache[k] = acc
                cn = False
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2)
            full.append({"af_ratio": af, "update_noise_ratio": unr,
                         "acc": round(acc, 2), "from_cache": cn})
            print(f"  AF={af:>4}, UNR={unr:>3} → {acc:.2f}%  "
                  f"(elapsed {(time.time()-t0)/60:.1f}min)")

    elapsed = time.time() - t0
    best = max(full, key=lambda r: r["acc"])
    final = {
        "method": "direct_singleRPU_6t1c_gamma_af",
        "rank": None, "lr": LR, "lifetime_phys": LIFETIME_PHYS,
        "best_acc": best["acc"],
        "best_af_ratio": best["af_ratio"],
        "best_update_noise_ratio": best["update_noise_ratio"],
        "n_trials": len(full),
        "wall_seconds": round(elapsed, 1),
        "all_trials": full,
    }
    with open(f"{output_dir}/results_final.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nDone. {elapsed/60:.1f} min")
    print(f"Best @ AF={best['af_ratio']}, UNR={best['update_noise_ratio']}: {best['acc']}%")

    grid = {(r["af_ratio"], r["update_noise_ratio"]): r["acc"] for r in full}
    print("  AF \\ UNR " + "".join(f" {n:>6}" for n in UNR_GRID))
    for af in AF_GRID:
        row = f"  af={af:>4} "
        for unr in UNR_GRID:
            row += f" {grid[(af, unr)]:6.2f}"
        print(row)


if __name__ == "__main__":
    main()

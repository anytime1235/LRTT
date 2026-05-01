#!/usr/bin/env python3
"""TikiTaka v1 (ChoppedTransferCompound, no_buffer=True) AF/noise sweep, rank=8.

Adapted from `transformer:main_results/scripts/analysis/optuna_bert_squad_tiki.py
:: create_tikitaka_config(... use_v2=False)`. The same MNIST MLP environment
as the v1/v2 LRTT sweeps, so the heatmap can be overlaid.

Architecture (2-tile, NOT 3-tile like LRTT):
  - Fast tile (A) = 6T1C LinearStepDevice  → analog hardware (sweep target)
  - Slow tile (B) = SoftBoundsDevice noise=0  → idealized (no sweep)

Hyperparameters are FIXED per the user's spec — no HP search:
  - lr        = 0.1
  - transfer_lr (t_lr) = 1.0
  - fast_lr   = 1.0
  - transfer_every (te) = 10  (matches LRTT for fairness)
  - lifetime_phys = 1000      (per user request — overrides optuna_bert_squad_tiki default)
  - gamma     = 0.0           (slow tile only visible)
  - desired_bl = 31           (transfer_update default)
  - units_in_mbatch=True, n_reads_per_transfer=1, transfer_columns=True
  - scale_transfer_lr=False   (v1: use_v2=False)
  - in_chop_prob=0.0, out_chop_prob=0.0  (v1: no chopping)
  - no_buffer=True            (v1)
  - auto_scale=False
  - mapping.weight_scaling_omega=0.6 (matches LRTT v1/v2 sweep convention)

Sweep variables (applied to FAST tile only):
  ab_up_down ∈ {0.5, 1.0, 2.0, 5.0, 10.0}
  ab_noise_ratio ∈ {0.5, 1.0, 2.0, 5.0, 10.0}
(JSON keys reuse `ab_up_down` / `ab_noise_ratio` for plotting compatibility.
Semantically: tikitaka has only ONE physical tile (Fast), not A+B both.)

Total: 1 cell × 25 trials = 25 trials (~30–40 min on 1 GPU).
Output: /root/LRTT/results/hp_search_tikitaka_v1_rank8_b_af_noise/

Usage:
  python hp_search_tikitaka_v1_rank8_b_af_noise.py
  python hp_search_tikitaka_v1_rank8_b_af_noise.py --smoke
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
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig, UnitCellRPUConfig,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.parameters.enums import NoiseManagementType, BoundManagementType
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.training import UpdateParameters


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

# Fixed hyperparameters (no HP sweep, per user)
LR = 0.1
TRANSFER_LR = 1.0
FAST_LR = 1.0
TE = 10
LIFETIME_PHYS = 1000      # fixed at 1000 per user request
DESIRED_BL = 31
GAMMA = 0.0

# Sweep grid (applied to FAST tile only)
AB_UP_DOWN_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
AB_NOISE_RATIO_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
N_TRIALS = len(AB_UP_DOWN_GRID) * len(AB_NOISE_RATIO_GRID)


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


def build_loaders(smoke: bool = False, smoke_train_n: int = 1024,
                  smoke_val_n: int = 1024, smoke_batch: int = 128):
    global _TRAIN_DS, _VAL_DS
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if _TRAIN_DS is None:
        _TRAIN_DS = datasets.MNIST(
            "/tmp/mnist", download=True, train=True, transform=transform
        )
        _VAL_DS = datasets.MNIST(
            "/tmp/mnist", download=True, train=False, transform=transform
        )
    if smoke:
        train_ds = Subset(_TRAIN_DS, range(min(smoke_train_n, len(_TRAIN_DS))))
        val_ds = Subset(_VAL_DS, range(min(smoke_val_n, len(_VAL_DS))))
        bs = smoke_batch; nw = 0
    else:
        train_ds = _TRAIN_DS; val_ds = _VAL_DS
        bs = BATCH_SIZE; nw = 4
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=max(0, nw - 2), pin_memory=(DEVICE.type == "cuda"),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Devices — Fast (A) is the swept 6T1C; Slow (B) is the idealized SoftBounds
# ---------------------------------------------------------------------------
def _make_fast_device(lifetime_param, ab_up_down, ab_noise_ratio):
    """Fast tile: 6T1C LinearStepDevice. up_down and d2d/c2c noise are swept,
    matching the AB-sweep convention used in the LRTT v1/v2 scripts."""
    return LinearStepDevice(
        dw_min=0.001981,
        up_down=ab_up_down,
        w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1 * ab_noise_ratio,
        up_down_dtod=0.01,
        w_max_dtod=0.05 * ab_noise_ratio,
        w_min_dtod=0.05 * ab_noise_ratio,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3 * ab_noise_ratio,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime_param,
        lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


def _make_slow_device():
    """Slow tile: noise-free SoftBoundsDevice (matches optuna_bert_squad_tiki)."""
    return SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=False,
    )


def create_tikitaka_v1_rpu(ab_up_down, ab_noise_ratio):
    """TikiTaka v1: ChoppedTransferCompound with use_v2=False settings.
    Mirrors `optuna_bert_squad_tiki.py:create_tikitaka_config(use_v2=False)`."""
    lifetime_param = _ab_lifetime_param(LIFETIME_PHYS)
    fast = _make_fast_device(lifetime_param, ab_up_down, ab_noise_ratio)
    slow = _make_slow_device()

    rpu = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[fast, slow],
            transfer_every=TE,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=GAMMA,
            transfer_lr=TRANSFER_LR,
            fast_lr=FAST_LR,
            scale_transfer_lr=False,                # v1
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=DESIRED_BL,
                update_bl_management=True,          # v1
                update_management=True,             # v1
            ),
            no_buffer=True,                          # v1
            in_chop_prob=0.0,                        # v1
            out_chop_prob=0.0,                       # v1
            auto_scale=False,
            auto_momentum=0.99,
        )
    )

    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6   # match LRTT v1/v2 sweep convention
    return rpu


def create_model(ab_up_down, ab_noise_ratio):
    rpu_config = create_tikitaka_v1_rpu(ab_up_down, ab_noise_ratio)
    model = AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(HIDDEN, OUT, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    return model


def run_trial(ab_up_down, ab_noise_ratio,
              *, epochs=EPOCHS, smoke=False, trial=None):
    torch.manual_seed(SEED)
    train_loader, val_loader = build_loaders(smoke=smoke)
    model = create_model(ab_up_down, ab_noise_ratio)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience = 0
    history = []
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
        history.append((epoch, acc))
        if acc > best_acc:
            best_acc = acc; patience = 0
        else:
            patience += 1
        if trial is not None and not smoke:
            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not smoke and epoch >= 5 and best_acc < 50.0:
            break
        if not smoke and patience >= EARLY_STOP_PATIENCE:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc, history


# ---------------------------------------------------------------------------
# Sweep driver — single cell, 5×5 grid
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("=== smoke trial ===")
        acc, hist = run_trial(ab_up_down=2.0, ab_noise_ratio=1.0,
                              epochs=2, smoke=True)
        print(f"  acc={acc:.2f}%  hist={hist}")
        return

    output_dir = args.out or "/root/LRTT/results/hp_search_tikitaka_v1_rank8_b_af_noise"
    os.makedirs(output_dir, exist_ok=True)
    cell_path = f"{output_dir}/cell_lr{LR}_tlr{TRANSFER_LR}_fastlr{FAST_LR}_te{TE}.json"

    print("=" * 74)
    print("TikiTaka v1 (ChoppedTransferCompound, no_buffer=True)  AF/noise sweep")
    print(f"Fixed: lr={LR}  transfer_lr={TRANSFER_LR}  fast_lr={FAST_LR}  "
          f"te={TE}  lifetime_phys={LIFETIME_PHYS}")
    print(f"Sweep: ab_up_down ∈ {AB_UP_DOWN_GRID} × "
          f"ab_noise_ratio ∈ {AB_NOISE_RATIO_GRID}  (applied to FAST tile)")
    print(f"Total: {N_TRIALS} trials (single cell)")
    print(f"Output: {cell_path}")
    print("=" * 74)

    if os.path.exists(cell_path):
        print(f"  -> already done, exiting ({cell_path})")
        return

    search_space = {
        "ab_up_down": AB_UP_DOWN_GRID,
        "ab_noise_ratio": AB_NOISE_RATIO_GRID,
    }
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space, seed=SEED),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3, max_resource=EPOCHS, reduction_factor=3,
        ),
    )

    def objective(trial):
        ab_up_down = trial.suggest_categorical("ab_up_down", AB_UP_DOWN_GRID)
        ab_noise_ratio = trial.suggest_categorical(
            "ab_noise_ratio", AB_NOISE_RATIO_GRID
        )
        acc, _ = run_trial(ab_up_down=ab_up_down, ab_noise_ratio=ab_noise_ratio,
                           trial=trial)
        return acc

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\nBest: {best.value:.2f}%  "
          f"ab_up_down={best.params['ab_up_down']:.4f}  "
          f"ab_noise_ratio={best.params['ab_noise_ratio']:.4f}  "
          f"({elapsed/60:.1f} min)")

    record = {
        "method": "tikitaka_v1_chopped_no_buffer",
        "rank": None,                   # tikitaka has no rank
        "te": TE, "lr": LR,
        "transfer_lr": TRANSFER_LR, "fast_lr": FAST_LR,
        "lifetime_phys": LIFETIME_PHYS,
        "best_acc": round(best.value, 2),
        "best_ab_up_down": best.params["ab_up_down"],
        "best_ab_noise_ratio": best.params["ab_noise_ratio"],
        "n_trials": N_TRIALS,
        "wall_seconds": round(elapsed, 1),
        "all_trials": [],
    }
    for t in study.trials:
        if t.value is None:
            continue
        record["all_trials"].append({
            "ab_up_down": t.params["ab_up_down"],
            "ab_noise_ratio": t.params["ab_noise_ratio"],
            "acc": round(t.value, 2),
        })

    with open(cell_path, "w") as f:
        json.dump(record, f, indent=2)
    with open(f"{output_dir}/results_final.json", "w") as f:
        json.dump([record], f, indent=2)

    print("\n" + "=" * 74)
    print("Acc grid (ab_up_down rows × ab_noise_ratio cols):")
    print("  AF \\ noise " + "".join(f"  {n:>6}" for n in AB_NOISE_RATIO_GRID))
    grid = {(t["ab_up_down"], t["ab_noise_ratio"]): t["acc"]
            for t in record["all_trials"]}
    for af in AB_UP_DOWN_GRID:
        row = f"  af={af:>4} "
        for nr in AB_NOISE_RATIO_GRID:
            v = grid.get((af, nr))
            row += f"  {v:6.2f}" if v is not None else "       -"
        print(row)


if __name__ == "__main__":
    main()

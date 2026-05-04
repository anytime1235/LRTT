#!/usr/bin/env python3
"""Per-cell Optuna TPE search over (rank, C-tile) for the 6T1C-AB heatmap.

Fixed: A/B tiles use the canonical 6T1C device from
``aihwkit.simulator.configs.lrtt_python.PythonLRTTPreset.sixt1c_ab`` (Li VLSI
2018 calibrated LinearStepDevice with realistic write noise, mult_noise,
gamma asymmetry, dtod variation, and capacitor retention).

Swept:
  - rank ∈ {1, 2, 4, 8, 16}    (v1/v2 only — TikiTaka has no rank)
  - C tile ∈ {ideal, ecram, rram}
      ideal → IdealizedPresetDevice (Gokmen-Vlasov, ~10000 states)
      ecram → EcRamPresetDevice (Tang IEDM 2018, Li-ECRAM)
      rram  → ReRamESPresetDevice (Gong Nat. Commun. 2018, ExpStep)
  - method ∈ {lrtt_v1, lrtt_v2, tikitaka_v1}

For each (method, rank, C) combination we run an Optuna TPE search with 30
trials over the same log-uniform HP space used by per_cell_tpe_30.py. The
baseline-heatmap best HP is enqueued as the first trial (warm-start).
TikiTaka has no rank dimension, so it gets one TPE search per C-tile.

Total cells: 2 methods × 5 ranks × 3 C = 30 + TikiTaka 3 = 33.
Cost @ ~120 s/trial on clean GPU: 33 × 30 × 120 s ≈ 33 hours.

Output: results/sweep_rank_device/<method>/rank{R}_C{tile}.json
TikiTaka cells are written as: results/sweep_rank_device/tikitaka_v1/norank_C{tile}.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("LRTT_SILENT", "1")
sys.path.insert(0, "/root/LRTT/experiments")

import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    FloatingPointRPUConfig, UnitCellRPUConfig,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import (
    PythonLRTTDevice, PythonLRTTPreset,
)
from aihwkit.simulator.configs.utils import (
    BoundManagementType, IOParameters,
    NoiseManagementType, UpdateParameters,
)
from aihwkit.simulator.presets.devices import (
    EcRamPresetDevice, IdealizedPresetDevice, ReRamESPresetDevice,
)

# Reuse data-loader / training constants from the v1 module so this sweep
# is directly comparable to the heatmap baseline.
import hp_search_v1_decay_gamma_af_2stage as v1_mod

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
RANKS    = [1, 2, 4, 8, 16]
C_TILES  = ["ideal", "ecram", "rram"]
METHODS  = ["lrtt_v1", "lrtt_v2", "tikitaka_v1"]

TE                  = 10
N_TRIALS            = 30
SEED                = 42
EPOCHS              = v1_mod.EPOCHS
EARLY_STOP_PATIENCE = v1_mod.EARLY_STOP_PATIENCE
HIDDEN              = v1_mod.HIDDEN
OUT                 = v1_mod.OUT
DEVICE              = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HP warm-start (heatmap baseline best). Used as the first enqueued trial of
# every cell, regardless of (rank, C). Per-cell TPE is expected to refine.
WARM_START_HP = {
    "lrtt_v1":     {"lr": 0.1, "tlr": 0.001, "clr": 0.3},
    "lrtt_v2":     {"lr": 1.0, "tlr": 0.1,   "clr": 1.0},
    "tikitaka_v1": {"transfer_lr": 1.0, "fast_lr": 0.3, "classifier_lr": 1.0},
}

# Search space mirrors per_cell_tpe_30.py for direct comparability.
SEARCH_SPACE = {
    "lrtt_v1": {
        "lr":  FloatDistribution(0.01, 1.0,  log=True),
        "tlr": FloatDistribution(1e-4, 1e-2, log=True),
        "clr": FloatDistribution(0.03, 3.0,  log=True),
    },
    "lrtt_v2": {
        "lr":  FloatDistribution(0.1, 10.0, log=True),
        "tlr": FloatDistribution(0.01, 1.0, log=True),
        "clr": FloatDistribution(0.1, 10.0, log=True),
    },
    "tikitaka_v1": {
        "transfer_lr":   FloatDistribution(0.1, 10.0, log=True),
        "fast_lr":       FloatDistribution(0.03, 3.0,  log=True),
        "classifier_lr": FloatDistribution(0.1, 10.0, log=True),
    },
}

OUT_ROOT = "/root/LRTT/results/sweep_rank_device"


# ---------------------------------------------------------------------------
# Device factories
# ---------------------------------------------------------------------------
def make_c_device(name: str):
    if name == "ideal":
        return IdealizedPresetDevice()
    if name == "ecram":
        return EcRamPresetDevice()
    if name == "rram":
        return ReRamESPresetDevice()
    raise ValueError(f"unknown C-tile: {name}")


def make_6t1c_device():
    """Manual factory matching PythonLRTTPreset.sixt1c_ab's AB tile, used for
    TikiTaka's fast tile (which doesn't go through the LRTT preset path)."""
    TAU_SEC = 46505.0
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    lifetime = 1.0 / delta
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0,
        w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0182,
        mean_bound_reference=True,
        lifetime=lifetime, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


# ---------------------------------------------------------------------------
# RPU builders
# ---------------------------------------------------------------------------
def build_v1_rpu(rank: int, c_name: str, tlr: float):
    c_dev = make_c_device(c_name)
    dev = PythonLRTTPreset.sixt1c_ab(
        rank=rank, transfer_every=TE, lora_alpha=1.0,
        c_device=c_dev,
        reinit_mode="decay", decay_factor=1.0,
    )
    dev.update_mode = "lora"
    dev.transfer_method = "onehot"
    dev.transfer_mode = "off"
    dev.transfer_lr = tlr
    dev.forward_inject = False
    rpu = PythonLRTTRPUConfig(device=dev)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return rpu


def build_v2_rpu(rank: int, c_name: str, tlr: float):
    c_dev = make_c_device(c_name)
    dev = PythonLRTTPreset.sixt1c_ab(
        rank=rank, transfer_every=TE, lora_alpha=1.0,
        c_device=c_dev,
        reinit_mode="standard", decay_factor=1.0,
    )
    dev.update_mode = "selector_reconstruction"
    dev.transfer_method = "blockwise"
    dev.b_init_mode = "zero"
    dev.transfer_lr = tlr
    dev.transfer_mode = "off"
    dev.forward_inject = False
    dev.selector_axis = "row"
    dev.selector_policy = "shuffled_cycle"
    dev.selector_seed = SEED
    dev.selector_reset_b_on_advance = True
    dev.cap_stabilizer_enabled = True
    dev.cap_rho = 1.0
    dev.cap_compensate_transfer = True
    rpu = PythonLRTTRPUConfig(device=dev)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return rpu


def build_tikitaka_rpu(c_name: str, transfer_lr: float, fast_lr: float):
    fast = make_6t1c_device()
    slow = make_c_device(c_name)
    rpu = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[fast, slow],
            transfer_every=TE,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=False,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=31,
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


# ---------------------------------------------------------------------------
# Single training trial
# ---------------------------------------------------------------------------
def run_one(method: str, rank, c_name: str, hp: dict) -> float:
    if method == "lrtt_v1":
        rpu = build_v1_rpu(rank, c_name, hp["tlr"])
        analog_lr, classifier_lr = hp["lr"], hp["clr"]
    elif method == "lrtt_v2":
        rpu = build_v2_rpu(rank, c_name, hp["tlr"])
        analog_lr, classifier_lr = hp["lr"], hp["clr"]
    elif method == "tikitaka_v1":
        rpu = build_tikitaka_rpu(c_name, hp["transfer_lr"], hp["fast_lr"])
        analog_lr, classifier_lr = 0.1, hp["classifier_lr"]
    else:
        raise ValueError(method)

    torch.manual_seed(SEED)
    train_loader, val_loader = v1_mod.build_loaders(smoke=False)
    model = AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(HIDDEN, OUT, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    optimizer = AnalogSGD(model.parameters(), lr=analog_lr)
    optimizer.regroup_param_groups(model)
    for pg in optimizer.param_groups:
        if not pg.get("analog", False):
            pg["lr"] = classifier_lr
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience = 0
    for epoch in range(1, EPOCHS + 1):
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
        if patience >= EARLY_STOP_PATIENCE:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc


# ---------------------------------------------------------------------------
# Per-cell TPE state
# ---------------------------------------------------------------------------
def cell_path(method: str, rank, c_name: str) -> str:
    if rank is None:
        return f"{OUT_ROOT}/{method}/norank_C{c_name}.json"
    return f"{OUT_ROOT}/{method}/rank{rank}_C{c_name}.json"


def load_cell(method: str, rank, c_name: str) -> dict:
    p = cell_path(method, rank, c_name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    distributions = SEARCH_SPACE[method]
    return {
        "method": method,
        "rank": rank,
        "c_tile": c_name,
        "warm_start": None,
        "search_space": {k: {"low": d.low, "high": d.high, "log": d.log}
                          for k, d in distributions.items()},
        "trials": [],
    }


def save_cell(state: dict, method: str, rank, c_name: str) -> None:
    p = cell_path(method, rank, c_name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, p)


def search_cell(method: str, rank, c_name: str, n_trials: int = N_TRIALS) -> dict:
    state = load_cell(method, rank, c_name)
    distributions = SEARCH_SPACE[method]
    n_done = len(state["trials"])
    if n_done >= n_trials:
        print(f"  [{method}] rank={rank}, C={c_name} — already {n_done}/{n_trials}, skip",
              flush=True)
        return state

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    if state["warm_start"] is None:
        state["warm_start"] = {"hp": dict(WARM_START_HP[method])}
        save_cell(state, method, rank, c_name)
    ws = state["warm_start"]
    if n_done == 0:
        try:
            study.enqueue_trial(ws["hp"])
        except Exception as e:
            print(f"  [warn] warm-start enqueue failed: {e}", flush=True)

    for t in state["trials"]:
        try:
            study.add_trial(create_trial(
                params=t["hp"], distributions=distributions, value=t["acc"],
            ))
        except Exception as e:
            print(f"  [warn] trial replay failed: {e}", flush=True)

    remaining = n_trials - n_done

    def objective(trial: optuna.Trial) -> float:
        hp = {
            k: trial.suggest_float(k, dist.low, dist.high, log=dist.log)
            for k, dist in distributions.items()
        }
        t0 = time.time()
        acc = run_one(method, rank, c_name, hp)
        wall = time.time() - t0
        rec = {"hp": hp, "acc": round(float(acc), 4),
               "wall_seconds": round(wall, 1)}
        state["trials"].append(rec)
        save_cell(state, method, rank, c_name)
        n = len(state["trials"])
        print(f"    trial {n}/{n_trials}: hp={hp} → acc={acc:.2f}  ({wall:.0f}s)",
              flush=True)
        return acc

    study.optimize(objective, n_trials=remaining,
                   gc_after_trial=True, show_progress_bar=False)

    best_acc = -float("inf")
    best_hp = ws["hp"]
    for t in state["trials"]:
        if t["acc"] > best_acc:
            best_acc = t["acc"]
            best_hp = t["hp"]
    state["best_acc"] = round(float(best_acc), 4)
    state["best_hp"] = best_hp
    state["n_completed"] = len(state["trials"])
    save_cell(state, method, rank, c_name)
    return state


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=METHODS,
                    choices=METHODS)
    p.add_argument("--ranks",   nargs="+", type=int, default=RANKS)
    p.add_argument("--c_tiles", nargs="+", default=C_TILES,
                    choices=C_TILES)
    p.add_argument("--n_trials", type=int, default=N_TRIALS)
    p.add_argument("--single_combo", nargs=3, default=None,
                    metavar=("METHOD", "RANK", "C"),
                    help="Run a single (method, rank, C) cell. Use 'none' for "
                         "rank when method=tikitaka_v1.")
    return p.parse_args()


def cells_to_run(args):
    out = []
    for m in args.methods:
        if m == "tikitaka_v1":
            for c in args.c_tiles:
                out.append((m, None, c))
        else:
            for r in args.ranks:
                for c in args.c_tiles:
                    out.append((m, r, c))
    return out


def main():
    args = parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.time()
    print(f"Per-cell TPE search ({args.n_trials} trials/cell) → {OUT_ROOT}",
          flush=True)
    print(f"Methods: {args.methods}", flush=True)
    print(f"Ranks (v1/v2): {args.ranks}", flush=True)
    print(f"C-tiles: {args.c_tiles}", flush=True)

    if args.single_combo is not None:
        method, rank_str, c_name = args.single_combo
        rank = None if rank_str.lower() == "none" else int(rank_str)
        cells = [(method, rank, c_name)]
    else:
        cells = cells_to_run(args)

    print(f"\nTotal cells to process: {len(cells)}", flush=True)

    for method, rank, c_name in cells:
        print(f"\n{'='*70}\n  {method.upper()}  rank={rank}  C={c_name}"
              f"  warm-start={WARM_START_HP[method]}\n{'='*70}", flush=True)
        t0 = time.time()
        s = search_cell(method, rank, c_name, n_trials=args.n_trials)
        dt = (time.time() - t0) / 60
        if "best_acc" in s:
            print(f"  cell done in {dt:.1f}min — best acc={s['best_acc']:.2f}, "
                  f"best HP={s['best_hp']}", flush=True)

    elapsed_h = (time.time() - t_total) / 3600
    print(f"\n{'='*70}\nALL DONE — total wall time: {elapsed_h:.2f} h\n{'='*70}",
          flush=True)


if __name__ == "__main__":
    main()

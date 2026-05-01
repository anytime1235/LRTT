#!/usr/bin/env python3
"""LRTT-v2 (selector / blockwise) AB-tile AF/noise cell-grid sweep, rank=8.

Companion to hp_search_v1_decay_b_af_noise.py, but uses the LRTT-v2 algorithm.
A and B tiles share the SAME swept (ab_up_down, ab_noise_ratio) — both A and B
are 6T1C LinearStepDevice with identical statistics (consistent with the
reference unit_cell_devices=[ab, ab, c] in lrtt_v2_mnist_6t1c.py).

DIFFERENCE FROM hp_search_v2_rank8_b_af_noise.py (the original v2 sweep):
  - That script swept B-tile only (A fixed at up_down=0). Here A=B swept together
    so the result is directly comparable to the v1 AB-symmetric sweep.

LRTT-v2 backbone (matches hp_search_v2_rank8_b_af_noise.py):
  - update_mode      = "selector_reconstruction"
  - transfer_method  = "blockwise"
  - reinit_mode      = "standard" (unused in v2, kept for config validation)
  - selector_axis    = "row"
  - selector_policy  = "shuffled_cycle"
  - cap_stabilizer_enabled = True, cap_rho = 1.0
  - te (transfer_every) = 10  (matches v2 default and v1-decay-best-rank8)

Cell axes (8 cells = 2 × 4 × 1):
  LR_GRID        = {1.0, 0.1}
  TLR_GRID       = {10.0, 1.0, 0.1, 0.01}
  LIFETIME_GRID  = {1000}                     # fixed at 1000 per user request

Per-cell AB-tile grid (25 trials, GridSampler):
  ab_up_down ∈ {0.5, 1.0, 2.0, 5.0, 10.0}     (applied to both A and B)
  ab_noise_ratio ∈ {0.5, 1.0, 2.0, 5.0, 10.0} (applied to both A and B)

C-tile = idealized.

Total: 16 × 25 = 400 trials, ~5–6 h on 1 GPU.

Usage:
  python hp_search_v2_rank8_b_af_noise_AB.py
  python hp_search_v2_rank8_b_af_noise_AB.py --smoke
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

# Cell axes — anchored on v2's known optimum from prior B-only sweep
LR_GRID = [1.0, 0.1]
TLR_GRID = [10.0, 1.0, 0.1, 0.01]
LIFETIME_GRID = [1000]  # fixed at 1000 per user request

AB_UP_DOWN_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
AB_NOISE_RATIO_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
N_TRIALS_PER_CELL = len(AB_UP_DOWN_GRID) * len(AB_NOISE_RATIO_GRID)


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
# Devices — A and B share the same swept (up_down, noise_ratio)
# ---------------------------------------------------------------------------
def _make_ab_device(ab_lifetime_param, up_down, noise_ratio):
    """6T1C-style LinearStepDevice for both A and B tiles in v2 (AB symmetric)."""
    return LinearStepDevice(
        dw_min=0.001981,
        up_down=up_down,
        w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1 * noise_ratio,
        up_down_dtod=0.01,
        w_max_dtod=0.05 * noise_ratio,
        w_min_dtod=0.05 * noise_ratio,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3 * noise_ratio,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime_param,
        lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


def _make_c_device():
    return LinearStepDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        gamma_up=0.0, gamma_down=0.0,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0,
    )


def create_model(rank, te, tlr, lifetime_phys, ab_up_down, ab_noise_ratio,
                 selector_policy="shuffled_cycle", cap_rho=1.0):
    """Build the LRTT-v2 (selector / blockwise) analog MLP.
    A and B tiles share the same swept (ab_up_down, ab_noise_ratio)."""
    ab_lifetime_param = _ab_lifetime_param(lifetime_phys)
    a_device = _make_ab_device(ab_lifetime_param, ab_up_down, ab_noise_ratio)
    b_device = _make_ab_device(ab_lifetime_param, ab_up_down, ab_noise_ratio)
    c_device = _make_c_device()
    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=te,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode="standard",
        decay_factor=1.0,
        b_init_mode="zero",
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        forward_inject=False,
        selector_axis="row",
        selector_policy=selector_policy,
        selector_seed=SEED,
        selector_reset_b_on_advance=True,
        cap_stabilizer_enabled=True,
        cap_rho=cap_rho,
        cap_compensate_transfer=True,
        unit_cell_devices=[a_device, b_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.weight_scaling_omega = 0.6

    model = AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(HIDDEN, OUT, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    return model


def run_trial(rank, te, lr, tlr, lifetime_phys, ab_up_down, ab_noise_ratio,
              *, epochs=EPOCHS, smoke=False, trial=None):
    torch.manual_seed(SEED)
    train_loader, val_loader = build_loaders(smoke=smoke)
    model = create_model(rank, te, tlr, lifetime_phys, ab_up_down, ab_noise_ratio)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
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
# Sweep driver
# ---------------------------------------------------------------------------
def sweep_cell(lr, tlr, lifetime_phys, output_dir):
    cell_tag = f"lr{lr}_tlr{tlr}_lt{lifetime_phys}"
    cell_path = f"{output_dir}/cell_{cell_tag}.json"
    if os.path.exists(cell_path):
        with open(cell_path) as f:
            return json.load(f)

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
        acc, _ = run_trial(
            RANK, TE, lr, tlr, lifetime_phys,
            ab_up_down=ab_up_down, ab_noise_ratio=ab_noise_ratio,
            trial=trial,
        )
        return acc

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS_PER_CELL)
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"  Best: {best.value:.2f}% "
          f"ab_up_down={best.params['ab_up_down']:.4f} "
          f"ab_noise_ratio={best.params['ab_noise_ratio']:.4f}   "
          f"({elapsed/60:.1f} min)")

    record = {
        "method": "lrtt_v2_selector_blockwise_AB",
        "rank": RANK, "te": TE,
        "lr": lr, "tlr": tlr, "lifetime_phys": lifetime_phys,
        "best_acc": round(best.value, 2),
        "best_ab_up_down": best.params["ab_up_down"],
        "best_ab_noise_ratio": best.params["ab_noise_ratio"],
        "n_trials": N_TRIALS_PER_CELL,
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
    return record


def _lifetime_label(lt):
    return "none" if lt is None else str(lt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("=== smoke trial ===")
        acc, hist = run_trial(
            RANK, TE, lr=1.0, tlr=10.0, lifetime_phys=1000,
            ab_up_down=2.0, ab_noise_ratio=1.0, epochs=2, smoke=True,
        )
        print(f"  acc={acc:.2f}%  hist={hist}")
        return

    output_dir = args.out or "/root/LRTT/results/hp_search_v2_rank8_b_af_noise_AB"
    os.makedirs(output_dir, exist_ok=True)

    cells = [(lr, tlr, lt) for lr in LR_GRID for tlr in TLR_GRID
             for lt in LIFETIME_GRID]
    total = len(cells)
    total_trials = total * N_TRIALS_PER_CELL

    print("=" * 74)
    print(f"LRTT-v2 (selector / blockwise)  AB-tile AF/noise sweep — rank={RANK}")
    print(f"Cell axes: lr ∈ {LR_GRID}, tlr ∈ {TLR_GRID}, "
          f"lifetime ∈ {[_lifetime_label(l) for l in LIFETIME_GRID]}")
    print(f"Per-cell grid: ab_up_down ∈ {AB_UP_DOWN_GRID} × "
          f"ab_noise_ratio ∈ {AB_NOISE_RATIO_GRID}  ({N_TRIALS_PER_CELL} pts)")
    print(f"Note: same (ab_up_down, ab_noise_ratio) is applied to BOTH A and B tiles")
    print(f"{total} cells × {N_TRIALS_PER_CELL} trials = {total_trials} total")
    print(f"Output dir : {output_dir}")
    print("=" * 74)

    all_results = []
    partial_path = f"{output_dir}/results_partial.json"
    done = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        done = {(r["lr"], r["tlr"], r["lifetime_phys"]) for r in all_results}
        print(f"Resuming: {len(done)} cells already done.")

    overall_t0 = time.time()
    for cell_idx, (lr, tlr, lifetime_phys) in enumerate(cells, 1):
        cell_key = (lr, tlr, lifetime_phys)
        if cell_key in done:
            print(f"[{cell_idx}/{total}] lr={lr} tlr={tlr} "
                  f"lifetime={_lifetime_label(lifetime_phys)} -- SKIP")
            continue
        elapsed = time.time() - overall_t0
        print(f"\n[{cell_idx}/{total}] lr={lr} tlr={tlr} "
              f"lifetime={_lifetime_label(lifetime_phys)}  "
              f"(elapsed {elapsed/3600:.2f} h)")
        record = sweep_cell(lr, tlr, lifetime_phys, output_dir)
        all_results.append(record)
        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2)

    with open(f"{output_dir}/results_final.json", "w") as f:
        json.dump(all_results, f, indent=2)
    overall = time.time() - overall_t0
    print(f"\nDone. Total wall time: {overall/60:.1f} min "
          f"({overall/3600:.2f} h). Results: {output_dir}/results_final.json")

    print("\n" + "=" * 74)
    print(f"Per-cell best (rank={RANK}, te={TE})")
    by = {(r["lr"], r["tlr"], r["lifetime_phys"]): r for r in all_results}
    for lt in LIFETIME_GRID:
        print(f"\nlifetime = {_lifetime_label(lt)}")
        header = "    lr\\tlr   " + "".join(f"  {t:>9.4f}" for t in TLR_GRID)
        print(header)
        for lr in LR_GRID:
            row = f"  {lr:>8.2f}  "
            for tlr in TLR_GRID:
                r = by.get((lr, tlr, lt))
                row += f"  {r['best_acc']:7.2f}% " if r else "         - "
            print(row)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LRTT-v2 6T1C grid sweep over (lifetime × transfer_lr × transfer_every).

Grid points (32 cells):
  lifetime       ∈ {10, 100, 1000, 46505}                    # 4 log-spaced
  transfer_lr    ∈ {1e-3, 1e-2}                              # 2 values
  transfer_every ∈ {1, 10, 100, 1000}                        # 4 values

Each cell = 1 trial of full MNIST × 10 epochs. Optuna GridSampler iterates over
the Cartesian product. HyperbandPruner kills bad trials at ep2/4/7. A 3-hour
deadline stops the sweep early; remaining cells can be resumed via the SQLite
study DB on a later run.

Fixed:
  lr = 0.187, rank = 8, hidden = 128, batch = 128
  policy = shuffled_cycle, cap_rho = 1.0
  A/B = 6T1C LinearStepDevice, C = Idealized LinearStepDevice
"""

import os
os.environ.setdefault("LRTT_SILENT", "1")

import argparse
import json
import time
from datetime import datetime
from itertools import product

import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42

# Fixed HPs
RANK = 8
HIDDEN = 128
BATCH_SIZE = 128
LR = 0.187
EPOCHS = 10
POLICY = "shuffled_cycle"
CAP_RHO = 1.0

# Grid axes
LIFETIME_GRID = [10.0, 100.0, 1000.0, 46505.0]
TLR_GRID = [1e-3, 1e-2, 1.0]
TE_GRID = [1, 10, 100]


def make_6t1c_ab_device(lifetime: float):
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0,
        w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=float(lifetime), lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


def make_idealized_c_device():
    return LinearStepDevice(
        dw_min=0.001,
        w_max=1.0, w_min=-1.0,
        gamma_up=0.0, gamma_down=0.0,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False,
        mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0,
    )


def build_v2_lrtt_config(lifetime: float, transfer_every: int, transfer_lr: float):
    ab = make_6t1c_ab_device(lifetime=lifetime)
    c = make_idealized_c_device()
    dev = PythonLRTTDevice(
        rank=RANK,
        transfer_every=int(transfer_every),
        transfer_lr=float(transfer_lr),
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        forward_inject=False,
        b_init_mode="zero",
        reinit_mode="standard",
        decay_factor=1.0,
        selector_axis="row",
        selector_policy=POLICY,
        selector_seed=SEED,
        selector_reset_b_on_advance=True,
        cap_stabilizer_enabled=True,
        cap_rho=CAP_RHO,
        cap_compensate_transfer=False,
        unit_cell_devices=[ab, ab, c],
    )
    rpu = PythonLRTTRPUConfig(device=dev)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return rpu


def build_model(lifetime, te, tlr):
    rpu = build_v2_lrtt_config(lifetime, te, tlr)
    return AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(HIDDEN, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)


# Cached MNIST loaders (built once, reused across trials)
_TRAIN_LOADER = None
_VAL_LOADER = None


def get_loaders():
    global _TRAIN_LOADER, _VAL_LOADER
    if _TRAIN_LOADER is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform)
        val_ds = datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform)
        _TRAIN_LOADER = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=False, persistent_workers=True,
        )
        _VAL_LOADER = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=1, pin_memory=False, persistent_workers=True,
        )
    return _TRAIN_LOADER, _VAL_LOADER


def run_trial(trial, lifetime, te, tlr):
    """Train one config for EPOCHS with intermediate reporting for pruning."""
    torch.manual_seed(SEED)
    train_loader, val_loader = get_loaders()
    model = build_model(lifetime, te, tlr)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for data, target in train_loader:
            data = data.to(DEVICE).view(data.shape[0], -1)
            target = target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * data.shape[0]
            n += data.shape[0]
        train_loss = running / max(1, n)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(DEVICE).view(data.shape[0], -1)
                target = target.to(DEVICE)
                correct += model(data).argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        scheduler.step()
        acc = 100.0 * correct / max(1, total)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": acc})
        best_acc = max(best_acc, acc)

        trial.report(acc, step=epoch)
        if trial.should_prune():
            del model
            trial.set_user_attr("history", history)
            raise optuna.TrialPruned(f"epoch {epoch}: pruned at acc={acc:.2f}%")

    trial.set_user_attr("history", history)
    del model
    return best_acc


def make_objective(deadline_seconds: float, start_time: float):
    def objective(trial):
        elapsed = time.time() - start_time
        if elapsed > deadline_seconds:
            print(f"[deadline] {elapsed/60:.1f} min elapsed > "
                  f"{deadline_seconds/60:.0f} min budget; stopping", flush=True)
            trial.study.stop()
            return float("nan")

        # GridSampler returns the next combination (deterministic order)
        lifetime = trial.suggest_categorical("lifetime", LIFETIME_GRID)
        tlr = trial.suggest_categorical("transfer_lr", TLR_GRID)
        te = trial.suggest_categorical("transfer_every", TE_GRID)

        t0 = time.time()
        print(f"\n[trial {trial.number}] lifetime={lifetime:.1f}, "
              f"tlr={tlr:.5f}, te={te}", flush=True)
        try:
            best_acc = run_trial(trial, lifetime, te, tlr)
            dt = time.time() - t0
            print(f"  → best_acc={best_acc:.2f}%, took {dt/60:.1f} min "
                  f"(elapsed {(time.time()-start_time)/60:.1f} min)", flush=True)
            return best_acc
        except optuna.TrialPruned as e:
            dt = time.time() - t0
            print(f"  → PRUNED ({e}), took {dt/60:.1f} min "
                  f"(elapsed {(time.time()-start_time)/60:.1f} min)", flush=True)
            raise
        except Exception as e:
            print(f"  → ERROR: {e}", flush=True)
            raise

    return objective


def dump_partial(study, output_dir: str):
    out = []
    for t in study.trials:
        if t.value is None and t.state.name not in ("PRUNED", "COMPLETE"):
            continue
        out.append({
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "history": t.user_attrs.get("history"),
        })
    with open(f"{output_dir}/results_partial.json", "w") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seconds", type=int, default=10800,
                        help="Deadline in seconds (3 hours = 10800)")
    parser.add_argument("--study_name", default="lrtt_v2_6t1c_grid")
    args = parser.parse_args()

    output_dir = "results/sweep_lrtt_v2_6t1c_3hr"
    os.makedirs(output_dir, exist_ok=True)

    grid_total = len(LIFETIME_GRID) * len(TLR_GRID) * len(TE_GRID)
    print("=" * 70)
    print(f"LRTT-v2 6T1C GRID SWEEP (3-hour budget)")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Total grid cells: {grid_total} "
          f"({len(LIFETIME_GRID)}×{len(TLR_GRID)}×{len(TE_GRID)})")
    print(f"  Deadline: {args.max_seconds}s ({args.max_seconds/3600:.1f}h)")
    print(f"  lifetime ∈ {LIFETIME_GRID}")
    print(f"  transfer_lr ∈ {TLR_GRID}")
    print(f"  transfer_every ∈ {TE_GRID}")
    print(f"  Fixed: lr={LR}, rank={RANK}, hidden={HIDDEN}, "
          f"epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"  Devices: A/B=6T1C LinearStep, C=Idealized LinearStep")
    print(f"  Pruner: HyperbandPruner (min_resource=2, reduction_factor=3)")
    print(f"  Output: {output_dir}/")
    print(f"  Compute: {DEVICE}")
    print("=" * 70, flush=True)

    storage = f"sqlite:///{output_dir}/{args.study_name}.db"

    # Order the grid: prioritize lifetime variation first (most informative axis),
    # then tlr, then te.
    search_space = {
        "lifetime": LIFETIME_GRID,
        "transfer_lr": TLR_GRID,
        "transfer_every": TE_GRID,
    }
    sampler = optuna.samplers.GridSampler(search_space, seed=SEED)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2, max_resource=EPOCHS, reduction_factor=3,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    start_time = time.time()
    objective = make_objective(args.max_seconds, start_time)

    def callback(study, trial):
        dump_partial(study, output_dir)
        elapsed_min = (time.time() - start_time) / 60
        completed = sum(1 for t in study.trials if t.state.name == "COMPLETE")
        pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
        try:
            best = study.best_trial
            print(f"  [study] {len(study.trials)} trials "
                  f"(complete={completed}, pruned={pruned}), "
                  f"best={best.value:.2f}% (trial {best.number}), "
                  f"elapsed={elapsed_min:.1f}/{args.max_seconds/60:.0f} min",
                  flush=True)
        except ValueError:
            pass

    try:
        study.optimize(
            objective,
            n_trials=grid_total,        # all grid cells
            callbacks=[callback],
            gc_after_trial=True,
            catch=(RuntimeError,),
        )
    except KeyboardInterrupt:
        print("Interrupted; saving partial results.", flush=True)

    dump_partial(study, output_dir)
    elapsed = (time.time() - start_time) / 60

    completed = [t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    completed.sort(key=lambda t: t.value, reverse=True)

    print("\n" + "=" * 70)
    print(f"SWEEP DONE: {len(study.trials)} trials, "
          f"complete={len(completed)}, pruned={len(pruned)}, "
          f"elapsed {elapsed:.1f} min")
    print("=" * 70)

    if completed:
        best = completed[0]
        print(f"Best trial #{best.number}: {best.value:.2f}%")
        print(f"  lifetime={best.params['lifetime']:.1f}")
        print(f"  transfer_lr={best.params['transfer_lr']:.5f}")
        print(f"  transfer_every={best.params['transfer_every']}")
        with open(f"{output_dir}/best_curve.json", "w") as f:
            json.dump({
                "best_trial": best.number,
                "best_acc": best.value,
                "params": best.params,
                "history": best.user_attrs.get("history", []),
            }, f, indent=2)

        print(f"\nTop {min(8, len(completed))}:")
        for i, t in enumerate(completed[:8]):
            print(f"  {i+1}. trial {t.number}: {t.value:.2f}% — "
                  f"life={t.params['lifetime']:.1f}, "
                  f"tlr={t.params['transfer_lr']:.4f}, "
                  f"te={t.params['transfer_every']}")
    else:
        print("No completed trials.")

    # Also dump the full grid status (for resume awareness)
    grid_status = []
    for life, tlr, te in product(LIFETIME_GRID, TLR_GRID, TE_GRID):
        match = next((t for t in study.trials
                      if t.params.get("lifetime") == life
                      and t.params.get("transfer_lr") == tlr
                      and t.params.get("transfer_every") == te), None)
        grid_status.append({
            "lifetime": life, "transfer_lr": tlr, "transfer_every": te,
            "status": match.state.name if match else "TODO",
            "value": match.value if match else None,
        })
    with open(f"{output_dir}/grid_status.json", "w") as f:
        json.dump(grid_status, f, indent=2)


if __name__ == "__main__":
    main()

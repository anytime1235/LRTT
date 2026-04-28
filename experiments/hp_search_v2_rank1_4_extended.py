#!/usr/bin/env python3
"""LRTT-v2 HP sweep — rank ∈ {1, 4} × te ∈ {1, 10, 100, 500} × lifetime ∈ {10,100,1000,6T1C}.

Adapted from hp_search_full_grid_lrtt_v2.py with these expansions:
  - search ranges widened: lr log-uniform [0.05, 10.0], tlr log-uniform [1e-4, 10.0]
  - new sweep variable: lifetime as an Optuna categorical {10, 100, 1000, 46505}
  - reduced grid: ranks ∈ {1, 4} only, te ∈ {1, 10, 100, 500}
  - device: 6T1C-style LinearStepDevice; only the lifetime parameter is swept

Each trial reports best validation accuracy over EPOCHS=30 with early stopping.
Results saved to results/hp_search_v2_rank1_4/.

Usage:
  python hp_search_v2_rank1_4_extended.py --mode shuffled_cycle [--smoke]
"""

from __future__ import annotations

import os
os.environ.setdefault("LRTT_SILENT", "1")

import argparse
import json
import math

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
N_TRIALS = 12         # 30 → 12: 4 lifetime-baseline enqueues + 8 TPE explorations
SEED = 42
HIDDEN = 256
OUT = 10
RANKS_DEFAULT = [1, 4]
TES_DEFAULT = [1, 10, 100, 500]
# Categorical lifetime sweep — physical retention τ in seconds
# 6T1C nominal = 46505 (matches published baseline)
LIFETIME_GRID = [10, 100, 1000, 46505]
TAU_SEC = 46505.0


# Reuse v-1 best HP cells as Optuna seeds where the cell exists.
# Sources: hp_search_full_grid_no_multnoise.py (rank 1,4,8,16,32,64), and
# lrtt_sweep_best_configs.json (rank=8 published best at te=10).
SEED_HPS = {
    (1, 10): (0.1908, 0.04726), (1, 100): (0.1218, 0.04350),
    (1, 500): (0.2560, 0.06478),
    (4, 10): (0.3890, 0.001048), (4, 100): (0.3418, 0.003911),
    (4, 500): (0.2412, 0.04950),
    (8, 10): (0.5604, 0.000657),    # published best at rank=8 te=10
    (8, 100): (0.1719, 0.002665), (8, 500): (0.1502, 0.02262),
    (16, 10): (0.1407, 0.08573), (16, 100): (0.1872, 0.002161),
    (16, 500): (0.2335, 0.004402),
    (32, 10): (0.1699, 0.001749), (32, 100): (0.9934, 0.001219),
    (32, 500): (0.6024, 0.001898),
    (64, 10): (0.5644, 0.000082),  # published best at rank=64 te=10
    (64, 100): (0.2520, 0.001336), (64, 500): (0.2520, 0.001336),
}


# ---------------------------------------------------------------------------
# Data loaders (lazy)
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


def _ab_lifetime_param(lifetime_phys: int) -> float:
    """Convert physical retention (seconds) → AIHWKit lifetime parameter.
    Reproduces the conversion used by lrtt_v2_mnist_6t1c.make_6t1c_ab_device."""
    dt_batch_sec = -TAU_SEC * math.log(1 - 1.0 / lifetime_phys)
    return 1.0 / (1 - math.exp(-dt_batch_sec / TAU_SEC))


def create_model(rank, te, tlr, selector_policy, cap_rho, lifetime_phys):
    """Build the 2-layer MLP. Only the hidden layer is LRTT-v2."""
    ab_lifetime_param = _ab_lifetime_param(lifetime_phys)
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=False,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime_param, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )
    c_device = LinearStepDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        gamma_up=0.0, gamma_down=0.0,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0,
    )
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
        unit_cell_devices=[ab_device, ab_device, c_device],
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


def run_trial(rank, te, lr, tlr, selector_policy, cap_rho, lifetime_phys,
              *, epochs=EPOCHS, smoke=False, trial=None):
    torch.manual_seed(SEED)
    train_loader, val_loader = build_loaders(smoke=smoke)
    model = create_model(rank, te, tlr, selector_policy, cap_rho, lifetime_phys)
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
        # Optuna pruner integration: report intermediate value, prune mediocre trials
        if trial is not None and not smoke:
            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not smoke and epoch >= 5 and best_acc < 50.0:
            break
        if not smoke and patience >= EARLY_STOP_PATIENCE:
            break

    info = {}
    last_lrtt = None
    for m in model.modules():
        ctrl = getattr(getattr(m, "tile", None), "controller", None)
        if ctrl is not None:
            last_lrtt = ctrl
    if last_lrtt is not None:
        info = {
            "num_a_updates": last_lrtt.num_a_updates,
            "num_b_updates": last_lrtt.num_b_updates,
            "num_transfers": last_lrtt.num_transfers,
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc, history, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="shuffled_cycle",
                        choices=["cyclic", "shuffled_cycle", "random"])
    parser.add_argument("--cap_rho", type=float, default=1.0,
                        help="Capacitor leakage factor; 1.0 = no controller leak.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="results/hp_search_v2_rank1_4")
    parser.add_argument("--ranks", default=",".join(str(r) for r in RANKS_DEFAULT),
                        help="Comma-separated ranks to sweep (default: 1,4). "
                             "Examples: --ranks 8  or  --ranks 8,16")
    parser.add_argument("--tes", default=",".join(str(t) for t in TES_DEFAULT),
                        help="Comma-separated TEs to sweep (default: 1,10,100,500)")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                        help=f"Trials per cell (default: {N_TRIALS}). "
                             "First 4 are lifetime baselines.")
    args = parser.parse_args()
    RANKS = [int(r) for r in args.ranks.split(",") if r.strip()]
    TES = [int(t) for t in args.tes.split(",") if t.strip()]
    n_trials_per_cell = args.n_trials

    if args.smoke:
        rank, te = 1, 10
        seed_lr, seed_tlr = SEED_HPS.get((rank, te), (0.1, 0.001))
        print(f"=== smoke trial rank={rank} te={te} lifetime=46505 ===")
        acc, hist, info = run_trial(
            rank, te, seed_lr, seed_tlr,
            selector_policy=args.mode, cap_rho=args.cap_rho,
            lifetime_phys=46505, epochs=2, smoke=True,
        )
        print(f"  acc={acc:.2f}%  hist={hist}  info={info}")
        return

    output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)
    total = len(RANKS) * len(TES)
    print("=" * 70)
    print(f"LRTT-v2 HP sweep — rank ∈ {RANKS}, TE ∈ {TES}, "
          f"lifetime ∈ {LIFETIME_GRID}, policy={args.mode}")
    print(f"Search space per cell: lr log-unif [0.05, 10.0], "
          f"tlr log-unif [1e-4, 10.0], lifetime categorical")
    print(f"{total} (rank,te) cells × {N_TRIALS} trials = {total*N_TRIALS} total")
    print("=" * 70)

    all_results = []
    partial_path = f"{output_dir}/results_partial.json"
    done = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        done = {(r["rank"], r["te"]) for r in all_results}
        print(f"Resuming: {len(done)} cells already done.")

    cell_idx = 0
    for rank in RANKS:
        if HIDDEN % rank != 0:
            print(f"  skip rank={rank}: HIDDEN={HIDDEN} not divisible")
            continue
        for te in TES:
            cell_idx += 1
            if (rank, te) in done:
                print(f"[{cell_idx}/{total}] rank={rank} TE={te} -- SKIP")
                continue
            seed_lr, seed_tlr = SEED_HPS.get((rank, te), (0.1, 0.001))
            print(f"\n[{cell_idx}/{total}] rank={rank} TE={te} "
                  f"(seed lr={seed_lr:.4f}, tlr={seed_tlr:.6f})")

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=3, max_resource=EPOCHS, reduction_factor=3,
                ),
            )
            # Seed each lifetime category with v-1's best (lr, tlr) → fair
            # category coverage before TPE/Hyperband begin pruning.
            for L in LIFETIME_GRID:
                study.enqueue_trial({
                    "lr": seed_lr,
                    "tlr": seed_tlr,
                    "lifetime_phys": L,
                })

            def objective(trial, _rank=rank, _te=te):
                lr = trial.suggest_float("lr", 0.05, 10.0, log=True)
                tlr = trial.suggest_float("tlr", 1e-4, 10.0, log=True)
                lifetime_phys = trial.suggest_categorical(
                    "lifetime_phys", LIFETIME_GRID
                )
                acc, _, _ = run_trial(
                    _rank, _te, lr, tlr,
                    selector_policy=args.mode, cap_rho=args.cap_rho,
                    lifetime_phys=lifetime_phys, trial=trial,
                )
                return acc

            study.optimize(objective, n_trials=n_trials_per_cell)

            best = study.best_trial
            print(f"  Best: {best.value:.2f}% lr={best.params['lr']:.4f} "
                  f"tlr={best.params['tlr']:.6f} lifetime={best.params['lifetime_phys']}")

            cell_record = {
                "rank": rank, "te": te,
                "policy": args.mode, "cap_rho": args.cap_rho,
                "best_acc": round(best.value, 2),
                "best_lr": best.params["lr"],
                "best_tlr": best.params["tlr"],
                "best_lifetime_phys": best.params["lifetime_phys"],
                "n_trials": n_trials_per_cell,
                # Per-lifetime best for honest ablation
                "best_per_lifetime": {},
                "all_trials": [],
            }
            for L in LIFETIME_GRID:
                trials_for_L = [
                    t for t in study.trials
                    if t.value is not None and t.params.get("lifetime_phys") == L
                ]
                if trials_for_L:
                    bt = max(trials_for_L, key=lambda t: t.value)
                    cell_record["best_per_lifetime"][str(L)] = {
                        "acc": round(bt.value, 2),
                        "lr": bt.params["lr"],
                        "tlr": bt.params["tlr"],
                    }
            for t in study.trials:
                if t.value is None:
                    continue
                cell_record["all_trials"].append({
                    "lr": t.params["lr"],
                    "tlr": t.params["tlr"],
                    "lifetime_phys": t.params["lifetime_phys"],
                    "acc": round(t.value, 2),
                })

            all_results.append(cell_record)
            with open(partial_path, "w") as f:
                json.dump(all_results, f, indent=2)

    final_path = f"{output_dir}/results_final.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved to {final_path}")

    # Print best-per-cell heatmap
    print("\n" + "=" * 70)
    print(f"BEST ACCURACY (rank x TE), policy={args.mode}, cap_rho={args.cap_rho}")
    header = f"{'rank':>6s}" + "".join(f"  TE={te:<5d}" for te in TES)
    print(header)
    by = {(r["rank"], r["te"]): r for r in all_results}
    for rank in RANKS:
        row = f"{rank:>6d}"
        for te in TES:
            r = by.get((rank, te))
            row += f"  {r['best_acc']:5.2f}% " if r else f"  {' ':>6s} "
        print(row)


if __name__ == "__main__":
    main()

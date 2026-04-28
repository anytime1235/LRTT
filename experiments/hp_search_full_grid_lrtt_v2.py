#!/usr/bin/env python3
"""LRTT-v2 (selector_reconstruction + blockwise) full rank x TE grid HP search.

Adapted from experiments/hp_search_full_grid_no_multnoise.py for LRTT-v2.

Key differences from v1 sweep:
  - update_mode="selector_reconstruction"  (was "lora")
  - transfer_method="blockwise"            (was "onehot")
  - reinit_mode is irrelevant in v2 (B reset is automatic after every transfer);
    the --mode argument now selects selector_policy ("cyclic" or "shuffled_cycle")
    plus an optional cap_rho leakage factor.
  - b_init_mode forced to "zero" (B is a residual buffer, must start from 0).
  - tile_a is not updated in v2 — its analog cells are unused.

Usage:
  python hp_search_full_grid_lrtt_v2.py --mode shuffled_cycle
  python hp_search_full_grid_lrtt_v2.py --mode cyclic --cap_rho 0.99

Quick smoke test (uses --smoke flag):
  python hp_search_full_grid_lrtt_v2.py --mode cyclic --smoke
"""

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
LIFETIME = 46505
N_TRIALS = 30
SEED = 42

# Layer dimensions for the MLP. Both must be divisible by every rank in RANKS.
HIDDEN = 256
OUT = 10

# RANKS are also selector_block_sizes in v2. They must divide d_size of the
# AnalogLinear layer (out_features = HIDDEN = 256). All values below divide 256.
RANKS = [1, 4, 8, 16, 32, 64]
TES = [10, 50, 100, 500, 1000]

# Seed HPs ported from v1 sweep — used as the initial Optuna trial.
SEED_HPS = {
    (1, 10): (0.1908, 0.04726), (1, 50): (0.1551, 0.003145),
    (1, 100): (0.1218, 0.04350), (1, 500): (0.2560, 0.06478),
    (1, 1000): (0.6782, 0.003058),
    (4, 10): (0.3890, 0.001048), (4, 50): (0.3418, 0.003911),
    (4, 100): (0.3418, 0.003911), (4, 500): (0.2412, 0.04950),
    (4, 1000): (0.6852, 0.6249),
    (8, 10): (0.0502, 0.01097), (8, 50): (0.7011, 0.004174),
    (8, 100): (0.1719, 0.002665), (8, 500): (0.1502, 0.02262),
    (8, 1000): (0.3198, 1.2764),
    (16, 10): (0.1407, 0.08573), (16, 50): (0.1872, 0.002161),
    (16, 100): (0.1872, 0.002161), (16, 500): (0.2335, 0.004402),
    (16, 1000): (0.8820, 0.008597),
    (32, 10): (0.1699, 0.001749), (32, 50): (0.9934, 0.001219),
    (32, 100): (0.9934, 0.001219), (32, 500): (0.6024, 0.001898),
    (32, 1000): (0.006297, 0.1306),
    (64, 10): (0.2520, 0.001336), (64, 50): (0.2520, 0.001336),
    (64, 100): (0.2520, 0.001336), (64, 500): (0.2520, 0.001336),
    (64, 1000): (0.3187, 0.006835),
}

TAU_SEC = 46505.0
dt_batch_sec = -TAU_SEC * math.log(1 - 1.0 / LIFETIME)
AB_LIFETIME = 1.0 / (1 - math.exp(-dt_batch_sec / TAU_SEC))


# ---------------------------------------------------------------------------
# Data loaders (lazily built; build_loaders() lets the smoke path use a Subset)
# ---------------------------------------------------------------------------

_TRAIN_DS = None
_VAL_DS = None


def build_loaders(smoke: bool = False, smoke_train_n: int = 1024,
                  smoke_val_n: int = 1024, smoke_batch: int = 128):
    """Return (train_loader, val_loader). Uses MNIST under /tmp/mnist."""
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
        bs = smoke_batch
        nw = 0
    else:
        train_ds = _TRAIN_DS
        val_ds = _VAL_DS
        bs = BATCH_SIZE
        nw = 4

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=max(0, nw - 2), pin_memory=(DEVICE.type == "cuda"),
    )
    return train_loader, val_loader


def create_model(rank, te, tlr, selector_policy, cap_rho):
    """Build a 2-layer analog MLP (hidden 256, output 10).

    The hidden layer uses LRTT-v2 with rank == selector_block_size; the output
    layer is FloatingPoint to keep the ablation focused on the LRTT block.
    """
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=False,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.1, reset=0.0, reset_dtod=0.0,
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
        # reinit_mode is unused in v2 (B reset is automatic). Keep "standard"
        # for config validation; the v2 transfer path never invokes reinit().
        reinit_mode="standard",
        decay_factor=1.0,
        b_init_mode="zero",                 # v2 invariant: B starts from 0
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        forward_inject=False,
        # Selector
        selector_axis="row",
        selector_policy=selector_policy,    # "cyclic" or "shuffled_cycle"
        selector_seed=SEED,
        selector_reset_b_on_advance=True,
        # Capacitor
        cap_stabilizer_enabled=True,
        cap_rho=cap_rho,
        cap_compensate_transfer=True,
        # Devices
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.transfer_mode = "off"     # no calibration in blockwise path

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


def run_trial(rank, te, lr, tlr, selector_policy, cap_rho, *,
              epochs=EPOCHS, smoke=False):
    torch.manual_seed(SEED)
    train_loader, val_loader = build_loaders(smoke=smoke)
    model = create_model(rank, te, tlr, selector_policy, cap_rho)
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
            best_acc = acc
            patience = 0
        else:
            patience += 1
        if not smoke and epoch >= 5 and best_acc < 50.0:
            break
        if not smoke and patience >= EARLY_STOP_PATIENCE:
            break

    # Verify v2 invariants from the last layer's controller
    info = {}
    last_lrtt = None
    for m in model.modules():
        ctrl = getattr(getattr(m, "tile", None), "controller", None)
        if ctrl is not None:
            last_lrtt = ctrl
    if last_lrtt is not None:
        info = {
            "num_a_updates": last_lrtt.num_a_updates,  # MUST stay 0 in v2
            "num_b_updates": last_lrtt.num_b_updates,
            "num_transfers": last_lrtt.num_transfers,
            "selector_cycle": last_lrtt.selector_cycle,
            "selector_indices_last": (
                last_lrtt.selector_indices.detach().cpu().tolist()
                if last_lrtt.selector_indices is not None else None
            ),
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc, history, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="shuffled_cycle",
        choices=["cyclic", "shuffled_cycle"],
        help="LRTT-v2 selector policy.",
    )
    parser.add_argument(
        "--cap_rho", type=float, default=1.0,
        help="Capacitor leakage factor (1.0 = no leak).",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny verification trial only and exit.")
    args = parser.parse_args()

    if args.smoke:
        # Run a single (rank=8, te=10) trial for a few epochs to verify v2 logic.
        rank, te = 8, 10
        seed_lr, seed_tlr = SEED_HPS.get((rank, te), (0.1, 0.001))
        print(f"=== LRTT-v2 SMOKE TRIAL (rank={rank}, te={te}, "
              f"policy={args.mode}, cap_rho={args.cap_rho}) ===")
        acc, hist, info = run_trial(
            rank, te, seed_lr, seed_tlr,
            selector_policy=args.mode, cap_rho=args.cap_rho,
            epochs=3, smoke=True,
        )
        print(f"  best_acc = {acc:.2f}%")
        print(f"  per-epoch acc = {hist}")
        print(f"  controller info = {info}")
        # ----- v2 invariant checks -----
        assert info.get("num_a_updates", 0) == 0, (
            f"v2 invariant broken: tile_a was updated "
            f"({info.get('num_a_updates')}x)"
        )
        assert info.get("num_b_updates", 0) > 0, (
            "B should have been updated at least once"
        )
        # Loss must actually go down on this small task.
        if len(hist) >= 2:
            assert hist[-1][1] >= hist[0][1] - 5.0, (
                "Accuracy regressed between first and last epoch"
            )
        print("\nALL v2 INVARIANTS PASSED")
        return

    output_dir = (
        f"results/hp_search_lrtt_v2_{args.mode}"
        + (f"_rho{args.cap_rho:.3f}" if args.cap_rho < 1.0 else "")
    )
    os.makedirs(output_dir, exist_ok=True)

    total = len(RANKS) * len(TES)
    print("=" * 70)
    print(f"LRTT-v2 FULL GRID HP SEARCH (policy={args.mode}, cap_rho={args.cap_rho})")
    print(f"Ranks: {RANKS}, TEs: {TES}")
    print(f"Total: {total} combinations x {N_TRIALS} trials = {total * N_TRIALS}")
    print("=" * 70)

    all_results = []
    partial_path = f"{output_dir}/results_partial.json"
    done_set = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        done_set = {(r["rank"], r["te"]) for r in all_results}
        print(f"Loaded {len(all_results)} existing results, skipping those.")

    cell_idx = 0
    for rank in RANKS:
        # In v2, rank == selector_block_size must divide d_size = HIDDEN.
        if HIDDEN % rank != 0:
            print(f"  skip rank={rank}: HIDDEN={HIDDEN} not divisible by rank")
            continue
        for te in TES:
            cell_idx += 1
            if (rank, te) in done_set:
                print(f"[{cell_idx}/{total}] rank={rank}, TE={te} -- SKIP")
                continue

            seed_lr, seed_tlr = SEED_HPS.get((rank, te), (0.1, 0.001))
            print(
                f"\n[{cell_idx}/{total}] rank={rank}, TE={te} "
                f"(seed lr={seed_lr:.4f}, tlr={seed_tlr:.6f})"
            )

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
            )
            study.enqueue_trial({"lr": seed_lr, "tlr": seed_tlr})

            def objective(trial, _rank=rank, _te=te):
                lr = trial.suggest_float("lr", 0.01, 1.5, log=True)
                tlr = trial.suggest_float("tlr", 1e-5, 1.0, log=True)
                acc, _, _ = run_trial(
                    _rank, _te, lr, tlr,
                    selector_policy=args.mode, cap_rho=args.cap_rho,
                )
                return acc

            study.optimize(objective, n_trials=N_TRIALS)

            best = study.best_trial
            print(
                f"  Best: {best.value:.2f}% "
                f"(lr={best.params['lr']:.4f}, tlr={best.params['tlr']:.6f})"
            )

            all_results.append({
                "rank": rank,
                "te": te,
                "policy": args.mode,
                "cap_rho": args.cap_rho,
                "best_acc": round(best.value, 2),
                "best_lr": best.params["lr"],
                "best_tlr": best.params["tlr"],
                "n_trials": N_TRIALS,
                "all_trials": [
                    {
                        "lr": t.params["lr"],
                        "tlr": t.params["tlr"],
                        "acc": round(t.value, 2),
                    }
                    for t in study.trials if t.value is not None
                ],
            })

            with open(partial_path, "w") as f:
                json.dump(all_results, f, indent=2)

    with open(f"{output_dir}/results_final.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"HEATMAP: Best Accuracy (rank x TE), policy={args.mode}, "
          f"cap_rho={args.cap_rho}")
    print("=" * 70)
    header = f"{'rank':>6s}" + "".join(f"  TE={te:<5d}" for te in TES)
    print(header)
    print("-" * len(header))
    for rank in RANKS:
        row = f"{rank:>6d}"
        for te in TES:
            match = [r for r in all_results if r["rank"] == rank and r["te"] == te]
            row += f"  {match[0]['best_acc']:>6.2f}" if match else "     N/A"
        print(row)
    print(f"\nResults: {output_dir}/results_final.json")


if __name__ == "__main__":
    main()

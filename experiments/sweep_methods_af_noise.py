#!/usr/bin/env python3
"""Method comparison: AF / noise sensitivity at C-tile bits=8, lifetime_steps=1000.

Compares 4 methods on MNIST (784->256->10).

Methods:
  - direct       SingleRPUConfig + LinearStepDevice (bits=8)            [ANCHOR ONLY]
  - tikitaka_v1  UnitCellRPUConfig + ChoppedTransferCompound, v1 flags  [ANCHOR ONLY]
                 (no_buffer=True, in_chop_prob=0, update_bl_management=True,
                  update_management=True) per transformer-branch
                 main_results/scripts/analysis/optuna_bert_squad_tiki.py.
                 Fast = 6T1C clean, Slow = perturbed LinearStepDevice (bits=8).
  - lrtt_v1      PythonLRTTRPUConfig + PythonLRTTDevice                 [SWEEP]
                 update_mode='lora', transfer_mode='off'.
                 A,B = 6T1C clean, C = perturbed LinearStepDevice (bits=8).
  - lrtt_v2      PythonLRTTRPUConfig + PythonLRTTDevice                 [SWEEP]
                 update_mode='selector_reconstruction', transfer_method='blockwise'.
                 A,B = 6T1C clean, C = perturbed LinearStepDevice (bits=8).

Sweep methods (lrtt_v1, lrtt_v2) get the full 1-D grid on each axis:
  af     : gamma_up = gamma_down in {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10}
           (12 levels), noise_ratio = 0
  noise  : noise_ratio in {0, 0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4}
           (12 levels), gamma = 0
           Scales 6T1C-style noise fields on the perturbed tile multiplicatively
           against NOISE_TEMPLATE.

Anchor methods (direct, tikitaka_v1) run only at (gamma=0, noise_ratio=0).
Plotted as horizontal reference lines on both AF and noise plots.

Fixed:
  bits = 8 (dw_min = 2/256 on perturbed tile)
  lifetime_steps = 1000 (universal: 6T1C Fast/A/B AND perturbed C/Slow/Single)
  rank = 8, TE = 10, omega = 0.6 (LRTT methods)
  30 epochs, batch 64, AnalogSGD + StepLR(step=10, gamma=0.5), early-stop patience 5

Per-method HPs (fixed; no Optuna):
  direct      : lr=0.1
  tikitaka_v1 : lr=0.1, fast_lr=1.0, transfer_lr=1.0
  lrtt_v1     : lr=0.116, transfer_lr=0.000935   (from sweep_ctile_bits_gamma.py bits=8)
  lrtt_v2     : lr=1.879, transfer_lr=0.0071     (from sweep_ctile_bits_gamma_v2.py)

Per cell: 10 runs without manual seed (natural RNG advance).

Total runs (default): sweep 2 methods x 2 axes x 12 levels x 10 runs = 480
                    + anchor 2 methods x 10 runs                    =  20
                    = 500 runs (target wall-clock ~10-12 h on A100).

Usage:
  python experiments/sweep_methods_af_noise.py
  python experiments/sweep_methods_af_noise.py --axis af
  python experiments/sweep_methods_af_noise.py --skip_anchor   # already done
  python experiments/sweep_methods_af_noise.py --smoke         # 1 run, 2 levels, 3 epochs
"""

import os; os.environ.setdefault("LRTT_SILENT", "1")
import argparse, math, json, time, sys
import torch, torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    SingleRPUConfig, UnitCellRPUConfig, FloatingPointRPUConfig,
    IOParameters, UpdateParameters,
)
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.parameters.enums import NoiseManagementType, BoundManagementType
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5

BITS = 8
RANK = 8
TE = 10
OMEGA = 0.6
LIFETIME_STEPS = 1000
TAU_SEC = 46505.0

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
NOISE_RATIOS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
N_RUNS_PER_CELL = 10
ANCHOR_RUNS = 10
ANCHOR_GAMMA = 0.0
ANCHOR_NOISE_RATIO = 0.0

METHODS_ALL = ["direct", "tikitaka_v1", "lrtt_v1", "lrtt_v2"]
SWEEP_METHODS = ["lrtt_v1", "lrtt_v2"]      # full AF / noise grid (12 levels)
ANCHOR_METHODS = ["direct", "tikitaka_v1"]  # single (gamma=0, noise=0) baseline only

HP_BY_METHOD = {
    "direct":      {"lr": 0.1},
    "tikitaka_v1": {"lr": 0.1, "fast_lr": 1.0, "transfer_lr": 1.0},
    "lrtt_v1":     {"lr": 0.11577, "transfer_lr": 0.000935},
    "lrtt_v2":     {"lr": 1.8789664173379752, "transfer_lr": 0.0071217696140313865},
}

# 6T1C noise template at noise_ratio=1.0 (matches A/B 6T1C noise profile).
# Each field is multiplied by noise_ratio for the perturbed tile.
NOISE_TEMPLATE = {
    "dw_min_std":      0.3,
    "dw_min_dtod":     0.1,
    "gamma_up_dtod":   0.05,
    "gamma_down_dtod": 0.05,
    "w_max_dtod":      0.05,
    "w_min_dtod":      0.05,
    "write_noise_std": 0.0,
}


def _ab_lifetime_value(lifetime_steps):
    if lifetime_steps is None or lifetime_steps <= 0:
        return 0.0
    dt_batch_sec = -TAU_SEC * math.log(1.0 - 1.0 / lifetime_steps)
    return 1.0 / (1.0 - math.exp(-dt_batch_sec / TAU_SEC))


AB_LIFETIME = _ab_lifetime_value(LIFETIME_STEPS)


def make_6t1c_device():
    """6T1C LinearStepDevice — clean baseline used for A/B (LRTT) and Fast (Tikitaka)."""
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


def make_perturbed_device(gamma, noise_ratio):
    """Perturbed LinearStepDevice — used for C (LRTT), Slow (Tikitaka), Single (direct).

    bits=8 -> dw_min = 2/256. AF via gamma_up = gamma_down = gamma.
    Noise via 6T1C-style noise template scaled by noise_ratio.
    Universal lifetime=1000 (lifetime_dtod=0.1 if active).
    """
    dw_min = 2.0 / (2 ** BITS)
    kwargs = dict(
        dw_min=dw_min, w_max=1.0, w_min=-1.0,
        gamma_up=gamma, gamma_down=gamma,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=False, mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=(0.1 if AB_LIFETIME > 0 else 0.0),
        reset=0.0, reset_dtod=0.0,
    )
    for field, base in NOISE_TEMPLATE.items():
        kwargs[field] = base * noise_ratio
    return LinearStepDevice(**kwargs)


def _apply_common_mapping(rpu):
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = OMEGA
    rpu.mapping.weight_scaling_columnwise = True
    rpu.mapping.learn_out_scaling = True
    rpu.mapping.out_scaling_columnwise = True
    return rpu


def make_rpu_direct(gamma, noise_ratio):
    rpu = SingleRPUConfig(device=make_perturbed_device(gamma, noise_ratio))
    return _apply_common_mapping(rpu)


def make_rpu_tikitaka_v1(gamma, noise_ratio):
    """Tikitaka v1 via ChoppedTransferCompound with v1-equivalent flags
    (per transformer-branch optuna_bert_squad_tiki.py:create_tikitaka_config(use_v2=False))."""
    fast = make_6t1c_device()
    slow = make_perturbed_device(gamma, noise_ratio)
    hp = HP_BY_METHOD["tikitaka_v1"]
    transfer_io = IOParameters(
        noise_management=NoiseManagementType.NONE,
        bound_management=BoundManagementType.NONE,
    )
    transfer_update = UpdateParameters(
        desired_bl=31,
        update_bl_management=True,   # v1
        update_management=True,      # v1
    )
    rpu = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[fast, slow],
            transfer_every=TE,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=hp["transfer_lr"],
            fast_lr=hp["fast_lr"],
            scale_transfer_lr=False,     # v1
            transfer_forward=transfer_io,
            transfer_update=transfer_update,
            no_buffer=True,              # v1
            in_chop_prob=0.0,            # v1: chopper off
            out_chop_prob=0.0,
            auto_scale=False,            # v1
            auto_momentum=0.99,
        ),
    )
    return _apply_common_mapping(rpu)


def _common_lrtt_mapping(rpu):
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = OMEGA
    rpu.mapping.weight_scaling_columnwise = True
    rpu.mapping.learn_out_scaling = True
    rpu.mapping.out_scaling_columnwise = True
    return rpu


def make_rpu_lrtt_v1(gamma, noise_ratio):
    a = make_6t1c_device(); b = make_6t1c_device()
    c = make_perturbed_device(gamma, noise_ratio)
    hp = HP_BY_METHOD["lrtt_v1"]
    dev = PythonLRTTDevice(
        rank=RANK, transfer_every=TE,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode="decay", decay_factor=1.0,
        unit_cell_devices=[a, b, c],
    )
    dev.transfer_lr = hp["transfer_lr"]
    dev.forward_inject = False
    dev.update_mode = "lora"
    dev.transfer_mode = "off"
    rpu = PythonLRTTRPUConfig(device=dev)
    return _common_lrtt_mapping(rpu)


def make_rpu_lrtt_v2(gamma, noise_ratio):
    a = make_6t1c_device(); b = make_6t1c_device()
    c = make_perturbed_device(gamma, noise_ratio)
    hp = HP_BY_METHOD["lrtt_v2"]
    dev = PythonLRTTDevice(
        rank=RANK, transfer_every=TE, transfer_lr=hp["transfer_lr"],
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        forward_inject=False,
        b_init_mode="zero",
        reinit_mode="standard", decay_factor=1.0,
        selector_axis="row", selector_policy="shuffled_cycle", selector_seed=42,
        selector_reset_b_on_advance=True,
        cap_stabilizer_enabled=True, cap_rho=1.0, cap_compensate_transfer=False,
        unit_cell_devices=[a, b, c],
    )
    rpu = PythonLRTTRPUConfig(device=dev)
    return _common_lrtt_mapping(rpu)


_METHOD_BUILDERS = {
    "direct":      make_rpu_direct,
    "tikitaka_v1": make_rpu_tikitaka_v1,
    "lrtt_v1":     make_rpu_lrtt_v1,
    "lrtt_v2":     make_rpu_lrtt_v2,
}


def create_model(method, gamma, noise_ratio):
    rpu = _METHOD_BUILDERS[method](gamma, noise_ratio)
    return AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)


_loaders = {}
def get_loaders():
    if "train" in _loaders:
        return _loaders["train"], _loaders["val"]
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    _loaders["train"] = DataLoader(
        datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    _loaders["val"] = DataLoader(
        datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return _loaders["train"], _loaders["val"]


def run_trial(method, gamma, noise_ratio, epochs=EPOCHS):
    """Train one model. No manual seed — natural RNG advance across runs."""
    train_loader, val_loader = get_loaders()
    model = create_model(method, gamma, noise_ratio)
    lr = HP_BY_METHOD[method]["lr"]
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
        history.append(round(acc, 2))
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
    torch.cuda.empty_cache()
    return best_acc, history


def axis_levels(axis):
    if axis == "af":
        return [{"gamma": g, "noise_ratio": 0.0, "level": g} for g in GAMMAS]
    if axis == "noise":
        return [{"gamma": 0.0, "noise_ratio": n, "level": n} for n in NOISE_RATIOS]
    raise ValueError(f"unknown axis: {axis}")


def run_axis(axis, methods, output_dir, n_runs, epochs):
    """Run sweep methods at all axis levels, n_runs per cell."""
    levels = axis_levels(axis)
    out_path = os.path.join(output_dir, f"{axis}_results.json")
    partial = os.path.join(output_dir, f"{axis}_results_partial.json")

    if os.path.exists(partial):
        with open(partial) as f:
            results = json.load(f)
        done = {(r["method"], r["level"], r["run_idx"]) for r in results}
        print(f"[{axis}] resuming with {len(done)} runs already complete")
    else:
        results = []
        done = set()

    total = len(methods) * len(levels) * n_runs
    completed = len(done)
    t0 = time.time()
    print(f"[{axis}] methods={methods}  levels={[L['level'] for L in levels]}  "
          f"n_runs={n_runs}  total={total}  remaining={total - completed}")

    for method in methods:
        for L in levels:
            for run_idx in range(n_runs):
                key = (method, L["level"], run_idx)
                if key in done:
                    continue
                t_run = time.time()
                acc, hist = run_trial(method, L["gamma"], L["noise_ratio"], epochs=epochs)
                completed += 1
                elapsed = time.time() - t0
                eta = (elapsed / max(1, completed - len(done))) * (total - completed) / 3600.0
                print(f"  [{completed}/{total}] {method:<11} "
                      f"{axis}={L['level']:<5}  run={run_idx:<2}  "
                      f"acc={acc:5.2f}  ({time.time()-t_run:5.1f}s)  ETA~{eta:.2f}h")
                results.append({
                    "method": method,
                    "axis": axis,
                    "level": L["level"],
                    "gamma": L["gamma"],
                    "noise_ratio": L["noise_ratio"],
                    "run_idx": run_idx,
                    "best_acc": round(acc, 2),
                    "history": hist,
                    "wall_sec": round(time.time() - t_run, 1),
                })
                with open(partial, "w") as f:
                    json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{axis}] DONE in {(time.time()-t0)/3600.0:.2f}h -> {out_path}")
    return results


def run_anchor(methods, output_dir, n_runs, epochs):
    """Run anchor methods at the single (gamma=0, noise_ratio=0) baseline."""
    out_path = os.path.join(output_dir, "anchor_results.json")
    partial = os.path.join(output_dir, "anchor_results_partial.json")

    if os.path.exists(partial):
        with open(partial) as f:
            results = json.load(f)
        done = {(r["method"], r["run_idx"]) for r in results}
        print(f"[anchor] resuming with {len(done)} runs already complete")
    else:
        results = []
        done = set()

    total = len(methods) * n_runs
    completed = len(done)
    t0 = time.time()
    print(f"[anchor] methods={methods}  cell=(gamma={ANCHOR_GAMMA}, noise_ratio={ANCHOR_NOISE_RATIO})  "
          f"n_runs={n_runs}  total={total}  remaining={total - completed}")

    for method in methods:
        for run_idx in range(n_runs):
            key = (method, run_idx)
            if key in done:
                continue
            t_run = time.time()
            acc, hist = run_trial(method, ANCHOR_GAMMA, ANCHOR_NOISE_RATIO, epochs=epochs)
            completed += 1
            elapsed = time.time() - t0
            eta = (elapsed / max(1, completed - len(done))) * (total - completed) / 3600.0
            print(f"  [{completed}/{total}] {method:<11} anchor    run={run_idx:<2}  "
                  f"acc={acc:5.2f}  ({time.time()-t_run:5.1f}s)  ETA~{eta:.2f}h")
            results.append({
                "method": method,
                "axis": "anchor",
                "level": 0.0,
                "gamma": ANCHOR_GAMMA,
                "noise_ratio": ANCHOR_NOISE_RATIO,
                "run_idx": run_idx,
                "best_acc": round(acc, 2),
                "history": hist,
                "wall_sec": round(time.time() - t_run, 1),
            })
            with open(partial, "w") as f:
                json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[anchor] DONE in {(time.time()-t0)/3600.0:.2f}h -> {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["af", "noise", "both"], default="both")
    ap.add_argument("--sweep_methods", default=",".join(SWEEP_METHODS),
                    help="methods that get the full AF/noise grid; subset of " + ",".join(METHODS_ALL))
    ap.add_argument("--anchor_methods", default=",".join(ANCHOR_METHODS),
                    help="methods that only run at the (gamma=0, noise=0) anchor; "
                         "subset of " + ",".join(METHODS_ALL))
    ap.add_argument("--skip_anchor", action="store_true",
                    help="skip the anchor pass (e.g., already run)")
    ap.add_argument("--skip_sweep", action="store_true",
                    help="skip the AF/noise sweep pass (anchor only)")
    ap.add_argument("--n_runs", type=int, default=N_RUNS_PER_CELL,
                    help="runs per (sweep) cell (default 10)")
    ap.add_argument("--anchor_runs", type=int, default=ANCHOR_RUNS,
                    help="runs per anchor method (default 10)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--output_dir", default="results/methods_af_noise")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny test: 1 run, 2 levels, 3 epochs")
    args = ap.parse_args()

    sweep_methods = [m.strip() for m in args.sweep_methods.split(",") if m.strip()]
    anchor_methods = [m.strip() for m in args.anchor_methods.split(",") if m.strip()]
    bad = [m for m in (sweep_methods + anchor_methods) if m not in METHODS_ALL]
    if bad:
        sys.exit(f"unknown methods: {bad}; choices: {METHODS_ALL}")

    if args.smoke:
        global GAMMAS, NOISE_RATIOS
        GAMMAS = [0.0, 2.0]
        NOISE_RATIOS = [0.0, 1.0]
        n_runs = 1
        anchor_runs = 1
        epochs = 3
        args.output_dir = args.output_dir + "_smoke"
    else:
        n_runs = args.n_runs
        anchor_runs = args.anchor_runs
        epochs = args.epochs

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[config] device={DEVICE}  axis={args.axis}  "
          f"sweep={sweep_methods} ({n_runs} runs/cell)  "
          f"anchor={anchor_methods} ({anchor_runs} runs)  epochs={epochs}")
    print(f"[config] bits={BITS}  rank={RANK}  TE={TE}  omega={OMEGA}  "
          f"lifetime_steps={LIFETIME_STEPS}  AB_LIFETIME={AB_LIFETIME:.4f}")
    print(f"[config] gammas={GAMMAS}")
    print(f"[config] noise_ratios={NOISE_RATIOS}")

    n_sweep_cells = len(sweep_methods) * (
        (len(GAMMAS) if args.axis in ("af", "both") else 0) +
        (len(NOISE_RATIOS) if args.axis in ("noise", "both") else 0))
    n_sweep_runs = 0 if args.skip_sweep else n_sweep_cells * n_runs
    n_anchor_runs = 0 if args.skip_anchor else len(anchor_methods) * anchor_runs
    print(f"[config] total runs: {n_sweep_runs} sweep + {n_anchor_runs} anchor = "
          f"{n_sweep_runs + n_anchor_runs}")

    manifest = {
        "bits": BITS, "rank": RANK, "transfer_every": TE, "omega": OMEGA,
        "lifetime_steps": LIFETIME_STEPS, "ab_lifetime_value": AB_LIFETIME,
        "epochs": epochs, "batch_size": BATCH_SIZE,
        "n_runs_per_sweep_cell": n_runs, "n_anchor_runs": anchor_runs,
        "gammas": GAMMAS, "noise_ratios": NOISE_RATIOS,
        "noise_template": NOISE_TEMPLATE,
        "hp_by_method": HP_BY_METHOD,
        "sweep_methods": sweep_methods, "anchor_methods": anchor_methods,
        "axis": args.axis,
        "anchor_cell": {"gamma": ANCHOR_GAMMA, "noise_ratio": ANCHOR_NOISE_RATIO},
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if not args.skip_anchor and anchor_methods:
        run_anchor(anchor_methods, args.output_dir, anchor_runs, epochs)

    if not args.skip_sweep and sweep_methods:
        if args.axis in ("af", "both"):
            run_axis("af", sweep_methods, args.output_dir, n_runs, epochs)
        if args.axis in ("noise", "both"):
            run_axis("noise", sweep_methods, args.output_dir, n_runs, epochs)


if __name__ == "__main__":
    main()

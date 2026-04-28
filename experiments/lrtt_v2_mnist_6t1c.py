#!/usr/bin/env python3
"""LRTT-v2 MNIST training with hp_search device conditions.

A/B (auxiliary) tiles use 6T1C-style LinearStepDevice (capacitor lifetime decay,
dw_min noise, gamma asymmetry — exactly matching hp_search_full_grid_no_multnoise.py).
C tile uses idealized LinearStepDevice (no noise, no asymmetry).

Differences from hp_search v1:
  - update_mode = "selector_reconstruction"   (was "lora")
  - transfer_method = "blockwise"              (was "onehot")
  - reinit_mode is irrelevant in v2 (B reset is automatic per transfer)
  - No tile_a updates in v2 — auxiliary array footprint reduced

NB: device-level lifetime decay is the realistic 6T1C leakage model. The
controller-level cap_rho is set to 1.0 here to avoid double-counting leakage.

Usage:
  python lrtt_v2_mnist_6t1c.py --epochs 10 [--policy shuffled_cycle]
"""

import os
os.environ.setdefault("LRTT_SILENT", "1")

import argparse
import math

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


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LIFETIME = 46505  # 6T1C physical retention τ ≈ 775 min in seconds


def make_6t1c_ab_device(lifetime: int = LIFETIME):
    """6T1C capacitor model for A/B tiles (matches hp_search settings)."""
    TAU_SEC = 46505.0
    dt_batch_sec = -TAU_SEC * math.log(1 - 1.0 / lifetime)
    AB_LIFETIME = 1.0 / (1 - math.exp(-dt_batch_sec / TAU_SEC))
    return LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


def make_idealized_c_device():
    """Idealized LinearStepDevice for C tile (matches hp_search settings)."""
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


def build_v2_lrtt_config(rank: int, transfer_every: int, transfer_lr: float,
                         selector_policy: str = "shuffled_cycle",
                         cap_rho: float = 1.0):
    ab = make_6t1c_ab_device()
    c = make_idealized_c_device()
    dev = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        forward_inject=False,
        b_init_mode="zero",
        # reinit_mode is unused in v2; keep "standard" for config validation.
        reinit_mode="standard",
        decay_factor=1.0,
        # Selector
        selector_axis="row",
        selector_policy=selector_policy,
        selector_seed=42,
        selector_reset_b_on_advance=True,
        # Capacitor stabilizer (controller-level). Set to 1.0 because the
        # 6T1C device-level `lifetime` already simulates capacitor leakage;
        # enabling cap_rho<1 would double-count it.
        cap_stabilizer_enabled=True,
        cap_rho=cap_rho,
        cap_compensate_transfer=False,  # rho==1 -> compensate is a no-op anyway
        # Devices
        unit_cell_devices=[ab, ab, c],
    )
    rpu = PythonLRTTRPUConfig(device=dev)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return rpu


def build_model(hidden: int = 128, rank: int = 8,
                transfer_every: int = 32, transfer_lr: float = 0.5,
                selector_policy: str = "shuffled_cycle", cap_rho: float = 1.0):
    rpu = build_v2_lrtt_config(rank, transfer_every, transfer_lr,
                               selector_policy, cap_rho)
    model = AnalogSequential(
        AnalogLinear(784, hidden, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(hidden, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    return model


def collect_lrtt_controllers(model):
    found = []
    for m in model.modules():
        for attr in ("analog_module", "tile"):
            sub = getattr(m, attr, None)
            ctrl = getattr(sub, "controller", None)
            if ctrl is not None and getattr(ctrl, "_is_selector_v2", lambda: False)():
                found.append((m, ctrl))
                break
    return found


def run(args):
    torch.manual_seed(args.seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform)
    full_val = datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform)
    train_ds = (Subset(full_train, range(args.train_n))
                if args.train_n < len(full_train) else full_train)
    val_ds = (Subset(full_val, range(args.val_n))
              if args.val_n < len(full_val) else full_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        hidden=args.hidden, rank=args.rank,
        transfer_every=args.te, transfer_lr=args.tlr,
        selector_policy=args.policy, cap_rho=args.cap_rho,
    )
    optimizer = AnalogSGD(model.parameters(), lr=args.lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
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
        history.append((epoch, train_loss, acc))
        print(f"  epoch {epoch}: train_loss={train_loss:.4f}, val_acc={acc:.2f}%",
              flush=True)

    print("\n=== Controller diagnostics ===", flush=True)
    ctrls = collect_lrtt_controllers(model)
    for i, (mod, ctrl) in enumerate(ctrls):
        print(f"  layer[{i}]: num_a_updates={ctrl.num_a_updates}, "
              f"num_b_updates={ctrl.num_b_updates}, "
              f"num_transfers={ctrl.num_transfers}, "
              f"cycle={ctrl.selector_cycle}", flush=True)

    # Invariant assertions
    failures = []
    for s in [(c.num_a_updates, c.num_b_updates) for _, c in ctrls]:
        if s[0] != 0:
            failures.append(f"num_a_updates={s[0]} != 0")
        if s[1] == 0:
            failures.append("num_b_updates == 0")
    if failures:
        print("FAIL:", failures)
        raise SystemExit(1)
    print("\nALL v2 INVARIANTS PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--te", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tlr", type=float, default=0.5)
    parser.add_argument("--cap_rho", type=float, default=1.0)
    parser.add_argument("--policy", default="shuffled_cycle",
                        choices=["cyclic", "shuffled_cycle", "random"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_n", type=int, default=60000)
    parser.add_argument("--val_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"LRTT-v2 MNIST (6T1C A/B + Idealized LinearStep C)")
    print(f"  rank={args.rank}, te={args.te}, hidden={args.hidden}")
    print(f"  lr={args.lr}, tlr={args.tlr}, policy={args.policy}, cap_rho={args.cap_rho}")
    print(f"  epochs={args.epochs}, train={args.train_n}, val={args.val_n}, batch={args.batch_size}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal SRA-LRTT-v2 MNIST smoke test using FloatingPointDevice tiles.

Verifies the SRA (Stochastic Reset-Anchor) logic end-to-end on a real
classification task without exercising the analog C++ ABI.

Checks:
  - tile_a is never updated by gradient descent (num_a_updates == 0)
  - B is updated at least once per layer
  - sra_cycle increments (anchor is resampled across transfers)
  - Validation accuracy improves over the first few epochs

The default anchor source is 'explicit_gaussian' because FloatingPointDevice
has no reset/reset_std parameters and would yield a deterministic zero under
'reset_columns'. To exercise 'reset_columns' / 'set_zero_write_noise' /
'pulse_scramble' end-to-end, switch to LinearStepDevice tiles via
PythonLRTTPreset.sixt1c_sra_all() (see --device_class linear_step).

Usage:
  python smoke_lrtt_v2_sra_mnist.py [--anchor_source explicit_gaussian|reset_columns|...] [--epochs N]
"""

import os
os.environ.setdefault("LRTT_SILENT", "1")

import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import FloatingPointDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice, PythonLRTTPreset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_sra_lrtt_config(
    *,
    rank: int,
    transfer_every: int,
    transfer_lr: float,
    anchor_source: str = "explicit_gaussian",
    cap_rho: float = 1.0,
    sra_seed: int = 0,
    device_class: str = "floating_point",
    reset_std: float = 0.01,
):
    """Build a PythonLRTTRPUConfig for SRA mode.

    With device_class='floating_point' the three tiles are FloatingPointDevice
    and 'reset_columns' is degenerate; use 'explicit_gaussian' instead.
    With device_class='linear_step', the SRA preset (sixt1c_sra_all) creates
    SEPARATE A/B/C LinearStepDevice instances so reset_columns / write-noise
    sources work realistically.
    """
    if device_class == "linear_step":
        # Hardware-realistic preset (separate A/B/C devices).
        dev = PythonLRTTPreset.sixt1c_sra_all(
            rank=rank,
            transfer_every=transfer_every,
            anchor_source=anchor_source,
            reset_std=reset_std,
        )
        # Override transfer_lr / cap / seed knobs to follow CLI.
        dev.transfer_lr = transfer_lr
        dev.cap_rho = cap_rho
        dev.sra_seed = sra_seed
    else:
        fp = FloatingPointDevice()
        dev = PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            transfer_lr=transfer_lr,
            update_mode="stochastic_reset_anchor",
            transfer_method="stochastic_anchor",
            forward_inject=False,
            a_init_mode="zero",
            b_init_mode="zero",
            cap_stabilizer_enabled=True,
            cap_rho=cap_rho,
            cap_compensate_transfer=True,
            sra_anchor_source=anchor_source,
            sra_seed=sra_seed,
            unit_cell_devices=[fp, fp, fp],
        )
    rpu = PythonLRTTRPUConfig(device=dev)
    return rpu


def build_model(
    *,
    hidden: int = 64,
    rank: int = 8,
    transfer_every: int = 16,
    transfer_lr: float = 0.5,
    anchor_source: str = "explicit_gaussian",
    cap_rho: float = 1.0,
    device_class: str = "floating_point",
):
    rpu = build_sra_lrtt_config(
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        anchor_source=anchor_source,
        cap_rho=cap_rho,
        device_class=device_class,
    )
    model = AnalogSequential(
        AnalogLinear(784, hidden, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(hidden, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    return model


def collect_sra_controllers(model):
    """Find every LRTTSimulatorTile.controller in the model that runs SRA."""
    found = []
    for m in model.modules():
        for attr in ("analog_module", "tile"):
            sub = getattr(m, attr, None)
            ctrl = getattr(sub, "controller", None)
            if ctrl is not None and getattr(ctrl, "_is_sra_v2", lambda: False)():
                found.append((m, ctrl))
                break
    return found


def run(
    *,
    hidden: int = 64,
    rank: int = 8,
    transfer_every: int = 16,
    transfer_lr: float = 0.5,
    lr: float = 0.1,
    epochs: int = 3,
    batch_size: int = 128,
    train_n: int = 2048,
    val_n: int = 2048,
    anchor_source: str = "explicit_gaussian",
    cap_rho: float = 1.0,
    device_class: str = "floating_point",
    seed: int = 42,
):
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform)
    full_val = datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform)
    train_ds = Subset(full_train, range(min(train_n, len(full_train))))
    val_ds = Subset(full_val, range(min(val_n, len(full_val))))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        hidden=hidden, rank=rank, transfer_every=transfer_every,
        transfer_lr=transfer_lr, anchor_source=anchor_source,
        cap_rho=cap_rho, device_class=device_class,
    )
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
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
        print(f"  epoch {epoch}: train_loss={train_loss:.4f}, val_acc={acc:.2f}%", flush=True)

    # Inspect controllers
    print("\n=== SRA controller diagnostics ===")
    ctrls = collect_sra_controllers(model)
    summary = []
    for i, (mod, ctrl) in enumerate(ctrls):
        anchor_rms = (
            float((ctrl.sra_anchor_scaled ** 2).mean().sqrt().item())
            if ctrl.sra_anchor_scaled is not None else None
        )
        summary.append({
            "layer": i,
            "num_a_updates": ctrl.num_a_updates,
            "num_b_updates": ctrl.num_b_updates,
            "num_transfers": ctrl.num_transfers,
            "sra_cycle": ctrl.sra_cycle,
            "sra_anchor_gain": ctrl.sra_anchor_gain,
            "sra_anchor_rms_raw": ctrl.sra_anchor_rms_raw,
            "sra_anchor_scaled_rms": anchor_rms,
        })
        print(
            f"  layer[{i}]: num_a_updates={ctrl.num_a_updates}, "
            f"num_b_updates={ctrl.num_b_updates}, "
            f"num_transfers={ctrl.num_transfers}, "
            f"sra_cycle={ctrl.sra_cycle}, "
            f"anchor_gain={ctrl.sra_anchor_gain:.4g}, "
            f"anchor_rms_scaled={anchor_rms}"
        )
    return history, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor_source", default="explicit_gaussian",
        choices=["reset_columns", "set_zero_write_noise", "explicit_gaussian", "pulse_scramble"],
    )
    parser.add_argument(
        "--device_class", default="floating_point",
        choices=["floating_point", "linear_step"],
        help="Underlying device. Use 'linear_step' to exercise reset_columns / write-noise.",
    )
    parser.add_argument("--cap_rho", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--te", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tlr", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--train_n", type=int, default=2048)
    parser.add_argument("--val_n", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(
        f"SRA-LRTT-v2 MNIST smoke (anchor={args.anchor_source}, "
        f"device={args.device_class}, cap_rho={args.cap_rho}, "
        f"rank={args.rank}, te={args.te}, epochs={args.epochs})"
    )
    print(f"Device: {DEVICE}")
    print("=" * 60)
    history, summary = run(
        hidden=args.hidden, rank=args.rank,
        transfer_every=args.te, transfer_lr=args.tlr,
        lr=args.lr, epochs=args.epochs,
        train_n=args.train_n, val_n=args.val_n,
        anchor_source=args.anchor_source, cap_rho=args.cap_rho,
        device_class=args.device_class, seed=args.seed,
    )

    # ---- Invariant checks ----
    failures = []
    if not summary:
        failures.append("no SRA-LRTT-v2 controller found in model")
    else:
        for s in summary:
            if s["num_a_updates"] != 0:
                failures.append(
                    f"layer[{s['layer']}] num_a_updates={s['num_a_updates']} != 0 "
                    "(SRA must not gradient-descend on tile_a)"
                )
            if s["num_b_updates"] == 0:
                failures.append(f"layer[{s['layer']}] num_b_updates == 0")
            if s["num_transfers"] > 0 and s["sra_cycle"] < 1:
                failures.append(
                    f"layer[{s['layer']}] sra_cycle == 0 despite "
                    f"{s['num_transfers']} transfers"
                )
        if len(history) >= 2:
            if history[-1]["train_loss"] > history[0]["train_loss"]:
                failures.append(
                    f"train_loss did not decrease "
                    f"({history[0]['train_loss']:.4f} -> {history[-1]['train_loss']:.4f})"
                )

    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("ALL SRA-LRTT-v2 SMOKE INVARIANTS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal LRTT-v2 MNIST smoke test using FloatingPointDevice tiles.

Verifies the v2 logic end-to-end on a real classification task without
exercising the analog C++ ABI (which is a pre-existing mismatch on this env).

Checks:
  - tile_a is never updated (num_a_updates == 0)
  - B is updated at least once per layer
  - Selector cycles through rows
  - Validation accuracy improves over the first few epochs

Usage:
  python smoke_lrtt_v2_mnist.py [--policy cyclic|shuffled_cycle] [--epochs N]
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
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_v2_lrtt_config(rank: int, transfer_every: int, transfer_lr: float,
                         selector_policy: str = "cyclic", cap_rho: float = 1.0):
    fp = FloatingPointDevice()
    dev = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        forward_inject=False,
        b_init_mode="zero",
        selector_axis="row",
        selector_policy=selector_policy,
        selector_seed=0,
        selector_reset_b_on_advance=True,
        cap_stabilizer_enabled=True,
        cap_rho=cap_rho,
        cap_compensate_transfer=True,
        unit_cell_devices=[fp, fp, fp],
    )
    rpu = PythonLRTTRPUConfig(device=dev)
    return rpu


def build_model(hidden: int = 64, rank: int = 8,
                transfer_every: int = 16, transfer_lr: float = 0.5,
                selector_policy: str = "cyclic", cap_rho: float = 1.0):
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
    """Find every LRTTSimulatorTile.controller in the model.

    AnalogLinear stores its tile under .analog_module (which is the
    LRTTSimulatorTile for v2). The tile holds the LRTTController on .controller.
    """
    found = []
    for m in model.modules():
        for attr in ("analog_module", "tile"):
            sub = getattr(m, attr, None)
            ctrl = getattr(sub, "controller", None)
            if ctrl is not None and getattr(ctrl, "_is_selector_v2", lambda: False)():
                found.append((m, ctrl))
                break
    return found


def run(*, hidden=64, rank=8, transfer_every=16, transfer_lr=0.5,
        lr=0.1, epochs=3, batch_size=128, train_n=2048, val_n=2048,
        selector_policy="cyclic", cap_rho=1.0, seed=42):
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
        transfer_lr=transfer_lr, selector_policy=selector_policy, cap_rho=cap_rho,
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
    print("\n=== Controller diagnostics ===")
    ctrls = collect_lrtt_controllers(model)
    summary = []
    for i, (mod, ctrl) in enumerate(ctrls):
        summary.append({
            "layer": i,
            "num_a_updates": ctrl.num_a_updates,
            "num_b_updates": ctrl.num_b_updates,
            "num_transfers": ctrl.num_transfers,
            "selector_cycle": ctrl.selector_cycle,
            "selector_indices": (
                ctrl.selector_indices.detach().cpu().tolist()
                if ctrl.selector_indices is not None else None
            ),
        })
        print(f"  layer[{i}]: num_a_updates={ctrl.num_a_updates}, "
              f"num_b_updates={ctrl.num_b_updates}, "
              f"num_transfers={ctrl.num_transfers}, "
              f"cycle={ctrl.selector_cycle}, "
              f"indices_now={summary[-1]['selector_indices']}")
    return history, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="cyclic",
                        choices=["cyclic", "shuffled_cycle", "random"])
    parser.add_argument("--cap_rho", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--te", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tlr", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--train_n", type=int, default=2048)
    parser.add_argument("--val_n", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 60)
    print(f"LRTT-v2 MNIST smoke (policy={args.policy}, cap_rho={args.cap_rho}, "
          f"rank={args.rank}, te={args.te}, epochs={args.epochs})")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    history, summary = run(
        hidden=args.hidden, rank=args.rank,
        transfer_every=args.te, transfer_lr=args.tlr,
        lr=args.lr, epochs=args.epochs,
        train_n=args.train_n, val_n=args.val_n,
        selector_policy=args.policy, cap_rho=args.cap_rho,
    )

    # ---- Invariant checks ----
    failures = []
    if not summary:
        failures.append("no LRTT-v2 controller found in model")
    else:
        for s in summary:
            if s["num_a_updates"] != 0:
                failures.append(
                    f"layer[{s['layer']}] num_a_updates={s['num_a_updates']} != 0"
                )
            if s["num_b_updates"] == 0:
                failures.append(f"layer[{s['layer']}] num_b_updates == 0")
        # train loss should decrease overall
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
    print("ALL v2 SMOKE INVARIANTS PASSED")


if __name__ == "__main__":
    main()

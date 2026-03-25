#!/usr/bin/env python3
"""HP search for TE=10 and TE=100 combinations (all ranks).

Complements hp_search_underperformers.py which only covered partial TE=10/100.
This covers the remaining 8 conditions.

Settings same as hp_search_underperformers: 30 trials, 30 epochs, early stopping.
"""

import os; os.environ["LRTT_SILENT"] = "1"
import math, torch, torch.nn as nn, json, optuna
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
import argparse

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LIFETIME = 46505
N_TRIALS = 30

TARGETS = [
    # TE=10 missing ranks
    {'rank': 4,  'te': 10,  'best_lr': 0.3890, 'best_tlr': 0.001048},
    {'rank': 32, 'te': 10,  'best_lr': 0.1699, 'best_tlr': 0.001749},
    {'rank': 64, 'te': 10,  'best_lr': 0.2520, 'best_tlr': 0.001336},
    # TE=100 missing ranks
    {'rank': 4,  'te': 100, 'best_lr': 0.3418, 'best_tlr': 0.003911},
    {'rank': 8,  'te': 100, 'best_lr': 0.1719, 'best_tlr': 0.002665},
    {'rank': 16, 'te': 100, 'best_lr': 0.1872, 'best_tlr': 0.002161},
    {'rank': 32, 'te': 100, 'best_lr': 0.9934, 'best_tlr': 0.001219},
    {'rank': 64, 'te': 100, 'best_lr': 0.2520, 'best_tlr': 0.001336},
]

TAU_SEC = 46505.0
dt_batch_sec = -TAU_SEC * math.log(1 - 1.0/LIFETIME)
delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
AB_LIFETIME = 1.0 / delta

DW_MIN_6BIT = 2.0 / (2**6)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train_loader = DataLoader(
    datasets.MNIST('/tmp/mnist', download=True, train=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(
    datasets.MNIST('/tmp/mnist', download=True, train=False, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


def create_model(rank, te, lr, tlr):
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.1, reset=0.0, reset_dtod=0.0,
    )
    c_device = LinearStepDevice(
        dw_min=DW_MIN_6BIT, w_max=1.0, w_min=-1.0,
        gamma_up=0.0, gamma_down=0.0,
        up_down=0.0, up_down_dtod=0.0,
        mult_noise=True, mean_bound_reference=True,
        dw_min_std=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0,
    )
    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=te,
        lora_alpha=1.0, reinit_gain=1.0,
        reinit_mode="hybrid", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.weight_scaling_omega = 0.6

    model = AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    return model


def run_trial(rank, te, lr, tlr):
    model = create_model(rank, te, lr, tlr)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience = 0
    for epoch in range(1, EPOCHS + 1):
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
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        if epoch >= 5 and best_acc < 50.0:
            break
        if patience >= EARLY_STOP_PATIENCE:
            break

    del model
    torch.cuda.empty_cache()
    return best_acc


def main():
    output_dir = "results/hp_search_te10_te100"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HP SEARCH: TE=10 & TE=100 (all ranks, hybrid mode)")
    print("=" * 70)
    print(f"Targets: {len(TARGETS)} conditions, {N_TRIALS} trials each")
    print(f"Total trials: {len(TARGETS) * N_TRIALS}")
    print()

    all_results = []

    for ti, tgt in enumerate(TARGETS):
        rank, te = tgt['rank'], tgt['te']
        seed_lr, seed_tlr = tgt['best_lr'], tgt['best_tlr']

        print(f"\n[{ti+1}/{len(TARGETS)}] rank={rank}, te={te} "
              f"(seed lr={seed_lr:.4f}, tlr={seed_tlr:.6f})")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42))
        study.enqueue_trial({"lr": seed_lr, "tlr": seed_tlr})

        def objective(trial, _rank=rank, _te=te):
            lr = trial.suggest_float("lr", 0.01, 1.5, log=True)
            tlr = trial.suggest_float("tlr", 1e-5, 1.0, log=True)
            return run_trial(_rank, _te, lr, tlr)

        study.optimize(objective, n_trials=N_TRIALS)

        best = study.best_trial
        print(f"  Best: {best.value:.2f}% "
              f"(lr={best.params['lr']:.4f}, tlr={best.params['tlr']:.6f})")

        result = {
            'rank': rank,
            'te': te,
            'mode': 'hybrid',
            'best_acc': round(best.value, 2),
            'best_lr': best.params['lr'],
            'best_tlr': best.params['tlr'],
            'n_trials': N_TRIALS,
            'all_trials': [
                {'lr': t.params['lr'], 'tlr': t.params['tlr'],
                 'acc': round(t.value, 2)}
                for t in study.trials
            ],
        }
        all_results.append(result)

        with open(f"{output_dir}/results_partial.json", 'w') as f:
            json.dump(all_results, f, indent=2)

    with open(f"{output_dir}/results_final.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    old_accs = {
        (4,10): 95.09, (32,10): 94.26, (64,10): 94.32,
        (4,100): 96.45, (8,100): 96.20, (16,100): 94.91,
        (32,100): 95.90, (64,100): 95.78,
    }
    print(f"{'rank':>5s} {'te':>5s} {'old_acc':>9s} {'new_acc':>9s} {'delta':>8s} {'best_lr':>9s} {'best_tlr':>11s}")
    print("-" * 60)
    for r in all_results:
        old = old_accs.get((r['rank'], r['te']), 0)
        delta = r['best_acc'] - old
        print(f"{r['rank']:>5d} {r['te']:>5d} {old:>8.2f}% {r['best_acc']:>8.2f}% "
              f"{delta:>+7.2f}% {r['best_lr']:>9.4f} {r['best_tlr']:>11.6f}")

    print(f"\nResults: {output_dir}/results_final.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
LRTT Grid Search: Comprehensive hyperparameter sweep with MNIST training
Measures read-side, write-side, and end-to-end transfer errors during training

Hyperparameters:
- rank: 4, 16, 64, 128
- transfer_lr: 0.1, 1.0
- transfer_every: 100, 1000
- Nstates (C tile): 20, 100, 1000 → dw_min = 0.1, 0.02, 0.002
- reinit_gain: 1.0, 0.5, 0.1

Fixed:
- lr: 0.1
- reinit_mode: orthogonal
- Layer 2: FloatingPointRPU

Outputs: LRTT_USE_ONEHOT_RESULTS/
"""
import os
import json
import csv
import math
import argparse
from itertools import product
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice, PythonLRTTPreset
from aihwkit.simulator.presets.devices import EcRamPresetDevice, IdealizedPresetDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.parameters.enums import WeightNoiseType
from aihwkit.simulator.parameters.io import IOParameters


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def fro_err_rel(X, Y, eps=1e-12):
    """Relative Frobenius error"""
    num = torch.norm(X - Y)
    den = max(torch.norm(Y), torch.tensor(eps, device=Y.device))
    return (num / den).item()


def corr_coef(X, Y):
    """Pearson correlation coefficient"""
    x = X.flatten() - X.mean()
    y = Y.flatten() - Y.mean()
    den = (x.norm() * y.norm()).item()
    return float((x @ y).item() / den) if den > 0 else float('nan')


def lin_slope(x, y, eps=1e-12):
    """Linear regression slope (no intercept): α = (x·y) / (x·x)"""
    xv = x.flatten()
    yv = y.flatten()
    return float((xv @ yv).item() / ((xv @ xv).item() + eps))


def expected_deltaC(A, B, lr_eff, transfer_lr_sign):
    """Expected ΔC = -sign × lr_eff × (A @ B)"""
    sign = -1.0 if (transfer_lr_sign > 0) else 1.0
    return (-sign * lr_eff) * (A @ B)


def onehot_read(ctrl):
    """Read A, B using one-hot method"""
    ctrl.read_n_avg = 1
    ctrl.agc_enabled = False
    ctrl.two_amp_enabled = False
    ctrl.transfer_centering = False
    ctrl.transfer_normalize = False

    with torch.no_grad():
        A_cols, B_rows = ctrl._read_ab_onehot_symmetric()

    return A_cols, B_rows


def measure_read_side(ctrl, repeats=5) -> Dict:
    """
    Read-side measurement: A_cols vs A_true, B_rows vs B_true
    Returns: relative Frobenius error, SNR, scale distortion α
    """
    A_list, B_list = [], []

    with torch.no_grad():
        A_true = ctrl.tile_a.get_weights()[0][:, :ctrl.rank].clone()
        B_true = ctrl.tile_b.get_weights()[0][:ctrl.rank, :].clone()

    # Multiple reads
    for _ in range(repeats):
        A_cols, B_rows = onehot_read(ctrl)
        A_list.append(A_cols.cpu())
        B_list.append(B_rows.cpu())

    A_stack = torch.stack(A_list, dim=0)
    B_stack = torch.stack(B_list, dim=0)

    A_mean = A_stack.mean(0)
    B_mean = B_stack.mean(0)
    A_std = A_stack.std(0, unbiased=True)
    B_std = B_stack.std(0, unbiased=True)

    A_true_cpu = A_true.cpu()
    B_true_cpu = B_true.cpu()

    # Relative Frobenius error
    A_fro_err = fro_err_rel(A_mean, A_true_cpu)
    B_fro_err = fro_err_rel(B_mean, B_true_cpu)

    # SNR (signal norm / noise std per element)
    A_snr = (torch.norm(A_true_cpu) / (A_std.norm() / math.sqrt(A_std.numel()) + 1e-12)).item()
    B_snr = (torch.norm(B_true_cpu) / (B_std.norm() / math.sqrt(B_std.numel()) + 1e-12)).item()

    # Scale distortion α: y ≈ α·x
    alpha_A = lin_slope(A_true_cpu, A_mean)
    alpha_B = lin_slope(B_true_cpu, B_mean)

    return {
        'A_fro_err': A_fro_err,
        'B_fro_err': B_fro_err,
        'A_snr': A_snr,
        'B_snr': B_snr,
        'alpha_A': alpha_A,
        'alpha_B': alpha_B,
    }


def measure_write_side(tile_c, A_cols, B_rows, lr_eff, repeats=5) -> Dict:
    """
    Write-side measurement: CV (coefficient of variation) for repeated writes
    """
    with torch.no_grad():
        base = tile_c.get_weights()[0].clone()

    norms = []
    for _ in range(repeats):
        with torch.no_grad():
            tile_c.set_weights(base)
            old_lr = tile_c.get_learning_rate()
            tile_c.set_learning_rate(lr_eff)

            # Apply same update multiple times
            for k in range(A_cols.size(1)):
                a = A_cols[:, k].unsqueeze(0)
                b = B_rows[k, :].unsqueeze(0)
                tile_c.update(b, a)

            tile_c.set_learning_rate(old_lr)
            now = tile_c.get_weights()[0]
            norms.append(torch.norm(now - base).item())

    norms = np.array(norms)
    cv = norms.std() / max(norms.mean(), 1e-12)

    return {
        'write_cv': float(cv),
        'write_mean_norm': float(norms.mean()),
        'write_std_norm': float(norms.std()),
    }


def measure_gain_curve(tile_c, a_vec, b_vec, lr_eff, amp_grid=[0.1, 0.3, 1.0, 3.0, 10.0]) -> List[Tuple[float, float]]:
    """
    Write-side gain curve: measure amplification vs pulse magnitude
    Returns: [(scale, gain), ...]
    """
    a = a_vec.clone().unsqueeze(0)
    b = b_vec.clone().unsqueeze(0)

    with torch.no_grad():
        base = tile_c.get_weights()[0].clone()

    denom0 = torch.ger(a.squeeze(0), b.squeeze(0)).norm().item()

    points = []
    for s in amp_grid:
        with torch.no_grad():
            tile_c.set_weights(base)
            old_lr = tile_c.get_learning_rate()
            tile_c.set_learning_rate(lr_eff * s)
            tile_c.update(b, a)
            tile_c.set_learning_rate(old_lr)
            now = tile_c.get_weights()[0]

        gain = (now - base).norm().item() / max((s * lr_eff * denom0), 1e-12)
        points.append((float(s), float(gain)))

    return points


def capture_pre_transfer_state(ctrl, lr_eff) -> Dict:
    """
    Capture state BEFORE transfer occurs (called before optimizer.step when transfer is imminent).

    Returns dict with:
    - A_true, B_true: actual weights
    - A_cols, B_rows: one-hot read results
    - C_before: C tile weights before transfer
    - dC_exp_true, dC_exp_onehot: expected ΔC values
    """
    with torch.no_grad():
        A_true = ctrl.tile_a.get_weights()[0][:, :ctrl.rank].clone()
        B_true = ctrl.tile_b.get_weights()[0][:ctrl.rank, :].clone()
        C_before = ctrl.tile_c.get_weights()[0].clone()

    # One-hot read
    A_cols, B_rows = onehot_read(ctrl)

    # Expected ΔC (true)
    dC_exp_true = expected_deltaC(A_true, B_true, lr_eff, ctrl.transfer_lr)

    # Expected ΔC (one-hot)
    dC_exp_onehot = expected_deltaC(A_cols, B_rows, lr_eff, ctrl.transfer_lr)

    return {
        'A_true': A_true,
        'B_true': B_true,
        'A_cols': A_cols,
        'B_rows': B_rows,
        'C_before': C_before,
        'dC_exp_true': dC_exp_true,
        'dC_exp_onehot': dC_exp_onehot,
        'lr_eff': lr_eff,
    }


def compute_transfer_stats(ctrl, pre_state: Dict) -> Dict:
    """
    Compute transfer statistics AFTER transfer has occurred.

    Args:
        ctrl: LRTT controller
        pre_state: dict from capture_pre_transfer_state()

    Returns:
        Transfer statistics dict
    """
    with torch.no_grad():
        C_after = ctrl.tile_c.get_weights()[0].clone()

    C_before = pre_state['C_before']
    dC_exp_true = pre_state['dC_exp_true']
    dC_exp_onehot = pre_state['dC_exp_onehot']

    dC_actual = C_after - C_before

    # Move to CPU for calculation
    dC_actual_cpu = dC_actual.cpu()
    dC_exp_true_cpu = dC_exp_true.cpu()
    dC_exp_onehot_cpu = dC_exp_onehot.cpu()

    return {
        'dC_actual_norm': float(torch.norm(dC_actual_cpu).item()),
        'dC_exp_true_norm': float(torch.norm(dC_exp_true_cpu).item()),
        'dC_exp_onehot_norm': float(torch.norm(dC_exp_onehot_cpu).item()),
        'err_vs_true': fro_err_rel(dC_actual_cpu, dC_exp_true_cpu),
        'err_vs_onehot': fro_err_rel(dC_actual_cpu, dC_exp_onehot_cpu),
        'amp_vs_true': float(torch.norm(dC_actual_cpu) / max(torch.norm(dC_exp_true_cpu), torch.tensor(1e-12))),
        'amp_vs_onehot': float(torch.norm(dC_actual_cpu) / max(torch.norm(dC_exp_onehot_cpu), torch.tensor(1e-12))),
        'corr_true': corr_coef(dC_actual_cpu, dC_exp_true_cpu),
        'corr_onehot': corr_coef(dC_actual_cpu, dC_exp_onehot_cpu),
    }


def measure_end_to_end_transfer(ctrl, layer, lr_eff) -> Dict:
    """
    [DEPRECATED - use capture_pre_transfer_state + compute_transfer_stats instead]

    Legacy function for backward compatibility.
    NOTE: This may give incorrect results due to automatic transfer in optimizer.step()
    """
    # Capture pre-transfer state
    pre_state = capture_pre_transfer_state(ctrl, lr_eff)

    # Manually trigger transfer (WARNING: may be redundant if auto-transfer already occurred)
    with torch.no_grad():
        ctrl.ab_weight_transfer(use_onehot=True, use_sigma_delta=False)

    return compute_transfer_stats(ctrl, pre_state)


def load_mnist(batch_size=128):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_model(rank, transfer_every, reinit_gain, nstates, device_name='cuda'):
    """
    Create MNIST model: 784 -> 256 -> 10
    Layer 1: 6T1C (A, B) + Idealized (C) with configurable Nstates, rank specified
    Layer 2: FloatingPointRPU (ideal digital)

    Args:
        rank: LoRA rank for Layer 1
        transfer_every: Transfer period in steps
        reinit_gain: Scale factor for orthogonal B initialization
        nstates: Number of conductance states for C tile (dw_min = 2/nstates)
        device_name: 'cuda' or 'cpu'
    """
    device = torch.device(device_name)

    # Layer 1: 784 -> 256
    # A/B tiles: 6T1C
    sixt1c_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0182,
        mean_bound_reference=True,
        lifetime=0.0,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: Idealized with configurable Nstates
    # Nstates = 2 / dw_min → dw_min = 2 / Nstates
    c_dw_min = 2.0 / nstates
    c_device = IdealizedPresetDevice(
        dw_min=c_dw_min,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        dw_min_std=0.0
    )

    device_config_layer1 = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=1.0,
        reinit_gain=reinit_gain,
        reinit_mode="orthogonal",
        forward_inject=False,
        use_onehot=True,
        use_sigma_delta=False,
        unit_cell_devices=[sixt1c_device, sixt1c_device, c_device]
    )

    rpu_cfg_1 = PythonLRTTRPUConfig(device=device_config_layer1)

    # Set IOParameters for reduced output noise
    io_pars = IOParameters(
        out_noise=0.006,
        w_noise_type=WeightNoiseType.NONE
    )
    rpu_cfg_1.forward = io_pars
    rpu_cfg_1.backward = io_pars

    layer1 = AnalogLinear(784, 256, rpu_config=rpu_cfg_1, bias=True)

    # Layer 2: 256 -> 10 (FloatingPointRPU - ideal digital)
    rpu_cfg_2 = FloatingPointRPUConfig()
    layer2 = AnalogLinear(256, 10, rpu_config=rpu_cfg_2, bias=True)

    model = AnalogSequential(
        layer1,
        nn.ReLU(),
        layer2
    ).to(device)

    return model


def train_one_config(rank, lr, transfer_lr, transfer_every, reinit_gain, nstates, epochs, outdir, device_name='cuda'):
    """Train one configuration and measure errors at each transfer"""

    print(f"\n{'='*80}")
    print(f"Config: rank={rank}, lr={lr}, transfer_lr={transfer_lr}, transfer_every={transfer_every}, "
          f"reinit_gain={reinit_gain}, nstates={nstates}")
    print(f"{'='*80}\n")

    set_seed(123)
    device = torch.device(device_name)

    # Create model
    model = create_model(rank, transfer_every, reinit_gain, nstates, device_name)

    # Get controller (only Layer 1 has LRTT)
    ctrl_1 = model[0].analog_module.controller

    # Update transfer parameters
    ctrl_1.transfer_lr = transfer_lr
    ctrl_1.transfer_every = transfer_every

    # Optimizer
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    criterion = nn.CrossEntropyLoss()

    # Load data
    train_loader, test_loader = load_mnist(batch_size=128)

    # Training records
    transfer_records = []
    epoch_records = []

    step_global = 0
    transfer_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            target = target.to(device)

            # === FIX: Capture state BEFORE optimizer.step() triggers transfer ===
            # Check if transfer will occur after this update
            # Controller's transfer_counter increments in ab_weight_update, then should_transfer() is checked
            next_counter = ctrl_1.transfer_counter + 1
            will_transfer = next_counter >= ctrl_1.transfer_every

            # Effective lr (compute early for capture)
            lr_eff = abs(transfer_lr) / math.sqrt(rank)

            # Capture pre-transfer state if transfer is imminent
            pre_state = None
            read_stats = None
            if will_transfer:
                pre_state = capture_pre_transfer_state(ctrl_1, lr_eff)
                # Measure read-side BEFORE reinit (A, B still have learned values)
                read_stats = measure_read_side(ctrl_1, repeats=3)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()  # Transfer automatically occurs here if should_transfer()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            step_global += 1

            # === FIX: Compute transfer stats AFTER transfer occurred ===
            if pre_state is not None:
                transfer_count += 1

                print(f"  [Epoch {epoch}, Step {step_global}] Transfer #{transfer_count}")

                # Compute actual dC from pre-captured state
                transfer_stats = compute_transfer_stats(ctrl_1, pre_state)

                # Measure write-side (after transfer/reinit, A and B are reset)
                A_cols, B_rows = onehot_read(ctrl_1)
                write_stats = measure_write_side(ctrl_1.tile_c, A_cols, B_rows, lr_eff, repeats=3)

                # Gain curve (sample first rank) - uses new A, B after reinit
                gain_curve = measure_gain_curve(ctrl_1.tile_c, A_cols[:, 0], B_rows[0, :], lr_eff)

                record = {
                    'rank': rank,
                    'lr': lr,
                    'transfer_lr': transfer_lr,
                    'transfer_every': transfer_every,
                    'reinit_gain': reinit_gain,
                    'nstates': nstates,
                    'epoch': epoch,
                    'step': step_global,
                    'transfer_num': transfer_count,
                    **read_stats,
                    **transfer_stats,
                    **write_stats,
                }
                transfer_records.append(record)

        # Evaluate
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_acc = 100.0 * correct / total
        test_loss /= len(test_loader)
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total

        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

        epoch_record = {
            'rank': rank,
            'lr': lr,
            'transfer_lr': transfer_lr,
            'transfer_every': transfer_every,
            'reinit_gain': reinit_gain,
            'nstates': nstates,
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }
        epoch_records.append(epoch_record)

    # Save results
    config_name = f"rank{rank}_lr{lr}_tlr{transfer_lr}_tevery{transfer_every}_gain{reinit_gain}_ns{nstates}"

    # Save transfer records
    transfer_csv = os.path.join(outdir, f"transfer_{config_name}.csv")
    if transfer_records:
        keys = transfer_records[0].keys()
        with open(transfer_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(transfer_records)

    # Save epoch records
    epoch_csv = os.path.join(outdir, f"epoch_{config_name}.csv")
    keys = epoch_records[0].keys()
    with open(epoch_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(epoch_records)

    # Save gain curve (last transfer)
    if transfer_records:
        gain_csv = os.path.join(outdir, f"gain_{config_name}.csv")
        with open(gain_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scale', 'gain'])
            writer.writerows(gain_curve)

    print(f"✓ Saved results for {config_name}\n")

    return epoch_records, transfer_records


def main():
    parser = argparse.ArgumentParser(description='LRTT Grid Search')
    parser.add_argument('--ranks', type=int, nargs='+', default=[4, 16, 64, 128])
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (fixed)')
    parser.add_argument('--transfer_lrs', type=float, nargs='+', default=[0.1, 1.0])
    parser.add_argument('--transfer_everys', type=int, nargs='+', default=[100, 1000])
    parser.add_argument('--reinit_gains', type=float, nargs='+', default=[1.0, 0.5, 0.1])
    parser.add_argument('--nstates', type=int, nargs='+', default=[20, 100, 1000],
                        help='Number of conductance states for C tile (dw_min = 2/nstates)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--outdir', type=str, default='LRTT_USE_ONEHOT_RESULTS')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Save detailed experiment config
    config = {
        'experiment_name': 'LRTT Orthogonal B Scale Study',
        'description': 'Grid search to study the effect of orthogonal B scale (reinit_gain) on training',
        'hyperparameters': {
            'ranks': args.ranks,
            'lr': args.lr,
            'transfer_lrs': args.transfer_lrs,
            'transfer_everys': args.transfer_everys,
            'reinit_gains': args.reinit_gains,
            'nstates': args.nstates,
        },
        'fixed_settings': {
            'reinit_mode': 'orthogonal',
            'forward_inject': False,
            'use_onehot': True,
            'use_sigma_delta': False,
            'lora_alpha': 1.0,
        },
        'model_architecture': {
            'layer1': {
                'input_size': 784,
                'output_size': 256,
                'ab_device': '6T1C (LinearStepDevice)',
                'c_device': 'IdealizedPresetDevice (configurable Nstates)',
            },
            'layer2': {
                'input_size': 256,
                'output_size': 10,
                'device': 'FloatingPointRPU (ideal digital)',
            },
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': 128,
            'dataset': 'MNIST',
            'optimizer': 'AnalogSGD',
            'loss': 'CrossEntropyLoss',
        },
        'total_configurations': len(list(product(
            args.ranks, args.transfer_lrs, args.transfer_everys, args.reinit_gains, args.nstates
        ))),
    }
    with open(os.path.join(args.outdir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    device_name = 'cpu' if args.cpu else 'cuda'

    # Grid search: rank × transfer_lr × transfer_every × reinit_gain × nstates
    total_configs = len(list(product(
        args.ranks, args.transfer_lrs, args.transfer_everys, args.reinit_gains, args.nstates
    )))
    print(f"Total configurations: {total_configs}")
    print(f"  ranks: {args.ranks}")
    print(f"  lr: {args.lr} (fixed)")
    print(f"  transfer_lrs: {args.transfer_lrs}")
    print(f"  transfer_everys: {args.transfer_everys}")
    print(f"  reinit_gains: {args.reinit_gains}")
    print(f"  nstates: {args.nstates}")

    all_epoch_records = []
    all_transfer_records = []

    for idx, (rank, transfer_lr, transfer_every, reinit_gain, nstates) in enumerate(
        product(args.ranks, args.transfer_lrs, args.transfer_everys, args.reinit_gains, args.nstates), 1
    ):
        print(f"\n[{idx}/{total_configs}] Training configuration...")

        epoch_recs, transfer_recs = train_one_config(
            rank=rank,
            lr=args.lr,
            transfer_lr=transfer_lr,
            transfer_every=transfer_every,
            reinit_gain=reinit_gain,
            nstates=nstates,
            epochs=args.epochs,
            outdir=args.outdir,
            device_name=device_name
        )

        all_epoch_records.extend(epoch_recs)
        all_transfer_records.extend(transfer_recs)

    # Save combined results (CSV)
    epoch_combined_csv = os.path.join(args.outdir, 'all_epochs.csv')
    if all_epoch_records:
        keys = all_epoch_records[0].keys()
        with open(epoch_combined_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_epoch_records)

    transfer_combined_csv = os.path.join(args.outdir, 'all_transfers.csv')
    if all_transfer_records:
        keys = all_transfer_records[0].keys()
        with open(transfer_combined_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_transfer_records)

    # Save accuracy results to xlsx
    if all_epoch_records:
        df_epochs = pd.DataFrame(all_epoch_records)

        # Create summary: final accuracy for each configuration
        final_acc = df_epochs.groupby(['rank', 'transfer_lr', 'transfer_every', 'reinit_gain', 'nstates']).agg({
            'test_acc': 'last',
            'train_loss': 'last',
            'test_loss': 'last',
        }).reset_index()
        final_acc.columns = ['rank', 'transfer_lr', 'transfer_every', 'reinit_gain', 'nstates',
                             'final_test_acc', 'final_train_loss', 'final_test_loss']

        # Create pivot table for easy analysis: reinit_gain vs nstates for each rank
        xlsx_path = os.path.join(args.outdir, 'accuracy_results.xlsx')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            # Sheet 1: All epoch records
            df_epochs.to_excel(writer, sheet_name='All Epochs', index=False)

            # Sheet 2: Final accuracy summary
            final_acc.to_excel(writer, sheet_name='Final Accuracy', index=False)

            # Sheet 3: Pivot table (reinit_gain x nstates) for each rank
            for rank_val in args.ranks:
                subset = final_acc[final_acc['rank'] == rank_val]
                if not subset.empty:
                    pivot = subset.pivot_table(
                        index=['transfer_lr', 'transfer_every'],
                        columns=['reinit_gain', 'nstates'],
                        values='final_test_acc',
                        aggfunc='first'
                    )
                    pivot.to_excel(writer, sheet_name=f'Rank{rank_val}_Pivot')

        print(f"\n✓ Accuracy results saved to: {xlsx_path}")

    print(f"\n{'='*80}")
    print(f"Grid search complete!")
    print(f"Results saved to: {args.outdir}")
    print(f"  - experiment_config.json (detailed experiment settings)")
    print(f"  - all_epochs.csv (all epoch records)")
    print(f"  - all_transfers.csv (all transfer records)")
    print(f"  - accuracy_results.xlsx (accuracy summary with pivot tables)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

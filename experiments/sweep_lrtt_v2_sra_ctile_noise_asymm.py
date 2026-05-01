#!/usr/bin/env python3
"""SRA-LRTT-v2 C-tile AF × update-noise sweep (Tiki-Taka / LRTT defense convention).

Sweep axes (C-tile only; updated noise policy):
    AF ratio              → up_down       = 0.05 * af_ratio   (signed asymmetry)
    UPDATE-NOISE ratio    → dw_min_std    = 0.3  * unr        (cycle-to-cycle pulse)

    Defaults:
        AF_RATIO_GRID           = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        UPDATE_NOISE_RATIO_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]
    The (AF=0, UNR=0) baseline cell is excluded by default
    (use --include_baseline to add it as the ideal-C reference).

Held FIXED at 6T1C nominal (NOT scaled by either sweep axis):
    write_noise_std  = 0.0182              (apparent state perturbation)
    dw_min_dtod      = 0.1, w_*_dtod=0.05, gamma_*_dtod=0.05, up_down_dtod=0.01

Other fixed:
    RANK=8, TE=10, LR=0.11577, LIFETIME=1000, EPOCHS=30, BATCH_SIZE=64,
    HIDDEN=256, weight_scaling_omega=0.6.
    C tile precision = 8 bits (dw_min = 2/2^8 = 0.0078125), LinearStepDevice 6T1C.
    A,B = nominal 6T1C LinearStepDevice (mult_noise=True).
    update_mode='stochastic_reset_anchor', transfer_method='stochastic_anchor'.

SRA-specific fixed (placeholder; tune via a separate nominal HP search):
    sigma_A (= sra_anchor_target_rms) = 0.10
    transfer_lr_base                   = 1e-3        # before σ_A^2 compensation
    Effective transfer LR = transfer_lr_base / (rank * sigma_A^2) ≈ 0.0125

Usage:
    python sweep_lrtt_v2_sra_ctile_noise_asymm.py
    python sweep_lrtt_v2_sra_ctile_noise_asymm.py --include_baseline --runs_per_config 3
    python sweep_lrtt_v2_sra_ctile_noise_asymm.py --tlr_base 5e-4 --sigma_A 0.06
    python sweep_lrtt_v2_sra_ctile_noise_asymm.py --af_grid 0 1 2 5 --update_noise_grid 0 1 2 3
"""

import os
os.environ["LRTT_SILENT"] = "1"

import argparse
import json
import math
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Fixed HP — UNIFIED with sweep_ctile_bits_gamma.py (rank=8, TE=10, ...)
# =============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
HIDDEN = 256

RANK = 8                       # FIXED rank (matches recent C-sweep)
TE = 10                        # FIXED transfer_every (matches recent C-sweep)
LR = 0.11577                   # FIXED outer LR (matches sweep_ctile_bits_gamma.py)
LIFETIME = 1000                # FIXED 6T1C lifetime (LRTT-v2 standard baseline,
                               # matches run_lrtt_v2_lt1000_* family)
TAU_SEC = 46505.0              # physical capacitor time constant (s)

# AB_LIFETIME is the per-step decay constant fed to LinearStepDevice. The v2
# baseline (run_lrtt_v2_lt1000_*) uses `lifetime=LIFETIME` directly without
# the dt round-trip, so we mirror that here.
AB_LIFETIME = float(LIFETIME)

# C tile precision (FIXED at 8 bits, matches sweep_ctile_bits_gamma.py @ bits=8)
C_BITS = 8
C_DW_MIN = 2.0 / (2 ** C_BITS)  # = 0.0078125

# =============================================================================
# Sweep policy (Tiki-Taka / LRTT canonical, defense-grade):
#   - Main UPDATE noise axis = `dw_min_std` ONLY (cycle-to-cycle pulse variation).
#     write_noise_std is held FIXED (apparent-state perturbation, not the
#     update-path stochasticity we want to characterize).
#   - Asymmetry axis = `up_down` ONLY (signed). D2D variation params
#     (dw_min_dtod, w_*_dtod, gamma_*_dtod, up_down_dtod) are HELD FIXED at
#     6T1C nominal so the heatmap reflects only the two main mechanisms.
#   - 6T1C nominal anchor: dw_min=0.001981, dw_min_std=0.3, write_noise_std=0.0182,
#     dw_min_dtod=0.1, w_*_dtod=0.05, gamma_*_dtod=0.05, up_down_dtod=0.01.
# =============================================================================
BASE_DW_MIN_STD = 0.3            # 6T1C nominal cycle-to-cycle pulse variation
BASE_AF_TO_UP_DOWN = 0.05        # AF=1.0 ⇔ up_down=0.05 (5% asymmetry)
WRITE_NOISE_STD_FIXED = 0.0182   # 6T1C nominal apparent write noise (FIXED, not swept)

# C-tile sweep axes (Lee et al. / Rasch et al. defense-grade convention)
AF_RATIO_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]      # asymmetry ratio
UPDATE_NOISE_RATIO_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]  # dw_min_std ratio


# =============================================================================
# sigma_A handling (review §6/§7)
# =============================================================================

def sigma_A_to_target_rms(sigma_A: float) -> float:
    """Pass σ_A directly to sra_anchor_target_rms.

    σ_A == 0 IS allowed: it is the negative control (A_q := 0  ⇒  A_q^T G = 0).
    The controller's _sra_cache_anchor handles target_rms <= 0 by short-circuit
    (gain := 0 ⇒ A_scaled := 0), so no learning signal flows through B.
    """
    return float(sigma_A)


def sigma_A_compensation_scale(sigma_A: float, rank: int) -> float:
    """transfer_lr_scale = 1/(r*sigma_A^2) absorbs sigma_A^2 effective-LR dependence."""
    if sigma_A <= 0:
        return 1.0
    return 1.0 / (rank * sigma_A * sigma_A)


# =============================================================================
# Device builders
# =============================================================================

def make_ab_device_nominal():
    """6T1C LinearStepDevice nominal (matches sweep_noise_asymmetry.py exactly).
    mult_noise=True, write_noise_std=0.0; only the SRA-required reset_std=0.01
    is added so reset_columns produces a non-degenerate anchor.
    """
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=True,                    # MATCH sweep_noise_asymmetry.py
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        apply_write_noise_on_set=False,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.1,
        reset=0.0, reset_std=0.01, reset_dtod=0.0,
    )


def make_c_device_6t1c_8bit_swept(*, af_ratio: float, update_noise_ratio: float):
    """C tile: LinearStepDevice 6T1C parameters at 8-bit precision.

    Sweep mapping (Tiki-Taka / LRTT defense-grade convention):
        up_down     = BASE_AF_TO_UP_DOWN * af_ratio          # signed asymmetry
        dw_min_std  = BASE_DW_MIN_STD    * update_noise_ratio # cycle-to-cycle update
                                                              # variation (main update noise)

    Held FIXED at 6T1C nominal (NOT scaled by either sweep axis):
        write_noise_std  = WRITE_NOISE_STD_FIXED  (apparent-state noise)
        dw_min_dtod      = 0.1   (D2D update-step variation)
        up_down_dtod     = 0.01  (D2D asymmetry variation)
        w_max_dtod       = 0.05  (D2D bound variation)
        w_min_dtod       = 0.05
        gamma_up_dtod    = 0.05  (D2D gamma variation)
        gamma_down_dtod  = 0.05

    Only the dw_min is pinned at 8 bits (= 2/2^8 = 0.0078125) per user spec
    so that ALL methods in the comparison share the same C-tile precision.
    """
    return LinearStepDevice(
        dw_min=C_DW_MIN, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=True,
        # Main asymmetry axis (signed)
        up_down=BASE_AF_TO_UP_DOWN * float(af_ratio),
        # FIXED 6T1C D2D variations
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        # Main update-noise axis (cycle-to-cycle pulse variation)
        dw_min_std=BASE_DW_MIN_STD * float(update_noise_ratio),
        # FIXED apparent write noise (NOT scaled with sweep axes)
        write_noise_std=WRITE_NOISE_STD_FIXED,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )


# =============================================================================
# Model builder (SRA mode + C-tile asymm/noise; AB nominal 6T1C)
# =============================================================================

def create_model(*, sigma_A: float, transfer_lr_base: float,
                 af_ratio: float, update_noise_ratio: float):
    """SRA-LRTT-v2 model with FIXED HP, AB-tile nominal 6T1C, and C-tile
    sweep on (af_ratio × update_noise_ratio).

    af_ratio        → up_down       (signed asymmetry; main asymm axis)
    update_noise_ratio → dw_min_std (cycle-to-cycle pulse update variation;
                                     main update-noise axis)
    """
    a_device = make_ab_device_nominal()
    b_device = make_ab_device_nominal()
    c_device = make_c_device_6t1c_8bit_swept(
        af_ratio=af_ratio, update_noise_ratio=update_noise_ratio,
    )

    target_rms = sigma_A_to_target_rms(sigma_A)
    tlr_scale = sigma_A_compensation_scale(sigma_A, RANK)

    device_config = PythonLRTTDevice(
        rank=RANK,
        transfer_every=TE,
        transfer_lr=float(transfer_lr_base),
        transfer_lr_scale=float(tlr_scale),
        lora_alpha=1.0,
        reinit_gain=0.0,
        reinit_mode="standard",
        decay_factor=1.0,
        a_init_mode="zero",
        b_init_mode="zero",
        forward_inject=False,
        update_mode="stochastic_reset_anchor",
        transfer_method="stochastic_anchor",
        cap_stabilizer_enabled=True,
        cap_rho=1.0,
        cap_compensate_transfer=True,
        sra_anchor_source="reset_columns",
        sra_anchor_target_rms=target_rms,
        sra_anchor_gain_max=1.0e3,
        sra_b_reset_mode="set_zero",
        sra_seed=42,
        unit_cell_devices=[a_device, b_device, c_device],
    )
    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.weight_scaling_omega = 0.6

    model = AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(HIDDEN, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)
    return model


# =============================================================================
# Data
# =============================================================================

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST('/tmp/mnist', download=True, train=True, transform=transform)
    val_set = datasets.MNIST('/tmp/mnist', download=True, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


# =============================================================================
# Training / validation primitives
# =============================================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for data, target in loader:
        data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
        target = target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()


def validate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            correct += model(data).argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / max(1, total)


def run_training(*, sigma_A, transfer_lr_base, af_ratio, update_noise_ratio,
                 run_id, train_loader, val_loader, use_wandb):
    model = create_model(sigma_A=sigma_A, transfer_lr_base=transfer_lr_base,
                         af_ratio=af_ratio, update_noise_ratio=update_noise_ratio)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    af_label = f"AF{af_ratio:g}"
    ns_label = f"UNR{update_noise_ratio:g}"
    run_name = f"sra_r{RANK}_te{TE}_{af_label}_{ns_label}_run{run_id}"

    if use_wandb:
        wandb.init(
            project="lrtt-sra-ctile-af-update_noise-sweep",
            name=run_name,
            tags=[f"rank={RANK}", f"te={TE}", af_label, ns_label,
                  f"sigma_A={sigma_A}"],
            config={
                "rank": RANK, "te": TE, "lr": LR,
                "sigma_A": sigma_A, "transfer_lr_base": transfer_lr_base,
                "tlr_scale": sigma_A_compensation_scale(sigma_A, RANK),
                "effective_tlr": transfer_lr_base * sigma_A_compensation_scale(sigma_A, RANK),
                "lifetime": LIFETIME, "c_bits": C_BITS, "c_dw_min": C_DW_MIN,
                # Sweep axes (new policy)
                "af_ratio": af_ratio,
                "up_down_applied": BASE_AF_TO_UP_DOWN * af_ratio,
                "update_noise_ratio": update_noise_ratio,
                "dw_min_std_applied": BASE_DW_MIN_STD * update_noise_ratio,
                # Held-fixed
                "write_noise_std_fixed": WRITE_NOISE_STD_FIXED,
                "run": run_id,
            },
            reinit=True,
        )

    best_acc = 0.0
    patience_counter = 0
    epochs_trained = 0
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        scheduler.step()
        epochs_trained = epoch

        if use_wandb:
            wandb.log({"epoch": epoch, "val_acc": val_acc,
                       "best_acc": max(best_acc, val_acc)})

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= 5 and best_acc < 30.0:
            print(f"      Early stopped at epoch {epoch} (low accuracy)")
            break
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"      Early stopped at epoch {epoch}")
            break

    if use_wandb:
        wandb.log({"best_acc": best_acc, "epochs_trained": epochs_trained})
        wandb.finish()

    del model
    torch.cuda.empty_cache()
    return best_acc


# =============================================================================
# Driver
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_A", type=float, default=0.10,
                        help="Anchor RMS target. Default 0.10. Set <=0 for negative control.")
    parser.add_argument("--tlr_base", type=float, default=1e-3,
                        help="Base transfer_lr before σ_A^2 compensation. "
                             "Effective tlr = tlr_base * 1/(rank*σ_A^2).")
    parser.add_argument("--runs_per_config", type=int, default=1)
    parser.add_argument("--results_path", default="results/sra_ctile_noise_asymmetry/results.json")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Include the (AF=0, UNR=0) baseline cell (perfectly ideal C tile).")
    parser.add_argument("--af_grid", type=float, nargs="*", default=AF_RATIO_GRID,
                        help="AF (asymmetry) ratio grid. up_down = 0.05 * af_ratio.")
    parser.add_argument("--update_noise_grid", type=float, nargs="*",
                        default=UPDATE_NOISE_RATIO_GRID,
                        help="Update-noise ratio grid. dw_min_std = 0.3 * ratio.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    use_wandb = WANDB_AVAILABLE and not args.no_wandb

    eff_tlr = args.tlr_base * sigma_A_compensation_scale(args.sigma_A, RANK)
    af_grid = list(args.af_grid)
    unr_grid = list(args.update_noise_grid)

    print("=" * 72)
    print("SRA-LRTT-v2 C-TILE AF × UPDATE-NOISE Sweep (rank=8, TE=10 fixed)")
    print("=" * 72)
    print(f"  Fixed HP : rank={RANK}, TE={TE}, LR={LR}, lifetime={LIFETIME}, "
          f"epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"  AB tile  : 6T1C LinearStepDevice (mult_noise=True, write_noise_std=0)")
    print(f"  C tile   : LinearStepDevice 6T1C 8b (dw_min={C_DW_MIN:.6f}), "
          f"write_noise_std={WRITE_NOISE_STD_FIXED} FIXED")
    print(f"  SRA HP   : sigma_A={args.sigma_A}, tlr_base={args.tlr_base}, "
          f"effective_tlr={eff_tlr:.4g}")
    print(f"  Sweep    :")
    print(f"    AF ratio          : {af_grid}  → up_down = "
          f"{BASE_AF_TO_UP_DOWN}·AF")
    print(f"    Update-noise ratio: {unr_grid}  → dw_min_std = "
          f"{BASE_DW_MIN_STD}·UNR")
    print(f"  Output   : {args.results_path}")
    print()

    experiments = []
    for af_ratio in af_grid:
        for update_noise_ratio in unr_grid:
            if af_ratio == 0.0 and update_noise_ratio == 0.0 and not args.include_baseline:
                continue
            experiments.append({
                'af_ratio': af_ratio,
                'update_noise_ratio': update_noise_ratio,
            })
    total_experiments = len(experiments)
    total_runs = total_experiments * args.runs_per_config
    print(f"  Combinations: {total_experiments} cells × {args.runs_per_config} runs = "
          f"{total_runs} runs\n")

    train_loader, val_loader = load_data()

    # Resume support
    all_results = []
    completed_keys = set()
    if os.path.exists(args.results_path):
        try:
            with open(args.results_path) as f:
                all_results = json.load(f)
            for r in all_results:
                completed_keys.add((r['af_ratio'], r['update_noise_ratio']))
            print(f"  Resuming: found {len(all_results)} completed entries")
        except Exception as e:
            print(f"  Warning: could not load existing results: {e}")
            all_results = []

    exp_count = 0
    for exp in experiments:
        exp_count += 1
        af_ratio = exp['af_ratio']
        update_noise_ratio = exp['update_noise_ratio']

        key = (af_ratio, update_noise_ratio)
        if key in completed_keys:
            print(f"\n[{exp_count}/{total_experiments}] AF={af_ratio:g} "
                  f"UNR={update_noise_ratio:g} - SKIPPED")
            continue

        print(f"\n[{exp_count}/{total_experiments}] AF={af_ratio:g} "
              f"UNR={update_noise_ratio:g}")
        print("-" * 50)

        results = []
        for run_id in range(args.runs_per_config):
            acc = run_training(sigma_A=args.sigma_A, transfer_lr_base=args.tlr_base,
                               af_ratio=af_ratio, update_noise_ratio=update_noise_ratio,
                               run_id=run_id, train_loader=train_loader,
                               val_loader=val_loader, use_wandb=use_wandb)
            results.append(acc)
            print(f"    run {run_id}: {acc:.2f}%")

        best = max(results)
        avg = sum(results) / len(results)
        print(f"  -> best: {best:.2f}%, avg: {avg:.2f}%")

        all_results.append({
            'mode': 'sra',
            'rank': RANK,
            'te': TE,
            'lr': LR,
            'sigma_A': args.sigma_A,
            'tlr_base': args.tlr_base,
            'tlr_scale': sigma_A_compensation_scale(args.sigma_A, RANK),
            'effective_tlr': eff_tlr,
            'c_bits': C_BITS,
            'c_dw_min': C_DW_MIN,
            'lifetime': LIFETIME,
            # Sweep axes (new policy)
            'af_ratio': af_ratio,
            'up_down_applied': BASE_AF_TO_UP_DOWN * af_ratio,
            'update_noise_ratio': update_noise_ratio,
            'dw_min_std_applied': BASE_DW_MIN_STD * update_noise_ratio,
            'write_noise_std_fixed': WRITE_NOISE_STD_FIXED,
            'results': results,
            'best': best,
            'avg': avg,
            'timestamp': datetime.utcnow().isoformat(),
        })
        with open(args.results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary heatmap
    print("\n" + "=" * 72)
    print("HEATMAP: SRA-LRTT-v2 C-tile AF × UNR (best val_acc, %)")
    print("=" * 72)
    label_corner = "UNR\\AF"
    header = f"{label_corner:>8s}" + "".join(f"  AF={a:<5g}" for a in af_grid)
    print(header)
    print("-" * len(header))
    for unr in unr_grid:
        row = f"{unr:>8g}"
        for af in af_grid:
            cell = next((r for r in all_results
                         if r['af_ratio'] == af and r['update_noise_ratio'] == unr), None)
            row += f"  {cell['best']:>6.2f}" if cell else "     N/A"
        print(row)
    print(f"\nResults: {args.results_path}")


if __name__ == "__main__":
    main()

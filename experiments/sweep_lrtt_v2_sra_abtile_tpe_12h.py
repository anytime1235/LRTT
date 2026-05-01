#!/usr/bin/env python3
"""SRA-LRTT-v2 A,B-tile sweep with per-cell Optuna TPE search (~12 h total).

A,B-tile sweep on the Tiki-Taka / LRTT canonical AF × UPDATE-NOISE axes.
For each cell, a small Optuna TPE study tunes the SRA-only HPs
`transfer_lr` (base, before σ_A^2 compensation) and `sigma_A`
(:= sra_anchor_target_rms) in continuous priors.

Outer grid (Tiki-Taka / LRTT defense convention):
    AF ratio              → up_down       = 0.05 * af_ratio   (signed asymmetry)
    UPDATE-NOISE ratio    → dw_min_std    = 0.3  * unr        (cycle-to-cycle pulse)

    Default (4 × 4 = 16 cells, fits 12h with 6 trials/cell):
        AF_RATIO_GRID           = [0.0, 1.0, 2.0, 5.0]
        UPDATE_NOISE_RATIO_GRID = [0.0, 1.0, 2.0, 3.0]
    The (AF=0, UNR=0) baseline cell is excluded by default (use --include_baseline).

Held FIXED at 6T1C nominal (NOT scaled by either sweep axis):
    write_noise_std  = 0.0182
    dw_min_dtod      = 0.1, w_*_dtod=0.05, gamma_*_dtod=0.05, up_down_dtod=0.01

Inner TPE search (per cell):
    transfer_lr ∈ [1e-5, 1e-1]   log-uniform
    sigma_A     ∈ [0.03, 0.30]   log-uniform
    Sampler  = TPESampler(multivariate=True, group=True, n_startup=2, seed=42)
    Pruner   = HyperbandPruner(min=5, max=30, factor=3)
    Trials/cell = configurable (default 6 → 16×6 ≈ 96 trials ≈ 12 h)

Fixed:
    rank=8, TE=10, LR=0.11577, lifetime=1000
    epochs=30, batch=64, hidden=256
    C tile = nominal LinearStepDevice 6T1C 8-bit (dw_min = 2/2^8 = 0.0078125)
    A,B tile = LinearStepDevice 6T1C, swept on AF × UNR axes
    update_mode='stochastic_reset_anchor', transfer_method='stochastic_anchor'
    sra_anchor_source='reset_columns', cap_rho=1.0

Output:
    results/<output_dir>/study.db        Optuna SQLite (one study per cell)
    results/<output_dir>/per_cell.json   best HP + best_acc per cell
    results/<output_dir>/heatmap.txt     printed heatmap of best_acc

Run on its own GPU; pair with sweep_lrtt_v2_sra_ctile_tpe_12h.py on a 2nd GPU.

Usage:
    python sweep_lrtt_v2_sra_abtile_tpe_12h.py
    python sweep_lrtt_v2_sra_abtile_tpe_12h.py --trials_per_cell 8 --include_baseline
    python sweep_lrtt_v2_sra_abtile_tpe_12h.py --output_dir results/sra_abtile_tpe
"""

import os
os.environ.setdefault("LRTT_SILENT", "1")

import argparse
import json
import time
from datetime import datetime

import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


# =============================================================================
# Fixed conditions (UNIFIED with all SRA sweep scripts)
# =============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42

RANK = 8
TE = 10
LR = 0.11577
LIFETIME = 1000.0
AB_LIFETIME = float(LIFETIME)

BATCH_SIZE = 64
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
HIDDEN = 256

C_BITS = 8
C_DW_MIN = 2.0 / (2 ** C_BITS)


# =============================================================================
# Sweep policy (Tiki-Taka / LRTT canonical):
#   - Main UPDATE noise axis = `dw_min_std` ONLY (cycle-to-cycle pulse variation).
#   - Asymmetry axis = `up_down` ONLY (signed).
#   - Other 6T1C non-idealities (write_noise, *_dtod) FIXED at 6T1C nominal.
# =============================================================================
BASE_DW_MIN_STD = 0.3
BASE_AF_TO_UP_DOWN = 0.05
WRITE_NOISE_STD_FIXED = 0.0182

# Outer grid axes (smaller default to fit per-cell TPE budget in 12 h)
AF_RATIO_GRID = [0.0, 1.0, 2.0, 5.0]              # 4 values
UPDATE_NOISE_RATIO_GRID = [0.0, 1.0, 2.0, 3.0]    # 4 values  → 16 cells

# Inner TPE search ranges (HP only)
TLR_LOW, TLR_HIGH = 1.0e-5, 1.0e-1
SIGMA_A_LOW, SIGMA_A_HIGH = 0.03, 0.30


# =============================================================================
# Device builders
# =============================================================================

def make_ab_device_swept(*, af_ratio: float, update_noise_ratio: float):
    """6T1C LinearStepDevice swept on the new AF × UPDATE-NOISE axes.

    Sweep mapping:
        up_down     = BASE_AF_TO_UP_DOWN * af_ratio          # signed asymm (main)
        dw_min_std  = BASE_DW_MIN_STD    * update_noise_ratio # cycle-to-cycle (main)

    Held FIXED at 6T1C nominal (NOT scaled by either sweep axis):
        write_noise_std  = WRITE_NOISE_STD_FIXED
        dw_min_dtod      = 0.1
        up_down_dtod     = 0.01
        w_max_dtod       = 0.05
        w_min_dtod       = 0.05
        gamma_up_dtod    = 0.05
        gamma_down_dtod  = 0.05
    """
    return LinearStepDevice(
        dw_min=0.001981,
        up_down=BASE_AF_TO_UP_DOWN * float(af_ratio),
        w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=BASE_DW_MIN_STD * float(update_noise_ratio),
        write_noise_std=WRITE_NOISE_STD_FIXED,
        apply_write_noise_on_set=False,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.1,
        reset=0.0, reset_std=0.01, reset_dtod=0.0,
    )


def make_c_device_nominal_8bit():
    """C tile: LinearStepDevice 6T1C parameters at 8-bit precision (no AF/noise scaling).

    Per the user's defense plan, ALL tiles use LinearStepDevice 6T1C. This
    nominal C tile pins `dw_min` at 8 bits and zeros all C2C/D2D noise so
    only the AB-tile sweep axes drive accuracy variation.
    """
    return LinearStepDevice(
        dw_min=C_DW_MIN, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=True,
        up_down=0.0,
        up_down_dtod=0.0,
        dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        gamma_up_dtod=0.0, gamma_down_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=AB_LIFETIME, lifetime_dtod=0.0,
        reset=0.0, reset_dtod=0.0,
    )


# =============================================================================
# σ_A helpers
# =============================================================================

def sigma_A_to_target_rms(sigma_A: float) -> float:
    """Pass σ_A directly to sra_anchor_target_rms.

    σ_A == 0 IS allowed: it is the negative control (A_q := 0  ⇒  A_q^T G = 0).
    """
    return float(sigma_A)


def sigma_A_compensation_scale(sigma_A: float, rank: int) -> float:
    if sigma_A <= 0:
        return 1.0
    return 1.0 / (rank * sigma_A * sigma_A)


# =============================================================================
# Model builder (cell device fixed; HP from TPE)
# =============================================================================

def build_model(*, transfer_lr: float, sigma_A: float,
                af_ratio: float, update_noise_ratio: float):
    a = make_ab_device_swept(af_ratio=af_ratio, update_noise_ratio=update_noise_ratio)
    b = make_ab_device_swept(af_ratio=af_ratio, update_noise_ratio=update_noise_ratio)
    c = make_c_device_nominal_8bit()

    target_rms = sigma_A_to_target_rms(sigma_A)
    tlr_scale = sigma_A_compensation_scale(sigma_A, RANK)

    cfg = PythonLRTTDevice(
        rank=RANK,
        transfer_every=TE,
        transfer_lr=float(transfer_lr),
        transfer_lr_scale=float(tlr_scale),
        update_mode="stochastic_reset_anchor",
        transfer_method="stochastic_anchor",
        forward_inject=False,
        a_init_mode="zero",
        b_init_mode="zero",
        cap_stabilizer_enabled=True,
        cap_rho=1.0,
        cap_compensate_transfer=True,
        sra_anchor_source="reset_columns",
        sra_anchor_target_rms=target_rms,
        sra_anchor_gain_max=1.0e3,
        sra_b_reset_mode="set_zero",
        sra_seed=SEED,
        unit_cell_devices=[a, b, c],
    )
    rpu = PythonLRTTRPUConfig(device=cfg)
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    rpu.mapping.weight_scaling_omega = 0.6
    return AnalogSequential(
        AnalogLinear(784, HIDDEN, bias=True, rpu_config=rpu),
        nn.ReLU(),
        AnalogLinear(HIDDEN, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)


# =============================================================================
# Cached MNIST loaders
# =============================================================================
_TRAIN = _VAL = None


def get_loaders():
    global _TRAIN, _VAL
    if _TRAIN is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform)
        val_ds = datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform)
        _TRAIN = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=False, persistent_workers=True)
        _VAL = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=1, pin_memory=False, persistent_workers=True)
    return _TRAIN, _VAL


# =============================================================================
# Trial runner (HyperbandPruner reporting)
# =============================================================================

def run_trial(trial, *, transfer_lr, sigma_A, af_ratio, update_noise_ratio):
    torch.manual_seed(SEED)
    train_loader, val_loader = get_loaders()
    model = build_model(transfer_lr=transfer_lr, sigma_A=sigma_A,
                        af_ratio=af_ratio, update_noise_ratio=update_noise_ratio)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    history = []
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = n = 0
        for data, target in train_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * data.shape[0]
            n += data.shape[0]
        train_loss = running / max(1, n)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
                target = target.to(DEVICE, non_blocking=True)
                correct += model(data).argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        scheduler.step()
        acc = 100.0 * correct / max(1, total)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": acc})
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1

        trial.report(acc, step=epoch)
        if trial.should_prune():
            del model
            torch.cuda.empty_cache()
            trial.set_user_attr("history", history)
            raise optuna.TrialPruned(f"epoch {epoch}: pruned at acc={acc:.2f}%")

        if epoch >= 5 and best_acc < 25.0:
            break
        if patience >= EARLY_STOP_PATIENCE:
            break

    trial.set_user_attr("history", history)
    del model
    torch.cuda.empty_cache()
    return best_acc


# =============================================================================
# Per-cell Optuna study
# =============================================================================

def run_cell(*, af_ratio, update_noise_ratio,
             trials_per_cell, storage, n_startup):
    """Run one Optuna TPE study for a single (af_ratio, update_noise_ratio) cell."""
    study_name = f"sra_abtile_AF{af_ratio:g}_UNR{update_noise_ratio:g}"
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup,
        multivariate=True,
        group=True,
        seed=SEED,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5, max_resource=EPOCHS, reduction_factor=3,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial):
        transfer_lr = trial.suggest_float("transfer_lr", TLR_LOW, TLR_HIGH, log=True)
        sigma_A = trial.suggest_float("sigma_A", SIGMA_A_LOW, SIGMA_A_HIGH, log=True)
        eff_tlr = transfer_lr * sigma_A_compensation_scale(sigma_A, RANK)
        trial.set_user_attr("tlr_scale", sigma_A_compensation_scale(sigma_A, RANK))
        trial.set_user_attr("effective_transfer_lr", eff_tlr)
        t0 = time.time()
        print(f"   [trial {trial.number}] tlr={transfer_lr:.3e}  σ_A={sigma_A:.4f}  "
              f"(eff_tlr={eff_tlr:.3e})", flush=True)
        try:
            best = run_trial(trial, transfer_lr=transfer_lr, sigma_A=sigma_A,
                             af_ratio=af_ratio, update_noise_ratio=update_noise_ratio)
            dt = time.time() - t0
            print(f"     → best_acc={best:.2f}%  took {dt/60:.1f} min", flush=True)
            return best
        except optuna.TrialPruned as e:
            dt = time.time() - t0
            print(f"     → PRUNED ({e})  took {dt/60:.1f} min", flush=True)
            raise
        except Exception as e:
            print(f"     → ERROR: {e}", flush=True)
            raise

    n_existing = len(study.trials)
    n_to_run = max(0, trials_per_cell - n_existing)
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    if completed:
        b = max(completed, key=lambda t: t.value)
        return b.value, dict(b.params), dict(b.user_attrs), len(completed), len(pruned)
    return None, None, None, len(completed), len(pruned)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials_per_cell", type=int, default=6,
                        help="Per-cell TPE budget. Default 6 → 16 cells × 6 ≈ 96 "
                             "trials × ~7 min/trial ≈ 11-12 h.")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Include the (AF=0, UNR=0) baseline cell.")
    parser.add_argument("--af_grid", type=float, nargs="*", default=AF_RATIO_GRID)
    parser.add_argument("--update_noise_grid", type=float, nargs="*",
                        default=UPDATE_NOISE_RATIO_GRID)
    parser.add_argument("--output_dir", default="results/sra_abtile_tpe_12h")
    parser.add_argument("--storage", default=None,
                        help="Optuna SQLite URL. Default = sqlite:///<output_dir>/study.db")
    parser.add_argument("--n_startup_trials", type=int, default=2,
                        help="TPE random startup per cell.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    storage = args.storage or f"sqlite:///{args.output_dir}/study.db"

    af_grid = list(args.af_grid)
    unr_grid = list(args.update_noise_grid)

    cells = []
    for af_ratio in af_grid:
        for update_noise_ratio in unr_grid:
            if af_ratio == 0.0 and update_noise_ratio == 0.0 and not args.include_baseline:
                continue
            cells.append({
                "af_ratio": af_ratio,
                "update_noise_ratio": update_noise_ratio,
            })

    print("=" * 72)
    print("SRA-LRTT-v2 A,B-TILE PER-CELL TPE SEARCH (~12 h budget)")
    print("=" * 72)
    print(f"  Device:         {DEVICE}")
    print(f"  Mode:           SRA only (update_mode='stochastic_reset_anchor')")
    print(f"  Fixed:  rank={RANK}, TE={TE}, LR={LR}, lifetime={LIFETIME}, "
          f"epochs={EPOCHS}, batch={BATCH_SIZE}, hidden={HIDDEN}")
    print(f"          C tile = nominal LinearStepDevice 6T1C {C_BITS}b "
          f"(dw_min={C_DW_MIN:.6f})")
    print(f"  Outer grid (Tiki-Taka / LRTT canonical AF × UNR axes):")
    print(f"    AF ratio          ∈ {af_grid}  → up_down = "
          f"{BASE_AF_TO_UP_DOWN}·AF")
    print(f"    Update-noise ratio∈ {unr_grid}  → dw_min_std = "
          f"{BASE_DW_MIN_STD}·UNR")
    print(f"  Cells: {len(cells)}  (baseline (AF=0,UNR=0) "
          f"{'INCLUDED' if args.include_baseline else 'excluded'})")
    print(f"  Inner TPE search per cell:")
    print(f"    transfer_lr ∈ [{TLR_LOW:.0e}, {TLR_HIGH:.0e}]  log-uniform")
    print(f"    sigma_A     ∈ [{SIGMA_A_LOW:.2f}, {SIGMA_A_HIGH:.2f}]      log-uniform")
    print(f"  trials/cell: {args.trials_per_cell}  "
          f"(total ≈ {len(cells) * args.trials_per_cell} trials, "
          f"≈ {len(cells)*args.trials_per_cell*7/60:.1f} h)")
    print(f"  Storage: {storage}")
    print(f"  Output:  {args.output_dir}")
    print()

    per_cell = []
    t_start = time.time()
    for ci, cell in enumerate(cells, 1):
        elapsed = time.time() - t_start
        rem = (len(cells) - ci + 1) * (elapsed / max(1, ci - 1)) if ci > 1 else 0
        eta_str = f", ETA ~{rem/60:.0f} min" if rem else ""
        af_ratio = cell['af_ratio']
        update_noise_ratio = cell['update_noise_ratio']
        print(f"\n[{ci}/{len(cells)}] cell: AB AF={af_ratio:g} "
              f"UNR={update_noise_ratio:g}{eta_str}")
        print("-" * 50)
        best_val, best_params, best_attrs, ncomp, npruned = run_cell(
            af_ratio=af_ratio, update_noise_ratio=update_noise_ratio,
            trials_per_cell=args.trials_per_cell, storage=storage,
            n_startup=args.n_startup_trials,
        )
        rec = {
            "af_ratio": af_ratio,
            "ab_up_down_applied": BASE_AF_TO_UP_DOWN * af_ratio,
            "update_noise_ratio": update_noise_ratio,
            "ab_dw_min_std_applied": BASE_DW_MIN_STD * update_noise_ratio,
            "ab_write_noise_std_fixed": WRITE_NOISE_STD_FIXED,
            "best_acc": best_val,
            "best_params": best_params,
            "best_user_attrs": best_attrs,
            "n_completed": ncomp,
            "n_pruned": npruned,
            "timestamp": datetime.utcnow().isoformat(),
        }
        per_cell.append(rec)
        msg = f"best={best_val:.2f}%" if best_val is not None else "best=N/A"
        print(f"  cell summary: {msg}, completed={ncomp}, pruned={npruned}")
        if best_params:
            print(f"    best HP: tlr={best_params.get('transfer_lr'):.4g}, "
                  f"σ_A={best_params.get('sigma_A'):.4g}")
        with open(f"{args.output_dir}/per_cell.json", "w") as f:
            json.dump(per_cell, f, indent=2)

    # Final heatmap
    print("\n" + "=" * 72)
    print("HEATMAP: SRA-LRTT-v2 A,B-tile per-cell best val_acc (%)")
    print("=" * 72)
    label_corner = "UNR\\AF"
    header = f"{label_corner:>8s}" + "".join(f"  AF={a:<5g}" for a in af_grid)
    lines = [header, "-" * len(header)]
    for unr in unr_grid:
        row = f"{unr:>8g}"
        for af in af_grid:
            cell = next((r for r in per_cell
                         if r['af_ratio'] == af and r['update_noise_ratio'] == unr), None)
            if cell and cell['best_acc'] is not None:
                row += f"  {cell['best_acc']:>6.2f}"
            else:
                row += "     N/A"
        lines.append(row)
    heatmap_str = "\n".join(lines)
    print(heatmap_str)
    with open(f"{args.output_dir}/heatmap.txt", "w") as f:
        f.write(heatmap_str + "\n")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed/3600:.2f} h")
    print(f"Outputs -> {args.output_dir}/{{per_cell.json, heatmap.txt, study.db}}")


if __name__ == "__main__":
    main()

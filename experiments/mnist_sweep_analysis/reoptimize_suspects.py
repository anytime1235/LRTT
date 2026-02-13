#!/usr/bin/env python3
"""Re-optimize under-optimized (rank, TE) combinations.

- Targets ~10 suspect combos from decay + hybrid sweeps
- Lifetime fixed to 46505 (sixt1c)
- Searches lr, tlr only (log-uniform + neighbor seeds)
- 10 trials per combo, global budget 200 trials, early stopping
- Updates lrtt_sweep_best_configs.json on improvement
"""

import os
os.environ["LRTT_SILENT"] = "1"

import math
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time
from datetime import datetime
import json

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset, PythonLRTTDevice

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LIFETIME = 46505  # sixt1c fixed

# Stage 1: fast search
STAGE1_EPOCHS = 10
STAGE1_PATIENCE = 3
STAGE1_TRIALS = 20       # per combo

# Stage 2: full training with top-k from stage 1
STAGE2_EPOCHS = 30
STAGE2_PATIENCE = 5
STAGE2_TOP_K = 3

GLOBAL_TRIAL_BUDGET = 200  # total stage1+stage2 trials across all combos

# ─── Resume data from crashed run (combo 0, stage1 18/20 done) ───
RESUME_COMBO0_STAGE2 = [
    {"lr": 0.332709, "tlr": 0.004094, "best_acc": 96.36},
    {"lr": 0.710188, "tlr": 0.012043, "best_acc": 94.60},
]

RESUME_COMBO0_STAGE1 = [
    {"lr": 0.089054, "tlr": 0.001277, "best_acc": 93.75},
    {"lr": 0.674107, "tlr": 0.011775, "best_acc": 93.73},
    {"lr": 0.040615, "tlr": 1.692439, "best_acc": 82.74},
    {"lr": 0.081323, "tlr": 0.000477, "best_acc": 93.34},
    {"lr": 0.058496, "tlr": 0.708833, "best_acc": 87.33},
    {"lr": 0.724506, "tlr": 0.021375, "best_acc": 10.28},
    {"lr": 0.247849, "tlr": 0.017195, "best_acc": 94.65},
    {"lr": 0.031256, "tlr": 0.002743, "best_acc": 91.31},
    {"lr": 0.332709, "tlr": 0.004094, "best_acc": 96.08},
    {"lr": 0.047056, "tlr": 0.001014, "best_acc": 92.55},
    {"lr": 0.069147, "tlr": 2.778727, "best_acc": 85.99},
    {"lr": 0.743224, "tlr": 0.010389, "best_acc": 93.26},
    {"lr": 0.068701, "tlr": 0.002504, "best_acc": 93.38},
    {"lr": 0.077524, "tlr": 1.132214, "best_acc": 85.87},
    {"lr": 0.083443, "tlr": 0.000896, "best_acc": 94.02},
    {"lr": 0.036858, "tlr": 1.675473, "best_acc": 85.07},
    {"lr": 0.094594, "tlr": 0.001183, "best_acc": 93.91},
    {"lr": 0.065344, "tlr": 0.001778, "best_acc": 93.14},
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_CONFIGS_PATH = os.path.join(SCRIPT_DIR, "lrtt_sweep_best_configs.json")

# ─── Suspect combinations ───
# Goal: Decay should peak at low TE (1,10), Hybrid should peak at high TE (100~1000)
# Selected: combos where the evidence is weak or contradictory
SUSPECTS = [
    # DECAY: low TE should be best, but these are under-performing at low TE
    {"mode": "decay",  "rank": 1,  "te": 10,  "current": 90.13,
     "reason": "TE=10 catastrophically low (90.13 vs TE=50=97.01), HP failure"},
    {"mode": "decay",  "rank": 1,  "te": 1,   "current": 95.07,
     "reason": "TE=1 should be ~97 like r=32/64, but only 95.07"},
    {"mode": "decay",  "rank": 4,  "te": 1,   "current": 96.16,
     "reason": "TE=1 (96.16) < TE=100 (97.34), low TE should win"},
    {"mode": "decay",  "rank": 4,  "te": 10,  "current": 96.77,
     "reason": "TE=10 (96.77) < TE=100 (97.34), low TE should win"},
    {"mode": "decay",  "rank": 8,  "te": 1,   "current": 96.57,
     "reason": "TE=1 (96.57) < TE=10 (97.17), TE=1 should be at least as good"},
    # HYBRID: high TE (100~1000) should be best, but these are under-performing
    {"mode": "hybrid", "rank": 16, "te": 100,  "current": 94.91,
     "reason": "TE=100 (94.91) < TE=50 (95.89), contradicts high-TE claim"},
    {"mode": "hybrid", "rank": 8,  "te": 500,  "current": 93.55,
     "reason": "TE=500 (93.55) << TE=50 (96.20), high TE should not drop this much"},
    {"mode": "hybrid", "rank": 32, "te": 1000, "current": 91.38,
     "reason": "TE=1000 (91.38) far below TE=50~500 (~96), too steep decline"},
    {"mode": "hybrid", "rank": 8,  "te": 100,  "current": 96.20,
     "reason": "TE=100 (96.20) = TE=50 (96.20), TE=100 should clearly beat TE=50"},
    {"mode": "hybrid", "rank": 1,  "te": 1000, "current": 94.47,
     "reason": "TE=1000 (94.47) < TE=500 (95.73), high TE should be sustained"},
]

# ─── Neighbor HP seeds (from best configs JSON: by_rank_te) ───
# decay: HYPERPARAMETERS dict from sweep_softbounds_lifetime.py
DECAY_HP = {
    1:  {1: (0.089054, 0.001277), 10: (0.040615, 1.692439), 50: (0.674107, 0.011775),
         100: (0.023799, 0.038241), 500: (0.013959, 0.048825), 1000: (0.007858, 4.472098)},
    4:  {1: (0.089054, 0.001277), 10: (0.001735, 0.008158), 50: (0.674107, 0.011775),
         100: (0.493706, 0.011245), 500: (0.818908, 9.649071), 1000: (0.007858, 4.472098)},
    8:  {1: (0.089054, 0.001277), 10: (0.001735, 0.008158), 50: (0.092537, 3.202363),
         100: (0.493706, 0.011245), 500: (0.013959, 0.048825), 1000: (0.003435, 0.011312)},
    16: {1: (0.089054, 0.001277), 10: (0.045302, 7.80662),  50: (0.008838, 0.008245),
         100: (0.023799, 0.038241), 500: (0.006316, 0.53314), 1000: (0.003435, 0.011312)},
    32: {1: (0.089054, 0.001277), 10: (0.040615, 1.692439), 50: (0.008838, 0.008245),
         100: (0.023799, 0.038241), 500: (0.013959, 0.048825), 1000: (0.003435, 0.011312)},
    64: {1: (0.089054, 0.001277), 10: (0.040615, 1.692439), 50: (0.008838, 0.008245),
         100: (0.493706, 0.011245), 500: (0.006316, 0.53314), 1000: (0.002874, 0.359582)},
}

# hybrid: best found lr/tlr from by_rank_te in JSON
HYBRID_HP = {
    1:  {1: (0.518826, 0.006708), 10: (0.190820, 0.047258), 50: (0.155101, 0.003145),
         100: (0.121848, 0.043497), 500: (0.255999, 0.064779), 1000: (0.678205, 0.003058)},
    4:  {1: (0.461206, 0.004641), 10: (0.388981, 0.001048), 50: (0.341834, 0.003911),
         100: (0.341834, 0.003911), 500: (0.241216, 0.049499), 1000: (0.685213, 0.624920)},
    8:  {1: (0.113545, 0.001522), 10: (0.050209, 0.010974), 50: (0.701092, 0.004174),
         100: (0.171931, 0.002665), 500: (0.150189, 0.022623), 1000: (0.319825, 1.276405)},
    16: {1: (0.175230, 0.002210), 10: (0.140737, 0.085730), 50: (0.187206, 0.002161),
         100: (0.187206, 0.002161), 500: (0.233461, 0.004402), 1000: (0.882021, 0.008597)},
    32: {1: (0.169914, 0.001749), 10: (0.169914, 0.001749), 50: (0.993416, 0.001219),
         100: (0.993416, 0.001219), 500: (0.602355, 0.001898), 1000: (0.006297, 0.130551)},
    64: {1: (0.252012, 0.001336), 10: (0.252012, 0.001336), 50: (0.252012, 0.001336),
         100: (0.252012, 0.001336), 500: (0.252012, 0.001336), 1000: (0.318716, 0.006835)},
}

# SoftBounds config (no noise)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}


def log_uniform(low, high):
    """Sample from log-uniform distribution."""
    return math.exp(random.uniform(math.log(low), math.log(high)))


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    """Convert lifetime to dt_batch_sec for sixt1c_ab preset."""
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    dt = -TAU_SEC * math.log(1 - delta)
    return dt


def load_data():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('/tmp/mnist', download=True, train=True, transform=transform)
    val_set = datasets.MNIST('/tmp/mnist', download=True, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


def create_model(rank: int, te: int, lifetime: float, lr: float, tlr: float, mode: str = "hybrid"):
    """Create LRTT model. mode='hybrid' (A=0 hard reset) or 'decay' (A&B decay)."""
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)

    TAU_SEC = 46505.0
    if dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        ab_lifetime = 1.0 / delta
    else:
        ab_lifetime = 0.0

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )

    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode=mode,
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    model = AnalogSequential(
        AnalogLinear(784, 256, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
        nn.LogSoftmax(dim=1),
    )
    model.to(DEVICE)
    return model


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    for data, target in train_loader:
        data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
        target = target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def validate(model, val_loader):
    """Validate model."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def get_neighbor_seeds(mode, rank, te):
    """Collect seed HPs from neighboring TE combos (same rank, adjacent TEs)."""
    hp_dict = DECAY_HP if mode == "decay" else HYBRID_HP
    all_tes = [1, 10, 50, 100, 500, 1000]
    ti = all_tes.index(te)

    seeds = []
    # Adjacent TEs at the same rank (most relevant neighbors)
    for dj in [-1, 1]:
        nj = ti + dj
        if 0 <= nj < len(all_tes):
            seeds.append(hp_dict[rank][all_tes[nj]])

    # Current TE's own HP as fallback
    seeds.append(hp_dict[rank][te])
    return seeds


def generate_trial_hps(seeds, n_trials):
    """Generate n_trials HPs: centered around neighbor TE seeds with perturbation.

    Strategy:
      - trials 0..len(seeds)-1: exact seed HPs (unperturbed)
      - remaining: perturbations of random seed, scale shrinks per batch
        × 0.3~3.0 (wide), × 0.5~2.0 (medium), × 0.7~1.4 (narrow)
    """
    hps = []
    # Exact seeds first
    for lr, tlr in seeds:
        hps.append((lr, tlr))

    scales = [3.0, 2.0, 1.4]  # wide -> narrow perturbation
    while len(hps) < n_trials:
        idx = len(hps) - len(seeds)
        phase = min(idx // 6, len(scales) - 1)
        scale = scales[phase]

        base_lr, base_tlr = random.choice(seeds)
        lr = base_lr * math.exp(random.uniform(-math.log(scale), math.log(scale)))
        tlr = base_tlr * math.exp(random.uniform(-math.log(scale), math.log(scale)))
        lr = max(0.0005, min(2.0, lr))
        tlr = max(0.0002, min(15.0, tlr))
        hps.append((lr, tlr))

    return hps[:n_trials]


def run_trial(mode, rank, te, lr, tlr, epochs, patience, train_loader, val_loader):
    """Run a single training trial."""
    model = create_model(rank, te, LIFETIME, lr, tlr, mode=mode)
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion)
        val_acc = validate(model, val_loader)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= 3 and best_acc < 50.0:
            break
        if patience_counter >= patience:
            break

    del model
    torch.cuda.empty_cache()
    return {"lr": lr, "tlr": tlr, "best_acc": best_acc}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = SCRIPT_DIR
    results_file = os.path.join(results_dir, f"reopt_results_{timestamp}.json")

    print("=" * 70)
    print("2-Stage Re-optimization of suspect (rank, TE) combinations")
    print(f"Lifetime fixed: {LIFETIME} (sixt1c)")
    print(f"Stage 1: {STAGE1_TRIALS} trials x {STAGE1_EPOCHS}ep (neighbor-seeded)")
    print(f"Stage 2: top-{STAGE2_TOP_K} x {STAGE2_EPOCHS}ep full training")
    print(f"Global budget: {GLOBAL_TRIAL_BUDGET} trials | Suspects: {len(SUSPECTS)}")
    print("=" * 70)

    with open(BEST_CONFIGS_PATH) as f:
        best_configs = json.load(f)

    train_loader, val_loader = load_data()
    global_trial_count = 0
    no_improve_streak = 0
    all_results = []
    updates = []

    for si, suspect in enumerate(SUSPECTS):
        mode = suspect["mode"]
        rank = suspect["rank"]
        te = suspect["te"]
        current_best = suspect["current"]

        print(f"\n{'='*70}")
        print(f"[{si+1}/{len(SUSPECTS)}] {mode} r={rank} TE={te} | current={current_best:.2f}%")
        print(f"  Reason: {suspect['reason']}")
        print(f"  Global trials used: {global_trial_count}/{GLOBAL_TRIAL_BUDGET}")
        print("-" * 70)

        if global_trial_count >= GLOBAL_TRIAL_BUDGET:
            print("  GLOBAL BUDGET REACHED — skipping")
            break

        # ── Stage 1: Fast search (10ep) seeded from neighbor TEs ──
        seeds = get_neighbor_seeds(mode, rank, te)
        trial_hps = generate_trial_hps(seeds, STAGE1_TRIALS)

        print(f"  Stage 1: {STAGE1_TRIALS} trials x {STAGE1_EPOCHS}ep")
        print(f"  Seeds from neighbor TEs: {len(seeds)} base HPs")
        for i, (lr, tlr) in enumerate(seeds):
            print(f"    seed {i}: lr={lr:.6f}, tlr={tlr:.6f}")

        # Resume: combo 0 has partial stage1 from previous run
        stage1_results = []
        if si == 0 and RESUME_COMBO0_STAGE1:
            stage1_results = list(RESUME_COMBO0_STAGE1)
            global_trial_count += len(stage1_results)
            print(f"  RESUMED {len(stage1_results)} trials from previous run")
            for ri, r in enumerate(stage1_results):
                print(f"    [resumed] t{ri:02d}: lr={r['lr']:.6f} tlr={r['tlr']:.6f} -> {r['best_acc']:.2f}%")

        for ti, (lr, tlr) in enumerate(trial_hps):
            if ti < len(stage1_results):
                continue  # skip already-done trials
            if global_trial_count >= GLOBAL_TRIAL_BUDGET:
                break

            try:
                result = run_trial(mode, rank, te, lr, tlr,
                                   STAGE1_EPOCHS, STAGE1_PATIENCE,
                                   train_loader, val_loader)
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"    [ERR] t{ti:02d}: {type(e).__name__} - skipped")
                continue
            global_trial_count += 1
            stage1_results.append(result)

            src = "seed" if ti < len(seeds) else "pert"
            print(f"    [{src}] t{ti:02d}: lr={lr:.6f} tlr={tlr:.6f} -> {result['best_acc']:.2f}%")

        # ── Stage 2: Full training with top-K from stage 1 ──
        stage1_sorted = sorted(stage1_results, key=lambda x: x["best_acc"], reverse=True)
        top_k = stage1_sorted[:STAGE2_TOP_K]

        print(f"\n  Stage 2: top-{STAGE2_TOP_K} -> {STAGE2_EPOCHS}ep full training")
        config_best_acc = current_best
        config_best_lr = None
        config_best_tlr = None
        stage2_results = []

        # Resume: combo 0 has partial stage2 from previous run
        if si == 0 and RESUME_COMBO0_STAGE2:
            stage2_results = list(RESUME_COMBO0_STAGE2)
            global_trial_count += len(stage2_results)
            for ri, r in enumerate(stage2_results):
                if r["best_acc"] > config_best_acc:
                    config_best_acc = r["best_acc"]
                    config_best_lr = r["lr"]
                    config_best_tlr = r["tlr"]
                print(f"    [resumed] top{ri}: lr={r['lr']:.6f} tlr={r['tlr']:.6f} -> {r['best_acc']:.2f}%")

        for ki, candidate in enumerate(top_k):
            if ki < len(stage2_results):
                continue  # skip already-done
            if global_trial_count >= GLOBAL_TRIAL_BUDGET:
                break

            lr, tlr = candidate["lr"], candidate["tlr"]
            try:
                result = run_trial(mode, rank, te, lr, tlr,
                                   STAGE2_EPOCHS, STAGE2_PATIENCE,
                                   train_loader, val_loader)
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"    top{ki}: ERROR ({type(e).__name__}) - skipped")
                continue
            global_trial_count += 1
            stage2_results.append(result)

            improved_marker = " ** NEW BEST" if result["best_acc"] > config_best_acc else ""
            print(f"    top{ki}: lr={lr:.6f} tlr={tlr:.6f} "
                  f"(s1={candidate['best_acc']:.2f}%) -> s2={result['best_acc']:.2f}%{improved_marker}")

            if result["best_acc"] > config_best_acc:
                config_best_acc = result["best_acc"]
                config_best_lr = lr
                config_best_tlr = tlr

        # ── Summary ──
        improved = config_best_acc > current_best
        delta = config_best_acc - current_best
        status = f"IMPROVED +{delta:.2f}%" if improved else "no improvement"
        print(f"\n  >>> {mode} r={rank} TE={te}: {current_best:.2f} -> {config_best_acc:.2f}% ({status})")

        combo_result = {
            "mode": mode, "rank": rank, "te": te,
            "old_best": current_best, "new_best": config_best_acc,
            "improved": improved, "delta": delta,
            "best_lr": config_best_lr, "best_tlr": config_best_tlr,
            "stage1_trials": stage1_results,
            "stage2_trials": stage2_results,
        }
        all_results.append(combo_result)

        # Update best_configs JSON if improved
        if improved and config_best_lr is not None:
            sweep_key = "hybrid_sweep" if mode == "hybrid" else "decay_sweep"
            by_rank_te = best_configs[sweep_key]["by_rank_te"]

            if mode == "hybrid":
                by_rank_te[str(rank)][str(te)] = {
                    "best_acc": config_best_acc,
                    "lifetime": LIFETIME,
                    "lr": config_best_lr,
                    "tlr": config_best_tlr,
                    "note": f"reopt from {current_best:.2f}%"
                }
            else:
                by_rank_te[str(rank)][str(te)] = config_best_acc

            updates.append(f"{mode} r={rank} TE={te}: {current_best:.2f} -> {config_best_acc:.2f}%")
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        # Save intermediate
        with open(results_file, 'w') as f:
            json.dump({"suspects": SUSPECTS, "results": all_results, "updates": updates,
                        "global_trials": global_trial_count}, f, indent=2)

        # Global early stop: 3 consecutive combos with no improvement
        if no_improve_streak >= 6:
            print(f"\n  EARLY STOP: {no_improve_streak} consecutive combos without improvement")
            break

    # Save updated best_configs
    if updates:
        best_configs["reopt_updates"] = {
            "timestamp": datetime.now().isoformat(),
            "updates": updates,
            "total_trials": global_trial_count,
        }
        with open(BEST_CONFIGS_PATH, 'w') as f:
            json.dump(best_configs, f, indent=2)
        print(f"\nUpdated {BEST_CONFIGS_PATH}")

    # Final summary
    print("\n" + "=" * 70)
    print("RE-OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Total trials: {global_trial_count}")
    print(f"Configs tested: {len(all_results)}/{len(SUSPECTS)}")
    n_improved = sum(1 for r in all_results if r["improved"])
    print(f"Improved: {n_improved}/{len(all_results)}")
    if updates:
        print("\nUpdates applied to best_configs.json:")
        for u in updates:
            print(f"  {u}")
    else:
        print("\nNo improvements found.")
    print(f"\nDetailed results: {results_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-cell Optuna TPE HP search (30 trials/cell) for the 5x5 (AF, UNR) grid.

Methods supported:
  Baselines (warm-started via sweep_5x5_fixed_hp completed trials):
    - tikitaka_v1 (TikiTaka v1 best 3-HP)
    - lrtt_v1     (LRTT-v1 decay/lora/onehot)
    - lrtt_v2     (LRTT-v2 selector_reconstruction/blockwise)
  Reset-mode ablations (warm-started via enqueue_trial of baseline best HP, no
  prior fixed-HP sweep — first trial evaluates the enqueued HP for real):
    - tikitaka_reset (= tikitaka_v1 + ChoppedTransferCompound.with_reset_prob=1.0)
    - lrtt_v1_reset  (= lrtt_v1 + PythonLRTTDevice.reinit_mode='standard')

For each (method, AF, UNR) cell:
  1. Pre-load the prior best HP. If a fixed-HP eval is cached, add it as a
     completed warm-start trial (TPE prior); otherwise enqueue it so the first
     real evaluation uses that HP.
  2. Run Optuna TPE for N_TRIALS=30 trials in a log-uniform box around best HP.
  3. Save per-trial records, best HP, best acc to a JSON file.

The script is resumable: a cell's JSON tracks completed trials, so a re-run
adds them all back via add_trial() and only runs the remaining budget.

Output: /root/LRTT/results/per_cell_tpe_30/<method>/af{a:g}_unr{u:g}.json
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("LRTT_SILENT", "1")
sys.path.insert(0, "/root/LRTT/experiments")

import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial

import hp_search_tikitaka_3hp_gamma_af as tt3_mod
import hp_search_v1_decay_gamma_af_2stage as v1_mod
import hp_search_v2_rank8_gamma_af_2stage as v2_mod
import sweep_5x5_reset_modes as reset_mod

optuna.logging.set_verbosity(optuna.logging.WARNING)

AF_GRID  = [0.0, 1.0, 2.0, 5.0, 10.0]
UNR_GRID = [0.0, 1.0, 3.0, 5.0, 10.0]
N_TRIALS = 30
SEED = 42

BEST_HP = {
    "tikitaka_v1":     {"transfer_lr": 1.0, "fast_lr": 0.3, "classifier_lr": 1.0},
    "lrtt_v1":         {"lr": 0.1, "tlr": 0.001, "clr": 0.3},
    "lrtt_v2":         {"lr": 1.0, "tlr": 0.1, "clr": 1.0},
    # Reset-mode ablations reuse the baseline best HP as the warm-start point
    # (no prior fixed-HP sweep exists yet).
    "tikitaka_reset":  {"transfer_lr": 1.0, "fast_lr": 0.3, "classifier_lr": 1.0},
    "lrtt_v1_reset":   {"lr": 0.1, "tlr": 0.001, "clr": 0.3},
    "lrtt_v2_noreset": {"lr": 1.0, "tlr": 0.1, "clr": 1.0},
}

# Search space mirrors the corresponding baseline so the ablation is
# directly comparable cell-for-cell.
_SS_TIKITAKA = {
    "transfer_lr":   FloatDistribution(0.1, 10.0, log=True),
    "fast_lr":       FloatDistribution(0.03, 3.0,  log=True),
    "classifier_lr": FloatDistribution(0.1, 10.0, log=True),
}
_SS_LRTT_V1 = {
    "lr":  FloatDistribution(0.01, 1.0,  log=True),
    "tlr": FloatDistribution(1e-4, 1e-2, log=True),
    "clr": FloatDistribution(0.03, 3.0,  log=True),
}
_SS_LRTT_V2 = {
    "lr":  FloatDistribution(0.1, 10.0, log=True),
    "tlr": FloatDistribution(0.01, 1.0, log=True),
    "clr": FloatDistribution(0.1, 10.0, log=True),
}
SEARCH_SPACE = {
    "tikitaka_v1":     _SS_TIKITAKA,
    "lrtt_v1":         _SS_LRTT_V1,
    "lrtt_v2":         _SS_LRTT_V2,
    "tikitaka_reset":  _SS_TIKITAKA,
    "lrtt_v1_reset":   _SS_LRTT_V1,
    "lrtt_v2_noreset": _SS_LRTT_V2,
}

OUT_ROOT = "/root/LRTT/results/per_cell_tpe_30"
# When a SWEEP_FILE entry is None the warm-start has no prior acc; the script
# enqueues the HP via study.enqueue_trial so it gets evaluated as trial 1.
SWEEP_FILES = {
    "tikitaka_v1":     "/root/LRTT/results/sweep_5x5_fixed_hp/tikitaka_bestHP.json",
    "lrtt_v1":         "/root/LRTT/results/sweep_5x5_fixed_hp/lrtt_v1.json",
    "lrtt_v2":         "/root/LRTT/results/sweep_5x5_fixed_hp/lrtt_v2.json",
    "tikitaka_reset":  None,
    "lrtt_v1_reset":   None,
    "lrtt_v2_noreset": None,
}


def warm_start_acc(method: str, af: float, unr: float):
    p = SWEEP_FILES.get(method)
    if p is None:
        return None  # caller will enqueue_trial instead of add_trial
    with open(p) as f:
        d = json.load(f)
    for r in d["grid"]:
        if r["af_ratio"] == af and r["update_noise_ratio"] == unr:
            return float(r["acc"])
    return None


def run_one(method: str, hp: dict, af: float, unr: float) -> float:
    if method == "tikitaka_v1":
        return tt3_mod.run_trial(
            hp["transfer_lr"], hp["fast_lr"], hp["classifier_lr"],
            af_ratio=af, unr=unr,
        )
    if method == "lrtt_v1":
        return v1_mod.run_trial(
            hp["lr"], hp["tlr"], hp["clr"], af_ratio=af, unr=unr,
        )
    if method == "lrtt_v2":
        return v2_mod.run_trial(
            hp["lr"], hp["tlr"], hp["clr"],
            af_ratio=af, update_noise_ratio=unr,
        )
    if method == "tikitaka_reset":
        return reset_mod.run_tikitaka_reset(
            hp["transfer_lr"], hp["fast_lr"], hp["classifier_lr"], af, unr,
        )
    if method == "lrtt_v1_reset":
        return reset_mod.run_lrtt_v1_reset(
            hp["lr"], hp["tlr"], hp["clr"], af, unr,
        )
    if method == "lrtt_v2_noreset":
        return reset_mod.run_lrtt_v2_noreset(
            hp["lr"], hp["tlr"], hp["clr"], af, unr,
        )
    raise ValueError(method)


def cell_path(method: str, af: float, unr: float) -> str:
    return f"{OUT_ROOT}/{method}/af{af:g}_unr{unr:g}.json"


def load_cell(method: str, af: float, unr: float) -> dict:
    p = cell_path(method, af, unr)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {
        "method": method,
        "af_ratio": af,
        "update_noise_ratio": unr,
        "warm_start": None,
        "search_space": {k: {"low": d.low, "high": d.high, "log": d.log}
                          for k, d in SEARCH_SPACE[method].items()},
        "trials": [],
    }


def save_cell(state: dict, method: str, af: float, unr: float) -> None:
    os.makedirs(f"{OUT_ROOT}/{method}", exist_ok=True)
    p = cell_path(method, af, unr)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, p)


def search_cell(method: str, af: float, unr: float, n_trials: int = N_TRIALS) -> dict:
    state = load_cell(method, af, unr)
    distributions = SEARCH_SPACE[method]
    n_done = len(state["trials"])
    if n_done >= n_trials:
        print(f"  [{method}] AF={af}, UNR={unr} — already {n_done}/{n_trials}, skip", flush=True)
        return state

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    if state["warm_start"] is None:
        ws_acc = warm_start_acc(method, af, unr)
        state["warm_start"] = {"hp": dict(BEST_HP[method]), "acc": ws_acc}
        save_cell(state, method, af, unr)
    ws = state["warm_start"]
    if ws.get("acc") is not None:
        # Prior fixed-HP eval exists → seed TPE with the completed trial.
        try:
            study.add_trial(create_trial(
                params=ws["hp"], distributions=distributions, value=ws["acc"],
            ))
        except Exception as e:
            print(f"  [warn] warm-start add failed: {e}", flush=True)
    elif n_done == 0:
        # No prior eval and nothing saved yet → enqueue the baseline HP so the
        # first trial of optimize() evaluates it (counts toward n_trials).
        try:
            study.enqueue_trial(ws["hp"])
        except Exception as e:
            print(f"  [warn] warm-start enqueue failed: {e}", flush=True)

    for t in state["trials"]:
        try:
            study.add_trial(create_trial(
                params=t["hp"], distributions=distributions, value=t["acc"],
            ))
        except Exception as e:
            print(f"  [warn] trial replay failed: {e}", flush=True)

    remaining = n_trials - n_done

    def objective(trial: optuna.Trial) -> float:
        hp = {
            k: trial.suggest_float(k, dist.low, dist.high, log=dist.log)
            for k, dist in distributions.items()
        }
        t0 = time.time()
        acc = run_one(method, hp, af, unr)
        wall = time.time() - t0
        rec = {"hp": hp, "acc": round(float(acc), 4),
               "wall_seconds": round(wall, 1)}
        state["trials"].append(rec)
        save_cell(state, method, af, unr)
        n = len(state["trials"])
        print(f"    trial {n}/{n_trials}: hp={hp} → acc={acc:.2f}  ({wall:.0f}s)",
              flush=True)
        return acc

    study.optimize(objective, n_trials=remaining,
                   gc_after_trial=True, show_progress_bar=False)

    best_acc = ws["acc"] if ws.get("acc") is not None else -float("inf")
    best_hp = ws["hp"]
    for t in state["trials"]:
        if t["acc"] > best_acc:
            best_acc = t["acc"]
            best_hp = t["hp"]
    state["best_acc"] = round(float(best_acc), 4)
    state["best_hp"] = best_hp
    state["n_completed"] = len(state["trials"])
    save_cell(state, method, af, unr)
    return state


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--methods", nargs="+",
        default=["tikitaka_v1", "lrtt_v1", "lrtt_v2"],
        choices=["tikitaka_v1", "lrtt_v1", "lrtt_v2",
                  "tikitaka_reset", "lrtt_v1_reset", "lrtt_v2_noreset"],
    )
    p.add_argument("--n_trials", type=int, default=N_TRIALS)
    p.add_argument("--single_cell", nargs=2, type=float, default=None,
                    metavar=("AF", "UNR"),
                    help="Run only one (AF, UNR) cell (for smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.time()
    print(f"Per-cell TPE search ({args.n_trials} trials/cell) → {OUT_ROOT}", flush=True)
    print(f"Methods: {args.methods}", flush=True)
    if args.single_cell:
        af_list, unr_list = [args.single_cell[0]], [args.single_cell[1]]
        print(f"Single cell: AF={af_list[0]}, UNR={unr_list[0]}", flush=True)
    else:
        af_list, unr_list = AF_GRID, UNR_GRID
        print(f"AF: {af_list}\nUNR: {unr_list}", flush=True)

    for method in args.methods:
        print(f"\n{'='*70}\n  {method.upper()}  warm-start HP={BEST_HP[method]}\n"
              f"  search={ {k: (d.low, d.high) for k, d in SEARCH_SPACE[method].items()} }\n{'='*70}",
              flush=True)
        for af in af_list:
            for unr in unr_list:
                t0 = time.time()
                print(f"\n[{method}] AF={af}, UNR={unr}", flush=True)
                s = search_cell(method, af, unr, n_trials=args.n_trials)
                dt = (time.time() - t0) / 60
                if "best_acc" in s:
                    print(f"  cell done in {dt:.1f}min — best acc={s['best_acc']:.2f}, "
                          f"best HP={s['best_hp']}", flush=True)

    elapsed_h = (time.time() - t_total) / 3600
    print(f"\n{'='*70}\nALL DONE — total wall time: {elapsed_h:.2f} h\n{'='*70}",
          flush=True)


if __name__ == "__main__":
    main()

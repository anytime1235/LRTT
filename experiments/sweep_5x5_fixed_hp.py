#!/usr/bin/env python3
"""5×5 fixed-HP sweep on gamma-AF / 10-bit C (extension of 3×3 grid).

Uses best HP from previous 27-cell HP search:
  Direct       — lr=0.1
  TikiTaka v1  — lr=0.1, tlr=1.0, fast_lr=1.0 (script defaults)
  LRTT-v1      — lr=0.1, tlr=0.001, classifier_lr=0.3 (best mean cell)
  LRTT-v2      — lr=1.0, tlr=0.1, classifier_lr=1.0 (best mean cell)

Grid: AF ∈ {0, 1, 2, 5, 10}, UNR ∈ {0, 1, 3, 5, 10} = 25 trials per method.

Reuses /root/LRTT/results/hp_search_*_gamma_af_noise_10bitC/trial_cache.json
to avoid recomputing the 9 grid points already evaluated → 16 new trials/method.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("LRTT_SILENT", "1")

sys.path.insert(0, "/root/LRTT/experiments")

import hp_search_direct_gamma_af as direct_mod
import hp_search_tikitaka_v1_gamma_af as tt_mod
import hp_search_v1_decay_gamma_af_2stage as v1_mod
import hp_search_v2_rank8_gamma_af_2stage as v2_mod


AF_GRID  = [0.0, 1.0, 2.0, 5.0, 10.0]
UNR_GRID = [0.0, 1.0, 3.0, 5.0, 10.0]

BEST_HP = {
    "direct":   {},
    "tikitaka": {},
    "lrtt_v1":  {"lr": 0.1, "tlr": 0.001, "clr": 0.3},
    "lrtt_v2":  {"lr": 1.0, "tlr": 0.1, "clr": 1.0},
}

CACHE_PATHS = {
    "direct":   "/root/LRTT/results/hp_search_direct_rank8_gamma_af_noise_10bitC/trial_cache.json",
    "tikitaka": "/root/LRTT/results/hp_search_tikitaka_v1_rank8_gamma_af_noise_10bitC/trial_cache.json",
    "lrtt_v1":  "/root/LRTT/results/hp_search_v1_decay_rank8_gamma_af_noise_10bitC/trial_cache.json",
    "lrtt_v2":  "/root/LRTT/results/hp_search_v2_rank8_gamma_af_noise_10bitC/trial_cache.json",
}


def cache_key(method, af, unr):
    if method in ("direct", "tikitaka"):
        return f"af{af}_unr{unr}"
    hp = BEST_HP[method]
    return f"lr{hp['lr']}_tlr{hp['tlr']}_clr{hp['clr']}_af{af}_unr{unr}"


def run_one(method, af, unr):
    if method == "direct":
        return direct_mod.run_trial(af_ratio=af, unr=unr)
    if method == "tikitaka":
        return tt_mod.run_trial(af_ratio=af, unr=unr)
    if method == "lrtt_v1":
        hp = BEST_HP["lrtt_v1"]
        return v1_mod.run_trial(hp["lr"], hp["tlr"], hp["clr"], af, unr)
    if method == "lrtt_v2":
        hp = BEST_HP["lrtt_v2"]
        return v2_mod.run_trial(hp["lr"], hp["tlr"], hp["clr"],
                                af_ratio=af, update_noise_ratio=unr)
    raise ValueError(method)


def run_method(method, out_dir):
    cache_path = CACHE_PATHS[method]
    with open(cache_path) as f:
        cache = json.load(f)
    print(f"\n{'='*70}\n{method.upper()} — best HP: {BEST_HP[method] or '(fixed in script)'}\n"
          f"{'='*70}", flush=True)
    print(f"Cache: {cache_path} ({len(cache)} entries)", flush=True)

    grid = []
    t0 = time.time()
    n_new = 0
    for af in AF_GRID:
        for unr in UNR_GRID:
            k = cache_key(method, af, unr)
            if k in cache:
                acc = cache[k]
                src = "cache"
            else:
                t_start = time.time()
                acc = run_one(method, af, unr)
                cache[k] = acc
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2)
                n_new += 1
                src = f"new ({time.time()-t_start:.0f}s)"
            grid.append({
                "af_ratio": af, "update_noise_ratio": unr,
                "acc": round(acc, 2), "source": src,
            })
            elapsed = (time.time() - t0) / 60
            print(f"  AF={af:>4}, UNR={unr:>4} → {acc:6.2f}%  [{src}]  "
                  f"elapsed={elapsed:.1f}min  new={n_new}", flush=True)

    out_path = f"{out_dir}/{method}.json"
    with open(out_path, "w") as f:
        json.dump({
            "method": method,
            "best_hp": BEST_HP[method],
            "af_grid": AF_GRID,
            "unr_grid": UNR_GRID,
            "wall_seconds": round(time.time() - t0, 1),
            "n_new_trials": n_new,
            "grid": grid,
        }, f, indent=2)
    print(f"\n  Saved {out_path}  (wall={(time.time()-t0)/60:.1f}min, {n_new} new trials)",
          flush=True)
    return grid


def main():
    out_dir = "/root/LRTT/results/sweep_5x5_fixed_hp"
    os.makedirs(out_dir, exist_ok=True)
    print(f"5×5 fixed-HP sweep — output: {out_dir}", flush=True)
    print(f"AF  ∈ {AF_GRID}", flush=True)
    print(f"UNR ∈ {UNR_GRID}", flush=True)

    t_total = time.time()
    all_results = {}
    for method in ["direct", "tikitaka", "lrtt_v1", "lrtt_v2"]:
        all_results[method] = run_method(method, out_dir)

    with open(f"{out_dir}/all_methods.json", "w") as f:
        json.dump({
            "af_grid": AF_GRID,
            "unr_grid": UNR_GRID,
            "best_hp": BEST_HP,
            "results": all_results,
            "wall_seconds_total": round(time.time() - t_total, 1),
        }, f, indent=2)

    elapsed_min = (time.time() - t_total) / 60
    print(f"\n{'='*70}\nALL DONE — total wall time: {elapsed_min:.1f} min\n"
          f"Output: {out_dir}\n{'='*70}", flush=True)


if __name__ == "__main__":
    main()

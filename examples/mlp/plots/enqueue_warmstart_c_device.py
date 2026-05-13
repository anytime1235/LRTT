#!/usr/bin/env python3
"""Pre-fill the c-device sweep study (per-device, per-reinit) with the top-N
best trials from the prior gauss_a/b_zero (a/b/cconstantstepideal) log. The
enqueued trials override rank_exp to the current phase's fixed value.

Usage:
    python enqueue_warmstart_c_device.py \\
        --reinit-mode gauss_b_zero \\
        --c-device idealizedpreset \\
        --rank-exp 0 \\
        --top-n 10
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def parse_old_log(path: Path):
    params_by_trial = defaultdict(dict)
    results = {}
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = r.get("op_code")
            if op == 5:
                params_by_trial[r["trial_id"]][r["param_name"]] = r["param_value_internal"]
            elif op == 6 and r.get("state") == 1:
                results[r["trial_id"]] = r["values"][0]
    return params_by_trial, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reinit-mode", required=True, choices=["gauss_a_zero", "gauss_b_zero"])
    ap.add_argument("--c-device", required=True,
                    choices=["idealizedpreset", "reramespreset", "ecrampreset"])
    ap.add_argument("--rank-exp", type=int, required=True,
                    help="rank_exp for this phase (e.g., 0/2/4/6 → rank 1/4/16/64)")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--transfer-every", type=int, default=10)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--old-log", default=None)
    args = ap.parse_args()

    here = Path(__file__).parent.parent   # plots/ → mlp/
    results_dir = Path(args.results_dir) if args.results_dir else here / "results" / "optuna_mlp_mnist_lrtt"

    old_log = Path(args.old_log) if args.old_log else (
        results_dir
        / f"optuna_mlp_mnist_lrtt_bs64_sgd_{args.reinit_mode}_nowd_nomom_nonest_"
          f"onehot_aconstantstepideal_bconstantstepideal_cconstantstepideal_"
          f"perfect_no-stlr_split-reset_std_linear1_30ep.log"
    )
    if not old_log.exists():
        raise SystemExit(f"old log not found: {old_log}")

    print(f"Reading prior log: {old_log.name}")
    p_by_t, results = parse_old_log(old_log)
    print(f"  COMPLETE trials: {len(results)}")

    ranked = sorted(results.items(), key=lambda kv: -kv[1])[: args.top_n]
    print(f"\nTop {len(ranked)} trials → enqueue (rank_exp={args.rank_exp}, transfer_every={args.transfer_every}):")
    top_params = []
    for tid, val in ranked:
        p = dict(p_by_t[tid])
        p["rank_exp"] = args.rank_exp
        p["transfer_every"] = args.transfer_every
        # Optuna IntDistribution stores as float; coerce so enqueue lookup matches.
        int_keys = ("rank_exp", "transfer_every", "ab_desired_bl",
                    "a_desired_bl", "b_desired_bl", "c_desired_bl",
                    "ab_multilevel", "a_multilevel", "b_multilevel")
        for k in int_keys:
            if k in p:
                p[k] = int(p[k])
        top_params.append((tid, val, p))
        print(f"  trial {tid:>4}  val={val:6.3f}")

    # Build study name matching what optuna_mlp_mnist_lrtt.py auto-derives for
    # --ab-device constantstepideal --c-device {cdev} --is-perfect ... config.
    study_name = (
        f"mlp_mnist_lrtt_bs64_sgd_{args.reinit_mode}_nowd_nomom_nonest_"
        f"onehot_constantstepideal_c{args.c_device}_perfect_"
        f"no-stlr_split-reset_std_linear1_30ep"
    )
    log_path = results_dir / f"optuna_{study_name}.log"
    results_dir.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(log_path)))
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction="maximize", load_if_exists=True,
    )
    for tid, val, p in top_params:
        study.enqueue_trial(p, skip_if_exists=True)
    print(f"\n→ Enqueued {len(top_params)} trials into {log_path.name}")


if __name__ == "__main__":
    main()

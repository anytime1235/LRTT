#!/usr/bin/env python3
"""Pre-fill each af×unr cell study with the top-N best trials from the prior
gauss_a_zero / gauss_b_zero log (constantstepideal, no fixed rank/te). The
enqueued trials override rank_exp and transfer_every to match the new fixed
search space; the remaining hyperparameters carry over as warm-start hints.

Run BEFORE launching sweep_af_unr.sh.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


AF_VALUES = [0, 1, 2, 5, 10]
UNR_VALUES = [0, 1, 3, 5, 10]


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
    ap.add_argument("--reinit-mode", default="gauss_a_zero")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--old-log", default=None,
                    help="Path to the prior log; default: constantstepideal log for the same reinit-mode")
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
    print(f"\nTop {len(ranked)} trials → enqueue (rank_exp=3, transfer_every=10 fixed via suggest_int(v, v)):")
    top_params = []
    for tid, val in ranked:
        p = dict(p_by_t[tid])
        # Override rank_exp / transfer_every to the fixed values used in the
        # current search space (suggest_int(rank_exp, 3, 3), suggest_int(transfer_every, 10, 10)).
        p["rank_exp"] = 3
        p["transfer_every"] = 10
        # Optuna IntDistribution stores as float; coerce so enqueue lookup matches.
        int_keys = ("rank_exp", "transfer_every", "ab_desired_bl",
                    "a_desired_bl", "b_desired_bl", "c_desired_bl",
                    "ab_multilevel", "a_multilevel", "b_multilevel")
        for k in int_keys:
            if k in p:
                p[k] = int(p[k])
        top_params.append((tid, val, p))
        print(f"  trial {tid:>4}  val={val:6.3f}")

    # Enqueue into each of 25 cell studies
    for af in AF_VALUES:
        for unr in UNR_VALUES:
            study_name = (
                f"mlp_mnist_lrtt_bs64_sgd_{args.reinit_mode}_nowd_nomom_nonest_"
                f"onehot_ascaledideal_bscaledideal_cconstantstepideal_perfect_"
                f"no-stlr_af{af}_unr{unr}_split-reset_std_linear1_30ep"
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
            print(f"  af={af} unr={unr}: enqueued {len(top_params)} → {log_path.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate config JSON for re-experiment.

Usage:
  # Step 1: Fixed HP (lr=0.3, tlr=0.005)
  python generate_reexp_config.py

  # Step 1b: tlr = 0.009/sqrt(rank), rank=1 excluded
  python generate_reexp_config.py --tlr_rule sqrt_rank
"""

import argparse
import json
import math

RANKS = [1, 4, 8, 16, 32, 64]
TES = [1, 10, 50, 100, 500, 1000]
LR = 0.3
TLR_FIXED = 0.005
TRIALS_PER_CELL = 3


def get_tlr(rule, rank):
    if rule == "fixed":
        return TLR_FIXED
    elif rule == "sqrt_rank":
        return 0.009 / math.sqrt(rank)
    else:
        raise ValueError(f"Unknown rule: {rule}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tlr_rule", type=str, default="fixed",
                        choices=["fixed", "sqrt_rank"],
                        help="fixed: tlr=0.005 | sqrt_rank: tlr=0.009/sqrt(rank)")
    args = parser.parse_args()

    ranks = RANKS
    if args.tlr_rule == "sqrt_rank":
        ranks = [r for r in RANKS if r > 1]  # rank=1 excluded

    suffix = "" if args.tlr_rule == "fixed" else f"_{args.tlr_rule}"
    output_file = f"reexp_sweep_configs{suffix}.json"

    config = {
        "metadata": {
            "ranks": ranks,
            "lifetime": 0,
            "strategy": args.tlr_rule,
            "lr": LR,
            "tlr": TLR_FIXED if args.tlr_rule == "fixed" else "0.009/sqrt(rank)",
            "trials_per_te": TRIALS_PER_CELL,
            "total_experiments_per_mode": len(ranks) * len(TES) * TRIALS_PER_CELL,
            "note": f"tlr_rule={args.tlr_rule}"
        },
        "decay": {
            "mode": "decay (A and B both decay)",
            "configs": []
        },
        "hybrid": {
            "mode": "hybrid (A=0 hard reset, B unchanged)",
            "configs": []
        }
    }

    print(f"Generating: {output_file}")
    print(f"  tlr_rule: {args.tlr_rule}")
    print(f"  Ranks: {ranks}")
    print(f"  TEs: {TES}")
    print()

    for rank in ranks:
        tlr = get_tlr(args.tlr_rule, rank)
        for te in TES:
            trials = [{"lr": LR, "tlr": round(tlr, 6)} for _ in range(TRIALS_PER_CELL)]
            entry = {
                "te": te,
                "rank": rank,
                "lr_base": LR,
                "tlr_base": round(tlr, 6),
                "trials": trials
            }
            config["decay"]["configs"].append(dict(entry))
            config["hybrid"]["configs"].append(dict(entry))

        print(f"  Rank={rank:>2d}: lr={LR}, tlr={tlr:.6f}")

    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)

    total = len(ranks) * len(TES) * TRIALS_PER_CELL
    print(f"\n  Total per mode: {total} runs")
    print(f"  Total (both modes): {total * 2} runs")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()

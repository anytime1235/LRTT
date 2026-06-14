#!/usr/bin/env python3
"""4-condition x 5-seed replication for fig 6c noise asymmetry analysis.

Conditions (A_DEVICE, B_DEVICE, hyperparams):
  no_noise: (constantstep6t1cgamma, constantstep6t1cgamma)  T249 hyperparams
  a_only:   (6t1c, constantstep6t1cgamma)                   T6 hyperparams (= T249)
  b_only:   (constantstep6t1cgamma, 6t1c)                   T13 hyperparams (D2 best)
  both:     (6t1c, 6t1c)                                    T98 hyperparams (full 6t1c best)

5 seeds (42-46) x 4 conditions = 20 runs.
Source-patches fine_bert_squad_lrtt.py (does NOT touch optuna logs).

Usage:
  python replicate_4conditions.py --parallel --num-gpus 4
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC = REPO / "examples/bert/fine_bert_squad_lrtt.py"
RESULTS_DIR = REPO / "examples/bert/results/BERT_SQUAD_LRTT_FINE"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 4 conditions: A_DEVICE, B_DEVICE, and hyperparams (best-per-condition)
CONDITIONS = {
    "no_noise": dict(
        a_device="constantstep6t1cgamma",
        b_device="constantstep6t1cgamma",
        lr=0.0038, tlr=0.095, te=1, fast_lr=0.474,  # T249
    ),
    "a_only": dict(
        a_device="6t1c",
        b_device="constantstep6t1cgamma",
        lr=0.0038, tlr=0.095, te=1, fast_lr=0.474,  # T6 (D1 best, equals T249)
    ),
    "b_only": dict(
        a_device="constantstep6t1cgamma",
        b_device="6t1c",
        lr=0.003655, tlr=0.08468, te=1, fast_lr=0.3016,  # T13 (D2 best)
    ),
    "both": dict(
        a_device="6t1c",
        b_device="6t1c",
        lr=0.0019, tlr=0.20, te=4, fast_lr=0.092,  # T98 (full 6t1c best)
    ),
}

COMMON = dict(
    rank=32,
    ab_dw_min=0.0004883,
    c_dw_min=0.001953,
    abml=None,
    warmup_steps=365,
    batch_size=48,
)

SEEDS = [42, 43, 44, 45, 46]


def patch(content, key, new_value):
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def apply_config(src, cond, seed):
    """Apply condition-specific config + seed to fine script source."""
    # General settings
    src = patch(src, "N_EPOCHS", "5")
    src = patch(src, "SCHEDULE_EPOCHS", "5")
    src = patch(src, "BATCH_SIZE", str(COMMON["batch_size"]))
    src = patch(src, "WARMUP_STEPS", str(COMMON["warmup_steps"]))
    src = patch(src, "SEED", str(seed))
    src = patch(src, "LRTT_RANK", str(COMMON["rank"]))
    src = patch(src, "AB_DW_MIN", repr(COMMON["ab_dw_min"]))
    src = patch(src, "C_DW_MIN", repr(COMMON["c_dw_min"]))
    src = patch(src, "AB_MULTILEVEL", str(COMMON["abml"]))
    src = patch(src, "REINIT_MODE", '"decay"')
    src = patch(src, "TRANSFER_METHOD", '"onehot"')
    src = patch(src, "C_DEVICE", '"constantstepideal"')
    src = patch(src, "LORA_TARGET", '"qkvo"')
    src = patch(src, "FORWARD_INJECT", "False")
    src = patch(src, "FI_CONTINUOUS_ALPHA", "False")
    src = patch(src, "LEARN_OUT_SCALING", "False")
    src = patch(src, "IS_PERFECT", "True")
    src = patch(src, "OUT_NOISE", "0.0")
    src = patch(src, "ENABLE_DIAGNOSTIC", "False")
    src = patch(src, "TRAIN_SUBSET_SIZE", "0")
    src = patch(src, "EVAL_SUBSET_SIZE", "0")

    # Condition-specific
    src = patch(src, "AB_DEVICE", '"6t1c"')  # placeholder; overridden by A/B
    src = patch(src, "A_DEVICE", f'"{cond["a_device"]}"')
    src = patch(src, "B_DEVICE", f'"{cond["b_device"]}"')
    src = patch(src, "LEARNING_RATE", repr(cond["lr"]))
    src = patch(src, "TRANSFER_LR", repr(cond["tlr"]))
    src = patch(src, "TRANSFER_EVERY", str(cond["te"]))
    src = patch(src, "FAST_LR", repr(cond["fast_lr"]))
    return src


def _parse_f1(log_path):
    f1 = None
    log_text = log_path.read_text(errors="replace")
    for line in reversed(log_text.split("\n")):
        if "Best F1" in line or "best_f1" in line:
            m = re.search(r"(\d+\.\d+)", line)
            if m:
                f1 = float(m.group(1))
                break
    if f1 is None:
        for line in reversed(log_text.split("\n")):
            m = re.search(r"f1[=:]\s*(\d+\.\d+)", line, re.IGNORECASE)
            if m:
                f1 = float(m.group(1))
                break
    return f1


def _prepare_trial(tag, src_patched):
    unique_stamp = f"{RUN_STAMP}_{tag}"
    src_patched = src_patched.replace(
        'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
        f'stamp = f"{unique_stamp}"'
    )
    if unique_stamp not in src_patched:
        raise RuntimeError("Failed to patch stamp")

    tmp = SRC.parent / f"_tmp_replicate_{tag}.py"
    tmp.write_text(src_patched)

    log_path = RESULTS_DIR / f"runlog_replicate_{unique_stamp}.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return tag, tmp, log_path


def run_parallel(trials, num_gpus=4):
    queue = list(trials)
    active = {}
    results = []

    def _launch(gpu_id, tag, src):
        tag, tmp, log_path = _prepare_trial(tag, src)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logf = open(log_path, "wb")
        proc = subprocess.Popen(
            [sys.executable, str(tmp)], cwd=SRC.parent,
            stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
        print(f"  [GPU {gpu_id}] Started: {tag}  (pid={proc.pid}, log={log_path.name})")
        active[gpu_id] = (tag, proc, log_path, tmp, logf)

    def _collect(gpu_id):
        tag, proc, log_path, tmp, logf = active.pop(gpu_id)
        logf.close()
        f1 = _parse_f1(log_path)
        tmp.unlink(missing_ok=True)
        print(f"  [GPU {gpu_id}] Done: {tag}  exit={proc.returncode}, F1={f1}")
        return {"tag": tag, "f1": f1, "exit_code": proc.returncode}

    for gpu_id in range(min(num_gpus, len(queue))):
        tag, src = queue.pop(0)
        _launch(gpu_id, tag, src)

    while active:
        time.sleep(5)
        for gpu_id in list(active.keys()):
            tag, proc, _, _, _ = active[gpu_id]
            if proc.poll() is not None:
                results.append(_collect(gpu_id))
                if queue:
                    next_tag, next_src = queue.pop(0)
                    _launch(gpu_id, next_tag, next_src)

    return results


def run_sequential(trials):
    results = []
    for tag, src in trials:
        tag, tmp, log_path = _prepare_trial(tag, src)
        print(f"\n{'='*60}\nRunning: {tag}\n{'='*60}")
        with open(log_path, "wb") as logf:
            ret = subprocess.run(
                [sys.executable, str(tmp)], cwd=SRC.parent,
                stdout=logf, stderr=subprocess.STDOUT,
            )
        f1 = _parse_f1(log_path)
        tmp.unlink(missing_ok=True)
        print(f"  exit={ret.returncode}, F1={f1}")
        results.append({"tag": tag, "f1": f1, "exit_code": ret.returncode})
    return results


def build_trials():
    trials = []
    base_src = SRC.read_text()
    for cond_name, cond in CONDITIONS.items():
        for seed in SEEDS:
            tag = f"{cond_name}_s{seed}"
            src = apply_config(base_src, cond, seed)
            trials.append((tag, src))
    return trials


def aggregate(raw):
    out = {"timestamp": RUN_STAMP, "common": COMMON, "seeds": SEEDS, "conditions": {}}
    for cond_name, cond in CONDITIONS.items():
        per_seed = {}
        f1s = []
        for r in raw:
            if not r["tag"].startswith(cond_name + "_s"):
                continue
            seed = int(r["tag"].rsplit("_s", 1)[1])
            per_seed[seed] = r["f1"]
            if r["f1"] is not None:
                f1s.append(r["f1"])
        if len(f1s) >= 2:
            mean = sum(f1s) / len(f1s)
            std = (sum((x - mean) ** 2 for x in f1s) / (len(f1s) - 1)) ** 0.5
        elif f1s:
            mean, std = f1s[0], 0.0
        else:
            mean, std = None, None
        out["conditions"][cond_name] = {
            "hyperparams": cond,
            "f1_per_seed": per_seed,
            "mean": mean, "std": std, "n": len(f1s),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="4-condition x 5-seed replication")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--parallel", action="store_true",
                        help="Parallel across GPUs (default if --num-gpus>1)")
    parser.add_argument("--sequential", action="store_true", help="Force sequential")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated subset (e.g. 'no_noise,both'). Default: all 4.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list (e.g. '42,43'). Default: all 5.")
    args = parser.parse_args()

    if args.conditions:
        keep = set(args.conditions.split(","))
        for k in list(CONDITIONS.keys()):
            if k not in keep:
                del CONDITIONS[k]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        SEEDS.clear()
        SEEDS.extend(seeds)

    trials = build_trials()
    print(f"Total trials: {len(trials)} ({len(CONDITIONS)} conditions x {len(SEEDS)} seeds)")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"GPUs: {args.num_gpus} (parallel={'no' if args.sequential else 'yes'})\n")

    if args.sequential or args.num_gpus <= 1:
        raw = run_sequential(trials)
    else:
        raw = run_parallel(trials, num_gpus=args.num_gpus)

    out = aggregate(raw)
    out_path = Path(__file__).parent / f"replicate_4conditions_{RUN_STAMP}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

    print("\nSummary:")
    print(f"  {'condition':<12} {'mean':>7} {'std':>6} {'n':>3}  per-seed F1s")
    for cond_name, c in out["conditions"].items():
        if c["mean"] is not None:
            ps = "  ".join(f"{c['f1_per_seed'].get(s, '-')!s:>5}" for s in SEEDS)
            print(f"  {cond_name:<12} {c['mean']:>7.3f} {c['std']:>6.3f} {c['n']:>3}  {ps}")
        else:
            print(f"  {cond_name:<12} no successful runs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostic replication: 4 conditions x seed=42, ENABLE_DIAGNOSTIC=True.

Same hyperparams + seed across 4 device configs to enable apples-to-apples
comparison of weight/erank/gradient dynamics. Each run dumps:
  squad_diagnostic_log_<stamp>.json  (per-step diagnostics)

Usage:
  python replicate_4conditions_diag.py --num-gpus 4
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC = REPO / "examples/bert/fine_bert_squad_lrtt.py"
RESULTS_DIR = REPO / "examples/bert/results/BERT_SQUAD_LRTT_FINE"
OUT_DIR = Path(__file__).parent / "diag_4conditions_multitile"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Use T6/T249 hyperparams uniformly across conditions for apples-to-apples comparison
# (same hyperparams isolate the noise effect; we already know per-condition optima from
# Phase 1, but this analysis is about *mechanism* not optimum)
HYPER = dict(
    lr=0.0038, tlr=0.095, te=1, fast_lr=0.474,
    rank=32, ab_dw_min=0.0004883, c_dw_min=0.001953,
    abml=None, warmup_steps=365, batch_size=48, seed=42,
)

CONDITIONS = {
    "no_noise": dict(a_device="constantstep6t1cgamma", b_device="constantstep6t1cgamma"),
    "a_only":   dict(a_device="6t1c",                  b_device="constantstep6t1cgamma"),
    "b_only":   dict(a_device="constantstep6t1cgamma", b_device="6t1c"),
    "both":     dict(a_device="6t1c",                  b_device="6t1c"),
}


def patch(content, key, new_value):
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def apply_config(src, cond):
    src = patch(src, "N_EPOCHS", "5")
    src = patch(src, "SCHEDULE_EPOCHS", "5")
    src = patch(src, "BATCH_SIZE", str(HYPER["batch_size"]))
    src = patch(src, "WARMUP_STEPS", str(HYPER["warmup_steps"]))
    src = patch(src, "SEED", str(HYPER["seed"]))
    src = patch(src, "LRTT_RANK", str(HYPER["rank"]))
    src = patch(src, "AB_DW_MIN", repr(HYPER["ab_dw_min"]))
    src = patch(src, "C_DW_MIN", repr(HYPER["c_dw_min"]))
    src = patch(src, "AB_MULTILEVEL", str(HYPER["abml"]))
    src = patch(src, "REINIT_MODE", '"decay"')
    src = patch(src, "TRANSFER_METHOD", '"onehot"')
    src = patch(src, "C_DEVICE", '"constantstepideal"')
    src = patch(src, "LORA_TARGET", '"qkvo"')
    src = patch(src, "FORWARD_INJECT", "False")
    src = patch(src, "FI_CONTINUOUS_ALPHA", "False")
    src = patch(src, "LEARN_OUT_SCALING", "False")
    src = patch(src, "IS_PERFECT", "True")
    src = patch(src, "OUT_NOISE", "0.0")
    # Diagnostic ON; rate-limit erank to every ~10 steps to save SVD time
    src = patch(src, "ENABLE_DIAGNOSTIC", "True")
    src = patch(src, "DIAG_TILES", '"first_last"')  # original Phase 2 schema (first/last tile only); renamed from MULTI_TILE_DIAG
    src = patch(src, "ERANK_RATE_LIMIT_STEPS", "10")
    src = patch(src, "TRAIN_SUBSET_SIZE", "0")
    src = patch(src, "EVAL_SUBSET_SIZE", "0")

    src = patch(src, "AB_DEVICE", '"6t1c"')
    src = patch(src, "A_DEVICE", f'"{cond["a_device"]}"')
    src = patch(src, "B_DEVICE", f'"{cond["b_device"]}"')
    src = patch(src, "LEARNING_RATE", repr(HYPER["lr"]))
    src = patch(src, "TRANSFER_LR", repr(HYPER["tlr"]))
    src = patch(src, "TRANSFER_EVERY", str(HYPER["te"]))
    src = patch(src, "FAST_LR", repr(HYPER["fast_lr"]))
    # Enable weight histogram (hist_A / hist_B / hist_C_eff at HIST_RATE_STEPS cadence)
    src = src.replace(
        '"g3c_weight_hist": False,',
        '"g3c_weight_hist": True,',
        1,
    )
    return src


def _parse_f1(log_path):
    f1 = None
    for line in reversed(log_path.read_text(errors="replace").split("\n")):
        if "Best F1" in line or "best_f1" in line:
            m = re.search(r"(\d+\.\d+)", line)
            if m:
                f1 = float(m.group(1)); break
    return f1


def _prepare_trial(tag, src_patched):
    # Make stamp unique across conditions so squad_diagnostic_log_<stamp>.json is per-condition
    unique_stamp = f"diag_4cond_{RUN_STAMP}_{tag}"
    src_patched = src_patched.replace(
        'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
        f'stamp = f"{unique_stamp}"'
    )
    if unique_stamp not in src_patched:
        raise RuntimeError("Failed to patch stamp")

    tmp = SRC.parent / f"_tmp_replicate_diag_{tag}.py"
    tmp.write_text(src_patched)
    log_path = RESULTS_DIR / f"runlog_diag_{unique_stamp}.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return tag, tmp, log_path, unique_stamp


def run_parallel(trials, num_gpus=4):
    queue = list(trials)
    active = {}
    results = []

    def _launch(gpu_id, tag, src):
        tag, tmp, log_path, stamp = _prepare_trial(tag, src)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logf = open(log_path, "wb")
        proc = subprocess.Popen(
            [sys.executable, str(tmp)], cwd=SRC.parent,
            stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
        print(f"  [GPU {gpu_id}] Started: {tag}  (pid={proc.pid})")
        active[gpu_id] = (tag, proc, log_path, tmp, logf, stamp)

    def _collect(gpu_id):
        tag, proc, log_path, tmp, logf, stamp = active.pop(gpu_id)
        logf.close()
        f1 = _parse_f1(log_path)
        # Find squad_diagnostic_log JSON
        diag_json = RESULTS_DIR / f"squad_diagnostic_log_{stamp}.json"
        diag_target = OUT_DIR / f"diag_{tag}.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if diag_json.exists():
            shutil.copy(diag_json, diag_target)
            print(f"  [GPU {gpu_id}] Done: {tag}  exit={proc.returncode}, F1={f1}, diag-> {diag_target.name}")
        else:
            print(f"  [GPU {gpu_id}] Done: {tag}  exit={proc.returncode}, F1={f1}, !!! NO DIAG JSON ({diag_json})")
        tmp.unlink(missing_ok=True)
        return {"tag": tag, "f1": f1, "exit_code": proc.returncode,
                "diag_json": str(diag_target) if diag_json.exists() else None}

    for gpu_id in range(min(num_gpus, len(queue))):
        tag, src = queue.pop(0)
        _launch(gpu_id, tag, src)
    while active:
        time.sleep(5)
        for gpu_id in list(active.keys()):
            tag, proc, *_ = active[gpu_id]
            if proc.poll() is not None:
                results.append(_collect(gpu_id))
                if queue:
                    next_tag, next_src = queue.pop(0)
                    _launch(gpu_id, next_tag, next_src)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=4)
    args = parser.parse_args()

    base_src = SRC.read_text()
    trials = [(name, apply_config(base_src, cond)) for name, cond in CONDITIONS.items()]
    print(f"Trials: {[t[0] for t in trials]}")
    print(f"Hyperparams: {HYPER}")
    print(f"Output dir: {OUT_DIR}\n")

    raw = run_parallel(trials, num_gpus=args.num_gpus)
    summary = {"timestamp": RUN_STAMP, "hyperparams": HYPER,
               "conditions": CONDITIONS, "results": raw}
    summary_path = OUT_DIR / f"summary_{RUN_STAMP}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nSummary saved: {summary_path}")
    print("\nResults:")
    for r in sorted(raw, key=lambda x: x["tag"]):
        print(f"  {r['tag']:<12}  F1={r['f1']}  diag={r['diag_json']}")


if __name__ == "__main__":
    main()

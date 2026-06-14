#!/usr/bin/env python3
"""Investigate post-pull regression hypothesis — is the L11_output explosion
caused by lrtt_controller.py code changes pulled in via commit 463188d
(`Add gauss_a/selector reinit modes + per-tile mapping/desired_bl override`)?

Round 1: 4 parallel runs, all no_noise condition with multi-tile diag:
  GPU 0: post-pull controller (current HEAD), seed=42
  GPU 1: post-pull controller (current HEAD), seed=43
  GPU 2: pre-pull  controller (463188d^),    seed=42
  GPU 3: pre-pull  controller (463188d^),    seed=43

The pre-pull runs use importlib magic to override the lrtt_controller module
with the pre-pull copy at /root/LRTT/src/aihwkit/simulator/tiles/lrtt_controller_prepull.py
WITHOUT modifying the on-disk post-pull file.

Output: diag_prepull_vs_current/diag_<tag>.json
"""
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
PREPULL_PATH = REPO / "src/aihwkit/simulator/tiles/lrtt_controller_prepull.py"
OUT_DIR = Path(__file__).parent / "diag_prepull_vs_current"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if not PREPULL_PATH.exists():
    raise FileNotFoundError(f"Pre-pull controller not at {PREPULL_PATH}. "
                            f"Extract via: git show 463188d^:src/aihwkit/simulator/tiles/lrtt_controller.py > {PREPULL_PATH}")

# All variants share these hyperparams (T6/T249 region)
HYPER = dict(
    lr=0.0038, tlr=0.095, te=1, fast_lr=0.474,
    rank=32, ab_dw_min=0.0004883, c_dw_min=0.001953,
    abml=None, warmup_steps=365, batch_size=48,
    min_lr_rate=0.0,
)

# 4 variants
VARIANTS = [
    dict(tag="postpull_seed42", seed=42, controller="postpull"),
    dict(tag="postpull_seed43", seed=43, controller="postpull"),
    dict(tag="prepull_seed42",  seed=42, controller="prepull"),
    dict(tag="prepull_seed43",  seed=43, controller="prepull"),
]

# importlib magic to override lrtt_controller module — prepended to pre-pull _tmp scripts
PREPULL_HEADER = f"""# Override lrtt_controller with pre-pull (commit 463188d^) version
import importlib.util as _imp_util
import sys as _sys
_spec = _imp_util.spec_from_file_location(
    "aihwkit.simulator.tiles.lrtt_controller",
    "{PREPULL_PATH}",
)
_module = _imp_util.module_from_spec(_spec)
_sys.modules["aihwkit.simulator.tiles.lrtt_controller"] = _module
_spec.loader.exec_module(_module)
print("[PREPULL_OVERRIDE] lrtt_controller loaded from {PREPULL_PATH}")
del _imp_util, _sys, _spec, _module
"""


def patch(content, key, new_value):
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def apply_config(src, variant):
    src = patch(src, "N_EPOCHS", "5")
    src = patch(src, "SCHEDULE_EPOCHS", "5")
    src = patch(src, "BATCH_SIZE", str(HYPER["batch_size"]))
    src = patch(src, "WARMUP_STEPS", str(HYPER["warmup_steps"]))
    src = patch(src, "SEED", str(variant["seed"]))
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
    src = patch(src, "ENABLE_DIAGNOSTIC", "True")
    src = patch(src, "MULTI_TILE_DIAG", "True")
    src = patch(src, "ERANK_RATE_LIMIT_STEPS", "10")
    src = patch(src, "TRAIN_SUBSET_SIZE", "0")
    src = patch(src, "EVAL_SUBSET_SIZE", "0")
    src = patch(src, "MIN_LR_RATE", repr(HYPER["min_lr_rate"]))

    # All variants use no_noise device config
    src = patch(src, "AB_DEVICE", '"6t1c"')
    src = patch(src, "A_DEVICE", '"constantstep6t1cgamma"')
    src = patch(src, "B_DEVICE", '"constantstep6t1cgamma"')
    src = patch(src, "LEARNING_RATE", repr(HYPER["lr"]))
    src = patch(src, "TRANSFER_LR", repr(HYPER["tlr"]))
    src = patch(src, "TRANSFER_EVERY", str(HYPER["te"]))
    src = patch(src, "FAST_LR", repr(HYPER["fast_lr"]))
    return src


def _prepare(variant, src):
    tag = variant["tag"]
    unique_stamp = f"prepull_check_{RUN_STAMP}_{tag}"
    src = src.replace(
        'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
        f'stamp = f"{unique_stamp}"'
    )
    if unique_stamp not in src:
        raise RuntimeError(f"Failed to patch stamp for {tag}")

    # Prepend importlib magic if pre-pull
    if variant["controller"] == "prepull":
        src = PREPULL_HEADER + "\n" + src

    tmp = SRC.parent / f"_tmp_prepull_{tag}.py"
    tmp.write_text(src)
    log_path = RESULTS_DIR / f"runlog_{unique_stamp}.txt"
    return unique_stamp, tmp, log_path


def main():
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)
    base_src = SRC.read_text()

    print(f"Run stamp: {RUN_STAMP}")
    print(f"Pre-pull controller: {PREPULL_PATH}")
    print(f"Post-pull controller (HEAD): {REPO}/src/aihwkit/simulator/tiles/lrtt_controller.py")
    print(f"Output dir: {OUT_DIR}")
    print(f"Launching {len(VARIANTS)} variants on GPU 0..{len(VARIANTS)-1}\n")

    procs = []
    for gpu_id, variant in enumerate(VARIANTS):
        src = apply_config(base_src, variant)
        unique_stamp, tmp, log_path = _prepare(variant, src)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logf = open(log_path, "wb")
        proc = subprocess.Popen(
            [sys.executable, str(tmp)], cwd=SRC.parent,
            stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
        print(f"  [GPU {gpu_id}] launched: {variant['tag']:<22} controller={variant['controller']:<8} pid={proc.pid}")
        procs.append(dict(gpu=gpu_id, tag=variant["tag"], proc=proc, tmp=tmp,
                          log=log_path, stamp=unique_stamp, logf=logf,
                          controller=variant["controller"]))

    print(f"\nAll {len(procs)} processes launched. Waiting for completion...\n")

    results = []
    while procs:
        time.sleep(15)
        still_running = []
        for p in procs:
            if p["proc"].poll() is not None:
                p["logf"].close()
                diag_json = RESULTS_DIR / f"squad_diagnostic_log_{p['stamp']}.json"
                target = OUT_DIR / f"diag_{p['tag']}.json"
                if diag_json.exists():
                    shutil.copy(diag_json, target)
                    print(f"  [GPU {p['gpu']}] Done: {p['tag']}  exit={p['proc'].returncode}  -> {target.name}")
                else:
                    print(f"  [GPU {p['gpu']}] Done: {p['tag']}  exit={p['proc'].returncode}  !!! NO DIAG JSON ({diag_json})")
                p["tmp"].unlink(missing_ok=True)
                results.append(dict(tag=p["tag"], controller=p["controller"],
                                    exit_code=p["proc"].returncode,
                                    diag_json=str(target) if diag_json.exists() else None))
            else:
                still_running.append(p)
        procs = still_running

    summary = {"timestamp": RUN_STAMP, "hyper": HYPER, "variants": VARIANTS, "results": results}
    summary_path = OUT_DIR / f"summary_{RUN_STAMP}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()

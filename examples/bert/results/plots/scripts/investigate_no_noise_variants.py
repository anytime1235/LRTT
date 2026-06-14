#!/usr/bin/env python3
"""Investigate no_noise F1 collapse — 4-variant parallel experiment.

Launches 4 no_noise runs on GPU 0..3, all with MULTI_TILE_DIAG=True so all
48 LRTT tiles are tracked. Variants:

  GPU 0: seed=42                    — baseline reproducer
  GPU 1: seed=43                    — reproducibility test
  GPU 2: seed=44                    — reproducibility test
  GPU 3: seed=42, min_lr_rate=0.01  — tests if LR→0 is the trigger

Output: diag_no_noise_variants/diag_<tag>.json
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
OUT_DIR = Path(__file__).parent / "diag_no_noise_variants"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

HYPER_BASE = dict(
    lr=0.0038, tlr=0.095, te=1, fast_lr=0.474,
    rank=32, ab_dw_min=0.0004883, c_dw_min=0.001953,
    abml=None, warmup_steps=365, batch_size=48,
    min_lr_rate=0.0,  # default linear decay to 0
)

# 4 variants — same condition (no_noise = both A,B = constantstep6t1cgamma)
VARIANTS = [
    dict(tag="seed42",            seed=42, min_lr_rate=0.0),
    dict(tag="seed43",            seed=43, min_lr_rate=0.0),
    dict(tag="seed44",            seed=44, min_lr_rate=0.0),
    dict(tag="seed42_lrfloor001", seed=42, min_lr_rate=0.01),
]


def patch(content, key, new_value):
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def apply_config(src, variant):
    src = patch(src, "N_EPOCHS", "5")
    src = patch(src, "SCHEDULE_EPOCHS", "5")
    src = patch(src, "BATCH_SIZE", str(HYPER_BASE["batch_size"]))
    src = patch(src, "WARMUP_STEPS", str(HYPER_BASE["warmup_steps"]))
    src = patch(src, "SEED", str(variant["seed"]))
    src = patch(src, "LRTT_RANK", str(HYPER_BASE["rank"]))
    src = patch(src, "AB_DW_MIN", repr(HYPER_BASE["ab_dw_min"]))
    src = patch(src, "C_DW_MIN", repr(HYPER_BASE["c_dw_min"]))
    src = patch(src, "AB_MULTILEVEL", str(HYPER_BASE["abml"]))
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
    src = patch(src, "MIN_LR_RATE", repr(variant["min_lr_rate"]))

    # All variants use no_noise device config: A=B=constantstep6t1cgamma
    src = patch(src, "AB_DEVICE", '"6t1c"')   # placeholder (ignored when A_DEVICE/B_DEVICE set)
    src = patch(src, "A_DEVICE", '"constantstep6t1cgamma"')
    src = patch(src, "B_DEVICE", '"constantstep6t1cgamma"')
    src = patch(src, "LEARNING_RATE", repr(HYPER_BASE["lr"]))
    src = patch(src, "TRANSFER_LR", repr(HYPER_BASE["tlr"]))
    src = patch(src, "TRANSFER_EVERY", str(HYPER_BASE["te"]))
    src = patch(src, "FAST_LR", repr(HYPER_BASE["fast_lr"]))
    return src


def _prepare(variant, src):
    tag = variant["tag"]
    unique_stamp = f"investigate_variants_{RUN_STAMP}_{tag}"
    src = src.replace(
        'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
        f'stamp = f"{unique_stamp}"'
    )
    if unique_stamp not in src:
        raise RuntimeError(f"Failed to patch stamp for {tag}")
    tmp = SRC.parent / f"_tmp_invvar_{tag}.py"
    tmp.write_text(src)
    log_path = RESULTS_DIR / f"runlog_{unique_stamp}.txt"
    return unique_stamp, tmp, log_path


def main():
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)
    base_src = SRC.read_text()

    print(f"Run stamp: {RUN_STAMP}")
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
        print(f"  [GPU {gpu_id}] launched: {variant['tag']}  pid={proc.pid}  stamp={unique_stamp}")
        procs.append(dict(gpu=gpu_id, tag=variant["tag"], proc=proc, tmp=tmp,
                          log=log_path, stamp=unique_stamp, logf=logf))

    print(f"\nAll {len(procs)} processes launched. Waiting for completion...")

    # Poll until all done
    results = []
    while procs:
        time.sleep(15)
        still_running = []
        for p in procs:
            if p["proc"].poll() is not None:
                p["logf"].close()
                # Find the diag JSON
                diag_json = RESULTS_DIR / f"squad_diagnostic_log_{p['stamp']}.json"
                target = OUT_DIR / f"diag_{p['tag']}.json"
                if diag_json.exists():
                    shutil.copy(diag_json, target)
                    print(f"  [GPU {p['gpu']}] Done: {p['tag']}  exit={p['proc'].returncode}  -> {target.name}")
                else:
                    print(f"  [GPU {p['gpu']}] Done: {p['tag']}  exit={p['proc'].returncode}  !!! NO DIAG JSON")
                p["tmp"].unlink(missing_ok=True)
                results.append(dict(tag=p["tag"], exit_code=p["proc"].returncode,
                                    diag_json=str(target) if diag_json.exists() else None))
            else:
                still_running.append(p)
        procs = still_running

    summary = {"timestamp": RUN_STAMP, "hyper_base": HYPER_BASE,
               "variants": VARIANTS, "results": results}
    summary_path = OUT_DIR / f"summary_{RUN_STAMP}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Investigate why no_noise's F1 collapses at epoch 5 despite weights being
stationary in the first/last LRTT tile.

Re-runs only the no_noise condition with MULTI_TILE_DIAG=True so all 48 LRTT
tiles (12 layers × qkvo) are tracked. The hypothesis is that some middle-layer
tile diverges in epoch 5 while first/last stay calm.

Output: diag_no_noise_multitile/diag_no_noise_multitile.json
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC = REPO / "examples/bert/fine_bert_squad_lrtt.py"
RESULTS_DIR = REPO / "examples/bert/results/BERT_SQUAD_LRTT_FINE"
OUT_DIR = Path(__file__).parent / "diag_no_noise_multitile"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Same hyperparams as the 4-condition diag (T6/T249 region).
HYPER = dict(
    lr=0.0038, tlr=0.095, te=1, fast_lr=0.474,
    rank=32, ab_dw_min=0.0004883, c_dw_min=0.001953,
    abml=None, warmup_steps=365, batch_size=48, seed=42,
)

# no_noise: A=B=constantstep6t1cgamma (gamma asymmetry, no stochastic write noise)
COND = dict(a_device="constantstep6t1cgamma", b_device="constantstep6t1cgamma")


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
    # Diagnostics: ENABLE all + MULTI_TILE_DIAG=True (key change)
    src = patch(src, "ENABLE_DIAGNOSTIC", "True")
    src = patch(src, "MULTI_TILE_DIAG", "True")           # ← THIS is the key change
    src = patch(src, "ERANK_RATE_LIMIT_STEPS", "10")
    src = patch(src, "TRAIN_SUBSET_SIZE", "0")
    src = patch(src, "EVAL_SUBSET_SIZE", "0")

    src = patch(src, "AB_DEVICE", '"6t1c"')   # placeholder
    src = patch(src, "A_DEVICE", f'"{cond["a_device"]}"')
    src = patch(src, "B_DEVICE", f'"{cond["b_device"]}"')
    src = patch(src, "LEARNING_RATE", repr(HYPER["lr"]))
    src = patch(src, "TRANSFER_LR", repr(HYPER["tlr"]))
    src = patch(src, "TRANSFER_EVERY", str(HYPER["te"]))
    src = patch(src, "FAST_LR", repr(HYPER["fast_lr"]))
    return src


def main():
    base_src = SRC.read_text()
    src = apply_config(base_src, COND)

    unique_stamp = f"investigate_no_noise_multitile_{RUN_STAMP}"
    src = src.replace(
        'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
        f'stamp = f"{unique_stamp}"'
    )
    if unique_stamp not in src:
        raise RuntimeError("Failed to patch stamp")

    tmp = SRC.parent / f"_tmp_investigate_no_noise.py"
    tmp.write_text(src)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"runlog_{unique_stamp}.txt"

    print(f"Running no_noise with MULTI_TILE_DIAG=True ...")
    print(f"  stamp     : {unique_stamp}")
    print(f"  tmp script: {tmp}")
    print(f"  log       : {log_path}")
    print(f"  out dir   : {OUT_DIR}")

    with open(log_path, "wb") as logf:
        ret = subprocess.run(
            [sys.executable, str(tmp)], cwd=SRC.parent,
            stdout=logf, stderr=subprocess.STDOUT,
        )

    # Find squad_diagnostic_log JSON
    diag_json = RESULTS_DIR / f"squad_diagnostic_log_{unique_stamp}.json"
    diag_target = OUT_DIR / "diag_no_noise_multitile.json"
    if diag_json.exists():
        shutil.copy(diag_json, diag_target)
        print(f"\nDone. exit={ret.returncode}, diag-> {diag_target}")
    else:
        print(f"\nDone but no diag JSON found at {diag_json}, exit={ret.returncode}")
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

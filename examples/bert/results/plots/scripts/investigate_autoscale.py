#!/usr/bin/env python3
"""Direct hypothesis test: AUTO_SCALE_MODE="separate" prevents bilinear instability collapse.

Theoretical analysis:
  - AUTO_SCALE_MODE="none" (baseline): ΔA = -fast_lr · G · B^T, no ||B|| normalization
                                       → bilinear feedback unbounded → stochastic collapse
  - AUTO_SCALE_MODE="separate": lr_eff_a = fast_lr/(m_xb·m_d), m_xb = EMA of |XB|_max ~ |x|·||B||
                                → ||ΔA|| ~ fast_lr × m_batch (constant, no ||B|| factor)
                                → bilinear amplification fully bounded

4 GPUs × 4 seeds, all AUTO_SCALE_MODE="separate":
  GPU 0: seed=42  (already collapsed in baseline post-pull, multi-tile diag, prepull check)
  GPU 1: seed=43  (stable in baseline)
  GPU 2: seed=44  (stable in baseline)
  GPU 3: seed=45  (untested seed)

Prediction: 0/4 F1 collapse if hypothesis correct.
Falsification: 1+/4 collapse → bilinear hypothesis wrong or AUTO_SCALE not enough.

Output: diag_autoscale/diag_<tag>.json
"""
import datetime, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC = REPO / "examples/bert/fine_bert_squad_lrtt.py"
RESULTS_DIR = REPO / "examples/bert/results/BERT_SQUAD_LRTT_FINE"
OUT_DIR = Path(__file__).parent / "diag_autoscale"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Same hyperparams as known-collapse-prone setup
HYPER = dict(
    lr=0.0038, tlr=0.095, te=1, fast_lr=0.474,
    rank=32, ab_dw_min=0.0004883, c_dw_min=0.001953,
    abml=None, warmup_steps=365, batch_size=48,
    min_lr_rate=0.0,
)

VARIANTS = [
    dict(tag="separate_seed42", seed=42, autoscale="separate"),
    dict(tag="separate_seed43", seed=43, autoscale="separate"),
    dict(tag="separate_seed44", seed=44, autoscale="separate"),
    dict(tag="separate_seed45", seed=45, autoscale="separate"),
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
    src = patch(src, "ERANK_RATE_LIMIT_STEPS", "99999")  # effectively disable erank — not needed for collapse detection
    src = patch(src, "TRAIN_SUBSET_SIZE", "0")
    src = patch(src, "EVAL_SUBSET_SIZE", "0")
    src = patch(src, "MIN_LR_RATE", repr(HYPER["min_lr_rate"]))

    # no_noise condition (where collapse is observed)
    src = patch(src, "AB_DEVICE", '"6t1c"')
    src = patch(src, "A_DEVICE", '"constantstep6t1cgamma"')
    src = patch(src, "B_DEVICE", '"constantstep6t1cgamma"')
    src = patch(src, "LEARNING_RATE", repr(HYPER["lr"]))
    src = patch(src, "TRANSFER_LR", repr(HYPER["tlr"]))
    src = patch(src, "TRANSFER_EVERY", str(HYPER["te"]))
    src = patch(src, "FAST_LR", repr(HYPER["fast_lr"]))

    # ★ THE KEY VARIATION
    src = patch(src, "AUTO_SCALE_MODE", f'"{variant["autoscale"]}"')
    return src


def _prepare(variant, src):
    tag = variant["tag"]
    unique_stamp = f"autoscale_{RUN_STAMP}_{tag}"
    src = src.replace(
        'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
        f'stamp = f"{unique_stamp}"'
    )
    if unique_stamp not in src:
        raise RuntimeError(f"Failed to patch stamp for {tag}")
    tmp = SRC.parent / f"_tmp_autoscale_{tag}.py"
    tmp.write_text(src)
    log_path = RESULTS_DIR / f"runlog_{unique_stamp}.txt"
    return unique_stamp, tmp, log_path


def main():
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)
    base_src = SRC.read_text()

    print(f"Run stamp: {RUN_STAMP}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Hypothesis: AUTO_SCALE_MODE='separate' prevents bilinear instability")
    print(f"  shared mode:   may not be sufficient (still has ||B|| factor)")
    print(f"  separate mode: predicted to fully bound updates\n")

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
        print(f"  [GPU {gpu_id}] launched: {variant['tag']:<22}  pid={proc.pid}")
        procs.append(dict(gpu=gpu_id, tag=variant["tag"], proc=proc, tmp=tmp,
                          log=log_path, stamp=unique_stamp, logf=logf,
                          autoscale=variant["autoscale"]))

    print(f"\nAll {len(procs)} processes launched. Waiting...\n")

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
                    print(f"  [GPU {p['gpu']}] Done: {p['tag']}  exit={p['proc'].returncode}")
                else:
                    print(f"  [GPU {p['gpu']}] Done: {p['tag']}  exit={p['proc'].returncode}  !!! NO DIAG JSON")
                p["tmp"].unlink(missing_ok=True)
                results.append(dict(tag=p["tag"], autoscale=p["autoscale"],
                                    exit_code=p["proc"].returncode,
                                    diag_json=str(target) if diag_json.exists() else None))
            else:
                still_running.append(p)
        procs = still_running

    summary = {"timestamp": RUN_STAMP, "hyper": HYPER,
               "variants": VARIANTS, "results": results}
    (OUT_DIR / f"summary_{RUN_STAMP}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone.")


if __name__ == "__main__":
    main()

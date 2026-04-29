#!/usr/bin/env python3
"""2D sweep: gamma_ratio × w_max based on T249 (rank=32, F1=84.06).

Purpose: study interaction between
  - gamma_up/gamma_down (soft bound from device asymmetry)
  - w_max (hard bound from device clipping)

Both are properties of the analog A/B tile device. By independently varying
each, we can disentangle their contributions to learning behavior.

T249 baseline = (gamma_ratio=1.0, w_max=1.0) → constantstep6t1cgamma with default
abml=None (which means w_max=1.0 in the linearstep family, abml=12 effective).

Device used: linearstepideal (no noise; only gamma + w_max varied).
Noise params kept at 0 throughout.

** Bit count is FIXED at abml=12 (T249's effective resolution). **
To vary w_max while keeping bit count fixed, we vary ab_dw_min instead:
  Device formula: w_max = 2^abml × dw_min / 2  →  dw_min = 2 × w_max / 2^abml
  With abml = 12 (4096 levels):
    w_max = 0.0312 → dw_min = 1.524e-5
    w_max = 0.0624 → dw_min = 3.047e-5
    w_max = 0.125  → dw_min = 6.104e-5
    w_max = 0.25   → dw_min = 1.221e-4
    w_max = 0.5    → dw_min = 2.441e-4
    w_max = 1.0    → dw_min = 4.883e-4   (matches T249 default)

Caveat: varying dw_min changes per-step update granularity together with
w_max. Bit count (number of levels) is held constant; the device's
"resolution per range" stays the same — only the absolute physical scale
of weights and updates changes proportionally.

Sweep grid: 5 (gamma) × 6 (w_max) = 30 trials.
T249 baseline reused at (gamma=1.0, w_max=1.0).

Usage:
  python run_sweep_gamma_wmax_t249.py --parallel --num-gpus 4
  python run_sweep_gamma_wmax_t249.py [no parallel = sequential]

Output: sweep_gamma_wmax_t249_{TIMESTAMP}.json
"""
import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC = REPO / "examples/bert/fine_bert_squad_lrtt.py"
RESULTS_DIR = REPO / "examples/bert/results/BERT_SQUAD_LRTT_FINE"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── T249 base config (qkvo study, rank=32, F1=84.06) ──
T249 = dict(
    lr=0.0038, tlr=0.095, te=1, rank=32,
    fast_lr=0.474, ab_dw_min=0.0004883, c_dw_min=0.001953,
    abml=None, warmup_steps=365, batch_size=48,
    seed=42,
)

# ── 6T1C reference values ──
REF_GAMMA_UP = -0.1678
REF_GAMMA_DOWN = 0.1410

# ── Fixed bit count (T249's effective resolution) ──
ABML_FIXED = 12  # 2^12 = 4096 levels — same as T249 baseline

# ── Sweep grid ──
GAMMA_RATIOS = [0.0, 0.5, 1.0, 2.0, 5.0]
W_MAX_TARGETS = [0.0312, 0.0624, 0.125, 0.25, 0.5, 1.0]

# T249 baseline F1 at (gamma_ratio=1.0, w_max=1.0)
T249_BASELINE_F1 = 84.06
T249_BASELINE_TRIAL = "qkvo_T249"


def patch(content: str, key: str, new_value: str) -> str:
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def wmax_to_dw_min(w_max: float, abml: int = ABML_FIXED) -> float:
    """Compute ab_dw_min for target w_max with bit count fixed.

    Formula: w_max = 2^abml * dw_min / 2  →  dw_min = 2 * w_max / 2^abml
    """
    return 2.0 * w_max / (2 ** abml)


def apply_t249_base(src: str, ab_dw_min: float) -> str:
    """Apply T249 base params; ab_dw_min varies per trial, abml fixed."""
    cfg = T249
    src = patch(src, "N_EPOCHS", "5")
    src = patch(src, "SCHEDULE_EPOCHS", "5")
    src = patch(src, "BATCH_SIZE", str(cfg["batch_size"]))
    src = patch(src, "WARMUP_STEPS", str(cfg["warmup_steps"]))
    src = patch(src, "SEED", str(cfg["seed"]))
    src = patch(src, "LEARNING_RATE", repr(cfg["lr"]))
    src = patch(src, "TRANSFER_LR", repr(cfg["tlr"]))
    src = patch(src, "TRANSFER_EVERY", str(cfg["te"]))
    src = patch(src, "LRTT_RANK", str(cfg["rank"]))
    src = patch(src, "FAST_LR", repr(cfg["fast_lr"]))
    src = patch(src, "AB_DW_MIN", repr(ab_dw_min))  # varies per trial
    src = patch(src, "C_DW_MIN", repr(cfg["c_dw_min"]))
    src = patch(src, "AB_MULTILEVEL", str(ABML_FIXED))  # FIXED at 12 bits
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
    return src


def _nz(v):
    return 0.0 if v == 0.0 else v


def _build_linearstep_block(gamma_ratio: float):
    """Build a clean LinearStepDevice constructor with custom gamma, no noise.
    w_max comes from caller (controlled via AB_MULTILEVEL)."""
    gamma_up = _nz(REF_GAMMA_UP * gamma_ratio)
    gamma_down = _nz(REF_GAMMA_DOWN * gamma_ratio)
    return f"""    if name == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max, w_min=w_min,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            gamma_up={gamma_up}, gamma_down={gamma_down},
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            write_noise_std=0.0, reset_std=0.0,
            up_down=0.0, mult_noise=False,
            lifetime=lifetime,
        )"""


ORIG_LINEARSTEPIDEAL = """    if name == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max, w_min=w_min,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            write_noise_std=0.0, reset_std=0.0,
            up_down=0.0, mult_noise=False,
            lifetime=lifetime,
        )"""


def _parse_f1(log_path: Path):
    log_text = log_path.read_text(errors="replace")
    for line in reversed(log_text.split("\n")):
        if "Best F1" in line or "best_f1" in line:
            m = re.search(r"(\d+\.\d+)", line)
            if m:
                return float(m.group(1))
    return None


def _build_trials():
    """Build all (tag, src) pairs for the 2D grid (skip the T249 baseline point)."""
    trials = []
    base_template = SRC.read_text()
    for w_max in W_MAX_TARGETS:
        ab_dw_min = wmax_to_dw_min(w_max)
        for gamma_r in GAMMA_RATIOS:
            # Skip T249 baseline point (gamma=1.0, w_max=1.0)
            if abs(gamma_r - 1.0) < 1e-9 and abs(w_max - 1.0) < 1e-9:
                continue
            tag = f"g{gamma_r:.1f}_w{w_max:.4f}".replace(".", "p")
            src = apply_t249_base(base_template, ab_dw_min=ab_dw_min)
            src = patch(src, "AB_DEVICE", '"linearstepideal"')
            src = src.replace(ORIG_LINEARSTEPIDEAL, _build_linearstep_block(gamma_r))
            if f"gamma_up={_nz(REF_GAMMA_UP * gamma_r)}" not in src:
                raise RuntimeError(f"Failed to patch gamma for ratio={gamma_r}")
            trials.append((tag, src, gamma_r, w_max, ab_dw_min))
    return trials


def _prepare_trial(tag: str, src_patched: str):
    unique_stamp = f"{RUN_STAMP}_{tag}"
    src_patched = src_patched.replace(
        'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
        f'stamp = f"{unique_stamp}"'
    )
    if unique_stamp not in src_patched:
        raise RuntimeError("Failed to patch stamp")
    tmp = SRC.parent / f"_tmp_sweep2d_{tag}.py"
    tmp.write_text(src_patched)
    log_path = RESULTS_DIR / f"runlog_sweep2d_{unique_stamp}.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return tag, tmp, log_path


def run_one(tag: str, src_patched: str) -> dict:
    tag, tmp, log_path = _prepare_trial(tag, src_patched)
    print(f"\n{'='*60}\nRunning: {tag}  (log -> {log_path})\n{'='*60}")
    with open(log_path, "wb") as logf:
        ret = subprocess.run([sys.executable, str(tmp)], cwd=SRC.parent,
                             stdout=logf, stderr=subprocess.STDOUT)
    f1 = _parse_f1(log_path)
    tmp.unlink(missing_ok=True)
    print(f"  exit={ret.returncode}, F1={f1}")
    return {"tag": tag, "f1": f1, "exit_code": ret.returncode}


def run_parallel(trials, num_gpus=4):
    import time
    queue = list(trials)
    active = {}  # gpu_id -> (tag, proc, log_path, tmp_path, logf)
    results = []

    def _launch(gpu_id, tag, src_patched):
        tag, tmp, log_path = _prepare_trial(tag, src_patched)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logf = open(log_path, "wb")
        proc = subprocess.Popen([sys.executable, str(tmp)], cwd=SRC.parent,
                                stdout=logf, stderr=subprocess.STDOUT, env=env)
        print(f"  [GPU {gpu_id}] Started: {tag}  (pid={proc.pid})")
        active[gpu_id] = (tag, proc, log_path, tmp, logf)

    def _collect(gpu_id):
        tag, proc, log_path, tmp, logf = active.pop(gpu_id)
        logf.close()
        f1 = _parse_f1(log_path)
        tmp.unlink(missing_ok=True)
        print(f"  [GPU {gpu_id}] Done: {tag}  exit={proc.returncode}  F1={f1}")
        return {"tag": tag, "f1": f1, "exit_code": proc.returncode}

    # Initial launches
    for gpu_id in range(min(num_gpus, len(queue))):
        _launch(gpu_id, *queue.pop(0)[:2])

    while active:
        time.sleep(10)
        for gpu_id in list(active):
            tag, proc, log_path, tmp, logf = active[gpu_id]
            if proc.poll() is not None:
                results.append(_collect(gpu_id))
                if queue:
                    _launch(gpu_id, *queue.pop(0)[:2])
    return results


def main():
    parser = argparse.ArgumentParser(description="2D sweep: gamma_ratio × w_max (T249 base)")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=4)
    args = parser.parse_args()

    trials = _build_trials()
    print(f"\n2D grid: {len(GAMMA_RATIOS)} gamma × {len(W_MAX_TARGETS)} w_max")
    print(f"Total to run: {len(trials)} (skipping T249 baseline at gamma=1.0, w_max=1.0)")
    print(f"Estimated time on {args.num_gpus} GPUs: ~{len(trials) * 30 / max(args.num_gpus, 1):.0f}min")

    # Print grid
    print("\nGrid (gamma × w_max):")
    for gr in GAMMA_RATIOS:
        for wm in W_MAX_TARGETS:
            mark = " (T249)" if (abs(gr - 1.0) < 1e-9 and abs(wm - 1.0) < 1e-9) else ""
            print(f"  gamma={gr:.1f}, w_max={wm:.4f}{mark}")

    if args.parallel:
        raw = run_parallel([(t[0], t[1]) for t in trials], num_gpus=args.num_gpus)
    else:
        raw = [run_one(t[0], t[1]) for t in trials]

    # Map results back with grid info
    tag_to_grid = {t[0]: (t[2], t[3], t[4]) for t in trials}
    data = []
    for r in raw:
        gr, wm, dwm = tag_to_grid[r["tag"]]
        data.append({"gamma_ratio": gr, "w_max": wm,
                     "abml": ABML_FIXED, "ab_dw_min": dwm,
                     "gamma_up": REF_GAMMA_UP * gr, "gamma_down": REF_GAMMA_DOWN * gr,
                     "f1": r["f1"], "exit_code": r["exit_code"]})

    # Inject T249 baseline at (gamma=1.0, w_max=1.0)
    data.append({"gamma_ratio": 1.0, "w_max": 1.0,
                 "abml": ABML_FIXED, "ab_dw_min": T249["ab_dw_min"],
                 "gamma_up": REF_GAMMA_UP, "gamma_down": REF_GAMMA_DOWN,
                 "f1": T249_BASELINE_F1, "exit_code": 0,
                 "source": T249_BASELINE_TRIAL})
    data.sort(key=lambda x: (x["gamma_ratio"], x["w_max"]))

    out = {
        "gamma_wmax_2d_sweep": {
            "title": "Gamma Ratio × w_max 2D Sweep (T249 base, fixed bit count)",
            "xlabel": "Gamma Ratio",
            "ylabel": "w_max",
            "f1_label": "F1",
            "gamma_ratios": GAMMA_RATIOS,
            "w_max_targets": W_MAX_TARGETS,
            "abml_fixed": ABML_FIXED,
            "data": data,
            "source": {"base": "T249 (rank=32, F1=84.06)", "timestamp": RUN_STAMP},
        }
    }

    out_path = RESULTS_DIR / f"sweep_gamma_wmax_t249_{RUN_STAMP}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (rows=gamma, cols=w_max)")
    print("=" * 70)
    label = "gamma\\w_max"
    header = f"{label:>12}" + "".join(f"{wm:>10.4f}" for wm in W_MAX_TARGETS)
    print(header)
    for gr in GAMMA_RATIOS:
        row = f"{gr:>12.1f}"
        for wm in W_MAX_TARGETS:
            entry = next((d for d in data if abs(d["gamma_ratio"]-gr)<1e-9 and abs(d["w_max"]-wm)<1e-9), None)
            v = entry["f1"] if entry and entry["f1"] is not None else "FAIL"
            row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.2f}"
        print(row)


if __name__ == "__main__":
    main()

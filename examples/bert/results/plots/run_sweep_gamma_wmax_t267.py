#!/usr/bin/env python3
"""2D sweep: gamma_ratio x w_max based on T267 (constantstepideal, rank=32, F1=84.98).

Purpose: study interaction between gamma (soft bound) and w_max (hard bound)
for the constantstepideal baseline device.

T267 baseline = (gamma_ratio=0, w_max=0.0624) -> constantstepideal with abml=10.

** Bit count FIXED at abml=10 (T267's resolution, 1024 levels). **
To vary w_max while keeping bit count fixed, ab_dw_min varies:
  dw_min = 2 * w_max / 2^abml
  With abml = 10:
    w_max = 0.0312 -> dw_min = 6.094e-5
    w_max = 0.0624 -> dw_min = 1.219e-4  (T267 baseline)
    w_max = 0.125  -> dw_min = 2.441e-4
    w_max = 0.25   -> dw_min = 4.883e-4
    w_max = 0.5    -> dw_min = 9.766e-4
    w_max = 1.0    -> dw_min = 1.953e-3

Cross-reuse with noise_gamma sweep:
  (gamma=0, w_max=0.0624) = T267 baseline (F1=84.98, injected)
  (gamma=1.0, w_max=0.0624) = gamma_sweep[1.0] = noise_sweep[0] (from noise_gamma sweep)

Usage:
  python run_sweep_gamma_wmax_t267.py --parallel --num-gpus 4

Output: sweep_gamma_wmax_t267_{TIMESTAMP}.json
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent.parent
SRC = REPO / "examples/bert/fine_bert_squad_lrtt.py"
RESULTS_DIR = REPO / "examples/bert/results/BERT_SQUAD_LRTT_FINE"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── T267 base config ──
T267 = dict(
    lr=0.00328, tlr=0.25, te=2, rank=32,
    fast_lr=0.3, ab_dw_min=0.0001210937, c_dw_min=0.001953,
    abml=10, warmup_steps=365, batch_size=48,
    seed=42,
)

# ── 6T1C reference values ──
REF_GAMMA_UP = -0.1678
REF_GAMMA_DOWN = 0.1410

# ── Fixed bit count ──
ABML_FIXED = 10

# ── Sweep grid ──
GAMMA_RATIOS = [0.0, 0.5, 1.0, 2.0, 5.0]
W_MAX_TARGETS = [0.0312, 0.0624, 0.125, 0.25, 0.5, 1.0]

T267_BASELINE_F1 = 84.98
T267_BASELINE_TRIAL = "abml_qkvo_T267"


def patch(content: str, key: str, new_value: str) -> str:
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def wmax_to_dw_min(w_max: float, abml: int = ABML_FIXED) -> float:
    return 2.0 * w_max / (2 ** abml)


def apply_t267_base(src: str, ab_dw_min: float) -> str:
    cfg = T267
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
    src = patch(src, "AB_DW_MIN", repr(ab_dw_min))
    src = patch(src, "C_DW_MIN", repr(cfg["c_dw_min"]))
    src = patch(src, "AB_MULTILEVEL", str(ABML_FIXED))
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
    gamma_up = _nz(REF_GAMMA_UP * gamma_ratio)
    gamma_down = _nz(REF_GAMMA_DOWN * gamma_ratio)
    return f"""    if AB_DEVICE == "linearstepideal":
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


ORIG_LINEARSTEPIDEAL = """    if AB_DEVICE == "linearstepideal":
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
            if "f1=" in line.lower() or "exact_match" in line.lower():
                m = re.search(r"f1[=:]\s*(\d+\.\d+)", line, re.IGNORECASE)
                if m:
                    f1 = float(m.group(1))
                    break
    return f1


def _build_trials():
    trials = []
    base_template = SRC.read_text()
    for w_max in W_MAX_TARGETS:
        ab_dw_min = wmax_to_dw_min(w_max)
        for gamma_r in GAMMA_RATIOS:
            # Skip T267 baseline (gamma=0, w_max=0.0624)
            if abs(gamma_r - 0.0) < 1e-9 and abs(w_max - 0.0624) < 1e-4:
                continue
            tag = f"g{gamma_r:.1f}_w{w_max:.4f}".replace(".", "p")
            src = apply_t267_base(base_template, ab_dw_min=ab_dw_min)
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
    active = {}
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
    parser = argparse.ArgumentParser(description="2D sweep: gamma_ratio x w_max (T267 base)")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=4)
    args = parser.parse_args()

    trials = _build_trials()
    print(f"\n2D grid: {len(GAMMA_RATIOS)} gamma x {len(W_MAX_TARGETS)} w_max")
    print(f"Total to run: {len(trials)} (skipping T267 baseline at gamma=0, w_max=0.0624)")

    if args.parallel:
        raw = run_parallel([(t[0], t[1]) for t in trials], num_gpus=args.num_gpus)
    else:
        raw = [run_one(t[0], t[1]) for t in trials]

    tag_to_grid = {t[0]: (t[2], t[3], t[4]) for t in trials}
    data = []
    for r in raw:
        gr, wm, dwm = tag_to_grid[r["tag"]]
        data.append({"gamma_ratio": gr, "w_max": wm,
                     "abml": ABML_FIXED, "ab_dw_min": dwm,
                     "gamma_up": REF_GAMMA_UP * gr, "gamma_down": REF_GAMMA_DOWN * gr,
                     "f1": r["f1"], "exit_code": r["exit_code"]})

    # Inject T267 baseline at (gamma=0, w_max=0.0624)
    data.append({"gamma_ratio": 0.0, "w_max": 0.0624,
                 "abml": ABML_FIXED, "ab_dw_min": T267["ab_dw_min"],
                 "gamma_up": 0.0, "gamma_down": 0.0,
                 "f1": T267_BASELINE_F1, "exit_code": 0,
                 "source": T267_BASELINE_TRIAL})
    data.sort(key=lambda x: (x["gamma_ratio"], x["w_max"]))

    out = {
        "gamma_wmax_2d_sweep": {
            "title": "Gamma Ratio x w_max 2D Sweep (T267 constantstepideal base, fixed 10bit)",
            "xlabel": "Gamma Ratio",
            "ylabel": "w_max",
            "f1_label": "F1",
            "gamma_ratios": GAMMA_RATIOS,
            "w_max_targets": W_MAX_TARGETS,
            "abml_fixed": ABML_FIXED,
            "data": data,
            "source": {"base": f"T267 (rank=32, F1={T267_BASELINE_F1})", "timestamp": RUN_STAMP},
        }
    }

    out_path = RESULTS_DIR / f"sweep_gamma_wmax_t267_{RUN_STAMP}.json"
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
            entry = next((d for d in data if abs(d["gamma_ratio"]-gr)<1e-9 and abs(d["w_max"]-wm)<1e-4), None)
            v = entry["f1"] if entry and entry["f1"] is not None else "FAIL"
            row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.2f}"
        print(row)


if __name__ == "__main__":
    main()

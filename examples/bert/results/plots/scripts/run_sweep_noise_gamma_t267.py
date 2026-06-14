#!/usr/bin/env python3
"""Noise ratio & Gamma ratio sweep based on T267 (constantstepideal, rank=32, F1=84.98).

T267 base config (qkvo abml study, constantstepideal device):
  lr=0.00328, tlr=0.25, te=2, flr=0.3
  ab_dw_min=1.2109e-4, c_dw_min=0.001953
  abml=10 (w_max=0.0624), rank=32, lora_target=qkvo

6T1C reference values (ratio=1.0):
  Device noise (dtod, std):
    dw_min_dtod=0.1, dw_min_std=0.3
    up_down_dtod=0.01
    w_max_dtod=0.05, w_min_dtod=0.05
    gamma_up_dtod=0.05, gamma_down_dtod=0.05
    write_noise_std=0.0
  Asymmetry (gamma):
    gamma_up=-0.1678, gamma_down=0.1410

Noise ratio sweep:
  Gamma FIXED at 6T1C (gamma_ratio=1.0). Only noise params scale with ratio.
  noise_ratio=0 → gamma=6T1C, no noise (= gamma_ratio=1.0 point in gamma sweep)

Gamma ratio sweep:
  Only gamma scaled. All noise params = 0.
  gamma_ratio=0 → constantstepideal baseline = T267 (F1=84.98)
  gamma_ratio=1.0 → 6T1C gamma, no noise (= noise_ratio=0 point)

Cross-reuse:
  gamma_sweep[gamma=0] = T267 baseline (F1=84.98, injected)
  gamma_sweep[gamma=1.0] = noise_sweep[noise=0] (run once, shared)

Usage:
  python run_sweep_noise_gamma_t267.py --parallel --num-gpus 4
  python run_sweep_noise_gamma_t267.py --noise-only --parallel
  python run_sweep_noise_gamma_t267.py --gamma-only --parallel

Output: sweep_noise_gamma_t267_{TIMESTAMP}.json
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

# ── T267 base config (abml qkvo study, constantstepideal, F1=84.98) ──
T267 = dict(
    lr=0.00328, tlr=0.25, te=2, rank=32,
    fast_lr=0.3, ab_dw_min=0.0001210937, c_dw_min=0.001953,
    abml=10, warmup_steps=365, batch_size=48,
    seed=42,
)

# ── 6T1C reference values (ratio=1.0) ──
REF_GAMMA_UP = -0.1678
REF_GAMMA_DOWN = 0.1410
REF_DW_MIN_DTOD = 0.1
REF_DW_MIN_STD = 0.3
REF_UP_DOWN_DTOD = 0.01
REF_W_MAX_DTOD = 0.05
REF_W_MIN_DTOD = 0.05
REF_GAMMA_UP_DTOD = 0.05
REF_GAMMA_DOWN_DTOD = 0.05
REF_WRITE_NOISE_STD = 0.0

# ── Sweep points ──
NOISE_RATIOS = [0.1, 0.3, 0.5, 0.7, 1.0]
# gamma=0 → T267 baseline (injected), gamma=1.0 → shared with noise_ratio=0 (run in gamma sweep)
GAMMA_RATIOS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

T267_BASELINE_F1 = 84.98
T267_BASELINE_TRIAL = "abml_qkvo_T267"


def patch(content: str, key: str, new_value: str) -> str:
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def apply_t267_base(src: str) -> str:
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
    src = patch(src, "AB_DW_MIN", repr(cfg["ab_dw_min"]))
    src = patch(src, "C_DW_MIN", repr(cfg["c_dw_min"]))
    src = patch(src, "AB_MULTILEVEL", str(cfg["abml"]))
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


def _build_linearstep_block(ratio, gamma_ratio=None):
    gr = gamma_ratio if gamma_ratio is not None else ratio
    dw_min_dtod = _nz(REF_DW_MIN_DTOD * ratio)
    dw_min_std = _nz(REF_DW_MIN_STD * ratio)
    up_down_dtod = _nz(REF_UP_DOWN_DTOD * ratio)
    w_max_dtod = _nz(REF_W_MAX_DTOD * ratio)
    w_min_dtod = _nz(REF_W_MIN_DTOD * ratio)
    gamma_up = _nz(REF_GAMMA_UP * gr)
    gamma_down = _nz(REF_GAMMA_DOWN * gr)
    gamma_up_dtod = _nz(REF_GAMMA_UP_DTOD * ratio)
    gamma_down_dtod = _nz(REF_GAMMA_DOWN_DTOD * ratio)
    write_noise_std = _nz(REF_WRITE_NOISE_STD * ratio)
    mult_noise = 'False'
    return f"""    if name == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max, w_min=w_min,
            dw_min_dtod={dw_min_dtod}, dw_min_std={dw_min_std},
            up_down_dtod={up_down_dtod}, w_max_dtod={w_max_dtod}, w_min_dtod={w_min_dtod},
            gamma_up={gamma_up}, gamma_down={gamma_down},
            gamma_up_dtod={gamma_up_dtod}, gamma_down_dtod={gamma_down_dtod},
            write_noise_std={write_noise_std}, reset_std=0.0,
            up_down=0.0, mult_noise={mult_noise},
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


def _prepare_trial(tag: str, src_patched: str):
    unique_stamp = f"{RUN_STAMP}_{tag}"
    src_patched = src_patched.replace(
        'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
        f'stamp = f"{unique_stamp}"'
    )
    if unique_stamp not in src_patched:
        raise RuntimeError("Failed to patch stamp")
    tmp = SRC.parent / f"_tmp_sweep_{tag}.py"
    tmp.write_text(src_patched)
    log_path = RESULTS_DIR / f"runlog_sweep_{unique_stamp}.txt"
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
        print(f"  [GPU {gpu_id}] Done: {tag}  exit={proc.returncode}, F1={f1}")
        return {"tag": tag, "f1": f1, "exit_code": proc.returncode}

    for gpu_id in range(min(num_gpus, len(queue))):
        _launch(gpu_id, *queue.pop(0))

    while active:
        time.sleep(5)
        for gpu_id in list(active):
            tag, proc, log_path, tmp, logf = active[gpu_id]
            if proc.poll() is not None:
                results.append(_collect(gpu_id))
                if queue:
                    _launch(gpu_id, *queue.pop(0))
    return results


def _build_noise_trials():
    trials = []
    base_src = apply_t267_base(SRC.read_text())
    base_src = patch(base_src, "AB_DEVICE", '"linearstepideal"')
    for ratio in NOISE_RATIOS:
        tag = f"noise_r{ratio:.1f}".replace(".", "p")
        src = base_src.replace(ORIG_LINEARSTEPIDEAL, _build_linearstep_block(ratio, gamma_ratio=1.0))
        if f"dw_min_dtod={_nz(REF_DW_MIN_DTOD * ratio)}" not in src:
            raise RuntimeError(f"Failed to patch for noise ratio={ratio}")
        trials.append((tag, src))
    return trials


def _build_gamma_trials():
    trials = []
    base_src = apply_t267_base(SRC.read_text())
    base_src = patch(base_src, "AB_DEVICE", '"linearstepideal"')
    for ratio in GAMMA_RATIOS:
        tag = f"gamma_r{ratio:.1f}".replace(".", "p")
        src = base_src.replace(ORIG_LINEARSTEPIDEAL, _build_linearstep_block(ratio=0.0, gamma_ratio=ratio))
        if f"gamma_up={_nz(REF_GAMMA_UP * ratio)}" not in src:
            raise RuntimeError(f"Failed to patch gamma for ratio={ratio}")
        trials.append((tag, src))
    return trials


def main():
    parser = argparse.ArgumentParser(description="Noise & Gamma sweep (T267 constantstepideal base)")
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument("--gamma-only", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=4)
    args = parser.parse_args()

    do_noise = not args.gamma_only
    do_gamma = not args.noise_only

    if args.parallel:
        all_trials = []
        if do_gamma:
            all_trials.extend(_build_gamma_trials())
        if do_noise:
            all_trials.extend(_build_noise_trials())
        print(f"\nParallel: {len(all_trials)} trials on {args.num_gpus} GPUs")
        raw = run_parallel(all_trials, num_gpus=args.num_gpus)
        noise_results = []
        gamma_results = []
        for r in raw:
            if r["tag"].startswith("noise_"):
                r["noise_ratio"] = float(r["tag"].replace("noise_r", "").replace("p", "."))
                noise_results.append(r)
            elif r["tag"].startswith("gamma_"):
                ratio = float(r["tag"].replace("gamma_r", "").replace("p", "."))
                r["gamma_ratio"] = ratio
                r["gamma_up"] = REF_GAMMA_UP * ratio
                r["gamma_down"] = REF_GAMMA_DOWN * ratio
                gamma_results.append(r)
    else:
        noise_results = []
        gamma_results = []
        if do_gamma:
            gamma_results = [run_one(t, s) for t, s in _build_gamma_trials()]
            for r in gamma_results:
                ratio = float(r["tag"].replace("gamma_r", "").replace("p", "."))
                r["gamma_ratio"] = ratio
                r["gamma_up"] = REF_GAMMA_UP * ratio
                r["gamma_down"] = REF_GAMMA_DOWN * ratio
        if do_noise:
            noise_results = [run_one(t, s) for t, s in _build_noise_trials()]
            for r in noise_results:
                r["noise_ratio"] = float(r["tag"].replace("noise_r", "").replace("p", "."))

    # Find gamma=1.0 result (shared with noise_ratio=0)
    gamma1_f1 = None
    for r in gamma_results:
        if abs(r["gamma_ratio"] - 1.0) < 1e-9:
            gamma1_f1 = r["f1"]
            break

    all_results = {}
    if noise_results or gamma1_f1 is not None:
        # noise_ratio=0 = gamma=1.0 (reuse from gamma sweep)
        data = []
        if gamma1_f1 is not None:
            data.append({"noise_ratio": 0.0, "f1": gamma1_f1, "source": "gamma_r1.0 (shared)"})
        data += [{"noise_ratio": r["noise_ratio"], "f1": r["f1"]} for r in noise_results]
        data.sort(key=lambda x: x["noise_ratio"])
        all_results["noise_ratio_sweep"] = {
            "title": "Noise Ratio Sweep: F1 vs Noise Ratio",
            "xlabel": "Noise Ratio",
            "ylabel": "F1",
            "data": data,
            "source": {"base": f"T267 (rank=32, abml=10, F1={T267_BASELINE_F1})", "timestamp": RUN_STAMP}
        }
    if gamma_results:
        # gamma=0 = T267 baseline (constantstepideal, injected)
        data = [{"gamma_ratio": 0.0, "f1": T267_BASELINE_F1, "source": T267_BASELINE_TRIAL}]
        data += [{"gamma_ratio": r["gamma_ratio"], "f1": r["f1"]} for r in gamma_results]
        data.sort(key=lambda x: x["gamma_ratio"])
        all_results["asymmetry_ratio_sweep"] = {
            "title": "Asymmetry Ratio Sweep: F1 vs Gamma Ratio",
            "xlabel": "Gamma Ratio",
            "ylabel": "F1",
            "data": data,
            "source": {"base": f"T267 (rank=32, abml=10, F1={T267_BASELINE_F1})", "timestamp": RUN_STAMP}
        }

    out_path = RESULTS_DIR / f"sweep_noise_gamma_t267_{RUN_STAMP}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if gamma1_f1 is not None:
        print(f"\nnoise_ratio=0 (= gamma=1.0): F1={gamma1_f1}")
    if noise_results:
        print("\nNoise Ratio Sweep:")
        for r in sorted(noise_results, key=lambda x: x["noise_ratio"]):
            print(f"  ratio={r['noise_ratio']:.1f}  F1={r['f1']}")
    if gamma_results:
        print(f"\ngamma_ratio=0 (= T267 baseline): F1={T267_BASELINE_F1}")
        print("\nGamma Ratio Sweep:")
        for r in sorted(gamma_results, key=lambda x: x["gamma_ratio"]):
            print(f"  ratio={r['gamma_ratio']:.1f}  F1={r['f1']}")


if __name__ == "__main__":
    main()

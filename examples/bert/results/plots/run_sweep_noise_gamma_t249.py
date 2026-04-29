#!/usr/bin/env python3
"""Noise ratio & Gamma ratio sweep based on T249 (rank=32, F1=84.06).

T249 base config (qkvo study with constantstep6t1cgamma device):
  flr=0.474, te=1, tlr=0.095, lr=0.0038
  ab_dw_min=0.0004883, c_dw_min=0.001953
  rank=32, no ab_multilevel, lora_target=qkvo

Generates patched copies of fine_bert_squad_lrtt.py and runs them in parallel
across GPUs. Each run produces F1; results saved to a JSON file for plotting.

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
  Gamma is FIXED at T249's baseline (gamma_ratio=1.0, full 6T1C asymmetry).
  Only noise params (dtod, std) scale with ratio.
  Uses linearstepideal device (string-replaced).
  ratio=0.0 → T249 baseline = constantstep6t1cgamma equivalent (gamma=1.0, no noise)
  ratio=1.0 → T249 baseline + full 6T1C noise on top
  → measures how device noise degrades T249's F1

Gamma ratio sweep:
  Only gamma_up/gamma_down scaled by ratio. All noise params = 0.
  Uses linearstepideal device (string-replaced).
  ratio=0.0 → symmetric (no asymmetry, equivalent to constantstepideal)
  ratio=1.0 → T249 baseline (6T1C asymmetry, no noise)
  ratio>1.0 → larger asymmetry than 6T1C
  → measures how asymmetry magnitude affects F1

Usage:
  # Sequential (one GPU at a time)
  CUDA_VISIBLE_DEVICES=0 python run_sweep_noise_gamma_t249.py [--noise-only] [--gamma-only]

  # Parallel across GPUs (async task queue)
  python run_sweep_noise_gamma_t249.py --parallel [--num-gpus 4]
  python run_sweep_noise_gamma_t249.py --parallel --noise-only --num-gpus 4
  python run_sweep_noise_gamma_t249.py --parallel --gamma-only --num-gpus 4

Output: sweep_noise_gamma_t249_{TIMESTAMP}.json
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

# ── T249 base config (qkvo study, rank=32, F1=84.06, ab_device=constantstep6t1cgamma) ──
# Source command: --batch-size 48 --epochs 5 --warmup-steps 365 --transfer-method onehot
#                 --ab-device constantstep6t1cgamma --c-device constantstepideal
#                 --is-perfect --no-learn-out-scaling --lora-target qkvo
#                 --optimizer AnalogAdam --reinit-mode decay
#                 --no-wd --no-momentum --no-nesterov
# Trial params: flr=0.474, te=1, tlr=0.095, lr=0.0038
#               ab_dw_min=0.0004883, c_dw_min=0.001953
#               (no --ab-multilevel; AB_MULTILEVEL=None)
T249 = dict(
    lr=0.0038, tlr=0.095, te=1, rank=32,
    fast_lr=0.474, ab_dw_min=0.0004883, c_dw_min=0.001953,
    abml=None, warmup_steps=365, batch_size=48,
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
# noise_ratio=0.0 and gamma_ratio=1.0 omitted because they both equal T249's
# exact config (constantstep6t1cgamma, no noise) → reuse T249_BASELINE_F1.
NOISE_RATIOS = [2.0, 3.0, 5.0, 10.0]
GAMMA_RATIOS = [0.0, 0.5, 2.0, 3.0, 5.0, 10.0]

# T249 baseline F1 (qkvo study, optuna trial T249, F1=84.06)
T249_BASELINE_F1 = 84.06
T249_BASELINE_TRIAL = "qkvo_T249"


def patch(content: str, key: str, new_value: str) -> str:
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def apply_t249_base(src: str) -> str:
    """Apply T249 base params to script source — exactly matches qkvo trial T249."""
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
    src = patch(src, "AB_DW_MIN", repr(cfg["ab_dw_min"]))
    src = patch(src, "C_DW_MIN", repr(cfg["c_dw_min"]))
    src = patch(src, "AB_MULTILEVEL", str(cfg["abml"]))  # None
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
    """Normalize -0.0 to 0.0 for clean output."""
    return 0.0 if v == 0.0 else v


def _build_linearstep_block(ratio, gamma_ratio=None):
    """Build a LinearStepDevice constructor string with scaled 6T1C params.

    If gamma_ratio is provided, use it for gamma scaling (independent of ratio).
    Otherwise gamma scales with ratio like everything else.
    """
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


# Original linearstepideal block in fine_bert_squad_lrtt.py (for string replacement)
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
    """Parse best F1 from a run log file."""
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
    """Write patched script and return (tag, tmp_path, log_path)."""
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
    """Write patched script, run it synchronously, parse F1 from output."""
    tag, tmp, log_path = _prepare_trial(tag, src_patched)

    print(f"\n{'='*60}")
    print(f"Running: {tag}  (log -> {log_path})")
    print(f"{'='*60}")

    with open(log_path, "wb") as logf:
        ret = subprocess.run(
            [sys.executable, str(tmp)], cwd=SRC.parent,
            stdout=logf, stderr=subprocess.STDOUT
        )

    f1 = _parse_f1(log_path)
    tmp.unlink(missing_ok=True)
    print(f"  exit={ret.returncode}, F1={f1}")
    return {"tag": tag, "f1": f1, "exit_code": ret.returncode}


def run_parallel(trials: list[tuple[str, str]], num_gpus: int = 4) -> list[dict]:
    """Run trials in parallel across GPUs with async task queue.

    Args:
        trials: list of (tag, src_patched) tuples
        num_gpus: number of GPUs to use (0..num_gpus-1)

    Returns:
        list of result dicts with tag, f1, exit_code
    """
    import time

    queue = list(trials)  # remaining trials
    active = {}  # gpu_id -> (tag, proc, log_path, tmp_path, logf)
    results = []

    def _launch(gpu_id, tag, src_patched):
        tag, tmp, log_path = _prepare_trial(tag, src_patched)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logf = open(log_path, "wb")
        proc = subprocess.Popen(
            [sys.executable, str(tmp)], cwd=SRC.parent,
            stdout=logf, stderr=subprocess.STDOUT, env=env
        )
        print(f"  [GPU {gpu_id}] Started: {tag}  (pid={proc.pid}, log={log_path})")
        active[gpu_id] = (tag, proc, log_path, tmp, logf)

    def _collect(gpu_id):
        tag, proc, log_path, tmp, logf = active.pop(gpu_id)
        logf.close()
        f1 = _parse_f1(log_path)
        tmp.unlink(missing_ok=True)
        print(f"  [GPU {gpu_id}] Done: {tag}  exit={proc.returncode}, F1={f1}")
        return {"tag": tag, "f1": f1, "exit_code": proc.returncode}

    # Fill initial GPU slots
    for gpu_id in range(min(num_gpus, len(queue))):
        tag, src = queue.pop(0)
        _launch(gpu_id, tag, src)

    # Poll and refill
    while active:
        time.sleep(5)
        for gpu_id in list(active.keys()):
            tag, proc, log_path, tmp, logf = active[gpu_id]
            if proc.poll() is not None:
                results.append(_collect(gpu_id))
                if queue:
                    next_tag, next_src = queue.pop(0)
                    _launch(gpu_id, next_tag, next_src)

    return results


def _build_noise_trials():
    """Build (tag, src) pairs for noise ratio sweep.

    Gamma is FIXED at T249's baseline (gamma_ratio=1.0, full 6T1C asymmetry).
    Only noise params (dtod, std) scale with ratio.
    """
    trials = []
    base_src = apply_t249_base(SRC.read_text())
    base_src = patch(base_src, "AB_DEVICE", '"linearstepideal"')

    for ratio in NOISE_RATIOS:
        tag = f"noise_r{ratio:.1f}".replace(".", "p")
        # gamma_ratio=1.0 fixed (T249 baseline gamma) — only noise scales
        src = base_src.replace(ORIG_LINEARSTEPIDEAL, _build_linearstep_block(ratio, gamma_ratio=1.0))
        if f"dw_min_dtod={_nz(REF_DW_MIN_DTOD * ratio)}" not in src:
            raise RuntimeError(f"Failed to patch linearstepideal for noise ratio={ratio}")
        trials.append((tag, src))
    return trials


def _build_gamma_trials():
    """Build (tag, src) pairs for gamma ratio sweep."""
    trials = []
    base_src = apply_t249_base(SRC.read_text())
    base_src = patch(base_src, "AB_DEVICE", '"linearstepideal"')

    for ratio in GAMMA_RATIOS:
        tag = f"gamma_r{ratio:.1f}".replace(".", "p")
        new_block = _build_linearstep_block(ratio=0.0, gamma_ratio=ratio)
        src = base_src.replace(ORIG_LINEARSTEPIDEAL, new_block)
        if f"gamma_up={_nz(REF_GAMMA_UP * ratio)}" not in src:
            raise RuntimeError(f"Failed to patch gamma for ratio={ratio}")
        trials.append((tag, src))
    return trials


def noise_sweep(parallel=False, num_gpus=4):
    """Sweep noise params by ratio (gamma fixed at T249's 1.0)."""
    trials = _build_noise_trials()

    if parallel:
        raw = run_parallel(trials, num_gpus=num_gpus)
    else:
        raw = [run_one(tag, src) for tag, src in trials]

    for r in raw:
        ratio = float(r["tag"].replace("noise_r", "").replace("p", "."))
        r["noise_ratio"] = ratio
        print(f"  noise_ratio={ratio}, F1={r['f1']}")
    return raw


def gamma_sweep(parallel=False, num_gpus=4):
    """Sweep gamma_up/gamma_down only. All noise params = 0."""
    trials = _build_gamma_trials()

    if parallel:
        raw = run_parallel(trials, num_gpus=num_gpus)
    else:
        raw = [run_one(tag, src) for tag, src in trials]

    for r in raw:
        ratio = float(r["tag"].replace("gamma_r", "").replace("p", "."))
        r["gamma_ratio"] = ratio
        r["gamma_up"] = REF_GAMMA_UP * ratio
        r["gamma_down"] = REF_GAMMA_DOWN * ratio
        print(f"  gamma_ratio={ratio}, gamma_up={REF_GAMMA_UP * ratio:.4f}, gamma_down={REF_GAMMA_DOWN * ratio:.4f}, F1={r['f1']}")
    return raw


def main():
    parser = argparse.ArgumentParser(description="Noise ratio & Gamma ratio sweep (T249 base)")
    parser.add_argument("--noise-only", action="store_true", help="Run noise sweep only")
    parser.add_argument("--gamma-only", action="store_true", help="Run gamma sweep only")
    parser.add_argument("--parallel", action="store_true", help="Run trials in parallel across GPUs")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs for parallel mode (default: 4)")
    args = parser.parse_args()

    do_noise = not args.gamma_only
    do_gamma = not args.noise_only

    if args.parallel:
        # Parallel mode: collect all trials and run them together
        # Gamma first, then noise
        all_trials = []
        if do_gamma:
            all_trials.extend(_build_gamma_trials())
        if do_noise:
            all_trials.extend(_build_noise_trials())

        print(f"\nParallel mode: {len(all_trials)} trials on {args.num_gpus} GPUs")
        print(f"Trials: {[t[0] for t in all_trials]}")
        raw = run_parallel(all_trials, num_gpus=args.num_gpus)

        # Split results back into noise/gamma
        noise_results = []
        gamma_results = []
        for r in raw:
            if r["tag"].startswith("noise_"):
                ratio = float(r["tag"].replace("noise_r", "").replace("p", "."))
                r["noise_ratio"] = ratio
                noise_results.append(r)
            elif r["tag"].startswith("gamma_"):
                ratio = float(r["tag"].replace("gamma_r", "").replace("p", "."))
                r["gamma_ratio"] = ratio
                r["gamma_up"] = REF_GAMMA_UP * ratio
                r["gamma_down"] = REF_GAMMA_DOWN * ratio
                gamma_results.append(r)
    else:
        # Sequential mode (original behavior) — gamma first
        noise_results = []
        gamma_results = []

        if do_gamma:
            print("\n" + "=" * 70)
            print("GAMMA (ASYMMETRY) RATIO SWEEP")
            print(f"Ratios: {GAMMA_RATIOS}")
            print("=" * 70)
            gamma_results = gamma_sweep()

        if do_noise:
            print("\n" + "=" * 70)
            print("NOISE RATIO SWEEP")
            print(f"Ratios: {NOISE_RATIOS}")
            print("=" * 70)
            noise_results = noise_sweep()

    all_results = {}
    if noise_results:
        # Inject T249 baseline at noise_ratio=0.0 (= constantstep6t1cgamma equivalent)
        data = [{"noise_ratio": 0.0, "f1": T249_BASELINE_F1, "source": T249_BASELINE_TRIAL}]
        data += [{"noise_ratio": r["noise_ratio"], "f1": r["f1"]} for r in noise_results]
        data.sort(key=lambda x: x["noise_ratio"])
        all_results["noise_ratio_sweep"] = {
            "title": "Noise Ratio Sweep: F1 vs Noise Ratio",
            "xlabel": "Noise Ratio",
            "ylabel": "F1",
            "data": data,
            "source": {"base": "T249 (rank=32, F1=84.06)", "timestamp": RUN_STAMP}
        }
    if gamma_results:
        # Inject T249 baseline at gamma_ratio=1.0 (= constantstep6t1cgamma exact)
        data = [{"gamma_ratio": 1.0, "f1": T249_BASELINE_F1, "source": T249_BASELINE_TRIAL}]
        data += [{"gamma_ratio": r["gamma_ratio"], "f1": r["f1"]} for r in gamma_results]
        data.sort(key=lambda x: x["gamma_ratio"])
        all_results["asymmetry_ratio_sweep"] = {
            "title": "Asymmetry Ratio Sweep: F1 vs Gamma Ratio",
            "xlabel": "Gamma Ratio",
            "ylabel": "F1",
            "data": data,
            "source": {"base": "T249 (rank=32, F1=84.06)", "timestamp": RUN_STAMP}
        }

    # Save results
    out_path = RESULTS_DIR / f"sweep_noise_gamma_t249_{RUN_STAMP}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if noise_results:
        print("\nNoise Ratio Sweep (all 6T1C params scaled):")
        for r in sorted(noise_results, key=lambda x: x["noise_ratio"]):
            print(f"  ratio={r['noise_ratio']:.1f}  F1={r['f1']}")
    if gamma_results:
        print("\nGamma Ratio Sweep (gamma only, no noise):")
        for r in sorted(gamma_results, key=lambda x: x["gamma_ratio"]):
            print(f"  ratio={r['gamma_ratio']:.1f}  gamma_up={r['gamma_up']:.4f}  gamma_down={r['gamma_down']:.4f}  F1={r['f1']}")


if __name__ == "__main__":
    main()

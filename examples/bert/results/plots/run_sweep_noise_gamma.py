#!/usr/bin/env python3
"""Noise ratio & Gamma ratio sweep based on T102 (rank=32, F1=83.66).

Generates patched copies of fine_bert_squad_lrtt.py and runs them sequentially.
Each run produces F1; results are saved to a JSON file for plotting.

6T1C reference values (ratio=1.0):
  Device noise:
    dw_min_dtod=0.1, dw_min_std=0.3
    up_down_dtod=0.01
    w_max_dtod=0.05, w_min_dtod=0.05
    gamma_up_dtod=0.05, gamma_down_dtod=0.05
    write_noise_std=0.0
  Asymmetry:
    gamma_up=-0.1678, gamma_down=0.1410

Noise ratio sweep:
  All 6T1C non-ideality params (dtod, std, gamma) scaled by ratio.
  Uses linearstep device (full 6T1C model).
  ratio=0.0 → ideal (no noise, no gamma)
  ratio=1.0 → full 6T1C

Gamma ratio sweep:
  Only gamma_up/gamma_down scaled by ratio. All noise params = 0.
  Uses linearstepideal device.
  ratio=0.0 → symmetric (no asymmetry)
  ratio=1.0 → 6T1C asymmetry only

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_sweep_noise_gamma.py [--noise-only] [--gamma-only]
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
SRC = REPO / "examples/bert/fine_bert_squad_lrtt.py"
RESULTS_DIR = REPO / "examples/bert/results/BERT_SQUAD_LRTT_FINE"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── T102 base config (rank=32, abml=11, F1=83.66) ──
T102 = dict(
    lr=0.003277, tlr=0.2499, te=2, rank=32,
    fast_lr=0.1164, ab_dw_min=6.097e-05, abml=11,
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
NOISE_RATIOS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
GAMMA_RATIOS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def patch(content: str, key: str, new_value: str) -> str:
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    out, n = pattern.subn(f"{key} = {new_value}", content, count=1)
    if n == 0:
        raise RuntimeError(f"Could not patch {key}")
    return out


def apply_t102_base(src: str) -> str:
    """Apply T102 base params to script source."""
    cfg = T102
    src = patch(src, "N_EPOCHS", "5")
    src = patch(src, "SCHEDULE_EPOCHS", "5")
    src = patch(src, "SEED", str(cfg["seed"]))
    src = patch(src, "LEARNING_RATE", repr(cfg["lr"]))
    src = patch(src, "TRANSFER_LR", repr(cfg["tlr"]))
    src = patch(src, "TRANSFER_EVERY", str(cfg["te"]))
    src = patch(src, "LRTT_RANK", str(cfg["rank"]))
    src = patch(src, "FAST_LR", repr(cfg["fast_lr"]))
    src = patch(src, "AB_DW_MIN", repr(cfg["ab_dw_min"]))
    src = patch(src, "AB_MULTILEVEL", str(cfg["abml"]))
    src = patch(src, "REINIT_MODE", '"decay"')
    src = patch(src, "TRANSFER_METHOD", '"onehot"')
    src = patch(src, "C_DEVICE", '"constantstepideal"')
    src = patch(src, "FORWARD_INJECT", "False")
    src = patch(src, "FI_CONTINUOUS_ALPHA", "False")
    src = patch(src, "LEARN_OUT_SCALING", "False")
    src = patch(src, "IS_PERFECT", "True")
    src = patch(src, "OUT_NOISE", "0.0")
    src = patch(src, "ENABLE_DIAGNOSTIC", "False")
    src = patch(src, "TRAIN_SUBSET_SIZE", "0")
    src = patch(src, "EVAL_SUBSET_SIZE", "0")
    return src


def _build_linearstep_block(ratio, gamma_ratio=None):
    """Build a LinearStepDevice constructor string with scaled 6T1C params.

    If gamma_ratio is provided, use it for gamma scaling (independent of ratio).
    Otherwise gamma scales with ratio like everything else.
    """
    gr = gamma_ratio if gamma_ratio is not None else ratio
    return f"""    if AB_DEVICE == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            dw_min_dtod={REF_DW_MIN_DTOD * ratio},
            dw_min_std={REF_DW_MIN_STD * ratio},
            up_down_dtod={REF_UP_DOWN_DTOD * ratio},
            w_max_dtod={REF_W_MAX_DTOD * ratio},
            w_min_dtod={REF_W_MIN_DTOD * ratio},
            gamma_up={REF_GAMMA_UP * gr},
            gamma_down={REF_GAMMA_DOWN * gr},
            gamma_up_dtod={REF_GAMMA_UP_DTOD * ratio},
            gamma_down_dtod={REF_GAMMA_DOWN_DTOD * ratio},
            write_noise_std={REF_WRITE_NOISE_STD * ratio},
            reset_std=0.0,
            up_down=0.0,
            mult_noise={'True' if ratio > 0 else 'False'},
            mean_bound_reference={'True' if ratio > 0 else 'False'},
            lifetime=lifetime,
        )"""


# Original linearstepideal block in fine_bert_squad_lrtt.py (for string replacement)
ORIG_LINEARSTEPIDEAL = """    if AB_DEVICE == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=0.0,
            up_down=0.0,
            mult_noise=False,
            lifetime=lifetime,
        )"""


def run_one(tag: str, src_patched: str) -> dict:
    """Write patched script, run it, parse F1 from output."""
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

    print(f"\n{'='*60}")
    print(f"Running: {tag}  (log -> {log_path})")
    print(f"{'='*60}")

    with open(log_path, "wb") as logf:
        ret = subprocess.run(
            [sys.executable, str(tmp)], cwd=SRC.parent,
            stdout=logf, stderr=subprocess.STDOUT
        )

    # Parse best F1 from log
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

    tmp.unlink(missing_ok=True)
    print(f"  exit={ret.returncode}, F1={f1}")
    return {"tag": tag, "f1": f1, "exit_code": ret.returncode}


def noise_sweep():
    """Sweep all 6T1C non-ideality params (dtod, std, gamma) by ratio.

    Uses linearstepideal device with scaled 6T1C parameters.
    ratio=0.0 -> ideal device (all noise/gamma = 0)
    ratio=1.0 -> full 6T1C non-ideality
    """
    results = []
    base_src = apply_t102_base(SRC.read_text())
    base_src = patch(base_src, "AB_DEVICE", '"linearstepideal"')

    for ratio in NOISE_RATIOS:
        tag = f"noise_r{ratio:.1f}".replace(".", "p")
        src = base_src.replace(ORIG_LINEARSTEPIDEAL, _build_linearstep_block(ratio))
        if f"dw_min_dtod={REF_DW_MIN_DTOD * ratio}" not in src:
            raise RuntimeError(f"Failed to patch linearstepideal for noise ratio={ratio}")

        result = run_one(tag, src)
        result["noise_ratio"] = ratio
        results.append(result)
        print(f"  noise_ratio={ratio}, F1={result['f1']}")

    return results


def gamma_sweep():
    """Sweep gamma_up/gamma_down only. All noise params = 0.

    Uses linearstepideal device with only gamma scaled.
    ratio=0.0 -> symmetric (no asymmetry, equivalent to constantstepideal)
    ratio=1.0 -> 6T1C asymmetry (gamma_up=-0.1678, gamma_down=0.1410)
    """
    results = []
    base_src = apply_t102_base(SRC.read_text())
    base_src = patch(base_src, "AB_DEVICE", '"linearstepideal"')

    for ratio in GAMMA_RATIOS:
        tag = f"gamma_r{ratio:.1f}".replace(".", "p")
        # noise ratio=0 (no dtod/std), but gamma scales by ratio
        new_block = _build_linearstep_block(ratio=0.0, gamma_ratio=ratio)
        src = base_src.replace(ORIG_LINEARSTEPIDEAL, new_block)
        if f"gamma_up={REF_GAMMA_UP * ratio}" not in src:
            raise RuntimeError(f"Failed to patch gamma for ratio={ratio}")

        result = run_one(tag, src)
        result["gamma_ratio"] = ratio
        result["gamma_up"] = REF_GAMMA_UP * ratio
        result["gamma_down"] = REF_GAMMA_DOWN * ratio
        results.append(result)
        print(f"  gamma_ratio={ratio}, gamma_up={REF_GAMMA_UP * ratio:.4f}, gamma_down={REF_GAMMA_DOWN * ratio:.4f}, F1={result['f1']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Noise ratio & Gamma ratio sweep (T102 base)")
    parser.add_argument("--noise-only", action="store_true", help="Run noise sweep only")
    parser.add_argument("--gamma-only", action="store_true", help="Run gamma sweep only")
    args = parser.parse_args()

    do_noise = not args.gamma_only
    do_gamma = not args.noise_only

    all_results = {}

    if do_noise:
        print("\n" + "=" * 70)
        print("NOISE RATIO SWEEP")
        print("All 6T1C params (dtod, std, gamma) scaled by ratio")
        print(f"Ratios: {NOISE_RATIOS}")
        print("=" * 70)
        noise_results = noise_sweep()
        all_results["noise_ratio_sweep"] = {
            "title": "Noise Ratio Sweep: F1 vs Noise Ratio",
            "xlabel": "Noise Ratio",
            "ylabel": "F1",
            "data": [{"noise_ratio": r["noise_ratio"], "f1": r["f1"]} for r in noise_results],
            "source": {"base": "T102 (rank=32, abml=11)", "timestamp": RUN_STAMP}
        }

    if do_gamma:
        print("\n" + "=" * 70)
        print("GAMMA (ASYMMETRY) RATIO SWEEP")
        print(f"Only gamma scaled. Reference: gamma_up={REF_GAMMA_UP}, gamma_down={REF_GAMMA_DOWN}")
        print(f"Ratios: {GAMMA_RATIOS}")
        print("=" * 70)
        gamma_results = gamma_sweep()
        all_results["asymmetry_ratio_sweep"] = {
            "title": "Asymmetry Ratio Sweep: F1 vs Gamma Ratio",
            "xlabel": "Gamma Ratio",
            "ylabel": "F1",
            "data": [{"gamma_ratio": r["gamma_ratio"], "f1": r["f1"]} for r in gamma_results],
            "source": {"base": "T102 (rank=32, abml=11)", "timestamp": RUN_STAMP}
        }

    # Save results
    out_path = RESULTS_DIR / f"sweep_noise_gamma_{RUN_STAMP}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if do_noise:
        print("\nNoise Ratio Sweep (all 6T1C params scaled):")
        for r in noise_results:
            print(f"  ratio={r['noise_ratio']:.1f}  F1={r['f1']}")
    if do_gamma:
        print("\nGamma Ratio Sweep (gamma only, no noise):")
        for r in gamma_results:
            print(f"  ratio={r['gamma_ratio']:.1f}  gamma_up={r['gamma_up']:.4f}  gamma_down={r['gamma_down']:.4f}  F1={r['f1']}")


if __name__ == "__main__":
    main()

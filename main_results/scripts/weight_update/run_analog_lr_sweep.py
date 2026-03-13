"""Sweep: analog LR only — digital LR fixed at per-bit best from TPE sweep.

Each trial runs full SQuAD 2-epoch training with ConstantStepDevice.
Optuna TPE explores lr_analog in log-uniform space while lr_digital is fixed.
Objective: maximize F1 score.

Fixed digital LR (from previous TPE sweep best):
  8-bit:  lr_d = 0.9394
  10-bit: lr_d = 0.7620
  12-bit: lr_d = 0.8637
  14-bit: lr_d = 0.6330

Analog LR range (log-uniform): [0.001, 10.0]

GPU assignment: 1 bit per GPU, 10 TPE trials each.

Usage:
  python run_analog_lr_sweep.py [--n-trials 10] [--n-gpus 4] [--epochs 2]
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import optuna
from optuna.samplers import TPESampler

# ── Config ──────────────────────────────────────────────────────────
BITS_LIST = [8, 10, 12, 14]

GPU_BITS = {
    0: [8],
    1: [10],
    2: [12],
    3: [14],
}

# Fixed digital LR per bit (best from previous TPE sweep)
FIXED_LR_DIGITAL = {
    8:  0.9394,
    10: 0.7620,
    12: 0.8637,
    14: 0.6330,
}

# Analog LR search range (log-uniform) — wider than before
LR_A_MIN, LR_A_MAX = 0.001, 10.0

SQUAD_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "analysis", "optuna_bert_squad_tiki.py")


# ── Run single SQuAD trial ─────────────────────────────────────────

def run_squad_trial(bits, dw_min, lr_analog, lr_digital, epochs, gpu_id,
                    trial_idx, n_trials, train_subset=0):
    """Run SQuAD training and return (f1, result_dict)."""
    study_name = (f"alr_b{bits}_t{trial_idx:02d}_"
                  f"lrA{lr_analog:.2e}_lrD{lr_digital:.2e}").replace("+", "")

    cmd = [
        sys.executable, SQUAD_SCRIPT,
        "--n-trials", "1",
        "--target-analog",
        "--device-type", "constant_step",
        "--dw-min", f"{dw_min:.10f}",
        "--lr", f"{lr_analog:.10f}",
        "--classifier-lr", f"{lr_digital:.10f}",
        "--lora-target", "qkv",
        "--forward-perfect",
        "--backward-perfect",
        "--nontarget-digital",
        "--epochs", str(epochs),
        "--study-name", study_name,
    ]
    if train_subset > 0:
        cmd += ["--train-subset", str(train_subset)]

    prefix = f"[GPU{gpu_id}][{trial_idx+1}/{n_trials}]"
    print(f"\n{prefix} SQuAD {bits}-bit | "
          f"lr_a={lr_analog:.6e} lr_d={lr_digital:.6e} (fixed) "
          f"dw_min={dw_min:.4e}", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(SQUAD_SCRIPT) or ".", env=env)
    elapsed = time.time() - t0

    # Extract F1 from stdout
    f1 = None
    em = None
    for line in result.stdout.split("\n"):
        m_f1 = re.search(r'[Ff]1[=:\s]+([0-9.]+)', line)
        if m_f1:
            try:
                f1 = float(m_f1.group(1))
            except ValueError:
                pass
        m_em = re.search(r'[Ee]xact[_ ]?[Mm]atch[=:\s]+([0-9.]+)', line)
        if m_em:
            try:
                em = float(m_em.group(1))
            except ValueError:
                pass

    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"  {prefix} {status} ({elapsed/60:.1f}min) F1={f1} EM={em}", flush=True)

    if result.returncode != 0:
        err_lines = result.stderr.strip().split("\n")
        for line in err_lines[-10:]:
            print(f"  {prefix} ERR: {line}", flush=True)

    return {
        "bits": bits,
        "dw_min": dw_min,
        "lr_analog": lr_analog,
        "lr_digital": lr_digital,
        "study_name": study_name,
        "status": status,
        "elapsed_min": round(elapsed / 60, 1),
        "f1": f1,
        "em": em,
    }


# ── Optuna TPE sweep per bit ──────────────────────────────────────

def _run_optuna_for_bit(gpu_id, bits, n_trials, epochs, train_subset):
    """Run Optuna TPE study for one bit level on one GPU — analog LR only."""
    dw_min = 2.0 / (2 ** bits)
    lr_digital = FIXED_LR_DIGITAL[bits]
    results = []

    sampler = TPESampler(seed=42 + bits)
    study = optuna.create_study(
        study_name=f"alr_sweep_b{bits}",
        sampler=sampler,
        direction="maximize",  # maximize F1
    )

    def objective(trial):
        lr_analog = trial.suggest_float("lr_analog", LR_A_MIN, LR_A_MAX, log=True)

        row = run_squad_trial(
            bits=bits, dw_min=dw_min,
            lr_analog=lr_analog, lr_digital=lr_digital,
            epochs=epochs, gpu_id=gpu_id,
            trial_idx=trial.number, n_trials=n_trials,
            train_subset=train_subset,
        )
        results.append(row)

        if row["f1"] is None:
            return 0.0  # failed trial
        return row["f1"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    # Print study summary
    print(f"\n{'='*60}")
    print(f"  [GPU{gpu_id}] {bits}-bit analog LR sweep done ({n_trials} trials):")
    print(f"    Fixed lr_d = {lr_digital:.4f}")
    print(f"    Best F1 = {study.best_value:.4f}")
    print(f"    Best lr_a = {study.best_params['lr_analog']:.6e}")
    print(f"{'='*60}", flush=True)

    return results


def run_sweep(args):
    """TPE sweep: each GPU runs one bit level's Optuna study (analog LR only)."""
    n_gpus = args.n_gpus
    n_trials = args.n_trials
    epochs = args.epochs

    total = len(BITS_LIST) * n_trials
    print(f"{'='*60}")
    print(f"=== Analog LR Sweep: SQuAD {epochs}ep (digital LR fixed) ===")
    print(f"  {len(BITS_LIST)} bits × {n_trials} trials = {total} runs")
    print(f"  IO: perfect, Device: ConstantStep, Layers: QKVO only")
    print(f"  LR analog:  [{LR_A_MIN}, {LR_A_MAX}] (log-uniform, swept)")
    print(f"  LR digital: FIXED per bit (from previous TPE best)")
    print(f"  Objective: maximize F1")
    print(f"GPU assignment:")
    for gpu_id, bits_list in GPU_BITS.items():
        bits = bits_list[0]
        dw = 2.0 / (2 ** bits)
        lr_d = FIXED_LR_DIGITAL[bits]
        print(f"  GPU {gpu_id}: {bits}-bit (dw_min={dw:.4e}, lr_d={lr_d:.4f}, {n_trials} trials)")
    print(f"{'='*60}", flush=True)

    all_results = []
    t_total = time.time()
    train_subset = getattr(args, 'train_subset', 0)

    if n_gpus <= 1:
        for gpu_id, bits_list in GPU_BITS.items():
            for bits in bits_list:
                res = _run_optuna_for_bit(gpu_id, bits, n_trials, epochs, train_subset)
                all_results.extend(res)
    else:
        with ProcessPoolExecutor(max_workers=n_gpus) as executor:
            futures = {}
            for gpu_id, bits_list in GPU_BITS.items():
                for bits in bits_list:
                    fut = executor.submit(
                        _run_optuna_for_bit, gpu_id, bits, n_trials,
                        epochs, train_subset
                    )
                    futures[fut] = (gpu_id, bits)

            for future in as_completed(futures):
                gpu_id, bits = futures[future]
                try:
                    gpu_results = future.result()
                    all_results.extend(gpu_results)
                    print(f"  GPU {gpu_id} ({bits}-bit) finished: {len(gpu_results)} runs",
                          flush=True)
                except Exception as e:
                    print(f"  ERROR on GPU {gpu_id} ({bits}-bit): {e}", flush=True)

    all_results.sort(key=lambda r: (r["bits"], -(r["f1"] or 0)))

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(all_results)}/{total} runs in {elapsed_total/60:.1f} min")

    # Save summary CSV
    csv_path = os.path.join(args.out_dir, "single", "analog_lr_sweep_summary.csv")
    if all_results:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        keys = all_results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Summary saved: {csv_path}")

    # Print best per bit
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (sorted by F1 within each bit):")
    print(f"{'='*60}")
    for bits in BITS_LIST:
        bit_res = [r for r in all_results if r["bits"] == bits]
        lr_d = FIXED_LR_DIGITAL[bits]
        print(f"\n  {bits}-bit (dw_min={2.0/(2**bits):.4e}, lr_d={lr_d:.4f} fixed):")
        for r in bit_res:
            f1_str = f"{r['f1']:.2f}" if r['f1'] is not None else "N/A"
            em_str = f"{r['em']:.2f}" if r['em'] is not None else "N/A"
            print(f"    F1={f1_str:>6} EM={em_str:>6} | "
                  f"lr_a={r['lr_analog']:.4e} | "
                  f"{r['elapsed_min']:.0f}min | {r['status']}")

    return csv_path, all_results


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analog LR Sweep: SQuAD training (digital LR fixed at best)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Optuna TPE trials per bit level")
    parser.add_argument("--epochs", type=int, default=2,
                        help="SQuAD training epochs per trial")
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="Number of GPUs for parallel execution")
    parser.add_argument("--out-dir", type=str,
                        default="./main_results/weight_update/squad")
    parser.add_argument("--train-subset", type=int, default=0,
                        help="Limit SQuAD training data (0=full, for testing)")
    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()

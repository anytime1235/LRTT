"""validate_csv.py — Validate paper_figures v3 CSV outputs.

Usage:
  python validate_csv.py --out-dir /data/results/tikitakav1 --run-tag v3
"""

import argparse
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Validate paper_figures CSV outputs")
parser.add_argument("--out-dir", type=str, default="/data/results/tikitakav1")
parser.add_argument("--run-tag", type=str, default="v3")
args = parser.parse_args()

OUT_DIR = args.out_dir
RUN_TAG = args.run_tag

# ---------------------------------------------------------------------------
# Common column sets
# ---------------------------------------------------------------------------
COMMON_COLS = [
    "figure_id", "run_tag", "variant", "dac_bits", "adc_bits",
    "inp_bound", "inp_res", "res_ratio", "step_size", "nm_thres",
    "layer_idx", "sublayer",
]
METRIC_COLS = [
    "EZR", "QZR_all", "QZR_nonzero", "ODR", "cosine_sim",
    "l2_retention", "rel_l2_error", "clip_rate_scaled",
    "ratio_q50", "ratio_q90", "ratio_q99",
    "absmax_q50", "absmax_q90", "absmax_q99", "absmax_q999",
]
STEP_EXTRA = ["step_idx", "n_vec"]
CDF_COLS = ["layer_idx", "sublayer", "layer_name", "ratio", "cdf"]

SUMMARY_COLS = COMMON_COLS + METRIC_COLS
STEPS_COLS = COMMON_COLS + STEP_EXTRA + METRIC_COLS

CRITICAL_SUMMARY = ["QZR_nonzero", "cosine_sim", "l2_retention", "EZR", "ODR"]
CRITICAL_STEPS = ["QZR_nonzero"]
CRITICAL_CDF = ["ratio", "cdf"]

N_STEP = 200  # full run default


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------
def validate_one(filepath, required_cols, min_rows, critical_cols):
    """Validate a single CSV file.

    Returns (passed: bool, messages: list[str]).
    """
    msgs = []
    basename = os.path.basename(filepath)

    # 1. Existence
    if not os.path.exists(filepath):
        return False, [f"File not found: {filepath}"]

    # 2. Read
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return False, [f"Cannot read {basename}: {e}"]

    passed = True

    # 3. Columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        msgs.append(f"Missing columns: {missing_cols}")
        passed = False

    # 4. Row count
    if len(df) < min_rows:
        msgs.append(f"Row count {len(df)} < min {min_rows}")
        passed = False

    # 5. Critical NaN check
    for col in critical_cols:
        if col in df.columns:
            n_nan = int(df[col].isna().sum())
            if n_nan > 0:
                msgs.append(f"NaN in critical column '{col}': {n_nan} rows")
                passed = False

    # 6. Range sanity
    if "QZR_nonzero" in df.columns:
        qmin, qmax = df["QZR_nonzero"].min(), df["QZR_nonzero"].max()
        if qmin < -0.01 or qmax > 1.01:
            msgs.append(f"QZR_nonzero range [{qmin:.4f}, {qmax:.4f}] outside [0,1]")
            passed = False
    if "cosine_sim" in df.columns:
        cmin = df["cosine_sim"].min()
        if cmin < -1.01:
            msgs.append(f"cosine_sim min={cmin:.4f} < -1")
            passed = False

    if passed:
        msgs.append(f"{len(df)} rows, {len(df.columns)} cols")

    return passed, msgs


def print_report(results):
    """Print validation report. Returns exit code (0 = all pass)."""
    n_pass = sum(1 for ok, _ in results.values() if ok)
    n_total = len(results)

    print("=" * 60)
    print("CSV Validation Report")
    print("=" * 60)

    for name, (ok, msgs) in results.items():
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"  {tag} {name}")
        for m in msgs:
            print(f"         {m}")

    print("-" * 60)
    print(f"  {n_pass}/{n_total} PASSED")
    print("=" * 60)

    return 0 if n_pass == n_total else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    csv_specs = {
        "A_summary": (
            os.path.join(OUT_DIR, "metrics_paper_A_rootcause_summary.csv"),
            SUMMARY_COLS, 72, CRITICAL_SUMMARY,
        ),
        "A_steps": (
            os.path.join(OUT_DIR, "metrics_paper_A_rootcause_steps.csv"),
            STEPS_COLS, int(72 * N_STEP * 0.8), CRITICAL_STEPS,
        ),
        "A_cdf": (
            os.path.join(OUT_DIR, "metrics_paper_A_rootcause_cdf.csv"),
            CDF_COLS, 100, CRITICAL_CDF,
        ),
        "B_summary": (
            os.path.join(OUT_DIR, "metrics_paper_B_bitsweep_summary.csv"),
            SUMMARY_COLS, 432, CRITICAL_SUMMARY,
        ),
        "B_steps": (
            os.path.join(OUT_DIR, "metrics_paper_B_bitsweep_steps.csv"),
            STEPS_COLS, int(50400 * 0.8), CRITICAL_STEPS,
        ),
        "C_summary": (
            os.path.join(OUT_DIR, "metrics_paper_C_solutions_summary.csv"),
            SUMMARY_COLS, 288, CRITICAL_SUMMARY,
        ),
        "C_steps": (
            os.path.join(OUT_DIR, "metrics_paper_C_solutions_steps.csv"),
            STEPS_COLS, int(57600 * 0.8), CRITICAL_STEPS,
        ),
    }

    results = {}
    for name, (filepath, req_cols, min_rows, crit_cols) in csv_specs.items():
        ok, msgs = validate_one(filepath, req_cols, min_rows, crit_cols)
        results[name] = (ok, msgs)

    exit_code = print_report(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

# Reset-mode ablation for the AF heatmap

This experiment isolates **per-transfer reset** as a candidate root cause for
the AF (γ-saturation) insensitivity observed in `LRTT-v2` on the
`fig_heatmaps_only.png` figure. We add two new method variants whose only
change vs. their baseline is to enable a hard reset of the fast-tile state at
every transfer event, and re-run the *same* per-cell Optuna TPE search used
for the existing baselines.

If the AF curve flattens for the reset variants at their per-cell best HP,
reset is sufficient. If it flattens partially (but not all the way to v2),
reset is dominant but other v1↔v2 differences (single-tile updates, small
per-step gradient magnitude into B, additive vs. multiplicative transfer)
also contribute.

## Variants

| Method | Baseline | Diff |
|---|---|---|
| `tikitaka_reset`  | TikiTaka v1 (`tikitaka_v1`) | `ChoppedTransferCompound(chopper-off) → TransferCompound` + `with_reset_prob=1.0` (see note below) |
| `lrtt_v1_reset`   | LRTT-v1 (`lrtt_v1`)         | `PythonLRTTDevice.reinit_mode: "decay" → "standard"` |
| `lrtt_v2_noreset` | LRTT-v2 (`lrtt_v2`)         | `PythonLRTTDevice.selector_reset_b_on_advance: True → False` |

All other tile / training / device settings are identical to the heatmap
baseline (`experiments/heatmap_hyperparameters.json`).

**Note on the TikiTaka compound class.** The MNIST baseline `tikitaka_v1`
uses `ChoppedTransferCompound` with `in_chop_prob=out_chop_prob=0` and
`no_buffer=True`, which is **bit-identical at runtime** to plain
`TransferCompound`: with chop_prob=0 the in/out chopper arrays stay all-`false`,
the `(in_chop != out_chop) ? -val : val` branch always returns `val`, and the
chop_prob>0 branch (the only place `rw_rng_.sampleUniform()` is called) is
skipped entirely, so RNG state is not perturbed. The reset variant
`tikitaka_reset` uses plain `TransferCompound` instead because the chopped
device's C++ `checkSupported()` (`rpu_chopped_transfer_device.cpp:82-87`)
hard-rejects any `with_reset_prob > 0` regardless of whether the chopper is
actually enabled at runtime. The base `TransferRPUDevice` C++ supports
`with_reset_prob > 0` natively (`rpu_transfer_device.cpp:92, 198`, requires
`transfer_columns=true` which we already set). So the swap toggles only the
reset axis between baseline and ablation.

## Search setup (matches the running TPE-30 sweep)

- 5 × 5 = **25 cells per method** (AF ∈ {0, 1, 2, 5, 10}, UNR ∈ {0, 1, 3, 5, 10})
- **30 Optuna TPE trials per cell** (TPESampler, `seed=42`, `n_startup_trials=5`)
- Search space (log-uniform, mirrors the corresponding baseline):
  - `tikitaka_reset` : transfer_lr ∈ [0.1, 10], fast_lr ∈ [0.03, 3], classifier_lr ∈ [0.1, 10]
  - `lrtt_v1_reset`  : lr ∈ [0.01, 1], tlr ∈ [1e-4, 1e-2], clr ∈ [0.03, 3]
  - `lrtt_v2_noreset`: lr ∈ [0.1, 10],  tlr ∈ [0.01, 1],  clr ∈ [0.1, 10]
- Warm-start: the baseline best HP is enqueued via `study.enqueue_trial`, so the
  first real trial evaluates it. (Baselines have the warm-start as a *completed*
  trial since their fixed-HP acc is already known; the three ablations don't
  yet, so trial 1 of each cell = the warm-start eval.)
- Total: **3 methods × 25 cells × 30 trials = 2,250 trials**, ~45–60 hours on
  one A100 (clean GPU). Roughly 2× if another sweep is sharing the GPU.

## How to run (primary entry point — TPE)

```bash
PYTHONPATH=/root/LRTT/src \
  LD_LIBRARY_PATH=/root/.venv310/lib/python3.10/site-packages/aihwkit.libs:/root/.venv310/lib/python3.10/site-packages/torch/lib \
  /root/.venv310/bin/python -u /root/LRTT/experiments/per_cell_tpe_30.py \
    --methods tikitaka_reset lrtt_v1_reset lrtt_v2_noreset \
    > /root/LRTT/logs/per_cell_tpe_30_reset.log 2>&1 &
```

Or run all six methods in one call (baselines + ablations — note that the
baselines may already be in progress in another process; the per-cell JSONs
are resumable):

```bash
... per_cell_tpe_30.py --methods tikitaka_v1 lrtt_v1 lrtt_v2 tikitaka_reset lrtt_v1_reset lrtt_v2_noreset ...
```

Output layout:

```
results/per_cell_tpe_30/
  tikitaka_reset/af{a:g}_unr{u:g}.json    # 25 files, each with 30-trial log + best HP
  lrtt_v1_reset/af{a:g}_unr{u:g}.json
  lrtt_v2_noreset/af{a:g}_unr{u:g}.json
```

The script is **resumable**: each cell JSON saves after every trial, so
killing/restarting picks up where it left off.

## Quick alternative — fixed-HP single-trial sweep (≈ 1.5 h)

For a fast sanity check before committing 30+ hours, the same two variants
can be swept at the baseline best HP only (matches the `sweep_5x5_fixed_hp`
heatmap setup, 1 trial per cell):

```bash
PYTHONPATH=/root/LRTT/src \
  LD_LIBRARY_PATH=/root/.venv310/lib/python3.10/site-packages/aihwkit.libs:/root/.venv310/lib/python3.10/site-packages/torch/lib \
  /root/.venv310/bin/python /root/LRTT/experiments/sweep_5x5_reset_modes.py
```

Output: `results/sweep_5x5_reset_modes/{tikitaka_reset,lrtt_v1_reset}.json`.

## Expected interpretation

| Observation | Conclusion |
|---|---|
| `lrtt_v1_reset` AF=10/UNR=0 best acc rises to ≈ v2's 96.7 % | reset alone explains AF insensitivity |
| `lrtt_v1_reset` AF=10/UNR=0 best acc rises ≈ halfway (≈ 94 %) but plateaus | reset is dominant; selector-only-B updates and additive blockwise transfer add the rest |
| `lrtt_v1_reset` AF=10/UNR=0 best acc remains ≈ 92 % | reset is *not* the root cause; v2's mechanism is mainly the selector / single-tile update path |
| `tikitaka_reset` better than `tikitaka_v1` at high AF | confirms reset helps in the column-based architecture too |
| `tikitaka_reset` AF=0 acc drops well below 97 % | the chosen HP search box doesn't cover the post-reset optimum; widen the box and retry |
| `lrtt_v2_noreset` AF=0 acc drops noticeably below v2's 96.87 % but AF=10 stays high | confirms reset gives a clean-condition accuracy boost but is **not** the source of v2's AF insensitivity (selector + single-tile update path is) |
| `lrtt_v2_noreset` AF=0 collapses (< 50 %) | the coordinate-mixing from skipped reset is structurally incompatible with the selector path |

## Comparison with existing baseline data (TPE-30)

```python
import json, glob, os
ROOT = '/root/LRTT/results/per_cell_tpe_30'

def best_at(method, af, unr):
    p = f'{ROOT}/{method}/af{af:g}_unr{unr:g}.json'
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get('best_acc')

methods = ['tikitaka_v1', 'lrtt_v1', 'lrtt_v2',
           'tikitaka_reset', 'lrtt_v1_reset']
for m in methods:
    a0  = best_at(m, 0.0, 0.0)
    a10 = best_at(m, 10.0, 0.0)
    if a0 is None or a10 is None:
        print(f'{m:18s}  (incomplete)')
        continue
    print(f'{m:18s}  AF=0→{a0:.2f}  AF=10→{a10:.2f}  Δ={a10-a0:+.2f}')
```

Running interpretation: a `lrtt_v1_reset` Δ near `lrtt_v2`'s (≈ −0.2 pp) is
strong evidence that reset is the dominant lever; a Δ closer to `lrtt_v1`'s
(≈ −4.9 pp) means other v1↔v2 differences carry most of the signal.

## Preliminary smoke-test results (1 trial @ default HP)

Run before launching the full TPE-30 sweep, on the *no-reset* v2 variant only,
to answer "does v2 still train without reset?".

| Cell | `lrtt_v2` (with reset) | `lrtt_v2_noreset` (no reset) |
|---|---|---|
| AF=0,  UNR=0 | 96.87 | **93.85** |
| AF=10, UNR=0 | 96.70 | **95.98** |

**Findings**

1. v2 *does* still train without reset. The "Required for coordinate
   consistency" comment in `lrtt_python.py:234-237` is directionally right —
   accuracy drops at clean conditions — but it's a ~3 pp loss, not a collapse.
2. **AF insensitivity is preserved without reset** (Δ(AF=0→10) is +2.13 pp at
   default HP, vs. baseline v2's −0.17 pp). This means the dominant mechanism
   behind v2's flat AF curve is the **selector + single-tile-only update**
   (A is unused, only the selected b=rank rows of d are accumulated into B,
   per-step gradient magnitude is small), *not* the per-transfer reset.
3. Reset gives a moderate clean-condition accuracy boost on top of an already
   AF-robust update path.

The TPE-30 sweep on `lrtt_v2_noreset` will confirm whether per-cell HP tuning
recovers some of the 3 pp clean-condition gap.

## Files

- `experiments/per_cell_tpe_30.py` — TPE driver. Now supports 5 methods
  (`tikitaka_v1`, `lrtt_v1`, `lrtt_v2`, `tikitaka_reset`, `lrtt_v1_reset`).
- `experiments/sweep_5x5_reset_modes.py` — defines `run_tikitaka_reset`,
  `run_lrtt_v1_reset` (imported by `per_cell_tpe_30.py`); also runnable as a
  standalone fixed-HP single-trial sweep.
- `experiments/heatmap_hyperparameters.json` — canonical HP / device-config
  catalog for the four heatmap baselines and the two reset variants.
- `experiments/RESET_ABLATION.md` — this file.

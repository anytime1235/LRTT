# Reset-mode ablation for the AF heatmap

This experiment isolates **per-transfer reset** as a candidate root cause for the
AF (γ-saturation) insensitivity observed in `LRTT-v2` on the
`fig_heatmaps_only.png` figure. We add two new method variants whose only
change vs. their baseline is to enable a hard reset of the fast-tile state at
every transfer event, and re-run the *exact same* fixed-HP / 5×5 / 1-trial
sweep used for the heatmap.

If the AF curve flattens for the reset variants, reset is sufficient. If it
flattens partially (but not all the way to v2), reset is dominant but other
v1↔v2 differences (single-tile updates, small per-step gradient magnitude into
B, additive vs. multiplicative transfer) also contribute.

## Variants

| Method | Baseline | Diff |
|---|---|---|
| `tikitaka_reset` | TikiTaka v1 (`tikitaka_v1`) | `ChoppedTransferCompound.with_reset_prob: 0.0 → 1.0` |
| `lrtt_v1_reset`  | LRTT-v1 (`lrtt_v1`)         | `PythonLRTTDevice.reinit_mode: "decay" → "standard"` |

All other tile / training / device settings are identical to the heatmap
baseline (`experiments/heatmap_hyperparameters.json`). HPs are reused from the
prior best-HP search; they are *not* re-tuned for the reset variants by design,
so a possible drop in clean-AF accuracy will tell us how brittle the original
HP was to the reset switch.

## Grid (matches heatmap)

- `AF  ∈ {0.0, 1.0, 2.0, 5.0, 10.0}` (5 values; γ_up = γ_down = AF, up_down = 0)
- `UNR ∈ {0.0, 1.0, 3.0, 5.0, 10.0}` (5 values; scales `dw_min_std` by 0.3·UNR)
- 5 × 5 = 25 cells per method, **1 trial per cell** (same as `sweep_5x5_fixed_hp.py`)
- Fixed HP per method (see `heatmap_hyperparameters.json` → `reset_ablation_methods`)

## How to run

```bash
PYTHONPATH=/root/LRTT/src \
  LD_LIBRARY_PATH=/root/.venv310/lib/python3.10/site-packages/aihwkit.libs:/root/.venv310/lib/python3.10/site-packages/torch/lib \
  /root/.venv310/bin/python /root/LRTT/experiments/sweep_5x5_reset_modes.py
```

Optional method selection (run only one):

```bash
... python /root/LRTT/experiments/sweep_5x5_reset_modes.py --methods lrtt_v1_reset
... python /root/LRTT/experiments/sweep_5x5_reset_modes.py --methods tikitaka_reset
```

The script writes a per-cell cache (`<method>_cache.json`) and a final per-method
result (`<method>.json`) plus a combined `all_methods.json`, all into:

```
results/sweep_5x5_reset_modes/
  tikitaka_reset.json
  tikitaka_reset_cache.json
  lrtt_v1_reset.json
  lrtt_v1_reset_cache.json
  all_methods.json
```

The cache is keyed by `af{AF}_unr{UNR}`, so the script is resumable: re-running
skips cells already evaluated.

## Wall-time estimate

- TikiTaka trial (epochs=30, early-stop): ≈ 130 – 170 s
- LRTT-v1 trial: ≈ 80 – 130 s (early-terminates at acc < 50 from epoch 5)
- 25 × 2 = 50 trials → **≈ 1.5 – 2 hours** on one A100 (clean GPU). About 2× if
  another sweep is sharing the GPU.

## Expected interpretation

| Observation | Conclusion |
|---|---|
| `lrtt_v1_reset` AF=10 acc rises to ≈ v2's 96 % range | reset alone explains AF insensitivity |
| `lrtt_v1_reset` AF=10 acc rises ≈ halfway (≈ 94 %) but plateaus | reset is dominant; selector-only-B updates and additive blockwise transfer add the rest |
| `lrtt_v1_reset` AF=10 acc remains ≈ 92 % (no improvement) | reset is *not* the root cause; the v2 mechanism is mainly the selector / single-tile update path |
| `tikitaka_reset` better than `tikitaka_v1` at high AF | confirms reset helps in the column-based architecture too |
| `lrtt_v1_reset` AF=0 acc drops noticeably below v1's 97 % | the chosen HP (tuned for `decay`) is not optimal for `standard` — a quick HP re-tune at AF=0 may be needed to fully isolate the reset effect |

## Comparison with existing heatmap data

After the run, the per-cell accuracies can be compared to the heatmap baseline
in `results/sweep_5x5_fixed_hp/`:

```python
# quick comparison
import json
base = {m: json.load(open(f'results/sweep_5x5_fixed_hp/{f}.json'))['grid']
        for m, f in [('tikitaka_v1','tikitaka_bestHP'),('lrtt_v1','lrtt_v1'),
                     ('lrtt_v2','lrtt_v2')]}
new  = {m: json.load(open(f'results/sweep_5x5_reset_modes/{m}.json'))['grid']
        for m in ['tikitaka_reset','lrtt_v1_reset']}
# AF=10, UNR=0 spread:
for m, g in {**base, **new}.items():
    cell = next(r for r in g if r['af_ratio']==10.0 and r['update_noise_ratio']==0.0)
    print(f'{m:20s} AF=10 UNR=0 acc = {cell["acc"]:.2f}')
```

Or extend `fig_heatmaps_only.py` to add two more panels by appending entries to
its `files` dict pointing at the new JSONs.

## Files in this experiment

- `experiments/sweep_5x5_reset_modes.py` — the sweep driver (defines the two reset variants and runs the 5×5 grid).
- `experiments/heatmap_hyperparameters.json` — the canonical HP / device-config catalog for the original heatmap baselines and the reset variants.
- `experiments/RESET_ABLATION.md` — this file.

Sources reused via Python import (no copy):

- `experiments/hp_search_tikitaka_3hp_gamma_af.py` — `_make_fast_device`,
  `_make_slow_device`, `build_loaders`, training constants for `tikitaka_reset`.
- `experiments/hp_search_v1_decay_gamma_af_2stage.py` — `_make_ab_device`,
  `_make_c_device`, `build_loaders`, training constants for `lrtt_v1_reset`.

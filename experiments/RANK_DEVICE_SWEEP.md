# Rank × C-tile sweep with 6T1C A/B fixed

This experiment sweeps **rank** and **C-tile device model** while holding A/B
tiles at the calibrated **6T1C** device (Li VLSI 2018,
`PythonLRTTPreset.sixt1c_ab` in
`aihwkit/simulator/configs/lrtt_python.py:918`). It answers: how does the
LRTT-v1 / LRTT-v2 / TikiTaka pipeline behave when the long-term store (C tile)
is replaced with a realistic resistive device, as a function of the LoRA rank?

## Fixed (A/B tiles): canonical 6T1C

| Field | Value |
|---|---|
| Device | `LinearStepDevice` from `PythonLRTTPreset.sixt1c_ab` |
| `dw_min` | 0.001981 |
| `gamma_up`, `gamma_down` | −0.1678, +0.1410 (slight asymmetry) |
| `mult_noise` | True |
| `dw_min_dtod` | 0.1 |
| `dw_min_std` (cycle-to-cycle) | 0.3 |
| `write_noise_std` | 0.0182 |
| Retention | `lifetime ≈ TAU_SEC / (1 − exp(−dt/TAU))`, `TAU_SEC=46505 s` (12.9 h capacitor τ) |

## Swept axes

- **rank** ∈ {1, 2, 4, 8, 16}  (LRTT-v1 / LRTT-v2 only)
- **C-tile** ∈ {`ideal`, `ecram`, `rram`}
  - `ideal` → `IdealizedPresetDevice` (Gokmen-Vlasov, ~10 000 states, ConstantStep)
  - `ecram` → `EcRamPresetDevice` (Tang IEDM 2018, Li-ECRAM, γ_up=0.115, γ_down=0.509)
  - `rram`  → `ReRamESPresetDevice` (Gong Nat. Commun. 2018, ExpStep)
- **method** ∈ {`lrtt_v1`, `lrtt_v2`, `tikitaka_v1`}
  - TikiTaka has no LoRA-rank dimension → one cell per C-tile (3 cells total)

Cells: 2 methods × 5 ranks × 3 C-tiles + TikiTaka 3 = **33 cells**.

## HP search

Per-cell **Optuna TPE, 30 trials**, log-uniform search box mirroring the
heatmap-baseline `per_cell_tpe_30.py`. The heatmap-baseline best HP is
enqueued as the first trial (warm-start), so trial 1 of each cell evaluates
that HP for real and trials 2–30 are TPE-suggested.

| Method | Search variables | Range (log-uniform) |
|---|---|---|
| `lrtt_v1` | `lr`, `tlr` (transfer_lr), `clr` (classifier_lr) | [0.01, 1] × [1e-4, 1e-2] × [0.03, 3] |
| `lrtt_v2` | `lr`, `tlr`, `clr` | [0.1, 10] × [0.01, 1] × [0.1, 10] |
| `tikitaka_v1` | `transfer_lr`, `fast_lr`, `classifier_lr` (analog `lr` fixed at 0.1) | [0.1, 10] × [0.03, 3] × [0.1, 10] |

Total: 33 cells × 30 trials = **990 trials**, ~30–35 hours on a clean A100.

## How to run

```bash
PYTHONPATH=/root/LRTT/src \
  LD_LIBRARY_PATH=/root/.venv310/lib/python3.10/site-packages/aihwkit.libs:/root/.venv310/lib/python3.10/site-packages/torch/lib \
  /root/.venv310/bin/python -u /root/LRTT/experiments/sweep_rank_device.py \
    > /root/LRTT/logs/sweep_rank_device.log 2>&1 &
```

Subset runs:

```bash
# Only LRTT-v2 across all (rank, C)
... sweep_rank_device.py --methods lrtt_v2

# Only one C-tile across all methods/ranks
... sweep_rank_device.py --c_tiles ecram

# A single cell (smoke / debug)
... sweep_rank_device.py --single_combo lrtt_v2 8 ideal
... sweep_rank_device.py --single_combo tikitaka_v1 none rram
```

The script is **resumable** — each cell JSON saves after every trial; killing
and restarting picks up the remaining budget.

## Output layout

```
results/sweep_rank_device/
  lrtt_v1/rank{R}_C{tile}.json     # 15 files
  lrtt_v2/rank{R}_C{tile}.json     # 15 files
  tikitaka_v1/norank_C{tile}.json  # 3 files
```

Each cell JSON contains:
- `warm_start.hp` — the heatmap-baseline best HP enqueued as trial 1
- `trials[]` — all 30 trials with `hp`, `acc`, `wall_seconds`
- `best_hp`, `best_acc`, `n_completed`

## Comparison snippet

```python
import json, glob, os
ROOT = '/root/LRTT/results/sweep_rank_device'

def best_at(method, rank, c):
    if rank is None:
        p = f'{ROOT}/{method}/norank_C{c}.json'
    else:
        p = f'{ROOT}/{method}/rank{rank}_C{c}.json'
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get('best_acc')

ranks = [1, 2, 4, 8, 16]
c_tiles = ['ideal', 'ecram', 'rram']

print('LRTT-v1 best acc')
print('  rank        ' + '   '.join(f'{c:>7s}' for c in c_tiles))
for r in ranks:
    row = [best_at('lrtt_v1', r, c) for c in c_tiles]
    cells = '   '.join('   ?  ' if v is None else f'{v:7.2f}' for v in row)
    print(f'  rank={r:<5d} {cells}')

print('\nLRTT-v2 best acc')
print('  rank        ' + '   '.join(f'{c:>7s}' for c in c_tiles))
for r in ranks:
    row = [best_at('lrtt_v2', r, c) for c in c_tiles]
    cells = '   '.join('   ?  ' if v is None else f'{v:7.2f}' for v in row)
    print(f'  rank={r:<5d} {cells}')

print('\nTikiTaka best acc')
print('  ' + '   '.join(f'{c:>7s}' for c in c_tiles))
row = [best_at('tikitaka_v1', None, c) for c in c_tiles]
cells = '   '.join('   ?  ' if v is None else f'{v:7.2f}' for v in row)
print(f'  {cells}')
```

## Files

- `experiments/sweep_rank_device.py` — sweep driver. Builds RPU configs via
  `PythonLRTTPreset.sixt1c_ab` for v1/v2 and via a manual `ChoppedTransferCompound`
  for TikiTaka (using a 6T1C `LinearStepDevice` with the same parameters as the
  preset).
- `experiments/RANK_DEVICE_SWEEP.md` — this file.

Imports reused from existing repo (no edits needed):

- `aihwkit.simulator.configs.lrtt_python.PythonLRTTPreset.sixt1c_ab` — canonical
  6T1C A/B + configurable C
- `aihwkit.simulator.presets.devices.{IdealizedPresetDevice, EcRamPresetDevice, ReRamESPresetDevice}`
- `experiments/hp_search_v1_decay_gamma_af_2stage.py` — only for `build_loaders`
  and training constants (`EPOCHS`, `HIDDEN`, etc.)

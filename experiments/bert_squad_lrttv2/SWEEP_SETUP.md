# LRTT-v2 BERT/SQuAD 4-Server Sweep — Setup & Run Guide

LRTT-v2 (row-coordinate **selector** + **shuffled-cycle** blockwise transfer)
BERT-base / SQuAD v1.1 finetuning, swept across **4 servers** by
**HP-region partition** (disjoint learning-rate shards), 2D grid over
`learning_rate × transfer_lr`.

Analog arrays: **Core array AND Auxiliary array both 10-bit ConstantStepDevice**
(`dw_min = 2/1024 = 0.001953125`).

---

## 0. Prerequisites (already satisfied on the target servers)

aihwkit-gpu is **already installed** on every server, identical to the
reference environment:

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.3.1+cu121 |
| aihwkit | 1.0.0+cuda121 (GPU wheel) |
| transformers | 4.47.1 |
| datasets | 4.5.0 |
| optuna | 4.7.0 |

> **No install / no build / no editable step.** Per `ENVIRONMENT_SETUP.md`
> Step 6, the LRTT-v2 Python modules are loaded via `sys.path`, shadowing the
> installed wheel (which only supplies the compiled `rpu_base` backend).

Sanity check (optional, one line):

```bash
~/.venv310/bin/python -c "from aihwkit.nn import AnalogLinear; \
from aihwkit.simulator.configs import InferenceRPUConfig; import torch; \
AnalogLinear(4,4,rpu_config=InferenceRPUConfig()).cuda()(torch.randn(2,4).cuda()); \
print('aihwkit GPU OK')"
```

---

## 1. Get the code (every server)

```bash
git clone https://github.com/nmdlkg/LRTT.git    # or git pull if already cloned
cd LRTT
git checkout MLP
git pull origin MLP                              # must include the LRTT-v2 selector
                                                 # 3D-tensor fix + this script
```

The MLP branch contains:
- `src/aihwkit/simulator/tiles/lrtt_controller.py` — LRTT-v2 selector/shuffle
  controller **incl. the transformer 3D-tensor fix** (collapses
  `[batch, seq, feat]` → `[N, feat]` in the selector path).
- `experiments/bert_squad_lrttv2/optuna_bert_squad_lrttv2.py` — this sweep
  script.

`LRTT_SRC` defaults to `/root/LRTT/src`. If your clone is elsewhere, export it:

```bash
export LRTT_SRC=/abs/path/to/LRTT/src
```

(Optional) results dir override (default:
`<repo>/experiments/bert_squad_lrttv2/results`):

```bash
export LRTT_RESULTS=/abs/path/to/results
```

---

## 2. Fixed configuration (identical on all 4 servers)

| Item | Value |
|---|---|
| Model / task | `bert-base-uncased` / SQuAD v1.1 (F1) |
| Target layers | `--lora-target qkv` → attention **Q, K, V, O** (48 modules) |
| LRTT-v2 mode | `update_mode=selector_reconstruction` |
| Selector | `selector_policy=shuffled_cycle`, `selector_axis=row` |
| Rank / block | `--rank 8` (= selector block size) |
| Core array | 10-bit `ConstantStepDevice` (`dw_min=0.001953125`) |
| Auxiliary array (A, B) | 10-bit `ConstantStepDevice` (`dw_min=0.001953125`) |
| transfer_every | `3` (steps; `units_in_mbatch=True`) |
| Epochs / batch | `2` / `16` |
| Data | full SQuAD v1.1 (omit `--train-subset`/`--eval-subset`) |

Also digitally trained alongside the analog QKVO tiles: `qa_outputs` head and
LayerNorm (standard QA finetuning). FFN + embeddings are frozen.

### Sweep grid (2D, HP-region partitioned)

- **LR grid (8 values, log-spaced):**
  `1e-4 3.16e-4 1e-3 3.16e-3 1e-2 3.16e-2 1e-1 3.16e-1`
- **transfer_lr grid (7 values):** `0.1 0.3 1.0 3.0 10.0 30.0 100.0`

The LR grid is split **round-robin across 4 servers** (each server owns 2 LR
values spanning low+high), and every server sweeps the **full** TLR grid:

| Server (`--server-id`) | LR shard | Trials (LR×TLR) |
|---|---|---|
| 0 | `1e-4`, `1e-2` | 2 × 7 = 14 |
| 1 | `3.16e-4`, `3.16e-2` | 14 |
| 2 | `1e-3`, `1e-1` | 14 |
| 3 | `3.16e-3`, `3.16e-1` | 14 |

Total = **56 trials** (8 LR × 7 TLR), no overlap, no shared DB.

---

## 3. Run command — per server

Run the **same command on all 4 servers**, changing only `--server-id`
(0, 1, 2, 3):

```bash
cd LRTT
~/.venv310/bin/python -u experiments/bert_squad_lrttv2/optuna_bert_squad_lrttv2.py \
  --mode lrtt_v2 \
  --lora-target qkv --rank 8 --selector-policy shuffled_cycle \
  --lrtt-device-type constant_step --lrtt-weight-bits 10 \
  --transfer-every 3 --epochs 2 --batch-size 16 \
  --lr-grid 1e-4 3.16e-4 1e-3 3.16e-3 1e-2 3.16e-2 1e-1 3.16e-1 \
  --tlr-grid 0.1 0.3 1.0 3.0 10.0 30.0 100.0 \
  --num-servers 4 --server-id <0|1|2|3> \
  --study-name lrttv2_qkvo_cs10 \
  2>&1 | tee experiments/bert_squad_lrttv2/logs/sweep_srv<0|1|2|3>.log
```

- `--server-id` is the **only** value that differs between servers.
- Each server writes its own study + SQLite DB:
  `results/optuna_lrttv2_qkvo_cs10_srv<k>of4.db`
  and `results/all_trials_bert_squad.json`.
- `n_trials` is derived automatically from the grid (no `--n-trials` needed).

> Runtime note: full SQuAD + 2 epochs ≈ thousands of steps **per trial** ×
> 14 trials/server. To smoke-test the pipeline first, add
> `--train-subset 1500 --eval-subset 400 --epochs 1` (then drop them for the
> real sweep). Grids are fully overridable via `--lr-grid` / `--tlr-grid`.

---

## 4. Merge results (after all servers finish)

Copy each server's `results/optuna_lrttv2_qkvo_cs10_srv<k>of4.db` (or the
per-server `all_trials_bert_squad.json`) to one machine, then:

```bash
~/.venv310/bin/python - <<'PY'
import json, glob, optuna
rows=[]
for db in glob.glob("optuna_lrttv2_qkvo_cs10_srv*of4.db"):
    name=db[len("optuna_"):-3]
    st=optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
    for t in st.trials:
        rows.append({"server":name,"trial":t.number,"f1":t.value,
                     "lr":t.params.get("learning_rate"),
                     "transfer_lr":t.params.get("transfer_lr"),
                     "state":str(t.state)})
rows=[r for r in rows if r["f1"] is not None]
rows.sort(key=lambda r:r["f1"], reverse=True)
json.dump(rows, open("merged_sweep.json","w"), indent=2)
print("best:", rows[0] if rows else "none", "| total", len(rows))
PY
```

The combined 8×7 grid is reconstructed exactly (HP-region partition → disjoint
union with no duplicate cells).

---

## 5. Notes

- **Shuffle coverage:** BERT hidden = 768, block = rank = 8 → one
  `shuffled_cycle` = 768/8 = **96 transfers** before the row permutation
  reshuffles. With `transfer_every=3` and full SQuAD (~5.5k steps/epoch ×2),
  there are ≫96 transfers per trial, so the selector reshuffle path is
  exercised many times.
- Changing only `--server-id` keeps every other factor identical, so results
  across servers are directly comparable.
- `--mode tiki` (default) reproduces the original TikiTaka script behavior
  unchanged; LRTT-v2 applies only with `--mode lrtt_v2`.

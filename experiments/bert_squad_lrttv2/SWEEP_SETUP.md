# LRTT-v2 BERT/SQuAD 4-Server Sweep — Setup & Run Guide

LRTT-v2 (row-coordinate **selector** + **shuffled-cycle** blockwise transfer)
BERT-base / SQuAD v1.1 finetuning, swept across **4 servers**: the LR range
is split into 4 equal log sub-ranges and each server runs an **independent
TPE search** over (`learning_rate`, `transfer_lr`) within its sub-range.
Training regime (optimizer / warmup / min-lr / batch) matches
`experiments/paper/paper_experiment.py`.

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
| Rank / block | `--rank 32` (= selector block size) |
| Core array | 10-bit `ConstantStepDevice` (`dw_min=0.001953125`) |
| Auxiliary array (A, B) | 10-bit `ConstantStepDevice` (`dw_min=0.001953125`) |
| transfer_every | `3` |
| Batch / grad-accum | `--batch-size 16` × `--grad-accum-steps 3` = **effective 48** (paper regime) |
| Epochs | `5` |
| Optimizer | **AnalogAdam** (paper_experiment.py) |
| Warmup | ratio `0.05`, applied to **all** param groups (`--no-analog-only-warmup`, paper behavior) |
| min_lr_rate | `0.05` (`--min-lr-rate`; linear decay to 5% of peak — deviates from paper default 0.5 by request) |
| Weight decay | `0` ; Seed `42` |
| Data | full SQuAD v1.1 (omit `--train-subset`/`--eval-subset`) |

Also digitally trained alongside the analog QKVO tiles: `qa_outputs` head and
LayerNorm.

**LR separation (now active for `--mode lrtt_v2`):** when `--classifier-lr`
(or `--classifier-lr-range`) is given, the optimizer uses **two separate
LRs**:

| Param group | LR | Swept by |
|---|---|---|
| analog LRTT auxiliary tiles (QKVO A/B) | `learning_rate` | **TPE** over this server's LR sub-range |
| digital `qa_outputs` + LayerNorm (same LR) | `classifier_lr` | fixed `2e-3` (`--classifier-lr`) |

Mechanism (aihwkit-specific): the analog tile LR is taken from the
optimizer's top-level `lr=` (`defaults["lr"]`) because
`regroup_param_groups()` rebuilds one group per analog tile without an
explicit lr; the digital group keeps its own `classifier_lr`. Verified:
analog tile LR == `learning_rate`, digital group lr == `classifier_lr`.
If `--classifier-lr` is **omitted**, behavior falls back to a single shared
`learning_rate` (old behavior). `transfer_lr` is a separate device-level B→C
transfer LR (not an optimizer LR).

### Sweep strategy: per-server TPE over a 1/4 LR sub-range

The **full LR range** `[1e-4, 3.16e-1]` is split into **4 equal log
sub-ranges**; each server runs an **independent TPE search** within its own
sub-range. `transfer_lr` is TPE-searched over the **full**
`--tpe-tlr-range` on every server. Per-server TPE seed = `42 + server_id`.

| Server (`--server-id`) | LR sub-range (TPE searches within) |
|---|---|
| 0 | `[1.00e-4, 7.50e-4]` |
| 1 | `[7.50e-4, 5.62e-3]` |
| 2 | `[5.62e-3, 4.21e-2]` |
| 3 | `[4.21e-2, 3.16e-1]` |

`learning_rate` (= analog LRTT auxiliary LR) and `transfer_lr` are the two
TPE-searched dimensions. `--n-trials` per server is chosen by you (e.g. 25
→ 100 total across 4 servers). No grid, no overlap, no shared DB.

---

## 3. Run command — per server

Run the **same command on all 4 servers**, changing only `--server-id`
(0, 1, 2, 3):

```bash
cd LRTT
~/.venv310/bin/python -u experiments/bert_squad_lrttv2/optuna_bert_squad_lrttv2.py \
  --mode lrtt_v2 \
  --lora-target qkv --rank 32 --selector-policy shuffled_cycle \
  --lrtt-device-type constant_step --lrtt-weight-bits 10 \
  --transfer-every 3 --epochs 5 \
  --optimizer AnalogAdam \
  --warmup-ratio 0.05 --no-analog-only-warmup \
  --batch-size 16 --grad-accum-steps 3 \
  --classifier-lr 2e-3 \
  --lr-range 1e-4 3.16e-1 \
  --tpe-tlr-range 1e-3 100 \
  --min-lr-rate 0.05 \
  --n-trials 25 \
  --num-servers 4 --server-id <0|1|2|3> \
  --study-name lrttv2_qkvo_cs10_r32te3 \
  2>&1 | tee experiments/bert_squad_lrttv2/logs/sweep_srv<0|1|2|3>.log
```

- `--server-id` is the **only** value that differs between servers.
- `--lr-range 1e-4 3.16e-1` is the **full** LR range; the script
  auto-assigns this server its 1/4 log sub-range (table above) and runs TPE
  inside it. `--tpe-tlr-range 1e-3 100` is the `transfer_lr` range
  (**10⁻³ – 10²**, log), TPE-searched **in full on every server**. For
  `--mode lrtt_v2` the TikiTaka-v2 `scale_transfer_lr` heuristic (which would
  cap the upper at `1/learning_rate`, coupling it to lr per trial/server) is
  **force-disabled** so transfer_lr uses the literal `[1e-3, 100]` range
  consistently across all 4 servers.
- `--batch-size 16 --grad-accum-steps 3` → **effective batch 48** (paper
  regime): `loss/=3`, `optimizer.step()` only every 3 micro-batches.
- `--classifier-lr 2e-3` separates LR: analog LRTT auxiliary tiles use the
  TPE-searched `learning_rate`; digital `qa_outputs` **and** LayerNorm share
  the fixed `2e-3` (= original TikiTaka `SQUAD_LR`). classifier_lr == ln_lr
  (they are one digital group).
- `--optimizer AnalogAdam --warmup-ratio 0.05 --no-analog-only-warmup`
  matches `paper_experiment.py`: AnalogAdam, linear warmup (ratio 0.05)
  applied to **all** param groups, seed 42, weight_decay 0. **Deviation:**
  `--min-lr-rate 0.05` (paper default is 0.5) — LR decays to 5% of peak by
  request.
- `--n-trials 25` → 25 TPE trials/server (≈100 total). Tune as needed.
- Each server writes its own study + SQLite DB:
  `results/optuna_lrttv2_qkvo_cs10_r32te3_srv<k>of4.db`
  and `results/all_trials_bert_squad.json`.

> Runtime note: full SQuAD + 5 epochs ≈ tens of thousands of optimizer steps
> **per trial**. To smoke-test the pipeline first, add
> `--train-subset 1500 --eval-subset 400 --epochs 1 --n-trials 1` (then drop
> them for the real sweep).

---

## 4. Merge results (after all servers finish)

Copy each server's `results/optuna_lrttv2_qkvo_cs10_r32te3_srv<k>of4.db` (or
the per-server `all_trials_bert_squad.json`) to one machine, then:

```bash
~/.venv310/bin/python - <<'PY'
import json, glob, optuna
rows=[]
for db in glob.glob("optuna_lrttv2_qkvo_cs10_r32te3_srv*of4.db"):
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

The 4 servers' TPE searches cover disjoint LR sub-ranges (their union = the
full `[1e-4, 3.16e-1]` range); merging gives the global best across the whole
range with no overlap.

---

## 5. Notes

- **Shuffle coverage:** BERT hidden = 768, block = rank = 32 → one
  `shuffled_cycle` = 768/32 = **24 transfers** before the row permutation
  reshuffles (768 % 32 = 0, no partial block). With `transfer_every=3` and
  full SQuAD × 5 epochs there are far more than 24 transfers per trial, so
  the selector reshuffle path is exercised many times.
- Changing only `--server-id` keeps every other factor identical (same
  optimizer/warmup/seed/data), so results across servers are directly
  comparable and the 4 LR sub-ranges tile the full range.
- `--mode tiki` (default) reproduces the original TikiTaka script behavior
  unchanged; LRTT-v2 applies only with `--mode lrtt_v2`.

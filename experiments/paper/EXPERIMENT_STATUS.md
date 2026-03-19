# Experiment Status & Pending Runs

Last updated: 2026-03-19

## Critical Bug Notice

**`tile.rpu_config` modification after creation has NO effect in aihwkit.**
All experiments using `_apply_per_layer_io_bits()` (pre-fix) ran as **uniform base IO bits**.
Only experiments using `specific_rpu_config_fun` at `convert_to_analog()` time are valid.

Fix commit: `4203ff9` (transformer branch)

---

## 1. Completed Experiments (Valid Results)

### 1.1 Uniform IO Bit Sweep (IdealDevice, all layers, abs_max NM)

| IO bits | BM | min_lr | best F1 | final F1 | EM | Status |
|---------|------|--------|---------|----------|------|--------|
| 4 | iterative | 0.5 | 6.96 | 5.21 | 1.57 | collapse |
| 4 | iterative | 0.05 | 6.96 | 3.61 | 0.81 | collapse |
| 5 | iterative | 0.5 | 7.19 | 4.97 | 1.44 | collapse |
| 6 | iterative | 0.05 | 85.61 | — | 77.56 | epoch 3 best (interrupted ep4 78%) |
| 7 | iterative | 0.05 | 83.85 | — | 75.73 | epoch 1 only (interrupted) |
| 8 | iterative | 0.5 | 86.84 | — | 79.26 | epoch 3 best (interrupted ep4) |
| 10 | iterative | 0.05 | 87.31 | 87.31 | 80.06 | completed |
| 12 | iterative | 0.5 | 87.36 | 87.36 | 80.11 | completed |

Source: `experiments/paper/results/paper/io_sweep/io_sweep_summary.csv`

### 1.2 Noise Management Ablation

| IO bits | NM | best F1 | Status |
|---------|------|---------|--------|
| 4 | none | 12.13 | collapse |
| 8 | none | 13.24 | collapse |
| 10 | none | 13.10 | collapse |

**abs_max noise management is essential.** Without it, even 10b collapses.

### 1.3 SA Allocation e2e (Pre-bug, SA_DIR allocation)

| Budget | best F1 | Note |
|--------|---------|------|
| budget 9 | 87.42 | Pre-bug, but allocation was QZR-based with b_min=4 |
| budget 8 | — | No summary (may have failed) |

---

## 2. Invalidated Experiments (Bug: per-layer-bits not applied)

These all ran as **uniform 8b** despite per-layer-bits being specified:

| Experiment | Intended | Actual | F1 |
|-----------|----------|--------|-----|
| `sa_v4_salpa_ideal/salpa_minimax_avg5.0` | avg 5b mixed | uniform 8b | 87.19 |
| `sa_v4_salpa_ideal/salpa_minimax_avg6.0` | avg 6b mixed | uniform 8b | (incomplete) |
| `sa_v4_io6_cs10/salpa_minimax_avg6.0` | avg 6b mixed | uniform 8b | (no summary) |
| `sa_v4_io6_cs10/salpa_minimax_avg8.0` | avg 8b mixed | uniform 8b | (no summary) |
| `paper/mixed_L4` | mixed layer 4 | uniform 8b | (no summary) |
| `paper/mixed_L5` | mixed layer 5 | uniform 8b | (no summary) |

---

## 3. In-Progress Experiments (Fixed Code)

### 3.1 Training-Aware Sensitivity Analysis (Phase 1)

**Goal**: Determine which sublayer **group** causes collapse when lowered to 4b.
Base = 8b, one group → 4b, **4 epochs** full training.

Groups:
- **QKV** (in-projection): Q=4, K=4, V=4, rest=8
- **O** (out-projection): O=4, rest=8
- **FFN1**: FFN1=4, rest=8
- **FFN2**: FFN2=4, rest=8

```bash
cd /root && nohup bash run_training_sensitivity.sh > results/sa_v4_training_sensitivity/nohup.log 2>&1 &
```

| Experiment | Setting | Status | best F1 |
|-----------|---------|--------|---------|
| sens_QKV_4b | Q=K=V=4, rest=8 | pending | — |
| sens_O_4b | O=4, rest=8 | pending | — |
| sens_FFN1_4b | FFN1=4, rest=8 | pending | — |
| sens_FFN2_4b | FFN2=4, rest=8 | pending | — |

Expected runtime: ~7h per run × 4 = ~28h total.

Previous observation (aborted 1ep run): Q=4b alone (rest=8b) showed normal
loss descent (5.9 → 1.1 at 41%), suggesting individual sublayer at 4b may
not collapse. This 4-group design tests whether grouped low-bit causes collapse.

### 3.2 SALPA Fixed (Collapsed — Stopped)

| Budget | FFN1 bits | QKVO bits | Loss trend | Status |
|--------|-----------|-----------|------------|--------|
| avg 5.0b | 6-8b | 4-5b | 5.93 → 5.98 (rising) | **collapsed, stopped** |
| avg 5.5b | started | — | killed immediately | stopped |

**Key finding**: QZR-based allocation (protect FFN1, sacrifice QKVO) causes
training collapse. Loss never decreases, matching uniform 5b collapse pattern.

---

## 4. Experiments To Run

### 4.1 After Phase 1 Sensitivity (Priority: HIGH)

Based on Phase 1 results, design SALPA allocation with **training-aware** constraints:

If QKVO is sensitive (Phase 1 shows collapse at 4b):
- Protect QKVO (b_min=6), sacrifice FFN1/FFN2 (b_min=3-4)
- Run avg 4.0, 4.5, 5.0, 5.5, 6.0

If FFN1 is sensitive (Phase 1 shows collapse at 4b):
- Current QZR-based allocation was correct
- Need higher avg budget (cannot go below 6b avg)

### 4.2 Uniform Completions (Priority: MEDIUM)

| Experiment | Command | Note |
|-----------|---------|------|
| uniform 6b full | `--io-bits 6 --epochs 4` | Currently only epoch 3 best (85.61), need full 4ep |
| uniform 7b full | `--io-bits 7 --epochs 4` | Only epoch 1 completed (83.85) |
| uniform 8b full | `--io-bits 8 --epochs 4` | Only epoch 3 best (86.84), need full 4ep |

### 4.3 Phase 2 Sensitivity — Per-Layer (Priority: LOW)

After identifying the most sensitive sublayer type from Phase 1,
run per-layer sensitivity (12 runs × 1 epoch):

```bash
# Example: if Q is most sensitive
python paper_experiment.py ... --per-layer-bits "L0:Q=4;rest=8" --epochs 1
```

### 4.4 QKVO-Protection SALPA (Priority: HIGH, after Phase 1)

If Phase 1 confirms QKVO needs protection:

```bash
# avg 5.0b with QKVO=6b, FFN=3-4b
python paper_experiment.py --method ideal --target-layers all \
  --noise-management abs_max --io-bits 8 \
  --per-layer-bits "Q=6,K=6,V=6,O=6,FFN1=3,FFN2=3" \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --epochs 4 --seed 42 \
  --mode fixed --min-lr-rate 0.05 \
  --output-dir results/salpa_qkvo_protect/avg5.0
```

---

## 5. Common Run Commands

### Base settings (all experiments use these)
```bash
source ~/.venv310/bin/activate  # or appropriate venv
cd /root  # or experiments/paper in LRTT repo

COMMON="--method ideal --target-layers all --noise-management abs_max \
  --analog-lr 0.0357 --classifier-lr 0.00076 --ln-lr 0.00076 \
  --batch-size 24 --grad-accum-steps 2 --seed 42 --mode fixed \
  --min-lr-rate 0.05"
```

### Uniform IO sweep
```bash
python paper_experiment.py $COMMON --io-bits 6 --epochs 4 --output-dir results/uniform_6b
```

### Per-sublayer sensitivity (1 epoch)
```bash
python paper_experiment.py $COMMON --io-bits 8 --epochs 1 \
  --per-layer-bits "Q=4,K=8,V=8,O=8,FFN1=8,FFN2=8" \
  --output-dir results/sens_Q_4b
```

### SALPA with precision map JSON
```bash
python paper_experiment.py $COMMON --io-bits 8 --epochs 4 \
  --per-layer-bits path/to/precision_map.json \
  --output-dir results/salpa_xxx
```

---

## 6. Key Files

| File | Location | Description |
|------|----------|-------------|
| paper_experiment.py | `/root/` or `experiments/paper/` | Main training script (fixed) |
| rpu_configs.py | `/root/` or `experiments/paper/` | RPU config builders |
| run_training_sensitivity.sh | `/root/` | Phase 1 sensitivity script |
| run_salpa_lowbit.sh | `experiments/paper/` | SALPA avg 4.0/4.5 launcher |
| SALPA_ALLOCATION_ANALYSIS.md | `experiments/paper/` | Analysis & design notes |
| precision_map_minimax_avg*.json | `results/sa_v4_io6_cs10/` | QZR-based precision maps |
| io_sweep_summary.csv | `experiments/paper/results/paper/io_sweep/` | Uniform sweep results |

# Experiment Plan: IO Bit Resolution & Layerwise Mixed-Precision

## Objective

Measure the **actual training impact** of ADC/DAC bit resolution on BERT-base SQuAD
fine-tuning with IdealDevice, and validate layerwise mixed-precision as a cost-effective
mitigation strategy.

## Background (from backward gradient diagnostic)

- Diagnostic data: `main_results/results/squad/seed_42/metrics_B_bitsweep_summary.csv`
- At **8-bit baseline**, gradient cosine_sim > 0.999 for most modules, but FFN1 (cosine=0.9993, QZR=0.32) and K (QZR=0.12) show elevated quantization error.
- QZR is a validated proxy metric (Spearman rho=0.913 vs cosine_sim and rel_l2_error).
- **No actual training experiments at different bit resolutions exist yet.**

---

## Experiment 1: Uniform Bit Resolution Sweep

**Goal**: Establish training accuracy vs bit resolution curve.

### Configurations

| Run | ADC/DAC bits | `--io-bits` | Notes |
|-----|-------------|-------------|-------|
| 1a  | 4           | `--io-bits 4`  | Extreme low-res |
| 1b  | 6           | `--io-bits 6`  | Low-res |
| 1c  | 8           | `--io-bits 8`  | **Baseline** |
| 1d  | 10          | `--io-bits 10` | High-res |
| 1e  | 12          | `--io-bits 12` | Very high-res |
| 1f  | perfect     | (no `--io-bits`) | FP32 reference (no IO quantization) |

### Command Template

```bash
PYTHON="/data/venvs/lrtt/bin/python"
SCRIPT="main_results/scripts/analysis/optuna_bert_squad_tiki.py"

COMMON="--target-ideal --n-trials 1 --epochs 2 --batch-size 48 \
        --lora-target all --shared-lr --lr 2e-3"

# 1a-1e: Uniform bit sweep
for BITS in 4 6 8 10 12; do
  $PYTHON $SCRIPT $COMMON --io-bits $BITS \
    --study-name "io_sweep_${BITS}b"
done

# 1f: Perfect IO (FP32 reference)
$PYTHON $SCRIPT $COMMON \
  --study-name "io_sweep_perfect"
```

### Expected Output

- F1 score at each bit resolution
- Training loss curve comparison
- Identify the minimum bit resolution where F1 matches FP32 within tolerance

---

## Experiment 2: Layerwise Mixed-Precision

**Goal**: Show that selective bit allocation achieves near-perfect F1 with lower average bit cost.

### Bit Assignment (from diagnostic analysis)

Based on backward gradient QZR analysis at 8-bit baseline (SQuAD seed_42),
minimum bits required per sublayer/layer for QZR < 0.10:

#### Detailed Bit Map

| Layer | Q  | K   | V  | O  | FFN1 | FFN2 |
|-------|----|-----|----|----|------|------|
| L0    | 8b | 8b  | 8b | 6b | 12b  | 6b   |
| L1    | 8b | 8b  | 8b | 6b | 12b  | 6b   |
| L2    | 12b| 10b | 8b | 8b | 12b  | 8b   |
| L3    | 8b | 8b  | 8b | 8b | 12b  | 8b   |
| L4    | 8b | 10b | 8b | 8b | 12b  | 6b   |
| L5    | 10b| 10b | 8b | 6b | 12b  | 6b   |
| L6    | 10b| 10b | 8b | 6b | 12b  | 6b   |
| L7    | 10b| 10b | 8b | 6b | 12b  | 6b   |
| L8    | 8b | 10b | 8b | 6b | 12b  | 6b   |
| L9    | 8b | 10b | 8b | 6b | 12b  | 6b   |
| L10   | 8b | 10b | 8b | 6b | 10b  | 6b   |
| L11   | 8b | 10b | 8b | 6b | 10b  | 6b   |

#### Summary

| Min bits | Modules | % of 72 | Sublayers |
|----------|---------|---------|-----------|
| 6b       | 18      | 25.0%   | O (L0-1,5-11), FFN2 (L0-1,4-11) |
| 8b       | 29      | 40.3%   | V (all), Q (most), O/FFN2 (middle) |
| 10b      | 14      | 19.4%   | K (L4-11), Q (L5-7), FFN1 (L10-11) |
| 12b      | 8       | 11.1%   | FFN1 (L2-9) |
| >12b     | 3       | 4.2%    | FFN1 (L0-1), Q (L2) — cap at 12b |

- **Optimal mix**: 612 total bits vs Uniform 12b (864) = **29.2% saving**
- **vs Uniform 8b**: only +6.2% bit-budget, but all modules meet QZR < 0.10

#### Simplified Option A (sublayer-level)

For implementation simplicity, assign by sublayer type:

| Sublayer | Bits | Modules | Rationale |
|----------|------|---------|-----------|
| V        | 8b   | 12      | QZR < 0.10 at 8b for all layers |
| O        | 6b   | 12      | QZR < 0.07 at 6b for 10/12 layers |
| FFN2     | 6b   | 12      | QZR < 0.07 at 6b for 10/12 layers |
| Q        | 8b   | 12      | QZR < 0.10 at 8b for 8/12 layers |
| K        | 10b  | 12      | QZR > 0.10 at 8b for 9/12 layers |
| FFN1     | 12b  | 12      | QZR > 0.10 at 10b for 10/12 layers |

### Implementation Note

**Current `--io-bits` applies uniformly to all modules.** Layerwise mixed-precision
requires code modification to `optuna_bert_squad_tiki.py`:

1. Add `--io-bits-map` argument accepting JSON or per-sublayer specification
2. Modify `create_tikitaka_config()` to accept per-module io_bits
3. After `convert_to_analog()`, iterate over `AnalogLinear` modules and set
   per-tile `rpu_config.forward.inp_res` / `backward.inp_res` individually

```python
# Example: per-module IO bit assignment after model conversion
SUBLAYER_BITS = {"Q": 8, "K": 10, "V": 8, "O": 6, "FFN1": 12, "FFN2": 6}

for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        sublayer = identify_sublayer(name)  # parse Q/K/V/O/FFN1/FFN2
        bits = SUBLAYER_BITS.get(sublayer, 8)
        io_res = 1.0 / (2 ** bits - 2)
        tile = module.analog_tile
        tile.rpu_config.forward.inp_res = io_res
        tile.rpu_config.forward.out_res = io_res
        tile.rpu_config.backward.inp_res = io_res
        tile.rpu_config.backward.out_res = io_res
```

### Experiment 2 Configurations

| Run | Description | Bit Assignment |
|-----|-------------|---------------|
| 2a  | Option A (sublayer-level) | O,FFN2=6b; Q,V=8b; K=10b; FFN1=12b |
| 2b  | Optimal (per-module) | Detailed bit map above |
| 2c  | Conservative uniform | All 10b (comparison) |

---

## Experiment Comparison Matrix

| Run | Avg bits/module | Expected F1 | Purpose |
|-----|----------------|-------------|---------|
| 1a (4b) | 4.0 | Degraded | Lower bound |
| 1b (6b) | 6.0 | Slightly degraded | |
| 1c (8b) | 8.0 | Near-perfect | Baseline |
| 1d (10b)| 10.0 | ~Perfect | |
| 1e (12b)| 12.0 | ~Perfect | |
| 1f (perf)| inf | Perfect | FP32 reference |
| 2a (Option A) | 8.3 | ~10b level | **Key result** |
| 2b (Optimal) | 8.5 | ~10b level | Best efficiency |

---

## Key Questions to Answer

1. **At what bit resolution does training F1 match FP32?** (Exp 1)
2. **Does layerwise assignment (avg 8.3b) match uniform 10b or 12b?** (Exp 2 vs 1)
3. **Is FFN1 truly the bottleneck sublayer?** (Compare 2a with FFN1@12b vs FFN1@8b)
4. **Does 6b for O/FFN2 cause any degradation?** (Compare 2a vs 1c)

---

## Data Sources

- Diagnostic CSV: `main_results/results/squad/seed_42/metrics_B_bitsweep_summary.csv`
- QZR validity figure: `main_results/results/figures/paper/FigS_qzr_validity_1x3.png`
- Plotting script: `main_results/scripts/plotting/plot_qzr_validity.py`
- Bit map derivation: `main_results/scripts/plotting/paper_main_figure_v2.py`

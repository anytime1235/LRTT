# SALPA: Sensitivity-Aware Layer-wise Precision Allocation — Analysis & Design Notes

## 1. Background: IO Quantization Collapse in Analog IMC

BERT-base fine-tuning on SQuAD v1.1 with IdealDevice (FP32 weight update) shows
sharp performance collapse below 6-bit uniform IO quantization:

| Uniform IO bits | best F1 | Status |
|-----------------|---------|--------|
| 12b | 87.36 | baseline |
| 10b | 87.31 | ~baseline |
| 8b | 86.84 | slight drop |
| 6b | 85.61 | noticeable |
| **5b** | **7.19** | **collapse** |
| **4b** | **6.96** | **collapse** |

The 5b→6b transition is a cliff: F1 drops from 85.61 to 7.19.
SALPA aims to recover performance below this cliff via mixed-precision allocation.

---

## 2. Layer Importance: QKVO vs FFN

### 2.1 Training Contribution (F1 on SQuAD)

When selectively converting layers to analog with 10b IO:

| Target Layers | F1 | Note |
|---------------|------|------|
| QKVO only | 87.5 | attention only |
| FFN only | 86.1 | feed-forward only |
| All layers | 87.6 | both |

**QKVO contributes more to learning** — attention-only conversion recovers nearly
all of the full-model performance, while FFN-only is 1.5 F1 points lower.

### 2.2 Forward Activation Magnitude

However, FFN1's **forward activation is the dominant signal** in each transformer layer:

| Layer | Residual Norm | FFN1 Output Norm | Ratio |
|-------|--------------|-------------------|-------|
| L0 | 37.4 | 136.9 | **3.66x** |
| L3 | 36.5 | 142.2 | **3.89x** |
| L8 | 38.5 | 115.3 | **3.00x** |
| L11 | 33.6 | 83.3 | **2.48x** |

FFN1 output is **3-4x larger** than the residual (attention output).
This means:

1. **Forward**: FFN1 output dominates each layer's output.
   Quantization errors in FFN1 are amplified downstream.
2. **Backward**: Gradients to QKVO must flow **through FFN1**.
   If FFN1 IO is heavily quantized, gradient signal reaching QKVO is corrupted.
3. **Residual cannot compensate**: Unlike a skip connection that can mask a
   small perturbation, FFN1's contribution is 3-4x the residual — destroying
   it cannot be absorbed.

### 2.3 The Paradox

| Perspective | Most Important | Reasoning |
|-------------|---------------|-----------|
| Training (F1) | QKVO | Attention learns task-specific patterns |
| Forward signal | FFN1 | Dominant activation magnitude |
| Gradient path | FFN1 | QKVO gradients flow through FFN1 |

**Resolution**: QKVO is the "learner" but FFN1 is the "highway."
Protecting FFN1's IO precision **indirectly protects QKVO's gradient quality**
because gradients must traverse FFN1 to reach QKVO.
Even if QKVO has perfect IO, corrupted gradients from a low-bit FFN1
upstream will degrade QKVO training.

---

## 3. Sensitivity Metrics: QZR vs Forward Activation

### 3.1 Current Metrics (Normalized Space)

The QZR-based sensitivity analysis computes metrics on **normalized** vectors
(scaled to [-1, 1] by per-vector absmax):

| Sublayer | QZR (4b) | Cosine Sim | Rel L2 Error | Sign Agree |
|----------|----------|------------|--------------|------------|
| **FFN1** | **0.919** | **0.934** | **0.360** | **0.081** |
| Q | 0.614 | 0.964 | 0.268 | 0.386 |
| K | 0.682 | 0.966 | 0.258 | 0.318 |
| V | 0.599 | 0.973 | 0.233 | 0.401 |
| O | 0.236 | 0.984 | 0.170 | 0.764 |
| FFN2 | 0.210 | 0.988 | 0.151 | 0.790 |

FFN1 is worst on **all** normalized metrics. But these metrics do not capture
the actual activation magnitude — they are computed in normalized [-1,1] space.

### 3.2 Why QZR Alone Is Insufficient

QZR measures "what fraction of non-zero values get zeroed by quantization."
It does **not** tell us:
- How large the affected activations are (magnitude)
- How much the overall network output changes (end-to-end impact)
- Whether the zeroed values are in the "body" or "tail" of the distribution

The true sensitivity should be:
```
Absolute Impact ≈ Quantization Error × Activation Magnitude
```

### 3.3 Why It Still Works for FFN1

In practice, FFN1 happens to be worst on **both** dimensions:
- Highest normalized distortion (QZR=0.92, cosine=0.93)
- Highest activation magnitude (3-4x residual)

So QZR-based allocation accidentally gives the correct answer for FFN1.
However, for future work or different architectures, a composite metric
incorporating activation magnitude would be more principled.

### 3.4 Comparison with Literature

Existing papers primarily study **weight quantization**:

| Paper | Finding | Metric |
|-------|---------|--------|
| Q-BERT (AAAI 2020) | FFN weights tolerate low-bit | Hessian eigenvalue |
| ZeroQuant (NeurIPS 2022) | FFN=INT4, Attention=INT8 | Weight quantization |
| SensiMix (PLOS ONE 2022) | Attention more sensitive than FFN | Weight sensitivity |
| IBM Analog BERT (2021) | FFN→analog, Attention→digital INT6 | Analog noise |

**Key distinction**: These papers analyze **weight** quantization, where FFN
tolerates lower precision. Our setting is **IO (activation) quantization** in
analog IMC, where the conclusions are **reversed** — FFN1's wide activation
dynamic range makes it the most sensitive sublayer to IO quantization.

This reversal is a potential contribution of our work:
> In analog in-memory computing, IO quantization sensitivity is dominated
> by forward activation dynamic range, not weight distribution. This causes
> FFN1 — traditionally considered quantization-robust — to become the most
> sensitive sublayer, requiring precision protection in mixed-precision schemes.

---

## 4. SALPA Allocation Design

### 4.1 Constrained Greedy Allocation

The allocator uses marginal-gain greedy optimization with per-sublayer
minimum bit constraints:

```
Per-sublayer b_min constraints:
  FFN1 >= 6b    (protected — highest sensitivity)
  Q/K/V >= 4b   (moderate)
  O/FFN2 >= 3b  (aggressive reduction — lowest sensitivity)
  b_max = 8b
```

Budget is distributed by greedy marginal gain: each extra bit goes to the
module where it reduces risk the most (QZR-lexi surrogate).

### 4.2 Generated Precision Maps

| Budget | Q | K | V | O | FFN1 | FFN2 |
|--------|---|---|---|---|------|------|
| avg 4.0b | 4.0 | 4.0 | 4.0 | 3.0 | **6.0** | 3.0 |
| avg 4.5b | 4.6 | 4.2 | 4.2 | 4.0 | **6.0** | 4.0 |
| avg 5.0b | 5.0 | 5.4 | 5.0 | 4.1 | **6.5** | 4.0 |
| avg 5.5b | 5.8 | 6.0 | 5.9 | 4.2 | **7.1** | 4.0 |
| avg 6.0b | 6.2 | 6.8 | 6.1 | 4.7 | **7.9** | 4.3 |

Note: Per-layer variation exists within each sublayer type (sensitivity-driven).
For example in avg 4.5b, Q ranges from 4 (L2,L5-L11) to 5 (L0,L1,L3,L4).

### 4.3 Design Rationale

1. **FFN1 always gets highest bits**: Protects forward signal (dominant
   activation) and backward gradient path to QKVO.
2. **O and FFN2 get lowest bits**: Lowest QZR, smallest activation magnitude,
   least impact on gradient flow.
3. **QKVO gets middle bits**: Important for learning, but their gradients are
   more dependent on FFN1 signal quality than on their own IO precision.
4. **Per-layer variation**: Early layers (L0-L1) tend to get slightly more
   bits due to higher marginal gain (sensitivity varies by depth).

---

## 5. Critical Bug: tile.rpu_config Post-Creation Modification

### 5.1 The Bug

```python
# THIS DOES NOT WORK in aihwkit:
for tile in module.analog_tiles():
    tile.rpu_config.forward.inp_res = new_res  # silently ignored
```

Modifying `tile.rpu_config` after tile creation has **no effect** on tile behavior.
The C++ backend reads config only at initialization time.

**Impact**: All SALPA experiments before the fix (F1=87.19 for avg 5.0b) were
actually running as **uniform 8b** — the per-layer-bits override was silently
ignored. The reported "SALPA recovery" was in fact just the uniform 8b baseline.

### 5.2 The Fix

```python
# CORRECT: use specific_rpu_config_fun at convert_to_analog() time
def specific_fn(module_name, module, rpu_config):
    key = _get_sublayer_key(module_name)
    if key is not None and key in plb_dict:
        cfg = deepcopy(rpu_config)
        cfg.forward.inp_res = io_res_from_bits(plb_dict[key])
        cfg.forward.out_res = io_res_from_bits(plb_dict[key])
        cfg.backward.inp_res = io_res_from_bits(plb_dict[key])
        cfg.backward.out_res = io_res_from_bits(plb_dict[key])
        return cfg
    return rpu_config

model = convert_to_analog(model, rpu_config,
                          specific_rpu_config_fun=specific_fn)
```

This applies IO resolution at tile creation time via `specific_rpu_config_fun`,
the only correct way to set per-tile IO parameters in aihwkit.

### 5.3 Verification

```
Applied to 72 modules
L0    Q: expected 4b (res=0.071429), actual res=0.071429 [OK]
L0 FFN1: expected 6b (res=0.016129), actual res=0.016129 [OK]
```

Initial F1 changed from 7.24 (uniform 8b) to 6.93 (mixed precision),
confirming per-layer-bits are now active.

---

## 6. Experiment Status & Next Steps

### 6.1 Running Experiments (Fixed Code)

| Budget | Precision Map | Status |
|--------|--------------|--------|
| avg 5.0b | `precision_map_minimax_avg5.0.json` | running |
| avg 5.5b | `precision_map_minimax_avg5.5.json` | queued |
| avg 6.0b | `precision_map_minimax_avg6.0.json` | queued |
| avg 4.0b | `precision_map_minimax_avg4.0.json` | ready (other env) |
| avg 4.5b | `precision_map_minimax_avg4.5.json` | ready (other env) |

### 6.2 Key Questions to Answer

1. **Does SALPA actually recover performance below the 5b cliff?**
   Previous "recovery" was a bug. The fixed experiments will give real answers.

2. **FFN1 protection vs QKVO protection**: If avg 5.0b (FFN1=6-8b, QKVO=4-5b)
   still collapses, try the inverse allocation (FFN1=4b, QKVO=6-8b) to test
   whether the literature's weight-quantization intuition applies to IO.

3. **Composite sensitivity metric**: Incorporate activation magnitude into the
   risk function: `R_m(b) = QZR_m(b) × activation_norm_m` for more principled
   allocation. This requires a forward pass to collect activation norms.

### 6.3 Files & Locations

```
/root/results/sa_v4_io6_cs10/
  precision_map_minimax_avg{4.0,4.5,5.0,5.5,6.0}.json  — precision maps

/root/results/sa_v4_salpa_ideal_fixed/
  salpa_minimax_avg{5.0,5.5,6.0}/                       — fixed experiment results

LRTT/experiments/paper/
  paper_experiment.py              — fixed per-layer-bits via specific_rpu_config_fun
  run_salpa_lowbit.sh              — launcher for avg 4.0b, 4.5b
  results/paper/salpa_lowbit/      — precision maps for 4.0b, 4.5b
```

---

## 7. Summary

| Insight | Evidence | Implication |
|---------|----------|-------------|
| FFN1 has highest IO sensitivity | QZR=0.92, cosine=0.93 at 4b | Protect FFN1 in allocation |
| FFN1 forward activation is dominant | 3-4x residual norm | Quantization errors amplified |
| QKVO gradients flow through FFN1 | Transformer architecture | FFN1 protection = QKVO gradient protection |
| Weight vs IO quantization differ | Literature says FFN robust (weights) | Our IO finding is reversed — novelty |
| tile.rpu_config modification is no-op | aihwkit C++ backend | Must use specific_rpu_config_fun |
| QZR alone is incomplete metric | Doesn't capture activation magnitude | Future: composite metric needed |

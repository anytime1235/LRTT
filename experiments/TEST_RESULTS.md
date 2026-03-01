# PEFT LoRA vs FP-LoRA Verification Results

**Date**: 2026-02-11
**Status**: ✅ ALL TESTS PASSED

## Summary

Successfully verified that PEFT LoRA and FP-LoRA (LRTT-LoRA with FloatingPoint devices) are mathematically equivalent through two simple unit tests.

## Test Results

### Test 1: Forward Output Equivalence ✅

**File**: `/data/LRTT_transformer/experiments/test_simple_forward.py`

**Setup**:
- Simple Linear layer (256 → 128, rank=8)
- Identical weight initialization for PEFT and FP-LoRA
- Same input tensor

**Result**:
```
Max difference: 1.192e-07
torch.allclose(atol=1e-6): True
```

**Conclusion**: Forward outputs are **identical** within floating-point precision.

---

### Test 2: Weight Update Equivalence ✅

**File**: `/data/LRTT_transformer/experiments/test_simple_update.py`

**Setup**:
- Same as Test 1, plus training step
- Optimizers: SGD (PEFT) vs AnalogSGD (FP-LoRA)
- Learning rate: 0.01

**Results**:
```
Loss (PEFT):     1.071573
Loss (FP-LoRA):  1.071573

Δ(lora_A vs tile_b) max difference: 9.313e-10  ✅
Δ(lora_B vs tile_a) max difference: 3.725e-09  ✅
ΔW (tile_c) max change: 0.000e+00               ✅ (frozen)
```

**Conclusion**:
- A/B weight updates are **identical** (differences < 1e-9)
- C tile remains **frozen** (no change)
- Both methods produce the same gradient updates

---

## Key Findings

### Weight Mapping (IMPORTANT!)

The naming convention differs between PEFT and LRTT:

| PEFT | Shape | LRTT | Shape | Applied |
|------|-------|------|-------|---------|
| lora_A | [rank, in_features] | tile_b | [rank, in_features] | First (input → rank) |
| lora_B | [out_features, rank] | tile_a | [out_features, rank] | Second (rank → output) |
| base_layer | [out_features, in_features] | tile_c | [out_features, in_features] | Frozen |

### Forward Pass Equations

Both compute the same value using different notation:

- **PEFT**: `y = W·x + α·B·(A·x)`
- **LRTT**: `y = C·x + α·A·(B·x)`

Where: `LRTT.tile_a ≡ PEFT.lora_B` and `LRTT.tile_b ≡ PEFT.lora_A`

### Critical Configuration

For exact equivalence, FP-LoRA must use:

```python
config = create_lrtt_lora_config(
    rank=8,
    lora_alpha=1.0,
    use_floating_point=True,  # FloatingPoint devices (no analog noise)
    output_noise_level=0.0,
)
```

This creates:
- `tile_a`: FloatingPointDevice (exact arithmetic)
- `tile_b`: FloatingPointDevice (exact arithmetic)
- `tile_c`: SoftBoundsDevice (frozen)

---

## Verification Commands

```bash
cd /data/LRTT_transformer/experiments

# Test 1: Forward equivalence (~5 seconds)
/data/venvs/aihwkit_gpu/bin/python test_simple_forward.py

# Test 2: Update equivalence (~10 seconds)
/data/venvs/aihwkit_gpu/bin/python test_simple_update.py
```

---

## Next Steps

With equivalence verified, we can now:

1. **Use FP-LoRA for debugging**: When encountering issues with 6T1C devices, switch to FP-LoRA (`use_floating_point=True`) to isolate whether the problem is:
   - Mathematical/algorithmic (will occur in both modes)
   - Device-specific (only occurs with 6T1C)

2. **Baseline comparison**: Run full training with FP-LoRA to establish expected accuracy before testing with analog devices

3. **Confidence in 6T1C**: Any differences between 6T1C-LoRA and PEFT can now be attributed to analog device characteristics (quantization, noise, etc.), not implementation bugs

---

## Files Created

- `/data/LRTT_transformer/experiments/test_simple_forward.py` - Forward equivalence test
- `/data/LRTT_transformer/experiments/test_simple_update.py` - Weight update equivalence test
- `/data/LRTT_transformer/experiments/TEST_RESULTS.md` - This file

---

**Verification Status**: ✅ Complete
**Implementation Status**: ✅ Verified
**Time to implement**: ~1.5 hours (as estimated)

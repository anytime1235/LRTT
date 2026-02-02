# Understanding Spatial LRTT

## 1. Standard LRTT (Original)

### Weight Dimensions
- Conv weight: `W[Cout, Cin, k, k]`
- Flatten spatial: `W_flat[Cout, Cin*k*k]`
- Decompose: `W_flat ≈ A[Cout, rank] @ B[rank, Cin*k*k]`

### Parameters
- A: `Cout × rank`
- B: `rank × Cin×k×k`
- Total: `rank × (Cout + Cin×k×k)`

### Example (Cout=64, Cin=64, k=3, rank=8)
- A: 64 × 8 = 512 params
- B: 8 × 576 = 4,608 params
- **Total: 5,120 params**

---

## 2. Spatial LRTT (LoRA-C)

### Key Idea
Instead of treating spatial dimensions as part of input channels, we decompose spatially!

### Weight Dimensions
- Conv weight: `W[Cout, Cin, k, k]`
- Reshape spatially: `W_spatial[Cout×k, Cin×k]`
- Decompose: `W_spatial ≈ A[Cout×k, rank×k] @ B[rank×k, Cin×k]`

### Parameters
- A: `Cout×k × rank×k`
- B: `rank×k × Cin×k`
- Total: `rank×k × (Cout×k + Cin×k) = rank × k² × (Cout + Cin)`

### Example (Cout=64, Cin=64, k=3, rank=8)
- A: 192 × 24 = 4,608 params
- B: 24 × 192 = 4,608 params
- **Total: 9,216 params**

### Parameter Ratio
- Spatial / Standard = `[k² × (Cout + Cin)] / [Cout + Cin×k×k]`
- For k=3: `9×128 / (64 + 576) = 1152/640 ≈ 1.8x`
- More parameters BUT better spatial structure!

---

## 3. The Critical Reshape Question

**Given standard conv weight `W[Cout, Cin, k, k]`, how do we reshape to `W_spatial[Cout×k, Cin×k]`?**

### Option 1: Current Implementation
```python
W.permute(0, 2, 1, 3)  # [Cout, k, Cin, k]
  .reshape(Cout*k, Cin*k)
```

This groups: `[Cout, k_height] × [Cin, k_width]`

### What does this mean?
For a 3×3 kernel:
```
Original W[i, j, :, :]:  (for output i, input j)
[[w00, w01, w02],
 [w10, w11, w12],
 [w20, w21, w22]]
```

After permute(0,2,1,3) and reshape, the C matrix structure is:
```
C[Cout×k, Cin×k] where:
  - Rows: (out_ch_0, row_0), (out_ch_0, row_1), (out_ch_0, row_2), (out_ch_1, row_0), ...
  - Cols: (in_ch_0, col_0), (in_ch_0, col_1), (in_ch_0, col_2), (in_ch_1, col_0), ...
```

---

## 4. The Question: Is This Reshape Correct?

To verify, we need to check:
**Does `y = Conv(x, W)` equal `y = SpatialLRTT(x, C)` where `C` comes from reshaping `W`?**

This is what we need to test!

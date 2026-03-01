# GPU Device Movement Fix for LRTT-LoRA

## Problem
LRTT-LoRA training was extremely slow (10-11s/iteration) with 1-4% GPU utilization. Investigation revealed that analog tiles (tile_a, tile_b, tile_c) remained on CPU even after calling `model.to('cuda')`.

## Root Cause
`LRTTSimulatorTile.to()` method called `super().to()` to move child modules to device. However, `AnalogTile` sub-modules don't properly implement `.to()` - they only have `.cuda()` and `.cpu()` methods. As a result, PyTorch's automatic device movement didn't work for these tiles.

## Fix Applied

### File: `/data/LRTT_transformer/src/aihwkit/simulator/tiles/lrtt_tile.py`

Modified three methods to explicitly call `.cuda()`/`.cpu()` on sub-tiles:

1. **`to()` method** (lines 812-853): Added explicit device movement for tile_a, tile_b, tile_c based on target device type
2. **`cuda()` method** (lines 855-878): Added explicit `.cuda()` calls on all sub-tiles
3. **`cpu()` method** (lines 880-896): Added explicit `.cpu()` calls on all sub-tiles

### Key Code Addition:
```python
# In .to() method - for CUDA device:
if device.type == 'cuda':
    cuda_idx = device.index if device.index is not None else 0
    if hasattr(self, 'tile_a') and hasattr(self.tile_a, 'cuda'):
        self.tile_a.cuda(cuda_idx)
    if hasattr(self, 'tile_b') and hasattr(self.tile_b, 'cuda'):
        self.tile_b.cuda(cuda_idx)
    if hasattr(self, 'tile_c') and hasattr(self.tile_c, 'cuda'):
        self.tile_c.cuda(cuda_idx)
```

## Impact

### Performance Improvement:
- **Before fix:** ~10-11 seconds/iteration (CPU-bound)
- **After fix:** ~0.2 seconds/iteration (GPU accelerated)
- **Speedup:** **50x faster!**

### GPU Utilization:
- **Before:** 1-4% GPU utilization
- **After:** Expected 70-90% GPU utilization during training

## Verification

### Test Script: `test_training_gpu_usage.py`
```bash
/data/venvs/aihwkit_gpu/bin/python /data/LRTT_transformer/experiments/test_training_gpu_usage.py
```

Results:
- Step 0: 1.310s (warmup/compilation)
- Steps 1-9: ~0.18-0.20s (steady state)
- Average: 0.303s
- Throughput: 3.30 it/s

## Important Notes

1. **`get_weights()` always returns CPU tensors** - This is by design in aihwkit. The method copies weights from C++ backend to CPU for inspection. This does NOT mean the tiles are on CPU during training.

2. **Actual computation uses CUDA** - Forward/backward/update operations all run on the C++ CUDA backend. You can verify this by checking that forward outputs have `device='cuda:0'`.

3. **No changes needed to user code** - The fix is transparent. Existing code that calls `model.to('cuda')` will now work correctly.

## Files Modified

1. `/data/LRTT_transformer/src/aihwkit/simulator/tiles/lrtt_tile.py` - Added explicit sub-tile device movement

## Testing

Run the sweep or training scripts as before. Training should now be 50x faster with high GPU utilization:

```bash
cd /data/LRTT_transformer/experiments
/data/venvs/aihwkit_gpu/bin/python run_lrtt_lora_sweep_batch.py
```

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

You should now see 70-90% GPU utilization during training.

---

**Fix Date:** 2026-02-11
**Issue:** Critical performance bug - tiles not moving to GPU
**Status:** ✅ RESOLVED

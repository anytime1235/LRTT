#!/usr/bin/env python
"""Test for memory leak in LRTT tiles."""
import torch
import gc

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

import sys
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

from transformers import AutoModelForQuestionAnswering, set_seed

DEVICE = torch.device("cuda")

def get_memory():
    """Get GPU memory in GB."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return allocated, reserved

def create_simple_model():
    """Create a minimal analog model (just 1 layer for testing)."""
    import torch.nn as nn

    # Simple model with just 1 linear layer
    model = nn.Sequential(
        nn.Linear(512, 512)
    )

    ab_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
    )
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
    )

    device_config = PythonLRTTDevice(
        rank=8,
        transfer_every=1000000,
        lora_alpha=1.0,
        reinit_gain=0.1,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.forward_inject = True

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    model = convert_to_analog(model, rpu_config)
    return model.to(DEVICE)

def test_single_layer_memory():
    """Test memory with just 1 analog layer."""
    print("="*60)
    print("Testing memory leak with single analog layer")
    print("="*60)

    gc.collect()
    torch.cuda.empty_cache()

    alloc, res = get_memory()
    print(f"Initial: allocated={alloc:.3f}GB, reserved={res:.3f}GB")

    model = create_simple_model()
    optimizer = AnalogAdam(model.parameters(), lr=1e-4)

    alloc, res = get_memory()
    print(f"After model: allocated={alloc:.3f}GB, reserved={res:.3f}GB")

    # Dummy input
    batch_size = 32
    x = torch.randn(batch_size, 512, device=DEVICE)
    target = torch.randn(batch_size, 512, device=DEVICE)

    for step in range(10):
        optimizer.zero_grad()

        y = model(x)
        loss = ((y - target) ** 2).mean()

        loss.backward()
        optimizer.step()

        alloc, res = get_memory()
        print(f"Step {step+1}: loss={loss.item():.4f}, allocated={alloc:.3f}GB, reserved={res:.3f}GB")

        if alloc > 10:
            print("WARNING: Memory exceeded 10GB!")
            break

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    alloc, res = get_memory()
    print(f"After cleanup: allocated={alloc:.3f}GB, reserved={res:.3f}GB")

def test_mobilebert_memory():
    """Test memory with MobileBERT (V only)."""
    print("\n" + "="*60)
    print("Testing memory leak with MobileBERT (V only)")
    print("="*60)

    gc.collect()
    torch.cuda.empty_cache()

    alloc, res = get_memory()
    print(f"Initial: allocated={alloc:.3f}GB, reserved={res:.3f}GB")

    # Load MobileBERT
    set_seed(42)
    model = AutoModelForQuestionAnswering.from_pretrained("google/mobilebert-uncased")

    # List V layers only
    exclude = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "value" not in name:
                exclude.append(name)

    ab_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
    )
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
    )

    device_config = PythonLRTTDevice(
        rank=8,
        transfer_every=1000000,
        lora_alpha=1.0,
        reinit_gain=0.1,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.forward_inject = True

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)
    model = model.to(DEVICE)

    optimizer = AnalogAdam(model.parameters(), lr=1e-4)

    alloc, res = get_memory()
    print(f"After model: allocated={alloc:.3f}GB, reserved={res:.3f}GB")

    # Count analog layers
    from aihwkit.nn.modules.linear import AnalogLinear
    n_analog = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"Analog layers: {n_analog}")

    # Smaller batch for testing
    batch_size = 8  # Reduced from 32
    seq_len = 128   # Reduced from 384
    input_ids = torch.randint(0, 30000, (batch_size, seq_len), device=DEVICE)
    attention_mask = torch.ones(batch_size, seq_len, device=DEVICE)
    start_positions = torch.randint(0, seq_len, (batch_size,), device=DEVICE)
    end_positions = torch.randint(0, seq_len, (batch_size,), device=DEVICE)

    # Check analog contexts on sub-tiles
    from aihwkit.nn.modules.linear import AnalogLinear
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear):
            tile = next(m.analog_tiles())
            if isinstance(tile, LRTTSimulatorTile):
                print(f"\n{name}: LRTTSimulatorTile found")
                print(f"  tile_a has analog_ctx: {hasattr(tile.tile_a, 'analog_ctx')}")
                print(f"  tile_b has analog_ctx: {hasattr(tile.tile_b, 'analog_ctx')}")
                print(f"  tile_c has analog_ctx: {hasattr(tile.tile_c, 'analog_ctx')}")
                break

    for step in range(5):
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        alloc, res = get_memory()
        print(f"Step {step+1}: loss={loss.item():.4f}, allocated={alloc:.3f}GB, reserved={res:.3f}GB")

        # Check analog_input list sizes
        if step == 0:
            for name, m in model.named_modules():
                if isinstance(m, AnalogLinear):
                    tile = next(m.analog_tiles())
                    if isinstance(tile, LRTTSimulatorTile):
                        a_ctx = getattr(tile.tile_a, 'analog_ctx', None)
                        b_ctx = getattr(tile.tile_b, 'analog_ctx', None)
                        c_ctx = getattr(tile.tile_c, 'analog_ctx', None)
                        print(f"\n  After step 1:")
                        print(f"    tile_a analog_input len: {len(a_ctx.analog_input) if a_ctx else 'N/A'}")
                        print(f"    tile_b analog_input len: {len(b_ctx.analog_input) if b_ctx else 'N/A'}")
                        print(f"    tile_c analog_input len: {len(c_ctx.analog_input) if c_ctx else 'N/A'}")
                        break

        if alloc > 50:
            print("WARNING: Memory exceeded 50GB!")
            break

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    alloc, res = get_memory()
    print(f"After cleanup: allocated={alloc:.3f}GB, reserved={res:.3f}GB")

if __name__ == "__main__":
    test_single_layer_memory()
    test_mobilebert_memory()

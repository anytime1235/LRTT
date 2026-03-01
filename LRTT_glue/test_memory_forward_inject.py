#!/usr/bin/env python
"""Test memory usage with forward_inject=True vs False."""
import torch
import gc
from transformers import AutoModelForQuestionAnswering, set_seed

# aihwkit imports (use installed package first)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT config imports
import sys
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
DEVICE = torch.device("cuda")

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**3  # GB

def create_model(forward_inject: bool):
    """Create analog model with or without forward_inject."""
    set_seed(42)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # A/B tiles: LinearStepDevice (same as sweep script)
    ab_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=-0.1678,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    # C tile: SoftBoundsDevice
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        w_max_dtod=0,
        w_min_dtod=0,
        mult_noise=True,
    )

    # Sixt1c-LoRA Device config
    device_config = PythonLRTTDevice(
        rank=8,
        transfer_every=1000000,  # No transfer
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.units_in_mbatch = True
    device_config.forward_inject = forward_inject  # KEY PARAMETER
    device_config.transfer_method = "onehot"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    model = convert_to_analog(model, rpu_config, exclude_modules=["qa_outputs"])
    return model.to(DEVICE)

def test_training_steps(forward_inject: bool, n_steps: int = 20):
    """Run training steps and monitor memory."""
    print(f"\n{'='*60}")
    print(f"Testing forward_inject={forward_inject}")
    print(f"{'='*60}")

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Initial GPU memory: {get_gpu_memory():.2f} GB")

    model = create_model(forward_inject)
    print(f"After model creation: {get_gpu_memory():.2f} GB")

    optimizer = AnalogAdam(model.parameters(), lr=1e-4)

    # Create dummy batch
    batch_size = 32
    seq_len = 384
    input_ids = torch.randint(0, 30000, (batch_size, seq_len), device=DEVICE)
    attention_mask = torch.ones(batch_size, seq_len, device=DEVICE)
    start_positions = torch.randint(0, seq_len, (batch_size,), device=DEVICE)
    end_positions = torch.randint(0, seq_len, (batch_size,), device=DEVICE)

    print(f"After creating batch: {get_gpu_memory():.2f} GB")

    for step in range(n_steps):
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = outputs.loss

        if step == 0:
            print(f"Initial loss: {loss.item():.2f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mem = get_gpu_memory()
        print(f"Step {step+1}: loss={loss.item():.4f}, GPU mem={mem:.2f} GB")

        if mem > 100:
            print("WARNING: Memory exceeded 100GB, stopping!")
            break

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"After cleanup: {get_gpu_memory():.2f} GB")

if __name__ == "__main__":
    # Test without forward_inject first (should be stable)
    test_training_steps(forward_inject=False, n_steps=10)

    # Then test with forward_inject
    test_training_steps(forward_inject=True, n_steps=10)

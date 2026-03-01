#!/usr/bin/env python
"""Test analog conversion of LoRA layers only (not base layer).

This tests if OOM occurs when only converting lora_A/lora_B to analog
while keeping base_layer digital.
"""
import torch
import gc
import sys

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

from transformers import AutoModelForQuestionAnswering, set_seed
from peft import LoraConfig, get_peft_model

# Add lora_training to path for related_functions
sys.path.insert(0, '/data/LRTT_transformer/lora_training')
from related_functions import convert_lora_layers_only_to_analog

DEVICE = torch.device("cuda")

def get_mem():
    return torch.cuda.memory_allocated() / 1024**3

def create_lora_model():
    """Create MobileBERT with PEFT LoRA adapters."""
    model = AutoModelForQuestionAnswering.from_pretrained("google/mobilebert-uncased")

    # PEFT LoRA config - target V layers only
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["value"],  # V only
        bias="none",
        task_type="QUESTION_ANS"
    )

    model = get_peft_model(model, peft_config)
    return model

def test_lora_only_analog():
    """Test memory with only LoRA layers converted to analog."""
    print("=" * 60)
    print("Test: Convert only lora_A/lora_B to analog (base_layer stays digital)")
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Initial memory: {get_mem():.3f} GB")

    # Create model with PEFT LoRA
    set_seed(42)
    model = create_lora_model()
    print(f"After PEFT LoRA model creation: {get_mem():.3f} GB")

    # Count LoRA layers
    lora_count = 0
    for name, m in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(m, torch.nn.Linear):
            lora_count += 1
    print(f"Found {lora_count} digital LoRA layers")

    # Create RPU config for LoRA layers (similar to LRTT's A/B tiles)
    rpu_config = SingleRPUConfig(
        device=LinearStepDevice(
            dw_min=0.001981,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
        )
    )

    # Convert only LoRA layers to analog
    print("\nConverting LoRA layers to analog...")
    model = convert_lora_layers_only_to_analog(model, rpu_config)

    # Count analog LoRA layers
    analog_lora_count = 0
    for name, m in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(m, AnalogLinear):
            analog_lora_count += 1
    print(f"Analog LoRA layers: {analog_lora_count}")

    model = model.to(DEVICE)
    print(f"After moving to CUDA: {get_mem():.3f} GB")

    # Create optimizer
    optimizer = AnalogAdam(model.parameters(), lr=1e-4)

    # Create dummy batch
    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_len), device=DEVICE)
    attention_mask = torch.ones(batch_size, seq_len, device=DEVICE)
    start_positions = torch.randint(0, seq_len, (batch_size,), device=DEVICE)
    end_positions = torch.randint(0, seq_len, (batch_size,), device=DEVICE)

    print(f"\nStarting training loop...")

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

        gc.collect()
        torch.cuda.empty_cache()

        mem = get_mem()
        print(f"Step {step+1}: loss={loss.item():.4f}, memory={mem:.3f} GB")

        if mem > 50:
            print("WARNING: Memory exceeded 50GB!")
            break

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nAfter cleanup: {get_mem():.3f} GB")

if __name__ == "__main__":
    test_lora_only_analog()

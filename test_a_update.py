#!/home/jovyan/work/ml/.venv310/bin/python
"""Quick test to verify A matrix is being updated in Digital Merge."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
from aihwkit.optim import AnalogAdam
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, set_seed

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice, SoftBoundsDevice


def create_config() -> PythonLRTTRPUConfig:
    """Create LRTT config with Digital A,B + Analog C."""
    ab_device = FloatingPointDevice()
    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, write_noise_std=0.0, mult_noise=True,
    )

    device_config = PythonLRTTDevice(
        rank=8,
        transfer_every=63,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.00531
    device_config.forward_inject = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    return PythonLRTTRPUConfig(device=device_config)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create a simple model
    model_config = AutoConfig.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased", config=model_config
    )

    # Convert only first query layer
    all_linear = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_linear.append(name)

    exclude = [name for name in all_linear if "layer.0.attention.self.query" not in name]
    exclude.append("classifier")

    rpu_config = create_config()
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    # Freeze all except target
    for name, param in model.named_parameters():
        if "layer.0.attention.self.query" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.to(device)
    model.train()

    # Find the analog layer
    analog_layer = None
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'tile_a'):
            analog_layer = module
            layer_name = name
            break

    if analog_layer is None:
        print("ERROR: Could not find analog layer")
        return

    tile = analog_layer.analog_module
    print(f"Found layer: {layer_name}")
    print(f"  tile_a shape: {tile.tile_a.get_weights()[0].shape}")
    print(f"  tile_b shape: {tile.tile_b.get_weights()[0].shape}")
    print(f"  tile_c shape: {tile.tile_c.get_weights()[0].shape}")

    # Check initial A
    A_init = tile.tile_a.get_weights()[0]
    print(f"\nInitial A norm: {A_init.norm().item():.6f}")
    print(f"Initial A max: {A_init.abs().max().item():.6f}")

    # Check if _orig_update exists
    print(f"\ntile_a has _orig_update: {hasattr(tile.tile_a, '_orig_update')}")
    print(f"tile_b has _orig_update: {hasattr(tile.tile_b, '_orig_update')}")
    print(f"tile_c has _orig_update: {hasattr(tile.tile_c, '_orig_update')}")

    # Create optimizer and do one training step
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    optimizer = AnalogAdam(model.parameters(), lr=0.001)  # High LR to see effect

    # Create dummy input
    text = "This is a positive review. I love it!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs['labels'] = torch.tensor([1], device=device)

    print("\n--- Training step 1 ---")
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = nn.CrossEntropyLoss()(outputs.logits, inputs['labels'])
    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    optimizer.step()

    # Reset update flag (important for LRTT)
    tile._reset_update_flag()

    # Check A after one step
    A_after = tile.tile_a.get_weights()[0]
    print(f"\nA norm after step 1: {A_after.norm().item():.6f}")
    print(f"A max after step 1: {A_after.abs().max().item():.6f}")
    print(f"A changed: {not torch.allclose(A_init, A_after)}")

    # Check controller state
    print(f"\nController state:")
    print(f"  num_a_updates: {tile.controller.num_a_updates}")
    print(f"  num_b_updates: {tile.controller.num_b_updates}")
    print(f"  transfer_counter: {tile.controller.transfer_counter}")

    # Do a few more steps
    print("\n--- Training steps 2-5 ---")
    for step in range(4):
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss()(outputs.logits, inputs['labels'])
        loss.backward()
        optimizer.step()
        tile._reset_update_flag()

    A_after_5 = tile.tile_a.get_weights()[0]
    print(f"A norm after 5 steps: {A_after_5.norm().item():.6f}")
    print(f"A max after 5 steps: {A_after_5.abs().max().item():.6f}")
    print(f"  num_a_updates: {tile.controller.num_a_updates}")
    print(f"  transfer_counter: {tile.controller.transfer_counter}")


if __name__ == "__main__":
    main()

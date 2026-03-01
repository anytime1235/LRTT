"""Diagnostic: Track A/B tile weight changes in sixt1c_lora mode."""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
import math
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.devices import (
    LinearStepDevice,
    SoftBoundsDevice,
)
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.optim import AnalogAdam

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
TASK = "rte"
NUM_STEPS = 10


def create_sixt1c_lora_config(rank, lora_alpha, reinit_gain=0.1):
    TAU_SEC = 46505.0
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mean_bound_reference=True,
        lifetime=lifetime, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
    )
    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )
    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=1000000,
        lora_alpha=lora_alpha, reinit_gain=reinit_gain,
        reinit_mode="hybrid", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.units_in_mbatch = True
    device_config.forward_inject = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def get_ab_stats(model):
    """Get A/B tile weight stats for all analog layers."""
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'tile_a'):
            tile = module.analog_module
            a_w = tile.tile_a.get_weights()[0]
            b_w = tile.tile_b.get_weights()[0]
            c_w = tile.tile_c.get_weights()[0]
            stats[name] = {
                'a_norm': a_w.norm().item(),
                'a_mean': a_w.mean().item(),
                'a_std': a_w.std().item(),
                'a_max': a_w.abs().max().item(),
                'b_norm': b_w.norm().item(),
                'b_mean': b_w.mean().item(),
                'b_std': b_w.std().item(),
                'b_max': b_w.abs().max().item(),
                'c_norm': c_w.norm().item(),
                'c_max': c_w.abs().max().item(),
                'ab_product_norm': (a_w @ b_w).norm().item(),
            }
    return stats


def get_controller_stats(model):
    """Get controller update counters."""
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            ctrl = module.analog_module.controller
            stats[name] = {
                'num_a_updates': ctrl.num_a_updates,
                'num_b_updates': ctrl.num_b_updates,
                'transfer_counter': ctrl.transfer_counter,
                'lora_alpha': ctrl.lora_alpha,
                'lr_eff_info': f"lr * alpha = lr * {ctrl.lora_alpha}",
            }
    return stats


def run_diagnostic(lora_alpha):
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: lora_alpha = {lora_alpha}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Convert only query layer for speed
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [n for n in all_linear if 'query' not in n]
    exclude.append('classifier')

    rpu_config = create_sixt1c_lora_config(RANK, lora_alpha)
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        if 'query' in name and 'bias' not in name:
            param.requires_grad = True
        elif 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.to(device)

    # Load small amount of data
    raw = load_dataset("glue", "rte", split="train[:64]")
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                        truncation=True, padding="max_length", max_length=128)
    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    loader = DataLoader(tokenized, batch_size=16, shuffle=False, collate_fn=default_data_collator)

    optimizer = AnalogAdam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Get initial stats
    print("\n--- INITIAL STATE ---")
    init_stats = get_ab_stats(model)
    first_layer = list(init_stats.keys())[0]
    print(f"Layer: {first_layer}")
    for k, v in init_stats[first_layer].items():
        print(f"  {k}: {v:.6f}")

    # Store initial weights for comparison
    init_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'tile_a'):
            tile = module.analog_module
            init_weights[name] = {
                'a': tile.tile_a.get_weights()[0].clone(),
                'b': tile.tile_b.get_weights()[0].clone(),
                'c': tile.tile_c.get_weights()[0].clone(),
            }

    # Train for a few steps
    model.train()
    step = 0
    for batch in loader:
        if step >= NUM_STEPS:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()

        # Check gradient magnitudes before optimizer step
        if step == 0:
            print(f"\n--- GRADIENT MAGNITUDES (step 0) ---")
            for name, param in model.named_parameters():
                if 'query' in name and param.grad is not None:
                    print(f"  {name}: grad_norm={param.grad.norm().item():.6f}, "
                          f"grad_max={param.grad.abs().max().item():.6f}")
                    if 'layer.0' in name:
                        break  # Just show first layer

        optimizer.step()
        step += 1

        if step in [1, 5, NUM_STEPS]:
            print(f"\n--- AFTER STEP {step} (loss={loss.item():.4f}) ---")
            cur_stats = get_ab_stats(model)
            for k, v in cur_stats[first_layer].items():
                print(f"  {k}: {v:.6f}")

            # Weight change
            tile = None
            for n, m in model.named_modules():
                if n == first_layer and hasattr(m, 'analog_module'):
                    tile = m.analog_module
                    break
            if tile and first_layer in init_weights:
                a_diff = (tile.tile_a.get_weights()[0] - init_weights[first_layer]['a']).norm()
                b_diff = (tile.tile_b.get_weights()[0] - init_weights[first_layer]['b']).norm()
                c_diff = (tile.tile_c.get_weights()[0] - init_weights[first_layer]['c']).norm()
                print(f"  >> A weight change norm: {a_diff.item():.8f}")
                print(f"  >> B weight change norm: {b_diff.item():.8f}")
                print(f"  >> C weight change norm: {c_diff.item():.8f}")

    # Controller stats
    print(f"\n--- CONTROLLER STATS (after {step} steps) ---")
    ctrl_stats = get_controller_stats(model)
    for k, v in ctrl_stats[first_layer].items():
        print(f"  {k}: {v}")

    # Compute expected lr_eff
    lr = 0.001
    lr_eff = lr * lora_alpha
    expected_pulses = lr_eff / 0.001981
    print(f"\n--- PULSE ANALYSIS ---")
    print(f"  lr = {lr}")
    print(f"  lora_alpha = {lora_alpha}")
    print(f"  lr_eff = lr * alpha = {lr_eff:.8f}")
    print(f"  dw_min = 0.001981")
    print(f"  BL (default) = 31")
    print(f"  A = B = sqrt(lr_eff / (dw_min * BL)) = {math.sqrt(lr_eff / (0.001981 * 31)):.8f}")
    print(f"  For |x|=|d|=1: expected pulse coincidences ~ {expected_pulses:.4f}")
    print(f"  => {'VERY FEW/ZERO pulses!' if expected_pulses < 0.1 else 'Some pulses possible'}")


if __name__ == "__main__":
    for alpha in [0.01, 0.1, 1.0, 5.0]:
        run_diagnostic(alpha)

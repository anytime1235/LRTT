"""Diagnostic: Track C vs alpha*AB forward output scales across lora_alpha values."""
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
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.optim import AnalogAdam

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
NUM_STEPS = 5


def create_sixt1c_lora_config(rank, lora_alpha, reinit_gain=0.1):
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-1.0 / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=3.0, w_min=-3.0,
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


def hook_forward_components(model):
    """Install hooks to capture C and AB forward outputs separately."""
    records = {}

    def make_hook(layer_name):
        def hook_fn(module, args, output):
            tile = module
            if not hasattr(tile, 'tile_a'):
                return
            # Get the input that was used
            # We need to manually compute C*x and A*(B*x) separately
            x = tile._last_x_input if hasattr(tile, '_last_x_input') else None
            if x is None:
                return
            with torch.no_grad():
                x_bf = x  # assume batch-first
                y_c = tile.tile_c.forward(x_bf)
                y_b = tile.tile_b.forward(x_bf)
                y_ab = tile.tile_a.forward(y_b)
                alpha = tile.controller.lora_alpha

                records[layer_name] = {
                    'y_c_norm': y_c.norm().item(),
                    'y_c_mean': y_c.mean().item(),
                    'y_c_std': y_c.std().item(),
                    'y_c_max': y_c.abs().max().item(),
                    'y_ab_norm': y_ab.norm().item(),
                    'y_ab_mean': y_ab.mean().item(),
                    'y_ab_std': y_ab.std().item(),
                    'y_ab_max': y_ab.abs().max().item(),
                    'alpha_y_ab_norm': (alpha * y_ab).norm().item(),
                    'alpha_y_ab_max': (alpha * y_ab).abs().max().item(),
                    'ratio_ab_over_c': (alpha * y_ab).norm().item() / max(y_c.norm().item(), 1e-10),
                    'total_norm': (y_c + alpha * y_ab).norm().item(),
                }
        return hook_fn

    # We can't easily hook the internal forward_inject, so let's do it manually
    return records


def run_diagnostic(lora_alpha):
    print(f"\n{'='*70}")
    print(f"FORWARD SCALE DIAGNOSTIC: lora_alpha = {lora_alpha}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    # Load data
    raw = load_dataset("glue", "rte", split="train[:32]")
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                        truncation=True, padding="max_length", max_length=128)
    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    loader = DataLoader(tokenized, batch_size=16, shuffle=False, collate_fn=default_data_collator)

    optimizer = AnalogAdam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Storage for hook captures
    hook_data = {}

    def install_hooks(model):
        """Monkey-patch controller._forward_inject_analog_unified to capture C and AB."""
        handles = []
        for name, module in model.named_modules():
            if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
                analog = module.analog_module
                ctrl = analog.controller
                layer_name = name
                orig_fn = ctrl._forward_inject_analog_unified

                def make_patched(ctrl, orig, lname):
                    def patched(x, out_trans=False, in_trans=False):
                        x_bf = x.t() if in_trans else x
                        with torch.no_grad():
                            y_c = ctrl.tile_c.forward(x_bf)
                            y_b = ctrl.tile_b.forward(x_bf)
                            y_ab = ctrl.tile_a.forward(y_b)
                        hook_data[lname] = {
                            'y_c': y_c.detach(), 'y_ab': y_ab.detach(),
                            'x_input': x_bf.detach(),
                        }
                        return orig(x, out_trans, in_trans)
                    return patched

                ctrl._forward_inject_analog_unified = make_patched(ctrl, orig_fn, layer_name)
        return handles

    install_hooks(model)

    def measure_forward_scales(model, batch, step_label):
        """Run forward pass and report captured C vs AB scales."""
        model.eval()
        hook_data.clear()
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

        # Report layer 0 query
        target = None
        for k in hook_data:
            if 'layer.0' in k and 'query' in k:
                target = k
                break
        if target is None:
            # Use first available
            target = list(hook_data.keys())[0] if hook_data else None

        if target:
            d = hook_data[target]
            y_c = d['y_c']
            y_ab = d['y_ab']
            alpha = lora_alpha

            # Get analog module for weight info
            analog = None
            for n, m in model.named_modules():
                if n == target and hasattr(m, 'analog_module'):
                    analog = m.analog_module
                    break

            print(f"\n  [{step_label}] {target}")
            print(f"    x input:      norm={d['x_input'].norm():.4f}  std={d['x_input'].std():.4f}")
            print(f"    C output:     norm={y_c.norm():.4f}  mean={y_c.mean():.6f}  std={y_c.std():.4f}  max={y_c.abs().max():.4f}")
            print(f"    AB output:    norm={y_ab.norm():.4f}  mean={y_ab.mean():.6f}  std={y_ab.std():.4f}  max={y_ab.abs().max():.4f}")
            print(f"    α*AB output:  norm={(alpha*y_ab).norm():.4f}  max={(alpha*y_ab).abs().max():.4f}")
            print(f"    C+α*AB:       norm={(y_c + alpha*y_ab).norm():.4f}")
            print(f"    Ratio |α*AB|/|C|: {(alpha*y_ab).norm().item() / max(y_c.norm().item(), 1e-10):.6f}")

            if analog:
                a_w = analog.tile_a.get_weights()[0]
                b_w = analog.tile_b.get_weights()[0]
                c_w = analog.tile_c.get_weights()[0]
                out_scale = analog.tile_c.out_scaling_alpha.data if hasattr(analog.tile_c, 'out_scaling_alpha') else None
                print(f"    Weights: A_norm={a_w.norm():.4f}  B_norm={b_w.norm():.4f}  C_norm={c_w.norm():.4f}  AB_norm={(a_w@b_w).norm():.4f}")
                if out_scale is not None:
                    print(f"    out_scaling:   mean={out_scale.mean():.4f}  std={out_scale.std():.4f}  max={out_scale.abs().max():.4f}")

        model.train()

    # Get first batch
    first_batch = next(iter(loader))

    # Measure BEFORE training
    measure_forward_scales(model, first_batch, "INIT (step 0)")

    # Train and measure
    model.train()
    for step, batch in enumerate(loader):
        if step >= NUM_STEPS:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step in [0, 1, 4]:
            measure_forward_scales(model, first_batch, f"After step {step+1}")


if __name__ == "__main__":
    for alpha in [0.01, 0.1, 1.0, 5.0, 10.0]:
        run_diagnostic(alpha)

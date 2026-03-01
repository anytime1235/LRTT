"""
Complete verification of sixt1c configuration in sweep_sixt1c_lora_glue_adam.py
Checks:
1. Bias freeze status
2. Layer freeze status
3. Trainable parameters
4. Learn out scaling
5. Pretrained weight loading
6. Sixt1c device config (I/O, bounds, noise management)
"""

import sys
import torch
import torch.nn as nn
from collections import defaultdict

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig


def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def check_bias_status(model):
    """Check bias parameters: frozen or trainable."""
    print_section("1. BIAS STATUS")

    bias_params = {}
    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params[name] = {
                'requires_grad': param.requires_grad,
                'shape': param.shape,
                'is_zero': torch.allclose(param, torch.zeros_like(param)),
                'mean': param.mean().item(),
                'std': param.std().item(),
            }

    print(f"\nTotal bias parameters: {len(bias_params)}")

    frozen_bias = [n for n, p in bias_params.items() if not p['requires_grad']]
    trainable_bias = [n for n, p in bias_params.items() if p['requires_grad']]

    print(f"\nFrozen bias: {len(frozen_bias)}")
    print(f"Trainable bias: {len(trainable_bias)}")

    if frozen_bias:
        print(f"\n  Sample frozen bias (first 5):")
        for name in frozen_bias[:5]:
            info = bias_params[name]
            print(f"    {name}")
            print(f"      requires_grad={info['requires_grad']}, shape={info['shape']}")
            print(f"      mean={info['mean']:.6f}, std={info['std']:.6f}")

    if trainable_bias:
        print(f"\n  Sample trainable bias (first 5):")
        for name in trainable_bias[:5]:
            info = bias_params[name]
            print(f"    {name}")
            print(f"      requires_grad={info['requires_grad']}, shape={info['shape']}")
            print(f"      mean={info['mean']:.6f}, std={info['std']:.6f}")

    return bias_params


def check_layer_freeze_status(model):
    """Check which layers are frozen and which are trainable."""
    print_section("2. LAYER FREEZE STATUS")

    layer_status = defaultdict(lambda: {'frozen': [], 'trainable': []})

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, AnalogLinear)):
            # Check if any parameter is trainable
            is_trainable = False
            has_params = False

            for param_name, param in module.named_parameters(recurse=False):
                has_params = True
                if param.requires_grad:
                    is_trainable = True
                    break

            if not has_params:
                continue

            layer_type = None
            if isinstance(module, AnalogLinear):
                if 'lora_A' in name:
                    layer_type = 'lora_A (analog)'
                elif 'lora_B' in name:
                    layer_type = 'lora_B (analog)'
                elif 'base_layer' in name:
                    layer_type = 'base_layer (analog)'
                else:
                    layer_type = 'other_analog'
            else:
                if 'lora' in name:
                    layer_type = 'lora (digital)'
                elif 'classifier' in name or 'qa_outputs' in name:
                    layer_type = 'classifier (digital)'
                else:
                    layer_type = 'other (digital)'

            if is_trainable:
                layer_status[layer_type]['trainable'].append(name)
            else:
                layer_status[layer_type]['frozen'].append(name)

    print("\nLayer freeze status by type:")
    for layer_type in sorted(layer_status.keys()):
        frozen_count = len(layer_status[layer_type]['frozen'])
        trainable_count = len(layer_status[layer_type]['trainable'])
        total = frozen_count + trainable_count

        print(f"\n  {layer_type}:")
        print(f"    Total: {total}")
        print(f"    Frozen: {frozen_count}")
        print(f"    Trainable: {trainable_count}")

        if trainable_count > 0:
            print(f"    Sample trainable (first 2):")
            for name in layer_status[layer_type]['trainable'][:2]:
                print(f"      - {name}")

    return layer_status


def check_trainable_parameters(model):
    """Check all trainable parameters."""
    print_section("3. TRAINABLE PARAMETERS")

    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))

    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(p[1] for p in frozen_params)
    total_all = total_trainable + total_frozen

    print(f"\nTotal parameters: {total_all:,}")
    print(f"Trainable parameters: {total_trainable:,} ({100*total_trainable/total_all:.4f}%)")
    print(f"Frozen parameters: {total_frozen:,} ({100*total_frozen/total_all:.4f}%)")

    # Group by type
    trainable_by_type = defaultdict(list)
    for name, count in trainable_params:
        if 'lora_A' in name and 'analog' not in name.lower():
            trainable_by_type['lora_A (digital)'].append((name, count))
        elif 'lora_B' in name and 'analog' not in name.lower():
            trainable_by_type['lora_B (digital)'].append((name, count))
        elif 'classifier' in name or 'qa_outputs' in name:
            trainable_by_type['classifier'].append((name, count))
        elif 'bias' in name:
            trainable_by_type['bias'].append((name, count))
        else:
            trainable_by_type['other'].append((name, count))

    print("\n  Trainable parameters by type:")
    for ptype in sorted(trainable_by_type.keys()):
        count = sum(c for _, c in trainable_by_type[ptype])
        print(f"    {ptype}: {len(trainable_by_type[ptype])} params, {count:,} elements")
        if len(trainable_by_type[ptype]) <= 3:
            for name, c in trainable_by_type[ptype]:
                print(f"      - {name}: {c:,}")

    return trainable_params, frozen_params


def check_analog_tiles_and_learn_out_scaling(model):
    """Check analog tiles and learn_out_scaling configuration."""
    print_section("4. ANALOG TILES & LEARN_OUT_SCALING")

    analog_layers = []
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            # Get RPU config
            rpu_config = module.analog_tile.rpu_config if hasattr(module, 'analog_tile') else None

            # Check for out_scaling
            has_out_scaling = hasattr(module, 'out_scaling_alpha')
            out_scaling_alpha = module.out_scaling_alpha if has_out_scaling else None

            # Check learn_out_scaling from config
            learn_out_scaling = None
            out_scaling_columnwise = None
            if rpu_config:
                if hasattr(rpu_config, 'forward'):
                    learn_out_scaling = getattr(rpu_config.forward, 'learn_out_scaling', None)
                    out_scaling_columnwise = getattr(rpu_config.forward, 'out_scaling_columnwise', None)

            analog_layers.append({
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'has_bias': module.bias is not None,
                'has_out_scaling': has_out_scaling,
                'out_scaling_alpha': out_scaling_alpha,
                'learn_out_scaling': learn_out_scaling,
                'out_scaling_columnwise': out_scaling_columnwise,
                'rpu_config': rpu_config,
            })

    print(f"\nTotal analog layers: {len(analog_layers)}")

    # Categorize by layer type
    lora_a = [l for l in analog_layers if 'lora_A' in l['name']]
    lora_b = [l for l in analog_layers if 'lora_B' in l['name']]
    base_layer = [l for l in analog_layers if 'base_layer' in l['name']]

    print(f"\n  LoRA A: {len(lora_a)}")
    print(f"  LoRA B: {len(lora_b)}")
    print(f"  Base layer: {len(base_layer)}")

    # Check learn_out_scaling for each type
    if lora_a:
        sample = lora_a[0]
        print(f"\n  Sample LoRA A: {sample['name'].split('.')[-4:]}")
        print(f"    Shape: ({sample['in_features']}, {sample['out_features']})")
        print(f"    has_bias: {sample['has_bias']}")
        print(f"    learn_out_scaling: {sample['learn_out_scaling']}")
        print(f"    out_scaling_columnwise: {sample['out_scaling_columnwise']}")
        print(f"    has_out_scaling_alpha: {sample['has_out_scaling']}")
        if sample['has_out_scaling']:
            print(f"    out_scaling_alpha shape: {sample['out_scaling_alpha'].shape if sample['out_scaling_alpha'] is not None else None}")

    if lora_b:
        sample = lora_b[0]
        print(f"\n  Sample LoRA B: {sample['name'].split('.')[-4:]}")
        print(f"    Shape: ({sample['in_features']}, {sample['out_features']})")
        print(f"    has_bias: {sample['has_bias']}")
        print(f"    learn_out_scaling: {sample['learn_out_scaling']}")
        print(f"    out_scaling_columnwise: {sample['out_scaling_columnwise']}")
        print(f"    has_out_scaling_alpha: {sample['has_out_scaling']}")
        if sample['has_out_scaling']:
            print(f"    out_scaling_alpha shape: {sample['out_scaling_alpha'].shape if sample['out_scaling_alpha'] is not None else None}")

    if base_layer:
        sample = base_layer[0]
        print(f"\n  Sample base_layer: {sample['name'].split('.')[-4:]}")
        print(f"    Shape: ({sample['in_features']}, {sample['out_features']})")
        print(f"    has_bias: {sample['has_bias']}")
        print(f"    learn_out_scaling: {sample['learn_out_scaling']}")
        print(f"    out_scaling_columnwise: {sample['out_scaling_columnwise']}")
        print(f"    has_out_scaling_alpha: {sample['has_out_scaling']}")

    return analog_layers


def check_pretrained_weights(model, task_name):
    """Check if pretrained weights are properly loaded."""
    print_section("5. PRETRAINED WEIGHT VERIFICATION")

    # Load a fresh pretrained model for comparison
    print("\n  Loading fresh pretrained model for comparison...")
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from sweep_sixt1c_lora_glue_adam import MODEL_NAME, TASK_TO_NUM_LABELS

    num_labels = TASK_TO_NUM_LABELS[task_name]
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    fresh_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Compare weights
    print("\n  Comparing weights between pretrained and current model...")

    # Get base model weights (non-LoRA, non-analog)
    comparisons = []

    for name, param in fresh_model.named_parameters():
        # Skip classifier (newly initialized)
        if 'classifier' in name or 'qa_outputs' in name:
            continue

        # Find corresponding parameter in our model
        # Need to navigate through PEFT wrapper: base_model.model.X
        peft_name = f"base_model.model.{name}"

        found = False
        for model_name, model_param in model.named_parameters():
            if model_name == peft_name:
                found = True
                # Compare
                diff = (param.cpu() - model_param.cpu()).abs().max().item()
                comparisons.append({
                    'name': name,
                    'diff': diff,
                    'shape': param.shape,
                })
                break

        if not found:
            # Try to find in analog tiles
            for model_name, module in model.named_modules():
                if isinstance(module, AnalogLinear):
                    if peft_name in model_name or name in model_name:
                        try:
                            weights = module.get_weights()
                            if isinstance(weights, tuple):
                                w = weights[0]
                            else:
                                w = weights

                            if w.shape == param.shape:
                                diff = (param.cpu() - w.cpu()).abs().max().item()
                                comparisons.append({
                                    'name': name,
                                    'diff': diff,
                                    'shape': param.shape,
                                    'analog': True,
                                })
                                found = True
                                break
                        except:
                            pass

    print(f"\n  Compared {len(comparisons)} weight tensors")

    # Check how many are identical (diff < 1e-6)
    identical = [c for c in comparisons if c['diff'] < 1e-6]
    similar = [c for c in comparisons if 1e-6 <= c['diff'] < 1e-3]
    different = [c for c in comparisons if c['diff'] >= 1e-3]

    print(f"\n  Weight comparison results:")
    print(f"    Identical (diff < 1e-6): {len(identical)}/{len(comparisons)}")
    print(f"    Similar (1e-6 <= diff < 1e-3): {len(similar)}/{len(comparisons)}")
    print(f"    Different (diff >= 1e-3): {len(different)}/{len(comparisons)}")

    if different:
        print(f"\n  Layers with different weights (first 5):")
        for c in different[:5]:
            print(f"    {c['name']}: diff={c['diff']:.6f}, shape={c['shape']}")

    # Summary
    if len(identical) > len(comparisons) * 0.8:
        print(f"\n  ✓ Pretrained weights appear to be properly loaded")
    else:
        print(f"\n  ⚠ Many weights differ from pretrained - possible issue")

    return comparisons


def check_sixt1c_device_config(model):
    """Check sixt1c device configuration in detail."""
    print_section("6. SIXT1C DEVICE CONFIGURATION")

    # Find a sixt1c analog layer (lora_A or lora_B)
    sixt1c_layer = None
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and ('lora_A' in name or 'lora_B' in name):
            sixt1c_layer = (name, module)
            break

    if not sixt1c_layer:
        print("  ✗ No sixt1c analog layer found!")
        return

    name, module = sixt1c_layer
    print(f"\n  Analyzing: {name.split('.')[-4:]}")

    # Get RPU config
    if not hasattr(module, 'analog_tile'):
        print("  ✗ No analog_tile found!")
        return

    tile = module.analog_tile
    rpu_config = tile.rpu_config

    print(f"\n  RPU Config Type: {type(rpu_config).__name__}")

    # Check if SingleRPUConfig
    if isinstance(rpu_config, SingleRPUConfig):
        print(f"    ✓ SingleRPUConfig confirmed")

        # Device type
        device = rpu_config.device
        print(f"\n  Device Type: {type(device).__name__}")

        # Check device-specific parameters
        if hasattr(device, 'dw_min'):
            print(f"\n  Device Parameters:")
            print(f"    dw_min: {device.dw_min}")
            print(f"    dw_min_std: {getattr(device, 'dw_min_std', 'N/A')}")
            print(f"    up_down: {getattr(device, 'up_down', 'N/A')}")
            print(f"    up_down_dtod: {getattr(device, 'up_down_dtod', 'N/A')}")

        # Check forward pass I/O management
        if hasattr(rpu_config, 'forward'):
            forward_io = rpu_config.forward
            print(f"\n  Forward Pass I/O Management:")
            print(f"    inp_res: {getattr(forward_io, 'inp_res', 'N/A')}")
            print(f"    inp_bound: {getattr(forward_io, 'inp_bound', 'N/A')}")
            print(f"    inp_noise: {getattr(forward_io, 'inp_noise', 'N/A')}")
            print(f"    out_res: {getattr(forward_io, 'out_res', 'N/A')}")
            print(f"    out_bound: {getattr(forward_io, 'out_bound', 'N/A')}")
            print(f"    out_noise: {getattr(forward_io, 'out_noise', 'N/A')}")
            print(f"    out_scale: {getattr(forward_io, 'out_scale', 'N/A')}")
            print(f"    noise_management: {getattr(forward_io, 'noise_management', 'N/A')}")
            print(f"    bound_management: {getattr(forward_io, 'bound_management', 'N/A')}")
            print(f"    learn_out_scaling: {getattr(forward_io, 'learn_out_scaling', 'N/A')}")
            print(f"    out_scaling_columnwise: {getattr(forward_io, 'out_scaling_columnwise', 'N/A')}")

        # Check backward pass I/O management
        if hasattr(rpu_config, 'backward'):
            backward_io = rpu_config.backward
            print(f"\n  Backward Pass I/O Management:")
            print(f"    inp_res: {getattr(backward_io, 'inp_res', 'N/A')}")
            print(f"    inp_bound: {getattr(backward_io, 'inp_bound', 'N/A')}")
            print(f"    inp_noise: {getattr(backward_io, 'inp_noise', 'N/A')}")
            print(f"    out_res: {getattr(backward_io, 'out_res', 'N/A')}")
            print(f"    out_bound: {getattr(backward_io, 'out_bound', 'N/A')}")
            print(f"    out_noise: {getattr(backward_io, 'out_noise', 'N/A')}")
            print(f"    noise_management: {getattr(backward_io, 'noise_management', 'N/A')}")
            print(f"    bound_management: {getattr(backward_io, 'bound_management', 'N/A')}")

        # Check update configuration
        if hasattr(rpu_config, 'update'):
            update = rpu_config.update
            print(f"\n  Update Configuration:")
            print(f"    pulse_type: {getattr(update, 'pulse_type', 'N/A')}")
            print(f"    update_bl_management: {getattr(update, 'update_bl_management', 'N/A')}")
            print(f"    update_management: {getattr(update, 'update_management', 'N/A')}")
            if hasattr(update, 'desired_bl'):
                print(f"    desired_bl: {update.desired_bl}")

        # Check drift
        if hasattr(rpu_config, 'drift'):
            drift = rpu_config.drift
            print(f"\n  Drift Configuration:")
            print(f"    nu: {getattr(drift, 'nu', 'N/A')}")
            print(f"    nu_dtod: {getattr(drift, 'nu_dtod', 'N/A')}")
            print(f"    t_0: {getattr(drift, 't_0', 'N/A')}")

        # Weight scaling
        if hasattr(rpu_config, 'mapping'):
            mapping = rpu_config.mapping
            print(f"\n  Weight Mapping:")
            print(f"    digital_bias: {getattr(mapping, 'digital_bias', 'N/A')}")
            print(f"    weight_scaling_omega: {getattr(mapping, 'weight_scaling_omega', 'N/A')}")
            print(f"    learn_out_scaling: {getattr(mapping, 'learn_out_scaling', 'N/A')}")

    # Check for base_layer config
    print(f"\n{'='*80}")
    print("  BASE LAYER CONFIGURATION")
    print(f"{'='*80}")

    base_layer = None
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and 'base_layer' in name:
            base_layer = (name, module)
            break

    if base_layer:
        name, module = base_layer
        print(f"\n  Analyzing: {name.split('.')[-4:]}")

        tile = module.analog_tile
        rpu_config = tile.rpu_config

        print(f"\n  RPU Config Type: {type(rpu_config).__name__}")

        if isinstance(rpu_config, SingleRPUConfig):
            device = rpu_config.device
            print(f"  Device Type: {type(device).__name__}")

            # Check forward I/O
            if hasattr(rpu_config, 'forward'):
                forward_io = rpu_config.forward
                print(f"\n  Forward Pass I/O Management:")
                print(f"    out_noise: {getattr(forward_io, 'out_noise', 'N/A')}")
                print(f"    noise_management: {getattr(forward_io, 'noise_management', 'N/A')}")
                print(f"    bound_management: {getattr(forward_io, 'bound_management', 'N/A')}")


def main():
    print("\n" + "="*80)
    print("  COMPLETE SIXT1C CONFIGURATION VERIFICATION")
    print("="*80)

    task_name = "sst2"
    target_modules = ["value"]
    lora_alpha = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nConfiguration:")
    print(f"  Task: {task_name}")
    print(f"  Target modules: {target_modules}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  Mode: SIXT1C (analog)")

    # Create model
    print("\n[Creating model...]")
    model = create_glue_model(task_name, device, target_modules,
                             fp_lora=False, lora_alpha=lora_alpha)

    # Run all checks
    bias_params = check_bias_status(model)
    layer_status = check_layer_freeze_status(model)
    trainable_params, frozen_params = check_trainable_parameters(model)
    analog_layers = check_analog_tiles_and_learn_out_scaling(model)
    comparisons = check_pretrained_weights(model, task_name)
    check_sixt1c_device_config(model)

    # Final summary
    print_section("FINAL SUMMARY")

    print(f"\n✓ Bias Status:")
    frozen_bias = sum(1 for p in bias_params.values() if not p['requires_grad'])
    trainable_bias = sum(1 for p in bias_params.values() if p['requires_grad'])
    print(f"    Frozen: {frozen_bias}, Trainable: {trainable_bias}")

    print(f"\n✓ Layer Status:")
    print(f"    LoRA A (analog): {len(layer_status['lora_A (analog)']['trainable'])} trainable")
    print(f"    LoRA B (analog): {len(layer_status['lora_B (analog)']['trainable'])} trainable")
    print(f"    Base layer (analog): {len(layer_status['base_layer (analog)']['frozen'])} frozen")

    print(f"\n✓ Trainable Parameters:")
    total_trainable = sum(p[1] for p in trainable_params)
    print(f"    Total: {total_trainable:,}")

    print(f"\n✓ Learn Out Scaling:")
    lora_with_learn_out = [l for l in analog_layers if 'lora' in l['name'] and l['learn_out_scaling']]
    print(f"    LoRA layers with learn_out_scaling: {len(lora_with_learn_out)}/{len([l for l in analog_layers if 'lora' in l['name']])}")

    print(f"\n✓ Pretrained Weights:")
    identical = len([c for c in comparisons if c['diff'] < 1e-6])
    print(f"    Preserved: {identical}/{len(comparisons)}")

    print("\n" + "="*80)
    print("  VERIFICATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

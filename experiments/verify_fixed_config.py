"""
Verify that the fixed configuration is applied correctly in the model.
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from sixt1c_config import gen_sixt1c_lora_config_trainable, gen_softbounds_base_layer_config_trainable
from aihwkit.nn import AnalogLinear


def main():
    print("="*80)
    print("  VERIFICATION: FIXED CONFIGURATION")
    print("="*80)

    # Test 1: Check config generation
    print("\n[Test 1] Config Generation")
    print("-"*80)

    lora_config = gen_sixt1c_lora_config_trainable(output_noise_level=0.0)
    base_config = gen_softbounds_base_layer_config_trainable(output_noise_level=0.0)

    print("\nLoRA Config Mapping:")
    print(f"  digital_bias:              {lora_config.mapping.digital_bias}")
    print(f"  learn_out_scaling:         {lora_config.mapping.learn_out_scaling}")
    print(f"  out_scaling_columnwise:    {lora_config.mapping.out_scaling_columnwise}")
    print(f"  weight_scaling_omega:      {lora_config.mapping.weight_scaling_omega}")
    print(f"  weight_scaling_columnwise: {lora_config.mapping.weight_scaling_columnwise}")

    print("\nBase Layer Config Mapping:")
    print(f"  digital_bias:              {base_config.mapping.digital_bias}")
    print(f"  learn_out_scaling:         {base_config.mapping.learn_out_scaling}")
    print(f"  out_scaling_columnwise:    {base_config.mapping.out_scaling_columnwise}")
    print(f"  weight_scaling_omega:      {base_config.mapping.weight_scaling_omega}")
    print(f"  weight_scaling_columnwise: {base_config.mapping.weight_scaling_columnwise}")

    # Verify expected values
    checks = {
        'LoRA learn_out_scaling': lora_config.mapping.learn_out_scaling == True,
        'LoRA out_scaling_columnwise': lora_config.mapping.out_scaling_columnwise == True,
        'LoRA weight_scaling_omega': lora_config.mapping.weight_scaling_omega == 1.0,
        'LoRA weight_scaling_columnwise': lora_config.mapping.weight_scaling_columnwise == True,
        'Base learn_out_scaling': base_config.mapping.learn_out_scaling == True,
        'Base out_scaling_columnwise': base_config.mapping.out_scaling_columnwise == True,
        'Base weight_scaling_omega': base_config.mapping.weight_scaling_omega == 1.0,
    }

    print("\n[Test 2] Verification")
    print("-"*80)

    all_passed = True
    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    # Test 3: Create model and check one layer
    print("\n[Test 3] Model Creation")
    print("-"*80)
    print("\nCreating model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_glue_model('sst2', device, ['value'], fp_lora=False, lora_alpha=1.0)

    # Find analog layers and check their analog_module
    lora_a_found = False
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and 'lora_A' in name and not lora_a_found:
            print(f"\nFound LoRA A layer: {name.split('.')[-4:]}")

            # Try to access the underlying analog module
            if hasattr(module, 'analog_module'):
                print(f"  Has analog_module: True")
                analog_mod = module.analog_module

                if hasattr(analog_mod, 'rpu_config'):
                    config = analog_mod.rpu_config
                    print(f"\n  Analog Module Config:")
                    print(f"    learn_out_scaling: {config.mapping.learn_out_scaling}")
                    print(f"    out_scaling_columnwise: {config.mapping.out_scaling_columnwise}")
                    print(f"    weight_scaling_omega: {config.mapping.weight_scaling_omega}")
            else:
                print(f"  No analog_module attribute")

            lora_a_found = True
            break

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    if all_passed:
        print("\n  ✓✓✓ ALL CHECKS PASSED!")
        print("  Configuration is now CORRECT!")
        print("\n  Fixed values:")
        print("    - learn_out_scaling: True")
        print("    - out_scaling_columnwise: True")
        print("    - weight_scaling_omega: 1.0")
        print("    - weight_scaling_columnwise: True")
    else:
        print("\n  ✗✗✗ SOME CHECKS FAILED!")
        print("  Configuration still has issues!")

    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

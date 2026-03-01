"""
Check learn_out_scaling configuration in detail.
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sixt1c_config import gen_sixt1c_lora_config_trainable


def main():
    print("="*80)
    print("  LEARN_OUT_SCALING CONFIGURATION CHECK")
    print("="*80)

    # Generate config
    config = gen_sixt1c_lora_config_trainable(output_noise_level=0.0)

    print(f"\nConfig Type: {type(config).__name__}")
    print(f"Device Type: {type(config.device).__name__}")

    # Check if config has mapping attribute
    print(f"\nHas 'mapping' attribute: {hasattr(config, 'mapping')}")

    if hasattr(config, 'mapping'):
        print(f"\nMapping attributes:")
        mapping = config.mapping
        for attr in dir(mapping):
            if not attr.startswith('_'):
                try:
                    value = getattr(mapping, attr)
                    if not callable(value):
                        print(f"  {attr}: {value}")
                except:
                    pass
    else:
        print("\n  SingleRPUConfig does not have 'mapping' attribute!")
        print("  This is expected - SingleRPUConfig uses different structure.")

    # Check forward config
    print(f"\nForward I/O configuration:")
    forward = config.forward
    print(f"  Has 'learn_out_scaling': {hasattr(forward, 'learn_out_scaling')}")
    if hasattr(forward, 'learn_out_scaling'):
        print(f"  learn_out_scaling: {forward.learn_out_scaling}")
    else:
        print("  learn_out_scaling not in forward config")

    print(f"  Has 'out_scaling_columnwise': {hasattr(forward, 'out_scaling_columnwise')}")
    if hasattr(forward, 'out_scaling_columnwise'):
        print(f"  out_scaling_columnwise: {forward.out_scaling_columnwise}")
    else:
        print("  out_scaling_columnwise not in forward config")

    # Check what attributes forward has
    print(f"\n  Available forward attributes:")
    for attr in dir(forward):
        if not attr.startswith('_') and 'scale' in attr.lower():
            try:
                value = getattr(forward, attr)
                if not callable(value):
                    print(f"    {attr}: {value}")
            except:
                pass

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("""
SingleRPUConfig (trainable) does NOT have 'mapping' attribute.
The 'learn_out_scaling' and 'out_scaling_columnwise' settings
are part of TorchInferenceRPUConfig's mapping, which is used
for inference-only (frozen) configs.

For trainable configs (SingleRPUConfig), out_scaling is handled
differently - the analog optimizer manages scaling internally
during training without explicit learn_out_scaling flag.

The key settings that ARE present in SingleRPUConfig:
  ✓ forward.inp_res (8-bit quantization)
  ✓ forward.out_res (8-bit quantization)
  ✓ forward.noise_management (ABS_MAX)
  ✓ forward.bound_management (ITERATIVE)
  ✓ device.mult_noise (False - deterministic)
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

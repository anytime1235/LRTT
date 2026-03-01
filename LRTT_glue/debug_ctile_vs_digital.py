"""Verify: Is C tile forward equivalent to digital base layer forward?

Quantify the exact difference between:
1. Digital: y = W @ x (exact FP32)
2. C tile: y = tile_c.forward(x) * mapping_scale * out_scaling + bias
   - with inp_res, out_res, out_noise, bound_management, noise_management
"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
import math

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
LORA_ALPHA = 0.01


def create_config():
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-1.0 / TAU_SEC)
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
        rank=RANK, transfer_every=1000000,
        lora_alpha=LORA_ALPHA, reinit_gain=0.1,
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load digital model
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    digital_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)

    # Save digital query weights for layer 0
    digital_query = None
    for name, mod in digital_model.named_modules():
        if 'layer.0' in name and 'query' in name and isinstance(mod, nn.Linear):
            digital_query = mod
            break

    if digital_query is None:
        print("ERROR: Could not find layer.0 query")
        return

    digital_weight = digital_query.weight.data.clone()
    digital_bias = digital_query.bias.data.clone() if digital_query.bias is not None else None

    print("=" * 70)
    print("C TILE vs DIGITAL BASE LAYER FORWARD COMPARISON")
    print("=" * 70)
    print(f"\nDigital query weight: shape={digital_weight.shape}, "
          f"norm={digital_weight.norm():.4f}, max={digital_weight.abs().max():.4f}")

    # Convert to analog
    all_linear = [n for n, m in digital_model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [n for n in all_linear if 'query' not in n]
    exclude.append('classifier')

    rpu_config = create_config()
    analog_model = convert_to_analog(digital_model, rpu_config, exclude_modules=exclude).to(device)

    # Find the analog query module
    analog_query = None
    analog_name = None
    for name, mod in analog_model.named_modules():
        if 'layer.0' in name and 'query' in name and hasattr(mod, 'analog_module'):
            analog_query = mod.analog_module
            analog_name = name
            break

    if analog_query is None:
        print("ERROR: Could not find analog layer.0 query")
        return

    ctrl = analog_query.controller

    # ---- 1. Weight comparison ----
    print("\n[1] WEIGHT COMPARISON (after convert_to_analog):")
    c_w = ctrl.tile_c.get_weights()[0]
    print(f"  C tile weight:  shape={c_w.shape}, norm={c_w.norm():.4f}, max={c_w.abs().max():.4f}")
    print(f"  Digital weight:  shape={digital_weight.shape}, norm={digital_weight.norm():.4f}, max={digital_weight.abs().max():.4f}")
    print(f"  Weight diff:    {(c_w - digital_weight.cpu()).abs().max():.6f}")
    print(f"  Weights match:  {torch.allclose(c_w, digital_weight.cpu(), atol=1e-4)}")

    # Check mapping scales
    if hasattr(ctrl.tile_c, 'mapping_scales') and ctrl.tile_c.mapping_scales is not None:
        ms = ctrl.tile_c.mapping_scales
        print(f"\n  Mapping scales: shape={ms.shape}, mean={ms.mean():.4f}, min={ms.min():.4f}, max={ms.max():.4f}")
        print(f"  Mapping scales sample: {ms[:5]}")
    else:
        print(f"\n  No mapping_scales found")

    # Check out_scaling_alpha
    if hasattr(ctrl.tile_c, 'out_scaling_alpha') and ctrl.tile_c.out_scaling_alpha is not None:
        osa = ctrl.tile_c.out_scaling_alpha.data
        print(f"  Out_scaling_alpha: shape={osa.shape}, mean={osa.mean():.4f}, min={osa.min():.4f}, max={osa.max():.4f}")
    else:
        print(f"  No out_scaling_alpha found")

    # Reconstruct: C_stored * mapping_scale should ~ digital weight
    if hasattr(ctrl.tile_c, 'mapping_scales') and ctrl.tile_c.mapping_scales is not None:
        ms = ctrl.tile_c.mapping_scales.cpu()
        reconstructed = c_w * ms.view(-1, 1) if ms.dim() == 1 else c_w * ms
        print(f"\n  Reconstructed (C * mapping): norm={reconstructed.norm():.4f}, max={reconstructed.abs().max():.4f}")
        print(f"  Recon vs Digital diff: {(reconstructed - digital_weight.cpu()).abs().max():.6f}")

    # ---- 2. Forward IO settings ----
    print("\n[2] C TILE FORWARD IO SETTINGS:")
    rpu = ctrl.tile_c.rpu_config if hasattr(ctrl.tile_c, 'rpu_config') else None
    if rpu:
        fwd = rpu.forward
        print(f"  inp_res:       {fwd.inp_res}")
        print(f"  inp_bound:     {fwd.inp_bound}")
        print(f"  inp_noise:     {getattr(fwd, 'inp_noise', 'N/A')}")
        print(f"  out_res:       {fwd.out_res}")
        print(f"  out_bound:     {fwd.out_bound}")
        print(f"  out_noise:     {fwd.out_noise}")
        print(f"  w_noise:       {getattr(fwd, 'w_noise', 'N/A')}")
        print(f"  bound_mgmt:    {getattr(fwd, 'bound_management', 'N/A')}")
        print(f"  noise_mgmt:    {getattr(fwd, 'noise_management', 'N/A')}")
        print(f"  is_perfect:    {getattr(fwd, 'is_perfect', 'N/A')}")

        bwd = rpu.backward
        print(f"\n  [Backward]")
        print(f"  out_noise:     {bwd.out_noise}")
        print(f"  out_res:       {bwd.out_res}")
    else:
        print("  No rpu_config found")

    # ---- 3. Forward output comparison ----
    print("\n[3] FORWARD OUTPUT COMPARISON:")
    torch.manual_seed(42)
    x = torch.randn(4, 128, device=device)  # [batch, in_features]

    # Digital forward: y = x @ W^T + bias
    with torch.no_grad():
        y_digital = x @ digital_weight.t().to(device)
        if digital_bias is not None:
            y_digital += digital_bias.to(device)

    # C tile forward (through analog model)
    # Run multiple times to measure noise variance
    y_analogs = []
    with torch.no_grad():
        for i in range(10):
            y_a = ctrl.tile_c.forward(x)
            # Apply mapping_scales and out_scaling
            if hasattr(ctrl.tile_c, 'mapping_scales') and ctrl.tile_c.mapping_scales is not None:
                ms = ctrl.tile_c.mapping_scales.to(device)
                if ms.dim() == 1:
                    y_a = y_a * ms.unsqueeze(0)
            if hasattr(ctrl.tile_c, 'out_scaling_alpha') and ctrl.tile_c.out_scaling_alpha is not None:
                osa = ctrl.tile_c.out_scaling_alpha.data.to(device)
                y_a = y_a * osa.unsqueeze(0)
            y_analogs.append(y_a.clone())

    y_analog_mean = torch.stack(y_analogs).mean(dim=0)
    y_analog_std = torch.stack(y_analogs).std(dim=0)

    print(f"  Digital output:     norm={y_digital.norm():.4f}, mean={y_digital.mean():.6f}")
    print(f"  Analog mean (10x):  norm={y_analog_mean.norm():.4f}, mean={y_analog_mean.mean():.6f}")
    print(f"  Analog std (10x):   mean_std={y_analog_std.mean():.6f}, max_std={y_analog_std.max():.6f}")
    print(f"  Mean diff:          {(y_analog_mean - y_digital).abs().max():.6f}")
    print(f"  Relative diff:      {(y_analog_mean - y_digital).norm() / y_digital.norm():.6f}")

    # Single sample comparison
    diff_single = (y_analogs[0] - y_digital).abs()
    print(f"\n  Single pass diff:   max={diff_single.max():.6f}, mean={diff_single.mean():.6f}")
    print(f"  SNR (signal/noise): {y_digital.norm() / (y_analogs[0] - y_digital).norm():.2f}")

    # ---- 4. Full model forward comparison ----
    print("\n[4] FULL MODEL FORWARD COMPARISON (single batch):")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer("Hello world", return_tensors="pt", padding="max_length", max_length=128).to(device)

    # Digital model (fresh copy)
    digital_model2 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)
    digital_model2.eval()

    analog_model.eval()

    logits_list = []
    with torch.no_grad():
        y_dig = digital_model2(**inputs).logits
        for i in range(5):
            y_ana = analog_model(**inputs).logits
            logits_list.append(y_ana.clone())

    y_ana_mean = torch.stack(logits_list).mean(dim=0)
    y_ana_std = torch.stack(logits_list).std(dim=0)

    print(f"  Digital logits: {y_dig}")
    print(f"  Analog mean:    {y_ana_mean}")
    print(f"  Analog std:     {y_ana_std}")
    print(f"  Logit diff:     {(y_ana_mean - y_dig).abs()}")

    # ---- 5. Check: is tile_c forward exactly matmul? ----
    print("\n[5] TILE_C FORWARD vs EXACT MATMUL:")
    c_w_gpu = c_w.to(device)

    with torch.no_grad():
        # Raw tile forward (no scaling)
        y_raw_tile = ctrl.tile_c.forward(x)  # This includes IO effects

        # Manual exact matmul with C weights
        y_exact = x @ c_w_gpu.t()

    print(f"  Raw tile forward:   norm={y_raw_tile.norm():.4f}")
    print(f"  Exact matmul:       norm={y_exact.norm():.4f}")
    print(f"  Diff:               {(y_raw_tile - y_exact).abs().max():.6f}")
    print(f"  Relative diff:      {(y_raw_tile - y_exact).norm() / y_exact.norm():.6f}")


if __name__ == "__main__":
    main()

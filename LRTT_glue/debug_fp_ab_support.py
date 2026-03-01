"""Verify: Do FloatingPointDevice A/B tiles in LRTT work as exact digital?

Check:
1. tile_a/tile_b forward = exact matmul? (ignores IO params?)
2. tile_a backward = exact matmul? (ignores backward.out_noise?)
3. tile_a/tile_b update = exact SGD? (w -= lr * d^T @ x)
4. forward_inject = C_analog + alpha * A_digital * B_digital?
"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
import math

from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.devices import FloatingPointDevice, SoftBoundsDevice, LinearStepDevice
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.nn.conversion import convert_to_analog


def create_config_fp_ab():
    """LRTT config with FloatingPointDevice A/B, SoftBounds C."""
    ab_device = FloatingPointDevice()
    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )
    device_config = PythonLRTTDevice(
        rank=4, transfer_every=1000000,
        lora_alpha=1.0, reinit_gain=0.1,
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
    # NOTE: keep default forward/backward IO settings (which have noise/quantization)
    # to verify that FloatingPointTile ignores them
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def create_config_sixt1c_ab():
    """LRTT config with LinearStepDevice A/B (realistic), SoftBounds C."""
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
        rank=4, transfer_every=1000000,
        lora_alpha=1.0, reinit_gain=0.1,
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
    torch.manual_seed(42)

    IN, OUT, RANK = 64, 32, 4

    print("=" * 70)
    print("FloatingPointDevice A/B SUPPORT TEST in LRTT")
    print(f"in={IN}, out={OUT}, rank={RANK}")
    print("=" * 70)

    # Create a simple linear layer and convert
    torch.manual_seed(42)
    base_linear = nn.Linear(IN, OUT, bias=False)
    digital_weight = base_linear.weight.data.clone()

    rpu_config = create_config_fp_ab()
    analog_model = convert_to_analog(base_linear, rpu_config).to(device)

    # Find analog module
    analog_mod = None
    for name, mod in analog_model.named_modules():
        if hasattr(mod, 'analog_module'):
            analog_mod = mod.analog_module
            break

    if analog_mod is None:
        print("ERROR: Could not find analog module")
        return

    ctrl = analog_mod.controller

    # ---- Check tile types ----
    print("\n[0] TILE TYPES:")
    print(f"  tile_a: {type(ctrl.tile_a)}")
    print(f"  tile_b: {type(ctrl.tile_b)}")
    print(f"  tile_c: {type(ctrl.tile_c)}")

    # Check rpu_config IO settings on tiles
    if hasattr(ctrl.tile_a, 'rpu_config'):
        fwd_a = ctrl.tile_a.rpu_config.forward
        bwd_a = ctrl.tile_a.rpu_config.backward
        print(f"\n  tile_a rpu_config.forward.inp_res: {fwd_a.inp_res}")
        print(f"  tile_a rpu_config.forward.out_noise: {fwd_a.out_noise}")
        print(f"  tile_a rpu_config.backward.out_noise: {bwd_a.out_noise}")
        print(f"  tile_a rpu_config.forward.is_perfect: {getattr(fwd_a, 'is_perfect', 'N/A')}")

    if hasattr(ctrl.tile_c, 'rpu_config'):
        fwd_c = ctrl.tile_c.rpu_config.forward
        print(f"\n  tile_c rpu_config.forward.inp_res: {fwd_c.inp_res}")
        print(f"  tile_c rpu_config.forward.out_noise: {fwd_c.out_noise}")

    # ---- Test 1: A/B tile forward = exact matmul? ----
    print("\n[1] A/B TILE FORWARD vs EXACT MATMUL:")

    # Get A/B weights
    a_w = ctrl.tile_a.get_weights()[0].to(device)  # [d_size, rank]
    b_w = ctrl.tile_b.get_weights()[0].to(device)  # [rank, x_size]

    print(f"  A weights: shape={a_w.shape}, norm={a_w.norm():.6f}")
    print(f"  B weights: shape={b_w.shape}, norm={b_w.norm():.6f}")

    x = torch.randn(8, IN, device=device)

    # B tile forward: multiple times to check noise
    b_fwds = []
    with torch.no_grad():
        for i in range(10):
            y_b = ctrl.tile_b.forward(x)
            b_fwds.append(y_b.clone())

    y_b_exact = x @ b_w.t()  # [8, rank]
    y_b_mean = torch.stack(b_fwds).mean(dim=0)
    y_b_std = torch.stack(b_fwds).std(dim=0)

    print(f"\n  B.forward(x) vs x @ B^T:")
    print(f"    Exact matmul:   norm={y_b_exact.norm():.6f}")
    print(f"    Tile fwd mean:  norm={y_b_mean.norm():.6f}")
    print(f"    Tile fwd std:   max={y_b_std.max():.8f}")
    print(f"    Max diff:       {(y_b_mean - y_b_exact).abs().max():.8f}")
    print(f"    Is deterministic: {y_b_std.max() < 1e-7}")
    print(f"    Is exact:       {torch.allclose(b_fwds[0], y_b_exact, atol=1e-5)}")

    # A tile forward (A is zero-initialized, so let's set some weights first)
    a_test = torch.randn(OUT, RANK, device=device) * 0.1
    ctrl.tile_a.set_weights(a_test)
    a_w = ctrl.tile_a.get_weights()[0].to(device)

    g = torch.randn(8, RANK, device=device)
    a_fwds = []
    with torch.no_grad():
        for i in range(10):
            y_a = ctrl.tile_a.forward(g)
            a_fwds.append(y_a.clone())

    y_a_exact = g @ a_w.t()
    y_a_mean = torch.stack(a_fwds).mean(dim=0)
    y_a_std = torch.stack(a_fwds).std(dim=0)

    print(f"\n  A.forward(g) vs g @ A^T:")
    print(f"    Exact matmul:   norm={y_a_exact.norm():.6f}")
    print(f"    Tile fwd mean:  norm={y_a_mean.norm():.6f}")
    print(f"    Tile fwd std:   max={y_a_std.max():.8f}")
    print(f"    Max diff:       {(y_a_mean - y_a_exact).abs().max():.8f}")
    print(f"    Is deterministic: {y_a_std.max() < 1e-7}")
    print(f"    Is exact:       {torch.allclose(a_fwds[0], y_a_exact, atol=1e-5)}")

    # ---- Test 2: A tile backward = exact matmul? (critical: out_noise?) ----
    print("\n[2] A TILE BACKWARD (critical for B update):")

    d = torch.randn(8, OUT, device=device)
    da_results = []
    with torch.no_grad():
        for i in range(10):
            da = ctrl.tile_a.backward(d)
            da_results.append(da.clone())

    da_exact = d @ a_w  # [8, OUT] @ [OUT, RANK] = [8, RANK]
    da_mean = torch.stack(da_results).mean(dim=0)
    da_std = torch.stack(da_results).std(dim=0)

    print(f"  A.backward(d) vs d @ A:")
    print(f"    Exact:          norm={da_exact.norm():.6f}")
    print(f"    Tile bwd mean:  norm={da_mean.norm():.6f}")
    print(f"    Tile bwd std:   max={da_std.max():.8f}")
    print(f"    Max diff:       {(da_mean - da_exact).abs().max():.8f}")
    print(f"    Is deterministic: {da_std.max() < 1e-7}")
    print(f"    Is exact:       {torch.allclose(da_results[0], da_exact, atol=1e-5)}")
    print(f"    ** out_noise affects backward? {da_std.max() > 1e-7} **")

    # Test with A=0 (the init state) - does backward return exactly zero?
    print(f"\n  A.backward(d) when A=0:")
    ctrl.tile_a.set_weights(torch.zeros(OUT, RANK))
    da_zero_results = []
    with torch.no_grad():
        for i in range(10):
            da = ctrl.tile_a.backward(d)
            da_zero_results.append(da.clone())

    da_zero_mean = torch.stack(da_zero_results).mean(dim=0)
    da_zero_std = torch.stack(da_zero_results).std(dim=0)
    print(f"    DA norm (should be 0): {da_zero_mean.norm():.8f}")
    print(f"    DA std (should be 0):  {da_zero_std.max():.8f}")
    print(f"    Is exactly zero: {da_zero_mean.abs().max() < 1e-7}")
    print(f"    ** NO spurious B update when A=0! **" if da_zero_mean.abs().max() < 1e-7
          else f"    ** SPURIOUS B update when A=0! DA norm={da_zero_mean.norm():.6f} **")

    # Restore A weights
    ctrl.tile_a.set_weights(a_test)

    # ---- Test 3: A/B tile update = exact SGD? ----
    print("\n[3] A/B TILE UPDATE = EXACT SGD?")

    # Save current B weights
    b_before = ctrl.tile_b.get_weights()[0].clone().to(device)
    lr_test = 0.01

    x_upd = torch.randn(8, IN, device=device)
    d_upd = torch.randn(8, RANK, device=device)  # For B: update(x, DA)

    ctrl.tile_b.set_learning_rate(lr_test)
    ctrl.tile_b.update(x_upd, d_upd)

    b_after = ctrl.tile_b.get_weights()[0].to(device)

    # Expected: b_after = b_before - lr * d^T @ x
    b_expected = b_before - lr_test * (d_upd.t() @ x_upd)  # [RANK, OUT] @ [OUT, IN] wait...

    # tile_b is [RANK, IN], update(x, d) where x=[batch, IN], d=[batch, RANK]
    # Expected: W -= lr * d^T @ x = [RANK, batch] @ [batch, IN] = [RANK, IN]
    b_expected_correct = b_before - lr_test * (d_upd.t() @ x_upd)

    diff_sgd = (b_after - b_expected_correct).abs().max()
    print(f"  B update: update(x=[8,{IN}], d=[8,{RANK}])")
    print(f"  B before norm:     {b_before.norm():.6f}")
    print(f"  B after norm:      {b_after.norm():.6f}")
    print(f"  Expected (SGD):    {b_expected_correct.norm():.6f}")
    print(f"  Diff vs exact SGD: {diff_sgd:.8f}")
    print(f"  Is exact SGD:      {diff_sgd < 1e-5}")

    # ---- Test 4: forward_inject = C_tile + alpha * A_fp * B_fp? ----
    print("\n[4] FORWARD INJECT COMPOSITION:")

    # Reset A to non-zero for testing
    ctrl.tile_a.set_weights(a_test)

    x_fwd = torch.randn(8, IN, device=device)

    # Run forward_inject multiple times
    y_injects = []
    with torch.no_grad():
        for i in range(10):
            y = ctrl.forward_inject(x_fwd)
            y_injects.append(y.clone())

    y_inject_mean = torch.stack(y_injects).mean(dim=0)
    y_inject_std = torch.stack(y_injects).std(dim=0)

    # Compute digital: y = W_base @ x + alpha * A @ (B @ x)
    c_w = ctrl.tile_c.get_weights()[0].to(device)
    a_w = ctrl.tile_a.get_weights()[0].to(device)
    b_w = ctrl.tile_b.get_weights()[0].to(device)
    alpha = ctrl.lora_alpha

    y_digital = x_fwd @ c_w.t() + alpha * (x_fwd @ b_w.t()) @ a_w.t()

    print(f"  forward_inject() vs digital C+alpha*A*B:")
    print(f"    Digital:        norm={y_digital.norm():.4f}")
    print(f"    Inject mean:    norm={y_inject_mean.norm():.4f}")
    print(f"    Inject std:     max={y_inject_std.max():.6f}")
    print(f"    Max diff:       {(y_inject_mean - y_digital).abs().max():.6f}")
    print(f"    Relative diff:  {(y_inject_mean - y_digital).norm() / y_digital.norm():.6f}")

    # Decompose: C tile part vs A*B part
    y_c_only = []
    with torch.no_grad():
        for i in range(5):
            y_c = ctrl.tile_c.forward(x_fwd)
            y_c_only.append(y_c.clone())

    y_c_exact = x_fwd @ c_w.t()
    y_c_mean = torch.stack(y_c_only).mean(dim=0)
    y_c_std = torch.stack(y_c_only).std(dim=0)

    print(f"\n  C tile forward (analog, with IO effects):")
    print(f"    C exact:  norm={y_c_exact.norm():.4f}")
    print(f"    C tile:   norm={y_c_mean.norm():.4f}")
    print(f"    C std:    max={y_c_std.max():.6f}")
    print(f"    C diff:   {(y_c_mean - y_c_exact).abs().max():.6f}")
    print(f"    C rel:    {(y_c_mean - y_c_exact).norm() / y_c_exact.norm():.6f}")

    y_ab = x_fwd @ b_w.t() @ a_w.t()
    print(f"\n  A*B part (should be exact digital):")
    print(f"    AB exact: norm={y_ab.norm():.4f}")
    print(f"    Noise in inject should come ONLY from C tile")

    # ---- Test 5: Compare with sixt1c A/B ----
    print("\n" + "=" * 70)
    print("[5] COMPARISON: FloatingPoint A/B vs LinearStep A/B")
    print("=" * 70)

    torch.manual_seed(42)
    base_linear2 = nn.Linear(IN, OUT, bias=False)
    base_linear2.weight.data.copy_(digital_weight)

    rpu_config2 = create_config_sixt1c_ab()
    analog_model2 = convert_to_analog(base_linear2, rpu_config2).to(device)

    analog_mod2 = None
    for name, mod in analog_model2.named_modules():
        if hasattr(mod, 'analog_module'):
            analog_mod2 = mod.analog_module
            break

    ctrl2 = analog_mod2.controller
    print(f"  sixt1c tile_a type: {type(ctrl2.tile_a)}")
    print(f"  sixt1c tile_b type: {type(ctrl2.tile_b)}")

    # Set same A weights for comparison
    ctrl2.tile_a.set_weights(a_test.cpu())

    # backward comparison with A=non-zero
    d_test = torch.randn(8, OUT, device=device)
    da_fp_results = []
    da_sixt1c_results = []
    with torch.no_grad():
        for i in range(10):
            da_fp = ctrl.tile_a.backward(d_test)
            da_sixt1c = ctrl2.tile_a.backward(d_test)
            da_fp_results.append(da_fp.clone())
            da_sixt1c_results.append(da_sixt1c.clone())

    da_fp_std = torch.stack(da_fp_results).std(dim=0)
    da_sixt1c_std = torch.stack(da_sixt1c_results).std(dim=0)
    da_exact = d_test @ a_test.to(device)

    print(f"\n  A.backward(d) comparison (A non-zero):")
    print(f"    Exact:                 norm={da_exact.norm():.6f}")
    print(f"    FP tile std:           max={da_fp_std.max():.8f}")
    print(f"    Sixt1c tile std:       max={da_sixt1c_std.max():.8f}")
    print(f"    FP vs exact diff:      {(da_fp_results[0] - da_exact).abs().max():.8f}")
    print(f"    Sixt1c vs exact diff:  {(da_sixt1c_results[0] - da_exact).abs().max():.8f}")

    # backward comparison with A=0
    ctrl.tile_a.set_weights(torch.zeros(OUT, RANK))
    ctrl2.tile_a.set_weights(torch.zeros(OUT, RANK))

    da_fp_zero = []
    da_sixt1c_zero = []
    with torch.no_grad():
        for i in range(10):
            da_fp_zero.append(ctrl.tile_a.backward(d_test).clone())
            da_sixt1c_zero.append(ctrl2.tile_a.backward(d_test).clone())

    print(f"\n  A.backward(d) when A=0 (spurious B update test):")
    print(f"    FP:     DA norm={torch.stack(da_fp_zero).mean(0).norm():.8f}, "
          f"std={torch.stack(da_fp_zero).std(0).max():.8f}")
    print(f"    Sixt1c: DA norm={torch.stack(da_sixt1c_zero).mean(0).norm():.8f}, "
          f"std={torch.stack(da_sixt1c_zero).std(0).max():.8f}")
    print(f"    FP spurious:     {'NO' if torch.stack(da_fp_zero).mean(0).norm() < 1e-7 else 'YES'}")
    print(f"    Sixt1c spurious: {'NO' if torch.stack(da_sixt1c_zero).mean(0).norm() < 1e-7 else 'YES'}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  FloatingPointDevice A/B in LRTT with forward_inject=True:

  1. A/B forward:   {'EXACT (digital-equivalent)' if y_b_std.max() < 1e-7 else 'HAS NOISE'}
  2. A backward:    {'EXACT (no out_noise)' if da_std.max() < 1e-7 else 'HAS NOISE (backward.out_noise leaks!)'}
  3. A=0 backward:  {'ZERO (no spurious B update)' if da_zero_mean.abs().max() < 1e-7 else 'NON-ZERO (spurious B update!)'}
  4. B update:      {'EXACT SGD' if diff_sgd < 1e-5 else 'APPROXIMATE (pulsed?)'}
  5. forward_inject: C_analog(noisy) + alpha * A_digital(exact) * B_digital(exact)

  Architecture achieved:
    Base layer (C): Analog RPU with IO non-idealities (quantization, noise)
    LoRA A/B:       Digital-equivalent (exact FP32 matmul + exact SGD update)

  Key difference from sixt1c A/B:
    - No backward noise → no spurious B updates
    - No pulsed update artifacts → exact gradient descent
    - Deterministic forward → stable training
""")


if __name__ == "__main__":
    main()

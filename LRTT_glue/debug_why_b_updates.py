"""Trace exactly WHY tile B gets updated when A=0.

Key question: Is DA = tile_a.backward(d) truly zero when A weights are zero?
If not, what injects the non-zero values?
"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
import math
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD


def create_config(rank, lora_alpha, mode="analog"):
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-1.0 / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    if mode == "analog":
        ab_device = LinearStepDevice(
            dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=-0.1678, gamma_down=0.1410, mult_noise=False,
            dw_min_dtod=0.0, up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0, dw_min_std=0.0,
            write_noise_std=0.0, mean_bound_reference=True,
            lifetime=lifetime, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
        )
    else:
        ab_device = FloatingPointDevice()

    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )
    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=1000000,
        lora_alpha=lora_alpha, reinit_gain=0.1,
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
    device_hw = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a simple linear layer and convert
    linear = nn.Linear(128, 64, bias=False).to(device_hw)
    rpu_config = create_config(rank=8, lora_alpha=1.0, mode="analog")

    # Convert single layer
    from aihwkit.nn import AnalogLinear
    analog_linear = AnalogLinear(128, 64, bias=False, rpu_config=rpu_config).to(device_hw)

    # Get the LRTT tile
    tile = analog_linear.analog_module
    ctrl = tile.controller

    print("=" * 70)
    print("DEBUGGING: Why does tile B update when A=0?")
    print("=" * 70)

    # 1. Check initial A weights
    a_w = ctrl.tile_a.get_weights()[0]
    b_w = ctrl.tile_b.get_weights()[0]
    print(f"\n[1] Initial weights:")
    print(f"    A: norm={a_w.norm():.6f}  max={a_w.abs().max():.6f}  zeros={( a_w == 0).sum()}/{a_w.numel()}")
    print(f"    B: norm={b_w.norm():.6f}  max={b_w.abs().max():.6f}")

    # 2. Test tile_a.backward(d) when A=0
    d_test = torch.randn(16, 64, device=device_hw)  # [batch=16, d_size=64]
    print(f"\n[2] Testing tile_a.backward(d) with A=0:")
    print(f"    d input: norm={d_test.norm():.4f}  max={d_test.abs().max():.4f}")

    DA = ctrl.tile_a.backward(d_test)
    print(f"    DA output: norm={DA.norm():.6f}  max={DA.abs().max():.6f}  "
          f"zeros={(DA == 0).sum()}/{DA.numel()}  min_nonzero={DA[DA!=0].abs().min() if (DA!=0).any() else 0:.8f}")
    print(f"    DA shape: {DA.shape}")
    print(f"    DA sample values: {DA[0, :8]}")

    # 3. Compare with manual matmul
    DA_manual = d_test @ a_w.to(device_hw)  # [batch, d_size] @ [d_size, rank] = [batch, rank]
    print(f"\n[3] Manual matmul A^T @ d (should be identical if no noise):")
    print(f"    DA_manual: norm={DA_manual.norm():.6f}  max={DA_manual.abs().max():.6f}")
    print(f"    Difference: {(DA - DA_manual.to(DA.device)).abs().max():.8f}")

    # 4. Check tile_a backward IO settings
    print(f"\n[4] tile_a backward IO settings:")
    rpu = ctrl.tile_a.rpu_config if hasattr(ctrl.tile_a, 'rpu_config') else None
    if rpu:
        bwd = rpu.backward
        print(f"    out_noise: {bwd.out_noise}")
        print(f"    inp_noise: {getattr(bwd, 'inp_noise', 'N/A')}")
        print(f"    bound_management: {getattr(bwd, 'bound_management', 'N/A')}")
        print(f"    noise_management: {getattr(bwd, 'noise_management', 'N/A')}")
        print(f"    out_bound: {getattr(bwd, 'out_bound', 'N/A')}")
        print(f"    inp_bound: {getattr(bwd, 'inp_bound', 'N/A')}")
        print(f"    out_res: {getattr(bwd, 'out_res', 'N/A')}")
        print(f"    inp_res: {getattr(bwd, 'inp_res', 'N/A')}")
    else:
        print("    No rpu_config found on tile_a")

    # 5. Now test: record B weights before and after tile_b.update(x, DA)
    print(f"\n[5] Testing tile_b.update(x, DA) when DA is from A=0:")
    b_before = ctrl.tile_b.get_weights()[0].clone()
    x_test = torch.randn(16, 128, device=device_hw)  # [batch=16, x_size=128]

    ctrl.tile_b.set_learning_rate(0.001)
    ctrl.tile_b.update(x_test, DA)

    b_after = ctrl.tile_b.get_weights()[0]
    b_change = (b_after - b_before).norm()
    print(f"    B before: norm={b_before.norm():.6f}")
    print(f"    B after:  norm={b_after.norm():.6f}")
    print(f"    B change: {b_change:.6f}")
    print(f"    B max element change: {(b_after - b_before).abs().max():.8f}")

    # 6. Test with EXACTLY zero DA
    print(f"\n[6] Testing tile_b.update(x, DA=ZERO) directly:")
    b_before2 = ctrl.tile_b.get_weights()[0].clone()
    DA_zero = torch.zeros(16, 8, device=device_hw)

    ctrl.tile_b.set_learning_rate(0.001)
    ctrl.tile_b.update(x_test, DA_zero)

    b_after2 = ctrl.tile_b.get_weights()[0]
    b_change2 = (b_after2 - b_before2).norm()
    print(f"    B before: norm={b_before2.norm():.6f}")
    print(f"    B after:  norm={b_after2.norm():.6f}")
    print(f"    B change: {b_change2:.6f}")
    print(f"    B max element change: {(b_after2 - b_before2).abs().max():.8f}")

    # 7. Test with EXACTLY zero x
    print(f"\n[7] Testing tile_b.update(x=ZERO, DA=nonzero) directly:")
    b_before3 = ctrl.tile_b.get_weights()[0].clone()
    x_zero = torch.zeros(16, 128, device=device_hw)
    DA_nonzero = torch.randn(16, 8, device=device_hw)

    ctrl.tile_b.set_learning_rate(0.001)
    ctrl.tile_b.update(x_zero, DA_nonzero)

    b_after3 = ctrl.tile_b.get_weights()[0]
    b_change3 = (b_after3 - b_before3).norm()
    print(f"    B before: norm={b_before3.norm():.6f}")
    print(f"    B after:  norm={b_after3.norm():.6f}")
    print(f"    B change: {b_change3:.6f}")

    # 8. Now monkey-patch _ab_weight_update_lora to capture DA
    print(f"\n[8] Monkey-patching controller to inspect DA during actual training:")

    orig_update = ctrl._ab_weight_update_lora
    captured = {}

    def patched_update(x, d, lr, in_trans=False, out_trans=False):
        if in_trans:
            x = x.t()
        if out_trans:
            d = d.t()
        with torch.no_grad():
            XB = ctrl.tile_b.forward(x)
            DA = ctrl.tile_a.backward(d)
        captured['DA'] = DA.detach().clone()
        captured['XB'] = XB.detach().clone()
        captured['x'] = x.detach().clone()
        captured['d'] = d.detach().clone()
        captured['b_before'] = ctrl.tile_b.get_weights()[0].clone()
        captured['a_before'] = ctrl.tile_a.get_weights()[0].clone()
        # Call original
        orig_update(x, d, lr, in_trans=False, out_trans=False)  # already transposed
        captured['b_after'] = ctrl.tile_b.get_weights()[0].clone()
        captured['a_after'] = ctrl.tile_a.get_weights()[0].clone()

    ctrl._ab_weight_update_lora = patched_update

    # Run a single forward+backward+update
    x_input = torch.randn(4, 128, device=device_hw)
    target = torch.randn(4, 64, device=device_hw)

    analog_linear.train()
    out = analog_linear(x_input)
    loss = nn.MSELoss()(out, target)

    optimizer = AnalogSGD(analog_linear.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if captured:
        DA = captured['DA']
        XB = captured['XB']
        print(f"    DA (from A=0 backward):")
        print(f"      norm={DA.norm():.8f}  max={DA.abs().max():.8f}  "
              f"nonzero={(DA != 0).sum()}/{DA.numel()}")
        print(f"    XB (from B forward):")
        print(f"      norm={XB.norm():.4f}  max={XB.abs().max():.4f}")
        print(f"    d (gradient):")
        print(f"      norm={captured['d'].norm():.4f}  max={captured['d'].abs().max():.4f}")
        print(f"    A change: {(captured['a_after'] - captured['a_before']).norm():.6f}")
        print(f"    B change: {(captured['b_after'] - captured['b_before']).norm():.6f}")

        # Check: Is the DA non-zero from backward noise?
        a_w_check = captured['a_before']
        DA_expected = captured['d'].cpu() @ a_w_check  # manual: should be 0 if A=0
        print(f"\n    Expected DA (manual matmul): norm={DA_expected.norm():.8f}")
        print(f"    Actual DA norm: {DA.norm():.8f}")
        print(f"    Discrepancy (backward noise): {(DA.cpu() - DA_expected).norm():.8f}")
    else:
        print("    No captured data - update may not have been called")


if __name__ == "__main__":
    main()

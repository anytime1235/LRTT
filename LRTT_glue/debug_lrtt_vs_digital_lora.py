"""Verify: Does LRTT with FloatingPointDevice A/B == digital LoRA?

Compare step-by-step:
1. LRTT (FloatingPointDevice A/B) - through tile.update() path
2. Manual digital LoRA - standard PyTorch autograd + Adam

Check: forward output, gradients, weight updates, loss trajectory.
"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.devices import FloatingPointDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


# ============================================================
# 1. Manual Digital LoRA layer
# ============================================================
class DigitalLoRALinear(nn.Module):
    """Standard digital LoRA: y = Wx + alpha * A(Bx)."""
    def __init__(self, in_features, out_features, rank, lora_alpha, bias=False):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_B = nn.Linear(in_features, rank, bias=False)   # B: [rank, in]
        self.lora_A = nn.Linear(rank, out_features, bias=False)  # A: [out, rank]
        self.alpha = lora_alpha

        # Init: A=0, B=kaiming
        nn.init.zeros_(self.lora_A.weight)
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

        # Freeze base
        self.base.weight.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_A(self.lora_B(x))
        return base_out + self.alpha * lora_out


# ============================================================
# 2. Create LRTT analog model (FloatingPointDevice A/B)
# ============================================================
def create_lrtt_config(rank, lora_alpha):
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
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    IN, OUT, RANK, ALPHA = 64, 32, 4, 1.0
    BATCH = 8
    LR = 0.001
    STEPS = 10

    print("=" * 70)
    print("LRTT (FloatingPointDevice A/B) vs Digital LoRA Comparison")
    print(f"in={IN}, out={OUT}, rank={RANK}, alpha={ALPHA}, lr={LR}")
    print("=" * 70)

    # ---- Create digital LoRA ----
    torch.manual_seed(42)
    digital = DigitalLoRALinear(IN, OUT, RANK, ALPHA, bias=False).to(device)

    # ---- Create LRTT analog ----
    torch.manual_seed(42)
    base_linear = nn.Linear(IN, OUT, bias=False)
    # Copy same base weights
    base_linear.weight.data.copy_(digital.base.weight.data)

    rpu_config = create_lrtt_config(RANK, ALPHA)
    analog_model = convert_to_analog(base_linear, rpu_config).to(device)

    # Get the LRTT tile to inspect internals
    analog_mod = None
    for name, mod in analog_model.named_modules():
        if hasattr(mod, 'analog_module'):
            analog_mod = mod.analog_module
            break

    if analog_mod is None:
        print("ERROR: Could not find analog module")
        return

    ctrl = analog_mod.controller

    # ---- Compare initial weights ----
    print("\n[INIT] Weight comparison:")
    a_w_lrtt = ctrl.tile_a.get_weights()[0]
    b_w_lrtt = ctrl.tile_b.get_weights()[0]
    c_w_lrtt = ctrl.tile_c.get_weights()[0]

    a_w_dig = digital.lora_A.weight.data.cpu()
    b_w_dig = digital.lora_B.weight.data.cpu()
    w_dig = digital.base.weight.data.cpu()

    print(f"  LRTT  A: shape={a_w_lrtt.shape}, norm={a_w_lrtt.norm():.6f}")
    print(f"  Digit A: shape={a_w_dig.shape}, norm={a_w_dig.norm():.6f}")
    print(f"  LRTT  B: shape={b_w_lrtt.shape}, norm={b_w_lrtt.norm():.6f}")
    print(f"  Digit B: shape={b_w_dig.shape}, norm={b_w_dig.norm():.6f}")
    print(f"  LRTT  C: shape={c_w_lrtt.shape}, norm={c_w_lrtt.norm():.6f}")
    print(f"  Digit W: shape={w_dig.shape}, norm={w_dig.norm():.6f}")

    # Check if B init matches
    # LRTT B is [rank, in_features], digital B weight is [rank, in_features]
    print(f"\n  B weight match: {torch.allclose(b_w_lrtt, b_w_dig, atol=1e-4)}")
    print(f"  B diff: {(b_w_lrtt - b_w_dig).abs().max():.6f}")

    # ---- Compare forward pass ----
    print("\n[FORWARD] Same input comparison:")
    torch.manual_seed(123)
    x = torch.randn(BATCH, IN, device=device)

    with torch.no_grad():
        y_dig = digital(x)
        y_lrtt = analog_model(x)

    print(f"  Digital out: norm={y_dig.norm():.4f}, mean={y_dig.mean():.6f}")
    print(f"  LRTT out:    norm={y_lrtt.norm():.4f}, mean={y_lrtt.mean():.6f}")
    print(f"  Output diff: {(y_dig - y_lrtt).abs().max():.6f}")
    print(f"  Match: {torch.allclose(y_dig, y_lrtt, atol=1e-3)}")

    # ---- Compare training dynamics ----
    print(f"\n[TRAINING] Step-by-step comparison ({STEPS} steps):")

    # Optimizers
    dig_optimizer = torch.optim.Adam(
        [p for p in digital.parameters() if p.requires_grad], lr=LR
    )
    lrtt_optimizer = AnalogAdam(analog_model.parameters(), lr=LR)

    target = torch.randn(BATCH, OUT, device=device)
    criterion = nn.MSELoss()

    print(f"\n  {'Step':>4} | {'Dig Loss':>10} {'LRTT Loss':>10} {'Diff':>10} | "
          f"{'Dig A_norm':>10} {'LRTT A_norm':>10} | "
          f"{'Dig B_norm':>10} {'LRTT B_norm':>10} | "
          f"{'Out diff':>10}")
    print(f"  {'-'*96}")

    for step in range(STEPS):
        torch.manual_seed(step + 200)
        x = torch.randn(BATCH, IN, device=device)
        target = torch.randn(BATCH, OUT, device=device)

        # Digital forward + backward + step
        dig_optimizer.zero_grad()
        y_dig = digital(x)
        loss_dig = criterion(y_dig, target)
        loss_dig.backward()
        dig_optimizer.step()

        # LRTT forward + backward + step
        lrtt_optimizer.zero_grad()
        y_lrtt = analog_model(x)
        loss_lrtt = criterion(y_lrtt, target)
        loss_lrtt.backward()
        lrtt_optimizer.step()

        # Compare weights
        a_dig = digital.lora_A.weight.data.cpu()
        b_dig = digital.lora_B.weight.data.cpu()
        a_lrtt = ctrl.tile_a.get_weights()[0]
        b_lrtt = ctrl.tile_b.get_weights()[0]

        # Forward comparison after update
        with torch.no_grad():
            torch.manual_seed(step + 300)
            x_test = torch.randn(BATCH, IN, device=device)
            y_dig_test = digital(x_test)
            y_lrtt_test = analog_model(x_test)
            out_diff = (y_dig_test - y_lrtt_test).abs().max().item()

        print(f"  {step+1:>4} | {loss_dig.item():>10.4f} {loss_lrtt.item():>10.4f} "
              f"{abs(loss_dig.item()-loss_lrtt.item()):>10.6f} | "
              f"{a_dig.norm():>10.4f} {a_lrtt.norm():>10.4f} | "
              f"{b_dig.norm():>10.4f} {b_lrtt.norm():>10.4f} | "
              f"{out_diff:>10.6f}")

    # ---- Final detailed comparison ----
    print(f"\n[FINAL] Detailed weight comparison:")
    print(f"  A diff norm: {(a_dig - a_lrtt).norm():.6f}")
    print(f"  B diff norm: {(b_dig - b_lrtt).norm():.6f}")
    print(f"  A relative diff: {(a_dig - a_lrtt).norm() / max(a_dig.norm(), 1e-10):.6f}")
    print(f"  B relative diff: {(b_dig - b_lrtt).norm() / max(b_dig.norm(), 1e-10):.6f}")

    # ---- Check update mechanism ----
    print(f"\n[ANALYSIS] Update mechanism comparison:")
    print(f"  Digital LoRA: standard autograd -> Adam optimizer")
    print(f"  LRTT FP:      tile.update(x, d) -> w -= lr * d^T @ x (SGD-like)")
    print(f"  Key diff: Digital uses Adam (m/sqrt(v)), LRTT uses direct outer product")

    # Check: Does LRTT tile.update do SGD or go through AnalogAdam?
    print(f"\n[CHECK] LRTT tile_a type: {type(ctrl.tile_a)}")
    print(f"  tile_a.tile type: {type(ctrl.tile_a.tile) if hasattr(ctrl.tile_a, 'tile') else 'N/A'}")

    # Check what AnalogAdam actually does
    print(f"\n  AnalogAdam param groups:")
    for i, pg in enumerate(lrtt_optimizer.param_groups):
        names = [n for n, p in analog_model.named_parameters() if any(p is pp for pp in [pg['params'][0]] if len(pg['params']) > 0)]
        print(f"    Group {i}: lr={pg['lr']}, #params={len(pg['params'])}")


if __name__ == "__main__":
    main()

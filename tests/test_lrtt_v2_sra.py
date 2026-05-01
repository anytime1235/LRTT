# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Unit tests for SRA-LRTT-v2 (Stochastic Reset-Anchor).

Test sizes intentionally small and all tiles use FloatingPointDevice so the
analytical relations hold to float32 precision. Tests follow the SRA-LRTT-v2
implementation guide §10.

The default anchor source for these tests is 'explicit_gaussian' because
FloatingPointDevice has no reset/reset_std and would yield a deterministic zero
under 'reset_columns'. The 'reset_columns' source is exercised separately by
the smoke MNIST script with LinearStepDevice tiles.
"""

import math

import pytest
import torch

from aihwkit.simulator.configs.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.devices import FloatingPointDevice
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

D_SIZE = 8
X_SIZE = 4
RANK = 2
BATCH = 3


def _make_sra_tile(
    *,
    transfer_lr: float = 1.0,
    transfer_every: int = 4,
    anchor_source: str = "explicit_gaussian",
    cap_rho: float = 1.0,
    cap_compensate_transfer: bool = True,
    sra_seed: int = 1234,
    sra_resample_on_transfer: bool = True,
    sra_reset_b_on_transfer: bool = True,
):
    cfg = PythonLRTTDevice(
        rank=RANK,
        update_mode="stochastic_reset_anchor",
        transfer_method="stochastic_anchor",
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        forward_inject=False,
        a_init_mode="zero",
        b_init_mode="zero",
        cap_rho=cap_rho,
        cap_compensate_transfer=cap_compensate_transfer,
        sra_anchor_source=anchor_source,
        sra_seed=sra_seed,
        sra_resample_on_transfer=sra_resample_on_transfer,
        sra_reset_b_on_transfer=sra_reset_b_on_transfer,
        unit_cell_devices=[FloatingPointDevice() for _ in range(3)],
    )
    rpu = UnitCellRPUConfig(device=cfg)
    return LRTTSimulatorTile(d_size=D_SIZE, x_size=X_SIZE, rpu_config=rpu)


# ---------------------------------------------------------------------------
# Test 1: anchor shape and RMS normalization
# ---------------------------------------------------------------------------

def test_sra_anchor_shape_and_rms():
    torch.manual_seed(0)
    tile = _make_sra_tile(anchor_source="explicit_gaussian")
    ctrl = tile.controller

    A = ctrl.sra_anchor_scaled
    assert A is not None, "anchor must be cached after tile post-init"
    assert tuple(A.shape) == (D_SIZE, RANK), f"unexpected anchor shape {A.shape}"

    rms = float((A * A).mean().sqrt().item())
    target = 1.0 / math.sqrt(RANK)
    assert abs(rms - target) < 1e-3, (
        f"scaled anchor RMS {rms:.6f} should be ~{target:.6f} (1/sqrt(rank))"
    )

    # Sanity: gain is finite and the raw RMS was logged.
    assert ctrl.sra_anchor_gain > 0
    assert ctrl.sra_anchor_rms_raw > 0


# ---------------------------------------------------------------------------
# Test 2: B update equals projected gradient (analytical)
# ---------------------------------------------------------------------------

def test_sra_b_update_equals_projected_gradient():
    torch.manual_seed(7)
    tile = _make_sra_tile(anchor_source="explicit_gaussian")
    ctrl = tile.controller
    # Use the device of the simulator tile_b, not ctrl.device — the controller
    # stores its anchor on ctrl.device (which may be CUDA), but the underlying
    # FloatingPoint simulator tile lives on CPU, and tile_b.update requires CPU
    # tensors. Anchor is moved on demand inside the controller.
    dev = ctrl.tile_b.get_weights()[0].device

    A = ctrl.sra_anchor_scaled.detach().clone().to(dev)  # [d_size, rank]

    # Force B := 0 (LRTTSimulatorTile already sets it; assert it).
    B0 = ctrl.tile_b.get_weights()[0]
    assert torch.norm(B0).item() < 1e-6, "B must start at zero"

    x = torch.randn(BATCH, X_SIZE, device=dev)
    d = torch.randn(BATCH, D_SIZE, device=dev)
    lr = 0.5

    ctrl._ab_weight_update_stochastic_reset_anchor(x, d, lr)

    B_actual = ctrl.tile_b.get_weights()[0].to(dev)
    # tile.update(x, d_proj) implements B += -lr * d_proj.T @ x with d_proj = d @ A.
    # So B_actual ≈ -lr * (d @ A).T @ x = -lr * A.T @ d.T @ x.
    B_expected = -lr * (d @ A).t() @ x

    assert torch.allclose(B_actual, B_expected, atol=1e-5, rtol=1e-4), (
        f"B mismatch: actual_norm={B_actual.norm():.6f} expected_norm={B_expected.norm():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: transfer sign — C += transfer_lr * A @ B
# ---------------------------------------------------------------------------

def test_sra_transfer_sign():
    torch.manual_seed(11)
    tile = _make_sra_tile(
        anchor_source="explicit_gaussian",
        # disable resample so the post-transfer anchor still equals A_known.
        sra_resample_on_transfer=False,
        sra_reset_b_on_transfer=False,
    )
    ctrl = tile.controller
    # Use the device of the simulator tile_b, not ctrl.device — the controller
    # stores its anchor on ctrl.device (which may be CUDA), but the underlying
    # FloatingPoint simulator tile lives on CPU, and tile_b.update requires CPU
    # tensors. Anchor is moved on demand inside the controller.
    dev = ctrl.tile_b.get_weights()[0].device

    # Inject a known A and B (on the controller's device).
    A_known = torch.randn(D_SIZE, RANK, device=dev)
    B_known = torch.randn(RANK, X_SIZE, device=dev)
    # Update controller cache to the known anchor (gain folded in already).
    ctrl.sra_anchor_raw = A_known.clone()
    ctrl.sra_anchor_scaled = A_known.clone()
    ctrl.sra_anchor_gain = 1.0
    # Set the underlying B tile.
    ctrl.tile_b.set_weights(B_known)

    C0 = ctrl.tile_c.get_weights()[0].to(dev).clone()
    ctrl._ab_weight_transfer_stochastic_anchor()
    C1 = ctrl.tile_c.get_weights()[0].to(dev)

    # Default cap_rho=1 with cap_compensate_transfer=True yields kappa=1 at any tau.
    expected = C0 + ctrl.transfer_lr * (A_known @ B_known)
    assert torch.allclose(C1, expected, atol=1e-4, rtol=1e-3), (
        f"transfer sign/magnitude mismatch: |delta_actual|={torch.norm(C1-C0):.6f}, "
        f"|delta_expected|={torch.norm(expected-C0):.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4: anchor changes after transfer (resample)
# ---------------------------------------------------------------------------

def test_sra_anchor_changes_after_transfer():
    torch.manual_seed(13)
    tile = _make_sra_tile(
        anchor_source="explicit_gaussian",
        sra_resample_on_transfer=True,
        sra_reset_b_on_transfer=True,
    )
    ctrl = tile.controller

    A0 = ctrl.sra_anchor_scaled.clone()
    cycle0 = ctrl.sra_cycle

    ctrl._ab_weight_transfer_stochastic_anchor()

    A1 = ctrl.sra_anchor_scaled
    assert not torch.allclose(A0, A1), "anchor must be resampled after transfer"
    assert ctrl.sra_cycle == cycle0 + 1, (
        f"sra_cycle should increment after resample: {cycle0} -> {ctrl.sra_cycle}"
    )


# ---------------------------------------------------------------------------
# Test 5: B resets after transfer
# ---------------------------------------------------------------------------

def test_sra_b_resets_after_transfer():
    torch.manual_seed(17)
    tile = _make_sra_tile(anchor_source="explicit_gaussian")
    ctrl = tile.controller
    # Use the device of the simulator tile_b, not ctrl.device — the controller
    # stores its anchor on ctrl.device (which may be CUDA), but the underlying
    # FloatingPoint simulator tile lives on CPU, and tile_b.update requires CPU
    # tensors. Anchor is moved on demand inside the controller.
    dev = ctrl.tile_b.get_weights()[0].device

    # Run a few updates so B accumulates a non-trivial residual.
    x = torch.randn(BATCH, X_SIZE, device=dev)
    d = torch.randn(BATCH, D_SIZE, device=dev)
    for _ in range(3):
        ctrl._ab_weight_update_stochastic_reset_anchor(x, d, lr=0.1)

    B_pre = ctrl.tile_b.get_weights()[0]
    assert B_pre.norm().item() > 1e-6, "B should be non-zero before transfer"

    ctrl._ab_weight_transfer_stochastic_anchor()

    B_post = ctrl.tile_b.get_weights()[0]
    assert B_post.norm().item() < 1e-6, (
        f"B must be ~0 after transfer; got |B|={B_post.norm():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 6: forward_inject is rejected for SRA mode
# ---------------------------------------------------------------------------

def test_sra_no_forward_inject_static_validation():
    """Static config validation must reject SRA + forward_inject=True."""
    with pytest.raises(ValueError, match="forward_inject"):
        PythonLRTTDevice(
            rank=RANK,
            update_mode="stochastic_reset_anchor",
            transfer_method="stochastic_anchor",
            forward_inject=True,
            unit_cell_devices=[FloatingPointDevice() for _ in range(3)],
        )


def test_sra_no_forward_inject_runtime_guard():
    """Runtime guard must reject post-construction enabling of forward_inject."""
    tile = _make_sra_tile(anchor_source="explicit_gaussian")
    ctrl = tile.controller
    # Use the device of the simulator tile_b, not ctrl.device — the controller
    # stores its anchor on ctrl.device (which may be CUDA), but the underlying
    # FloatingPoint simulator tile lives on CPU, and tile_b.update requires CPU
    # tensors. Anchor is moved on demand inside the controller.
    dev = ctrl.tile_b.get_weights()[0].device
    # Force the runtime flag on (config-level validation is bypassed here).
    ctrl.forward_inject_enabled = True
    x = torch.randn(BATCH, X_SIZE, device=dev)
    with pytest.raises(ValueError, match="SRA-LRTT-v2"):
        ctrl.forward_inject(x)


# ---------------------------------------------------------------------------
# Test 7: projection coverage diagnostic — 1/Q Σ A_q A_q^T → I
# ---------------------------------------------------------------------------

def test_sra_projection_coverage_diagnostic():
    """For explicit_gaussian anchors with target_rms=1/sqrt(rank), the empirical
    1/Q * Σ A_q A_q^T should approach I_d as Q grows (variance ~ 1/(Q*rank)).

    We assert a loose monotonic decrease of the Frobenius residual rather than a
    strict bound to keep the test deterministic on small Q.
    """
    torch.manual_seed(0)
    tile = _make_sra_tile(anchor_source="explicit_gaussian")
    ctrl = tile.controller
    # Use the device of the simulator tile_b, not ctrl.device — the controller
    # stores its anchor on ctrl.device (which may be CUDA), but the underlying
    # FloatingPoint simulator tile lives on CPU, and tile_b.update requires CPU
    # tensors. Anchor is moved on demand inside the controller.
    dev = ctrl.tile_b.get_weights()[0].device

    def avg_AAT(Q: int) -> float:
        acc = torch.zeros(D_SIZE, D_SIZE, device=dev)
        for _ in range(Q):
            ctrl._reset_sra_anchor()
            A = ctrl.sra_anchor_scaled.to(dev)
            acc += A @ A.t()
        avg = acc / Q
        return float((avg - torch.eye(D_SIZE, device=dev)).norm().item())

    err_low = avg_AAT(16)
    err_high = avg_AAT(256)
    assert err_high < err_low, (
        f"projection coverage residual should shrink with Q: Q=16 -> {err_low:.4f}, "
        f"Q=256 -> {err_high:.4f}"
    )


# ---------------------------------------------------------------------------
# Smoke: existing v1 + selector_v2 modes must still construct.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode,method",
    [
        ("lora", "onehot"),
        ("reconstruction", "onehot"),
        ("selector_reconstruction", "blockwise"),
    ],
)
def test_sra_legacy_modes_still_construct(mode, method):
    """Adding SRA must not break construction of legacy modes."""
    cfg = PythonLRTTDevice(
        rank=RANK,
        update_mode=mode,
        transfer_method=method,
        forward_inject=False,
        unit_cell_devices=[FloatingPointDevice() for _ in range(3)],
    )
    rpu = UnitCellRPUConfig(device=cfg)
    tile = LRTTSimulatorTile(d_size=D_SIZE, x_size=X_SIZE, rpu_config=rpu)
    assert tile.controller.update_mode == mode
    assert tile.controller.transfer_method == method
    assert tile.controller._is_sra_v2() is False

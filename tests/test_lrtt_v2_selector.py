# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Unit tests for LRTT-v2 selector reconstruction + blockwise transfer.

Test sizes intentionally small (d_size=8, x_size=4, rank/block_size=2, batch=3) and
all tiles use FloatingPointDevice so the analytical relations hold to float32
precision. Tests follow md guide §12 + §21 acceptance criteria.
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
BLOCK = 2  # == rank
BATCH = 3


def _make_v2_tile(
    *,
    transfer_lr: float = 1.0,
    transfer_every: int = 4,
    cap_rho: float = 1.0,
    cap_compensate_transfer: bool = True,
    selector_policy: str = "cyclic",
    cap_max_rms: float = None,
    cap_monitor_every: int = 0,
    cap_soft_clip: bool = True,
    selector_seed: int = 0,
):
    cfg = PythonLRTTDevice(
        rank=BLOCK,
        update_mode="selector_reconstruction",
        transfer_method="blockwise",
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        forward_inject=False,
        b_init_mode="zero",
        selector_policy=selector_policy,
        selector_seed=selector_seed,
        cap_rho=cap_rho,
        cap_compensate_transfer=cap_compensate_transfer,
        cap_max_rms=cap_max_rms,
        cap_monitor_every=cap_monitor_every,
        cap_soft_clip=cap_soft_clip,
        unit_cell_devices=[FloatingPointDevice() for _ in range(3)],
    )
    rpu = UnitCellRPUConfig(device=cfg)
    return LRTTSimulatorTile(d_size=D_SIZE, x_size=X_SIZE, rpu_config=rpu)


def _make_v1_tile(*, update_mode: str = "lora", transfer_method: str = "onehot"):
    cfg = PythonLRTTDevice(
        rank=BLOCK,
        update_mode=update_mode,
        transfer_method=transfer_method,
        transfer_every=4,
        transfer_lr=1.0,
        forward_inject=False,
        unit_cell_devices=[FloatingPointDevice() for _ in range(3)],
    )
    rpu = UnitCellRPUConfig(device=cfg)
    return LRTTSimulatorTile(d_size=D_SIZE, x_size=X_SIZE, rpu_config=rpu)


# ---------------------------------------------------------------------------
# Test 1: B update equals selected gradient rows
# ---------------------------------------------------------------------------

def test_1_b_update_matches_selected_gradient():
    torch.manual_seed(42)
    tile = _make_v2_tile(cap_rho=1.0)
    ctrl = tile.controller

    # Force selector_indices to a known set so we can verify analytically.
    idx = torch.tensor([1, 3], dtype=torch.long, device=ctrl.device)
    ctrl.selector_indices = idx
    ctrl.selector_valid_mask = torch.ones(BLOCK, device=ctrl.device, dtype=ctrl.dtype)

    x = torch.randn(BATCH, X_SIZE)
    d = torch.randn(BATCH, D_SIZE)
    lr = 0.1

    # Pre-state of B (should be zero after tile init)
    B_pre = ctrl._read_b_buffer()
    assert torch.allclose(B_pre, torch.zeros_like(B_pre), atol=1e-6)

    ctrl._ab_weight_update_selector_reconstruction(x, d, lr=lr)

    B_post = ctrl._read_b_buffer()
    expected = -lr * d[:, idx].t() @ x  # tile.update => B += -lr * d_sel.T @ x
    assert torch.allclose(B_post, expected, atol=1e-5), (
        f"B mismatch. got=\n{B_post}\nexpected=\n{expected}"
    )


# ---------------------------------------------------------------------------
# Test 2: Blockwise transfer updates only selected C rows
# ---------------------------------------------------------------------------

def test_2_blockwise_transfer_updates_only_selected_rows():
    torch.manual_seed(7)
    tile = _make_v2_tile(transfer_lr=1.0, cap_rho=1.0)
    ctrl = tile.controller

    # Pin selector to a known block.
    idx = torch.tensor([2, 5], dtype=torch.long, device=ctrl.device)
    ctrl.selector_indices = idx
    ctrl.selector_valid_mask = torch.ones(BLOCK, device=ctrl.device, dtype=ctrl.dtype)

    # Inject a known B (set tile_b directly).
    B_known = torch.randn(BLOCK, X_SIZE)
    tile.tile_b.set_weights(B_known)

    # Snapshot C before.
    C_before = tile.tile_c.get_weights()[0].clone()

    ctrl._ab_weight_transfer_blockwise()

    C_after = tile.tile_c.get_weights()[0]

    # Transfer is C[idx] += transfer_lr * B (with transfer_lr=1.0).
    expected_after = C_before.clone()
    expected_after[idx] = C_before[idx] + 1.0 * B_known
    assert torch.allclose(C_after, expected_after, atol=1e-5), (
        f"C transfer mismatch: max diff = {(C_after - expected_after).abs().max()}"
    )

    # Rows NOT in idx must be unchanged.
    mask = torch.ones(D_SIZE, dtype=torch.bool)
    mask[idx] = False
    assert torch.allclose(C_after[mask], C_before[mask], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: B reset after transfer
# ---------------------------------------------------------------------------

def test_3_b_zero_after_blockwise_transfer():
    torch.manual_seed(11)
    tile = _make_v2_tile()
    ctrl = tile.controller

    # Inject nonzero B then transfer.
    B_known = torch.randn(BLOCK, X_SIZE)
    tile.tile_b.set_weights(B_known)
    assert tile.controller._read_b_buffer().norm() > 0.1

    ctrl._ab_weight_transfer_blockwise()

    B_after = ctrl._read_b_buffer()
    assert torch.allclose(B_after, torch.zeros_like(B_after), atol=1e-6), (
        f"B should be zero after transfer, norm={B_after.norm()}"
    )


# ---------------------------------------------------------------------------
# Test 4: Selector advances and covers all rows in one cycle
# ---------------------------------------------------------------------------

def test_4_selector_cyclic_full_coverage():
    """Cyclic policy on d_size=8, b=2 should produce blocks
    [0,1] -> [2,3] -> [4,5] -> [6,7] -> [0,1] (cycle increment)."""
    tile = _make_v2_tile(selector_policy="cyclic")
    ctrl = tile.controller

    seen_indices = []
    for _ in range(D_SIZE // BLOCK):
        seen_indices.append(ctrl.selector_indices.clone().tolist())
        ctrl._advance_selector()

    flat = sorted(i for block in seen_indices for i in block)
    assert flat == list(range(D_SIZE)), f"coverage failed: {seen_indices}"
    assert ctrl.selector_cycle == 1, f"selector_cycle should be 1, got {ctrl.selector_cycle}"


def test_4b_selector_shuffled_cycle_full_coverage():
    """Shuffled-cycle should also cover all rows in one cycle."""
    tile = _make_v2_tile(selector_policy="shuffled_cycle", selector_seed=123)
    ctrl = tile.controller

    seen_indices = []
    for _ in range(D_SIZE // BLOCK):
        seen_indices.append(ctrl.selector_indices.clone().tolist())
        ctrl._advance_selector()

    flat = sorted(i for block in seen_indices for i in block)
    assert flat == list(range(D_SIZE))
    assert ctrl.selector_cycle == 1


# ---------------------------------------------------------------------------
# Test 5: Full-cycle equivalence to projected SGD with cap_rho=1
# ---------------------------------------------------------------------------

def test_5_full_cycle_equivalence_to_projected_sgd():
    """With cap_rho=1.0 and constant (x, d), one full selector cycle should
    accumulate the full -lr_b * lr_tr * D^T X gradient into C
    (block-coordinate SGD partition)."""
    torch.manual_seed(2026)
    tile = _make_v2_tile(
        transfer_lr=1.0, transfer_every=1, cap_rho=1.0, selector_policy="cyclic"
    )
    ctrl = tile.controller

    x = torch.randn(BATCH, X_SIZE)
    d = torch.randn(BATCH, D_SIZE)
    lr_b = 0.1

    C_before = tile.tile_c.get_weights()[0].clone()

    n_blocks = D_SIZE // BLOCK
    for _ in range(n_blocks):
        ctrl._ab_weight_update_selector_reconstruction(x, d, lr=lr_b)
        # transfer_every=1 means we transfer immediately (cap_rho=1, no leak).
        ctrl._ab_weight_transfer_blockwise()

    C_after = tile.tile_c.get_weights()[0]

    # Each block contributes -lr_b * d_block.T @ x to its rows of C.
    # Aggregated over the full cycle this equals -lr_b * d.T @ x.
    G = d.t() @ x  # [d_size, x_size]
    expected_delta = -1.0 * lr_b * G  # transfer_lr=1.0
    actual_delta = C_after - C_before
    assert torch.allclose(actual_delta, expected_delta, atol=1e-4), (
        f"max diff = {(actual_delta - expected_delta).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Test 6: Capacitor leak compensation
# ---------------------------------------------------------------------------

def test_6_capacitor_leak_envelope_without_compensation():
    """Without compensation, a leaky B (cap_rho<1) accumulating a constant
    gradient over tau steps should reach (1-rho^tau)/(1-rho) * (-lr*g) instead
    of tau * (-lr*g)."""
    torch.manual_seed(0)
    rho = 0.9
    tau = 8
    tile = _make_v2_tile(
        transfer_every=tau, cap_rho=rho, cap_compensate_transfer=False
    )
    ctrl = tile.controller

    # Pin selector to constant block so the same rows accumulate every step.
    idx = torch.tensor([0, 1], dtype=torch.long, device=ctrl.device)
    ctrl.selector_indices = idx
    ctrl.selector_valid_mask = torch.ones(BLOCK, device=ctrl.device, dtype=ctrl.dtype)

    x = torch.randn(BATCH, X_SIZE)
    d = torch.randn(BATCH, D_SIZE)
    lr_b = 0.1

    for _ in range(tau):
        ctrl._ab_weight_update_selector_reconstruction(x, d, lr=lr_b)

    B_final = ctrl._read_b_buffer()

    # Per-step contribution (tile.update => -lr * d_sel.T @ x). With leakage:
    # B_t = sum_{k=0..t-1} rho^k * (-lr * d_sel.T @ x)
    g_block = -lr_b * d[:, idx].t() @ x
    geom = (1.0 - rho ** tau) / (1.0 - rho)
    expected_B = geom * g_block

    assert torch.allclose(B_final, expected_B, atol=5e-4), (
        f"leak envelope mismatch: max diff={(B_final - expected_B).abs().max()}"
    )

    # kappa_rho transfer gain should reproduce tau when applied.
    kappa = ctrl._cap_transfer_gain()  # cap_compensate_transfer=False -> 1.0
    assert kappa == 1.0


def test_6b_capacitor_compensation_recovers_unbiased_transfer():
    """With cap_compensate_transfer=True and constant gradient, transfer
    magnitude C += transfer_lr * kappa(tau) * B should equal the unbiased
    full-tau accumulation transfer_lr * tau * (-lr * G_block)."""
    torch.manual_seed(99)
    rho = 0.9
    tau = 8
    transfer_lr = 1.0
    tile = _make_v2_tile(
        transfer_lr=transfer_lr,
        transfer_every=tau,
        cap_rho=rho,
        cap_compensate_transfer=True,
        selector_policy="cyclic",
    )
    ctrl = tile.controller

    # Pin selector
    idx = torch.tensor([0, 1], dtype=torch.long, device=ctrl.device)
    ctrl.selector_indices = idx
    ctrl.selector_valid_mask = torch.ones(BLOCK, device=ctrl.device, dtype=ctrl.dtype)

    x = torch.randn(BATCH, X_SIZE)
    d = torch.randn(BATCH, D_SIZE)
    lr_b = 0.05

    C_before = tile.tile_c.get_weights()[0].clone()

    for _ in range(tau):
        ctrl._ab_weight_update_selector_reconstruction(x, d, lr=lr_b)
    # Single transfer at end of window
    # Manually trigger to bypass selector advance, then re-pin.
    # _ab_weight_transfer_blockwise will reset+advance.
    ctrl._ab_weight_transfer_blockwise()

    C_after = tile.tile_c.get_weights()[0]
    # Unbiased target: C[idx] += transfer_lr * tau * (-lr_b * d_sel.T @ x)
    g_block = -lr_b * d[:, idx].t() @ x
    expected_delta_block = transfer_lr * tau * g_block

    actual_delta = (C_after - C_before)[idx]
    assert torch.allclose(actual_delta, expected_delta_block, atol=2e-3), (
        f"compensation mismatch: max={ (actual_delta - expected_delta_block).abs().max() }"
    )


# ---------------------------------------------------------------------------
# Test 7: cap soft-clip triggers when RMS exceeds cap_max_rms
# ---------------------------------------------------------------------------

def test_7_cap_soft_clip_engages_above_threshold():
    """With cap_monitor_every=1 and cap_max_rms small, B should be soft-clipped
    on the next update so RMS does not exceed the threshold."""
    torch.manual_seed(0)
    tile = _make_v2_tile(
        cap_rho=1.0,
        cap_monitor_every=1,
        cap_max_rms=0.1,
        cap_soft_clip=True,
    )
    ctrl = tile.controller

    # Pin selector so we accumulate into the same block.
    idx = torch.tensor([0, 1], dtype=torch.long, device=ctrl.device)
    ctrl.selector_indices = idx
    ctrl.selector_valid_mask = torch.ones(BLOCK, device=ctrl.device, dtype=ctrl.dtype)

    # Inject a large B directly to trigger clipping at the next monitor tick.
    B_big = 5.0 * torch.ones(BLOCK, X_SIZE)
    tile.tile_b.set_weights(B_big)
    assert ctrl._read_b_buffer().abs().mean() > 1.0

    # An update with tiny lr should not change B much, but the monitor will run
    # and clip B because RMS is above cap_max_rms.
    x = torch.zeros(BATCH, X_SIZE)
    d = torch.zeros(BATCH, D_SIZE)
    ctrl._ab_weight_update_selector_reconstruction(x, d, lr=0.0)

    B_after = ctrl._read_b_buffer()
    rms_after = float(torch.sqrt((B_after * B_after).mean()).item())
    # After clip, RMS should be roughly cap_max_rms (some tolerance for sqrt).
    assert rms_after <= 0.11, f"clip failed: rms_after={rms_after}"


# ---------------------------------------------------------------------------
# Test 8: v1 modes still functional (smoke)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "update_mode,transfer_method",
    [
        ("lora", "onehot"),
        ("reconstruction", "onehot"),
        ("reconstruction", "set"),
    ],
)
def test_8_v1_modes_smoke(update_mode, transfer_method):
    """Existing v1 modes must continue to construct, update once, and transfer once."""
    tile = _make_v1_tile(update_mode=update_mode, transfer_method=transfer_method)
    ctrl = tile.controller

    x = torch.randn(BATCH, X_SIZE)
    d = torch.randn(BATCH, D_SIZE)
    ctrl.ab_weight_update(x, d, lr=0.1)
    ctrl.ab_weight_transfer()
    # Just check it ran without errors.
    assert ctrl.num_transfers == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

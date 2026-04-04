#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core cost model for AIMC BERT-base fine-tuning comparison.

Implements the common-path + method-specific-delta cost decomposition:
    T_step^(m) = T_common + DeltaT^(m)
    E_step^(m) = E_common + DeltaE^(m)

Three methods:
  1. on_device_digital_lora_training
  2. lrtt_6t1c_training
  3. tikitaka_fullrank_6t1c_training

The common term (T_common, E_common) is identical across all three methods.
Method-specific deltas are computed from explicit event counting derived from
the lrtt_controller.py and optuna_bert_squad_tiki.py code semantics.
"""

import math
import csv
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

from extract_layer_inventory import (
    LayerSpec,
    build_layer_inventory,
    get_targeted_layers,
    compute_adapter_tile_counts,
    tile_count,
    DEFAULT_TILE_SIZE,
)


# =============================================================================
# BERT-base constants
# =============================================================================

N_ENCODER_LAYERS = 12
N_HEADS = 12
D_MODEL = 768
D_HEAD = D_MODEL // N_HEADS  # 64
D_FF = 3072
BATCH_SIZE = 48


# =============================================================================
# Hardware Parameters (loaded from YAML or defaults)
# =============================================================================

@dataclass
class HardwareParams:
    """Hardware timing and energy parameters."""

    # Tile geometry
    tile_size: int = DEFAULT_TILE_SIZE
    io_bits: int = 8

    # AIMC tile MVM latency / energy (per tile per vector)
    t_tile_ns: float = 256.0        # Sensitivity sweep default
    e_tile_pj: Optional[float] = None

    # Pulsed update latency / energy (per tile per sample)
    t_update_ns: Optional[float] = None
    e_update_pj: Optional[float] = None

    # Digital ops
    t_digital_ns_per_flop: Optional[float] = None
    e_digital_pj_per_flop: Optional[float] = None

    # Common backward multiplier
    k_common_bwd: float = 2.5

    # Digital LoRA specific
    t_lora_gemm_ns_per_flop: Optional[float] = None
    e_lora_gemm_pj_per_flop: Optional[float] = None
    t_lora_opt_ns_per_param: Optional[float] = None
    e_lora_opt_pj_per_param: Optional[float] = None
    lora_overlap_with_base: bool = False

    # LRTT specific
    t_ab_read_ns: Optional[float] = None       # If None, uses t_tile_ns
    e_ab_read_pj: Optional[float] = None
    t_ab_update_ns: Optional[float] = None      # If None, uses t_update_ns
    e_ab_update_pj: Optional[float] = None
    t_rank_xfer_read_ns: Optional[float] = None
    e_rank_xfer_read_pj: Optional[float] = None
    t_rank_xfer_write_ns: Optional[float] = None
    e_rank_xfer_write_pj: Optional[float] = None

    # Tiki-Taka specific
    t_u_read_ns: Optional[float] = None         # If None, uses t_tile_ns
    e_u_read_pj: Optional[float] = None
    t_u_update_ns: Optional[float] = None        # If None, uses t_update_ns
    e_u_update_pj: Optional[float] = None
    t_col_xfer_read_ns: Optional[float] = None
    e_col_xfer_read_pj: Optional[float] = None
    t_col_xfer_write_ns: Optional[float] = None
    e_col_xfer_write_pj: Optional[float] = None

    # Shared periphery
    shared_periphery_serialize: bool = True
    t_periph_switch_ns: Optional[float] = None
    e_periph_switch_pj: Optional[float] = None


@dataclass
class ScenarioConfig:
    """Configuration for one cost-model scenario."""
    method: str                         # "digital_lora", "lrtt_6t1c", "tikitaka_6t1c"
    target: str                         # "attention", "ffn", "all"
    gamma: int                          # 0 or 1
    rank: int                           # LoRA / LR-TT rank (ignored for tikitaka)
    transfer_every: int = 4             # Transfer frequency (method-specific default)
    units_in_mbatch: bool = False       # Whether transfer_every counts samples
    num_reads: int = 1                  # num_reads per transfer (LRTT)
    batch_size: int = BATCH_SIZE
    forward_only: bool = False          # If True: zero out backward, update, transfer costs


# =============================================================================
# Cost Result
# =============================================================================

@dataclass
class StepCostResult:
    """Cost breakdown for one training step."""
    # Common path
    T_base_fwd_ns: float = 0.0
    T_base_bwd_ns: float = 0.0
    T_other_ns: float = 0.0
    T_common_ns: float = 0.0
    E_base_fwd_pj: float = 0.0
    E_base_bwd_pj: float = 0.0
    E_other_pj: float = 0.0
    E_common_pj: float = 0.0

    # Method-specific delta
    DeltaT_ns: float = 0.0
    DeltaE_pj: float = 0.0

    # Delta sub-components
    DeltaT_adapter_fwd_ns: float = 0.0
    DeltaT_adapter_bwd_ns: float = 0.0
    DeltaT_adapter_opt_ns: float = 0.0
    DeltaT_transfer_ns: float = 0.0
    DeltaE_adapter_fwd_pj: float = 0.0
    DeltaE_adapter_bwd_pj: float = 0.0
    DeltaE_adapter_opt_pj: float = 0.0
    DeltaE_transfer_pj: float = 0.0

    # Totals
    T_step_ns: float = 0.0
    E_step_pj: float = 0.0

    # Op counts
    base_mvm_events: int = 0
    adapter_mvm_events: int = 0
    adapter_update_events: int = 0
    transfer_events: int = 0

    # Digital ops
    digital_flops: int = 0
    digital_opt_params: int = 0

    # Missing data flags
    missing_params: List[str] = field(default_factory=list)


# =============================================================================
# Common Path Model
# =============================================================================

def compute_common_path(
    inventory: List[LayerSpec],
    hw: HardwareParams,
    S_pad: int,
    batch_size: int,
    forward_only: bool = False,
) -> StepCostResult:
    """Compute common-path cost shared by all three methods.

    This function returns identical results regardless of method, ensuring
    fair comparison.

    Args:
        inventory: Full 72-layer encoder inventory.
        hw: Hardware parameters.
        S_pad: Padded sequence length for this batch.
        batch_size: Batch size.
        forward_only: If True, set backward and other costs to zero.
    """
    result = StepCostResult()
    BT = batch_size * S_pad  # Total vectors (batch-tokens)

    # --- Base forward: sum over all 72 encoder linears ---
    t_fwd = 0.0
    e_fwd = 0.0
    mvm_events = 0

    for layer in inventory:
        # Latency: rows parallel, cols serial
        t_layer = BT * layer.n_tile_cols * hw.t_tile_ns
        t_fwd += t_layer

        # Energy: every tile activation
        n_events = BT * layer.n_tiles
        mvm_events += n_events
        if hw.e_tile_pj is not None:
            e_fwd += n_events * hw.e_tile_pj

    result.T_base_fwd_ns = t_fwd
    result.E_base_fwd_pj = e_fwd
    result.base_mvm_events = mvm_events

    if hw.e_tile_pj is None:
        result.missing_params.append("e_tile_pj")

    # --- Base backward: k * forward (zero if forward_only) ---
    if forward_only:
        result.T_base_bwd_ns = 0.0
        result.E_base_bwd_pj = 0.0
    else:
        result.T_base_bwd_ns = hw.k_common_bwd * t_fwd
        result.E_base_bwd_pj = hw.k_common_bwd * e_fwd

    # --- Other digital ops (per encoder layer) ---
    t_other = 0.0
    e_other = 0.0

    if hw.t_digital_ns_per_flop is not None:
        for _ in range(N_ENCODER_LAYERS):
            # Multi-head attention: QK^T + softmax + attn*V
            qkt_flops = batch_size * N_HEADS * S_pad * S_pad * D_HEAD * 2
            softmax_ops = batch_size * N_HEADS * S_pad * S_pad * 5  # exp, sum, div, etc.
            attn_v_flops = batch_size * N_HEADS * S_pad * S_pad * D_HEAD * 2
            # LayerNorm (x2)
            ln_ops = 2 * BT * D_MODEL * 5  # mean, var, normalize, scale, shift
            # GELU
            gelu_ops = BT * D_FF * 5  # approx ops for GELU
            # Residual adds (x2)
            res_ops = 2 * BT * D_MODEL

            layer_flops = qkt_flops + softmax_ops + attn_v_flops + ln_ops + gelu_ops + res_ops
            t_other += layer_flops * hw.t_digital_ns_per_flop

        if hw.e_digital_pj_per_flop is not None:
            e_other = t_other / hw.t_digital_ns_per_flop * hw.e_digital_pj_per_flop
    else:
        result.missing_params.append("t_digital_ns_per_flop")

    result.T_other_ns = t_other
    result.E_other_pj = e_other

    # --- Totals ---
    result.T_common_ns = result.T_base_fwd_ns + result.T_base_bwd_ns + result.T_other_ns
    result.E_common_pj = result.E_base_fwd_pj + result.E_base_bwd_pj + result.E_other_pj

    return result


# =============================================================================
# Digital LoRA Delta Model
# =============================================================================

def compute_digital_lora_delta(
    targeted: List[LayerSpec],
    hw: HardwareParams,
    cfg: ScenarioConfig,
    S_pad: int,
) -> StepCostResult:
    """Compute DeltaT/DeltaE for on_device_digital_lora_training.

    For each targeted layer W [M x N] with rank r:
    - Forward: 2 digital matmuls (XB^T then result*A^T)
    - Backward: 4 digital matmuls (dA, dZ, dB, dX_LoRA)
    - Optimizer: Adam update for A [M*r] and B [r*N] parameters

    LoRA forward:  Y_lora = A(Bx)  where A∈[M,r], B∈[r,N], x∈[BT,N]
    LoRA backward (4 GEMMs):
      dA       = dY^T @ Z        [M,BT] x [BT,r] → [M,r]      2*BT*M*r
      dZ       = dY @ A          [BT,M] x [M,r]  → [BT,r]      2*BT*M*r
      dB       = dZ^T @ X        [r,BT] x [BT,N] → [r,N]       2*BT*r*N
      dX_LoRA  = dZ @ B          [BT,r] x [r,N]  → [BT,N]      2*BT*r*N
    where Z = Bx ∈ [BT,r]
    """
    result = StepCostResult()
    BT = cfg.batch_size * S_pad  # Batch-tokens
    r = cfg.rank

    total_fwd_flops = 0
    total_bwd_flops = 0
    total_opt_params = 0

    for layer in targeted:
        M, N = layer.M, layer.N

        # Forward: X@B^T [BT,N]x[N,r] + result@A^T [BT,r]x[r,M]
        fwd_flops = 2 * BT * N * r + 2 * BT * r * M
        total_fwd_flops += fwd_flops

        # Backward (4 GEMMs, complete chain rule):
        # (1) dA      = dY^T @ Z:       [M,BT] x [BT,r]  = 2*BT*M*r
        # (2) dZ      = dY @ A:         [BT,M] x [M,r]    = 2*BT*M*r
        # (3) dB      = dZ^T @ X:       [r,BT] x [BT,N]   = 2*BT*r*N
        # (4) dX_LoRA = dZ @ B:         [BT,r] x [r,N]    = 2*BT*r*N
        bwd_flops = (2 * BT * M * r       # dA
                   + 2 * BT * M * r       # dZ
                   + 2 * BT * r * N       # dB
                   + 2 * BT * r * N)      # dX_LoRA
        total_bwd_flops += bwd_flops

        # Optimizer: Adam for A [M*r] + B [r*N]
        params = M * r + r * N
        total_opt_params += params

    # forward_only: zero out backward and optimizer
    if cfg.forward_only:
        total_bwd_flops = 0
        total_opt_params = 0

    result.digital_flops = total_fwd_flops + total_bwd_flops
    result.digital_opt_params = total_opt_params

    # Convert to time
    if hw.t_lora_gemm_ns_per_flop is not None:
        result.DeltaT_adapter_fwd_ns = total_fwd_flops * hw.t_lora_gemm_ns_per_flop
        result.DeltaT_adapter_bwd_ns = total_bwd_flops * hw.t_lora_gemm_ns_per_flop
    else:
        result.missing_params.append("t_lora_gemm_ns_per_flop")

    if hw.t_lora_opt_ns_per_param is not None:
        result.DeltaT_adapter_opt_ns = total_opt_params * 5 * hw.t_lora_opt_ns_per_param
    else:
        if not cfg.forward_only:
            result.missing_params.append("t_lora_opt_ns_per_param")

    # Convert to energy
    if hw.e_lora_gemm_pj_per_flop is not None:
        result.DeltaE_adapter_fwd_pj = total_fwd_flops * hw.e_lora_gemm_pj_per_flop
        result.DeltaE_adapter_bwd_pj = total_bwd_flops * hw.e_lora_gemm_pj_per_flop
    else:
        result.missing_params.append("e_lora_gemm_pj_per_flop")

    if hw.e_lora_opt_pj_per_param is not None:
        result.DeltaE_adapter_opt_pj = total_opt_params * 5 * hw.e_lora_opt_pj_per_param
    else:
        if not cfg.forward_only:
            result.missing_params.append("e_lora_opt_pj_per_param")

    # No transfer for digital LoRA
    result.DeltaT_transfer_ns = 0.0
    result.DeltaE_transfer_pj = 0.0

    # Totals
    result.DeltaT_ns = (result.DeltaT_adapter_fwd_ns +
                        result.DeltaT_adapter_bwd_ns +
                        result.DeltaT_adapter_opt_ns)
    result.DeltaE_pj = (result.DeltaE_adapter_fwd_pj +
                        result.DeltaE_adapter_bwd_pj +
                        result.DeltaE_adapter_opt_pj)

    return result


# =============================================================================
# LR-TT Delta Model
# =============================================================================

def compute_lrtt_delta(
    targeted: List[LayerSpec],
    hw: HardwareParams,
    cfg: ScenarioConfig,
    S_pad: int,
) -> StepCostResult:
    """Compute DeltaT/DeltaE for lrtt_6t1c_training.

    Derived from lrtt_controller.py:ab_weight_update_lora (lines 720-796).

    Per step per targeted layer:
      1. B projection read: tile_b.forward(x) -> MVM through B [rank x N]
      2. A projection read: tile_a.backward(d) -> MVM through A^T [rank x M]
      3. A tile update: tile_a.update(XB, d) -> pulsed update on A [M x rank]
      4. B tile update: tile_b.update(x, DA) -> pulsed update on B [rank x N]
      If gamma=1: visible forward through A,B path

    Transfer (amortized):
      Per event: r onehot reads from A + r onehot reads from B + r rank-1 updates to C
    """
    result = StepCostResult()
    BT = cfg.batch_size * S_pad
    r = cfg.rank

    # Resolve fallback latencies
    t_ab_read = hw.t_ab_read_ns if hw.t_ab_read_ns is not None else hw.t_tile_ns
    t_ab_update = hw.t_ab_update_ns if hw.t_ab_update_ns is not None else hw.t_update_ns

    e_ab_read = hw.e_ab_read_pj if hw.e_ab_read_pj is not None else hw.e_tile_pj
    e_ab_update = hw.e_ab_update_pj if hw.e_ab_update_pj is not None else hw.e_update_pj

    t_missing = t_ab_update is None
    e_tile_missing = e_ab_read is None
    e_update_missing = e_ab_update is None

    total_t_proj = 0.0
    total_t_update = 0.0
    total_t_gamma_fwd = 0.0
    total_t_transfer = 0.0
    total_e_proj = 0.0
    total_e_update = 0.0
    total_e_gamma_fwd = 0.0
    total_e_transfer = 0.0

    total_mvm_events = 0
    total_update_events = 0
    total_xfer_events = 0

    for layer in targeted:
        M, N = layer.M, layer.N
        atc = compute_adapter_tile_counts(layer, r, hw.tile_size)
        n_tiles_A = atc['n_tiles_A']
        n_tiles_B = atc['n_tiles_B']
        n_tiles_C = layer.n_tiles

        a_cols = atc['n_tiles_A_cols']
        a_rows = atc['n_tiles_A_rows']
        b_cols = atc['n_tiles_B_cols']
        b_rows = atc['n_tiles_B_rows']

        # --- Per-step projection MVMs (backward-path only: skip if forward_only) ---
        if not cfg.forward_only:
            t_b_fwd = BT * b_cols * t_ab_read
            t_a_bwd = BT * a_rows * t_ab_read
            total_t_proj += t_b_fwd + t_a_bwd
            mvm_evts = BT * (n_tiles_B + n_tiles_A)
            total_mvm_events += mvm_evts
            if not e_tile_missing:
                total_e_proj += mvm_evts * e_ab_read

        # --- Per-step pulsed updates (skip if forward_only) ---
        if not cfg.forward_only:
            if not t_missing:
                t_a_upd = BT * n_tiles_A * t_ab_update
                t_b_upd = BT * n_tiles_B * t_ab_update
                total_t_update += t_a_upd + t_b_upd
            else:
                result.missing_params.append("t_update_ns (for A/B)")

            upd_evts = BT * (n_tiles_A + n_tiles_B)
            total_update_events += upd_evts
            if not e_update_missing:
                total_e_update += upd_evts * e_ab_update

        # --- gamma=1: visible forward through A,B path ---
        if cfg.gamma == 1:
            # B.forward(x) + A.forward(Bx)
            t_gamma_b = BT * b_cols * t_ab_read
            t_gamma_a = BT * a_cols * t_ab_read
            total_t_gamma_fwd += t_gamma_b + t_gamma_a
            gamma_mvm = BT * (n_tiles_B + n_tiles_A)
            total_mvm_events += gamma_mvm
            if not e_tile_missing:
                total_e_gamma_fwd += gamma_mvm * e_ab_read

            # Shared periphery switch
            if hw.shared_periphery_serialize and hw.t_periph_switch_ns is not None:
                total_t_gamma_fwd += hw.t_periph_switch_ns

        # --- Amortized transfer cost (skip if forward_only) ---
        if not cfg.forward_only and cfg.transfer_every > 0:
            # units_in_mbatch=True: transfer_every counts in steps (mini-batches)
            # units_in_mbatch=False: transfer_every counts in mat-vec ops
            # Both cases: tau_eff = transfer_every steps
            tau_eff = max(1, cfg.transfer_every)

            n_reads = cfg.num_reads
            xfer_a_reads = r * n_tiles_A * n_reads
            xfer_b_reads = r * n_tiles_B * n_reads
            xfer_c_updates = r * n_tiles_C

            t_xfer_read_per = hw.t_rank_xfer_read_ns if hw.t_rank_xfer_read_ns is not None else t_ab_read
            t_xfer_write_per = hw.t_rank_xfer_write_ns if hw.t_rank_xfer_write_ns is not None else (t_ab_update if t_ab_update is not None else 0.0)

            t_xfer_event = (xfer_a_reads + xfer_b_reads) * t_xfer_read_per + xfer_c_updates * t_xfer_write_per
            total_t_transfer += t_xfer_event / tau_eff

            xfer_evts = xfer_a_reads + xfer_b_reads + xfer_c_updates
            total_xfer_events += xfer_evts

            # Transfer energy
            e_xfer_read = hw.e_rank_xfer_read_pj if hw.e_rank_xfer_read_pj is not None else (e_ab_read if e_ab_read is not None else 0.0)
            e_xfer_write = hw.e_rank_xfer_write_pj if hw.e_rank_xfer_write_pj is not None else (e_ab_update if e_ab_update is not None else 0.0)

            e_xfer_event = (xfer_a_reads + xfer_b_reads) * e_xfer_read + xfer_c_updates * e_xfer_write
            total_e_transfer += e_xfer_event / tau_eff

    # Missing checks
    if t_missing:
        result.missing_params.append("t_update_ns")
    if e_tile_missing:
        result.missing_params.append("e_tile_pj (for A/B reads)")
    if e_update_missing:
        result.missing_params.append("e_update_pj (for A/B updates)")

    # Assign sub-components
    # For LRTT, "adapter_fwd" = gamma visible path, "adapter_bwd" = projections + updates
    result.DeltaT_adapter_fwd_ns = total_t_gamma_fwd
    result.DeltaT_adapter_bwd_ns = total_t_proj + total_t_update
    result.DeltaT_adapter_opt_ns = 0.0  # No separate optimizer for analog tiles
    result.DeltaT_transfer_ns = total_t_transfer

    result.DeltaE_adapter_fwd_pj = total_e_gamma_fwd
    result.DeltaE_adapter_bwd_pj = total_e_proj + total_e_update
    result.DeltaE_adapter_opt_pj = 0.0
    result.DeltaE_transfer_pj = total_e_transfer

    result.adapter_mvm_events = total_mvm_events
    result.adapter_update_events = total_update_events
    result.transfer_events = total_xfer_events

    result.DeltaT_ns = (result.DeltaT_adapter_fwd_ns +
                        result.DeltaT_adapter_bwd_ns +
                        result.DeltaT_adapter_opt_ns +
                        result.DeltaT_transfer_ns)
    result.DeltaE_pj = (result.DeltaE_adapter_fwd_pj +
                        result.DeltaE_adapter_bwd_pj +
                        result.DeltaE_adapter_opt_pj +
                        result.DeltaE_transfer_pj)

    return result


# =============================================================================
# Tiki-Taka Delta Model
# =============================================================================

def compute_tikitaka_delta(
    targeted: List[LayerSpec],
    hw: HardwareParams,
    cfg: ScenarioConfig,
    S_pad: int,
) -> StepCostResult:
    """Compute DeltaT/DeltaE for tikitaka_fullrank_6t1c_training.

    Derived from ChoppedTransferCompound config in optuna_bert_squad_tiki.py:240-271.
    transfer_columns=True, n_reads_per_transfer=1.

    Per step per targeted layer:
      1. Full-rank U update: pulsed update on U [M x N]
      If gamma=1: visible forward through U path

    Transfer (amortized, column-by-column):
      Per event: 1 column read from U + 1 column write to C
      transfer_columns=True, n_reads_per_transfer=1
    """
    result = StepCostResult()
    BT = cfg.batch_size * S_pad

    # Resolve fallback latencies
    t_u_read = hw.t_u_read_ns if hw.t_u_read_ns is not None else hw.t_tile_ns
    t_u_update = hw.t_u_update_ns if hw.t_u_update_ns is not None else hw.t_update_ns

    e_u_read = hw.e_u_read_pj if hw.e_u_read_pj is not None else hw.e_tile_pj
    e_u_update = hw.e_u_update_pj if hw.e_u_update_pj is not None else hw.e_update_pj

    t_update_missing = t_u_update is None
    e_read_missing = e_u_read is None
    e_update_missing = e_u_update is None

    total_t_update = 0.0
    total_t_gamma_fwd = 0.0
    total_t_transfer = 0.0
    total_e_update = 0.0
    total_e_gamma_fwd = 0.0
    total_e_transfer = 0.0

    total_mvm_events = 0
    total_update_events = 0
    total_xfer_events = 0

    for layer in targeted:
        M, N = layer.M, layer.N
        n_tiles_U = layer.n_tiles     # Same size as base C
        n_tiles_C = layer.n_tiles
        tile_rows = layer.n_tile_rows
        tile_cols = layer.n_tile_cols

        # --- Per-step full-rank update (skip if forward_only) ---
        if not cfg.forward_only:
            if not t_update_missing:
                t_upd = BT * n_tiles_U * t_u_update
                total_t_update += t_upd

            upd_evts = BT * n_tiles_U
            total_update_events += upd_evts

            if not e_update_missing:
                total_e_update += upd_evts * e_u_update

        # --- gamma=1: visible forward through U ---
        if cfg.gamma == 1:
            # MVM through U [M x N]: serial across tile_cols
            t_gamma = BT * tile_cols * t_u_read
            total_t_gamma_fwd += t_gamma
            gamma_mvm = BT * n_tiles_U
            total_mvm_events += gamma_mvm

            if not e_read_missing:
                total_e_gamma_fwd += gamma_mvm * e_u_read

            # Shared periphery switch
            if hw.shared_periphery_serialize and hw.t_periph_switch_ns is not None:
                total_t_gamma_fwd += hw.t_periph_switch_ns

        # --- Amortized column transfer (skip if forward_only) ---
        if not cfg.forward_only and cfg.transfer_every > 0:
            # units_in_mbatch=True: transfer_every counts in steps (mini-batches)
            # units_in_mbatch=False: transfer_every counts in mat-vec ops
            # Both cases: tau_eff = transfer_every steps
            tau_eff = max(1, cfg.transfer_every)

            t_col_read = hw.t_col_xfer_read_ns if hw.t_col_xfer_read_ns is not None else t_u_read
            t_col_write = hw.t_col_xfer_write_ns if hw.t_col_xfer_write_ns is not None else (t_u_update if t_u_update is not None else 0.0)

            t_xfer_event = tile_rows * (t_col_read + t_col_write)
            total_t_transfer += t_xfer_event / tau_eff

            xfer_evts = 2 * tile_rows
            total_xfer_events += xfer_evts

            e_col_read = hw.e_col_xfer_read_pj if hw.e_col_xfer_read_pj is not None else (e_u_read if e_u_read is not None else 0.0)
            e_col_write = hw.e_col_xfer_write_pj if hw.e_col_xfer_write_pj is not None else (e_u_update if e_u_update is not None else 0.0)

            e_xfer_event = tile_rows * (e_col_read + e_col_write)
            total_e_transfer += e_xfer_event / tau_eff

    # Missing checks
    if t_update_missing:
        result.missing_params.append("t_update_ns (for U)")
    if e_read_missing:
        result.missing_params.append("e_tile_pj (for U reads)")
    if e_update_missing:
        result.missing_params.append("e_update_pj (for U updates)")

    # Assign sub-components
    result.DeltaT_adapter_fwd_ns = total_t_gamma_fwd
    result.DeltaT_adapter_bwd_ns = total_t_update  # Full-rank update is the "backward" analog
    result.DeltaT_adapter_opt_ns = 0.0
    result.DeltaT_transfer_ns = total_t_transfer

    result.DeltaE_adapter_fwd_pj = total_e_gamma_fwd
    result.DeltaE_adapter_bwd_pj = total_e_update
    result.DeltaE_adapter_opt_pj = 0.0
    result.DeltaE_transfer_pj = total_e_transfer

    result.adapter_mvm_events = total_mvm_events
    result.adapter_update_events = total_update_events
    result.transfer_events = total_xfer_events

    result.DeltaT_ns = (result.DeltaT_adapter_fwd_ns +
                        result.DeltaT_adapter_bwd_ns +
                        result.DeltaT_adapter_opt_ns +
                        result.DeltaT_transfer_ns)
    result.DeltaE_pj = (result.DeltaE_adapter_fwd_pj +
                        result.DeltaE_adapter_bwd_pj +
                        result.DeltaE_adapter_opt_pj +
                        result.DeltaE_transfer_pj)

    return result


# =============================================================================
# Unified Step Cost Computation
# =============================================================================

def compute_step_cost(
    inventory: List[LayerSpec],
    hw: HardwareParams,
    cfg: ScenarioConfig,
    S_pad: int,
) -> StepCostResult:
    """Compute full step cost = common + delta for one scenario.

    Args:
        inventory: Full 72-layer encoder inventory.
        hw: Hardware parameters.
        cfg: Scenario configuration.
        S_pad: Padded sequence length for this batch.

    Returns:
        StepCostResult with full breakdown.
    """
    # Common path (identical across methods)
    common = compute_common_path(inventory, hw, S_pad, cfg.batch_size,
                                  forward_only=cfg.forward_only)

    # Targeted layers
    targeted = get_targeted_layers(inventory, cfg.target)

    # Method-specific delta
    if cfg.method == "digital_lora":
        delta = compute_digital_lora_delta(targeted, hw, cfg, S_pad)
    elif cfg.method == "lrtt_6t1c":
        delta = compute_lrtt_delta(targeted, hw, cfg, S_pad)
    elif cfg.method == "tikitaka_6t1c":
        delta = compute_tikitaka_delta(targeted, hw, cfg, S_pad)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    # Merge common + delta
    result = StepCostResult()

    # Common
    result.T_base_fwd_ns = common.T_base_fwd_ns
    result.T_base_bwd_ns = common.T_base_bwd_ns
    result.T_other_ns = common.T_other_ns
    result.T_common_ns = common.T_common_ns
    result.E_base_fwd_pj = common.E_base_fwd_pj
    result.E_base_bwd_pj = common.E_base_bwd_pj
    result.E_other_pj = common.E_other_pj
    result.E_common_pj = common.E_common_pj
    result.base_mvm_events = common.base_mvm_events

    # Delta
    result.DeltaT_ns = delta.DeltaT_ns
    result.DeltaE_pj = delta.DeltaE_pj
    result.DeltaT_adapter_fwd_ns = delta.DeltaT_adapter_fwd_ns
    result.DeltaT_adapter_bwd_ns = delta.DeltaT_adapter_bwd_ns
    result.DeltaT_adapter_opt_ns = delta.DeltaT_adapter_opt_ns
    result.DeltaT_transfer_ns = delta.DeltaT_transfer_ns
    result.DeltaE_adapter_fwd_pj = delta.DeltaE_adapter_fwd_pj
    result.DeltaE_adapter_bwd_pj = delta.DeltaE_adapter_bwd_pj
    result.DeltaE_adapter_opt_pj = delta.DeltaE_adapter_opt_pj
    result.DeltaE_transfer_pj = delta.DeltaE_transfer_pj
    result.adapter_mvm_events = delta.adapter_mvm_events
    result.adapter_update_events = delta.adapter_update_events
    result.transfer_events = delta.transfer_events
    result.digital_flops = delta.digital_flops
    result.digital_opt_params = delta.digital_opt_params

    # Totals — with overlap handling for Digital LoRA
    #
    # Digital LoRA AHWA-LoRA style:
    #   overlap_with_base_path=False (default, serialized):
    #     T_step = T_other + T_base_fwd + T_base_bwd + DeltaT_lora_fwd + DeltaT_lora_bwd + DeltaT_lora_opt
    #   overlap_with_base_path=True (LoRA fwd overlaps base fwd):
    #     T_step = T_other + max(T_base_fwd, DeltaT_lora_fwd) + T_base_bwd + DeltaT_lora_bwd + DeltaT_lora_opt
    #
    # For analog methods (LRTT, TikiTaka): always serialized (shared periphery)
    if cfg.method == "digital_lora" and hw.lora_overlap_with_base and not cfg.forward_only:
        # Overlap: LoRA forward runs in parallel with base AIMC forward
        overlap_fwd = max(result.T_base_fwd_ns, delta.DeltaT_adapter_fwd_ns)
        non_overlap_delta = delta.DeltaT_adapter_bwd_ns + delta.DeltaT_adapter_opt_ns
        result.T_step_ns = result.T_other_ns + overlap_fwd + result.T_base_bwd_ns + non_overlap_delta
    else:
        result.T_step_ns = result.T_common_ns + result.DeltaT_ns
    result.E_step_pj = result.E_common_pj + result.DeltaE_pj

    # Merge missing params
    result.missing_params = list(set(common.missing_params + delta.missing_params))

    return result


# =============================================================================
# Batch-Trace Aggregation
# =============================================================================

def compute_epoch_cost(
    inventory: List[LayerSpec],
    hw: HardwareParams,
    cfg: ScenarioConfig,
    batch_trace: List[Dict],
) -> Dict[str, Any]:
    """Compute total cost over a batch trace (one epoch).

    Args:
        inventory: Full layer inventory.
        hw: Hardware parameters.
        cfg: Scenario configuration.
        batch_trace: List of dicts with S_pad, batch_size, etc.

    Returns:
        Summary dict with total and per-step averages.
    """
    total_T = 0.0
    total_E = 0.0
    total_T_common = 0.0
    total_DeltaT = 0.0
    total_tokens = 0
    n_steps = len(batch_trace)
    all_missing = set()

    for batch in batch_trace:
        s_pad = batch['S_pad']
        bs = batch['batch_size']

        step = compute_step_cost(inventory, hw, cfg, s_pad)
        total_T += step.T_step_ns
        total_E += step.E_step_pj
        total_T_common += step.T_common_ns
        total_DeltaT += step.DeltaT_ns
        total_tokens += bs * s_pad
        all_missing.update(step.missing_params)

    avg_T_step = total_T / n_steps if n_steps > 0 else 0.0
    avg_E_step = total_E / n_steps if n_steps > 0 else 0.0

    return {
        'method': cfg.method,
        'target': cfg.target,
        'gamma': cfg.gamma,
        'rank': cfg.rank,
        't_tile_ns': hw.t_tile_ns,
        'k_common_bwd': hw.k_common_bwd,
        'n_steps': n_steps,
        'total_T_ns': total_T,
        'total_E_pj': total_E,
        'total_T_common_ns': total_T_common,
        'total_DeltaT_ns': total_DeltaT,
        'avg_T_step_ns': avg_T_step,
        'avg_E_step_pj': avg_E_step,
        'avg_T_per_seq_ns': total_T / total_tokens if total_tokens > 0 else 0.0,
        'avg_E_per_seq_pj': total_E / total_tokens if total_tokens > 0 else 0.0,
        'total_tokens': total_tokens,
        'missing_params': sorted(all_missing),
    }


def compute_fixed_seq_cost(
    inventory: List[LayerSpec],
    hw: HardwareParams,
    cfg: ScenarioConfig,
    S: int,
) -> Dict[str, Any]:
    """Compute per-step cost for a fixed sequence length.

    Args:
        inventory: Full layer inventory.
        hw: Hardware parameters.
        cfg: Scenario configuration.
        S: Fixed sequence length.

    Returns:
        Result dict with per-step cost breakdown.
    """
    step = compute_step_cost(inventory, hw, cfg, S)

    return {
        'method': cfg.method,
        'target': cfg.target,
        'gamma': cfg.gamma,
        'rank': cfg.rank,
        'seq_mode': 'fixed',
        'S': S,
        't_tile_ns': hw.t_tile_ns,
        'k_common_bwd': hw.k_common_bwd,
        'T_common_ns': step.T_common_ns,
        'DeltaT_ns': step.DeltaT_ns,
        'T_step_ns': step.T_step_ns,
        'E_common_pj': step.E_common_pj,
        'DeltaE_pj': step.DeltaE_pj,
        'E_step_pj': step.E_step_pj,
        'T_base_fwd_ns': step.T_base_fwd_ns,
        'T_base_bwd_ns': step.T_base_bwd_ns,
        'T_other_ns': step.T_other_ns,
        'DeltaT_adapter_fwd_ns': step.DeltaT_adapter_fwd_ns,
        'DeltaT_adapter_bwd_ns': step.DeltaT_adapter_bwd_ns,
        'DeltaT_adapter_opt_ns': step.DeltaT_adapter_opt_ns,
        'DeltaT_transfer_ns': step.DeltaT_transfer_ns,
        'latency_per_seq_ns': step.T_step_ns / (cfg.batch_size * S) if S > 0 else 0.0,
        'energy_per_seq_pj': step.E_step_pj / (cfg.batch_size * S) if S > 0 else 0.0,
        'base_mvm_events': step.base_mvm_events,
        'adapter_mvm_events': step.adapter_mvm_events,
        'adapter_update_events': step.adapter_update_events,
        'transfer_events': step.transfer_events,
        'digital_flops': step.digital_flops,
        'digital_opt_params': step.digital_opt_params,
        'missing_params': step.missing_params,
    }


# =============================================================================
# Convergence Log Hook (time-to-target)
# =============================================================================

def load_convergence_log(path: str) -> List[Dict]:
    """Load externally supplied convergence log.

    Expected CSV columns: method, target, gamma, rank, step, f1
    """
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'method': row['method'],
                'target': row['target'],
                'gamma': int(row['gamma']),
                'rank': int(row['rank']),
                'step': int(row['step']),
                'f1': float(row['f1']),
            })
    return rows


def compute_time_to_target(
    convergence_log: List[Dict],
    step_costs: Dict[Tuple, float],
    target_f1: float = 80.0,
) -> List[Dict]:
    """Compute time-to-target-F1 from convergence log and step costs.

    Args:
        convergence_log: List of {method, target, gamma, rank, step, f1}.
        step_costs: Dict mapping (method, target, gamma, rank) -> avg_T_step_ns.
        target_f1: Target F1 score.

    Returns:
        List of {method, target, gamma, rank, step_to_target, time_to_target_ns, energy_to_target_pj}.
    """
    # Group by (method, target, gamma, rank) and find first step >= target_f1
    from collections import defaultdict
    groups = defaultdict(list)
    for entry in convergence_log:
        key = (entry['method'], entry['target'], entry['gamma'], entry['rank'])
        groups[key].append((entry['step'], entry['f1']))

    results = []
    for key, entries in groups.items():
        entries.sort(key=lambda x: x[0])
        step_to_target = None
        for step, f1 in entries:
            if f1 >= target_f1:
                step_to_target = step
                break

        if step_to_target is not None and key in step_costs:
            avg_T = step_costs[key]
            results.append({
                'method': key[0],
                'target': key[1],
                'gamma': key[2],
                'rank': key[3],
                'target_f1': target_f1,
                'step_to_target': step_to_target,
                'time_to_target_ns': step_to_target * avg_T,
                'energy_to_target_pj': None,  # Needs avg_E_step
            })

    return results


# =============================================================================
# Hardware Params YAML Loader
# =============================================================================

def load_hardware_params(yaml_path: str) -> HardwareParams:
    """Load hardware parameters from YAML file.

    Missing (null) values remain as None in the dataclass.
    """
    try:
        import yaml
    except ImportError:
        print("WARNING: PyYAML not installed. Using default hardware parameters.")
        return HardwareParams()

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    hw = HardwareParams()

    # Common section
    common = data.get('common', {})
    if common.get('tile_size') is not None:
        hw.tile_size = common['tile_size']
    if common.get('io_bits') is not None:
        hw.io_bits = common['io_bits']
    if common.get('base_tile_energy_pj') is not None:
        hw.e_tile_pj = common['base_tile_energy_pj']
    if common.get('tile_update_latency_ns') is not None:
        hw.t_update_ns = common['tile_update_latency_ns']
    if common.get('tile_update_energy_pj') is not None:
        hw.e_update_pj = common['tile_update_energy_pj']
    if common.get('t_digital_ns_per_flop') is not None:
        hw.t_digital_ns_per_flop = common['t_digital_ns_per_flop']
    if common.get('e_digital_pj_per_flop') is not None:
        hw.e_digital_pj_per_flop = common['e_digital_pj_per_flop']

    # Digital LoRA section
    dl = data.get('digital_lora', {})
    if dl.get('gemm_latency_ns_per_flop') is not None:
        hw.t_lora_gemm_ns_per_flop = dl['gemm_latency_ns_per_flop']
    if dl.get('gemm_energy_pj_per_flop') is not None:
        hw.e_lora_gemm_pj_per_flop = dl['gemm_energy_pj_per_flop']
    if dl.get('optimizer_update_latency_ns_per_param') is not None:
        hw.t_lora_opt_ns_per_param = dl['optimizer_update_latency_ns_per_param']
    if dl.get('optimizer_update_energy_pj_per_param') is not None:
        hw.e_lora_opt_pj_per_param = dl['optimizer_update_energy_pj_per_param']
    if dl.get('overlap_with_base_path') is not None:
        hw.lora_overlap_with_base = dl['overlap_with_base_path']

    # LRTT section
    lrtt = data.get('lrtt_6t1c', {})
    hw.t_ab_read_ns = lrtt.get('ab_read_latency_ns')
    hw.e_ab_read_pj = lrtt.get('ab_read_energy_pj')
    hw.t_ab_update_ns = lrtt.get('ab_update_latency_ns')
    hw.e_ab_update_pj = lrtt.get('ab_update_energy_pj')
    hw.t_rank_xfer_read_ns = lrtt.get('rank_transfer_read_latency_ns')
    hw.e_rank_xfer_read_pj = lrtt.get('rank_transfer_read_energy_pj')
    hw.t_rank_xfer_write_ns = lrtt.get('rank_transfer_write_latency_ns')
    hw.e_rank_xfer_write_pj = lrtt.get('rank_transfer_write_energy_pj')
    if lrtt.get('shared_periphery_serialize') is not None:
        hw.shared_periphery_serialize = lrtt['shared_periphery_serialize']
    hw.t_periph_switch_ns = lrtt.get('periphery_switch_latency_ns')
    hw.e_periph_switch_pj = lrtt.get('periphery_switch_energy_pj')

    # Tiki-Taka section
    tt = data.get('tikitaka_6t1c', {})
    hw.t_u_read_ns = tt.get('u_visible_latency_ns')
    hw.e_u_read_pj = tt.get('u_visible_energy_pj')
    hw.t_u_update_ns = tt.get('u_update_latency_ns')
    hw.e_u_update_pj = tt.get('u_update_energy_pj')
    hw.t_col_xfer_read_ns = tt.get('column_transfer_read_latency_ns')
    hw.e_col_xfer_read_pj = tt.get('column_transfer_read_energy_pj')
    hw.t_col_xfer_write_ns = tt.get('column_transfer_write_latency_ns')
    hw.e_col_xfer_write_pj = tt.get('column_transfer_write_energy_pj')

    return hw

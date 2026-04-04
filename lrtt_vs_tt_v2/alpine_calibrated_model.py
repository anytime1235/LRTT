#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALPINE-calibrated LR-TT vs TikiTaka absolute/sensitivity cost model.

Two latency models:
  Model A (MAIN): ALPINE-throughput-calibrated utilization-aware model
    - Uses element-level op counts, not tile-level counts
    - tau_acim = 100 ns / 131072 as per-op time anchor
    - alpha_t scenarios for update cost ratio
  Model B (SUPPLEMENTARY): Fixed-tile ALPINE upper-bound
    - Fixed 100 ns per 256x256 tile MVM
    - Rank-insensitive (too pessimistic)

ALPINE anchors (from spec):
  - Full-tile MVM latency: 100 ns
  - Full-tile size: 256 x 256
  - Full-tile MVM efficiency: 12.8 TOp/s/W
  - Full-tile ops = 2 * 256 * 256 = 131072
  - tau_acim = 100 ns / 131072 ≈ 7.629e-4 ns/op
  - eps_acim = 131072 / 12.8e12 / 131072 = 1 / 12.8e12 J/op ≈ 7.8125e-14 J/op
"""

import os
import sys
import csv
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Add cost_model to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COST_MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "cost_model")
sys.path.insert(0, COST_MODEL_DIR)

try:
    from extract_layer_inventory import (
        build_layer_inventory as _build_layer_inventory,
        get_targeted_layers,
        compute_adapter_tile_counts,
        tile_count,
        LayerSpec,
        DEFAULT_TILE_SIZE,
    )
except Exception:
    # Standalone fallback: define LayerSpec and load from CSV
    from dataclasses import dataclass as _dc
    DEFAULT_TILE_SIZE = 512

    @_dc
    class LayerSpec:
        layer_idx: int
        module_name: str
        sub_name: str
        group: str
        M: int
        N: int
        n_tile_rows: int
        n_tile_cols: int
        n_tiles: int

    def get_targeted_layers(inventory, target):
        if target == "all":
            return list(inventory)
        return [l for l in inventory if l.group == target]

    def tile_count(dim, tile_size=DEFAULT_TILE_SIZE):
        return math.ceil(dim / tile_size)

    def compute_adapter_tile_counts(layer, rank, tile_size=DEFAULT_TILE_SIZE):
        a_rows = tile_count(layer.M, tile_size)
        a_cols = tile_count(rank, tile_size)
        b_rows = tile_count(rank, tile_size)
        b_cols = tile_count(layer.N, tile_size)
        return {
            'n_tiles_A': a_rows * a_cols,
            'n_tiles_A_rows': a_rows,
            'n_tiles_A_cols': a_cols,
            'n_tiles_B': b_rows * b_cols,
            'n_tiles_B_rows': b_rows,
            'n_tiles_B_cols': b_cols,
        }

    _build_layer_inventory = None


def build_layer_inventory_from_csv(csv_path: str) -> List[LayerSpec]:
    """Load layer inventory from pre-built CSV."""
    inventory = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inventory.append(LayerSpec(
                layer_idx=int(row['layer_idx']),
                module_name=row['module_name'],
                sub_name=row['sub_name'],
                group=row['group'],
                M=int(row['M']),
                N=int(row['N']),
                n_tile_rows=int(row['n_tile_rows']),
                n_tile_cols=int(row['n_tile_cols']),
                n_tiles=int(row['n_tiles']),
            ))
    return inventory


def build_layer_inventory(tile_size: int = DEFAULT_TILE_SIZE) -> List[LayerSpec]:
    """Build layer inventory, preferring CSV fallback if transformers unavailable."""
    csv_path = os.path.join(COST_MODEL_DIR, "layer_inventory.csv")
    if os.path.exists(csv_path):
        return build_layer_inventory_from_csv(csv_path)
    if _build_layer_inventory is not None:
        return _build_layer_inventory(tile_size)
    raise RuntimeError("No layer inventory source available. "
                       "Provide layer_inventory.csv or install transformers+torch.")

# =============================================================================
# ALPINE Anchors
# =============================================================================

ALPINE_TILE_SIZE = 256
ALPINE_TILE_MVM_NS = 100.0          # Full 256x256 tile MVM latency [ns]
ALPINE_TILE_OPS = 2 * 256 * 256     # = 131072 MACs * 2 = ops per tile MVM
ALPINE_EFFICIENCY_TOPS_W = 12.8     # TOp/s/W

# Derived anchors
TAU_ACIM = ALPINE_TILE_MVM_NS / ALPINE_TILE_OPS  # ns per op
P_ACIM = ALPINE_TILE_OPS / ALPINE_TILE_MVM_NS    # ops per ns = 1.31072e6 ops/ns

# Energy anchor
E_TILE_FULL_J = ALPINE_TILE_OPS / (ALPINE_EFFICIENCY_TOPS_W * 1e12)  # joules per tile MVM
EPS_ACIM_J = E_TILE_FULL_J / ALPINE_TILE_OPS     # joules per op
EPS_ACIM_PJ = EPS_ACIM_J * 1e12                   # picojoules per op

# Scenario parameters
ALPHA_T_SCENARIOS = [1.00, 1.41, 1.55, 4.15, 9.30]  # t_upd / t_mvm_per_op
KAPPA_E_SCENARIOS = [1, 2, 5, 10]                     # e_upd / e_mvm_per_op

# Workload
BATCH_SIZE = 48
S_PAD_DEFAULT = 384
RANKS = [1, 2, 4, 8, 16, 32, 64]
TARGETS = ["attention", "ffn", "all"]
FIXED_SEQ_LENGTHS = [64, 128, 256, 384]

# Transfer parameters (from training configs)
LRTT_TRANSFER_EVERY = 4   # steps
LRTT_NUM_READS = 1
TIKITAKA_TRANSFER_EVERY = 1  # steps

# Transfer weighting factors (default = 1, transfer is tiny)
RHO_TR_T = 1.0
RHO_TR_E = 1.0


# =============================================================================
# Element Count Computation
# =============================================================================

@dataclass
class ElementCounts:
    """Element-level operation counts for adapter path."""
    N_proj_elem: int = 0      # Projection MVM ops (LR-TT only)
    N_upd_elem: int = 0       # Pulsed update ops
    N_vis_elem: int = 0       # Visible forward MVM ops (gamma=1 only)
    N_tr_src_elem: int = 0    # Transfer source ops (amortized per step)
    method: str = ""
    target: str = ""
    rank: int = 0
    gamma: int = 0


def compute_lrtt_element_counts(
    targeted: List[LayerSpec],
    rank: int,
    gamma: int,
    BT: int,
    transfer_every: int = LRTT_TRANSFER_EVERY,
    num_reads: int = LRTT_NUM_READS,
) -> ElementCounts:
    """Compute element-level operation counts for LR-TT adapter path.

    For each targeted layer W [M x N] with adapter A [M x r], B [r x N]:
      - Projection: B.forward(x) + A.backward(d) = BT * (M*r + r*N) element-ops
      - Update: A.update + B.update = BT * (M*r + r*N) element-ops
      - Visible (gamma=1): B.forward(x) + A.forward(Bx) = BT * (M*r + r*N) element-ops
      - Transfer: r reads from A + r reads from B → C, amortized

    Element ops use actual weight dimensions, NOT tile counts.
    This preserves rank sensitivity that tile-granularity would destroy.
    """
    ec = ElementCounts(method="lrtt_6t1c", target="", rank=rank, gamma=gamma)

    for layer in targeted:
        M, N = layer.M, layer.N

        # Projection MVMs: B.fwd = BT * r * N ops, A.bwd = BT * M * r ops
        # Each is a matrix-vector product counted as 2*dim1*dim2 multiply-accumulate ops
        proj_ops = BT * 2 * (M * rank + rank * N)
        ec.N_proj_elem += proj_ops

        # Pulsed updates: same element footprint as projection
        upd_ops = BT * 2 * (M * rank + rank * N)
        ec.N_upd_elem += upd_ops

        # Visible forward (gamma=1 only)
        if gamma == 1:
            vis_ops = BT * 2 * (M * rank + rank * N)
            ec.N_vis_elem += vis_ops

        # Transfer (amortized): r one-hot reads from A, r one-hot reads from B,
        # r rank-1 updates to C
        # One-hot read from A: reads M elements per rank slice
        # One-hot read from B: reads N elements per rank slice
        # Rank-1 update to C: M*N elements per rank slice
        tau_eff = max(1, transfer_every)
        tr_read_a = rank * M * num_reads
        tr_read_b = rank * N * num_reads
        tr_write_c = rank * M * N  # rank-1 outer product writes
        tr_total_per_event = 2 * (tr_read_a + tr_read_b + tr_write_c)
        ec.N_tr_src_elem += tr_total_per_event // tau_eff

    return ec


def compute_tikitaka_element_counts(
    targeted: List[LayerSpec],
    gamma: int,
    BT: int,
    transfer_every: int = TIKITAKA_TRANSFER_EVERY,
) -> ElementCounts:
    """Compute element-level operation counts for TikiTaka adapter path.

    For each targeted layer W [M x N] with full-rank U [M x N]:
      - Update: U.update(x, d) = BT * M * N element-ops
      - Visible (gamma=1): MVM through U = BT * M * N element-ops
      - Transfer: 1 column read from U + 1 column write to C, amortized
    """
    ec = ElementCounts(method="tikitaka_6t1c", target="", rank=0, gamma=gamma)

    for layer in targeted:
        M, N = layer.M, layer.N

        # Full-rank pulsed update: BT * 2*M*N ops
        upd_ops = BT * 2 * M * N
        ec.N_upd_elem += upd_ops

        # Visible forward (gamma=1 only)
        if gamma == 1:
            vis_ops = BT * 2 * M * N
            ec.N_vis_elem += vis_ops

        # Transfer: 1 column (M elements) read + write per event
        tau_eff = max(1, transfer_every)
        tr_per_event = 2 * M  # read column + write column
        ec.N_tr_src_elem += tr_per_event // tau_eff

    return ec


# =============================================================================
# Model A: ALPINE-Throughput-Calibrated Utilization-Aware Model
# =============================================================================

@dataclass
class ModelAResult:
    """Result from Model A (main paper model)."""
    method: str
    target: str
    rank: int
    gamma: int
    alpha_t: float
    kappa_e: float = 1.0
    S_pad: int = S_PAD_DEFAULT

    # Element counts
    N_proj_elem: int = 0
    N_upd_elem: int = 0
    N_vis_elem: int = 0
    N_tr_src_elem: int = 0

    # Latency [ns]
    DeltaT_proj_ns: float = 0.0
    DeltaT_upd_ns: float = 0.0
    DeltaT_vis_ns: float = 0.0
    DeltaT_tr_ns: float = 0.0
    DeltaT_total_ns: float = 0.0

    # Energy [pJ]
    DeltaE_proj_pj: float = 0.0
    DeltaE_upd_pj: float = 0.0
    DeltaE_vis_pj: float = 0.0
    DeltaE_tr_pj: float = 0.0
    DeltaE_total_pj: float = 0.0

    # Equivalent throughput
    EqOps: int = 0
    ETOPS: float = 0.0           # EqOps / DeltaT [ops/ns = TOp/s if *1e-3]
    ETOPS_per_W: float = 0.0     # EqOps / DeltaE


def compute_model_a(
    ec: ElementCounts,
    alpha_t: float,
    kappa_e: float = 1.0,
    S_pad: int = S_PAD_DEFAULT,
) -> ModelAResult:
    """Compute Model A: ALPINE-throughput-calibrated absolute costs.

    DeltaT = tau_acim * (N_proj + N_vis) + alpha_t * tau_acim * N_upd
             + rho_tr * tau_acim * N_tr
    DeltaE = eps_acim * (N_proj + N_vis) + kappa_e * eps_acim * N_upd
             + rho_tr_e * eps_acim * N_tr
    """
    r = ModelAResult(
        method=ec.method, target=ec.target, rank=ec.rank,
        gamma=ec.gamma, alpha_t=alpha_t, kappa_e=kappa_e, S_pad=S_pad,
        N_proj_elem=ec.N_proj_elem, N_upd_elem=ec.N_upd_elem,
        N_vis_elem=ec.N_vis_elem, N_tr_src_elem=ec.N_tr_src_elem,
    )

    # Latency components
    r.DeltaT_proj_ns = TAU_ACIM * ec.N_proj_elem
    r.DeltaT_vis_ns = TAU_ACIM * ec.N_vis_elem
    r.DeltaT_upd_ns = alpha_t * TAU_ACIM * ec.N_upd_elem
    r.DeltaT_tr_ns = RHO_TR_T * TAU_ACIM * ec.N_tr_src_elem
    r.DeltaT_total_ns = r.DeltaT_proj_ns + r.DeltaT_vis_ns + r.DeltaT_upd_ns + r.DeltaT_tr_ns

    # Energy components
    r.DeltaE_proj_pj = EPS_ACIM_PJ * ec.N_proj_elem
    r.DeltaE_vis_pj = EPS_ACIM_PJ * ec.N_vis_elem
    r.DeltaE_upd_pj = kappa_e * EPS_ACIM_PJ * ec.N_upd_elem
    r.DeltaE_tr_pj = RHO_TR_E * EPS_ACIM_PJ * ec.N_tr_src_elem
    r.DeltaE_total_pj = r.DeltaE_proj_pj + r.DeltaE_vis_pj + r.DeltaE_upd_pj + r.DeltaE_tr_pj

    # Equivalent throughput
    # EqOps counts ALL active operations including both MVM-class and update-class,
    # weighted equally. This makes ETOPS a throughput metric that reflects the
    # total work done per unit time, not just MVM throughput.
    r.EqOps = ec.N_proj_elem + ec.N_upd_elem + ec.N_vis_elem + ec.N_tr_src_elem

    if r.DeltaT_total_ns > 0:
        # ETOPS: EqOps / DeltaT_total_ns gives ops/ns; convert to TOp/s
        r.ETOPS = r.EqOps / r.DeltaT_total_ns * 1e-3  # TOp/s
    if r.DeltaE_total_pj > 0:
        # ETOPS/W: EqOps / DeltaE (in joules) gives ops/J = ops/W·s
        DeltaE_J = r.DeltaE_total_pj * 1e-12
        r.ETOPS_per_W = r.EqOps / DeltaE_J * 1e-12  # TOp/s/W


    return r


# =============================================================================
# Model B: Fixed-Tile ALPINE Upper-Bound (Supplementary)
# =============================================================================

@dataclass
class ModelBResult:
    """Result from Model B (fixed-tile upper-bound)."""
    method: str
    target: str
    rank: int
    gamma: int

    # Tile counts
    n_tiles_proj: int = 0
    n_tiles_upd: int = 0
    n_tiles_vis: int = 0
    n_tiles_tr: int = 0

    # Latency [ns] (fixed 100 ns per tile MVM)
    DeltaT_total_ns: float = 0.0

    # Note: this model does NOT properly separate update from MVM
    # because it uses the same fixed per-tile latency for everything


def compute_model_b(
    targeted: List[LayerSpec],
    method: str,
    rank: int,
    gamma: int,
    BT: int,
    tile_size: int = ALPINE_TILE_SIZE,
    tile_latency_ns: float = ALPINE_TILE_MVM_NS,
    transfer_every_lrtt: int = LRTT_TRANSFER_EVERY,
    transfer_every_tt: int = TIKITAKA_TRANSFER_EVERY,
) -> ModelBResult:
    """Compute Model B: fixed-tile ALPINE upper-bound.

    Uses ceil(dim/256) tiling with fixed 100 ns per tile.
    Rank <= 32 → ceil(rank/256) = 1, so rank sensitivity disappears.
    """
    r = ModelBResult(method=method, target="", rank=rank, gamma=gamma)
    total_t = 0.0

    for layer in targeted:
        M, N = layer.M, layer.N
        tc_M = math.ceil(M / tile_size)
        tc_N = math.ceil(N / tile_size)
        tc_r = math.ceil(rank / tile_size) if rank > 0 else 0

        if method == "lrtt_6t1c":
            # A [M x r]: tc_M * tc_r tiles,  B [r x N]: tc_r * tc_N tiles
            n_tiles_A = tc_M * tc_r
            n_tiles_B = tc_r * tc_N

            # Projection MVMs
            t_proj = BT * (tc_N * tile_latency_ns + tc_M * tile_latency_ns)  # B.fwd + A.bwd
            total_t += t_proj

            # Updates (use same tile latency as conservative estimate)
            t_upd = BT * (n_tiles_A + n_tiles_B) * tile_latency_ns
            total_t += t_upd

            # Visible forward (gamma=1)
            if gamma == 1:
                t_vis = BT * (tc_N + tc_r) * tile_latency_ns  # B.fwd + A.fwd
                total_t += t_vis

            # Transfer (amortized)
            tau_eff = max(1, transfer_every_lrtt)
            n_tiles_C = tc_M * tc_N
            t_tr = (rank * (n_tiles_A + n_tiles_B) + rank * n_tiles_C) * tile_latency_ns / tau_eff
            total_t += t_tr

        elif method == "tikitaka_6t1c":
            n_tiles_U = tc_M * tc_N

            # Full-rank update
            t_upd = BT * n_tiles_U * tile_latency_ns
            total_t += t_upd

            # Visible forward (gamma=1)
            if gamma == 1:
                t_vis = BT * tc_N * tile_latency_ns
                total_t += t_vis

            # Transfer (column-by-column, amortized)
            tau_eff = max(1, transfer_every_tt)
            t_tr = tc_M * 2 * tile_latency_ns / tau_eff
            total_t += t_tr

    r.DeltaT_total_ns = total_t
    return r


# =============================================================================
# Full Sweep Runner
# =============================================================================

def run_full_sweep(
    inventory: List[LayerSpec],
    S_pad: int = S_PAD_DEFAULT,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, List]:
    """Run complete Model A + Model B sweep across all scenarios.

    Returns dict with keys:
      'model_a': list of ModelAResult
      'model_b': list of ModelBResult
    """
    BT = batch_size * S_pad
    results_a = []
    results_b = []

    for target in TARGETS:
        targeted = get_targeted_layers(inventory, target)

        # --- TikiTaka scenarios ---
        for gamma in [0, 1]:
            ec_tt = compute_tikitaka_element_counts(targeted, gamma, BT)
            ec_tt.target = target

            # Model A: across alpha_t and kappa_e
            for alpha_t in ALPHA_T_SCENARIOS:
                for kappa_e in KAPPA_E_SCENARIOS:
                    r = compute_model_a(ec_tt, alpha_t, kappa_e, S_pad)
                    r.target = target
                    results_a.append(r)

            # Model B
            rb = compute_model_b(targeted, "tikitaka_6t1c", 0, gamma, BT)
            rb.target = target
            results_b.append(rb)

        # --- LR-TT scenarios ---
        for rank in RANKS:
            for gamma in [0, 1]:
                ec_lr = compute_lrtt_element_counts(targeted, rank, gamma, BT)
                ec_lr.target = target

                # Model A
                for alpha_t in ALPHA_T_SCENARIOS:
                    for kappa_e in KAPPA_E_SCENARIOS:
                        r = compute_model_a(ec_lr, alpha_t, kappa_e, S_pad)
                        r.target = target
                        results_a.append(r)

                # Model B
                rb = compute_model_b(targeted, "lrtt_6t1c", rank, gamma, BT)
                rb.target = target
                results_b.append(rb)

    return {'model_a': results_a, 'model_b': results_b}


def run_sequence_sweep(
    inventory: List[LayerSpec],
    batch_size: int = BATCH_SIZE,
    rank: int = 8,
    target: str = "all",
    alpha_t: float = 1.0,
    kappa_e: float = 1.0,
    seq_lengths: List[int] = None,
    dynamic_S_pad_mean: float = None,
) -> List[ModelAResult]:
    """Run sequence length sweep for a fixed rank/target."""
    if seq_lengths is None:
        seq_lengths = FIXED_SEQ_LENGTHS
    targeted = get_targeted_layers(inventory, target)
    results = []

    all_S = list(seq_lengths)
    if dynamic_S_pad_mean is not None:
        all_S.append(int(round(dynamic_S_pad_mean)))

    for S in all_S:
        BT = batch_size * S
        for gamma in [0, 1]:
            # LR-TT
            ec_lr = compute_lrtt_element_counts(targeted, rank, gamma, BT)
            ec_lr.target = target
            r = compute_model_a(ec_lr, alpha_t, kappa_e, S)
            r.target = target
            results.append(r)

            # TikiTaka
            ec_tt = compute_tikitaka_element_counts(targeted, gamma, BT)
            ec_tt.target = target
            r = compute_model_a(ec_tt, alpha_t, kappa_e, S)
            r.target = target
            results.append(r)

    return results


# =============================================================================
# CSV Writers
# =============================================================================

MODEL_A_COLUMNS = [
    'method', 'target', 'rank', 'gamma', 'alpha_t', 'kappa_e', 'S_pad',
    'N_proj_elem', 'N_upd_elem', 'N_vis_elem', 'N_tr_src_elem',
    'DeltaT_proj_ns', 'DeltaT_upd_ns', 'DeltaT_vis_ns', 'DeltaT_tr_ns',
    'DeltaT_total_ns',
    'DeltaE_proj_pj', 'DeltaE_upd_pj', 'DeltaE_vis_pj', 'DeltaE_tr_pj',
    'DeltaE_total_pj',
    'EqOps', 'ETOPS', 'ETOPS_per_W',
]

MODEL_B_COLUMNS = [
    'method', 'target', 'rank', 'gamma', 'DeltaT_total_ns',
]


def write_model_a_csv(results: List[ModelAResult], path: str) -> None:
    """Write Model A results to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_A_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {col: getattr(r, col, '') for col in MODEL_A_COLUMNS}
            writer.writerow(row)


def write_model_b_csv(results: List[ModelBResult], path: str) -> None:
    """Write Model B results to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_B_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {col: getattr(r, col, '') for col in MODEL_B_COLUMNS}
            writer.writerow(row)


# =============================================================================
# Dynamic padding mean from batch trace
# =============================================================================

def get_dynamic_S_pad_mean(trace_path: str) -> Optional[float]:
    """Load batch trace and compute mean S_pad."""
    if not os.path.exists(trace_path):
        return None
    s_pads = []
    with open(trace_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_pads.append(int(row['S_pad']))
    return np.mean(s_pads) if s_pads else None


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Run the full ALPINE-calibrated cost model."""
    OUT = os.path.join(SCRIPT_DIR)
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("ALPINE-Calibrated LR-TT vs TikiTaka Cost Model")
    print("=" * 70)
    print(f"\nALPINE Anchors:")
    print(f"  Tile size: {ALPINE_TILE_SIZE}x{ALPINE_TILE_SIZE}")
    print(f"  Full-tile MVM latency: {ALPINE_TILE_MVM_NS} ns")
    print(f"  Full-tile ops: {ALPINE_TILE_OPS}")
    print(f"  tau_acim: {TAU_ACIM:.6e} ns/op")
    print(f"  P_acim: {P_ACIM:.6e} ops/ns")
    print(f"  Efficiency: {ALPINE_EFFICIENCY_TOPS_W} TOp/s/W")
    print(f"  eps_acim: {EPS_ACIM_PJ:.6e} pJ/op")
    print(f"  E_tile_full: {E_TILE_FULL_J:.6e} J")

    # Build inventory
    print("\nBuilding layer inventory...")
    inventory = build_layer_inventory()
    print(f"  Total layers: {len(inventory)}")

    # Dynamic padding mean
    trace_path = os.path.join(COST_MODEL_DIR, "batch_trace.csv")
    dyn_mean = get_dynamic_S_pad_mean(trace_path)
    print(f"  Dynamic padding S_pad mean: {dyn_mean:.1f}" if dyn_mean else "  No batch trace found")

    # Run main sweep (S_pad = 384)
    print("\n--- Running Model A + B sweep (S=384) ---")
    sweep = run_full_sweep(inventory)
    print(f"  Model A results: {len(sweep['model_a'])}")
    print(f"  Model B results: {len(sweep['model_b'])}")

    # Write CSVs
    # Filter for latency interval CSV (all alpha_t, kappa_e=1)
    latency_results = [r for r in sweep['model_a'] if r.kappa_e == 1.0]
    write_model_a_csv(latency_results, os.path.join(OUT, "lrtt_vs_tt_latency_interval.csv"))

    # ETOPS interval CSV (all alpha_t, kappa_e=1)
    write_model_a_csv(latency_results, os.path.join(OUT, "lrtt_vs_tt_etops_interval.csv"))

    # Energy interval CSV (alpha_t=1, all kappa_e)
    energy_results = [r for r in sweep['model_a'] if r.alpha_t == 1.0]
    write_model_a_csv(energy_results, os.path.join(OUT, "lrtt_vs_tt_energy_interval.csv"))

    # ETOPS/W interval CSV (all alpha_t x kappa_e)
    write_model_a_csv(sweep['model_a'], os.path.join(OUT, "lrtt_vs_tt_etopsw_interval.csv"))

    # Sequence sweep CSV
    print("\n--- Running sequence sweep ---")
    seq_results = run_sequence_sweep(
        inventory, rank=8, target="all", alpha_t=1.0,
        dynamic_S_pad_mean=dyn_mean,
    )
    write_model_a_csv(seq_results, os.path.join(OUT, "lrtt_vs_tt_sequence_sweep.csv"))
    print(f"  Sequence sweep results: {len(seq_results)}")

    # Model B CSV
    write_model_b_csv(sweep['model_b'], os.path.join(OUT, "lrtt_vs_tt_model_b_upper_bound.csv"))

    print("\n--- Summary ---")
    # Print element count comparison for target=all, rank=8, gamma=0
    targeted_all = get_targeted_layers(inventory, "all")
    BT = BATCH_SIZE * S_PAD_DEFAULT
    ec_lr = compute_lrtt_element_counts(targeted_all, 8, 0, BT)
    ec_tt = compute_tikitaka_element_counts(targeted_all, 0, BT)

    print(f"\n  Element counts (target=all, rank=8, gamma=0, S=384, BS=48):")
    print(f"  LR-TT:")
    print(f"    N_proj_elem: {ec_lr.N_proj_elem:>15,}")
    print(f"    N_upd_elem:  {ec_lr.N_upd_elem:>15,}")
    print(f"    N_vis_elem:  {ec_lr.N_vis_elem:>15,}")
    print(f"    N_tr_elem:   {ec_lr.N_tr_src_elem:>15,}")
    print(f"  TikiTaka:")
    print(f"    N_upd_elem:  {ec_tt.N_upd_elem:>15,}")
    print(f"    N_vis_elem:  {ec_tt.N_vis_elem:>15,}")
    print(f"    N_tr_elem:   {ec_tt.N_tr_src_elem:>15,}")
    print(f"  Ratio TT_upd/LR_upd: {ec_tt.N_upd_elem / ec_lr.N_upd_elem:.2f}x")

    # Print latency at alpha_t=1.0
    r_lr = compute_model_a(ec_lr, 1.0)
    r_tt = compute_model_a(ec_tt, 1.0)
    print(f"\n  DeltaT at alpha_t=1.0:")
    print(f"    LR-TT gamma=0: {r_lr.DeltaT_total_ns:.2f} ns ({r_lr.DeltaT_total_ns/1e6:.4f} ms)")
    print(f"    TikiTaka gamma=0: {r_tt.DeltaT_total_ns:.2f} ns ({r_tt.DeltaT_total_ns/1e6:.4f} ms)")
    print(f"    Ratio TT/LR: {r_tt.DeltaT_total_ns / r_lr.DeltaT_total_ns:.2f}x")

    print(f"\n  ETOPS at alpha_t=1.0:")
    print(f"    LR-TT gamma=0: {r_lr.ETOPS:.4f} TOp/s")
    print(f"    TikiTaka gamma=0: {r_tt.ETOPS:.4f} TOp/s")

    print("\nDone. CSVs written to:", OUT)
    return sweep


if __name__ == "__main__":
    main()

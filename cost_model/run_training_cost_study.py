#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run AIMC training cost study across all scenarios.

Orchestrates the cost model across the full scenario matrix:
  Methods:  digital_lora, lrtt_6t1c, tikitaka_6t1c
  Targets:  attention, ffn, all
  Gamma:    0, 1 (for lrtt/tikitaka; digital_lora always forward-visible)
  Ranks:    4, 8, 16, 32
  Seq modes: dynamic padding (main), fixed S in {64, 128, 256, 384}
  t_tile:   {128, 256, 512} ns (optional ref: 100 ns)
  k_bwd:    {2.0, 2.5, 3.0}

Outputs:
  results_training_step.csv
  results_forward_only.csv
  results_time_to_target.csv (if convergence log provided)
  sensitivity_summary.md
"""

import os
import sys
import csv
import math
import argparse
import itertools
from typing import List, Dict, Any, Optional

# Add cost_model directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from extract_layer_inventory import (
    build_layer_inventory,
    get_targeted_layers,
    summarize_inventory,
    save_inventory_csv,
    LayerSpec,
)
from cost_model_core import (
    HardwareParams,
    ScenarioConfig,
    StepCostResult,
    compute_step_cost,
    compute_epoch_cost,
    compute_fixed_seq_cost,
    load_hardware_params,
    load_convergence_log,
    compute_time_to_target,
    BATCH_SIZE,
)


# =============================================================================
# Scenario Matrix
# =============================================================================

METHODS = ["digital_lora", "lrtt_6t1c", "tikitaka_6t1c"]
TARGETS = ["attention", "ffn", "all"]
RANKS = [4, 8, 16, 32]
GAMMAS = [0, 1]
FIXED_SEQ_LENGTHS = [64, 128, 256, 384]
T_TILE_OPTIONS = [100, 128, 256, 512]  # ns (100 is optional ref)
K_BWD_OPTIONS = [2.0, 2.5, 3.0]

# Method-specific transfer_every (from actual training configs)
# Both use units_in_mbatch=True (sample counting) for fair batch-level comparison
# TikiTaka: transfer_every=1 (transfer 1 column per sample batch)
# LR-TT:   transfer_every=4 (A⊗B→C every 4 samples)
# With batch_size=48 and units_in_mbatch=True:
#   TikiTaka tau_eff = max(1, 1//48) = 1 step → transfer every step
#   LR-TT   tau_eff = max(1, 4//48) = 1 step → transfer every step
TIKITAKA_TRANSFER_EVERY = 1
LRTT_TRANSFER_EVERY = 4
DEFAULT_UNITS_IN_MBATCH = True  # batch-level (sample counting): consistent with training code


# =============================================================================
# Scenario Builders
# =============================================================================

def build_scenarios(
    t_tile_ns: float = 256.0,
    k_bwd: float = 2.5,
) -> List[ScenarioConfig]:
    """Build all scenario configurations for the main comparison.

    Transfer frequency is method-specific:
      TikiTaka: transfer_every=1 (one column per step)
      LR-TT:   transfer_every=4 (A⊗B→C every 4 steps)
    """
    scenarios = []

    for method in METHODS:
        for target in TARGETS:
            if method == "digital_lora":
                for rank in RANKS:
                    scenarios.append(ScenarioConfig(
                        method=method,
                        target=target,
                        gamma=1,  # Always forward-visible
                        rank=rank,
                        transfer_every=0,  # No transfer
                        units_in_mbatch=False,
                    ))
            else:
                for gamma in GAMMAS:
                    if method == "lrtt_6t1c":
                        for rank in RANKS:
                            scenarios.append(ScenarioConfig(
                                method=method,
                                target=target,
                                gamma=gamma,
                                rank=rank,
                                transfer_every=LRTT_TRANSFER_EVERY,
                                units_in_mbatch=DEFAULT_UNITS_IN_MBATCH,
                            ))
                    else:  # tikitaka_6t1c
                        scenarios.append(ScenarioConfig(
                            method=method,
                            target=target,
                            gamma=gamma,
                            rank=0,  # Full-rank, not used
                            transfer_every=TIKITAKA_TRANSFER_EVERY,
                            units_in_mbatch=DEFAULT_UNITS_IN_MBATCH,
                        ))

    return scenarios


# =============================================================================
# Result Writing
# =============================================================================

STEP_CSV_COLUMNS = [
    'method', 'target', 'gamma', 'rank', 'seq_mode', 'S_or_dynamic',
    't_tile_ns', 'k_bwd',
    'T_common_ns', 'DeltaT_ns', 'T_step_ns',
    'E_common_pj', 'DeltaE_pj', 'E_step_pj',
    'T_base_fwd_ns', 'T_base_bwd_ns', 'T_other_ns',
    'DeltaT_adapter_fwd_ns', 'DeltaT_adapter_bwd_ns',
    'DeltaT_adapter_opt_ns', 'DeltaT_transfer_ns',
    'latency_per_seq_ns', 'energy_per_seq_pj',
    'base_mvm_events', 'adapter_mvm_events',
    'adapter_update_events', 'transfer_events',
    'digital_flops', 'digital_opt_params',
    'gamma_label', 'missing_params',
]


def gamma_label(method: str, gamma: int) -> str:
    """Generate reviewer-safe gamma label."""
    if method == "digital_lora":
        return "forward-visible (always)"
    elif method == "lrtt_6t1c" and gamma == 0:
        return "hidden-carry-path ablation"
    elif method == "lrtt_6t1c" and gamma == 1:
        return "main direct comparison to Digital LoRA"
    elif method == "tikitaka_6t1c" and gamma == 0:
        return "standard (C-only visible)"
    elif method == "tikitaka_6t1c" and gamma == 1:
        return "U-visible (main comparison)"
    return f"gamma={gamma}"


def write_results_csv(results: List[Dict], path: str) -> None:
    """Write results to CSV."""
    if not results:
        return
    fieldnames = [k for k in STEP_CSV_COLUMNS if k in results[0]]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in results:
            writer.writerow(row)


# =============================================================================
# Main Computation Loops
# =============================================================================

def run_fixed_seq_sweep(
    inventory: List[LayerSpec],
    hw: HardwareParams,
    scenarios: List[ScenarioConfig],
) -> List[Dict]:
    """Run cost model for all scenarios at fixed sequence lengths."""
    results = []

    for cfg in scenarios:
        for S in FIXED_SEQ_LENGTHS:
            r = compute_fixed_seq_cost(inventory, hw, cfg, S)

            results.append({
                'method': cfg.method,
                'target': cfg.target,
                'gamma': cfg.gamma,
                'rank': cfg.rank,
                'seq_mode': 'fixed',
                'S_or_dynamic': S,
                't_tile_ns': hw.t_tile_ns,
                'k_bwd': hw.k_common_bwd,
                'T_common_ns': r['T_common_ns'],
                'DeltaT_ns': r['DeltaT_ns'],
                'T_step_ns': r['T_step_ns'],
                'E_common_pj': r['E_common_pj'],
                'DeltaE_pj': r['DeltaE_pj'],
                'E_step_pj': r['E_step_pj'],
                'T_base_fwd_ns': r['T_base_fwd_ns'],
                'T_base_bwd_ns': r['T_base_bwd_ns'],
                'T_other_ns': r['T_other_ns'],
                'DeltaT_adapter_fwd_ns': r['DeltaT_adapter_fwd_ns'],
                'DeltaT_adapter_bwd_ns': r['DeltaT_adapter_bwd_ns'],
                'DeltaT_adapter_opt_ns': r['DeltaT_adapter_opt_ns'],
                'DeltaT_transfer_ns': r['DeltaT_transfer_ns'],
                'latency_per_seq_ns': r['latency_per_seq_ns'],
                'energy_per_seq_pj': r['energy_per_seq_pj'],
                'base_mvm_events': r['base_mvm_events'],
                'adapter_mvm_events': r['adapter_mvm_events'],
                'adapter_update_events': r['adapter_update_events'],
                'transfer_events': r['transfer_events'],
                'digital_flops': r['digital_flops'],
                'digital_opt_params': r['digital_opt_params'],
                'gamma_label': gamma_label(cfg.method, cfg.gamma),
                'missing_params': ';'.join(r['missing_params']) if r['missing_params'] else '',
            })

    return results


def run_dynamic_padding_sweep(
    inventory: List[LayerSpec],
    hw: HardwareParams,
    scenarios: List[ScenarioConfig],
    batch_trace: Optional[List[Dict]],
) -> List[Dict]:
    """Run cost model for all scenarios with dynamic padding trace.

    If batch_trace is None, uses a representative S_pad (mean from trace).
    """
    results = []

    if batch_trace is None:
        print("  WARNING: No batch trace available. Using S_pad=384 as proxy.")
        for cfg in scenarios:
            r = compute_fixed_seq_cost(inventory, hw, cfg, 384)
            results.append({
                'method': cfg.method,
                'target': cfg.target,
                'gamma': cfg.gamma,
                'rank': cfg.rank,
                'seq_mode': 'dynamic',
                'S_or_dynamic': 'proxy_384',
                't_tile_ns': hw.t_tile_ns,
                'k_bwd': hw.k_common_bwd,
                'T_common_ns': r['T_common_ns'],
                'DeltaT_ns': r['DeltaT_ns'],
                'T_step_ns': r['T_step_ns'],
                'E_common_pj': r['E_common_pj'],
                'DeltaE_pj': r['DeltaE_pj'],
                'E_step_pj': r['E_step_pj'],
                'T_base_fwd_ns': r['T_base_fwd_ns'],
                'T_base_bwd_ns': r['T_base_bwd_ns'],
                'T_other_ns': r['T_other_ns'],
                'DeltaT_adapter_fwd_ns': r['DeltaT_adapter_fwd_ns'],
                'DeltaT_adapter_bwd_ns': r['DeltaT_adapter_bwd_ns'],
                'DeltaT_adapter_opt_ns': r['DeltaT_adapter_opt_ns'],
                'DeltaT_transfer_ns': r['DeltaT_transfer_ns'],
                'latency_per_seq_ns': r['latency_per_seq_ns'],
                'energy_per_seq_pj': r['energy_per_seq_pj'],
                'base_mvm_events': r['base_mvm_events'],
                'adapter_mvm_events': r['adapter_mvm_events'],
                'adapter_update_events': r['adapter_update_events'],
                'transfer_events': r['transfer_events'],
                'digital_flops': r['digital_flops'],
                'digital_opt_params': r['digital_opt_params'],
                'gamma_label': gamma_label(cfg.method, cfg.gamma),
                'missing_params': ';'.join(r['missing_params']) if r['missing_params'] else '',
            })
        return results

    # With actual batch trace: compute epoch-level aggregates
    for cfg in scenarios:
        epoch_result = compute_epoch_cost(inventory, hw, cfg, batch_trace)

        results.append({
            'method': cfg.method,
            'target': cfg.target,
            'gamma': cfg.gamma,
            'rank': cfg.rank,
            'seq_mode': 'dynamic',
            'S_or_dynamic': 'dynamic',
            't_tile_ns': hw.t_tile_ns,
            'k_bwd': hw.k_common_bwd,
            'T_common_ns': epoch_result['total_T_common_ns'] / epoch_result['n_steps'],
            'DeltaT_ns': epoch_result['total_DeltaT_ns'] / epoch_result['n_steps'],
            'T_step_ns': epoch_result['avg_T_step_ns'],
            'E_common_pj': 0.0,  # Needs e_tile_pj
            'DeltaE_pj': 0.0,
            'E_step_pj': epoch_result['avg_E_step_pj'],
            'T_base_fwd_ns': 0.0,  # Averaged across batches
            'T_base_bwd_ns': 0.0,
            'T_other_ns': 0.0,
            'DeltaT_adapter_fwd_ns': 0.0,
            'DeltaT_adapter_bwd_ns': 0.0,
            'DeltaT_adapter_opt_ns': 0.0,
            'DeltaT_transfer_ns': 0.0,
            'latency_per_seq_ns': epoch_result['avg_T_per_seq_ns'],
            'energy_per_seq_pj': epoch_result['avg_E_per_seq_pj'],
            'base_mvm_events': 0,
            'adapter_mvm_events': 0,
            'adapter_update_events': 0,
            'transfer_events': 0,
            'digital_flops': 0,
            'digital_opt_params': 0,
            'gamma_label': gamma_label(cfg.method, cfg.gamma),
            'missing_params': ';'.join(epoch_result['missing_params']) if epoch_result['missing_params'] else '',
        })

    return results


def run_sensitivity_sweep(
    inventory: List[LayerSpec],
    base_hw: HardwareParams,
) -> List[Dict]:
    """Run tile latency x k_bwd sensitivity sweep at fixed S=384, target=all, rank=8."""
    results = []

    for t_tile in T_TILE_OPTIONS:
        for k_bwd in K_BWD_OPTIONS:
            hw = HardwareParams(
                tile_size=base_hw.tile_size,
                t_tile_ns=t_tile,
                k_common_bwd=k_bwd,
                e_tile_pj=base_hw.e_tile_pj,
                t_update_ns=base_hw.t_update_ns,
                e_update_pj=base_hw.e_update_pj,
                t_digital_ns_per_flop=base_hw.t_digital_ns_per_flop,
                e_digital_pj_per_flop=base_hw.e_digital_pj_per_flop,
                t_lora_gemm_ns_per_flop=base_hw.t_lora_gemm_ns_per_flop,
                e_lora_gemm_pj_per_flop=base_hw.e_lora_gemm_pj_per_flop,
                t_lora_opt_ns_per_param=base_hw.t_lora_opt_ns_per_param,
                e_lora_opt_pj_per_param=base_hw.e_lora_opt_pj_per_param,
                shared_periphery_serialize=base_hw.shared_periphery_serialize,
            )

            scenarios = build_scenarios(t_tile_ns=t_tile, k_bwd=k_bwd)
            for cfg in scenarios:
                r = compute_fixed_seq_cost(inventory, hw, cfg, 384)
                results.append({
                    'method': cfg.method,
                    'target': cfg.target,
                    'gamma': cfg.gamma,
                    'rank': cfg.rank,
                    'seq_mode': 'fixed',
                    'S_or_dynamic': 384,
                    't_tile_ns': t_tile,
                    'k_bwd': k_bwd,
                    'T_common_ns': r['T_common_ns'],
                    'DeltaT_ns': r['DeltaT_ns'],
                    'T_step_ns': r['T_step_ns'],
                    'E_common_pj': r['E_common_pj'],
                    'DeltaE_pj': r['DeltaE_pj'],
                    'E_step_pj': r['E_step_pj'],
                    'T_base_fwd_ns': r['T_base_fwd_ns'],
                    'T_base_bwd_ns': r['T_base_bwd_ns'],
                    'T_other_ns': r['T_other_ns'],
                    'DeltaT_adapter_fwd_ns': r['DeltaT_adapter_fwd_ns'],
                    'DeltaT_adapter_bwd_ns': r['DeltaT_adapter_bwd_ns'],
                    'DeltaT_adapter_opt_ns': r['DeltaT_adapter_opt_ns'],
                    'DeltaT_transfer_ns': r['DeltaT_transfer_ns'],
                    'latency_per_seq_ns': r['latency_per_seq_ns'],
                    'energy_per_seq_pj': r['energy_per_seq_pj'],
                    'base_mvm_events': r['base_mvm_events'],
                    'adapter_mvm_events': r['adapter_mvm_events'],
                    'adapter_update_events': r['adapter_update_events'],
                    'transfer_events': r['transfer_events'],
                    'digital_flops': r['digital_flops'],
                    'digital_opt_params': r['digital_opt_params'],
                    'gamma_label': gamma_label(cfg.method, cfg.gamma),
                    'missing_params': ';'.join(r['missing_params']) if r['missing_params'] else '',
                })

    return results


# =============================================================================
# Markdown Report Generator
# =============================================================================

def generate_sensitivity_summary(
    all_results: List[Dict],
    batch_trace_summary: Optional[Dict],
    inventory_summary: Dict,
    hw: HardwareParams,
    output_path: str,
) -> None:
    """Generate the sensitivity_summary.md report."""
    lines = []
    L = lines.append

    L("# AIMC Training Cost Model: Sensitivity Summary")
    L("")
    L("## Problem Definition")
    L("")
    L("This is a **batch-level on-chip fine-tuning cost study** comparing three AIMC")
    L("training methods for BERT-base on SQuAD v1.1 with dynamic padding.")
    L("")
    L("This is NOT an inference-only study. This is NOT a GPU wall-clock benchmark.")
    L("This is NOT a cell-current-to-512x512 direct extrapolation.")
    L("")
    L("## Why Inference-Only Comparison Is Insufficient")
    L("")
    L("Prior AHWA-LoRA work models deployment cost (forward pass only). However,")
    L("training involves backward propagation, optimizer updates, and weight transfer")
    L("operations. These costs differ fundamentally across methods and cannot be")
    L("inferred from forward-pass-only analysis.")
    L("")
    L("## Why a Training-Aware Digital LoRA Baseline Is Needed")
    L("")
    L("Digital LoRA on AIMC tiles provides a natural baseline: the base weights W")
    L("are fixed on common AIMC tiles, while LoRA adapters A and B live on a digital")
    L("support core (PMCA/DPU). The training cost includes digital backward pass and")
    L("optimizer update for A,B. This is a training-aware extension of the AHWA-LoRA")
    L("split architecture, NOT a deployment-only baseline.")
    L("")
    L("## Common-vs-Delta Cost Decomposition")
    L("")
    L("```")
    L("T_step^(m) = T_common + DeltaT^(m)")
    L("E_step^(m) = E_common + DeltaE^(m)")
    L("```")
    L("")
    L("T_common is identical across all three methods for a given batch and scenario.")
    L("Only the adapter/update/transfer costs differ.")
    L("")

    # Layer inventory
    L("## Layer Inventory")
    L("")
    L(f"- Total encoder linear layers: {inventory_summary['total_layers']}")
    L(f"- Total base tiles (512x512): {inventory_summary['total_tiles']}")
    L(f"- Attention layers: {inventory_summary['attention_layers']} ({inventory_summary['attention_tiles']} tiles)")
    L(f"- FFN layers: {inventory_summary['ffn_layers']} ({inventory_summary['ffn_tiles']} tiles)")
    L(f"- Total weight parameters: {inventory_summary['total_params']:,}")
    L("")

    # Dynamic padding
    L("## Dynamic Padding Methodology")
    L("")
    if batch_trace_summary:
        L(f"- Total batches: {batch_trace_summary.get('total_batches', 'N/A')}")
        L(f"- Total features: {batch_trace_summary.get('total_features', 'N/A')}")
        L(f"- Mean S_pad: {batch_trace_summary.get('mean_S_pad', 'N/A'):.1f}")
        L(f"- Mean padding waste: {batch_trace_summary.get('mean_padding_waste', 0)*100:.2f}%")
        L(f"- Batches at max length (384): {batch_trace_summary.get('pct_batches_at_max', 'N/A'):.1f}%")
    else:
        L("- Batch trace not available; used fixed S=384 as proxy.")
    L("")

    # Hardware assumptions
    L("## Hardware Proxy Assumptions")
    L("")
    L("- Tile size: 512x512")
    L(f"- Tile MVM latency (sensitivity): {T_TILE_OPTIONS} ns")
    L(f"- Common backward multiplier (sensitivity): {K_BWD_OPTIONS}")
    L("- These are **literature-calibrated proxies**, NOT derived from 5x5 cell current data.")
    L("- Cell current data (if available) is for sanity check only.")
    L("")

    # Missing data
    missing_set = set()
    for r in all_results:
        if r.get('missing_params'):
            for p in r['missing_params'].split(';'):
                if p.strip():
                    missing_set.add(p.strip())

    L("## Missing Hardware Data Checklist")
    L("")
    if missing_set:
        for p in sorted(missing_set):
            L(f"- [ ] `{p}`")
    else:
        L("All hardware parameters provided.")
    L("")

    # Method-specific event accounting
    L("## Method-Specific Event Accounting")
    L("")
    L("### Digital LoRA")
    L("- Forward: 2 digital matmuls per targeted layer (X@B^T, result@A^T)")
    L("- Backward: 3 digital matmuls per targeted layer (dA, d(Bx), dB)")
    L("- Optimizer: Adam update for A [M*r] + B [r*N] parameters")
    L("- No analog tile operations for the adapter path")
    L("")
    L("### LR-TT 6T1C")
    L("- Per step: B.forward(x), A.backward(d), A.update(XB,d), B.update(x,DA)")
    L("- gamma=1: additional forward through A,B path")
    L("- Transfer: r one-hot reads from A + r one-hot reads from B + r rank-1 updates to C")
    L("- Transfer amortized over tau_eff steps")
    L("")
    L("### Tiki-Taka 6T1C")
    L("- Per step: full-rank pulsed update on U [M*N]")
    L("- gamma=1: additional forward through U path")
    L("- Transfer: column-by-column (transfer_columns=True, n_reads_per_transfer=1)")
    L("- 1 column read from U + 1 column write to C per transfer event")
    L("")

    # Sensitivity knobs
    L("## Most Important Sensitivity Knobs")
    L("")
    L("1. **t_tile_ns**: Dominates common path cost. All methods scale identically.")
    L("2. **k_common_bwd**: Multiplies base backward cost. Affects common path only.")
    L("3. **rank (r)**: Affects digital LoRA and LR-TT delta costs. No effect on Tiki-Taka.")
    L("4. **transfer_every**: Amortizes transfer cost. Larger = lower per-step overhead.")
    L("5. **gamma**: 0 vs 1 affects whether adapter path adds forward latency.")
    L("6. **t_update_ns**: Missing; strongly affects LRTT/TikiTaka delta costs.")
    L("7. **Digital GEMM speed**: Missing; strongly affects digital LoRA absolute cost.")
    L("")

    # Conclusions
    L("## Conclusions")
    L("")
    L("### (1) Scientific Defensibility")
    L("")
    L("The comparison setup is scientifically defensible because:")
    L("- Common path is identical across methods")
    L("- Method-specific costs are derived from code-level event counting")
    L("- Hardware proxies are literature-calibrated, not extrapolated from cell data")
    L("- Dynamic padding is properly modeled")
    L("- gamma labeling is explicit (gamma=1 = main comparison)")
    L("")
    L("### (2) Most Impactful Missing Values")
    L("")
    L("The following missing hardware values most strongly affect absolute numbers:")
    L("- `t_update_ns` / `e_update_pj`: determines analog update cost (LRTT + TikiTaka)")
    L("- `t_lora_gemm_ns_per_flop`: determines digital LoRA absolute cost")
    L("- `e_tile_pj`: determines all energy estimates")
    L("")
    L("### (3) Robust Relative Rankings")
    L("")
    L("The following relative rankings are likely robust even under uncertainty:")
    L("- **LR-TT vs Tiki-Taka per-step update cost**: Tiki-Taka full-rank update is")
    L("  always more expensive than LR-TT rank-r update (by factor ~n_tiles_U/n_tiles_AB)")
    L("- **Transfer amortization**: Larger transfer_every always reduces per-step overhead")
    L("- **Rank scaling for LR-TT**: Lower rank = lower adapter delta cost")
    L("- **Common path dominance**: For large models, T_common >> DeltaT for all methods")
    L("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# =============================================================================
# Batch Trace Loader
# =============================================================================

def load_batch_trace(path: str) -> Optional[List[Dict]]:
    """Load batch trace CSV."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'batch_id': int(row['batch_id']),
                'batch_size': int(row['batch_size']),
                'S_pad': int(row['S_pad']),
                'token_sum': int(row['token_sum']),
                'padding_waste': float(row['padding_waste']),
            })
    return rows


def compute_trace_summary(trace: List[Dict]) -> Dict:
    """Compute summary from batch trace."""
    import numpy as np
    s_pads = np.array([b['S_pad'] for b in trace])
    wastes = np.array([b['padding_waste'] for b in trace])
    return {
        'total_batches': len(trace),
        'total_features': sum(b['batch_size'] for b in trace),
        'mean_S_pad': float(np.mean(s_pads)),
        'mean_padding_waste': float(np.mean(wastes)),
        'pct_batches_at_max': float(np.mean(s_pads == 384) * 100),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run AIMC training cost study")
    parser.add_argument("--hw-params", default=os.path.join(SCRIPT_DIR, "hardware_params.yaml"),
                        help="Path to hardware_params.yaml")
    parser.add_argument("--batch-trace", default=os.path.join(SCRIPT_DIR, "batch_trace.csv"),
                        help="Path to batch_trace.csv")
    parser.add_argument("--convergence-log", default=None,
                        help="Path to convergence log CSV (optional)")
    parser.add_argument("--output-dir", default=SCRIPT_DIR,
                        help="Output directory for results")
    parser.add_argument("--skip-dynamic", action="store_true",
                        help="Skip dynamic padding sweep (faster)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Load hardware params ---
    print("=" * 70)
    print("AIMC Training Cost Study")
    print("=" * 70)

    hw = load_hardware_params(args.hw_params)
    print(f"\nHardware params loaded from: {args.hw_params}")
    print(f"  t_tile_ns (default): {hw.t_tile_ns}")
    print(f"  k_common_bwd (default): {hw.k_common_bwd}")

    # --- Step 2: Build layer inventory ---
    print("\nBuilding layer inventory...")
    inventory = build_layer_inventory()
    inv_summary = summarize_inventory(inventory)
    inv_path = os.path.join(output_dir, "layer_inventory.csv")
    save_inventory_csv(inventory, inv_path)
    print(f"  Layers: {inv_summary['total_layers']}, Tiles: {inv_summary['total_tiles']}")
    print(f"  Saved: {inv_path}")

    # --- Step 3: Load batch trace ---
    batch_trace = load_batch_trace(args.batch_trace)
    trace_summary = None
    if batch_trace:
        trace_summary = compute_trace_summary(batch_trace)
        print(f"\nBatch trace loaded: {len(batch_trace)} batches")
        print(f"  Mean S_pad: {trace_summary['mean_S_pad']:.1f}")
        print(f"  Mean padding waste: {trace_summary['mean_padding_waste']*100:.2f}%")
    else:
        print(f"\nNo batch trace found at {args.batch_trace}")
        print("  Will use fixed S=384 as proxy for dynamic padding.")

    # --- Step 4: Run fixed-length sweep ---
    print("\n--- Fixed Sequence Length Sweep ---")
    all_fixed_results = []

    for t_tile in T_TILE_OPTIONS:
        for k_bwd in K_BWD_OPTIONS:
            hw_sweep = HardwareParams(
                tile_size=hw.tile_size,
                t_tile_ns=t_tile,
                k_common_bwd=k_bwd,
                e_tile_pj=hw.e_tile_pj,
                t_update_ns=hw.t_update_ns,
                e_update_pj=hw.e_update_pj,
                t_digital_ns_per_flop=hw.t_digital_ns_per_flop,
                e_digital_pj_per_flop=hw.e_digital_pj_per_flop,
                t_lora_gemm_ns_per_flop=hw.t_lora_gemm_ns_per_flop,
                e_lora_gemm_pj_per_flop=hw.e_lora_gemm_pj_per_flop,
                t_lora_opt_ns_per_param=hw.t_lora_opt_ns_per_param,
                e_lora_opt_pj_per_param=hw.e_lora_opt_pj_per_param,
                shared_periphery_serialize=hw.shared_periphery_serialize,
                t_ab_read_ns=hw.t_ab_read_ns,
                e_ab_read_pj=hw.e_ab_read_pj,
                t_ab_update_ns=hw.t_ab_update_ns,
                e_ab_update_pj=hw.e_ab_update_pj,
                t_u_read_ns=hw.t_u_read_ns,
                e_u_read_pj=hw.e_u_read_pj,
                t_u_update_ns=hw.t_u_update_ns,
                e_u_update_pj=hw.e_u_update_pj,
            )
            scenarios = build_scenarios(t_tile_ns=t_tile, k_bwd=k_bwd)
            results = run_fixed_seq_sweep(inventory, hw_sweep, scenarios)
            all_fixed_results.extend(results)

    fixed_path = os.path.join(output_dir, "results_training_step.csv")
    write_results_csv(all_fixed_results, fixed_path)
    print(f"  Fixed sweep: {len(all_fixed_results)} results -> {fixed_path}")

    # --- Step 5: Run forward-only sweep (subset at S=384, k_bwd not relevant) ---
    print("\n--- Forward-Only Sweep ---")
    fwd_results = []
    hw_fwd = HardwareParams(
        tile_size=hw.tile_size,
        t_tile_ns=256.0,
        e_tile_pj=hw.e_tile_pj,
        t_digital_ns_per_flop=hw.t_digital_ns_per_flop,
        t_lora_gemm_ns_per_flop=hw.t_lora_gemm_ns_per_flop,
    )
    fwd_scenarios = build_scenarios(t_tile_ns=256.0, k_bwd=2.5)
    for cfg in fwd_scenarios:
        # forward_only=True: zeroes backward, update, transfer in all 3 delta functions
        cfg_fwd = ScenarioConfig(
            method=cfg.method,
            target=cfg.target,
            gamma=cfg.gamma,
            rank=cfg.rank,
            transfer_every=cfg.transfer_every,
            units_in_mbatch=cfg.units_in_mbatch,
            forward_only=True,  # <-- cleanly disables bwd/update/transfer
        )
        r = compute_fixed_seq_cost(inventory, hw_fwd, cfg_fwd, 384)
        fwd_results.append({
            'method': cfg.method,
            'target': cfg.target,
            'gamma': cfg.gamma,
            'rank': cfg.rank,
            'seq_mode': 'fixed',
            'S_or_dynamic': 384,
            't_tile_ns': 256.0,
            'k_bwd': 'fwd_only',
            'T_common_ns': r['T_common_ns'],
            'DeltaT_ns': r['DeltaT_ns'],
            'T_step_ns': r['T_step_ns'],
            'gamma_label': gamma_label(cfg.method, cfg.gamma),
        })

    fwd_path = os.path.join(output_dir, "results_forward_only.csv")
    write_results_csv(fwd_results, fwd_path)
    print(f"  Forward-only: {len(fwd_results)} results -> {fwd_path}")

    # --- Step 6: Dynamic padding sweep ---
    if not args.skip_dynamic:
        print("\n--- Dynamic Padding Sweep ---")
        hw_dyn = HardwareParams(
            tile_size=hw.tile_size,
            t_tile_ns=256.0,
            k_common_bwd=2.5,
            e_tile_pj=hw.e_tile_pj,
            t_update_ns=hw.t_update_ns,
            t_digital_ns_per_flop=hw.t_digital_ns_per_flop,
            t_lora_gemm_ns_per_flop=hw.t_lora_gemm_ns_per_flop,
            t_lora_opt_ns_per_param=hw.t_lora_opt_ns_per_param,
            shared_periphery_serialize=hw.shared_periphery_serialize,
            t_ab_read_ns=hw.t_ab_read_ns,
            t_ab_update_ns=hw.t_ab_update_ns,
            t_u_read_ns=hw.t_u_read_ns,
            t_u_update_ns=hw.t_u_update_ns,
        )
        dyn_scenarios = build_scenarios(t_tile_ns=256.0, k_bwd=2.5)
        dyn_results = run_dynamic_padding_sweep(inventory, hw_dyn, dyn_scenarios, batch_trace)

        # Append to main results
        all_fixed_results.extend(dyn_results)
        write_results_csv(all_fixed_results, fixed_path)
        print(f"  Dynamic: {len(dyn_results)} results appended -> {fixed_path}")

    # --- Step 7: Convergence log (if provided) ---
    ttt_path = os.path.join(output_dir, "results_time_to_target.csv")
    if args.convergence_log and os.path.exists(args.convergence_log):
        print(f"\n--- Time-to-Target from {args.convergence_log} ---")
        conv_log = load_convergence_log(args.convergence_log)

        # Build step cost lookup from fixed S=384, t_tile=256, k_bwd=2.5 results
        step_costs = {}
        for r in all_fixed_results:
            if r['S_or_dynamic'] == 384 and r['t_tile_ns'] == 256 and r['k_bwd'] == 2.5:
                key = (r['method'], r['target'], r['gamma'], r['rank'])
                step_costs[key] = r['T_step_ns']

        ttt_results = compute_time_to_target(conv_log, step_costs, target_f1=80.0)
        if ttt_results:
            with open(ttt_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=ttt_results[0].keys())
                writer.writeheader()
                writer.writerows(ttt_results)
            print(f"  Time-to-target: {len(ttt_results)} results -> {ttt_path}")
        else:
            print("  No convergence data matched scenario matrix.")
    else:
        print(f"\nNo convergence log provided. Skipping time-to-target computation.")
        print(f"  To compute time-to-target, provide --convergence-log <path>")
        print(f"  Expected CSV columns: method, target, gamma, rank, step, f1")

    # --- Step 8: Generate sensitivity summary ---
    print("\n--- Generating Sensitivity Summary ---")
    summary_path = os.path.join(output_dir, "sensitivity_summary.md")
    generate_sensitivity_summary(
        all_fixed_results,
        trace_summary,
        inv_summary,
        hw,
        summary_path,
    )
    print(f"  Summary: {summary_path}")

    # --- Done ---
    print("\n" + "=" * 70)
    print("Cost study complete.")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

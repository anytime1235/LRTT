#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract BERT-base encoder linear layer inventory for AIMC cost modeling.

Instantiates bert-base-uncased QA model, walks all encoder nn.Linear layers,
classifies them into target groups (attention / ffn), and computes tile counts
for 512x512 AIMC crossbar mapping.

Output: layer_inventory.csv
"""

import math
import csv
import os
import sys
import re
from dataclasses import dataclass, fields
from typing import List, Dict, Optional

import torch
from torch import nn


# =============================================================================
# Constants (from optuna_bert_squad_tiki.py)
# =============================================================================

MODEL_NAME = "bert-base-uncased"
DEFAULT_TILE_SIZE = 512

# Target group mapping (from optuna_bert_squad_tiki.py lines 118-126)
TARGET_GROUP_MAP = {
    "attention": ["query", "key", "value", "attention"],
    "ffn": ["intermediate", "output.dense"],
    "all": None,  # all encoder linears
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LayerSpec:
    """Specification for one encoder linear layer."""
    layer_idx: int          # 0..11
    module_name: str        # Full module name (e.g. "bert.encoder.layer.0.attention.self.query")
    sub_name: str           # Short name (e.g. "attention.self.query")
    group: str              # "attention" or "ffn"
    M: int                  # out_features (rows)
    N: int                  # in_features (cols)
    n_tile_rows: int        # ceil(M / tile_size)
    n_tile_cols: int        # ceil(N / tile_size)
    n_tiles: int            # n_tile_rows * n_tile_cols


# =============================================================================
# Functions
# =============================================================================

def tile_count(dim: int, tile_size: int = DEFAULT_TILE_SIZE) -> int:
    """Number of tiles needed to cover one dimension."""
    return math.ceil(dim / tile_size)


def classify_encoder_layer(layer_name: str) -> str:
    """Classify BERT-base encoder Linear layer into target group.

    Replicates _classify_encoder_layer() from optuna_bert_squad_tiki.py:376-385.

    BERT-base encoder layer structure (per block, x12):
        attention:  attention.self.query/key/value, attention.output.dense (W_O)
        ffn:        intermediate.dense, output.dense
    """
    if 'attention' in layer_name:
        return 'attention'
    return 'ffn'


def is_encoder_linear(name: str) -> bool:
    """Check if a named module is an encoder linear layer."""
    return 'encoder' in name and 'layer' in name


def extract_layer_index(name: str) -> Optional[int]:
    """Extract encoder layer index (0-11) from module name."""
    m = re.search(r'layer\.(\d+)', name)
    return int(m.group(1)) if m else None


def get_sub_name(name: str) -> str:
    """Extract short sub-layer name from full module path.

    e.g. "bert.encoder.layer.0.attention.self.query" -> "attention.self.query"
    """
    m = re.search(r'layer\.\d+\.(.+)', name)
    return m.group(1) if m else name


def build_layer_inventory(tile_size: int = DEFAULT_TILE_SIZE) -> List[LayerSpec]:
    """Instantiate BERT-base QA model and extract encoder linear layer inventory.

    Returns:
        List of LayerSpec for all 72 encoder linear layers.
    """
    from transformers import AutoModelForQuestionAnswering

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    inventory = []
    always_digital = ["qa_outputs", "pooler"]

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(d in name for d in always_digital):
            continue
        if not is_encoder_linear(name):
            continue

        layer_idx = extract_layer_index(name)
        if layer_idx is None:
            continue

        sub_name = get_sub_name(name)
        group = classify_encoder_layer(name)
        M = module.out_features
        N = module.in_features
        tr = tile_count(M, tile_size)
        tc = tile_count(N, tile_size)

        inventory.append(LayerSpec(
            layer_idx=layer_idx,
            module_name=name,
            sub_name=sub_name,
            group=group,
            M=M,
            N=N,
            n_tile_rows=tr,
            n_tile_cols=tc,
            n_tiles=tr * tc,
        ))

    # Sort by layer index, then by sub_name for consistent ordering
    inventory.sort(key=lambda x: (x.layer_idx, x.sub_name))

    return inventory


def get_targeted_layers(
    inventory: List[LayerSpec],
    target: str,
) -> List[LayerSpec]:
    """Filter inventory to targeted layers.

    Args:
        inventory: Full layer inventory.
        target: "attention", "ffn", or "all".
    """
    if target == "all":
        return list(inventory)
    return [l for l in inventory if l.group == target]


def compute_adapter_tile_counts(
    layer: LayerSpec,
    rank: int,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> Dict[str, int]:
    """Compute adapter tile counts for a given layer and rank.

    Returns dict with:
        n_tiles_A: tiles for A matrix [M, rank]
        n_tiles_B: tiles for B matrix [rank, N]
        n_tiles_A_rows, n_tiles_A_cols, n_tiles_B_rows, n_tiles_B_cols
    """
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


def summarize_inventory(inventory: List[LayerSpec]) -> Dict:
    """Compute summary statistics for the layer inventory."""
    total_tiles = sum(l.n_tiles for l in inventory)
    attn_layers = [l for l in inventory if l.group == 'attention']
    ffn_layers = [l for l in inventory if l.group == 'ffn']

    return {
        'total_layers': len(inventory),
        'total_tiles': total_tiles,
        'attention_layers': len(attn_layers),
        'attention_tiles': sum(l.n_tiles for l in attn_layers),
        'ffn_layers': len(ffn_layers),
        'ffn_tiles': sum(l.n_tiles for l in ffn_layers),
        'total_params': sum(l.M * l.N for l in inventory),
    }


def save_inventory_csv(inventory: List[LayerSpec], path: str) -> None:
    """Save layer inventory to CSV."""
    fieldnames = [f.name for f in fields(LayerSpec)]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer in inventory:
            writer.writerow({f.name: getattr(layer, f.name) for f in fields(LayerSpec)})


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "layer_inventory.csv")

    print(f"Extracting BERT-base encoder linear layer inventory...")
    print(f"Model: {MODEL_NAME}")
    print(f"Tile size: {DEFAULT_TILE_SIZE}")

    inventory = build_layer_inventory()

    # Print summary
    summary = summarize_inventory(inventory)
    print(f"\nInventory Summary:")
    print(f"  Total encoder linear layers: {summary['total_layers']}")
    print(f"  Total base tiles (512x512):  {summary['total_tiles']}")
    print(f"  Attention layers:            {summary['attention_layers']} ({summary['attention_tiles']} tiles)")
    print(f"  FFN layers:                  {summary['ffn_layers']} ({summary['ffn_tiles']} tiles)")
    print(f"  Total weight params:         {summary['total_params']:,}")

    # Print per-layer details
    print(f"\nPer-layer details:")
    print(f"{'Layer':>3} {'Sub-name':<30} {'Group':<10} {'M':>5} {'N':>5} {'Tiles':>5}")
    print("-" * 65)
    for layer in inventory:
        print(f"{layer.layer_idx:>3} {layer.sub_name:<30} {layer.group:<10} "
              f"{layer.M:>5} {layer.N:>5} {layer.n_tiles:>5}")

    # Print adapter tile counts for various ranks
    print(f"\nAdapter tile counts per layer (rank -> [A tiles, B tiles]):")
    unique_shapes = set((l.M, l.N) for l in inventory)
    for M, N in sorted(unique_shapes):
        print(f"  [{M}x{N}]:", end="")
        example = next(l for l in inventory if l.M == M and l.N == N)
        for rank in [4, 8, 16, 32]:
            atc = compute_adapter_tile_counts(example, rank)
            print(f"  r={rank}: A={atc['n_tiles_A']}, B={atc['n_tiles_B']}", end="")
        print()

    # Save
    save_inventory_csv(inventory, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

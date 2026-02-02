#!/usr/bin/env python3
"""Compare trainable parameters: FullAnalog vs LRTT (Standard & Spatial) for different ranks."""

import torch
from torch import nn

# ResNet18 configuration for CIFAR-10
# Layer structure: conv1(digital) + 4 layers + fc(digital)
# Each layer has 2 blocks, each block has 2 convs
# Layer2, Layer3, Layer4 have downsample convs (1×1)

conv_layers = [
    # Layer1: 64→64
    {'name': 'layer1.0.conv1', 'in': 64, 'out': 64, 'k': 3},
    {'name': 'layer1.0.conv2', 'in': 64, 'out': 64, 'k': 3},
    {'name': 'layer1.1.conv1', 'in': 64, 'out': 64, 'k': 3},
    {'name': 'layer1.1.conv2', 'in': 64, 'out': 64, 'k': 3},

    # Layer2: 64→128 (with downsample)
    {'name': 'layer2.0.conv1', 'in': 64, 'out': 128, 'k': 3},
    {'name': 'layer2.0.conv2', 'in': 128, 'out': 128, 'k': 3},
    {'name': 'layer2.0.downsample', 'in': 64, 'out': 128, 'k': 1},
    {'name': 'layer2.1.conv1', 'in': 128, 'out': 128, 'k': 3},
    {'name': 'layer2.1.conv2', 'in': 128, 'out': 128, 'k': 3},

    # Layer3: 128→256 (with downsample)
    {'name': 'layer3.0.conv1', 'in': 128, 'out': 256, 'k': 3},
    {'name': 'layer3.0.conv2', 'in': 256, 'out': 256, 'k': 3},
    {'name': 'layer3.0.downsample', 'in': 128, 'out': 256, 'k': 1},
    {'name': 'layer3.1.conv1', 'in': 256, 'out': 256, 'k': 3},
    {'name': 'layer3.1.conv2', 'in': 256, 'out': 256, 'k': 3},

    # Layer4: 256→512 (with downsample)
    {'name': 'layer4.0.conv1', 'in': 256, 'out': 512, 'k': 3},
    {'name': 'layer4.0.conv2', 'in': 512, 'out': 512, 'k': 3},
    {'name': 'layer4.0.downsample', 'in': 256, 'out': 512, 'k': 1},
    {'name': 'layer4.1.conv1', 'in': 512, 'out': 512, 'k': 3},
    {'name': 'layer4.1.conv2', 'in': 512, 'out': 512, 'k': 3},
]

def count_fullanalog_params():
    """Count FullAnalog trainable parameters (C matrices only)."""
    total = 0
    for layer in conv_layers:
        c_in, c_out, k = layer['in'], layer['out'], layer['k']
        # C matrix: [c_out, c_in×k×k]
        params = c_out * c_in * k * k
        total += params
    return total

def count_standard_lora_params(rank):
    """Count Standard LoRA trainable parameters (A + B matrices)."""
    total = 0
    for layer in conv_layers:
        c_in, c_out, k = layer['in'], layer['out'], layer['k']
        # A matrix: [c_out, rank]
        # B matrix: [rank, c_in×k×k]
        params_A = c_out * rank
        params_B = rank * c_in * k * k
        total += params_A + params_B
    return total

def count_spatial_lora_params(rank):
    """Count Spatial LoRA-C trainable parameters (A + B matrices in spatial format)."""
    total = 0
    for layer in conv_layers:
        c_in, c_out, k = layer['in'], layer['out'], layer['k']
        # A matrix: [c_out×k, rank×k]
        # B matrix: [rank×k, c_in×k]
        spatial_rank = rank * k
        params_A = (c_out * k) * spatial_rank
        params_B = spatial_rank * (c_in * k)
        total += params_A + params_B
    return total

# Calculate for different ranks
ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]

fullanalog_params = count_fullanalog_params()

print("="*120)
print("Parameter Count Comparison: FullAnalog vs LRTT (Standard & Spatial LoRA-C)")
print("="*120)
print(f"ResNet18 for CIFAR-10: {len(conv_layers)} analog convolutional layers")
print()

# Print header
print(f"{'Rank':>6} | {'FullAnalog':>12} | {'Standard LoRA':>15} | {'Reduction':>10} | {'Spatial LoRA-C':>17} | {'Reduction':>10} | {'Spatial/Standard':>17}")
print("-" * 120)

for rank in ranks:
    standard_params = count_standard_lora_params(rank)
    spatial_params = count_spatial_lora_params(rank)

    standard_reduction = fullanalog_params / standard_params if standard_params > 0 else float('inf')
    spatial_reduction = fullanalog_params / spatial_params if spatial_params > 0 else float('inf')
    spatial_ratio = spatial_params / standard_params if standard_params > 0 else 1.0

    print(f"{rank:>6} | {fullanalog_params:>12,} | {standard_params:>15,} | {standard_reduction:>9.2f}× | {spatial_params:>17,} | {spatial_reduction:>9.2f}× | {spatial_ratio:>16.2f}×")

print("-" * 120)
print()

# Detailed breakdown for rank=8
print("="*120)
print("Detailed Breakdown for Rank=8")
print("="*120)
print(f"{'Layer':^30} | {'c_in':>5} × {'c_out':>5} × {'k':>3} | {'FullAnalog':>12} | {'Standard':>12} | {'Spatial':>12}")
print("-" * 120)

rank = 8
total_fullanalog = 0
total_standard = 0
total_spatial = 0

for layer in conv_layers:
    c_in, c_out, k = layer['in'], layer['out'], layer['k']

    # FullAnalog
    params_full = c_out * c_in * k * k

    # Standard LoRA
    params_std = c_out * rank + rank * c_in * k * k

    # Spatial LoRA-C
    spatial_rank = rank * k
    params_spatial = (c_out * k) * spatial_rank + spatial_rank * (c_in * k)

    total_fullanalog += params_full
    total_standard += params_std
    total_spatial += params_spatial

    print(f"{layer['name']:^30} | {c_in:>5} × {c_out:>5} × {k:>3} | {params_full:>12,} | {params_std:>12,} | {params_spatial:>12,}")

print("-" * 120)
print(f"{'TOTAL':^30} | {' ':>5}   {' ':>5}   {' ':>3} | {total_fullanalog:>12,} | {total_standard:>12,} | {total_spatial:>12,}")
print(f"{'':^30} | {' ':>5}   {' ':>5}   {' ':>3} | {'':>12} | {total_fullanalog/total_standard:>11.2f}× | {total_fullanalog/total_spatial:>11.2f}×")

print()
print("="*120)
print("Key Observations:")
print("="*120)
print("1. FullAnalog: Stores full C matrices for all layers")
print("2. Standard LoRA: A[c_out, rank] + B[rank, c_in×k×k]")
print("   - Lower parameter count for small ranks")
print("   - But rank is limited (can't capture full expressiveness)")
print()
print("3. Spatial LoRA-C: A[c_out×k, rank×k] + B[rank×k, c_in×k]")
print("   - Higher effective rank (rank × k)")
print("   - More parameters than Standard LoRA (spatial_params ≈ k² × standard_params for same rank)")
print("   - Better spatial structure preservation")
print()
print(f"For rank=8:")
print(f"  - Standard LoRA: {total_fullanalog/total_standard:.2f}× parameter reduction")
print(f"  - Spatial LoRA-C: {total_fullanalog/total_spatial:.2f}× parameter reduction")
print(f"  - Spatial uses {total_spatial/total_standard:.2f}× more parameters than Standard")
print(f"  - But Spatial has effective rank of {8}×k while Standard has rank {8}")
print("="*120)

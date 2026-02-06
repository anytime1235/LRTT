#!/usr/bin/env python
# coding=utf-8
"""Layer utilities for single layer comparison experiments."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


def list_linear_layers(module: nn.Module, parent_name: str = '') -> List[str]:
    """Recursively list all Linear layers in the module and return their names."""
    linear_layers = []
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, nn.Linear):
            linear_layers.append(full_name)
        else:
            child_layers = list_linear_layers(child, full_name)
            linear_layers.extend(child_layers)
    return linear_layers


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Get a layer from model by its dot-separated name."""
    parts = layer_name.split('.')
    current = model
    for part in parts:
        current = getattr(current, part)
    return current


def set_layer_by_name(model: nn.Module, layer_name: str, new_layer: nn.Module) -> None:
    """Set a layer in model by its dot-separated name."""
    parts = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_layer)


def freeze_all_except(model: nn.Module, target_layer_name: str) -> None:
    """Freeze all parameters except those in the target layer."""
    for name, param in model.named_parameters():
        if target_layer_name not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_params(model: nn.Module, detailed: bool = False) -> None:
    """Print trainable parameter summary."""
    trainable, total = count_trainable_params(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if detailed:
        print("\nTrainable parameters by name:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} ({param.shape})")


class LoRALayer(nn.Module):
    """Digital LoRA wrapper for a Linear layer.

    Implements: y = base_layer(x) + scaling * dropout(B(A(x)))

    where:
        - base_layer: Original Linear layer (can be analog)
        - A: Linear(in_features, r, bias=False)
        - B: Linear(r, out_features, bias=False)
        - scaling = lora_alpha / r
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Get dimensions from base layer
        if hasattr(base_layer, 'in_features'):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        elif hasattr(base_layer, 'in_size'):
            # AnalogLinear uses in_size/out_size
            in_features = base_layer.in_size
            out_features = base_layer.out_size
        else:
            raise ValueError(f"Cannot determine dimensions from {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

        # LoRA A and B matrices (digital)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize LoRA weights
        self._init_lora_weights()

    def _init_lora_weights(self):
        """Initialize LoRA weights: A with Kaiming, B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base + scaling * B(A(x))"""
        # Base layer output (can be analog)
        base_out = self.base_layer(x)

        # LoRA delta: scaling * B(A(dropout(x)))
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = self.scaling * lora_out

        return base_out + lora_out

    def get_effective_weight(self) -> torch.Tensor:
        """Get effective weight matrix: W_eff = C + scaling * B @ A"""
        with torch.no_grad():
            # Get C weight
            if hasattr(self.base_layer, 'get_weights'):
                # AnalogLinear
                C = self.base_layer.get_weights()[0]
            else:
                C = self.base_layer.weight

            # LoRA delta: scaling * B @ A
            delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)

            return C + delta


def create_lora_wrapper(
    layer: nn.Module,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
) -> LoRALayer:
    """Create a LoRA wrapper for a given layer."""
    return LoRALayer(layer, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

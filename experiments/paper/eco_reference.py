#!/usr/bin/env python
# coding=utf-8
"""Digital ECO (Error-Compensated Optimizer) reference implementation.

NOT an AIHWKit analog tile. This is a pure digital gold-standard that performs
weight-level error-feedback quantization after each optimizer step.

Algorithm (per target layer, per step):
  1. w_updated = module.weight.data           (FP32 after Adam step)
  2. w_corrected = w_updated + e_t            (inject accumulated error)
  3. w_hat = Q(w_corrected)                   (quantize to N-bit)
  4. e_{t+1} = w_corrected - w_hat            (store new error)
  5. module.weight.data.copy_(w_hat)           (set visible weight)

Quantization: Uniform symmetric, dw_min = 2*w_max / 2^n_bits,
              clamp to [-w_max, w_max].
"""

import torch
import torch.nn as nn


def quantize_rtn(weight, w_max, n_bits):
    """Round-to-nearest uniform symmetric quantization.

    Args:
        weight: FP32 tensor to quantize.
        w_max: Symmetric clamp range [-w_max, w_max].
        n_bits: Number of quantization bits.

    Returns:
        Quantized tensor (FP32 dtype, discrete levels).
    """
    n_levels = 2 ** n_bits
    dw_min = 2.0 * w_max / n_levels
    clamped = weight.clamp(-w_max, w_max)
    # Shift to [0, n_levels], round, shift back
    scaled = (clamped + w_max) / dw_min
    rounded = scaled.round()
    rounded = rounded.clamp(0, n_levels - 1)
    return rounded * dw_min - w_max


def quantize_stochastic(weight, w_max, n_bits):
    """Stochastic rounding uniform symmetric quantization.

    Like RTN but instead of rounding to nearest, the floor/ceil choice
    is stochastic with probability proportional to the fractional part.

    Args:
        weight: FP32 tensor to quantize.
        w_max: Symmetric clamp range [-w_max, w_max].
        n_bits: Number of quantization bits.

    Returns:
        Quantized tensor (FP32 dtype, discrete levels).
    """
    n_levels = 2 ** n_bits
    dw_min = 2.0 * w_max / n_levels
    clamped = weight.clamp(-w_max, w_max)
    scaled = (clamped + w_max) / dw_min
    floored = scaled.floor()
    frac = scaled - floored
    # Stochastic: round up with probability = frac
    rounded = floored + (torch.rand_like(frac) < frac).float()
    rounded = rounded.clamp(0, n_levels - 1)
    return rounded * dw_min - w_max


_QUANTIZERS = {
    "stochastic": quantize_stochastic,
    "rtn": quantize_rtn,
}


class EcoQuantizer:
    """Error-Compensated Optimizer quantizer for target nn.Linear layers.

    Manages per-layer error buffers (FP32 on CPU) and applies weight-level
    error-feedback quantization after each optimizer step.

    Args:
        model: nn.Module with target layers already identified.
        target_layer_names: List of full dotted names for layers to quantize.
        n_bits: Quantization bit-width. Default 10.
        w_max: Symmetric weight range. Default 1.0.
        rounding: 'stochastic' or 'rtn'. Default 'stochastic'.
    """

    def __init__(self, model, target_layer_names, n_bits=10, w_max=1.0,
                 rounding="stochastic"):
        self.n_bits = n_bits
        self.w_max = w_max
        self.dw_min = 2.0 * w_max / (2 ** n_bits)
        self.rounding = rounding
        self._quantize_fn = _QUANTIZERS[rounding]

        # Resolve target modules
        self.targets = {}  # name -> nn.Linear module
        module_dict = dict(model.named_modules())
        for name in target_layer_names:
            if name in module_dict and isinstance(module_dict[name], nn.Linear):
                self.targets[name] = module_dict[name]

        # Error buffers on CPU (FP32, same shape as weight)
        self.error_buffers = {}
        for name, module in self.targets.items():
            self.error_buffers[name] = torch.zeros_like(
                module.weight.data, device="cpu"
            )

        # Quantize initial weights (no error injection for init)
        self._quantize_initial_weights()

        print(f"  EcoQuantizer: {len(self.targets)} layers, "
              f"{n_bits}-bit, w_max={w_max}, rounding={rounding}")
        mem_mb = sum(e.numel() * 4 for e in self.error_buffers.values()) / 1e6
        print(f"  Error buffer memory: {mem_mb:.1f} MB (CPU)")

    def _quantize_initial_weights(self):
        """Quantize initial weights without error injection."""
        for name, module in self.targets.items():
            w = module.weight.data
            device = w.device
            w_q = self._quantize_fn(w, self.w_max, self.n_bits)
            # Store initial quantization error
            self.error_buffers[name] = (w - w_q).cpu()
            module.weight.data.copy_(w_q)

    @torch.no_grad()
    def post_step(self):
        """Apply ECO error-feedback quantization after optimizer.step().

        For each target layer:
          1. w_updated = module.weight.data
          2. w_corrected = w_updated + error_buffer
          3. w_hat = quantize(w_corrected)
          4. error_buffer = w_corrected - w_hat
          5. module.weight.data = w_hat
        """
        for name, module in self.targets.items():
            w_updated = module.weight.data  # FP32 on GPU
            device = w_updated.device

            # Move error to GPU briefly
            e_t = self.error_buffers[name].to(device)

            # Error-corrected weight
            w_corrected = w_updated + e_t

            # Quantize
            w_hat = self._quantize_fn(w_corrected, self.w_max, self.n_bits)

            # New error
            e_new = w_corrected - w_hat

            # Store
            self.error_buffers[name] = e_new.cpu()
            module.weight.data.copy_(w_hat)

    def get_weights(self, name):
        """Get quantized and pre-quantization weights for a named layer.

        Returns:
            (w_quantized, w_pre_quant) where w_pre_quant = w_quantized + error.
        """
        if name not in self.targets:
            raise KeyError(f"Layer '{name}' not in EcoQuantizer targets")
        module = self.targets[name]
        w_q = module.weight.data.clone()
        e = self.error_buffers[name].to(w_q.device)
        w_pre = w_q + e
        return w_q, w_pre

    def get_error_stats(self):
        """Get error buffer statistics for logging.

        Returns:
            Dict mapping layer name to {mean_abs, max_abs, rms}.
        """
        stats = {}
        for name, e in self.error_buffers.items():
            e_abs = e.abs()
            stats[name] = {
                "mean_abs": e_abs.mean().item(),
                "max_abs": e_abs.max().item(),
                "rms": e.pow(2).mean().sqrt().item(),
            }
        return stats

    def get_all_target_names(self):
        """Return list of all target layer names."""
        return list(self.targets.keys())

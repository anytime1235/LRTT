#!/usr/bin/env python
# coding=utf-8
"""Exact tile-level update diagnostics using AnalogContext internals.

Critical timing (must be called in this order):
  1. loss.backward()          -> AnalogContext stores analog_input + analog_grad_output
  2. diag.snapshot_before()   -> Capture x, d from AnalogContext BEFORE step clears them
  3. optimizer.step()         -> AnalogContext.reset() clears analog_input/grad_output
  4. diag.snapshot_after()    -> Compare w_after vs w_before

For BERT AnalogLinear (non-indexed case):
  x_input: cat(ctx.analog_input, axis=-1 if in_trans else 0)
  d_input: cat(ctx.analog_grad_output, axis=-1 if out_trans else 0)
  Target: -lr * d_input.T @ x_input -> shape [out_features, in_features]
"""

import os
import csv
import json
from collections import defaultdict

import torch
import numpy as np

from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext


class UpdateDiagnostics:
    """Exact per-tile update diagnostics.

    Tracks target (FP32) vs actual (pulsed) weight updates per AnalogLinear module.
    Uses wrapper-level tracking (not sub-tile level) since AnalogContext is shared.
    """

    def __init__(self, model, dw_min):
        self.dw_min = dw_min
        self.tile_registry = self._build_registry(model)
        self.cumulative_target = {}
        self.cumulative_actual = {}
        self.records = []
        self._before_cache = {}

        # Initialize cumulative trackers
        for entry in self.tile_registry:
            key = entry["key"]
            self.cumulative_target[key] = None
            self.cumulative_actual[key] = None

    @staticmethod
    def _build_registry(model):
        """Build registry of AnalogLinear modules with their contexts."""
        registry = []
        for name, module in model.named_modules():
            if not isinstance(module, AnalogLinear):
                continue

            # Get the AnalogContext parameter for this module
            analog_ctx = None
            for pname, param in module.named_parameters():
                if isinstance(param, AnalogContext):
                    analog_ctx = param
                    break

            if analog_ctx is None:
                continue

            # Determine subtype
            subtype = "unknown"
            if "query" in name:
                subtype = "query"
            elif "key" in name:
                subtype = "key"
            elif "value" in name:
                subtype = "value"
            elif "dense" in name:
                subtype = "dense"

            # Get the tile from the analog module
            tile = module.analog_module

            registry.append({
                "key": name,
                "module_name": name,
                "module": module,
                "tile": tile,
                "analog_ctx": analog_ctx,
                "subtype": subtype,
            })
        return registry

    def snapshot_before_step(self, model, optimizer):
        """Capture weights, activations, and gradients BEFORE optimizer.step().

        Must be called after loss.backward() but before optimizer.step().
        """
        lr = _get_current_analog_lr(optimizer)
        self._before_cache = {}

        for entry in self.tile_registry:
            key = entry["key"]
            tile = entry["tile"]
            ctx = entry["analog_ctx"]

            try:
                w_before = tile.get_weights()[0].clone().detach().cpu()
            except Exception:
                continue

            # Extract x, d from AnalogContext (available after backward, before step)
            x_input = None
            d_input = None
            if ctx.analog_input and ctx.analog_grad_output:
                try:
                    in_trans = tile.in_trans if hasattr(tile, 'in_trans') else False
                    out_trans = tile.out_trans if hasattr(tile, 'out_trans') else False
                    x_input = torch.cat(
                        ctx.analog_input, dim=-1 if in_trans else 0
                    ).detach().cpu().float()
                    d_input = torch.cat(
                        ctx.analog_grad_output, dim=-1 if out_trans else 0
                    ).detach().cpu().float()
                except Exception:
                    pass

            self._before_cache[key] = {
                "w_before": w_before,
                "x_input": x_input,
                "d_input": d_input,
                "lr": lr,
            }

    def snapshot_after_step(self, model, step):
        """Capture weights AFTER optimizer.step() and compute metrics.

        Must be called after optimizer.step().
        """
        for entry in self.tile_registry:
            key = entry["key"]
            tile = entry["tile"]

            if key not in self._before_cache:
                continue
            cache = self._before_cache[key]

            try:
                w_after = tile.get_weights()[0].clone().detach().cpu()
            except Exception:
                continue

            w_before = cache["w_before"]
            x_input = cache["x_input"]
            d_input = cache["d_input"]
            lr = cache["lr"]

            delta_actual = (w_after - w_before).float()

            # Compute FP32 target update
            delta_target = None
            if x_input is not None and d_input is not None:
                # Flatten to 2D: [N, features] — handles 3D indexed tiles
                if x_input.ndim > 2:
                    x_input = x_input.reshape(-1, x_input.shape[-1])
                if d_input.ndim > 2:
                    d_input = d_input.reshape(-1, d_input.shape[-1])

                # x_input: [N_x, in_features], d_input: [N_d, out_features]
                n_x = x_input.shape[0]
                n_d = d_input.shape[0]
                if n_x != n_d:
                    x_mean = x_input.mean(dim=0)  # [in_features]
                    d_mean = d_input.mean(dim=0)  # [out_features]
                    delta_target = -lr * torch.outer(d_mean, x_mean)
                else:
                    # Standard: -lr * d.T @ x / N
                    delta_target = -lr * (d_input.mT @ x_input) / n_x

                # Truncate to matching shape if tile has different size
                min_d = min(delta_target.shape[0], delta_actual.shape[0])
                min_x = min(delta_target.shape[1], delta_actual.shape[1])
                delta_target = delta_target[:min_d, :min_x]
                delta_actual_matched = delta_actual[:min_d, :min_x]
            else:
                delta_actual_matched = delta_actual

            # Compute metrics
            actual_norm = torch.norm(delta_actual).item()

            record = {
                "step": step,
                "tile_name": key,
                "subtype": entry["subtype"],
                "actual_norm": actual_norm,
                "lr": lr,
            }

            if delta_target is not None:
                residual = delta_actual_matched - delta_target

                target_norm = torch.norm(delta_target).item()
                residual_norm = torch.norm(residual).item()

                # Cosine similarity
                flat_actual = delta_actual_matched.flatten()
                flat_target = delta_target.flatten()
                cos_denom = (torch.norm(flat_actual) * torch.norm(flat_target))
                if cos_denom > 0:
                    cosine_sim = (flat_actual @ flat_target / cos_denom).item()
                else:
                    cosine_sim = 0.0

                # Signed bias: mean((actual - target) * sign(target))
                sign_target = torch.sign(delta_target)
                signed_bias = (residual * sign_target).mean().item()

                # Zero fraction: elements where |actual| < 0.5 * dw_min
                zero_frac = (delta_actual_matched.abs() < 0.5 * self.dw_min).float().mean().item()

                # Cumulative recovery
                if self.cumulative_target[key] is None:
                    self.cumulative_target[key] = delta_target.clone()
                    self.cumulative_actual[key] = delta_actual_matched.clone()
                else:
                    self.cumulative_target[key] += delta_target
                    self.cumulative_actual[key] += delta_actual_matched

                cum_target_norm = torch.norm(self.cumulative_target[key]).item()
                cum_actual_norm = torch.norm(self.cumulative_actual[key]).item()
                recovery_ratio = cum_actual_norm / max(cum_target_norm, 1e-12)

                record.update({
                    "target_norm": target_norm,
                    "residual_norm": residual_norm,
                    "cosine_sim": cosine_sim,
                    "signed_bias": signed_bias,
                    "zero_frac": zero_frac,
                    "recovery_ratio": recovery_ratio,
                })
            else:
                record.update({
                    "target_norm": 0.0,
                    "residual_norm": 0.0,
                    "cosine_sim": 0.0,
                    "signed_bias": 0.0,
                    "zero_frac": 0.0,
                    "recovery_ratio": 0.0,
                })

            self.records.append(record)

        self._before_cache = {}

    def save(self, output_dir):
        """Save diagnostics to CSV and JSON summary."""
        os.makedirs(output_dir, exist_ok=True)

        # CSV: per-step, per-tile metrics
        csv_path = os.path.join(output_dir, "update_diagnostics.csv")
        if self.records:
            fieldnames = list(self.records[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.records:
                    writer.writerow(record)

        # JSON: summary statistics per tile
        summary = {}
        tile_records = defaultdict(list)
        for r in self.records:
            tile_records[r["tile_name"]].append(r)

        for tile_name, recs in tile_records.items():
            if not recs:
                continue
            summary[tile_name] = {
                "n_steps": len(recs),
                "subtype": recs[0]["subtype"],
                "mean_target_norm": float(np.mean([r["target_norm"] for r in recs])),
                "mean_actual_norm": float(np.mean([r["actual_norm"] for r in recs])),
                "mean_residual_norm": float(np.mean([r["residual_norm"] for r in recs])),
                "mean_cosine_sim": float(np.mean([r["cosine_sim"] for r in recs])),
                "mean_signed_bias": float(np.mean([r["signed_bias"] for r in recs])),
                "mean_zero_frac": float(np.mean([r["zero_frac"] for r in recs])),
                "final_recovery_ratio": recs[-1]["recovery_ratio"],
            }

        summary_path = os.path.join(output_dir, "update_diagnostics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Diagnostics saved: {csv_path} ({len(self.records)} records)")
        return csv_path, summary_path


def _get_current_analog_lr(optimizer):
    """Get the current learning rate for analog (AnalogContext) param groups."""
    for pg in optimizer.param_groups:
        if any(isinstance(p, AnalogContext) for p in pg["params"]):
            return float(pg["lr"])
    return 0.0

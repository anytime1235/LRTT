#!/usr/bin/env python
# coding=utf-8
"""Carry-path diagnostics for ECO comparison experiments.

Extended diagnostics superseding UpdateDiagnostics when --diag-carry-path is used.
Supports all 6 methods: eco_ref, mixed_precision, ttv1, single_rpu, ideal, cttv2.

Architecture:
  CarryPathDiagnostics
    ├── _TileTracker          — per-tile windowed accumulators
    ├── _TTv1TransferTracker  — per-tile TTv1 column transfer state
    └── save()                — CSV, JSON

Output files:
  carry_path_step.csv     — per-step, per-tile immediate metrics
  carry_path_window.csv   — per-window-boundary, per-tile VRC/VRR/CPG
  carry_path_transfer.csv — TTv1 only: per-step transfer column metrics
  carry_path_summary.json — aggregate statistics
"""

import os
import csv
import json
from collections import defaultdict

import torch
import numpy as np

from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext


# ============================================================================
# Helpers
# ============================================================================

def _cosine_sim(a, b):
    """Cosine similarity between two flat tensors. Returns 0 if either is zero."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = torch.norm(a_flat) * torch.norm(b_flat)
    if denom < 1e-30:
        return 0.0
    return (a_flat @ b_flat / denom).item()


def _ratio(num_tensor, den_tensor):
    """||num|| / ||den||, safe division."""
    n = torch.norm(num_tensor.float()).item()
    d = torch.norm(den_tensor.float()).item()
    if d < 1e-30:
        return 0.0
    return n / d


def _get_layer_subtype(name):
    """Classify attention layer subtype from module name."""
    if "query" in name:
        return "Q"
    elif "key" in name:
        return "K"
    elif "value" in name:
        return "V"
    elif "dense" in name and "attention" in name:
        return "O"
    return "other"


def _get_layer_index(name):
    """Extract encoder layer index from module name."""
    import re
    m = re.search(r'layer\.(\d+)', name)
    return int(m.group(1)) if m else -1


# ============================================================================
# Windowed accumulator
# ============================================================================

class _TileTracker:
    """Per-tile windowed accumulators for VRC/VRR/CPG metrics.

    Maintains tumbling (non-overlapping) windows of target and visible
    delta accumulations. Emits metrics at window boundaries.
    """

    def __init__(self, window_sizes, device="cpu"):
        self.window_sizes = window_sizes
        self.device = device
        # Accumulators: {K: {"target": tensor, "visible": tensor, "count": int}}
        self.accumulators = {}
        for K in window_sizes:
            self.accumulators[K] = {
                "target": None,
                "visible": None,
                "count": 0,
            }

    def accumulate(self, delta_target, delta_visible):
        """Add one step's deltas to all window accumulators.

        Args:
            delta_target: FP32 tensor (ideally on CPU).
            delta_visible: FP32 tensor (ideally on CPU).

        Returns:
            List of (K, vrc_k, vrr_k) for each window that just completed.
        """
        completed = []
        for K in self.window_sizes:
            acc = self.accumulators[K]
            dt = delta_target.float()
            dv = delta_visible.float()
            if acc["target"] is None:
                acc["target"] = dt.clone()
                acc["visible"] = dv.clone()
            else:
                acc["target"] += dt
                acc["visible"] += dv
            acc["count"] += 1

            if acc["count"] >= K:
                vrc_k = _cosine_sim(acc["target"], acc["visible"])
                vrr_k = _ratio(acc["visible"], acc["target"])
                completed.append((K, vrc_k, vrr_k))
                # Reset
                acc["target"] = None
                acc["visible"] = None
                acc["count"] = 0

        return completed


# ============================================================================
# TTv1 transfer tracker
# ============================================================================

class _TTv1TransferTracker:
    """Per-tile TTv1 column transfer state tracker.

    Detects which column was transferred by looking at slow tile changes,
    tracks cumulative target per column, and computes transfer metrics.
    """

    def __init__(self, shape, device="cpu"):
        """
        Args:
            shape: (out_features, in_features) of the weight matrix.
        """
        self.shape = shape
        self.n_cols = shape[1]
        self.device = device

        # Cumulative target gradient per column, reset after transfer
        self.g_cum = torch.zeros(shape, device=device)
        # Previous fast/slow for detecting changes
        self.prev_fast = None
        self.prev_slow = None

    def init_weights(self, fast_w, slow_w):
        """Initialize previous weights at first step."""
        self.prev_fast = fast_w.clone().cpu()
        self.prev_slow = slow_w.clone().cpu()

    def step(self, fast_w, slow_w, delta_target):
        """Process one step, detect transfer, compute metrics.

        Args:
            fast_w: Current fast tile weights [out, in].
            slow_w: Current slow tile weights [out, in].
            delta_target: FP32 target update for this step [out, in].

        Returns:
            dict with transfer metrics, or None if no transfer or not initialized.
        """
        if self.prev_fast is None or self.prev_slow is None:
            self.init_weights(fast_w, slow_w)
            return None

        fast_w = fast_w.cpu().float()
        slow_w = slow_w.cpu().float()
        delta_target = delta_target.cpu().float()

        # Accumulate target per column
        self.g_cum += delta_target

        # Detect transferred column via slow tile change
        slow_delta = slow_w - self.prev_slow
        col_norms = slow_delta.norm(dim=0)  # [in_features]
        max_col_norm = col_norms.max().item()

        result = None
        if max_col_norm > 1e-12:
            # Transfer detected
            col_idx = col_norms.argmax().item()

            # Fast tile change at transferred column
            fast_delta_col = fast_w[:, col_idx] - self.prev_fast[:, col_idx]
            slow_delta_col = slow_delta[:, col_idx]

            # Accumulated target for this column
            g_cum_col = self.g_cum[:, col_idx]

            # FastAccumCos: cos(fast_before_col, g_cum_col)
            # (fast tile BEFORE transfer = prev_fast, which accumulated SGD updates)
            fast_accum_cos = _cosine_sim(self.prev_fast[:, col_idx], g_cum_col)
            fast_accum_ratio = _ratio(self.prev_fast[:, col_idx], g_cum_col)

            # HandoffCos: cos(slow_delta_col, prev_fast_col)
            handoff_cos = _cosine_sim(slow_delta_col, self.prev_fast[:, col_idx])
            handoff_ratio = _ratio(slow_delta_col, self.prev_fast[:, col_idx])

            # EndToEndCos: cos(slow_delta_col, g_cum_col)
            e2e_cos = _cosine_sim(slow_delta_col, g_cum_col)
            e2e_ratio = _ratio(slow_delta_col, g_cum_col)

            result = {
                "col_idx": col_idx,
                "A_pre_norm": self.prev_fast[:, col_idx].norm().item(),
                "A_post_norm": fast_w[:, col_idx].norm().item(),
                "B_delta_norm": slow_delta_col.norm().item(),
                "G_cum_norm": g_cum_col.norm().item(),
                "FastAccumCos": fast_accum_cos,
                "FastAccumRatio": fast_accum_ratio,
                "HandoffCos": handoff_cos,
                "HandoffRatio": handoff_ratio,
                "EndToEndCos": e2e_cos,
                "EndToEndRatio": e2e_ratio,
            }

            # Reset cumulative target for transferred column
            self.g_cum[:, col_idx] = 0.0

        # Update previous
        self.prev_fast = fast_w.clone()
        self.prev_slow = slow_w.clone()

        return result


# ============================================================================
# Main diagnostics class
# ============================================================================

# Default windowed tile selection: 1 per subtype from layers 0 and 11
_DEFAULT_WINDOW_LAYERS = {0, 11}
_DEFAULT_WINDOW_SUBTYPES = {"Q", "K", "V", "O"}


class CarryPathDiagnostics:
    """Extended carry-path diagnostics for ECO comparison experiments.

    Supports all methods: eco_ref (digital), mixed_precision, ttv1,
    single_rpu, ideal, cttv2.

    Args:
        model: The model (analog-converted or digital for eco_ref).
        method: One of "eco_ref", "mixed_precision", "ttv1", "single_rpu",
                "ideal", "cttv2".
        window_sizes: List of tumbling window sizes for VRC/VRR.
        eco_quantizer: EcoQuantizer instance (required for eco_ref method).
        gamma: TTv1 gamma value (for effective weight computation).
    """

    def __init__(self, model, method, window_sizes=None, eco_quantizer=None,
                 gamma=0.0):
        self.method = method
        self.gamma = gamma
        self.eco_quantizer = eco_quantizer
        self.window_sizes = window_sizes or [16, 64, 256]

        # Build tile/layer registry
        if method == "eco_ref":
            self.tile_registry = self._build_eco_registry(model, eco_quantizer)
        else:
            self.tile_registry = self._build_analog_registry(model)

        # Select tiles for windowed metrics
        self._windowed_tile_names = self._select_windowed_tiles()

        # Per-tile windowed accumulators (only for selected tiles)
        self._trackers = {}
        for name in self._windowed_tile_names:
            self._trackers[name] = _TileTracker(self.window_sizes)

        # TTv1 transfer trackers (all TTv1 tiles)
        self._transfer_trackers = {}
        if method == "ttv1":
            for entry in self.tile_registry:
                name = entry["name"]
                shape = entry["shape"]
                self._transfer_trackers[name] = _TTv1TransferTracker(shape)

        # Storage
        self.step_records = []
        self.window_records = []
        self.transfer_records = []

        # Caches for snapshot timing
        self._w_before = {}
        self._w_adam = {}  # eco_ref only: weights after Adam but before quant

        print(f"  CarryPathDiagnostics: {len(self.tile_registry)} tiles, "
              f"method={method}, windows={self.window_sizes}")
        print(f"  Windowed tiles: {len(self._windowed_tile_names)}")
        if self._transfer_trackers:
            print(f"  TTv1 transfer trackers: {len(self._transfer_trackers)}")

    @staticmethod
    def _build_analog_registry(model):
        """Build registry for analog methods (single_rpu, mixed_precision, ttv1, etc.)."""
        registry = []
        for name, module in model.named_modules():
            if not isinstance(module, AnalogLinear):
                continue
            subtype = _get_layer_subtype(name)
            layer_idx = _get_layer_index(name)
            for i, tile in enumerate(module.analog_tiles()):
                tile_key = f"{name}::tile{i}" if i > 0 else name
                shape = tile.get_weights()[0].shape
                # Get AnalogContext for this module
                analog_ctx = None
                for pname, param in module.named_parameters():
                    if isinstance(param, AnalogContext):
                        analog_ctx = param
                        break
                registry.append({
                    "name": tile_key,
                    "module_name": name,
                    "module": module,
                    "tile": tile,
                    "analog_ctx": analog_ctx,
                    "subtype": subtype,
                    "layer_idx": layer_idx,
                    "shape": shape,
                })
        return registry

    @staticmethod
    def _build_eco_registry(model, eco_quantizer):
        """Build registry for eco_ref method (digital nn.Linear layers)."""
        registry = []
        for name in eco_quantizer.get_all_target_names():
            module = eco_quantizer.targets[name]
            subtype = _get_layer_subtype(name)
            layer_idx = _get_layer_index(name)
            shape = module.weight.data.shape
            registry.append({
                "name": name,
                "module_name": name,
                "module": module,
                "tile": None,
                "analog_ctx": None,
                "subtype": subtype,
                "layer_idx": layer_idx,
                "shape": shape,
            })
        return registry

    def _select_windowed_tiles(self):
        """Select default subset of tiles for windowed metrics."""
        selected = []
        for entry in self.tile_registry:
            layer_idx = entry["layer_idx"]
            subtype = entry["subtype"]
            if (layer_idx in _DEFAULT_WINDOW_LAYERS and
                    subtype in _DEFAULT_WINDOW_SUBTYPES):
                selected.append(entry["name"])
        # Fallback: if no matching tiles, take first 8
        if not selected:
            selected = [e["name"] for e in self.tile_registry[:8]]
        return set(selected)

    # ------------------------------------------------------------------
    # Snapshot: BEFORE optimizer.step()
    # ------------------------------------------------------------------

    def snapshot_before_step(self, model, optimizer):
        """Capture weights before optimizer.step().

        For analog methods: reads tile weights and AnalogContext x/d.
        For eco_ref: reads nn.Linear weights.
        """
        self._w_before = {}
        self._analog_ctx_cache = {}

        if self.method == "eco_ref":
            for entry in self.tile_registry:
                name = entry["name"]
                module = entry["module"]
                self._w_before[name] = module.weight.data.clone().cpu()
        elif self.method == "ttv1":
            for entry in self.tile_registry:
                name = entry["name"]
                tile = entry["tile"]
                ctx = entry["analog_ctx"]
                try:
                    # For TTv1, get both fast and slow
                    hidden = tile.get_hidden_parameters()
                    fast_w = None
                    slow_w = None
                    for hname, htensor in hidden.items():
                        if "hidden_weights_0" in hname:
                            fast_w = htensor.clone().cpu()
                        elif "hidden_weights_1" in hname:
                            slow_w = htensor.clone().cpu()

                    if fast_w is not None and slow_w is not None:
                        if self.gamma > 0:
                            w_eff = slow_w + self.gamma * fast_w
                        else:
                            w_eff = slow_w.clone()
                        self._w_before[name] = w_eff
                        self._w_before[f"{name}::fast"] = fast_w
                        self._w_before[f"{name}::slow"] = slow_w
                    else:
                        # Fallback to combined weights
                        w = tile.get_weights()[0].clone().cpu()
                        self._w_before[name] = w
                except Exception:
                    try:
                        w = tile.get_weights()[0].clone().cpu()
                        self._w_before[name] = w
                    except Exception:
                        pass

                # Cache AnalogContext x, d for target computation
                if ctx is not None and ctx.analog_input and ctx.analog_grad_output:
                    try:
                        in_trans = tile.in_trans if hasattr(tile, 'in_trans') else False
                        out_trans = tile.out_trans if hasattr(tile, 'out_trans') else False
                        x = torch.cat(ctx.analog_input,
                                      dim=-1 if in_trans else 0).detach().cpu().float()
                        d = torch.cat(ctx.analog_grad_output,
                                      dim=-1 if out_trans else 0).detach().cpu().float()
                        self._analog_ctx_cache[name] = {"x": x, "d": d}
                    except Exception:
                        pass
        else:
            # single_rpu, mixed_precision, ideal, cttv2
            for entry in self.tile_registry:
                name = entry["name"]
                tile = entry["tile"]
                ctx = entry["analog_ctx"]
                try:
                    w = tile.get_weights()[0].clone().cpu()
                    self._w_before[name] = w
                except Exception:
                    continue

                if ctx is not None and ctx.analog_input and ctx.analog_grad_output:
                    try:
                        in_trans = tile.in_trans if hasattr(tile, 'in_trans') else False
                        out_trans = tile.out_trans if hasattr(tile, 'out_trans') else False
                        x = torch.cat(ctx.analog_input,
                                      dim=-1 if in_trans else 0).detach().cpu().float()
                        d = torch.cat(ctx.analog_grad_output,
                                      dim=-1 if out_trans else 0).detach().cpu().float()
                        self._analog_ctx_cache[name] = {"x": x, "d": d}
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # ECO-specific: snapshot between Adam and quantization
    # ------------------------------------------------------------------

    def snapshot_after_adam(self):
        """For eco_ref only: capture weights after Adam step but before ECO post_step.

        This gives us DeltaW_target = w_adam - w_before.
        """
        self._w_adam = {}
        if self.method != "eco_ref":
            return
        for entry in self.tile_registry:
            name = entry["name"]
            module = entry["module"]
            self._w_adam[name] = module.weight.data.clone().cpu()

    # ------------------------------------------------------------------
    # Snapshot: AFTER optimizer.step() (and eco post_step if applicable)
    # ------------------------------------------------------------------

    def snapshot_after_step(self, model, step, optimizer=None):
        """Capture weights after step and compute all metrics.

        For eco_ref: call AFTER eco_quantizer.post_step().
        For analog: call AFTER optimizer.step().
        """
        lr = self._get_lr(optimizer)

        for entry in self.tile_registry:
            name = entry["name"]
            subtype = entry["subtype"]

            if name not in self._w_before:
                continue

            w_before = self._w_before[name]

            # Get current weights
            if self.method == "eco_ref":
                w_after = entry["module"].weight.data.clone().cpu()
                # DeltaW_target = w_adam - w_before (Adam's intended update)
                if name in self._w_adam:
                    delta_target = (self._w_adam[name] - w_before).float()
                else:
                    delta_target = None
                delta_visible = (w_after - w_before).float()

            elif self.method == "ttv1":
                tile = entry["tile"]
                try:
                    hidden = tile.get_hidden_parameters()
                    fast_w = None
                    slow_w = None
                    for hname, htensor in hidden.items():
                        if "hidden_weights_0" in hname:
                            fast_w = htensor.clone().cpu()
                        elif "hidden_weights_1" in hname:
                            slow_w = htensor.clone().cpu()

                    if fast_w is not None and slow_w is not None:
                        if self.gamma > 0:
                            w_eff_after = slow_w + self.gamma * fast_w
                        else:
                            w_eff_after = slow_w.clone()

                        delta_visible = (w_eff_after - w_before).float()

                        # Transfer tracker
                        if name in self._transfer_trackers:
                            # Compute delta_target from AnalogContext
                            dt = self._compute_analog_target(name, lr)
                            if dt is not None:
                                t_result = self._transfer_trackers[name].step(
                                    fast_w, slow_w, dt)
                                if t_result is not None:
                                    self.transfer_records.append({
                                        "step": step,
                                        "tile_name": name,
                                        **t_result,
                                    })
                    else:
                        w_after = tile.get_weights()[0].clone().cpu()
                        delta_visible = (w_after - w_before).float()
                except Exception:
                    try:
                        w_after = tile.get_weights()[0].clone().cpu()
                        delta_visible = (w_after - w_before).float()
                    except Exception:
                        continue

                delta_target = self._compute_analog_target(name, lr)

            else:
                # single_rpu, mixed_precision, ideal, cttv2
                tile = entry["tile"]
                try:
                    w_after = tile.get_weights()[0].clone().cpu()
                except Exception:
                    continue
                delta_visible = (w_after - w_before).float()
                delta_target = self._compute_analog_target(name, lr)

            # Compute immediate metrics
            delta_visible_norm = torch.norm(delta_visible).item()

            if delta_target is not None:
                # Match shapes
                min_d = min(delta_target.shape[0], delta_visible.shape[0])
                min_x = min(delta_target.shape[1], delta_visible.shape[1])
                dt = delta_target[:min_d, :min_x]
                dv = delta_visible[:min_d, :min_x]

                delta_target_norm = torch.norm(dt).item()
                residual = dv - dt
                residual_norm = torch.norm(residual).item()
                cosine = _cosine_sim(dt, dv)
                res_ratio = residual_norm / max(delta_target_norm, 1e-30)
            else:
                dt = None
                delta_target_norm = 0.0
                residual_norm = 0.0
                cosine = 0.0
                res_ratio = 0.0

            self.step_records.append({
                "step": step,
                "tile_name": name,
                "subtype": subtype,
                "delta_target_norm": delta_target_norm,
                "delta_visible_norm": delta_visible_norm,
                "residual_norm": residual_norm,
                "cosine_sim": cosine,
                "residual_ratio": res_ratio,
            })

            # Windowed accumulation (selected tiles only)
            if name in self._windowed_tile_names and dt is not None:
                completed = self._trackers[name].accumulate(dt, dv)
                # Also compute VRC_1 = immediate cosine for CPG baseline
                for K, vrc_k, vrr_k in completed:
                    # CPG = VRC_K - VRC_1; we approximate VRC_1 as the
                    # mean cosine over the window (would need tracking).
                    # Simpler: just store VRC_K and VRR_K; compute CPG in post.
                    self.window_records.append({
                        "step": step,
                        "tile_name": name,
                        "subtype": subtype,
                        "window_K": K,
                        "VRC_K": vrc_k,
                        "VRR_K": vrr_k,
                    })

        # Clear caches
        self._w_before = {}
        self._w_adam = {}
        self._analog_ctx_cache = {}

    def _compute_analog_target(self, name, lr):
        """Compute FP32 target update from cached AnalogContext x, d."""
        if name not in self._analog_ctx_cache:
            return None
        cache = self._analog_ctx_cache[name]
        x = cache["x"]
        d = cache["d"]

        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        if d.ndim > 2:
            d = d.reshape(-1, d.shape[-1])

        n_x = x.shape[0]
        n_d = d.shape[0]
        if n_x != n_d:
            delta_target = -lr * torch.outer(d.mean(dim=0), x.mean(dim=0))
        else:
            delta_target = -lr * (d.mT @ x) / n_x

        return delta_target

    def _get_lr(self, optimizer):
        """Get current analog/target LR."""
        if optimizer is None:
            return 0.016  # default
        for pg in optimizer.param_groups:
            if any(isinstance(p, AnalogContext) for p in pg["params"]):
                return float(pg["lr"])
        # For eco_ref: first param group
        if optimizer.param_groups:
            return float(optimizer.param_groups[0]["lr"])
        return 0.016

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output_dir):
        """Save all diagnostics to CSV and JSON."""
        os.makedirs(output_dir, exist_ok=True)

        # Step CSV
        if self.step_records:
            csv_path = os.path.join(output_dir, "carry_path_step.csv")
            fieldnames = list(self.step_records[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.step_records)
            print(f"Carry-path step CSV: {csv_path} ({len(self.step_records)} records)")

        # Window CSV
        if self.window_records:
            csv_path = os.path.join(output_dir, "carry_path_window.csv")
            fieldnames = list(self.window_records[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.window_records)
            print(f"Carry-path window CSV: {csv_path} ({len(self.window_records)} records)")

        # Transfer CSV (TTv1 only)
        if self.transfer_records:
            csv_path = os.path.join(output_dir, "carry_path_transfer.csv")
            fieldnames = list(self.transfer_records[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.transfer_records)
            print(f"Carry-path transfer CSV: {csv_path} ({len(self.transfer_records)} records)")

        # Summary JSON
        summary = self._compute_summary()
        summary_path = os.path.join(output_dir, "carry_path_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Carry-path summary: {summary_path}")

    def _compute_summary(self):
        """Compute aggregate summary statistics."""
        summary = {"method": self.method, "n_tiles": len(self.tile_registry)}

        # Per-tile step metrics
        tile_step = defaultdict(list)
        for r in self.step_records:
            tile_step[r["tile_name"]].append(r)

        per_tile = {}
        for tile_name, recs in tile_step.items():
            per_tile[tile_name] = {
                "subtype": recs[0]["subtype"],
                "n_steps": len(recs),
                "mean_cosine_sim": float(np.mean([r["cosine_sim"] for r in recs])),
                "mean_residual_ratio": float(np.mean([r["residual_ratio"] for r in recs])),
                "mean_delta_target_norm": float(np.mean([r["delta_target_norm"] for r in recs])),
                "mean_delta_visible_norm": float(np.mean([r["delta_visible_norm"] for r in recs])),
            }
        summary["per_tile"] = per_tile

        # Aggregate
        all_cosines = [r["cosine_sim"] for r in self.step_records]
        all_res_ratios = [r["residual_ratio"] for r in self.step_records]
        if all_cosines:
            summary["aggregate"] = {
                "mean_cosine_sim": float(np.mean(all_cosines)),
                "std_cosine_sim": float(np.std(all_cosines)),
                "mean_residual_ratio": float(np.mean(all_res_ratios)),
                "std_residual_ratio": float(np.std(all_res_ratios)),
            }

        # Window metrics
        if self.window_records:
            window_by_k = defaultdict(list)
            for r in self.window_records:
                window_by_k[r["window_K"]].append(r)
            window_summary = {}
            for K, recs in window_by_k.items():
                window_summary[str(K)] = {
                    "mean_VRC_K": float(np.mean([r["VRC_K"] for r in recs])),
                    "std_VRC_K": float(np.std([r["VRC_K"] for r in recs])),
                    "mean_VRR_K": float(np.mean([r["VRR_K"] for r in recs])),
                }
            summary["windows"] = window_summary

        # Transfer summary (TTv1)
        if self.transfer_records:
            e2e_cos = [r["EndToEndCos"] for r in self.transfer_records]
            handoff_cos = [r["HandoffCos"] for r in self.transfer_records]
            summary["ttv1_transfer"] = {
                "n_transfers": len(self.transfer_records),
                "mean_EndToEndCos": float(np.mean(e2e_cos)),
                "mean_HandoffCos": float(np.mean(handoff_cos)),
                "std_EndToEndCos": float(np.std(e2e_cos)),
                "std_HandoffCos": float(np.std(handoff_cos)),
            }

        return summary

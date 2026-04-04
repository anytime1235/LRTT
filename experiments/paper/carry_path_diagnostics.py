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

    For TTv1, optionally tracks a separate slow-tile-only accumulator
    (VRC_slow) alongside the default W_eff accumulator.
    """

    def __init__(self, window_sizes, device="cpu", track_slow=False):
        self.window_sizes = window_sizes
        self.device = device
        self.track_slow = track_slow
        # Accumulators: {K: {"target": tensor, "visible": tensor, "count": int}}
        self.accumulators = {}
        for K in window_sizes:
            self.accumulators[K] = {
                "target": None,
                "visible": None,
                "count": 0,
            }
        # Slow-only accumulators (TTv1): same structure, accumulates delta_slow
        if track_slow:
            self.slow_accumulators = {}
            for K in window_sizes:
                self.slow_accumulators[K] = {
                    "target": None,
                    "visible": None,
                    "count": 0,
                }

    def accumulate(self, delta_target, delta_visible, delta_slow=None):
        """Add one step's deltas to all window accumulators.

        Args:
            delta_target: FP32 tensor (ideally on CPU).
            delta_visible: FP32 tensor (ideally on CPU). W_eff delta for TTv1.
            delta_slow: FP32 tensor (optional). Slow tile delta for TTv1.
                        If None and track_slow is True, zeros are accumulated.

        Returns:
            List of (K, vrc_k, vrr_k) for each window that just completed.
            If track_slow, each entry is (K, vrc_k, vrr_k, vrc_slow_k, vrr_slow_k).
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

            eff_done = acc["count"] >= K

            # Slow accumulator (parallel, same window boundaries)
            slow_vrc_k, slow_vrr_k = None, None
            if self.track_slow:
                sacc = self.slow_accumulators[K]
                ds = delta_slow.float() if delta_slow is not None else torch.zeros_like(dt)
                if sacc["target"] is None:
                    sacc["target"] = dt.clone()
                    sacc["visible"] = ds.clone()
                else:
                    sacc["target"] += dt
                    sacc["visible"] += ds
                sacc["count"] += 1

                if sacc["count"] >= K:
                    slow_vrc_k = _cosine_sim(sacc["target"], sacc["visible"])
                    slow_vrr_k = _ratio(sacc["visible"], sacc["target"])
                    sacc["target"] = None
                    sacc["visible"] = None
                    sacc["count"] = 0

            if eff_done:
                vrc_k = _cosine_sim(acc["target"], acc["visible"])
                vrr_k = _ratio(acc["visible"], acc["target"])
                if self.track_slow:
                    completed.append((K, vrc_k, vrr_k, slow_vrc_k, slow_vrr_k))
                else:
                    completed.append((K, vrc_k, vrr_k))
                # Reset eff accumulator
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

            # NTE: Normalized Transfer Error
            g_cum_norm_val = g_cum_col.norm().item()
            if g_cum_norm_val > 1e-30:
                nte = (slow_delta_col - g_cum_col).norm().item() / g_cum_norm_val
            else:
                nte = 0.0

            # PE: Projection Efficiency
            g_cum_norm_sq = g_cum_col.dot(g_cum_col).item()
            if g_cum_norm_sq > 1e-30:
                pe = slow_delta_col.dot(g_cum_col).item() / g_cum_norm_sq
            else:
                pe = 0.0

            # Signed Transfer Bias
            g_cum_l1 = g_cum_col.abs().sum().item()
            if g_cum_l1 > 1e-30:
                residual_col = slow_delta_col - g_cum_col
                signed_bias = (residual_col * g_cum_col.sign()).sum().item() / g_cum_l1
            else:
                signed_bias = 0.0

            # Reset detection: fast tile column near-zero after transfer
            a_pre = self.prev_fast[:, col_idx].norm().item()
            a_post = fast_w[:, col_idx].norm().item()
            reset_detected = (a_pre > 1e-8 and a_post < 0.01 * a_pre)

            result = {
                "col_idx": col_idx,
                "A_pre_norm": a_pre,
                "A_post_norm": a_post,
                "B_delta_norm": slow_delta_col.norm().item(),
                "G_cum_norm": g_cum_norm_val,
                "FastAccumCos": fast_accum_cos,
                "FastAccumRatio": fast_accum_ratio,
                "HandoffCos": handoff_cos,
                "HandoffRatio": handoff_ratio,
                "EndToEndCos": e2e_cos,
                "EndToEndRatio": e2e_ratio,
                "NTE": nte,
                "PE": pe,
                "SignedBias": signed_bias,
                "reset_detected": reset_detected,
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
                 gamma=0.0, layer_set=None):
        self.method = method
        self.gamma = gamma
        self.eco_quantizer = eco_quantizer
        self.window_sizes = window_sizes or [16, 64, 256]

        # Build tile/layer registry
        if method == "eco_ref":
            self.tile_registry = self._build_eco_registry(model, eco_quantizer, layer_set)
        else:
            self.tile_registry = self._build_analog_registry(model, layer_set)

        # Select tiles for windowed metrics
        self._windowed_tile_names = self._select_windowed_tiles()

        # Per-tile windowed accumulators (only for selected tiles)
        # For TTv1: also track slow-tile-only VRC (VRC_slow)
        self._trackers = {}
        for name in self._windowed_tile_names:
            self._trackers[name] = _TileTracker(
                self.window_sizes, track_slow=(method == "ttv1")
            )

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

        # Persistent w_before for windowed tiles (survives across steps)
        self._w_before_window = {}
        # Slow-only w_before for TTv1 VRC_slow (survives across steps)
        self._w_before_window_slow = {}

        # Fast lookup: name -> registry entry
        self._tile_by_name = {e["name"]: e for e in self.tile_registry}

        print(f"  CarryPathDiagnostics: {len(self.tile_registry)} tiles, "
              f"method={method}, windows={self.window_sizes}")
        print(f"  Windowed tiles: {len(self._windowed_tile_names)}")
        if self._transfer_trackers:
            print(f"  TTv1 transfer trackers: {len(self._transfer_trackers)}")

    @staticmethod
    def _build_analog_registry(model, layer_set=None):
        """Build registry for analog methods (single_rpu, mixed_precision, ttv1, etc.)."""
        registry = []
        for name, module in model.named_modules():
            if not isinstance(module, AnalogLinear):
                continue
            subtype = _get_layer_subtype(name)
            layer_idx = _get_layer_index(name)
            if layer_set is not None and layer_idx not in layer_set:
                continue
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
    def _build_eco_registry(model, eco_quantizer, layer_set=None):
        """Build registry for eco_ref method (digital nn.Linear layers)."""
        registry = []
        for name in eco_quantizer.get_all_target_names():
            module = eco_quantizer.targets[name]
            subtype = _get_layer_subtype(name)
            layer_idx = _get_layer_index(name)
            if layer_set is not None and layer_idx not in layer_set:
                continue
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
        """Select default subset of tiles for windowed metrics.

        Only selects main tiles (no ::tileN subtiles) because subtiles
        lack delta_target mapping and produce zero VRC metrics.
        """
        selected = []
        for entry in self.tile_registry:
            layer_idx = entry["layer_idx"]
            subtype = entry["subtype"]
            name = entry["name"]
            # Skip subtiles — they have no delta_target (key mismatch)
            if "::tile" in name:
                continue
            if (layer_idx in _DEFAULT_WINDOW_LAYERS and
                    subtype in _DEFAULT_WINDOW_SUBTYPES):
                selected.append(name)
        # Fallback: if no matching tiles, take first 8
        if not selected:
            selected = [e["name"] for e in self.tile_registry[:8]]
        return set(selected)

    # ------------------------------------------------------------------
    # Snapshot: weights before first microbatch (for grad_accum > 1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def snapshot_weights_before(self):
        """Capture w_before at the start of a grad_accum group.

        Must be called BEFORE the first microbatch's tile.update().
        For grad_accum > 1, tile updates happen per microbatch, so
        w_before must be captured before any of them fire.
        """
        self._w_before_early = {}
        if self.method == "eco_ref":
            for entry in self.tile_registry:
                name = entry["name"]
                module = entry["module"]
                self._w_before_early[name] = module.weight.data.clone().cpu()
        elif self.method == "ttv1":
            for entry in self.tile_registry:
                name = entry["name"]
                tile = entry["tile"]
                try:
                    hidden = tile.get_hidden_parameters()
                    fast_w, slow_w = None, None
                    for hname, htensor in hidden.items():
                        if "hidden_weights_0" in hname:
                            fast_w = htensor.clone().cpu()
                        elif "hidden_weights_1" in hname:
                            slow_w = htensor.clone().cpu()
                    if fast_w is not None and slow_w is not None:
                        if self.gamma > 0:
                            self._w_before_early[name] = slow_w + self.gamma * fast_w
                        else:
                            self._w_before_early[name] = slow_w.clone()
                        self._w_before_early[f"{name}::fast"] = fast_w
                        self._w_before_early[f"{name}::slow"] = slow_w
                    else:
                        self._w_before_early[name] = tile.get_weights()[0].clone().cpu()
                except Exception:
                    try:
                        self._w_before_early[name] = tile.get_weights()[0].clone().cpu()
                    except Exception:
                        pass
        else:
            for entry in self.tile_registry:
                name = entry["name"]
                tile = entry["tile"]
                try:
                    self._w_before_early[name] = tile.get_weights()[0].clone().cpu()
                except Exception:
                    continue

    # ------------------------------------------------------------------
    # Snapshot: BEFORE optimizer.step()
    # ------------------------------------------------------------------

    def snapshot_before_step(self, model, optimizer):
        """Capture weights before optimizer.step().

        For grad_accum > 1: uses pre-captured _w_before_early.
        For grad_accum == 1: captures weights directly.
        """
        use_early = bool(getattr(self, '_w_before_early', {}))
        self._w_before = {}
        self._analog_ctx_cache = {}

        if use_early:
            self._w_before = self._w_before_early
            self._w_before_early = {}
            return

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
    # Lightweight per-step update for TTv1 transfer tracker
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_transfer_tracker(self, model, step, delta_target_dict=None):
        """Lightweight per-step update for TTv1 transfer tracker.

        Called EVERY step (not just diag steps) to:
        1. Accumulate g_cum with target gradient
        2. Detect transfers by reading fast/slow tile changes
        3. Record transfer events

        Args:
            model: The model.
            step: Current global step.
            delta_target_dict: {tile_name: delta_target tensor} from update_diagnostics.
                               If None, g_cum accumulation is skipped.
        """
        if self.method != "ttv1" or not self._transfer_trackers:
            return

        for entry in self.tile_registry:
            name = entry["name"]
            tile = entry["tile"]

            if name not in self._transfer_trackers:
                continue

            try:
                hidden = tile.get_hidden_parameters()
                fast_w, slow_w = None, None
                for hname, htensor in hidden.items():
                    if "hidden_weights_0" in hname:
                        fast_w = htensor.clone().cpu()
                    elif "hidden_weights_1" in hname:
                        slow_w = htensor.clone().cpu()

                if fast_w is None or slow_w is None:
                    continue

                # Get delta_target for this tile
                dt = torch.zeros_like(fast_w)
                if delta_target_dict is not None and name in delta_target_dict:
                    dt = delta_target_dict[name].cpu().float()
                    # Match shape
                    min_d = min(dt.shape[0], fast_w.shape[0])
                    min_x = min(dt.shape[1], fast_w.shape[1])
                    dt_matched = torch.zeros_like(fast_w)
                    dt_matched[:min_d, :min_x] = dt[:min_d, :min_x]
                    dt = dt_matched

                t_result = self._transfer_trackers[name].step(fast_w, slow_w, dt)
                if t_result is not None:
                    self.transfer_records.append({
                        "step": step,
                        "tile_name": name,
                        **t_result,
                    })
            except Exception:
                continue

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
                        # Separate slow/fast/eff deltas
                        slow_before = self._w_before.get(f"{name}::slow")
                        fast_before = self._w_before.get(f"{name}::fast")

                        delta_slow = (slow_w - slow_before).float() if slow_before is not None else None
                        delta_fast = (fast_w - fast_before).float() if fast_before is not None else None

                        if self.gamma > 0:
                            w_eff_after = slow_w + self.gamma * fast_w
                        else:
                            w_eff_after = slow_w.clone()
                        delta_visible = (w_eff_after - w_before).float()

                        # Store separated deltas for metrics (used below)
                        self._ttv1_deltas = {
                            "delta_slow": delta_slow,
                            "delta_fast": delta_fast,
                            "delta_eff": delta_visible,
                        }

                        # Transfer tracker
                        if name in self._transfer_trackers:
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
                        self._ttv1_deltas = None
                except Exception:
                    try:
                        w_after = tile.get_weights()[0].clone().cpu()
                        delta_visible = (w_after - w_before).float()
                    except Exception:
                        continue
                    self._ttv1_deltas = None

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

            record = {
                "step": step,
                "tile_name": name,
                "subtype": subtype,
                "delta_target_norm": delta_target_norm,
                "delta_visible_norm": delta_visible_norm,
                "residual_norm": residual_norm,
                "cosine_sim": cosine,
                "residual_ratio": res_ratio,
            }

            # TTv1: add separate slow/fast/eff metrics
            ttv1_deltas = getattr(self, '_ttv1_deltas', None)
            if self.method == "ttv1" and ttv1_deltas is not None and dt is not None:
                ds = ttv1_deltas["delta_slow"]
                df = ttv1_deltas["delta_fast"]
                if ds is not None:
                    ds_matched = ds[:min_d, :min_x]
                    record["cosine_sim_slow"] = _cosine_sim(dt, ds_matched)
                    record["delta_visible_norm_slow"] = torch.norm(ds).item()
                    res_slow = ds_matched - dt
                    record["residual_ratio_slow"] = torch.norm(res_slow).item() / max(delta_target_norm, 1e-30)
                if df is not None:
                    df_matched = df[:min_d, :min_x]
                    record["cosine_sim_fast"] = _cosine_sim(dt, df_matched)
                    record["delta_visible_norm_fast"] = torch.norm(df).item()
                if self.gamma > 0:
                    record["cosine_sim_eff"] = cosine  # delta_visible is already W_eff
                    record["residual_ratio_eff"] = res_ratio

            self.step_records.append(record)
            self._ttv1_deltas = None  # clear

            # Windowed accumulation (selected tiles only)
            # For TTv1: use W_eff delta (gamma*fast + slow) for carry-path VRC
            # and also accumulate slow-only delta for VRC_slow
            if name in self._windowed_tile_names and dt is not None:
                if self.method == "ttv1" and ttv1_deltas is not None and ttv1_deltas["delta_eff"] is not None:
                    dv_window = ttv1_deltas["delta_eff"][:min_d, :min_x]
                else:
                    dv_window = dv
                # Slow delta for VRC_slow (TTv1 only)
                ds_window = None
                if self.method == "ttv1" and ttv1_deltas is not None and ttv1_deltas.get("delta_slow") is not None:
                    ds_window = ttv1_deltas["delta_slow"][:min_d, :min_x]

                completed = self._trackers[name].accumulate(dt, dv_window, delta_slow=ds_window)
                for item in completed:
                    if len(item) == 5:
                        K, vrc_k, vrr_k, vrc_slow_k, vrr_slow_k = item
                    else:
                        K, vrc_k, vrr_k = item
                        vrc_slow_k, vrr_slow_k = None, None
                    rec = {
                        "step": step,
                        "tile_name": name,
                        "subtype": subtype,
                        "window_K": K,
                        "VRC_K": vrc_k,
                        "VRR_K": vrr_k,
                    }
                    if vrc_slow_k is not None:
                        rec["VRC_slow_K"] = vrc_slow_k
                        rec["VRR_slow_K"] = vrr_slow_k
                    self.window_records.append(rec)

            # Sync _w_before_window so accumulate_windows_only stays consistent
            # For TTv1: store W_eff = slow + gamma*fast (matching window delta)
            if name in self._windowed_tile_names:
                if self.method == "ttv1":
                    try:
                        hidden = entry["tile"].get_hidden_parameters()
                        _fast_w_sync = None
                        _slow_w_sync = None
                        for hname, htensor in hidden.items():
                            if "hidden_weights_0" in hname:
                                _fast_w_sync = htensor.clone().cpu()
                            elif "hidden_weights_1" in hname:
                                _slow_w_sync = htensor.clone().cpu()
                        if _slow_w_sync is not None:
                            if self.gamma > 0 and _fast_w_sync is not None:
                                self._w_before_window[name] = _slow_w_sync + self.gamma * _fast_w_sync
                            else:
                                self._w_before_window[name] = _slow_w_sync
                            # Sync slow-only w_before for VRC_slow
                            self._w_before_window_slow[name] = _slow_w_sync.clone()
                        else:
                            self._w_before_window[name] = entry["tile"].get_weights()[0].clone().cpu()
                    except Exception:
                        pass
                elif self.method == "eco_ref":
                    self._w_before_window[name] = entry["module"].weight.data.clone().cpu()
                else:
                    try:
                        self._w_before_window[name] = entry["tile"].get_weights()[0].clone().cpu()
                    except Exception:
                        pass

        # Clear caches
        self._w_before = {}
        self._w_adam = {}
        self._analog_ctx_cache = {}

    @torch.no_grad()
    def accumulate_microbatch_for_windows(self, model, use_cpu=False):
        """Lightweight d^T @ x accumulation for windowed modules ONLY.

        Must be called after loss.backward() and BEFORE p.reset().
        Only processes modules that contain windowed tiles, skipping
        mu stats computation.  ~8 modules vs 12+ for full diagnostics.

        Args:
            use_cpu: If True, move x,d to CPU before matmul (avoids GPU OOM
                     on memory-constrained devices, slower but safe).
                     If False, compute on GPU then move result to CPU (faster
                     but requires ~40MB GPU headroom per module).
        """
        if not hasattr(self, '_window_grad_accum'):
            self._window_grad_accum = {}

        # Build module lookup on first call
        if not hasattr(self, '_windowed_modules'):
            self._windowed_modules = {}
            for name in self._windowed_tile_names:
                entry = self._tile_by_name.get(name)
                if entry is None:
                    continue
                mod_name = entry["module_name"]
                if mod_name not in self._windowed_modules:
                    self._windowed_modules[mod_name] = entry

        for mod_name, entry in self._windowed_modules.items():
            module = entry["module"]
            ctx = entry.get("analog_ctx")
            if ctx is None:
                for pname, param in module.named_parameters():
                    if isinstance(param, AnalogContext):
                        ctx = param
                        break
                entry["analog_ctx"] = ctx
            if ctx is None or not (ctx.analog_input and ctx.analog_grad_output):
                continue

            tile = entry["tile"]
            try:
                in_trans = tile.in_trans if hasattr(tile, 'in_trans') else False
                out_trans = tile.out_trans if hasattr(tile, 'out_trans') else False

                if use_cpu:
                    # CPU path: move chunks to CPU before cat (zero GPU overhead)
                    x_parts = [t.detach().float().cpu() for t in ctx.analog_input]
                    d_parts = [t.detach().float().cpu() for t in ctx.analog_grad_output]
                    x = torch.cat(x_parts, dim=-1 if in_trans else 0)
                    d = torch.cat(d_parts, dim=-1 if out_trans else 0)
                    del x_parts, d_parts
                else:
                    # GPU path: cat + matmul on GPU, then move result to CPU
                    x = torch.cat(ctx.analog_input, dim=-1 if in_trans else 0).detach().float()
                    d = torch.cat(ctx.analog_grad_output, dim=-1 if out_trans else 0).detach().float()

                if x.ndim > 2:
                    x = x.reshape(-1, x.shape[-1])
                if d.ndim > 2:
                    d = d.reshape(-1, d.shape[-1])

                if use_cpu:
                    G_m = d.mT @ x  # CPU matmul
                else:
                    G_m = (d.mT @ x).cpu()  # GPU matmul → CPU
                    del x, d  # Free GPU memory immediately

                # Accumulate across microbatches, keyed by tile name (not module)
                tile_name = entry["name"]
                if tile_name in self._window_grad_accum:
                    self._window_grad_accum[tile_name] += G_m
                else:
                    self._window_grad_accum[tile_name] = G_m.clone()
            except Exception:
                continue

    def flush_window_grad_accum(self, lr):
        """Convert accumulated G to delta_target = -lr * G and return, then clear."""
        if not hasattr(self, '_window_grad_accum'):
            return {}
        dt_dict = {k: (-lr * G).float() for k, G in self._window_grad_accum.items()}
        self._window_grad_accum = {}
        return dt_dict

    @torch.no_grad()
    def accumulate_windows_only(self, model, step, delta_target_dict=None):
        """Lightweight per-step window accumulation (windowed tiles only).

        Must be called at EVERY step (not just diagnostic steps) so that
        tumbling windows of size K > 1 can complete.  Only reads weights
        for windowed tiles (~8), keeping overhead low.

        Args:
            model: The model.
            step: Current global step.
            delta_target_dict: {tile_name: delta_target tensor} already
                               scaled by -lr, on CPU.  When None the
                               window accumulator is still called with
                               whatever _accumulated_targets were set.
        """
        for name in self._windowed_tile_names:
            entry = self._tile_by_name.get(name)
            if entry is None:
                continue

            # --- read current (post-step) weights -------------------------
            # For TTv1: use W_eff = slow + gamma*fast for carry-path window metrics
            # and also read slow_w separately for VRC_slow
            _slow_w_current = None  # TTv1 only: for VRC_slow
            if self.method == "ttv1":
                tile = entry["tile"]
                try:
                    hidden = tile.get_hidden_parameters()
                    _fast_w = None
                    _slow_w = None
                    for hname, htensor in hidden.items():
                        if "hidden_weights_0" in hname:
                            _fast_w = htensor.clone().cpu()
                        elif "hidden_weights_1" in hname:
                            _slow_w = htensor.clone().cpu()
                    if _slow_w is not None:
                        _slow_w_current = _slow_w  # save for VRC_slow
                        if self.gamma > 0 and _fast_w is not None:
                            w_after = _slow_w + self.gamma * _fast_w
                        else:
                            w_after = _slow_w
                    else:
                        w_after = tile.get_weights()[0].clone().cpu()
                except Exception:
                    continue
            elif self.method == "eco_ref":
                w_after = entry["module"].weight.data.clone().cpu()
            else:
                tile = entry["tile"]
                try:
                    w_after = tile.get_weights()[0].clone().cpu()
                except Exception:
                    continue

            # --- first call: initialise w_before_window --------------------
            if name not in self._w_before_window:
                self._w_before_window[name] = w_after.clone()
                if _slow_w_current is not None:
                    self._w_before_window_slow[name] = _slow_w_current.clone()
                continue  # no delta on the very first call

            w_before = self._w_before_window[name]
            delta_visible = (w_after - w_before).float()

            # Slow tile delta for VRC_slow
            delta_slow = None
            if _slow_w_current is not None and name in self._w_before_window_slow:
                delta_slow = (_slow_w_current - self._w_before_window_slow[name]).float()

            # --- delta_target: prefer explicit dict, fall back to cache ----
            dt = None
            if delta_target_dict is not None and name in delta_target_dict:
                dt = delta_target_dict[name]
            elif hasattr(self, '_accumulated_targets') and name in self._accumulated_targets:
                dt = self._accumulated_targets[name]

            if dt is not None:
                min_d = min(dt.shape[0], delta_visible.shape[0])
                min_x = min(dt.shape[1], delta_visible.shape[1])
                dt = dt[:min_d, :min_x]
                dv = delta_visible[:min_d, :min_x]

                # Slow delta matched
                ds = None
                if delta_slow is not None:
                    ds = delta_slow[:min_d, :min_x]

                completed = self._trackers[name].accumulate(dt, dv, delta_slow=ds)
                for item in completed:
                    if len(item) == 5:
                        K, vrc_k, vrr_k, vrc_slow_k, vrr_slow_k = item
                    else:
                        K, vrc_k, vrr_k = item
                        vrc_slow_k, vrr_slow_k = None, None
                    rec = {
                        "step": step,
                        "tile_name": name,
                        "subtype": entry["subtype"],
                        "window_K": K,
                        "VRC_K": vrc_k,
                        "VRR_K": vrr_k,
                    }
                    if vrc_slow_k is not None:
                        rec["VRC_slow_K"] = vrc_slow_k
                        rec["VRR_slow_K"] = vrr_slow_k
                    self.window_records.append(rec)

            # --- persist w_after as next step's w_before -------------------
            self._w_before_window[name] = w_after
            if _slow_w_current is not None:
                self._w_before_window_slow[name] = _slow_w_current.clone()

    def set_accumulated_targets(self, targets_dict, lr):
        """Set pre-accumulated G_l targets from update_diagnostics.

        For grad_accum > 1, update_diagnostics accumulates d^T @ x per microbatch.
        This method receives the accumulated G and stores delta_target = -lr * G
        so that carry_path can use it for VRC, transfer tracker, etc.

        Args:
            targets_dict: {tile_key: G_accumulated tensor} from update_diagnostics._grad_accum
                          or from before_cache after snapshot_before_step.
            lr: Current learning rate.
        """
        self._accumulated_targets = {}
        for key, G in targets_dict.items():
            # Map update_diagnostics key to carry_path tile name
            # They may differ (update_diag uses module name, carry_path may use same)
            self._accumulated_targets[key] = (-lr * G).cpu().float()

    def _compute_analog_target(self, name, lr):
        """Compute FP32 target update from cached AnalogContext x, d or accumulated targets."""
        # First check accumulated targets (grad_accum > 1)
        if hasattr(self, '_accumulated_targets') and name in self._accumulated_targets:
            return self._accumulated_targets[name]

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
                ws = {
                    "mean_VRC_K": float(np.mean([r["VRC_K"] for r in recs])),
                    "std_VRC_K": float(np.std([r["VRC_K"] for r in recs])),
                    "mean_VRR_K": float(np.mean([r["VRR_K"] for r in recs])),
                }
                # VRC_slow (TTv1 only)
                slow_vrcs = [r["VRC_slow_K"] for r in recs if "VRC_slow_K" in r and r["VRC_slow_K"] is not None]
                slow_vrrs = [r["VRR_slow_K"] for r in recs if "VRR_slow_K" in r and r["VRR_slow_K"] is not None]
                if slow_vrcs:
                    ws["mean_VRC_slow_K"] = float(np.mean(slow_vrcs))
                    ws["std_VRC_slow_K"] = float(np.std(slow_vrcs))
                if slow_vrrs:
                    ws["mean_VRR_slow_K"] = float(np.mean(slow_vrrs))
                window_summary[str(K)] = ws
            summary["windows"] = window_summary

        # TTv1 slow/fast/eff separated metrics
        if self.method == "ttv1" and self.step_records:
            slow_cos = [r["cosine_sim_slow"] for r in self.step_records if "cosine_sim_slow" in r]
            fast_cos = [r["cosine_sim_fast"] for r in self.step_records if "cosine_sim_fast" in r]
            eff_cos = [r["cosine_sim_eff"] for r in self.step_records if "cosine_sim_eff" in r]
            slow_res = [r["residual_ratio_slow"] for r in self.step_records if "residual_ratio_slow" in r]
            ttv1_metrics = {}
            if slow_cos:
                ttv1_metrics["mean_cosine_sim_slow"] = float(np.mean(slow_cos))
                ttv1_metrics["std_cosine_sim_slow"] = float(np.std(slow_cos))
            if fast_cos:
                ttv1_metrics["mean_cosine_sim_fast"] = float(np.mean(fast_cos))
            if eff_cos:
                ttv1_metrics["mean_cosine_sim_eff"] = float(np.mean(eff_cos))
            if slow_res:
                ttv1_metrics["mean_residual_ratio_slow"] = float(np.mean(slow_res))
            if ttv1_metrics:
                summary["ttv1_separated"] = ttv1_metrics

        # Transfer summary (TTv1)
        if self.transfer_records:
            e2e_cos = [r["EndToEndCos"] for r in self.transfer_records]
            handoff_cos = [r["HandoffCos"] for r in self.transfer_records]
            resets = [r.get("reset_detected", False) for r in self.transfer_records]
            n_resets = sum(resets)
            summary["ttv1_transfer"] = {
                "n_transfers": len(self.transfer_records),
                "mean_EndToEndCos": float(np.mean(e2e_cos)),
                "mean_HandoffCos": float(np.mean(handoff_cos)),
                "std_EndToEndCos": float(np.std(e2e_cos)),
                "std_HandoffCos": float(np.std(handoff_cos)),
                "n_resets_detected": n_resets,
                "reset_fraction": n_resets / max(len(self.transfer_records), 1),
            }

        return summary
